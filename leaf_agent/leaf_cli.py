import json
import os
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

try:  # imported as a package (tests, host side)
    from . import tool_runner
    from .tools import OPENAI_TOOLS, SYSTEM_PROMPT
except ImportError:  # run as a script inside the sandbox: sibling files
    import tool_runner  # type: ignore[no-redef]
    from tools import OPENAI_TOOLS, SYSTEM_PROMPT  # type: ignore[no-redef]

#: Advertised to the adapter, which ignores it and serves whatever sglang loaded.
MODEL_LABEL = "slime-actor"
REQUEST_TIMEOUT_SEC = 1800

_LENGTH_FINISH_REASONS = frozenset({"length", "max_tokens", "max_new_tokens"})

# Never route the adapter call through a proxy: the sandbox may inherit
# HTTP(S)_PROXY, which would hijack the adapter URL (the urllib equivalent of
# aiohttp's trust_env=False).
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class ChatError(RuntimeError):
    """A non-200 reply from the adapter.

    Not retried: 429 means the adapter's per-session turn cap was hit and 503
    means the session is already closed, so retrying can only waste the budget.
    """

    def __init__(self, status: int, body: str) -> None:
        super().__init__("HTTP {}: {}".format(status, body[:500]))
        self.status = status
        self.body = body


def request_body(
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_completion_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """The Chat Completions body Leaf sends.

    Deliberately absent, unlike FrogNano's client:
      * temperature / top_p / top_k -- the adapter forwards them to sglang
        (adapters/common.py _sampling_params), overriding slime's rollout
        sampling. In RL the trainer owns sampling.
      * stream -- the adapter would answer with SSE instead of one JSON body.
      * parallel_tool_calls -- the adapter returns only the first tool call.
    """
    body = {
        "model": MODEL_LABEL,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
    }  # type: Dict[str, Any]
    if max_completion_tokens is not None:
        # Safe to send: the adapter uses min(this, its own cap).
        body["max_completion_tokens"] = int(max_completion_tokens)
    return body


def complete(
    base_url: str,
    api_key: str,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    max_completion_tokens: Optional[int] = None,
    timeout: float = REQUEST_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """POST one turn to ``{base_url}/chat/completions`` and return the response dict.

    ``api_key`` is the slime session id: the adapter reads it from the Bearer
    header (adapters/openai.py _request_session_id) to route the turn into the
    right trajectory. Raises ChatError on any non-200 reply.
    """
    url = base_url.rstrip("/") + "/chat/completions"
    data = json.dumps(request_body(messages, tools, max_completion_tokens)).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": "Bearer " + api_key,
        },
    )
    try:
        with _OPENER.open(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise ChatError(exc.code, exc.read().decode("utf-8", "replace")) from exc


# --- response helpers: FrogNano's getattr-on-SDK-objects, rewritten for dicts ---


def message_dict(message: Dict[str, Any]) -> Dict[str, Any]:
    """The assistant message to append to the history (and echo next turn)."""
    result = {"role": "assistant", "content": message.get("content")}  # type: Dict[str, Any]
    tool_calls = message.get("tool_calls")
    if tool_calls:
        result["tool_calls"] = [
            {
                "id": call.get("id"),
                "type": "function",
                "function": {
                    "name": (call.get("function") or {}).get("name"),
                    "arguments": (call.get("function") or {}).get("arguments"),
                },
            }
            for call in tool_calls
        ]
    reasoning = message.get("reasoning_content")
    if reasoning:
        result["reasoning_content"] = reasoning
    return result


def is_length_finish(choice: Dict[str, Any]) -> bool:
    return str(choice.get("finish_reason") or "").lower() in _LENGTH_FINISH_REASONS


def usage_total(usage: Optional[Dict[str, Any]]) -> int:
    if not usage:
        return 0
    total = usage.get("total_tokens")
    if isinstance(total, int):
        return total
    prompt = usage.get("prompt_tokens")
    completion = usage.get("completion_tokens")
    if isinstance(prompt, int) and isinstance(completion, int):
        return prompt + completion
    return 0


# --- the agent loop (LeafAgent.run, minus k8s, checkpoints and token counting) ---

#: Normal endings: the workspace is still graded, so the process exits 0.
#: generate.py only logs a non-zero agent_exit_code; it does not drop the sample.
NORMAL_EXITS = frozenset({"agent", "max_turns", "max_time", "max_context_len", "max_tokens_per_turn"})


def parse_arguments(raw: Any) -> Dict[str, Any]:
    """Tool-call arguments (a JSON string on the wire) -> dict; anything malformed -> {}."""
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def is_context_overflow(error: Exception) -> bool:
    message = str(error).lower()
    return any(
        marker in message for marker in ("context_length_exceeded", "reduce the length", "maximum context length")
    ) or ("context length" in message and "exceed" in message)


def length_exit_reason(response: Dict[str, Any]) -> str:
    """Why a reply was truncated.

    slime's adapter answers a prompt that no longer fits with an *empty* reply and
    finish_reason="length" (adapters/common.py, call_sglang_generate), so zero
    completion tokens means the context is full; otherwise the per-turn cap hit.
    """
    completion = (response.get("usage") or {}).get("completion_tokens")
    if completion == 0:
        return "max_context_len"
    return "max_tokens_per_turn"


def _emit(log: Callable[[str], None], record: Dict[str, Any]) -> None:
    log(json.dumps(record, ensure_ascii=False))


def run_leaf(
    base_url: str,
    api_key: str,
    prompt: str,
    *,
    workdir: str,
    max_steps: int = 100,
    max_time_sec: Optional[float] = None,
    max_completion_tokens: Optional[int] = None,
    request_timeout: float = REQUEST_TIMEOUT_SEC,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Run the Leaf loop to completion and return its result dict.

    Each step: one model call; if the reply has no tool calls (final answer) or
    was truncated, stop; otherwise run every tool call in order and append one
    ``tool`` message per call. Tools run in-process in ``workdir`` -- this
    process is already inside the sandbox.
    """
    os.chdir(workdir)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]  # type: List[Dict[str, Any]]
    started = time.monotonic()
    exit_reason = "max_turns"
    error = None  # type: Optional[str]
    final_message = ""
    n_steps = 0

    for step in range(max_steps):
        if max_time_sec is not None and time.monotonic() - started >= max_time_sec:
            exit_reason = "max_time"
            break
        n_steps += 1

        try:
            response = complete(
                base_url,
                api_key,
                messages,
                OPENAI_TOOLS,
                max_completion_tokens=max_completion_tokens,
                timeout=request_timeout,
            )
            choice = response["choices"][0]
            assistant = message_dict(choice.get("message") or {})
        except Exception as exc:  # ChatError (429/503/...), connection errors, malformed JSON
            exit_reason = "max_context_len" if is_context_overflow(exc) else "llm_query_error"
            error = "{}: {}".format(type(exc).__name__, exc)
            _emit(log, {"step": step, "event": "llm_error", "error": error})
            break

        messages.append(assistant)
        tool_calls = assistant.get("tool_calls") or []
        record = {
            "step": step,
            "finish_reason": choice.get("finish_reason"),
            "tokens": usage_total(response.get("usage")),
            "tool_calls": [call["function"]["name"] for call in tool_calls],
        }  # type: Dict[str, Any]

        truncated = is_length_finish(choice)
        if not tool_calls or truncated:
            # Final answer, or a cut-off reply whose tool calls can't be trusted.
            final_message = str(assistant.get("content") or "")
            exit_reason = length_exit_reason(response) if truncated else "agent"
            _emit(log, record)
            break

        observations = []
        try:
            for call in tool_calls:
                name = call["function"]["name"]
                observation = tool_runner.run_tool(name, parse_arguments(call["function"]["arguments"]))
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": observation})
                observations.append(len(observation))
        except Exception as exc:  # run_tool returns errors as text; this is a last resort
            exit_reason = "tool_error"
            error = "{}: {}".format(type(exc).__name__, exc)
            _emit(log, dict(record, event="tool_error", error=error))
            break
        record["observation_chars"] = observations
        _emit(log, record)

    result = {
        "exit_reason": exit_reason,
        "n_steps": n_steps,
        "error": error,
        "final_message": final_message,
        "elapsed_sec": round(time.monotonic() - started, 3),
        "messages": messages,
    }
    _emit(log, {"event": "exit", **{k: v for k, v in result.items() if k != "messages"}})
    return result


def _env_number(name: str, cast: Callable[[str], Any], default: Any) -> Any:
    raw = os.environ.get(name, "").strip()
    return cast(raw) if raw else default


def main(argv: Optional[List[str]] = None) -> int:
    """Process entry point: ``leaf_cli.py <prompt_file>``; returns the exit code.

    0 for every normal ending (NORMAL_EXITS), 1 for an LLM or tool error,
    2 for bad usage or missing configuration.
    """
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        sys.stderr.write("usage: leaf_cli.py <prompt_file>\n")
        return 2
    try:
        with open(argv[0], encoding="utf-8") as handle:
            prompt = handle.read()
        # pop, not get: tool subprocesses (Bash) inherit os.environ, and the model
        # has no business seeing the adapter URL or its own session id.
        base_url = os.environ.pop("OPENAI_BASE_URL")
        api_key = os.environ.pop("OPENAI_API_KEY")
    except (OSError, KeyError) as exc:
        sys.stderr.write("leaf_cli: {}: {}\n".format(type(exc).__name__, exc))
        return 2

    result = run_leaf(
        base_url,
        api_key,
        prompt,
        workdir=os.environ.get("LEAF_WORKDIR") or os.getcwd(),
        max_steps=_env_number("LEAF_MAX_STEPS", int, 100),
        max_time_sec=_env_number("LEAF_MAX_TIME_SEC", float, None),
        max_completion_tokens=_env_number("LEAF_MAX_COMPLETION_TOKENS", int, None),
        log=lambda line: print(line, flush=True),
    )
    return 0 if result["exit_reason"] in NORMAL_EXITS else 1


if __name__ == "__main__":
    raise SystemExit(main())
