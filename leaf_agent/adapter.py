from __future__ import annotations

import json
import secrets
from typing import Any

from slime.agent.adapters import OpenAIAdapter
from slime.agent.adapters.common import Reply, manager_finish_reason
from slime.agent.parsing import ParsedModelOutput


class LeafChatAdapter(OpenAIAdapter):
    """OpenAIAdapter that returns every tool call, for clients that echo them all."""

    log_prefix = "leaf_adapter"

    def _build_reply(self, parsed, raw_finish, translated, tools_schema) -> Reply:
        wire_message, manager_message, wire_finish = build_reply_parts(parsed, raw_finish)
        return Reply(
            manager_message=manager_message,
            finish_reason=manager_finish_reason(parsed.tool_uses, raw_finish),
            wire=(wire_message, wire_finish),
        )


def build_reply_parts(parsed: ParsedModelOutput, finish: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """``openai._build_reply_parts`` for a verbatim-echo client (Leaf).

    Returns (wire_message, manager_message, wire_finish). Two changes from
    upstream: every tool call is kept, and the manager copy keeps
    reasoning_content. Everything else is upstream's, so the history match holds
    against Leaf's echo: wire arguments are a JSON string with a unique id per
    call; manager arguments are a dict with no id; content is null / "" when the
    turn calls tools.
    """
    wire_tool_calls: list[dict[str, Any]] = []
    manager_tool_calls: list[dict[str, Any]] = []
    for tu in parsed.tool_uses:
        name = tu.get("name", "tool")
        args_dict = tu.get("input") or {}
        if not isinstance(args_dict, dict):
            args_dict = {"_raw_arguments": str(args_dict)}
        wire_tool_calls.append(
            {
                "id": f"call_{secrets.token_hex(12)}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(args_dict, ensure_ascii=False, sort_keys=True),
                },
            }
        )
        manager_tool_calls.append({"type": "function", "function": {"name": name, "arguments": args_dict}})

    wire_message: dict[str, Any] = {
        "role": "assistant",
        "content": None if wire_tool_calls else (parsed.text or None),
    }
    manager_message: dict[str, Any] = {
        "role": "assistant",
        "content": "" if wire_tool_calls else (parsed.text or ""),
    }
    if parsed.reasoning:
        wire_message["reasoning_content"] = parsed.reasoning
        # Leaf echoes it; upstream drops it here, which mismatches every thinking turn.
        manager_message["reasoning_content"] = parsed.reasoning
    if wire_tool_calls:
        # all of them, not upstream's [:1]
        wire_message["tool_calls"] = wire_tool_calls
        manager_message["tool_calls"] = manager_tool_calls

    if parsed.tool_uses:
        wire_finish = "tool_calls"
    elif finish == "length":
        wire_finish = "length"
    else:
        wire_finish = "stop"

    return wire_message, manager_message, wire_finish
