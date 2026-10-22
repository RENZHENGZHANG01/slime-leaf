from __future__ import annotations

import ast
import asyncio
import re
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
# pinned slime submodule: provides slime.* and its test fakes (tests.test_agent._fakes)
SLIME_DIR = Path(os.environ.get("SLIME_DIR", REPO_ROOT / "slime")).resolve()
for path in (REPO_ROOT, SLIME_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from leaf_agent import leaf_cli, tool_runner  # noqa: E402
from leaf_agent.tools import OPENAI_TOOLS  # noqa: E402
from tests.test_agent._fakes import FakeTokenizer, fake_call_sglang_generate  # noqa: E402

from slime.agent.adapters import OpenAIAdapter  # noqa: E402
from slime.agent.adapters import common as adapters_common  # noqa: E402
from slime.agent.aiohttp_threaded import run_app_in_thread  # noqa: E402
from slime.utils.types import Sample  # noqa: E402

LEAF_DIR = REPO_ROOT / "leaf_agent"
#: Files copied into the sandbox, which may only have an old python3.
SANDBOX_FILES = ("leaf_cli.py", "tool_runner.py", "tools.py")
SID = "leaf-sess"
#: One whitespace-free "word" so FakeTokenizer round-trips it; slime's XML
#: fallback parser turns it into an OpenAI tool_call when Bash is in `tools`.
BASH_LS = "<tool_call><function=Bash><parameter=command>ls</parameter></function></tool_call>"


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """A fresh repo-like directory as cwd, mirroring tool_runner.main()'s os.chdir(workdir)."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


# ===========================================================================
# Step 1: tool_runner.run_tool
# ===========================================================================


def test_tool_schemas_match_runner():
    """Every advertised tool must be one run_tool can execute."""
    names = {tool["function"]["name"] for tool in OPENAI_TOOLS}
    assert names == {"Read", "Write", "Edit", "Glob", "Bash"}


# --- Write / Read ----------------------------------------------------------


def test_write_creates_file_and_parent_dirs(workdir):
    out = tool_runner.run_tool("Write", {"file_path": "pkg/sub/a.txt", "content": "x\ny\n"})
    assert out == "Wrote pkg/sub/a.txt (4 bytes)."
    assert (workdir / "pkg/sub/a.txt").read_text() == "x\ny\n"


def test_read_returns_numbered_lines(workdir):
    (workdir / "a.txt").write_text("alpha\nbeta\ngamma\n")
    assert tool_runner.run_tool("Read", {"file_path": "a.txt"}) == "a.txt\n1. alpha\n2. beta\n3. gamma"


def test_read_offset_and_limit(workdir):
    (workdir / "a.txt").write_text("".join(f"line{i}\n" for i in range(1, 11)))
    out = tool_runner.run_tool("Read", {"file_path": "a.txt", "offset": 4, "limit": 2})
    assert out == "a.txt\n4. line4\n5. line5"


def test_read_missing_file_returns_error_string(workdir):
    out = tool_runner.run_tool("Read", {"file_path": "nope.txt"})
    assert out.startswith("FileNotFoundError:"), "errors come back as observations, never raised"


# --- Edit ------------------------------------------------------------------


def test_edit_replaces_single_match(workdir):
    (workdir / "a.py").write_text("x = 1\ny = 2\n")
    out = tool_runner.run_tool("Edit", {"file_path": "a.py", "old_string": "x = 1", "new_string": "x = 10"})
    assert out == "Edited a.py (1 replacement)."
    assert (workdir / "a.py").read_text() == "x = 10\ny = 2\n"


def test_edit_no_match_is_an_error(workdir):
    (workdir / "a.py").write_text("x = 1\n")
    out = tool_runner.run_tool("Edit", {"file_path": "a.py", "old_string": "missing", "new_string": "z"})
    assert out == "ValueError: old_string was not found"
    assert (workdir / "a.py").read_text() == "x = 1\n", "a failed edit must not touch the file"


def test_edit_ambiguous_match_requires_replace_all(workdir):
    (workdir / "a.py").write_text("v = 0\nv = 0\n")
    out = tool_runner.run_tool("Edit", {"file_path": "a.py", "old_string": "v = 0", "new_string": "v = 1"})
    assert out.startswith("ValueError: old_string matched 2 times")
    assert (workdir / "a.py").read_text() == "v = 0\nv = 0\n"


def test_edit_replace_all(workdir):
    (workdir / "a.py").write_text("v = 0\nv = 0\n")
    out = tool_runner.run_tool(
        "Edit", {"file_path": "a.py", "old_string": "v = 0", "new_string": "v = 1", "replace_all": True}
    )
    assert out == "Edited a.py (2 replacements)."
    assert (workdir / "a.py").read_text() == "v = 1\nv = 1\n"


def test_edit_empty_old_string_is_an_error(workdir):
    (workdir / "a.py").write_text("x\n")
    out = tool_runner.run_tool("Edit", {"file_path": "a.py", "old_string": "", "new_string": "y"})
    assert out == "ValueError: old_string must be non-empty"


# --- Glob ------------------------------------------------------------------


def test_glob_recursive_files_only_sorted(workdir):
    for rel in ("b.py", "a.py", "src/c.py", "src/deep/d.py", "notes.txt"):
        (workdir / rel).parent.mkdir(parents=True, exist_ok=True)
        (workdir / rel).write_text("")
    (workdir / "dir.py").mkdir()  # a directory matching the pattern must be skipped
    out = tool_runner.run_tool("Glob", {"pattern": "**/*.py"})
    assert out.splitlines() == ["a.py", "b.py", "src/c.py", "src/deep/d.py"]


def test_glob_no_match(workdir):
    assert tool_runner.run_tool("Glob", {"pattern": "*.rs"}) == "No files matched."


# --- Bash ------------------------------------------------------------------


def test_bash_reports_exit_code_stdout_stderr(workdir):
    out = tool_runner.run_tool("Bash", {"command": "echo hi; echo oops >&2; exit 3"})
    assert out == "Exit code: 3\nSTDOUT:\nhi\n\nSTDERR:\noops\n"


def test_bash_runs_in_cwd(workdir):
    (workdir / "marker.txt").write_text("")
    out = tool_runner.run_tool("Bash", {"command": "ls"})
    assert out.startswith("Exit code: 0\n")
    assert "marker.txt" in out


def test_bash_timeout_returns_not_raises(workdir):
    out = tool_runner.run_tool("Bash", {"command": "sleep 5", "timeout": 0.5})
    assert out.startswith("Timed out after 0.5s")


def test_bash_output_is_truncated(workdir, monkeypatch):
    monkeypatch.setenv("LEAF_BASH_OUTPUT_CHARS", "50")
    out = tool_runner.run_tool("Bash", {"command": "printf 'a%.0s' $(seq 1 500)"})
    assert len(out.split("\n[truncated")[0]) == 50
    assert "[truncated" in out


# --- dispatch --------------------------------------------------------------


def test_unknown_tool(workdir):
    assert tool_runner.run_tool("Delete", {}) == "Unknown tool: Delete"


# ===========================================================================
# Step 2: leaf_cli.complete against the real OpenAIAdapter
# ===========================================================================
#
# The adapter runs in its own thread (run_app_in_thread) with sglang faked, so
# the blocking urllib client can be called straight from the test.


class AdapterHarness:
    """A live OpenAIAdapter on 127.0.0.1 with scripted model replies."""

    def __init__(self, monkeypatch, replies, *, max_turns_per_sid=None, host="127.0.0.1", adapter_cls=OpenAIAdapter):
        """``replies``: model outputs in order; each a text (finish "stop") or a (text, finish) pair."""
        self.tok = FakeTokenizer()
        script = [(r, "stop", None) if isinstance(r, str) else (r[0], r[1], None) for r in replies]
        fake = fake_call_sglang_generate(script, self.tok)
        self.bodies = []  # every request body the adapter handed to "sglang"

        async def recording(prompt_ids, session, body, *, adapter, session_id=None):
            self.bodies.append(body)
            return await fake(prompt_ids, session, body, adapter=adapter, session_id=session_id)

        monkeypatch.setattr(adapters_common, "call_sglang_generate", recording)
        self.adapter = adapter_cls(tokenizer=self.tok, sglang_url="http://unused", max_turns_per_sid=max_turns_per_sid)
        self.handle = run_app_in_thread(self.adapter.app, host=host, port=0, thread_name="test-leaf-adapter")
        self.base_url = f"http://127.0.0.1:{self.handle.port}/v1"
        self.adapter.open_session(SID)

    def complete(self, messages, sid=SID, **kwargs):
        return leaf_cli.complete(self.base_url, sid, messages, OPENAI_TOOLS, timeout=30, **kwargs)

    def finish(self):
        import asyncio

        return asyncio.run(self.adapter.finish_session(SID, base_sample=Sample(index=0, prompt=""), reward=1.0))

    def close(self):
        self.handle.stop()


@pytest.fixture
def make_adapter(monkeypatch):
    made = []

    def _make(replies, **kwargs):
        made.append(AdapterHarness(monkeypatch, replies, **kwargs))
        return made[-1]

    yield _make
    for server in made:
        server.close()


USER_ONLY = [{"role": "system", "content": "sys"}, {"role": "user", "content": "fix the bug"}]


def test_sandbox_files_are_python36_syntax():
    """Everything copied into the sandbox must parse as Python 3.6."""
    for name in SANDBOX_FILES:
        source = (LEAF_DIR / name).read_text()
        tree = ast.parse(source, filename=name, feature_version=(3, 6))
        future = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "__future__"
            for alias in node.names
        }
        assert "annotations" not in future, f"{name}: `from __future__ import annotations` needs 3.7+"


def test_request_body_leaves_sampling_to_slime():
    body = leaf_cli.request_body(USER_ONLY, OPENAI_TOOLS, max_completion_tokens=512)
    for key in ("temperature", "top_p", "top_k", "stream", "parallel_tool_calls"):
        assert key not in body, f"{key} must not be sent (see request_body docstring)"
    assert body["max_completion_tokens"] == 512
    assert body["tool_choice"] == "auto"
    assert "max_completion_tokens" not in leaf_cli.request_body(USER_ONLY, OPENAI_TOOLS)


def test_text_reply(make_adapter):
    server = make_adapter(["all done here"])
    response = server.complete(USER_ONLY)
    choice = response["choices"][0]
    assert choice["message"]["content"] == "all done here"
    assert choice["finish_reason"] == "stop"
    assert not leaf_cli.is_length_finish(choice)
    assert leaf_cli.usage_total(response["usage"]) > 0


def test_tool_call_reply_and_echo_shape(make_adapter):
    server = make_adapter([BASH_LS])
    response = server.complete(USER_ONLY)
    choice = response["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    message = choice["message"]
    assert message["content"] is None, "content is null when the model calls a tool"
    (call,) = message["tool_calls"]
    assert call["function"]["name"] == "Bash"
    assert json.loads(call["function"]["arguments"]) == {"command": "ls"}, "arguments arrive as a JSON string"

    echo = leaf_cli.message_dict(message)
    assert echo == {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": call["id"], "type": "function", "function": call["function"]}],
    }


def test_wire_body_reaching_sglang(make_adapter):
    """What the adapter forwarded: our max tokens, no client sampling overrides."""
    server = make_adapter(["ok then fine"])
    server.complete(USER_ONLY, max_completion_tokens=256)
    (body,) = server.bodies
    assert body["max_completion_tokens"] == 256
    assert "temperature" not in body and "top_p" not in body and "top_k" not in body


def test_api_key_routes_turn_into_session(make_adapter):
    """The Bearer key is the sid: the turn must land in that session's trajectory."""
    server = make_adapter(["routed reply here"])
    server.complete(USER_ONLY)
    samples = server.finish()
    assert len(samples) == 1
    response = samples[0].tokens[-samples[0].response_length :]
    trained = [t for t, m in zip(response, samples[0].loss_mask, strict=True) if m]
    assert server.tok.decode(trained) == "routed reply here"


def test_turn_cap_raises_429_without_retry(make_adapter):
    server = make_adapter(["first reply ok", "never served"], max_turns_per_sid=1)
    server.complete(USER_ONLY)
    with pytest.raises(leaf_cli.ChatError) as excinfo:
        server.complete(USER_ONLY)
    assert excinfo.value.status == 429
    assert len(server.bodies) == 1, "a capped turn must not reach the model, and must not be retried"


def test_closed_session_raises_503(make_adapter):
    server = make_adapter(["only reply here"])
    server.complete(USER_ONLY)
    server.finish()
    with pytest.raises(leaf_cli.ChatError) as excinfo:
        server.complete(USER_ONLY)
    assert excinfo.value.status == 503


def test_ignores_proxy_env(make_adapter, monkeypatch):
    """A proxy inherited by the sandbox must not intercept the adapter call."""
    monkeypatch.setenv("http_proxy", "http://127.0.0.1:9")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:9")
    server = make_adapter(["direct reply ok"])
    assert server.complete(USER_ONLY)["choices"][0]["message"]["content"] == "direct reply ok"


@pytest.mark.parametrize(
    ("choice", "expected"),
    [({"finish_reason": "length"}, True), ({"finish_reason": "stop"}, False), ({}, False)],
)
def test_is_length_finish(choice, expected):
    assert leaf_cli.is_length_finish(choice) is expected


@pytest.mark.parametrize(
    ("usage", "expected"),
    [
        ({"total_tokens": 7}, 7),
        ({"prompt_tokens": 3, "completion_tokens": 4}, 7),
        ({}, 0),
        (None, 0),
    ],
)
def test_usage_total(usage, expected):
    assert leaf_cli.usage_total(usage) == expected


def test_message_dict_keeps_reasoning():
    echo = leaf_cli.message_dict({"role": "assistant", "content": "x", "reasoning_content": "because"})
    assert echo == {"role": "assistant", "content": "x", "reasoning_content": "because"}


# ===========================================================================
# Step 3: the agent loop (run_leaf) and process entry point (main)
# ===========================================================================
#
# Tool calls are scripted as whitespace-free XML (see BASH_LS). Under
# FakeTokenizer every tool-call turn is REALIGNed and masked out (its chat
# template renders calls as "toolcall:<name>", not the generated XML), so these
# tests only assert training of the *final text*; Step 7 checks tool-call turns
# with a real tokenizer.


def _call(name, **params):
    body = "".join(f"<parameter={k}>{v}</parameter>" for k, v in params.items())
    return f"<tool_call><function={name}>{body}</function></tool_call>"


WRITE_OUT = _call("Write", file_path="out.txt", content="hello")
READ_OUT = _call("Read", file_path="out.txt")
EDIT_MISSING = _call("Edit", file_path="out.txt", old_string="zzz", new_string="yyy")


def _run(server, workdir, **kwargs):
    lines = []
    result = leaf_cli.run_leaf(
        server.base_url, SID, "fix the bug", workdir=str(workdir), request_timeout=30, log=lines.append, **kwargs
    )
    return result, [json.loads(line) for line in lines]


def _trained_text(sample, tok):
    response = sample.tokens[-sample.response_length :]
    return tok.decode([t for t, m in zip(response, sample.loss_mask, strict=True) if m])


def test_loop_runs_tools_then_answers(make_adapter, workdir):
    server = make_adapter([WRITE_OUT, READ_OUT, "all done"])
    result, log = _run(server, workdir)

    assert result["exit_reason"] == "agent"
    assert result["n_steps"] == 3
    assert result["final_message"] == "all done"
    assert (workdir / "out.txt").read_text() == "hello", "the Write tool really ran in workdir"

    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "assistant", "tool", "assistant"]
    first_call, first_result = result["messages"][2], result["messages"][3]
    assert first_result["tool_call_id"] == first_call["tool_calls"][0]["id"]
    assert first_result["content"] == "Wrote out.txt (5 bytes)."
    assert result["messages"][5]["content"] == "out.txt\n1. hello"

    assert [entry.get("tool_calls") for entry in log[:3]] == [["Write"], ["Read"], []]
    assert log[-1]["event"] == "exit" and log[-1]["exit_reason"] == "agent"

    samples = server.finish()
    assert len(samples) == 1, "a tool loop that extends one conversation is one Sample"
    assert _trained_text(samples[0], server.tok) == "all done"


def test_tool_error_is_an_observation_not_a_crash(make_adapter, workdir):
    (workdir / "out.txt").write_text("hello\n")
    server = make_adapter([EDIT_MISSING, "gave up"])
    result, _ = _run(server, workdir)
    assert result["exit_reason"] == "agent"
    assert result["messages"][3]["content"] == "ValueError: old_string was not found"
    assert result["final_message"] == "gave up"


def test_max_steps(make_adapter, workdir):
    server = make_adapter([READ_OUT, READ_OUT, READ_OUT])
    (workdir / "out.txt").write_text("x\n")
    result, _ = _run(server, workdir, max_steps=2)
    assert result["exit_reason"] == "max_turns"
    assert result["n_steps"] == 2
    assert len(server.bodies) == 2


def test_max_time_stops_before_calling_the_model(make_adapter, workdir):
    server = make_adapter(["never served"])
    result, _ = _run(server, workdir, max_time_sec=0)
    assert result["exit_reason"] == "max_time"
    assert result["n_steps"] == 0
    assert server.bodies == []


def test_context_full_is_empty_length_reply(make_adapter, workdir):
    """slime's adapter answers an over-long prompt with an empty reply, finish 'length'."""
    server = make_adapter([("", "length")])
    result, _ = _run(server, workdir)
    assert result["exit_reason"] == "max_context_len"


def test_cut_off_reply_is_max_tokens_per_turn(make_adapter, workdir):
    server = make_adapter([("partial answer that got", "length")])
    result, _ = _run(server, workdir)
    assert result["exit_reason"] == "max_tokens_per_turn"
    assert result["final_message"] == "partial answer that got"


def test_truncated_tool_call_is_not_executed(workdir, monkeypatch):
    """A cut-off reply's tool call may be incomplete, so Leaf stops instead of running it.

    slime's OpenAIAdapter never sends this combination (any parsed tool call is
    reported as finish "tool_calls", adapters/openai.py _build_reply_parts), so
    the reply is stubbed; other OpenAI-compatible servers do send it.
    """
    reply = {
        "choices": [
            {
                "finish_reason": "length",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "c1", "type": "function", "function": {"name": "Write", "arguments": '{"file_path": "out'}}
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 64, "total_tokens": 74},
    }
    monkeypatch.setattr(leaf_cli, "complete", lambda *a, **k: reply)
    result = leaf_cli.run_leaf("http://unused/v1", SID, "p", workdir=str(workdir), log=lambda _: None)
    assert result["exit_reason"] == "max_tokens_per_turn"
    assert result["messages"][-1]["role"] == "assistant", "no tool message appended"
    assert list(workdir.iterdir()) == [], "the truncated Write must not run"


def test_turn_cap_ends_the_loop(make_adapter, workdir):
    (workdir / "out.txt").write_text("x\n")
    server = make_adapter([READ_OUT, "never served"], max_turns_per_sid=1)
    result, log = _run(server, workdir)
    assert result["exit_reason"] == "llm_query_error"
    assert "HTTP 429" in result["error"]
    assert result["n_steps"] == 2
    assert log[-2]["event"] == "llm_error"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [('{"command": "ls"}', {"command": "ls"}), ("not json", {}), ("[1, 2]", {}), (None, {}), ({"a": 1}, {"a": 1})],
)
def test_parse_arguments(raw, expected):
    assert leaf_cli.parse_arguments(raw) == expected


def test_main_reads_env_and_prompt_file(make_adapter, workdir, monkeypatch, capsys):
    server = make_adapter(["done here"])
    prompt = workdir / "prompt.txt"
    prompt.write_text("fix the bug")
    monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", SID)
    monkeypatch.setenv("LEAF_WORKDIR", str(workdir))
    monkeypatch.setenv("LEAF_MAX_STEPS", "5")

    assert leaf_cli.main([str(prompt)]) == 0
    assert "OPENAI_API_KEY" not in leaf_cli.os.environ, "the sid must not leak into tool subprocesses"
    assert "OPENAI_BASE_URL" not in leaf_cli.os.environ
    last = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert last["event"] == "exit" and last["exit_reason"] == "agent"


def test_main_usage_errors(monkeypatch, tmp_path):
    assert leaf_cli.main([]) == 2
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    (tmp_path / "p.txt").write_text("x")
    assert leaf_cli.main([str(tmp_path / "p.txt")]) == 2, "missing OPENAI_BASE_URL"
    assert leaf_cli.main([str(tmp_path / "missing.txt")]) == 2, "missing prompt file"


def test_main_exit_code_on_error(make_adapter, workdir, monkeypatch):
    (workdir / "out.txt").write_text("x\n")
    server = make_adapter([READ_OUT], max_turns_per_sid=1)
    prompt = workdir / "prompt.txt"
    prompt.write_text("fix")
    monkeypatch.setenv("OPENAI_BASE_URL", server.base_url)
    monkeypatch.setenv("OPENAI_API_KEY", SID)
    monkeypatch.setenv("LEAF_WORKDIR", str(workdir))
    assert leaf_cli.main([str(prompt)]) == 1


def test_runs_as_a_standalone_script(make_adapter, tmp_path):
    """The sandbox layout: the three files side by side, run as a plain script (no package)."""
    import shutil
    import subprocess

    install = tmp_path / "opt_leaf"
    install.mkdir()
    for name in SANDBOX_FILES:
        shutil.copy(LEAF_DIR / name, install / name)
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "prompt.txt").write_text("fix the bug")

    server = make_adapter([WRITE_OUT, "all done"])
    env = {
        "PATH": "/usr/bin:/bin",
        "OPENAI_BASE_URL": server.base_url,
        "OPENAI_API_KEY": SID,
        "LEAF_WORKDIR": str(repo),
    }
    proc = subprocess.run(
        [sys.executable, str(install / "leaf_cli.py"), str(repo / "prompt.txt")],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    last = json.loads(proc.stdout.strip().splitlines()[-1])
    assert last["exit_reason"] == "agent" and last["n_steps"] == 2
    assert (repo / "out.txt").read_text() == "hello"


DOCKER_IMAGE = "python:3.6-slim"


def _docker_ready():
    import shutil
    import subprocess

    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=30).returncode == 0


@pytest.mark.skipif(not _docker_ready(), reason="needs a running Docker daemon")
def test_runs_inside_docker_and_dials_back_to_the_adapter(make_adapter, tmp_path):
    """The real topology: loop + tools inside a python:3.6 container, adapter on the host.

    The adapter binds 0.0.0.0 (like ADAPTER_BIND_HOST) and the container reaches it
    via host.docker.internal (playing ADAPTER_PUBLIC_HOST). Files go in and out with
    `docker cp`, so no host directory is mounted: the edit must happen in the container.
    """
    import subprocess

    def docker(*args, check=True):
        return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=300, check=check)

    server = make_adapter([WRITE_OUT, READ_OUT, "all done"], host="0.0.0.0")
    base_url = f"http://host.docker.internal:{server.handle.port}/v1"

    stage = tmp_path / "stage"
    (stage / "leaf").mkdir(parents=True)
    for name in SANDBOX_FILES:
        (stage / "leaf" / name).write_bytes((LEAF_DIR / name).read_bytes())
    (stage / "prompt.txt").write_text("fix the bug")

    docker("pull", "-q", DOCKER_IMAGE)
    cid = docker(
        "create",
        "--add-host=host.docker.internal:host-gateway",
        "-e", f"OPENAI_BASE_URL={base_url}",
        "-e", f"OPENAI_API_KEY={SID}",
        "-e", "LEAF_WORKDIR=/testbed",
        "-w", "/testbed",
        DOCKER_IMAGE,
        "python3", "/opt/leaf/leaf_cli.py", "/opt/prompt.txt",
    ).stdout.strip()  # fmt: skip
    try:
        docker("cp", str(stage / "leaf"), f"{cid}:/opt/leaf")
        docker("cp", str(stage / "prompt.txt"), f"{cid}:/opt/prompt.txt")
        run = docker("start", "-a", cid, check=False)
        assert run.returncode == 0, run.stdout + run.stderr

        last = json.loads(run.stdout.strip().splitlines()[-1])
        assert last["exit_reason"] == "agent" and last["n_steps"] == 3

        docker("cp", f"{cid}:/testbed/out.txt", str(tmp_path / "out.txt"))
        assert (tmp_path / "out.txt").read_text() == "hello", "the Write tool ran inside the container"
        assert not (stage / "out.txt").exists() and not Path("out.txt").exists(), "nothing written on the host"
    finally:
        docker("rm", "-f", cid, check=False)

    assert len(server.bodies) == 3, "all three model turns crossed the container boundary"
    samples = server.finish()
    assert len(samples) == 1
    assert _trained_text(samples[0], server.tok) == "all done"



# ===========================================================================
# Step 4: LeafHarness (install, config, launch) -- FakeSandbox, then real Docker
# ===========================================================================

from leaf_agent import harness as leaf_harness  # noqa: E402
from leaf_agent.harness import LeafHarness  # noqa: E402

from slime.agent import sandbox as sandbox_mod  # noqa: E402
from slime.agent.harness import HarnessContext  # noqa: E402
from slime.agent.harness import common as harness_common  # noqa: E402
from tests.test_agent._fakes import FakeSandbox  # noqa: E402

_REAL_SLEEP = asyncio.sleep


async def _fast_sleep(_secs):
    await _REAL_SLEEP(0)


def _ctx(workdir="/testbed", url="http://host:18001"):
    return HarnessContext(workdir=workdir, session_id=SID, adapter_url=url)


def _launch(sb, ctx, prompt="fix the bug", budget=1800):
    with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
        return asyncio.run(LeafHarness().launch_and_wait(sb, ctx, prompt=prompt, time_budget_sec=budget))


def test_harness_is_registered_by_name():
    assert LeafHarness.name == "leaf"
    assert LeafHarness() is LeafHarness(), "BaseHarness subclasses are singletons"


def test_install_cli_copies_the_three_files_and_checks_python():
    sb = FakeSandbox()
    asyncio.run(LeafHarness().install_cli(sb))

    for name in SANDBOX_FILES:
        assert sb.files[f"/opt/leaf/{name}"] == (LEAF_DIR / name).read_text(), name
    assert set(sb.files) == {f"/opt/leaf/{name}" for name in SANDBOX_FILES}, "adapter/harness stay on the host"
    check = [(cmd, user) for cmd, user in sb.exec_log if leaf_harness.CHECK_PYTHON in cmd]
    assert check and check[0][1] == "root"
    assert "chmod -R a+rX /opt/leaf" in check[0][0], "the agent user must be able to read the files"


def test_install_cli_fails_loudly_without_python():
    sb = FakeSandbox(responses=[("sys.version_info", (1, "", "no python"))])

    async def exec_with_check(cmd, *, check=False, **kwargs):
        rc, out, err = await FakeSandbox.exec(sb, cmd, **kwargs)
        if check and rc != 0:
            raise RuntimeError(err)
        return rc, out, err

    sb.exec = exec_with_check
    with pytest.raises(RuntimeError, match="no python"):
        asyncio.run(LeafHarness().install_cli(sb))


def test_write_config_writes_nothing():
    sb = FakeSandbox()
    assert asyncio.run(LeafHarness().write_config(sb, _ctx())) is None
    assert sb.exec_log == [] and sb.files == {}


def test_launch_writes_prompt_file_and_exports_wiring(monkeypatch):
    for name in (LeafHarness.max_steps_env, LeafHarness.max_completion_tokens_env, LeafHarness.extra_envs_env):
        monkeypatch.delenv(name, raising=False)
    seen = {}

    async def agent(env):
        seen["env"] = env
        return 0

    sb = FakeSandbox(on_launch=agent)
    prompt = "it's a \"quoted\" prompt\nwith $HOME and `backticks`"
    assert _launch(sb, _ctx(url="http://10.0.0.5:18001"), prompt=prompt, budget=1800) == 0

    assert sb.files["/testbed/.harness/prompt.txt"] == prompt, "prompt passes through a file, untouched"
    assert any("chown agent:agent /testbed/.harness/prompt.txt" in cmd for cmd, _ in sb.exec_log)
    assert seen["env"] == {
        "OPENAI_BASE_URL": "http://10.0.0.5:18001/v1",
        "OPENAI_API_KEY": SID,
        "LEAF_WORKDIR": "/testbed",
        "LEAF_MAX_TIME_SEC": "1740",
    }
    launcher = next(v for k, v in sb.files.items() if k.endswith(".sh"))
    assert '"$python_bin" /opt/leaf/leaf_cli.py /testbed/.harness/prompt.txt' in launcher
    assert leaf_harness.FIND_PYTHON in launcher
    assert prompt not in launcher, "the prompt never goes through the shell"
    assert [user for cmd, user in sb.exec_log if "setsid" in cmd] == ["agent"]


def test_launch_env_knobs_and_extra_envs(monkeypatch):
    monkeypatch.setenv(LeafHarness.max_steps_env, "40")
    monkeypatch.setenv(LeafHarness.max_completion_tokens_env, "4096")
    monkeypatch.setenv(LeafHarness.extra_envs_env, json.dumps({"LEAF_MAX_STEPS": "7", "FOO": "bar"}))
    seen = {}

    async def agent(env):
        seen.update(env)
        return 0

    _launch(FakeSandbox(on_launch=agent), _ctx())
    assert seen["LEAF_MAX_COMPLETION_TOKENS"] == "4096"
    assert seen["LEAF_MAX_STEPS"] == "7", "extra envs are merged last and win"
    assert seen["FOO"] == "bar"


@pytest.mark.parametrize("budget, expected", [(1800, "1740"), (120, "90"), (20, "15")])
def test_leaf_stops_before_the_harness_deadline(budget, expected):
    seen = {}

    async def agent(env):
        seen.update(env)
        return 0

    _launch(FakeSandbox(on_launch=agent), _ctx(), budget=budget)
    assert seen["LEAF_MAX_TIME_SEC"] == expected


def test_launch_propagates_exit_code():
    async def agent(_env):
        return 1

    assert _launch(FakeSandbox(on_launch=agent), _ctx()) == 1


def _thread_leaf(env, prompt, workdir):
    """What leaf_cli.main does with the env it is launched with, on a worker thread."""
    return leaf_cli.run_leaf(
        env["OPENAI_BASE_URL"],
        env["OPENAI_API_KEY"],
        prompt,
        workdir=str(workdir),
        max_time_sec=float(env["LEAF_MAX_TIME_SEC"]),
        max_steps=int(env.get("LEAF_MAX_STEPS", 100)),
        request_timeout=30,
        log=lambda _line: None,
    )


def test_full_harness_run_through_fake_sandbox(make_adapter, tmp_path):
    """harness.run -> ensure_agent_user -> write_config -> launch -> run_agent -> leaf loop -> adapter."""
    server = make_adapter([WRITE_OUT, READ_OUT, "all done"])
    sb = None
    results = []

    async def agent(env):
        prompt = sb.files[f"{tmp_path}/.harness/prompt.txt"]
        result = await asyncio.to_thread(_thread_leaf, env, prompt, tmp_path)
        results.append(result)
        return 0 if result["exit_reason"] in leaf_cli.NORMAL_EXITS else 1

    sb = FakeSandbox(on_launch=agent)
    adapter_url = server.base_url[: -len("/v1")]

    async def run():
        await LeafHarness().install_cli(sb)
        with patch.object(harness_common.asyncio, "sleep", new=_fast_sleep):
            return await LeafHarness().run(
                sb, workdir=str(tmp_path), session_id=SID, adapter_url=adapter_url, time_budget_sec=600, prompt="fix"
            )

    assert asyncio.run(run()) == 0
    assert any("useradd" in cmd for cmd, _ in sb.exec_log), "BaseHarness.run created the agent user"
    (result,) = results
    assert result["exit_reason"] == "agent" and result["n_steps"] == 3
    assert (tmp_path / "out.txt").read_text() == "hello"
    samples = server.finish()
    assert len(samples) == 1
    assert _trained_text(samples[0], server.tok) == "all done"


class DockerSandbox:
    """slime's Sandbox protocol over `docker exec`, for one real container."""

    def __init__(self, container_id):
        self.cid = container_id

    async def exec(self, cmd, *, user="root", env=None, timeout=120, check=False, idempotent=True):
        args = ["docker", "exec", "-u", user]
        for key, value in (env or {}).items():
            args += ["-e", f"{key}={value}"]
        proc = await asyncio.to_thread(
            subprocess.run, [*args, self.cid, "bash", "-c", cmd], capture_output=True, text=True, timeout=timeout
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"exit {proc.returncode}: {cmd}\n{proc.stderr}")
        return proc.returncode, proc.stdout, proc.stderr

    async def write_file(self, sandbox_path, content, *, user="root"):
        data = content.read_bytes() if isinstance(content, Path) else content
        data = data.encode() if isinstance(data, str) else data
        await asyncio.to_thread(
            subprocess.run,
            ["docker", "exec", "-i", "-u", user, self.cid, "sh", "-c", f"cat > {shlex.quote(sandbox_path)}"],
            input=data,
            check=True,
            timeout=60,
        )

    async def read_file(self, sandbox_path, *, user="root"):
        rc, out, _ = await self.exec(f"cat {shlex.quote(sandbox_path)}", user=user)
        return out if rc == 0 else ""


#: Whitespace-free so FakeTokenizer round-trips it: dumps the agent's env for inspection.
DUMP_ENV = _call("Bash", command="printenv>env.txt")


@pytest.mark.skipif(not _docker_ready(), reason="needs a running Docker daemon")
def test_harness_drives_leaf_in_a_real_container(make_adapter, monkeypatch):
    """LeafHarness against a real python:3.6 container: real install, setsid launch, agent user, polling."""
    for name in (LeafHarness.max_steps_env, LeafHarness.max_completion_tokens_env, LeafHarness.extra_envs_env):
        monkeypatch.delenv(name, raising=False)
    server = make_adapter([WRITE_OUT, DUMP_ENV, "all done"], host="0.0.0.0")
    adapter_url = f"http://host.docker.internal:{server.handle.port}"

    subprocess.run(["docker", "pull", "-q", DOCKER_IMAGE], check=True, capture_output=True, timeout=300)
    cid = subprocess.run(
        ["docker", "run", "-d", "--add-host=host.docker.internal:host-gateway", DOCKER_IMAGE, "sleep", "600"],
        check=True, capture_output=True, text=True, timeout=60,
    ).stdout.strip()  # fmt: skip
    sb = DockerSandbox(cid)

    async def run():
        # ensure_agent_user also needs git (absent from -slim); do its user part by hand.
        await sb.exec("useradd -m -s /bin/bash agent && mkdir /testbed && chown agent:agent /testbed", check=True)
        harness = LeafHarness()
        await harness.install_cli(sb)
        ctx = HarnessContext(workdir="/testbed", session_id=SID, adapter_url=adapter_url)
        assert await harness.write_config(sb, ctx) is None
        return await harness.launch_and_wait(sb, ctx, prompt="fix the bug", time_budget_sec=120)

    try:
        with patch.object(sandbox_mod.asyncio, "sleep", new=lambda _s: _REAL_SLEEP(0.2)):
            exit_code = asyncio.run(run())
        trajectory = asyncio.run(sb.read_file("/testbed/.harness/trajectory.jsonl"))
        assert exit_code == 0, trajectory

        last = json.loads(trajectory.strip().splitlines()[-1])
        assert last["exit_reason"] == "agent" and last["n_steps"] == 3
        assert asyncio.run(sb.read_file("/testbed/out.txt")) == "hello"
        _, owner, _ = asyncio.run(sb.exec("stat -c %U /testbed/out.txt"))
        assert owner.strip() == "agent", "tools ran as the unprivileged agent user"

        agent_env = asyncio.run(sb.read_file("/testbed/env.txt"))
        assert "LEAF_WORKDIR=/testbed" in agent_env and "LEAF_MAX_TIME_SEC=90" in agent_env
        assert "OPENAI_API_KEY" not in agent_env and "OPENAI_BASE_URL" not in agent_env, "popped before tools ran"
    finally:
        subprocess.run(["docker", "rm", "-f", cid], capture_output=True, timeout=60)

    assert len(server.bodies) == 3
    samples = server.finish()
    assert len(samples) == 1
    assert _trained_text(samples[0], server.tok) == "all done"



# ===========================================================================
# Step 5: LeafChatAdapter (every tool call) + the self-contained generate()
# ===========================================================================

from leaf_agent import adapter as leaf_adapter  # noqa: E402
from leaf_agent.adapter import LeafChatAdapter  # noqa: E402

from slime.agent.adapters import openai as openai_adapter  # noqa: E402
from slime.agent.parsing import ParsedModelOutput  # noqa: E402

#: Two calls in one whitespace-free reply: write a file, then read it back.
WRITE_THEN_READ = WRITE_OUT + READ_OUT


def test_leaf_adapter_returns_every_tool_call(make_adapter):
    server = make_adapter([WRITE_THEN_READ], adapter_cls=LeafChatAdapter)
    reply = server.complete(USER_ONLY)
    choice = reply["choices"][0]
    calls = choice["message"]["tool_calls"]

    assert [c["function"]["name"] for c in calls] == ["Write", "Read"]
    assert [json.loads(c["function"]["arguments"]) for c in calls] == [
        {"content": "hello", "file_path": "out.txt"},
        {"file_path": "out.txt"},
    ]
    assert len({c["id"] for c in calls}) == 2, "each call needs its own id for tool_call_id"
    assert all(c["id"].startswith("call_") and c["type"] == "function" for c in calls)
    assert choice["message"]["content"] is None
    assert choice["finish_reason"] == "tool_calls"


def test_plain_openai_adapter_keeps_only_the_first_call(make_adapter):
    """Why LeafChatAdapter exists: upstream truncates to one call per turn."""
    server = make_adapter([WRITE_THEN_READ])
    calls = server.complete(USER_ONLY)["choices"][0]["message"]["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["Write"]


def _without_ids(parts):
    wire, manager, finish = json.loads(json.dumps(parts))
    for call in wire.get("tool_calls", []):
        assert call.pop("id").startswith("call_")
    return wire, manager, finish


_ONE_CALL = [{"name": "Bash", "input": {"command": "ls"}}]


@pytest.mark.parametrize(
    "parsed, finish",
    [
        (ParsedModelOutput(reasoning="", text="done", tool_uses=[]), "stop"),
        (ParsedModelOutput(reasoning="", text="", tool_uses=[]), "stop"),
        (ParsedModelOutput(reasoning="", text="cut", tool_uses=[]), "length"),
        (ParsedModelOutput(reasoning="hmm", text="done", tool_uses=[]), "stop"),
        (ParsedModelOutput(reasoning="", text="", tool_uses=_ONE_CALL), "stop"),
        (ParsedModelOutput(reasoning="hmm", text="I'll look", tool_uses=_ONE_CALL), "stop"),
        (ParsedModelOutput(reasoning="", text="", tool_uses=_ONE_CALL), "length"),
        (ParsedModelOutput(reasoning="", text="", tool_uses=[{"name": "Bash", "input": "raw"}]), "stop"),
        (ParsedModelOutput(reasoning="", text="", tool_uses=[{"input": {}}]), "stop"),
    ],
)
def test_reply_parts_match_upstream_for_zero_or_one_call(parsed, finish):
    """build_reply_parts is upstream's minus [:1] plus the kept reasoning: catch drift if upstream changes."""
    ours = _without_ids(leaf_adapter.build_reply_parts(parsed, finish))
    upstream = _without_ids(openai_adapter._build_reply_parts(parsed, finish))
    if parsed.reasoning:
        assert ours[1].pop("reasoning_content") == parsed.reasoning, "manager copy keeps what Leaf echoes"
    assert ours == upstream


def test_reply_parts_keep_all_calls_in_both_messages():
    calls = [{"name": "Bash", "input": {"command": "ls"}}, {"name": "Read", "input": {"file_path": "a"}}]
    wire, manager, finish = leaf_adapter.build_reply_parts(ParsedModelOutput("", "", calls), "stop")
    assert [c["function"]["name"] for c in wire["tool_calls"]] == ["Bash", "Read"]
    assert manager["tool_calls"] == [
        {"type": "function", "function": {"name": "Bash", "arguments": {"command": "ls"}}},
        {"type": "function", "function": {"name": "Read", "arguments": {"file_path": "a"}}},
    ], "manager copy: dict arguments and no ids, so it matches Leaf's echo"
    assert finish == "tool_calls"


def test_loop_runs_every_call_in_order_and_stays_one_sample(make_adapter, workdir):
    server = make_adapter([WRITE_THEN_READ, "all done"], adapter_cls=LeafChatAdapter)
    result, log = _run(server, workdir)

    assert result["exit_reason"] == "agent" and result["n_steps"] == 2
    roles = [m["role"] for m in result["messages"]]
    assert roles == ["system", "user", "assistant", "tool", "tool", "assistant"]
    calls = result["messages"][2]["tool_calls"]
    results = result["messages"][3:5]
    assert [r["tool_call_id"] for r in results] == [c["id"] for c in calls]
    assert results[0]["content"] == "Wrote out.txt (5 bytes)."
    assert results[1]["content"] == "out.txt\n1. hello", "Read ran after Write, in call order"
    assert log[0]["tool_calls"] == ["Write", "Read"]

    samples = server.finish()
    assert len(samples) == 1, "the echo of a two-call turn still matches the recorded history"
    assert _trained_text(samples[0], server.tok) == "all done"



# --- task.py + generate.py: Leaf's own rollout pipeline (no other example imported) ---

import dataclasses  # noqa: E402

from leaf_agent import generate as leaf_generate  # noqa: E402
from leaf_agent import task as leaf_task  # noqa: E402
from leaf_agent.task import Task  # noqa: E402

from slime.agent.sandbox import E2BSandbox  # noqa: E402
from slime.utils.misc import SingletonMeta  # noqa: E402


def test_no_leaf_module_imports_another_example():
    for path in sorted(LEAF_DIR.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                assert not name.startswith("examples."), (path.name, name)


def _task_sample(**md):
    meta = {
        "instance_id": "demo-1",
        "image": "fake-image",
        "workdir": "/workspace/repo",
        "problem_statement": "fix the bug",
        "eval_cmd": "true",
        **md,
    }
    return Sample(index=0, group_index=0, prompt="the prompt", metadata=meta)


def test_task_from_sample_and_missing_fields():
    task = Task.from_sample(_task_sample())
    assert (task.instance_id, task.image, task.workdir, task.eval_cmd, task.eval_user) == (
        "demo-1",
        "fake-image",
        "/workspace/repo",
        "true",
        "agent",
    )
    assert task.missing() is None
    assert Task.from_sample(_task_sample(problem_statement=None)).problem_statement == "the prompt"
    assert Task.from_sample(Sample(index=0, prompt="p")).instance_id == "unknown"
    for field in ("image", "workdir", "eval_cmd"):
        assert Task.from_sample(_task_sample(**{field: ""})).missing() == f"missing_{field}"


def test_task_prompt_inlines_the_issue_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv(leaf_task.PROMPT_ENV, raising=False)
    prompt = Task.from_sample(_task_sample(problem_statement="Bug: {x} crashes")).prompt()
    assert prompt.startswith("Bug: {x} crashes\n\n"), "braces in the issue survive str.format"
    assert "/workspace/repo" in prompt
    monkeypatch.setenv(leaf_task.PROMPT_ENV, "Fix in {workdir}: {problem_statement}")
    assert Task.from_sample(_task_sample()).prompt() == "Fix in /workspace/repo: fix the bug"


def test_create_sandbox_defaults_to_e2b_and_is_pluggable(monkeypatch):
    monkeypatch.delenv(leaf_task.SANDBOX_FACTORY_ENV, raising=False)
    sb = leaf_task.create_sandbox("img")
    assert isinstance(sb, E2BSandbox) and sb.image == "img"

    monkeypatch.setenv(leaf_task.SANDBOX_FACTORY_ENV, "tests.test_agent._fakes:FakeSandbox")
    sb = leaf_task.create_sandbox("img")
    assert isinstance(sb, FakeSandbox) and sb.image == "img"

    monkeypatch.setenv(leaf_task.SANDBOX_FACTORY_ENV, "tests.test_agent._fakes")
    with pytest.raises(ValueError, match="package.module:callable"):
        leaf_task.create_sandbox("img")


def test_workspace_snapshot_and_diff_exclude_harness_files():
    sb = FakeSandbox(responses=[("git diff --binary", (0, "the diff", ""))])
    baseline = asyncio.run(leaf_task.prepare_workspace(sb, "/workspace/repo"))
    assert baseline == "0" * 40
    assert asyncio.run(leaf_task.git_diff(sb, "/workspace/repo", baseline)) == "the diff"
    git_cmds = [cmd for cmd, _ in sb.exec_log if "git add -A" in cmd or "git diff" in cmd]
    assert len(git_cmds) == 3 and all("':(exclude).harness/'" in cmd for cmd in git_cmds)
    assert any("useradd" in cmd for cmd, _ in sb.exec_log), "prepare_workspace creates the agent user"


def _grading_sandbox(monkeypatch, **kwargs):
    made = []

    def factory(image):
        made.append(FakeSandbox(image, **kwargs))
        return made[-1]

    monkeypatch.setattr(leaf_task, "create_sandbox", factory)
    return made


@pytest.mark.parametrize(
    "diff, responses, expected",
    [
        ("", [], (1.0, True)),
        ("--- a\n+++ b\n", [], (1.0, True)),
        ("--- a\n+++ b\n", [("&& pytest -q", (1, "", "1 failed"))], (0.0, True)),
        ("--- a\n+++ b\n", [("git apply", (1, "", "does not apply"))], (0.0, False)),
    ],
)
def test_grade_applies_the_diff_in_a_fresh_sandbox(monkeypatch, diff, responses, expected):
    made = _grading_sandbox(monkeypatch, responses=responses)
    task = Task.from_sample(_task_sample(eval_cmd="pytest -q", eval_user="root"))
    assert asyncio.run(leaf_task.grade(task, diff, timeout_sec=30)) == expected

    (ev,) = made
    assert ev.image == "fake-image", "graded on a fresh sandbox of the task image"
    assert (ev.files.get("/tmp/__leaf_patch__.diff") == diff) == bool(diff.strip())
    eval_runs = [(cmd, user) for cmd, user in ev.exec_log if cmd.endswith("&& pytest -q")]
    assert eval_runs == ([] if expected[1] is False else [("cd /workspace/repo && pytest -q", "root")])


def _patch_leaf_generate(monkeypatch, tok, script, on_launch, **config):
    """Wire generate()'s external edges to CPU fakes: tokenizer, sandboxes, sglang, polling."""
    monkeypatch.setattr(
        leaf_generate,
        "CONFIG",
        dataclasses.replace(
            leaf_generate.CONFIG,
            **{
                "adapter_public_host": "127.0.0.1",
                "adapter_bind_host": "127.0.0.1",
                "adapter_port": 0,
                "rollout_guard_sec": 60,
                "agent_time_budget_sec": 30,
                "eval_timeout_sec": 30,
                "boot_retries": 1,
                **config,
            },
        ),
    )
    monkeypatch.setattr(leaf_generate, "load_tokenizer", lambda _path: tok)
    made = []

    def factory(image):
        made.append(FakeSandbox(image, on_launch=on_launch))
        return made[-1]

    monkeypatch.setattr(leaf_task, "create_sandbox", factory)
    monkeypatch.setattr(harness_common.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(adapters_common, "call_sglang_generate", fake_call_sglang_generate(script, tok))
    for name in (LeafHarness.max_steps_env, LeafHarness.max_completion_tokens_env, LeafHarness.extra_envs_env):
        monkeypatch.delenv(name, raising=False)
    SingletonMeta.clear_instances(leaf_generate._AdapterService)
    return made


def _gen_args():
    return SimpleNamespace(
        hf_checkpoint="unused",
        rollout_max_context_len=0,
        sglang_tool_call_parser=None,
        sglang_reasoning_parser=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=1,
    )


def _leaf_launcher(tmp_path, results):
    async def leaf_agent(env):
        result = await asyncio.to_thread(_thread_leaf, env, "fix the bug", tmp_path)
        results.append(result)
        return 0 if result["exit_reason"] in leaf_cli.NORMAL_EXITS else 1

    return leaf_agent


TWO_CALLS_THEN_DONE = [(WRITE_THEN_READ, "stop", None), ("all done", "stop", None)]


def test_generate_runs_a_leaf_rollout_end_to_end(monkeypatch, tmp_path):
    """Leaf's own generate(): boot + install, prep, harness, loop, diff, fresh-sandbox eval, Samples."""
    tok, results = FakeTokenizer(), []
    sandboxes = _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, _leaf_launcher(tmp_path, results))
    monkeypatch.delenv(leaf_task.PROMPT_ENV, raising=False)
    base = _task_sample(workdir=str(tmp_path))

    samples = asyncio.run(leaf_generate.generate(_gen_args(), base, sampling_params={"max_new_tokens": 32}))

    (result,) = results
    assert result["exit_reason"] == "agent" and result["n_steps"] == 2
    assert (tmp_path / "out.txt").read_text() == "hello"
    assert re.fullmatch(r"leaf-demo-1-0-0-[0-9a-f]{16}", base.session_id), "traceable prefix + unguessable suffix"
    agent_sb, eval_sb = sandboxes
    assert agent_sb.files[f"{tmp_path}/.harness/prompt.txt"] == Task.from_sample(base).prompt()
    assert not any(cmd.endswith("leaf_cli.py") or "setsid" in cmd for cmd, _ in eval_sb.exec_log), "eval runs no agent"

    # fork_threshold_tokens=SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS (default 0): a
    # turn whose re-rendered history drifts is split into its own Sample instead
    # of being merged and masked (the adapter default, 1024, that the
    # AdapterHarness tests use). FakeTokenizer re-renders tool calls differently
    # from how they were generated, so the two-call turn splits off and is
    # trained intact. With a real tokenizer this is what Step 7 checks.
    assert [_trained_text(s, tok) for s in samples] == [WRITE_THEN_READ, "all done"]
    for sample in samples:
        assert sample.status == Sample.Status.COMPLETED
        assert sample.reward == 1.0, "eval_cmd `true` passes; the outcome reward is copied to every Sample"
        assert sample.metadata["agent"] == "leaf" and sample.metadata["instance_id"] == "demo-1"
        assert sample.metadata["agent_exit_code"] == 0 and sample.metadata["applied_cleanly"] is True
        assert sample.metadata["grading_solved"] is True
        assert len(sample.loss_mask) == len(sample.rollout_log_probs) == sample.response_length

    state = leaf_generate._AdapterService(_gen_args())
    assert type(state.adapter) is LeafChatAdapter
    assert state.adapter_url.startswith("http://127.0.0.1:")


def test_generate_reward_is_zero_when_eval_fails(monkeypatch, tmp_path):
    tok, results = FakeTokenizer(), []
    _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, _leaf_launcher(tmp_path, results))
    monkeypatch.setattr(
        leaf_task,
        "create_sandbox",
        FakeSandbox.factory(on_launch=_leaf_launcher(tmp_path, results), responses=[("&& false", (1, "", ""))]),
    )
    samples = asyncio.run(leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path), eval_cmd="false"), {}))
    assert samples and all(s.reward == 0.0 and s.metadata["grading_solved"] is False for s in samples)


def test_generate_eval_mode_returns_one_reward_placeholder(monkeypatch, tmp_path):
    tok, results = FakeTokenizer(), []
    _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, _leaf_launcher(tmp_path, results))
    (sample,) = asyncio.run(
        leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path)), {}, evaluation=True)
    )
    assert sample.status == Sample.Status.COMPLETED and sample.remove_sample
    assert sample.reward == 1.0 and sample.metadata["grading_solved"] is True
    assert sample.metadata["agent_exit_code"] == 0


@pytest.mark.parametrize(
    "md, launcher, reason",
    [
        ({"eval_cmd": ""}, None, "missing_eval_cmd"),
        ({"image": ""}, None, "missing_image"),
        ({}, "silent", "adapter_session_empty"),
        ({}, "crash", "exception:RuntimeError"),
    ],
)
def test_generate_aborts_cleanly(monkeypatch, tmp_path, md, launcher, reason):
    async def silent(_env):
        return 0  # never calls the model

    async def crash(_env):
        raise RuntimeError("sandbox died")

    tok = FakeTokenizer()
    _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, {"silent": silent, "crash": crash}.get(launcher))
    (sample,) = asyncio.run(leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path), **md), {}))
    assert sample.status == Sample.Status.ABORTED and sample.remove_sample
    assert sample.metadata["abort_reason"] == reason
    assert sample.metadata["agent"] == "leaf"


def test_public_url_override_for_nat_port_mappings(monkeypatch, tmp_path):
    tok = FakeTokenizer()
    _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, None, adapter_public_host=None)
    monkeypatch.setenv("SLIME_AGENT_LEAF_ADAPTER_PUBLIC_URL", "http://203.0.113.7:40123/")
    assert leaf_generate._AdapterService(_gen_args()).adapter_url == "http://203.0.113.7:40123"


def test_generate_requires_a_public_adapter_host(monkeypatch, tmp_path):
    tok = FakeTokenizer()
    _patch_leaf_generate(monkeypatch, tok, TWO_CALLS_THEN_DONE, None, adapter_public_host=None)
    monkeypatch.delenv("SLIME_AGENT_LEAF_ADAPTER_PUBLIC_URL", raising=False)
    with pytest.raises(RuntimeError, match="ADAPTER_PUBLIC_HOST"):
        asyncio.run(leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path)), {}))



# ===========================================================================
# Step 6: limits and failure paths, end to end
# ===========================================================================


def test_non_json_tool_arguments_become_an_error_observation(workdir, monkeypatch):
    """A model that emits broken JSON arguments gets an error back and keeps going."""
    replies = iter(
        [
            {
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {"id": "c1", "type": "function", "function": {"name": "Read", "arguments": "{not json"}}
                            ],
                        },
                    }
                ]
            },
            {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "recovered"}}]},
        ]
    )
    monkeypatch.setattr(leaf_cli, "complete", lambda *a, **k: next(replies))
    result = leaf_cli.run_leaf("http://unused/v1", SID, "p", workdir=str(workdir), log=lambda _: None)

    assert result["exit_reason"] == "agent" and result["final_message"] == "recovered"
    observation = result["messages"][3]
    assert observation["role"] == "tool" and observation["tool_call_id"] == "c1"
    assert observation["content"] == tool_runner.run_tool("Read", {}), "arguments fell back to {}"
    assert "Error" in observation["content"]


def _rollout(monkeypatch, tmp_path, script, *, launcher=None, env=None):
    tok, results = FakeTokenizer(), []
    _patch_leaf_generate(monkeypatch, tok, script, launcher or _leaf_launcher(tmp_path, results))
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    samples = asyncio.run(leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path)), {}))
    return samples, results, tok


def test_step_limit_still_emits_samples(monkeypatch, tmp_path):
    (tmp_path / "out.txt").write_text("x\n")
    samples, results, _ = _rollout(
        monkeypatch,
        tmp_path,
        [(READ_OUT, "stop", None)] * 3,
        env={LeafHarness.max_steps_env: "2"},
    )
    (result,) = results
    assert result["exit_reason"] == "max_turns" and result["n_steps"] == 2
    assert samples and all(s.status == Sample.Status.COMPLETED for s in samples)
    assert all(s.metadata["agent_exit_code"] == 0 for s in samples), "a step limit is a normal exit"


def test_context_full_ends_the_rollout_cleanly(monkeypatch, tmp_path):
    samples, results, _ = _rollout(monkeypatch, tmp_path, [(WRITE_OUT, "stop", None), ("", "length", None)])
    (result,) = results
    assert result["exit_reason"] == "max_context_len"
    assert (tmp_path / "out.txt").read_text() == "hello", "work done before the overflow is kept"
    assert samples and all(s.status == Sample.Status.COMPLETED for s in samples)
    assert all(s.metadata["agent_exit_code"] == 0 and s.reward == 1.0 for s in samples)


def test_llm_error_mid_run_keeps_the_turns_so_far(monkeypatch, tmp_path):
    """The sglang side fails on turn 2: Leaf exits 1, but turn 1 is still trained and graded."""
    calls = {"n": 0}
    tok = FakeTokenizer()
    real = fake_call_sglang_generate([(WRITE_OUT, "stop", None)], tok)

    async def flaky(prompt_ids, session, body, *, adapter, session_id=None):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("sglang went away")
        return await real(prompt_ids, session, body, adapter=adapter, session_id=session_id)

    results = []
    _patch_leaf_generate(monkeypatch, tok, [], _leaf_launcher(tmp_path, results))
    monkeypatch.setattr(adapters_common, "call_sglang_generate", flaky)
    samples = asyncio.run(leaf_generate.generate(_gen_args(), _task_sample(workdir=str(tmp_path)), {}))

    (result,) = results
    assert result["exit_reason"] == "llm_query_error"
    assert samples and all(s.status == Sample.Status.COMPLETED for s in samples)
    assert all(s.metadata["agent_exit_code"] == 1 for s in samples), "the error is visible on every Sample"
    assert [_trained_text(s, tok) for s in samples] == [WRITE_OUT]


def test_agent_killed_at_the_time_budget_keeps_its_turns(monkeypatch, tmp_path):
    results = []
    run_then_hang = _leaf_launcher(tmp_path, results)

    async def killed(env):
        await run_then_hang(env)
        return sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED  # what run_agent reports when the marker never appears

    samples, _, tok = _rollout(monkeypatch, tmp_path, TWO_CALLS_THEN_DONE, launcher=killed)
    assert samples and all(s.metadata["agent_exit_code"] == sandbox_mod.EXIT_TIME_BUDGET_EXCEEDED for s in samples)
    assert all(s.status == Sample.Status.COMPLETED for s in samples)



# ===========================================================================
# Step 7: real tokenizer (Qwen3-Coder chat template, FrogNano's tool format)
# ===========================================================================
#
# Only the tokenizer is real; sglang is still scripted. The scripted outputs end
# in <|im_end|> as sglang's do (see test_prompt_continuity.py). Tool calls use
# the Qwen3-Coder XML format, which slime's fallback parser reads, so no sglang
# parser is needed. Skipped when transformers or the tokenizer is unavailable.

REAL_TOKENIZER = "Qwen/Qwen3-Coder-30B-A3B-Instruct"


@pytest.fixture(scope="module")
def real_tok():
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoTokenizer.from_pretrained(REAL_TOKENIZER)
    except Exception as exc:  # offline, no HF access, ...
        pytest.skip(f"cannot load {REAL_TOKENIZER}: {exc}")


def _coder_call(name, **params):
    """A tool call exactly as the Qwen3-Coder chat template renders one."""
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={name}>\n{body}</function>\n</tool_call>"


CODER_TWO_CALLS = _coder_call("Write", file_path="out.txt", content="hello") + "\n" + _coder_call("Read", file_path="out.txt")


def _real_session(monkeypatch, workdir, tok, adapter_cls, outputs):
    """Run the Leaf loop through ``adapter_cls`` with production fork_threshold 0; return (result, samples)."""
    script = [(text + "<|im_end|>", "stop", None) for text in outputs]
    monkeypatch.setattr(adapters_common, "call_sglang_generate", fake_call_sglang_generate(script, tok))
    adapter = adapter_cls(tokenizer=tok, sglang_url="http://unused", fork_threshold_tokens=0)
    handle = run_app_in_thread(adapter.app, host="127.0.0.1", port=0, thread_name="test-leaf-real-tok")
    adapter.open_session(SID)
    try:
        result = leaf_cli.run_leaf(
            f"http://127.0.0.1:{handle.port}/v1", SID, "fix the bug", workdir=str(workdir), request_timeout=30,
            log=lambda _line: None,
        )  # fmt: skip
        samples = asyncio.run(adapter.finish_session(SID, base_sample=Sample(index=0, prompt=""), reward=1.0))
    finally:
        handle.stop()
    return result, samples


def test_real_template_trains_every_turn_in_one_sample(monkeypatch, workdir, real_tok):
    """The RL property that matters: tool-call turns are CLEAN, so they are trained, not masked."""
    result, samples = _real_session(monkeypatch, workdir, real_tok, LeafChatAdapter, [CODER_TWO_CALLS, "all done"])

    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "tool", "tool", "assistant"]
    assert result["messages"][4]["content"] == "out.txt\n1. hello", "both calls ran, in order"
    (sample,) = samples
    assert _trained_text(sample, real_tok) == CODER_TWO_CALLS + "<|im_end|>" + "all done<|im_end|>", (
        "exactly the policy's tokens are trained; prompts and tool results are masked"
    )
    assert 0 < sample.loss_mask.count(0) < len(sample.loss_mask)


def test_plain_openai_adapter_trains_a_call_that_never_ran(monkeypatch, workdir, real_tok):
    """Why LeafChatAdapter exists, with a real template.

    Upstream truncates the reply to the first call and stores that truncated
    message as history, so Leaf's echo still matches and nothing drifts: the
    session stays one CLEAN Sample. But the trained tokens are what the model
    generated -- both calls -- while only the first one was executed. The policy
    is trained on a Read whose result it never saw; no drift counter shows it.
    """
    result, samples = _real_session(monkeypatch, workdir, real_tok, OpenAIAdapter, [CODER_TWO_CALLS, "all done"])

    assert [m["role"] for m in result["messages"]] == ["system", "user", "assistant", "tool", "assistant"]
    assert [c["function"]["name"] for c in result["messages"][2]["tool_calls"]] == ["Write"], "Read never ran"
    (sample,) = samples
    assert "<function=Read>" in _trained_text(sample, real_tok), "...yet the Read call is trained"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
