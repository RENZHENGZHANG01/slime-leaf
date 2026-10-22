from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.harness.common import BaseHarness, HarnessContext, run_agent
from slime.agent.sandbox import Sandbox

LEAF_DIR = Path(__file__).resolve().parent
#: The files the sandbox needs; adapter.py / harness.py stay on the host.
SANDBOX_FILES = ("leaf_cli.py", "tool_runner.py", "tools.py")

# From FrogNano runtimes/python.py: prefer the system python3 over a repo venv
# that may sit earlier on PATH, then fall back to whatever is there.
FIND_PYTHON = (
    "python_bin=$(test -x /usr/bin/python3 && echo /usr/bin/python3 "
    "|| command -v python3 || command -v python)"
)
CHECK_PYTHON = (
    f'{FIND_PYTHON}; test -n "$python_bin" && '
    '"$python_bin" -I -S -c "import sys; sys.exit(sys.version_info < (3, 6))"'
)


class LeafHarness(BaseHarness):
    name = "leaf"

    install_dir = "/opt/leaf"

    # host-side knobs, under the agent-layer SLIME_AGENT_* prefix
    max_steps_env = "SLIME_AGENT_LEAF_MAX_STEPS"
    max_completion_tokens_env = "SLIME_AGENT_LEAF_MAX_COMPLETION_TOKENS"
    extra_envs_env = "SLIME_AGENT_LEAF_EXTRA_ENVS"

    # Leaf stops itself this long before run_agent's hard deadline, so it exits
    # between steps (workspace intact, exit code 0) instead of being cut off
    # mid-edit. Capped at a quarter of the budget for short budgets.
    exit_margin_sec = 60

    async def install_cli(self, sb: Sandbox) -> None:
        await sb.exec(f"mkdir -p {self.install_dir}", user="root", check=True, timeout=30)
        for name in SANDBOX_FILES:
            await sb.write_file(f"{self.install_dir}/{name}", (LEAF_DIR / name).read_text(encoding="utf-8"))
        # Readable by the unprivileged agent user; fail the boot (and let the
        # caller retry / skip the image) if no python3 >= 3.6 exists.
        await sb.exec(
            f"chmod -R a+rX {self.install_dir} && {CHECK_PYTHON}",
            user="root",
            check=True,
            timeout=60,
        )

    async def write_config(self, sb: Sandbox, ctx: HarnessContext) -> None:
        """Nothing to write: leaf_cli.py takes all of its settings from env vars."""
        return None

    async def launch_and_wait(self, sb: Sandbox, ctx: HarnessContext, prompt: str, time_budget_sec: int) -> int:
        # The prompt goes in a file, not argv, so a long or quote-heavy prompt
        # never has to survive shell quoting.
        meta_dir = f"{ctx.workdir}/.harness"
        prompt_path = f"{meta_dir}/prompt.txt"
        await sb.exec(f"mkdir -p {meta_dir} && chown agent:agent {meta_dir}", user="root", check=True, timeout=30)
        await sb.write_file(prompt_path, prompt)
        await sb.exec(f"chown agent:agent {prompt_path}", user="root", check=True, timeout=30)

        # No `exec`: the launcher appends `echo $? > done` after this command.
        cmd = f'{FIND_PYTHON}; "$python_bin" {self.install_dir}/leaf_cli.py {shlex.quote(prompt_path)}'
        margin = min(self.exit_margin_sec, time_budget_sec // 4)
        env = {
            # leaf_cli pops both before running any tool, so Bash never sees them.
            "OPENAI_BASE_URL": f"{ctx.adapter_url}/v1",
            "OPENAI_API_KEY": ctx.session_id,
            "LEAF_WORKDIR": ctx.workdir,
            "LEAF_MAX_TIME_SEC": str(time_budget_sec - margin),
        }
        if steps := os.environ.get(self.max_steps_env, "").strip():
            env["LEAF_MAX_STEPS"] = steps
        if tokens := os.environ.get(self.max_completion_tokens_env, "").strip():
            env["LEAF_MAX_COMPLETION_TOKENS"] = tokens
        # extra env vars as a JSON object, merged last so callers can override
        # the defaults above
        extra_envs = os.environ.get(self.extra_envs_env, "").strip()
        if extra_envs:
            env.update(json.loads(extra_envs))
        return await run_agent(sb, workdir=ctx.workdir, start_cmd=cmd, env=env, time_budget_sec=time_budget_sec)
