from __future__ import annotations

import importlib
import logging
import os
import shlex
from dataclasses import dataclass

from slime.agent import sandbox as agent_sandbox
from slime.agent.sandbox import E2BSandbox, Sandbox
from slime.utils.types import Sample

logger = logging.getLogger(__name__)

#: "package.module:callable" taking an image and returning a Sandbox; unset = E2B.
SANDBOX_FACTORY_ENV = "SLIME_AGENT_LEAF_SANDBOX_FACTORY"
PROMPT_ENV = "SLIME_AGENT_LEAF_PROMPT"

# The whole issue goes in the prompt, as FrogNano does with instruction.md.
DEFAULT_PROMPT = (
    "{problem_statement}\n\n"
    "The repository is checked out at {workdir}. Resolve the issue above by editing "
    "source files; do not modify tests and do not commit. Run the relevant tests to "
    "check your fix, then give a one-line summary."
)

# Excluded from both the baseline snapshot and the diff: run_agent's logs and
# the prompt file live in .harness/.
_EXCLUDE = "':(exclude).harness/'"
_PATCH = "/tmp/__leaf_patch__.diff"


@dataclass(frozen=True)
class Task:
    instance_id: str
    image: str
    workdir: str
    problem_statement: str
    eval_cmd: str
    eval_user: str = "agent"

    @classmethod
    def from_sample(cls, sample: Sample) -> Task:
        m = sample.metadata or {}
        prompt = sample.prompt if isinstance(sample.prompt, str) else ""
        return cls(
            instance_id=str(m.get("instance_id") or "unknown"),
            image=m.get("image") or "",
            workdir=m.get("workdir") or "",
            problem_statement=m.get("problem_statement") or prompt,
            eval_cmd=m.get("eval_cmd") or "",
            eval_user=m.get("eval_user") or "agent",
        )

    def missing(self) -> str | None:
        """Why this row cannot be rolled out, or None."""
        for field in ("image", "workdir", "eval_cmd"):
            if not getattr(self, field):
                return f"missing_{field}"
        return None

    def prompt(self) -> str:
        template = os.environ.get(PROMPT_ENV) or DEFAULT_PROMPT
        return template.format(problem_statement=self.problem_statement, workdir=self.workdir)


def create_sandbox(image: str) -> Sandbox:
    """A sandbox for ``image``, from SLIME_AGENT_LEAF_SANDBOX_FACTORY if set, else E2B."""
    spec = os.environ.get(SANDBOX_FACTORY_ENV, "").strip()
    if not spec:
        return E2BSandbox(image)
    module, _, attr = spec.partition(":")
    if not attr:
        raise ValueError(f"{SANDBOX_FACTORY_ENV}={spec!r}: expected 'package.module:callable'")
    return getattr(importlib.import_module(module), attr)(image)


async def prepare_workspace(sb: Sandbox, workdir: str) -> str:
    """Create the agent user and return the workspace's starting Git tree."""
    await agent_sandbox.ensure_agent_user(sb, workdir)
    return await _repository_tree(sb, workdir)


async def git_diff(sb: Sandbox, workdir: str, baseline: str) -> str:
    """Everything the agent changed since ``baseline``, as a binary-safe patch."""
    current = await _repository_tree(sb, workdir)
    cmd = (
        f"cd {shlex.quote(workdir)} && git diff --binary {shlex.quote(baseline)} {shlex.quote(current)}"
        f" -- . {_EXCLUDE}"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120, check=True)
    return out


async def _repository_tree(sb: Sandbox, workdir: str) -> str:
    # Snapshot the working tree through a temporary index, so staged edits,
    # untracked files and the image's own uncommitted state are all captured
    # without touching HEAD or the agent's real index.
    cmd = (
        f"set -e\ncd {shlex.quote(workdir)}\n"
        'tmp_index=$(mktemp)\nrm -f "$tmp_index"\n'
        'trap \'rm -f "$tmp_index" "$tmp_index.lock"\' EXIT\n'
        'export GIT_INDEX_FILE="$tmp_index"\n'
        "git read-tree HEAD\n"
        f"git add -A -- . {_EXCLUDE}\n"
        "git write-tree"
    )
    _, out, _ = await sb.exec(cmd, user="agent", timeout=120, check=True)
    tree = out.strip()
    if len(tree) not in (40, 64) or any(c not in "0123456789abcdef" for c in tree):
        raise RuntimeError(f"Invalid repository snapshot: {tree!r}")
    return tree


async def grade(task: Task, diff_text: str, *, timeout_sec: int) -> tuple[float, bool]:
    """(reward, applied_cleanly): apply the diff to a fresh sandbox and run eval_cmd."""
    async with create_sandbox(task.image) as ev:
        await agent_sandbox.ensure_agent_user(ev, task.workdir)
        if not await _apply_diff(ev, task.workdir, diff_text):
            return 0.0, False
        ec, _, _ = await ev.exec(
            f"cd {shlex.quote(task.workdir)} && {task.eval_cmd}",
            user=task.eval_user,
            check=False,
            timeout=timeout_sec,
        )
        return (1.0 if ec == 0 else 0.0), True


async def _apply_diff(ev: Sandbox, workdir: str, diff_text: str) -> bool:
    if not diff_text.strip():
        return True  # nothing to apply; eval_cmd decides (usually 0 reward)
    await ev.write_file(_PATCH, diff_text, user="agent")
    ladder = " || ".join(
        f"({cmd})"
        for cmd in (
            f"git apply --whitespace=nowarn {_PATCH}",
            f"git apply --3way --whitespace=nowarn {_PATCH}",
            f"patch -p1 --no-backup-if-mismatch < {_PATCH}",
        )
    )
    ec, _, _ = await ev.exec(f"cd {shlex.quote(workdir)} && ({ladder})", user="agent", check=False, timeout=120)
    return ec == 0
