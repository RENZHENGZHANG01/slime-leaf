from __future__ import annotations

import json
import os
import shlex
from pathlib import Path

from slime.agent.sandbox import ExecResult, FileContent

IMAGES_ENV = "SLIME_AGENT_LEAF_MODAL_IMAGES"
APP_ENV = "SLIME_AGENT_LEAF_MODAL_APP"
ALLOW_OUT_ENV = "SLIME_AGENT_LEAF_MODAL_ALLOW_OUT"
DEFAULT_BASE = "python:3.11-slim"

_IMAGES: dict[str, object] = {}


def _image(image: str, info: dict):
    """The Modal image for a task: base + git + the repository at workdir (built lazily, cached)."""
    import modal

    if image not in _IMAGES:
        _IMAGES[image] = (
            modal.Image.from_registry(info.get("base") or DEFAULT_BASE)
            .apt_install("git")
            .add_local_dir(info["source"], info["workdir"], copy=True)
        )
    return _IMAGES[image]


class ModalSandbox:
    def __init__(self, image: str) -> None:
        self.image = image
        self.info = json.loads(Path(os.environ[IMAGES_ENV]).read_text())[image]
        self.sandbox_id = ""
        self._sb = None

    async def __aenter__(self) -> ModalSandbox:
        import modal

        app = await modal.App.lookup.aio(os.environ.get(APP_ENV) or "slime-leaf-sandboxes", create_if_missing=True)
        kwargs = {
            "app": app,
            "image": _image(self.image, self.info),
            "timeout": int(os.environ.get("SLIME_AGENT_LEAF_MODAL_LIFETIME_SEC") or 3600),
            "tags": {"slime_image": self.image},
        }
        if cpu := os.environ.get("SLIME_AGENT_LEAF_MODAL_CPU"):
            kwargs["cpu"] = float(cpu)
        if mem := os.environ.get("SLIME_AGENT_LEAF_MODAL_MEMORY_MB"):
            kwargs["memory"] = int(mem)
        if allow := [a.strip() for a in os.environ.get(ALLOW_OUT_ENV, "").split(",") if a.strip()]:
            # Only the adapter (and whatever else is listed) is reachable; FrogNano's
            # model card asks for network access disabled by default.
            kwargs["outbound_cidr_allowlist"] = allow
        self._sb = await modal.Sandbox.create.aio("sleep", "infinity", **kwargs)
        self.sandbox_id = self._sb.object_id
        return self

    async def __aexit__(self, *exc) -> None:
        if self._sb is not None:
            await self._sb.terminate.aio()

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        argv = ["bash", "-c", cmd] if user == "root" else ["runuser", "-u", user, "--", "bash", "-c", cmd]
        proc = await self._sb.exec.aio(*argv, env=env or None, timeout=timeout)
        out = await proc.stdout.read.aio()
        err = await proc.stderr.read.aio()
        code = await proc.wait.aio()
        if check and code != 0:
            raise RuntimeError(f"modal exec failed (exit={code}): {cmd[:120]}\n{err[-400:]}")
        return code, out, err

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        data = content.read_bytes() if isinstance(content, Path) else content
        data = data.encode() if isinstance(data, str) else data
        await self._sb.filesystem.make_directory.aio(os.path.dirname(sandbox_path) or "/", create_parents=True)
        await self._sb.filesystem.write_bytes.aio(data, sandbox_path)
        if user != "root":
            await self.exec(f"chown {shlex.quote(user)}: {shlex.quote(sandbox_path)}", check=True)

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._sb.filesystem.read_text.aio(sandbox_path)
        except Exception:
            return ""
