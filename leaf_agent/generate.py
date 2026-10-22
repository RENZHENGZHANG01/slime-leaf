from __future__ import annotations

import asyncio
import logging
import os
import random
import secrets
import time
import traceback
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from slime.agent.aiohttp_threaded import FilteredAccessLogger, run_app_in_thread
from slime.agent.sandbox import Sandbox
from slime.utils.misc import SingletonMeta, get_current_node_ip
from slime.utils.types import Sample

from . import task as leaf_task
from .adapter import LeafChatAdapter
from .harness import LeafHarness

logger = logging.getLogger(__name__)

AGENT = "leaf"


@dataclass(frozen=True)
class LeafConfig:
    adapter_public_host: str | None
    adapter_bind_host: str
    adapter_port: int
    fork_merge_threshold: int
    agent_time_budget_sec: int
    eval_timeout_sec: int
    rollout_guard_sec: int
    boot_concurrency: int
    boot_retries: int

    @classmethod
    def from_env(cls) -> LeafConfig:
        budget = int(os.environ.get("SLIME_AGENT_LEAF_TIME_BUDGET_SEC", "1800"))
        eval_timeout = int(os.environ.get("SLIME_AGENT_LEAF_EVAL_TIMEOUT_SEC", "600"))
        guard = int(os.environ.get("SLIME_AGENT_LEAF_ROLLOUT_GUARD_SEC", "0") or 0) or (budget + eval_timeout + 180)
        return cls(
            adapter_public_host=os.environ.get("ADAPTER_PUBLIC_HOST"),
            adapter_bind_host=os.environ.get("ADAPTER_BIND_HOST", "0.0.0.0"),
            adapter_port=int(os.environ.get("ADAPTER_PORT", "18001")),
            fork_merge_threshold=int(os.environ.get("SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS") or "0"),
            agent_time_budget_sec=budget,
            eval_timeout_sec=eval_timeout,
            rollout_guard_sec=guard,
            boot_concurrency=int(os.environ.get("SLIME_AGENT_LEAF_BOOT_CONCURRENCY", "16")),
            boot_retries=int(os.environ.get("SLIME_AGENT_LEAF_BOOT_RETRIES", "2")),
        )


CONFIG = LeafConfig.from_env()

_BOOT_SEM = asyncio.Semaphore(CONFIG.boot_concurrency)


def load_tokenizer(path: str):
    # Imported lazily: processing_utils pulls in transformers.
    from slime.utils.processing_utils import load_tokenizer as _load

    return _load(path, trust_remote_code=True)


class _AdapterService(metaclass=SingletonMeta):
    """One LeafChatAdapter per rollout worker, serving every Leaf session."""

    def __init__(self, args) -> None:
        public_url = os.environ.get("SLIME_AGENT_LEAF_ADAPTER_PUBLIC_URL", "").strip().rstrip("/")
        if not (CONFIG.adapter_public_host or public_url):
            raise RuntimeError(
                "ADAPTER_PUBLIC_HOST is not set. Export the host IP that sandboxes can reach "
                "(or 'auto'); without it Leaf cannot dial the adapter back."
            )
        self.tokenizer = load_tokenizer(args.hf_checkpoint)
        self.max_context_len = int(getattr(args, "rollout_max_context_len", 0) or 0)
        self.adapter = LeafChatAdapter(
            tokenizer=self.tokenizer,
            sglang_url=f"http://{args.sglang_router_ip}:{args.sglang_router_port}",
            tool_parser=getattr(args, "sglang_tool_call_parser", None) or None,
            reasoning_parser=getattr(args, "sglang_reasoning_parser", None) or None,
            fork_threshold_tokens=CONFIG.fork_merge_threshold,
            rollout_args=args,
        )
        # handler_cancellation: a client disconnect cancels the handler, which
        # aborts the in-flight sglang request instead of leaving it running.
        self.app_handle = run_app_in_thread(
            self.adapter.app,
            host=CONFIG.adapter_bind_host,
            port=CONFIG.adapter_port,
            thread_name="leaf-adapter",
            runner_kwargs={"handler_cancellation": True, "access_log_class": FilteredAccessLogger},
        )
        if public_url:
            self.adapter_url = public_url  # e.g. a NAT'd public IP:port; the local port differs
        else:
            host = get_current_node_ip() if CONFIG.adapter_public_host == "auto" else CONFIG.adapter_public_host
            self.adapter_url = f"http://{host}:{self.app_handle.port}"
        logger.info("[leaf] adapter=%s tokenizer=%s", self.adapter_url, args.hf_checkpoint)


@asynccontextmanager
async def boot_sandbox(image: str, instance_id: str) -> AsyncIterator[Sandbox]:
    """A fresh sandbox with Leaf installed; transient boot/install failures are retried."""
    sb, last_err = None, None
    for attempt in range(CONFIG.boot_retries):
        cand = leaf_task.create_sandbox(image)
        try:
            async with _BOOT_SEM:
                await cand.__aenter__()
                try:
                    await LeafHarness().install_cli(cand)
                except BaseException:
                    await cand.__aexit__(None, None, None)
                    raise
            sb = cand
            break
        except Exception as e:
            last_err = e
            logger.warning(
                "[leaf] %s: boot attempt %d/%d failed: %s: %s",
                instance_id,
                attempt + 1,
                CONFIG.boot_retries,
                type(e).__name__,
                str(e)[:200],
            )
            await asyncio.sleep(1 + attempt + random.random())
    if sb is None:
        raise last_err or RuntimeError("no boot attempts (SLIME_AGENT_LEAF_BOOT_RETRIES=0)")
    try:
        yield sb
    finally:
        await sb.__aexit__(None, None, None)


async def generate(args, base_sample: Sample, sampling_params: dict[str, Any], evaluation: bool = False):
    """Roll out one task with Leaf; returns the session's Samples (or one aborted Sample)."""
    state = _AdapterService(args)
    adapter = state.adapter
    task = leaf_task.Task.from_sample(base_sample)
    reason = task.missing()
    if reason:
        return _abort_result(base_sample, reason, task.instance_id)

    session_id = base_sample.session_id = _session_id(base_sample, task.instance_id)
    adapter.open_session(
        session_id,
        sampling_defaults=sampling_params,
        max_context_tokens=state.max_context_len,
        apply_chat_template_kwargs={
            **(getattr(args, "apply_chat_template_kwargs", None) or {}),
            **(base_sample.apply_chat_template_kwargs or {}),
        },
    )
    t0 = time.time()
    try:
        async with asyncio.timeout(CONFIG.rollout_guard_sec):
            async with boot_sandbox(task.image, task.instance_id) as sb:
                baseline = await leaf_task.prepare_workspace(sb, task.workdir)
                agent_exit_code = await LeafHarness().run(
                    sb,
                    workdir=task.workdir,
                    session_id=session_id,
                    adapter_url=state.adapter_url,
                    time_budget_sec=CONFIG.agent_time_budget_sec,
                    prompt=task.prompt(),
                )
                diff_text = await leaf_task.git_diff(sb, task.workdir, baseline)

            reward, applied = await leaf_task.grade(task, diff_text, timeout_sec=CONFIG.eval_timeout_sec)
            logger.info(
                "[leaf] %s: reward=%.1f applied=%s agent_exit_code=%d elapsed=%.1fs%s",
                task.instance_id,
                reward,
                applied,
                agent_exit_code,
                time.time() - t0,
                " (eval)" if evaluation else "",
            )
            if evaluation:
                return _eval_result(base_sample, task, reward=reward, applied=applied, agent_exit_code=agent_exit_code)

            samples = await adapter.finish_session(
                session_id,
                base_sample=base_sample,
                reward=reward,
                extra_metadata={"grading_solved": reward == 1.0, "instance_id": task.instance_id, "agent": AGENT},
            )
            if not samples:
                return _abort_result(base_sample, "adapter_session_empty", task.instance_id)
            for s in samples:
                s.metadata = {**(s.metadata or {}), "agent_exit_code": agent_exit_code, "applied_cleanly": applied}
            return samples

    except asyncio.TimeoutError:
        return _abort_result(base_sample, "wall_clock_timeout", task.instance_id)
    except Exception as e:
        logger.warning("[leaf] %s: rollout failed: %s\n%s", task.instance_id, e, traceback.format_exc())
        return _abort_result(base_sample, f"exception:{type(e).__name__}", task.instance_id)
    finally:
        await adapter.drop_session(session_id, wait_timeout=30)  # idempotent cleanup
        # As coding_agent_rl does: give the fire-and-forget sglang aborts of
        # cancelled requests time to land before this worker moves on.
        await asyncio.sleep(10)


def _session_id(sample: Sample, instance_id: str) -> str:
    # The sid is the adapter's only credential (it is the Bearer token), so it
    # must not be guessable when the adapter is reachable from outside the node.
    if sample.session_id:
        return sample.session_id
    if sample.index is not None and sample.group_index is not None:
        return f"leaf-{instance_id}-{sample.index}-{sample.group_index}-{secrets.token_hex(8)}"
    return f"leaf-{instance_id}-{secrets.token_hex(8)}"


def _placeholder(sample: Sample, *, reward: float, status, metadata: dict) -> list[Sample]:
    """The single untrainable Sample this fan-out generate() returns when there are no real ones."""
    sample.tokens = [0, 0]
    sample.response = ""
    sample.response_length = 1
    sample.loss_mask = [0]
    sample.rollout_log_probs = [0.0]
    sample.reward = reward
    sample.remove_sample = True
    sample.status = status
    sample.metadata = {**(sample.metadata or {}), "agent": AGENT, **metadata}
    return [sample]


def _abort_result(sample: Sample, reason: str, instance_id: str) -> list[Sample]:
    logger.warning("[leaf] %s aborted: %s", instance_id, reason)
    return _placeholder(
        sample,
        reward=0.0,
        status=Sample.Status.ABORTED,
        metadata={"abort_reason": reason, "instance_id": instance_id},
    )


def _eval_result(sample: Sample, task, *, reward: float, applied: bool, agent_exit_code: int) -> list[Sample]:
    """Eval path: only the reward matters."""
    return _placeholder(
        sample,
        reward=reward,
        status=Sample.Status.COMPLETED,
        metadata={
            "instance_id": task.instance_id,
            "grading_solved": reward == 1.0,
            "applied_cleanly": applied,
            "agent_exit_code": agent_exit_code,
        },
    )
