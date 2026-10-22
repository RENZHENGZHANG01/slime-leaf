import argparse, asyncio, json, os, sys, time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(_ROOT), os.environ.get("SLIME_DIR", str(_ROOT / "slime"))]
from leaf_agent import generate as leaf_generate, task as leaf_task
from slime.utils.types import Sample

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/root/leaf_data/train.jsonl")
ap.add_argument("--hf-checkpoint", default="/root/FrogNano-4B-2609")
ap.add_argument("--sglang-port", type=int, default=18100)
ap.add_argument("--samples-per-task", type=int, default=2)
ap.add_argument("--max-new-tokens", type=int, default=4096)
ap.add_argument("--out", default="/root/leaf_runs/rollout_check")
a = ap.parse_args()
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

trajectories = {}
_real_diff = leaf_task.git_diff
async def diff_and_keep_trajectory(sb, workdir, baseline):
    # one asyncio task per rollout, so key the capture by task (rollouts run concurrently)
    trajectories[id(asyncio.current_task())] = await sb.read_file(f"{workdir}/.harness/trajectory.jsonl")
    return await _real_diff(sb, workdir, baseline)
leaf_task.git_diff = diff_and_keep_trajectory

args = SimpleNamespace(hf_checkpoint=a.hf_checkpoint, rollout_max_context_len=32768,
                       sglang_tool_call_parser="qwen3_coder", sglang_reasoning_parser="qwen3",
                       sglang_router_ip="127.0.0.1", sglang_router_port=a.sglang_port, apply_chat_template_kwargs={})
rows = [json.loads(l) for l in open(a.data) if l.strip()]

async def one(row, k):
    base = Sample(index=k, group_index=rows.index(row), prompt=row["prompt"], label=row.get("label"), metadata=dict(row["metadata"]))
    t0 = time.time()
    samples = await leaf_generate.generate(args, base, {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": a.max_new_tokens})
    tok = leaf_generate._AdapterService(args).tokenizer
    traj = [json.loads(l) for l in (trajectories.pop(id(asyncio.current_task()), "") or "").splitlines() if l.strip().startswith("{")]
    rec = {
        "task": row["metadata"]["instance_id"], "k": k, "elapsed_sec": round(time.time() - t0, 1),
        "status": [s.status.name for s in samples], "reward": [s.reward for s in samples],
        "metadata": {kk: v for kk, v in (samples[0].metadata or {}).items() if kk in ("agent_exit_code", "applied_cleanly", "grading_solved", "abort_reason")},
        "n_samples": len(samples),
        "trained_tokens": [int(sum(s.loss_mask)) for s in samples],
        "masked_tokens": [int(len(s.loss_mask) - sum(s.loss_mask)) for s in samples],
        "agent_exit": next((e for e in reversed(traj) if e.get("event") == "exit"), None),
        "tool_calls": [c for e in traj for c in (e.get("tool_calls") or [])],
        "first_trained_text": tok.decode([t for t, m in zip(samples[0].tokens[-samples[0].response_length:], samples[0].loss_mask) if m])[:600] if samples and sum(samples[0].loss_mask) else "",
    }
    (out / f"{rec['task']}_{k}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({x: rec[x] for x in ("task", "k", "status", "reward", "metadata", "n_samples", "trained_tokens", "agent_exit", "elapsed_sec")}), flush=True)
    return rec

async def main():
    recs = await asyncio.gather(*[one(r, k) for r in rows for k in range(a.samples_per_task)])
    solved = sum(1 for r in recs if r["reward"] and r["reward"][0] == 1.0)
    summary = {"rollouts": len(recs), "solved": solved, "samples_total": sum(r["n_samples"] for r in recs),
               "aborted": sum(1 for r in recs if "ABORTED" in r["status"])}
    (out / "summary.json").write_text(json.dumps(summary, indent=2)); print("SUMMARY", json.dumps(summary))
asyncio.run(main())
