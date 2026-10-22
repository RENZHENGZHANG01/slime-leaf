# slime-leaf

FrogNano's Leaf agent harness ([microsoft/FrogNano](https://github.com/microsoft/FrogNano)) ported to
[slime](https://github.com/THUDM/slime) for agentic RL.

- `leaf_agent/leaf_cli.py`, `tool_runner.py`, `tools.py` run inside the sandbox (stdlib only, Python >= 3.6).
- `leaf_agent/harness.py` installs and launches Leaf (`LeafHarness`).
- `leaf_agent/adapter.py` is `LeafChatAdapter`: it keeps every tool call and the reasoning in the
  recorded turn, so a Leaf session stays one CLEAN trajectory.
- `leaf_agent/task.py` and `generate.py` form the rollout: sandbox, harness, diff, grading in a
  fresh sandbox, then Samples.
- `leaf_agent/modal_sandbox.py` runs tools in Modal Sandboxes, with an optional egress allowlist.

## Setup

slime is a git submodule at `slime`, pinned to
[`05e3b017`](https://github.com/THUDM/slime/commit/05e3b017fed9e02de418c2fbac9d75602a534be5).

```bash
git clone --recursive https://github.com/RENZHENGZHANG01/slime-leaf.git
# or, in an existing clone:
git submodule update --init
```

To move the pin: `git -C slime fetch && git -C slime checkout <commit>`, run the
tests, then commit `slime`.

## Use with slime

```bash
export PYTHONPATH=$PWD:$PWD/slime:/root/Megatron-LM
python slime/train.py ... --custom-generate-function-path leaf_agent.generate.generate
```

Megatron-LM must match the pinned slime. The `slimerl/slime` Docker image ships its own slime and
Megatron-LM; the GPU smoke test ran on that image's slime, which was newer than this pin, with the same Leaf code.

See `scripts/run_leaf_frognano_smoke.sh` for a 1-GPU GRPO run with `microsoft/FrogNano-4B-2609`.

## Tests

```bash
python -m pytest tests/test_leaf_harness.py   # uses slime; override with SLIME_DIR
```
