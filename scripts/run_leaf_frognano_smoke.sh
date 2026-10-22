#!/bin/bash
set -ex
LEAF_DIR=${LEAF_DIR:-$(cd "$(dirname "$0")/.." && pwd)}
SLIME_DIR=${SLIME_DIR:-$LEAF_DIR/slime}
DATA=${DATA:-/root/leaf_data}
HF_CHECKPOINT=${HF_CHECKPOINT:-/root/FrogNano-4B-2609}
RUN_ROOT=${RUN_ROOT:-/root/leaf_runs/$(date +%Y%m%d-%H%M%S)}
NUM_ROLLOUT=${NUM_ROLLOUT:-2}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-2}
SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT:-4}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((ROLLOUT_BATCH_SIZE * SAMPLES_PER_PROMPT))}
MAX_CONTEXT_LEN=${MAX_CONTEXT_LEN:-32768}
MAX_RESPONSE_LEN=${MAX_RESPONSE_LEN:-4096}
mkdir -p "$RUN_ROOT"

pkill -9 sglang || true; ray stop --force || true; sleep 2

export PYTHONUNBUFFERED=1 CUDA_DEVICE_MAX_CONNECTIONS=1 NCCL_NVLS_ENABLE=0
# A100 (sm80) + head_dim 256 training: FA4 does not compile, FA3 is Hopper-only, FA2 is not in
# the image and cuDNN fused has no kernel for it, so only unfused attention works here.
export NVTE_FLASH_ATTN=0 NVTE_FUSED_ATTN=0
export PYTHONPATH="$LEAF_DIR:$SLIME_DIR:/root/Megatron-LM"
: "${ADAPTER_PUBLIC_URL:?set ADAPTER_PUBLIC_URL=http://<public-ip>:<mapped-port>}"
test -f ~/.modal.toml || { echo "Modal auth missing (~/.modal.toml)"; exit 1; }
export ADAPTER_BIND_HOST=0.0.0.0 ADAPTER_PORT=18091
export SLIME_AGENT_LEAF_ADAPTER_PUBLIC_URL="$ADAPTER_PUBLIC_URL"
export SLIME_AGENT_LEAF_SANDBOX_FACTORY=leaf_agent.modal_sandbox:ModalSandbox
export SLIME_AGENT_LEAF_MODAL_IMAGES="$DATA/images.json"
export SLIME_AGENT_LEAF_MODAL_APP=${SLIME_AGENT_LEAF_MODAL_APP:-slime-leaf-sandboxes}
export SLIME_AGENT_LEAF_MODAL_ALLOW_OUT=${SLIME_AGENT_LEAF_MODAL_ALLOW_OUT:-}
export SLIME_AGENT_LEAF_MODAL_LIFETIME_SEC=1800
export SLIME_AGENT_LEAF_TIME_BUDGET_SEC=${SLIME_AGENT_LEAF_TIME_BUDGET_SEC:-600}
export SLIME_AGENT_LEAF_EVAL_TIMEOUT_SEC=120
export SLIME_AGENT_LEAF_MAX_STEPS=${SLIME_AGENT_LEAF_MAX_STEPS:-30}
export SLIME_AGENT_LEAF_MAX_COMPLETION_TOKENS=$MAX_RESPONSE_LEN
export SLIME_FORK_MERGE_MAX_RESPONSE_TOKENS=0

ray start --head --node-ip-address 127.0.0.1 --num-gpus 1 --disable-usage-stats
RUNTIME_ENV_JSON=$(python3 - <<'PY'
import json, os
keep = [k for k in os.environ if k.startswith(("SLIME_AGENT_LEAF_", "ADAPTER_", "SLIME_FORK_", "MODAL_", "NVTE_")) or k in (
    "PYTHONPATH", "CUDA_DEVICE_MAX_CONNECTIONS", "NCCL_NVLS_ENABLE", "PYTHONUNBUFFERED")]
print(json.dumps({"env_vars": {k: os.environ[k] for k in keep}}))
PY
)
source "$SLIME_DIR/scripts/models/qwen3.5-4B.sh"
cd "$SLIME_DIR"
ray job submit --address=http://127.0.0.1:8265 --runtime-env-json="$RUNTIME_ENV_JSON" -- python3 -u "$SLIME_DIR/train.py" \
    --actor-num-nodes 1 --actor-num-gpus-per-node 1 --colocate \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" --load "$HF_CHECKPOINT" \
    --custom-generate-function-path leaf_agent.generate.generate \
    --prompt-data "$DATA/train.jsonl" --input-key prompt --label-key label --metadata-key metadata \
    --num-rollout "$NUM_ROLLOUT" --rollout-batch-size "$ROLLOUT_BATCH_SIZE" --n-samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --rollout-max-context-len "$MAX_CONTEXT_LEN" --rollout-max-response-len "$MAX_RESPONSE_LEN" \
    --rollout-temperature 1.0 --rollout-top-p 0.95 --rollout-stop-token-ids 248046 248044 \
    --num-steps-per-rollout 1 --global-batch-size "$GLOBAL_BATCH_SIZE" --micro-batch-size 1 \
    --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 \
    --recompute-granularity full --recompute-method uniform --recompute-num-layers 1 \
    --use-dynamic-batch-size --max-tokens-per-gpu "$MAX_CONTEXT_LEN" --log-probs-chunk-size 1024 \
    --advantage-estimator grpo --kl-loss-coef 0 --kl-coef 0 --entropy-coef 0 --eps-clip 0.2 \
    --optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 --adam-beta1 0.9 --adam-beta2 0.98 \
    --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer \
    --rollout-num-gpus-per-engine 1 --sglang-mem-fraction-static 0.45 --sglang-context-length "$MAX_CONTEXT_LEN" \
    --sglang-tool-call-parser qwen3_coder --sglang-reasoning-parser qwen3 \
    --attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 --attention-backend unfused \
    --save-debug-rollout-data "$RUN_ROOT/rollout_{rollout_id}.pt" \
    2>&1 | tee "$RUN_ROOT/train.log"
