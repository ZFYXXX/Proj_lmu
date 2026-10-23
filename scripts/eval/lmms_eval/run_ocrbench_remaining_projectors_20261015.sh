#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/WuHao/checkpoints/ckpts-vlm/Qwen3.8-27B
ROOT="$PROJECT_ROOT/outputs/qwen3_5-eval/ocrbench_projectors_remaining_20261015"
GPU_ID=${1:?usage: run_ocrbench_remaining_projectors.sh GPU_ID queue_name}
QUEUE_NAME=${2:?usage: run_ocrbench_remaining_projectors.sh GPU_ID queue_name}

case "$QUEUE_NAME" in
  gpu0) METHODS=(aim bifocal gap mmtok nuwa) ;;
  gpu1) METHODS=(restore saint vispruner zoo_prune) ;;
  *) echo "Unknown queue: $QUEUE_NAME" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/activate
export QWEN3_5_MODEL_PATH="$MODEL_PATH"
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_CACHE="$ROOT/parquet-cache"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

run_method() {
  local method="$1"
  local config="configs/eval/lmms_eval/scienceqa_projectors_20261015/${method}.yaml"
  local out="$ROOT/runs/${method}"
  local log="$ROOT/logs/${method}.log"
  mkdir -p "$(dirname "$log")" "$(dirname "$out")"
  if [[ -e "$out" ]]; then
    echo "SKIP $(date -Is) gpu=$GPU_ID method=$method reason=output_exists path=$out" | tee -a "$log"
    return
  fi
  echo "START $(date -Is) gpu=$GPU_ID method=$method task=qwen3_5_ocrbench_full cache=false" | tee "$log"
  if timeout 43200s python -m mllm_comp.cli.main eval \
    --config "$config" \
    --tasks qwen3_5_ocrbench_full \
    --runtime_batch_size 1 \
    --generation_max_new_tokens 128 \
    --no-use_cache \
    --checkpoint "$MODEL_PATH" \
    --output_dir "$out" \
    --manifest "$out/manifest.json" >> "$log" 2>&1; then
    echo "FINISH $(date -Is) gpu=$GPU_ID method=$method status=ok" | tee -a "$log"
  else
    local status=$?
    echo "FINISH $(date -Is) gpu=$GPU_ID method=$method status=$status" | tee -a "$log"
  fi
}

echo "QUEUE_START $(date -Is) queue=$QUEUE_NAME gpu=$GPU_ID methods=${METHODS[*]}"
for method in "${METHODS[@]}"; do
  run_method "$method"
done
echo "QUEUE_FINISH $(date -Is) queue=$QUEUE_NAME"
