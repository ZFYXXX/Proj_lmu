#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/WuHao/checkpoints/ckpts-vlm/Qwen3.8-27B
ROOT="$PROJECT_ROOT/outputs/qwen3_5-eval/crossbench_full_20261014"
GPU_ID=${1:?usage: run_crossbench_full.sh GPU_ID queue_name}
QUEUE_NAME=${2:?usage: run_crossbench_full.sh GPU_ID queue_name}

case "$QUEUE_NAME" in
  gpu0) METHODS=(base prumerge) ;;
  gpu1) METHODS=(tome divprune visionzip otprune) ;;
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

SHORT_TASKS=qwen3_5_scienceqa_img_full,qwen3_5_mmbench_en_dev_full
LONG_TASKS=qwen3_5_ocrbench_full,qwen3_5_refcoco_bbox_rec_val_full

run_group() {
  local method="$1" group="$2" tasks="$3" max_tokens="$4"
  local config="configs/eval/lmms_eval/crossbench_full_20261014/${method}.yaml"
  local out="$ROOT/runs/${method}/${group}"
  local log="$ROOT/logs/${method}_${group}.log"
  mkdir -p "$(dirname "$out")" "$(dirname "$log")"
  echo "START $(date -Is) gpu=$GPU_ID method=$method group=$group tasks=$tasks max_new_tokens=$max_tokens cache=false" | tee "$log"
  if timeout 43200s python -m mllm_comp.cli.main eval \
    --config "$config" \
    --tasks "$tasks" \
    --runtime_batch_size 1 \
    --generation_max_new_tokens "$max_tokens" \
    --no-use_cache \
    --checkpoint "$MODEL_PATH" \
    --output_dir "$out" \
    --manifest "$out/manifest.json" >> "$log" 2>&1; then
    echo "FINISH $(date -Is) gpu=$GPU_ID method=$method group=$group status=ok" | tee -a "$log"
  else
    local status=$?
    echo "FINISH $(date -Is) gpu=$GPU_ID method=$method group=$group status=$status" | tee -a "$log"
  fi
}

echo "QUEUE_START $(date -Is) queue=$QUEUE_NAME gpu=$GPU_ID methods=${METHODS[*]}"
for method in "${METHODS[@]}"; do
  run_group "$method" reasoning_general "$SHORT_TASKS" 16
  run_group "$method" ocr_grounding "$LONG_TASKS" 128
done
echo "QUEUE_FINISH $(date -Is) queue=$QUEUE_NAME"
