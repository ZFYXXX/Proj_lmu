#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/WuHao/checkpoints/ckpts-vlm/Qwen3.8-27B
ROOT="$PROJECT_ROOT/outputs/qwen3_5-eval/scienceqa_projectors_full_20261015_v2"
GPU_ID=${1:?usage: run_scienceqa_projectors_full_20261015_v2.sh GPU_ID queue_name}
QUEUE_NAME=${2:?usage: run_scienceqa_projectors_full_20261015_v2.sh GPU_ID queue_name}

case "$QUEUE_NAME" in
  gpu0) METHODS=(aim bifocal divprune gap mmtok nuwa tome) ;;
  gpu1) METHODS=(otprune prumerge restore saint vispruner visionzip zoo_prune) ;;
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

config_for() {
  case "$1" in
    tome|visionzip) echo "configs/eval/lmms_eval/crossbench_full_20261014/$1.yaml" ;;
    *) echo "configs/eval/lmms_eval/scienceqa_projectors_20261015/$1.yaml" ;;
  esac
}

run_method() {
  local method="$1"
  local config
  config=$(config_for "$method")
  local out="$ROOT/runs/$method"
  local log="$ROOT/logs/$method.log"
  mkdir -p "$(dirname "$log")" "$(dirname "$out")"
  if [[ -e "$out/result.json" ]] && grep -q '"status": "succeeded"' "$out/result.json"; then
    echo "SKIP $(date -Is) gpu=$GPU_ID method=$method reason=succeeded" | tee -a "$log"
    return
  fi
  echo "START $(date -Is) gpu=$GPU_ID method=$method task=qwen3_5_scienceqa_img_full cache=false" | tee "$log"
  if timeout 43200s python -m mllm_comp.cli.main eval \
    --config "$config" \
    --tasks qwen3_5_scienceqa_img_full \
    --runtime_batch_size 1 \
    --generation_max_new_tokens 16 \
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

mkdir -p "$ROOT/logs" "$ROOT/runs"
echo "QUEUE_START $(date -Is) queue=$QUEUE_NAME gpu=$GPU_ID methods=${METHODS[*]}"
for method in "${METHODS[@]}"; do
  run_method "$method"
done
echo "QUEUE_FINISH $(date -Is) queue=$QUEUE_NAME"
