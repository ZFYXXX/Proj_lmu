#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/WuHao/checkpoints/ckpts-vlm/Qwen3.8-27B
GPU_ID=${1:?usage: qwen3_5_mme_full_smoke_methods.sh GPU_ID queue_name}
QUEUE_NAME=${2:?usage: qwen3_5_mme_full_smoke_methods.sh GPU_ID queue_name}
ROOT=outputs/qwen3_5-eval/full-smoke-followup-20261014

case "$QUEUE_NAME" in
  gpu0)
    METHODS=(aim bifocal btp dart fastv feather finepruner halfv ivc_prune nuwa)
    ;;
  gpu1)
    METHODS=(otprune pyramid_drop restore saint sgl sparsevlm transprune vispruner zoo_prune)
    ;;
  *)
    echo "Unknown queue: $QUEUE_NAME" >&2
    exit 2
    ;;
esac

cd "$PROJECT_ROOT"
source /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/activate
export QWEN3_5_MODEL_PATH="$MODEL_PATH"
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_CACHE="$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_full"
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"

echo "QUEUE_START $(date -Is) queue=$QUEUE_NAME gpu=$GPU_ID methods=${METHODS[*]}"
for method in "${METHODS[@]}"; do
  config="configs/eval/lmms_eval/qwen3_5_mme_smoke_${method}.yaml"
  out="$ROOT/runs/$method"
  if [[ -e "$out" ]]; then
    echo "SKIP $(date -Is) method=$method reason=output_exists path=$out"
    continue
  fi
  echo "START $(date -Is) method=$method"
  if timeout 43200s python -m mllm_comp.cli.main eval \
    --config "$config" \
    --tasks mme_local_full \
    --runtime_batch_size 1 \
    --sample_limit 2374 \
    --generation_max_new_tokens 16 \
    --use_cache \
    --checkpoint "$MODEL_PATH" \
    --output_dir "$out" \
    --manifest "$out/manifest.json"; then
    echo "FINISH $(date -Is) method=$method status=ok"
  else
    status=$?
    echo "FINISH $(date -Is) method=$method status=$status"
  fi
done
echo "QUEUE_FINISH $(date -Is) queue=$QUEUE_NAME"
