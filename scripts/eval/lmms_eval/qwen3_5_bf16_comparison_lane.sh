#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/WuHao/checkpoints/ckpts-vlm/Qwen3.8-27B
RUN_ROOT=outputs/qwen3_5-eval/final-bf16-20261011-v2
GPU_ID=${1:?usage: qwen3_5_bf16_comparison_lane.sh GPU_ID (0 or 1)}

source /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/activate
export QWEN3_5_MODEL_PATH="$MODEL_PATH"
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTHONPATH="/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
cd "$PROJECT_ROOT"

run_eval() {
  local tag=$1 config=$2 task=$3 cache_dir=$4
  local out="$RUN_ROOT/$tag"
  local log="$RUN_ROOT/$tag.runner.log"
  if [[ -e "$out" ]]; then
    echo "Refusing to reuse existing run directory: $out" >&2
    return 3
  fi
  export HF_DATASETS_CACHE="$cache_dir"
  mkdir -p "$RUN_ROOT"
  echo "START $(date -Is) gpu=$GPU_ID run=$tag task=$task" > "$log"
  if timeout 43200s python -m mllm_comp.cli.main eval \
      --config "$config" \
      --tasks "$task" \
      --runtime_batch_size 1 \
      --num_processes 1 \
      --generation_max_new_tokens 16 \
      --use_cache \
      --output_dir "$out" \
      --manifest "$out/manifest.json" >> "$log" 2>&1; then
    echo "FINISH $(date -Is) gpu=$GPU_ID run=$tag status=ok" >> "$log"
  else
    local status=$?
    echo "FINISH $(date -Is) gpu=$GPU_ID run=$tag status=$status" >> "$log"
    return "$status"
  fi
}

case "$GPU_ID" in
  0)
    run_eval mme-base configs/eval/lmms_eval/qwen3_5_mme_full_base.yaml mme_local_full \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_full"
    run_eval gqa-base configs/eval/lmms_eval/qwen3_5_eval_base.yaml qwen3_5_gqa_local \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_gqa_local"
    run_eval mmbench-tome configs/eval/lmms_eval/qwen3_5_eval_tome.yaml qwen3_5_mmbench_en_dev_local_static \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_mmbench_local"
    run_eval mme-divprune configs/eval/lmms_eval/qwen3_5_mme_full_divprune.yaml mme_local_full \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_full"
    ;;
  1)
    run_eval mme-tome configs/eval/lmms_eval/qwen3_5_mme_full_tome.yaml mme_local_full \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_full"
    run_eval gqa-tome configs/eval/lmms_eval/qwen3_5_eval_tome.yaml qwen3_5_gqa_local \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_gqa_local"
    run_eval gqa-divprune configs/eval/lmms_eval/qwen3_5_eval_divprune.yaml qwen3_5_gqa_local \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_gqa_local"
    run_eval mmbench-base configs/eval/lmms_eval/qwen3_5_eval_base.yaml qwen3_5_mmbench_en_dev_local_static \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_mmbench_local"
    run_eval mmbench-divprune configs/eval/lmms_eval/qwen3_5_eval_divprune.yaml qwen3_5_mmbench_en_dev_local_static \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_mmbench_local"
    ;;
  *)
    echo "GPU_ID must be 0 or 1" >&2
    exit 2
    ;;
esac
