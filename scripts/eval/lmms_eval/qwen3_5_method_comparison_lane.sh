#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
MODEL_PATH=/home/ZhouFengyu/fastv-repro/models/qwen38-hf-cache/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
GPU_ID=${1:?usage: qwen3_5_method_comparison_lane.sh GPU_ID (0 or 1)}

source /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/activate
export QWEN3_5_MODEL_PATH="$MODEL_PATH"
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export PYTHONPATH="/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
unset HF_HUB_OFFLINE || true
cd "$PROJECT_ROOT"

run_eval() {
  local tag=$1 config=$2 task=$3 cache_dir=$4 devices=${5:-$GPU_ID}
  local out="outputs/qwen3_5-eval/final-20261011/$tag"
  if [[ -e "$out/manifest.json" ]]; then
    echo "Refusing to overwrite existing run: $out" >&2
    return 3
  fi
  export HF_DATASETS_CACHE="$cache_dir"
  mkdir -p "$out"
  echo "START $(date -Is) gpu=$GPU_ID run=$tag task=$task" > "$out/run.log"
  if timeout 43200s env CUDA_VISIBLE_DEVICES="$devices" python -m mllm_comp.cli.main eval \
      --config "$config" \
      --tasks "$task" \
      --runtime_batch_size 1 \
      --num_processes 1 \
      --generation_max_new_tokens 16 \
      --use_cache \
      --output_dir "$out" \
      --manifest "$out/manifest.json" >> "$out/run.log" 2>&1; then
    echo "FINISH $(date -Is) gpu=$GPU_ID run=$tag status=ok" >> "$out/run.log"
  else
    local status=$?
    echo "FINISH $(date -Is) gpu=$GPU_ID run=$tag status=$status" >> "$out/run.log"
    return "$status"
  fi
}

wait_for_mme_divprune() {
  local log="outputs/qwen3_5-eval/final-20261011/mme-divprune/run.log"
  while [[ ! -f "$log" ]] || ! grep -q 'FINISH .*run=mme-divprune status=' "$log"; do
    sleep 60
  done
  if ! grep -q 'FINISH .*run=mme-divprune status=ok' "$log"; then
    echo "The MME DivPrune run failed; not starting the GQA/MMBench lane." >&2
    return 1
  fi
}

case "$GPU_ID" in
  0)
    run_eval mme-divprune configs/eval/lmms_eval/qwen3_5_mme_full_divprune.yaml mme_local_full \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_full" 0,1
    run_eval gqa-base configs/eval/lmms_eval/qwen3_5_eval_base.yaml qwen3_5_gqa_local \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_gqa_local"
    run_eval mmbench-tome configs/eval/lmms_eval/qwen3_5_eval_tome.yaml qwen3_5_mmbench_en_dev_local_static \
      "$PROJECT_ROOT/outputs/qwen3_5-eval/hf_datasets_mmbench_local"
    ;;
  1)
    wait_for_mme_divprune
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
