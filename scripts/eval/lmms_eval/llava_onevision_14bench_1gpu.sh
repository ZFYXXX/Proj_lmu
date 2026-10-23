#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
run_stamp="${EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
run_dir="${EVAL_OUTPUT_DIR:-${root_dir}/outputs/llava_onevision_14bench/runs/${run_stamp}}"

export LLAVA_ONEVISION_MODEL_PATH="${MODEL_PATH:-${LLAVA_ONEVISION_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-vlm/llava-onevision-qwen2-7b-ov}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

nofile_limit="${EVAL_NOFILE_LIMIT:-65535}"
if (( $(ulimit -Sn) < nofile_limit )); then
  ulimit -Sn "${nofile_limit}"
fi

args=(
  --config "${root_dir}/configs/eval/lmms_eval/llava_onevision_14bench.yaml"
  --num_processes 1
  --output_dir "${run_dir}"
  --manifest "${run_dir}/manifest.json"
)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  args+=(--sample_limit "${EVAL_LIMIT}")
fi

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval "${args[@]}" "$@"
