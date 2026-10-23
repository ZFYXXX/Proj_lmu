#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
run_stamp="${EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
run_dir="${EVAL_OUTPUT_DIR:-${root_dir}/outputs/qwen3_vl_14bench/runs/${run_stamp}}"

export QWEN3_VL_MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-vlm/Qwen3-VL-8B-Instruct}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

args=(
  --config "${root_dir}/configs/eval/lmms_eval/qwen3_vl_14bench.yaml"
  --output_dir "${run_dir}"
  --manifest "${run_dir}/manifest.json"
)
if [[ -n "${EVAL_LIMIT:-}" ]]; then
  args+=(--sample_limit "${EVAL_LIMIT}")
fi

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval "${args[@]}" "$@"
