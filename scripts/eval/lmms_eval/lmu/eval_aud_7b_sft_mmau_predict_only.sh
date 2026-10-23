#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/audio_7b_sft}"
checkpoint_path="${CHECKPOINT_PATH:-${MODEL_PATH:-}}"

: "${checkpoint_path:?set CHECKPOINT_PATH=/absolute/path/to/audio_7b_sft/checkpoint-N}"

export QWEN25_OMNI_MODEL_PATH="${checkpoint_path}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

output_dir="${run_root}/mmau_predict_only"
mkdir -p "${output_dir}"

cd "${root_dir}"

MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmau_predict_only.yaml" \
  "$@" \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  --main_process_port "${MAIN_PROCESS_PORT:-29633}"
