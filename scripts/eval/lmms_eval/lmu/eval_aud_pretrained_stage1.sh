#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/audio_pretrained}"
output_dir="${run_root}/mmau_mmar_mmsu_audiomarathon"

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${AUDIOMARATHON_ROOT:-}" && -n "${DATASET_ROOT:-}" ]]; then
  export AUDIOMARATHON_ROOT="${DATASET_ROOT}/AudioMarathon"
fi

mkdir -p "${output_dir}"
cd "${root_dir}"

MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud.yaml" \
  "$@" \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  --main_process_port "${MAIN_PROCESS_PORT:-29631}"
