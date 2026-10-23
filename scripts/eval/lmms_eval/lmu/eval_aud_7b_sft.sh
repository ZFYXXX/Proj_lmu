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

cd "${root_dir}"

run_eval() {
  local config_path="$1"
  local output_dir="$2"
  local main_process_port="$3"
  shift 3

  mkdir -p "${output_dir}"
  MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
    --config "${config_path}" \
    "$@" \
    --output_dir "${output_dir}" \
    --manifest "${output_dir}/manifest.json" \
    --main_process_port "${main_process_port}"
}

run_eval \
  "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud.yaml" \
  "${run_root}/mmau_mmar_mmsu_audiomarathon" \
  "${MAIN_PROCESS_PORT:-29633}" \
  "$@"

bash "${script_dir}/run_mmau_pro_stage.sh" \
  "${root_dir}" \
  "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmau_pro.yaml" \
  "${run_root}/mmau_pro" \
  "${MMAU_PRO_MAIN_PROCESS_PORT:-29733}" \
  "$@"
