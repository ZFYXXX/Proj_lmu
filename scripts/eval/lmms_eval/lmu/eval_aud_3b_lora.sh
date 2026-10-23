#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/audio_3b_lora}"
adapter_path="${ADAPTER_PATH:-${CHECKPOINT_PATH:-}}"

: "${adapter_path:?set ADAPTER_PATH=/absolute/path/to/audio_3b_lora/checkpoint-N}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
merged_model_path="${MERGED_MODEL_PATH:-${adapter_path}-merged}"
plan_only=false
for arg in "$@"; do
  if [[ "${arg}" == "--plan_only" ]]; then
    plan_only=true
    break
  fi
done
if [[ "${plan_only}" == false && ! -f "${merged_model_path}/config.json" ]]; then
  "${SWIFT_BIN:-swift}" export \
    --adapters "${adapter_path}" \
    --merge_lora true \
    --output_dir "${merged_model_path}"
fi

export QWEN25_OMNI_MODEL_PATH="${merged_model_path}"
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
  "${MAIN_PROCESS_PORT:-29634}" \
  "$@"

bash "${script_dir}/run_mmau_pro_stage.sh" \
  "${root_dir}" \
  "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmau_pro.yaml" \
  "${run_root}/mmau_pro" \
  "${MMAU_PRO_MAIN_PROCESS_PORT:-29734}" \
  "$@"
