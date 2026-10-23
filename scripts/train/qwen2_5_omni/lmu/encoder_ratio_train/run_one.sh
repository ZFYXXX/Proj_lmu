#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  printf 'usage: run_one.sh <train|smoke> <encoder_topk|atome|progressive_fixed> <3b|7b> <10|25|50|75> [mllm-comp train options]\n' >&2
  exit 2
fi

mode="$1"
method="$2"
model_size="$3"
keep="$4"
shift 4

case "${mode}" in
  train|smoke) ;;
  *) printf 'mode must be train or smoke: %s\n' "${mode}" >&2; exit 2 ;;
esac
case "${method}" in
  encoder_topk) method_offset=0 ;;
  atome) method_offset=8 ;;
  progressive_fixed) method_offset=16 ;;
  *) printf 'unknown method: %s\n' "${method}" >&2; exit 2 ;;
esac
case "${model_size}" in
  3b) model_offset=0; selected_model="${MODEL_PATH_3B:-Qwen/Qwen2.5-Omni-3B}" ;;
  7b) model_offset=1; selected_model="${MODEL_PATH_7B:-${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}}" ;;
  *) printf 'model size must be 3b or 7b: %s\n' "${model_size}" >&2; exit 2 ;;
esac
case "${keep}" in
  10) ratio_offset=0 ;;
  25) ratio_offset=2 ;;
  50) ratio_offset=4 ;;
  75) ratio_offset=6 ;;
  *) printf 'target keep ratio must be 10, 25, 50, or 75: %s\n' "${keep}" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
default_root="$(cd -- "${script_dir}/../../../../.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${default_root}}"
: "${DATASET_PATH:?set DATASET_PATH=/absolute/path/to/audio_train.jsonl}"
export MODEL_PATH="${selected_model}"
export RUN_TAG="${RUN_TAG:-lmu-encoder-lora-$(date +%Y%m%d-%H%M%S)}"

log_root="${LOG_ROOT:-${PROJECT_ROOT}/logs/lmu/encoder_ratio_lora}"
result_root="${RESULT_ROOT:-${PROJECT_ROOT}/runs/lmu/encoder_ratio_lora}"
export OUTPUT_DIR="${result_root}/${RUN_TAG}/${mode}"
base_port="${MASTER_PORT_BASE:-${MASTER_PORT:-29504}}"
export MASTER_PORT="$((base_port + method_offset + ratio_offset + model_offset))"

log_dir="${log_root}/${RUN_TAG}/${mode}/${method}/keep${keep}"
log_path="${log_dir}/${model_size}_lora.log"
mkdir -p "${log_dir}" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

cmd=(bash "${script_dir}/run_${method}.sh" "${model_size}" "${keep}")
if [[ "${mode}" == smoke ]]; then
  cmd+=(--max_steps 1)
fi
cmd+=("$@")
{
  printf 'mode=%s method=%s model=%s keep=%s%% model_path=%s\n' \
    "${mode}" "${method}" "${model_size}" "${keep}" "${MODEL_PATH}"
  printf 'log=%s\noutput=%s/%s/keep%s/%s_lora\nport=%s\n' \
    "${log_path}" "${OUTPUT_DIR}" "${method}" "${keep}" "${model_size}" "${MASTER_PORT}"
  "${cmd[@]}"
} 2>&1 | tee "${log_path}"
