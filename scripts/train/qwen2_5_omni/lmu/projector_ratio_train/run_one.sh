#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  printf 'usage: run_one.sh <train|smoke> <segmentwise_topk|ltbm|sa_map|unseg|unisamp|uniavg|fastslm_hta> <3b|7b> <10|25|50|75|natural|default> [mllm-comp train options]\n' >&2
  exit 2
fi

mode="$1"
method="$2"
model_size="$3"
setting="$4"
shift 4
case "${mode}" in
  train|smoke) ;;
  *) printf 'mode must be train or smoke: %s\n' "${mode}" >&2; exit 2 ;;
esac
case "${model_size}" in
  3b) selected_model="${MODEL_PATH_3B:-Qwen/Qwen2.5-Omni-3B}" ;;
  7b) selected_model="${MODEL_PATH_7B:-${MODEL_PATH:-Qwen/Qwen2.5-Omni-7B}}" ;;
  *) printf 'model size must be 3b or 7b: %s\n' "${model_size}" >&2; exit 2 ;;
esac

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/common.bash"
setting_label="$(projector_train_setting_label "${method}" "${setting}")"
default_root="$(cd -- "${script_dir}/../../../../.." && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-${default_root}}"
: "${DATASET_PATH:?set DATASET_PATH=/absolute/path/to/audio_train.jsonl}"
export MODEL_PATH="${selected_model}"
export RUN_TAG="${RUN_TAG:-lmu-projector-lora-$(date +%Y%m%d-%H%M%S)}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-${MASTER_PORT:-29540}}"

log_root="${LOG_ROOT:-${PROJECT_ROOT}/logs/lmu/projector_ratio_lora}"
result_root="${RESULT_ROOT:-${PROJECT_ROOT}/runs/lmu/projector_ratio_lora}"
export OUTPUT_DIR="${result_root}/${RUN_TAG}/${mode}"
log_dir="${log_root}/${RUN_TAG}/${mode}/${method}/${setting_label}"
log_path="${log_dir}/${model_size}_lora.log"
mkdir -p "${log_dir}" "${OUTPUT_DIR}"
cd "${PROJECT_ROOT}"

cmd=(bash "${script_dir}/run_${method}.sh" "${model_size}" "${setting}")
if [[ "${mode}" == smoke ]]; then
  cmd+=(--max_steps 1)
fi
cmd+=("$@")
{
  printf 'mode=%s method=%s model=%s setting=%s model_path=%s\n' \
    "${mode}" "${method}" "${model_size}" "${setting_label}" "${MODEL_PATH}"
  printf 'log=%s\n' "${log_path}"
  "${cmd[@]}"
} 2>&1 | tee "${log_path}"
