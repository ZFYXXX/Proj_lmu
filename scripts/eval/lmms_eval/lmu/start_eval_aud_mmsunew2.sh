#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
eval_name="${EVAL_NAME:-audio_eval}"
output_dir="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/${eval_name}/mmsunew2}"

adapter_path="${ADAPTER_PATH:-}"
direct_model_path="${CHECKPOINT_PATH:-${MODEL_PATH:-}}"
if [[ -n "${adapter_path}" && -n "${direct_model_path}" ]]; then
  printf 'set ADAPTER_PATH or CHECKPOINT_PATH/MODEL_PATH, not both\n' >&2
  exit 2
fi
if [[ -z "${adapter_path}" && -z "${direct_model_path}" ]]; then
  printf 'set ADAPTER_PATH, CHECKPOINT_PATH, or MODEL_PATH\n' >&2
  exit 2
fi

if [[ -n "${adapter_path}" ]]; then
  if [[ ! -d "${adapter_path}" ]]; then
    printf 'LoRA adapter does not exist: %s\n' "${adapter_path}" >&2
    exit 2
  fi
  merged_model_path="${MERGED_MODEL_PATH:-${adapter_path}-merged}"
  if [[ ! -f "${merged_model_path}/config.json" ]]; then
    "${SWIFT_BIN:-swift}" export \
      --adapters "${adapter_path}" \
      --merge_lora true \
      --output_dir "${merged_model_path}"
  fi
  resolved_model_path="${merged_model_path}"
else
  resolved_model_path="${direct_model_path}"
fi

gpu_list="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
gpu_list="${gpu_list//[[:space:]]/}"
if [[ -z "${gpu_list}" ]]; then
  printf 'CUDA_VISIBLE_DEVICES must contain four GPUs\n' >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
num_processes="${EVAL_NUM_PROCESSES:-4}"
if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EVAL_NUM_PROCESSES must be a positive integer, got: %s\n' \
    "${num_processes}" >&2
  exit 2
fi
if [[ "${#gpu_ids[@]}" -lt "${num_processes}" ]]; then
  printf 'EVAL_NUM_PROCESSES=%s exceeds the %s visible GPUs in %s\n' \
    "${num_processes}" "${#gpu_ids[@]}" "${gpu_list}" >&2
  exit 2
fi

export QWEN25_OMNI_MODEL_PATH="${resolved_model_path}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

cd "${root_dir}"
mkdir -p "${output_dir}"

printf 'starting task=mmsunew2 model=%s gpus=%s processes=%s output=%s\n' \
  "${QWEN25_OMNI_MODEL_PATH}" "${CUDA_VISIBLE_DEVICES}" \
  "${num_processes}" "${output_dir}"

MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmsunew2.yaml" \
  --num_processes "${num_processes}" \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  --main_process_port "${MMSUNEW2_MAIN_PROCESS_PORT:-29636}" \
  "$@"
