#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
eval_name="${EVAL_NAME:-audio_eval}"
run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/${eval_name}}"

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

plan_only=false
cli_num_processes=""
expect_num_processes=false
for arg in "$@"; do
  if [[ "${expect_num_processes}" == true ]]; then
    cli_num_processes="${arg}"
    expect_num_processes=false
    continue
  fi
  if [[ "${arg}" == "--plan_only" ]]; then
    plan_only=true
  elif [[ "${arg}" == "--num_processes" ]]; then
    expect_num_processes=true
  elif [[ "${arg}" == --num_processes=* ]]; then
    cli_num_processes="${arg#--num_processes=}"
  fi
done
if [[ "${expect_num_processes}" == true ]]; then
  printf '%s requires a value\n' '--num_processes' >&2
  exit 2
fi

if [[ -n "${adapter_path}" ]]; then
  if [[ "${plan_only}" == false && ! -d "${adapter_path}" ]]; then
    printf 'LoRA adapter does not exist: %s\n' "${adapter_path}" >&2
    exit 2
  fi
  merged_model_path="${MERGED_MODEL_PATH:-${adapter_path}-merged}"
  if [[ "${plan_only}" == false && ! -f "${merged_model_path}/config.json" ]]; then
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
  printf 'CUDA_VISIBLE_DEVICES must contain at least one GPU\n' >&2
  exit 2
fi
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
for gpu_id in "${gpu_ids[@]}"; do
  if [[ -z "${gpu_id}" ]]; then
    printf 'invalid CUDA_VISIBLE_DEVICES value: %s\n' "${gpu_list}" >&2
    exit 2
  fi
done
num_processes="${EVAL_NUM_PROCESSES:-${cli_num_processes:-${#gpu_ids[@]}}}"
if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'EVAL_NUM_PROCESSES must be a positive integer, got: %s\n' "${num_processes}" >&2
  exit 2
fi
if (( num_processes > ${#gpu_ids[@]} )); then
  printf 'EVAL_NUM_PROCESSES=%s exceeds the %s visible GPUs in %s\n' \
    "${num_processes}" "${#gpu_ids[@]}" "${gpu_list}" >&2
  exit 2
fi

export QWEN25_OMNI_MODEL_PATH="${resolved_model_path}"
export CUDA_VISIBLE_DEVICES="${gpu_list}"
export MMAU_PRO_CUDA_VISIBLE_DEVICES="${MMAU_PRO_CUDA_VISIBLE_DEVICES:-${gpu_list}}"
export MMAU_PRO_NUM_GPUS="${MMAU_PRO_NUM_GPUS:-${num_processes}}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

if [[ -z "${AUDIOMARATHON_ROOT:-}" && -n "${DATASET_ROOT:-}" ]]; then
  export AUDIOMARATHON_ROOT="${DATASET_ROOT}/AudioMarathon"
fi

cd "${root_dir}"

run_eval() {
  local task_name="$1"
  local config_name="$2"
  local main_process_port="$3"
  local output_dir="${run_root}/${task_name}"
  shift 3

  printf 'starting stage 1 task=%s model=%s gpus=%s processes=%s output=%s\n' \
    "${task_name}" "${QWEN25_OMNI_MODEL_PATH}" "${CUDA_VISIBLE_DEVICES}" \
    "${num_processes}" "${output_dir}"
  mkdir -p "${output_dir}"
  MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
    --config "${root_dir}/configs/eval/lmms_eval/lmu/${config_name}" \
    --num_processes "${num_processes}" \
    "$@" \
    --output_dir "${output_dir}" \
    --manifest "${output_dir}/manifest.json" \
    --main_process_port "${main_process_port}"
}

run_eval \
  audiomarathon \
  eval_aud_audiomarathon.yaml \
  "${AUDIOMARATHON_MAIN_PROCESS_PORT:-29634}" \
  "$@"

run_eval \
  mmaunew2 \
  eval_aud_mmaunew2.yaml \
  "${MMAUNEW2_MAIN_PROCESS_PORT:-29635}" \
  "$@"

run_eval \
  mmsunew2 \
  eval_aud_mmsunew2.yaml \
  "${MMSUNEW2_MAIN_PROCESS_PORT:-29636}" \
  "$@"

run_eval \
  mmarnew \
  eval_aud_mmar.yaml \
  "${MMAR_MAIN_PROCESS_PORT:-29637}" \
  "$@"

printf 'starting stage 2 task=mmau_pro output=%s\n' "${run_root}/mmau_pro"
bash "${script_dir}/run_mmau_pro_stage.sh" \
  "${root_dir}" \
  "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmau_pro.yaml" \
  "${run_root}/mmau_pro" \
  "${MMAU_PRO_MAIN_PROCESS_PORT:-29734}" \
  --num_processes "${num_processes}" \
  "$@"
