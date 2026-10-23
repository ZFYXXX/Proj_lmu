#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
config="${root_dir}/configs/eval/lmms_eval/qwen25_omni_aud_fastslm_hta.yaml"

model_path="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-}}"
if [[ -z "${model_path}" ]]; then
  printf '%s\n' 'Set MODEL_PATH or QWEN25_OMNI_MODEL_PATH to the Qwen checkpoint.' >&2
  exit 2
fi
if [[ -z "${FASTSLM_HTA_CHECKPOINT_PATH:-}" ]]; then
  printf '%s\n' 'Set FASTSLM_HTA_CHECKPOINT_PATH to the trained HTA checkpoint.' >&2
  exit 2
fi

export QWEN25_OMNI_MODEL_PATH="${model_path}"
export FASTSLM_HTA_CHECKPOINT_PATH
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval --config "${config}" "$@"
