#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"

export QWEN3_VL_MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_MODEL_PATH:-/code/wuhao/checkpoints/ckpts-vlm/Qwen3-VL-4B-Instruct}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/qwen3_vl.yaml" \
  "$@"
