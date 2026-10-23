#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/qwen25_omni_aud_atome.yaml" \
  --num_processes 1 \
  "$@"
