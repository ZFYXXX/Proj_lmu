#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PYTHONUNBUFFERED=1

main_process_port="${MAIN_PROCESS_PORT:-29611}"

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/qwen25_omni_aud_segmentwise_topk.yaml" \
  --num_processes 2 \
  --distributed_launcher accelerate \
  --main_process_port "${main_process_port}" \
  "$@"
