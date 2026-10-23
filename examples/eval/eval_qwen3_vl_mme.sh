#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root_dir}"

export QWEN3_VL_MODEL_PATH="${QWEN3_VL_MODEL_PATH:-/code/wuhao/checkpoints/ckpts-vlm/Qwen3-VL-4B-Instruct}"
export MLLM_COMP_RUN_DIR="${MLLM_COMP_RUN_DIR:-${root_dir}/runs/qwen3_vl_mme}"

exec python -m mllm_comp.cli.main eval \
  --config examples/eval/qwen3_vl_mme.yaml \
  "$@"
