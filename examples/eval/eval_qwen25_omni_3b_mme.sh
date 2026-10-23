#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root_dir}"

export QWEN25_OMNI_MODEL_PATH="${QWEN25_OMNI_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B}"
export MLLM_COMP_RUN_DIR="${MLLM_COMP_RUN_DIR:-${root_dir}/runs/qwen25_omni_3b_mme}"

exec python -m mllm_comp.cli.main eval \
  --config examples/eval/qwen25_omni_3b_mme.yaml \
  --sample_limit 100 \
  "$@"
