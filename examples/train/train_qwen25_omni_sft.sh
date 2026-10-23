#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN25_OMNI_MODEL_PATH="${QWEN25_OMNI_MODEL_PATH:-/code/wuhao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B}"
export MLLM_COMP_TRAIN_DATASET="${MLLM_COMP_TRAIN_DATASET:-${root_dir}/examples/train/data/text_sft_smoke.jsonl}"
export MLLM_COMP_TRAIN_RUN_DIR="${MLLM_COMP_TRAIN_RUN_DIR:-${root_dir}/runs/qwen25_omni_3b_sft}"

nvidia-smi

exec python -m mllm_comp.cli.main train \
  --config examples/train/qwen25_omni_sft.yaml \
  "$@"
