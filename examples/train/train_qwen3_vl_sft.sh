#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${root_dir}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export QWEN3_VL_MODEL_PATH="${QWEN3_VL_MODEL_PATH:-/code/wuhao/checkpoints/ckpts-vlm/Qwen3-VL-4B-Instruct}"
export MLLM_COMP_TRAIN_DATASET="${MLLM_COMP_TRAIN_DATASET:-${root_dir}/examples/train/data/text_sft_smoke.jsonl}"
export MLLM_COMP_TRAIN_RUN_DIR="${MLLM_COMP_TRAIN_RUN_DIR:-${root_dir}/runs/qwen3_vl_4b_sft}"

nvidia-smi

exec python -m mllm_comp.cli.main train \
  --config examples/train/qwen3_vl_sft.yaml \
  "$@"
