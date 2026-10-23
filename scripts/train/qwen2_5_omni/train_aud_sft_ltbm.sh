#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
config="${root_dir}/configs/train/qwen2_5_omni/train_aud_sft_ltbm.yaml"

if [[ ! -f "${config}" ]]; then
  printf 'Missing LTBM train config: %s\n' "${config}" >&2
  exit 2
fi

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-omni/Qwen2.5-Omni-3B}}"
export MLLM_COMP_TRAIN_DATASET="${DATASET_PATH:-/home/WuHao/datasets/AudioMCQ/AudioMCQ_28k_swift.jsonl}"
export MLLM_COMP_TRAIN_RUN_DIR="${OUTPUT_DIR:-${root_dir}/runs/train/qwen2_5_omni/audio_sft_compressed}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29512}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5_omni}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main train \
  --config "${config}" \
  "$@"
