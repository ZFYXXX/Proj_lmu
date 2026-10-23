#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"

: "${DATASET_PATH:?set DATASET_PATH=/absolute/path/to/audio_train.jsonl}"
export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-3B}"
export MLLM_COMP_TRAIN_DATASET="${DATASET_PATH}"
export MLLM_COMP_TRAIN_RUN_DIR="${OUTPUT_DIR:-${root_dir}/runs/lmu/no_compression/${run_tag}/audio_3b_sft}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export NNODES="${NNODES:-1}"
export NODE_RANK="${NODE_RANK:-0}"
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5_omni_lmu}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main train \
  --config "${root_dir}/configs/train/qwen2_5_omni/lmu/train_aud_3b_sft.yaml" \
  "$@"
