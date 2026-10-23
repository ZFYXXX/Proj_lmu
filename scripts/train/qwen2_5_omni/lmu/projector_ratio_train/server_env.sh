#!/usr/bin/env bash
# Source this file before copying an individual command from commands/.

export PROJECT_ROOT="${PROJECT_ROOT:-/home/hk-project-p0022189/tum_yvc3016/haowu/Bowen/MLLM-Comp-Platform}"
projector_model_root=/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/workspace/ckpts/ckpts-omni
export MODEL_PATH_3B="${MODEL_PATH_3B:-${projector_model_root}/Qwen2.5-Omni-3B}"
export MODEL_PATH_7B="${MODEL_PATH_7B:-${projector_model_root}/Qwen2.5-Omni-7B}"
unset projector_model_root

export DATASET_PATH="${DATASET_PATH:-/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ/AudioMCQ_570k_swift_new.jsonl}"
export ROOT_IMAGE_DIR="${ROOT_IMAGE_DIR:-/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/AudioMCQ}"
export PYTHON_BIN="${PYTHON_BIN:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
export MASTER_PORT_BASE="${MASTER_PORT_BASE:-${MASTER_PORT:-29540}}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
export RUN_TAG="${RUN_TAG:-lmu-projector-lora-$(date +%Y%m%d-%H%M%S)}"
