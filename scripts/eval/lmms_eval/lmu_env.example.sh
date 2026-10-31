#!/bin/bash
# Copy this file outside Git or export the variables in the submitting shell.
# It contains no credentials. Replace every /path/to/... placeholder.

export EVAL_PYTHON="/path/to/venv/bin/python"
export DATA_ROOT="/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset"
export MODEL_PATH="/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct"
export RUN_DIR="/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/qwen25_lmu_full_v1"

# Identity file saved by the P6000 training run. The Slurm script validates the
# LMU model copy against it before starting any evaluation worker.
export REFERENCE_IDENTITY="/path/to/checkpoints/model_identity.json"

export OLD_MLP="/path/to/checkpoints/old_mlp"
export VISUAL_SELECTOR="/path/to/checkpoints/visual_selector"
export IMAGE_SELECTOR="/path/to/checkpoints/image_context_selector"
export QUESTION_SELECTOR="/path/to/checkpoints/question_context_selector"
export PROTECTED_SELECTOR="/path/to/checkpoints/protected_selector"

# Optional. Keep unset for the full run. Set to 2 only for the first LMU smoke.
# export EVAL_LIMIT=2
# export CHUNK_SIZE=512
# export HALLUSION_IMAGE="/path/to/hallusionbench-image-root"

# Submit from the repository root:
# source scripts/eval/lmms_eval/lmu_env.example.sh
# sbatch scripts/eval/lmms_eval/lmu_qwen25_queue.sbatch
