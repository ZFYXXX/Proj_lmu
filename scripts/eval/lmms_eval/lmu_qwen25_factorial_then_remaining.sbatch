#!/usr/bin/env bash
#SBATCH --job-name=28-factorial-rest
#SBATCH --account=hk-project-p0024767
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=23:50:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x_%j_output.log
#SBATCH --error=logs/slurm/%x_%j_error.log
#SBATCH --mail-type="NONE"

set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}"
CHAIN_REPO="${CHAIN_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
cd "$CHAIN_REPO"
export FALLBACK_REPO="$CHAIN_REPO" KEEP15_REPO="$CHAIN_REPO"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
unset MLLM_TOPK_SHARED_GPU MLLM_TOPK_NORMAL_GPU MLLM_TOPK_MEMORY_FRACTION MLLM_TOPK_DUTY_CYCLE
for CHAIN_SCRIPT in lmu_qwen25_local_factorial.sbatch lmu_qwen25_keep15_remaining.sbatch; do
  test -f "scripts/eval/lmms_eval/$CHAIN_SCRIPT"
  bash -n "scripts/eval/lmms_eval/$CHAIN_SCRIPT"
done
CHAIN_CHILD=""
trap 'if [[ -n "$CHAIN_CHILD" ]]; then kill -TERM "$CHAIN_CHILD" 2>/dev/null || true; wait "$CHAIN_CHILD" 2>/dev/null || true; fi; exit 143' TERM INT
# Child scripts run with bash, never sbatch: one allocation, sequential phases.
# Existing output paths and resume checks are retained by each child.
for CHAIN_SCRIPT in lmu_qwen25_local_factorial.sbatch lmu_qwen25_keep15_remaining.sbatch; do
  echo "CHAIN_START $CHAIN_SCRIPT job=$SLURM_JOB_ID gpus=$CUDA_VISIBLE_DEVICES"
  bash "scripts/eval/lmms_eval/$CHAIN_SCRIPT" &
  CHAIN_CHILD=$!
  if wait "$CHAIN_CHILD"; then
    CHAIN_CHILD=""
    echo "CHAIN_PASS $CHAIN_SCRIPT"
  else
    CHAIN_EXIT=$?
    CHAIN_CHILD=""
    echo "CHAIN_FAIL $CHAIN_SCRIPT exit=$CHAIN_EXIT" >&2
    exit "$CHAIN_EXIT"
  fi
done
echo "CHAIN_COMPLETE job=$SLURM_JOB_ID"
