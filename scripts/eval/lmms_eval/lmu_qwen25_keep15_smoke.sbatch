#!/usr/bin/env bash
#SBATCH --job-name=28-q25-15-smoke
#SBATCH --account=hk-project-p0024767
#SBATCH --partition=dev_accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --time=00:50:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x_%j_output.log
#SBATCH --error=logs/slurm/%x_%j_error.log
#SBATCH --mail-type="NONE"

set -euo pipefail
: "${SLURM_JOB_ID:?Submit this smoke through Slurm; do not run on a login node}"
KEEP15_REPO="${KEEP15_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
KEEP15_PYTHON="${KEEP15_PYTHON:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp/bin/python}"
KEEP15_PROFILE="${KEEP15_PROFILE:-configs/workflow/profiles/lmu_keep15.json}"
KEEP15_OUTPUT="${KEEP15_OUTPUT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_lmu_smoke_${SLURM_JOB_ID}}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2

cd "$KEEP15_REPO"
echo "Slurm job: $SLURM_JOB_ID"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Smoke output: $KEEP15_OUTPUT"
"$KEEP15_PYTHON" scripts/workflow.py smoke \
  --profile "$KEEP15_PROFILE" \
  --method-catalog configs/workflow/methods_keep15.json \
  --experiment configs/workflow/experiments/qwen25_keep15.json \
  --limit 2 --run-dir "$KEEP15_OUTPUT"
