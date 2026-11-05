#!/usr/bin/env bash
#SBATCH --job-name=28-q25-15-full
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
: "${SLURM_JOB_ID:?Submit through Slurm; never run inference on a login node}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose the allocated GPUs}"
KEEP15_REPO="${KEEP15_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
KEEP15_PYTHON="${KEEP15_PYTHON:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp/bin/python}"
KEEP15_PROFILE="${KEEP15_PROFILE:-configs/workflow/profiles/lmu_keep15.json}"
KEEP15_OUTPUT="${KEEP15_OUTPUT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_eight_methods_v1}"
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
cd "$KEEP15_REPO"

# Prepare a full immutable source/config snapshot before the smoke. Resume only
# when source, settings and checkpoint hashes match the existing full run.
KEEP15_RESUME=()
if [[ -f "$KEEP15_OUTPUT/workflow.json" ]]; then KEEP15_RESUME=(--resume); fi
"$KEEP15_PYTHON" scripts/workflow.py prepare \
  --profile "$KEEP15_PROFILE" \
  --method-catalog configs/workflow/methods_keep15.json \
  --experiment configs/workflow/experiments/qwen25_keep15.json \
  --run-dir "$KEEP15_OUTPUT" "${KEEP15_RESUME[@]}"

echo "SMOKE_START job=$SLURM_JOB_ID gpu=${CUDA_VISIBLE_DEVICES%%,*}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
"$KEEP15_PYTHON" scripts/workflow.py smoke \
  --profile "$KEEP15_PROFILE" \
  --method-catalog configs/workflow/methods_keep15.json \
  --experiment configs/workflow/experiments/qwen25_keep15.json \
  --limit 1 --run-dir "$KEEP15_OUTPUT/smoke_${SLURM_JOB_ID}"

# set -e prevents full evaluation if any smoke cell fails. The full supervisor
# owns all four originally allocated devices; no extra job is submitted.
echo "SMOKE_PASS; FULL_START gpus=$CUDA_VISIBLE_DEVICES output=$KEEP15_OUTPUT"
exec "$KEEP15_PYTHON" "$KEEP15_OUTPUT/source/scripts/workflow.py" \
  execute --run-dir "$KEEP15_OUTPUT"
