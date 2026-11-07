#!/usr/bin/env bash
#SBATCH --job-name=28-local-factorial
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
FACTORIAL_REPO="${FALLBACK_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
FACTORIAL_OUTPUT="${FACTORIAL_OUTPUT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_local_factorial_v1}"
export FALLBACK_EXPERIMENT=configs/workflow/experiments/qwen25_local_factorial.json
export FALLBACK_METHODS="${FALLBACK_METHODS:-importance_fps_balanced_local4_gap01_r15 importance_fps_balanced_grid4_local4_r15 importance_fps_balanced_grid4_gap01_r15}"
cd "$FACTORIAL_REPO"
FACTORIAL_CHILD=""
trap 'if [[ -n "$FACTORIAL_CHILD" ]]; then kill -TERM "$FACTORIAL_CHILD" 2>/dev/null || true; fi; exit 143' TERM INT
# One allocation, four shared chunk workers per phase, strict Ref -> OCR -> other.
for FACTORIAL_PHASE in ref ocr other; do
  case "$FACTORIAL_PHASE" in
    ref) export FALLBACK_TASKS="refcoco_bbox_rec_testB refcoco+_bbox_rec_testB refcoco_bbox_rec_testA refcoco+_bbox_rec_testA refcocog_bbox_rec_test" ;;
    ocr) export FALLBACK_TASKS="ocrbench docvqa textvqa" ;;
    other) export FALLBACK_TASKS="mmstar ai2d pope gqa scienceqa_img okvqa mmmu hallusionbench" ;;
  esac
  export FALLBACK_OUTPUT="$FACTORIAL_OUTPUT/$FACTORIAL_PHASE"
  bash scripts/eval/lmms_eval/lmu_qwen25_local_fallback.sbatch &
  FACTORIAL_CHILD=$!
  wait "$FACTORIAL_CHILD"
  FACTORIAL_CHILD=""
done
