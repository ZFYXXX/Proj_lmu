#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu}"
ENV="${ENV:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp}"
WS="${WS:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace}"
DATA_ROOT="${DATA_ROOT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset}"
MODEL_PATH="${MODEL_PATH:-$WS/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$WS/ckpts/ckpts-lmm/Proj}"
SEED_RUN_DIR="${SEED_RUN_DIR:-$WS/outputs/qwen25_remaining_benchmarks_20261031}"
RUN_ID="${RUN_ID:-qwen25_remaining_dynamic_testdev_20261101}"
RUN_DIR="${RUN_DIR:-$WS/outputs/$RUN_ID}"

export EVAL_PYTHON="${EVAL_PYTHON:-$ENV/bin/python}"
export DATA_ROOT MODEL_PATH RUN_DIR SEED_RUN_DIR
export REFERENCE_IDENTITY="${REFERENCE_IDENTITY:-$CKPT_ROOT/model_identity.json}"
export OLD_MLP="${OLD_MLP:-$CKPT_ROOT/checkpoints/old_mlp}"
export VISUAL_SELECTOR="${VISUAL_SELECTOR:-$CKPT_ROOT/checkpoints/visual_selector}"
export IMAGE_SELECTOR="${IMAGE_SELECTOR:-$CKPT_ROOT/checkpoints/image_context_selector}"
export QUESTION_SELECTOR="${QUESTION_SELECTOR:-$CKPT_ROOT/checkpoints/question_context_selector}"
export PROTECTED_SELECTOR="${PROTECTED_SELECTOR:-$CKPT_ROOT/checkpoints/protected_selector}"
export CACHE_DIR="${CACHE_DIR:-$WS/cache/qwen25_lmms_eval}"

cd "$REPO"
existing_jobs=$(squeue -h -u "${USER:?USER is not set}" -o '%i %j' | awk '$2 ~ /^28-/')
if [[ -n "$existing_jobs" ]]; then
  echo "An ID-28 job is already pending or running; keeping the one-node quota:" >&2
  echo "$existing_jobs" >&2
  exit 3
fi

for path in "$EVAL_PYTHON" "$DATA_ROOT" "$MODEL_PATH" "$REFERENCE_IDENTITY" \
            "$OLD_MLP" "$VISUAL_SELECTOR" "$IMAGE_SELECTOR" "$QUESTION_SELECTOR" \
            "$PROTECTED_SELECTOR" "$SEED_RUN_DIR"; do
  [[ -e "$path" ]] || { echo "Required path does not exist: $path" >&2; exit 2; }
done
mkdir -p logs/slurm "$RUN_DIR" "$CACHE_DIR"
echo "Submitting dynamic 4xA100 queue; each GPU claims the next unfinished request chunk."
echo "Completed cells will be seeded from: $SEED_RUN_DIR"
echo "VQAv2 uses test-dev (testdev-*.parquet); per-method submissions are saved for official scoring."
echo "Run directory: $RUN_DIR"
sbatch --export=ALL scripts/eval/lmms_eval/lmu_qwen25_dynamic_queue.sbatch
