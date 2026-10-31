#!/usr/bin/env bash
set -euo pipefail

# Run this lightweight submitter from the LMU project checkout. It only creates
# the log directory and submits one Slurm job; evaluation runs on 4 A100 GPUs.
REPO="${REPO:-/home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu}"
ENV="${ENV:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp}"
WS="${WS:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace}"
DATA_ROOT="${DATA_ROOT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset}"
MODEL_PATH="${MODEL_PATH:-$WS/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$WS/ckpts/ckpts-lmm/Proj}"
RUN_ID="${RUN_ID:-qwen25_remaining_benchmarks_20261031}"
RUN_DIR="${RUN_DIR:-$WS/outputs/$RUN_ID}"

export EVAL_PYTHON="${EVAL_PYTHON:-$ENV/bin/python}"
export DATA_ROOT MODEL_PATH RUN_DIR
export REFERENCE_IDENTITY="${REFERENCE_IDENTITY:-$CKPT_ROOT/model_identity.json}"
export OLD_MLP="${OLD_MLP:-$CKPT_ROOT/checkpoints/old_mlp}"
export VISUAL_SELECTOR="${VISUAL_SELECTOR:-$CKPT_ROOT/checkpoints/visual_selector}"
export IMAGE_SELECTOR="${IMAGE_SELECTOR:-$CKPT_ROOT/checkpoints/image_context_selector}"
export QUESTION_SELECTOR="${QUESTION_SELECTOR:-$CKPT_ROOT/checkpoints/question_context_selector}"
export PROTECTED_SELECTOR="${PROTECTED_SELECTOR:-$CKPT_ROOT/checkpoints/protected_selector}"
export CACHE_DIR="${CACHE_DIR:-$WS/cache/qwen25_lmms_eval}"

cd "$REPO"
if squeue -h -u "${USER:?USER is not set}" -n 28-qwen25-eval | grep -q .; then
  echo "A 28-qwen25-eval job is already pending or running; refusing duplicate submission." >&2
  exit 3
fi

for path in "$EVAL_PYTHON" "$DATA_ROOT" "$MODEL_PATH" "$REFERENCE_IDENTITY" \
            "$OLD_MLP" "$VISUAL_SELECTOR" "$IMAGE_SELECTOR" \
            "$QUESTION_SELECTOR" "$PROTECTED_SELECTOR"; do
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 2
  fi
done

mkdir -p logs/slurm "$RUN_DIR" "$CACHE_DIR"
echo "Submitting Qwen2.5-VL-7B evaluation on one 4xA100 node (max 24h)."
echo "Run directory: $RUN_DIR"
echo "Datasets: gqa, vqav2_val, scienceqa_img, textvqa, single-image MMMU, HallusionBench, RefCOCO testA/testB, RefCOCOg"
echo "Methods: dense, EADP, PruneSID, MMTok, old MLP, visual Top-K, image context, question context, protected selector"

sbatch --export=ALL scripts/eval/lmms_eval/lmu_qwen25_queue.sbatch
