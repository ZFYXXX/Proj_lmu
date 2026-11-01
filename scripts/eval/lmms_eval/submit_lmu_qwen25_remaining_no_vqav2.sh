#!/usr/bin/env bash
set -euo pipefail

REPO="${REPO:-/home/hk-project-p0022189/tum_yvc3016/haowu/Fengyu/Proj_lmu}"
ENV="${ENV:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp}"
WS="${WS:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace}"
DATA_ROOT="${DATA_ROOT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset}"
MODEL_PATH="${MODEL_PATH:-$WS/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct}"
CKPT_ROOT="${CKPT_ROOT:-$WS/ckpts/ckpts-lmm/Proj}"
RUN_ID="${RUN_ID:-qwen25_remaining_no_vqav2_20261101}"
RUN_DIR="${RUN_DIR:-$WS/outputs/$RUN_ID}"
PREVIOUS_RUN_DIR="${PREVIOUS_RUN_DIR:-$WS/outputs/qwen25_remaining_benchmarks_20261031}"

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
  echo "Cancel the 28-qwen25-eval job before submitting this continuation." >&2
  exit 3
fi
if squeue -h -u "$USER" -n 28-q25-remain | grep -q .; then
  echo "A 28-q25-remain job is already pending or running." >&2
  exit 3
fi
if [[ -e "$RUN_DIR/parallel_status.json" ]]; then
  echo "Continuation run already has a final status; choose another RUN_ID." >&2
  exit 3
fi

for path in "$EVAL_PYTHON" "$DATA_ROOT" "$MODEL_PATH" "$REFERENCE_IDENTITY" \
            "$OLD_MLP" "$VISUAL_SELECTOR" "$IMAGE_SELECTOR" \
            "$QUESTION_SELECTOR" "$PROTECTED_SELECTOR"; do
  [[ -e "$path" ]] || { echo "Required path does not exist: $path" >&2; exit 2; }
done

mkdir -p logs/slurm "$RUN_DIR" "$CACHE_DIR"

# Reuse the already completed dense and EADP cells for the five unfinished
# task splits. Copy only ledger markers; prediction payloads stay in the old run.
for worker_task in \
  "0 mmmu" \
  "1 hallusionbench" \
  "2 refcoco_bbox_rec_testA" \
  "2 refcoco_bbox_rec_testB" \
  "3 refcocog_bbox_rec_test"; do
  read -r worker task <<<"$worker_task"
  source_cell="$PREVIOUS_RUN_DIR/worker_3/cells"
  target_cell="$RUN_DIR/worker_$worker/cells"
  for method in dense eadp; do
    [[ -f "$source_cell/$method/$task/complete.json" ]] || {
      echo "Missing completed source cell: $source_cell/$method/$task" >&2
      exit 4
    }
    mkdir -p "$target_cell/$method/$task"
    cp -n "$source_cell/$method/$task/complete.json" "$target_cell/$method/$task/"
    for marker in "$source_cell/$method/$task"/chunk_*/complete.json; do
      [[ -f "$marker" ]] || continue
      chunk_dir="$(basename "$(dirname "$marker")")"
      mkdir -p "$target_cell/$method/$task/$chunk_dir"
      cp -n "$marker" "$target_cell/$method/$task/$chunk_dir/"
    done
  done
done

echo "Run directory: $RUN_DIR"
echo "Tasks: single-image MMMU, HallusionBench, RefCOCO testA/testB, RefCOCOg test"
echo "Methods: dense reference plus unfinished methods from PruneSID onward; completed EADP is retained in the previous run"
echo "VQAv2 val is excluded."
sbatch --export=ALL scripts/eval/lmms_eval/lmu_qwen25_remaining_no_vqav2.sbatch
