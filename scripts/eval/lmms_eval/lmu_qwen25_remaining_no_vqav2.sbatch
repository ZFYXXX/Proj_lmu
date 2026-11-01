#!/bin/bash
#SBATCH --job-name=28-q25-remain
#SBATCH --account=hk-project-p0024767
#SBATCH --partition=accelerated
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/slurm/%x_%j_output.log
#SBATCH --error=logs/slurm/%x_%j_error.log
#SBATCH --mail-type="NONE"

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
: "${EVAL_PYTHON:?Set EVAL_PYTHON}"
: "${DATA_ROOT:?Set DATA_ROOT}"
: "${MODEL_PATH:?Set MODEL_PATH}"
: "${RUN_DIR:?Set RUN_DIR}"
: "${REFERENCE_IDENTITY:?Set REFERENCE_IDENTITY}"
: "${OLD_MLP:?Set OLD_MLP}"
: "${VISUAL_SELECTOR:?Set VISUAL_SELECTOR}"
: "${IMAGE_SELECTOR:?Set IMAGE_SELECTOR}"
: "${QUESTION_SELECTOR:?Set QUESTION_SELECTOR}"
: "${PROTECTED_SELECTOR:?Set PROTECTED_SELECTOR}"

export WANDB_MODE=offline
mkdir -p "$RUN_DIR/shared" "$RUN_DIR/logs"
(while true; do
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
  sleep 60
done) >"$RUN_DIR/logs/gpu_usage.log" 2>&1 &
monitor_pid="$!"

"$EVAL_PYTHON" - "$REFERENCE_IDENTITY" "$MODEL_PATH" "$RUN_DIR/shared" <<'PY'
import sys
from pathlib import Path
from scripts.eval.lmms_eval.qwen25_portable_queue import resolve_identity
resolve_identity(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
PY
LOCAL_IDENTITY="$RUN_DIR/shared/model_identity.json"
SHARED_CACHE="${CACHE_DIR:-$RUN_DIR/cache}"
mkdir -p "$SHARED_CACHE"

run_worker() {
  local worker="$1" gpu="$2"
  shift 2
  local worker_dir="$RUN_DIR/worker_$worker"
  local args=(scripts/eval/lmms_eval/qwen25_portable_queue.py
    --data-root "$DATA_ROOT" --model "$MODEL_PATH" --output "$worker_dir"
    --cache-dir "$SHARED_CACHE" --gpu "$gpu"
    --reference-identity "$LOCAL_IDENTITY"
    --checkpoint "old_mlp=$OLD_MLP"
    --checkpoint "visual_selector=$VISUAL_SELECTOR"
    --checkpoint "image_context_selector=$IMAGE_SELECTOR"
    --checkpoint "question_context_selector=$QUESTION_SELECTOR"
    --checkpoint "protected_selector=$PROTECTED_SELECTOR"
    --methods dense prunesid mmtok old_mlp visual_selector image_context_selector question_context_selector protected_selector
    --tasks "$@" --mmmu-single-image-only --chunk-size "${CHUNK_SIZE:-512}")
  if [[ -n "${HALLUSION_IMAGE:-}" ]]; then
    args+=(--hallusion-image "$HALLUSION_IMAGE")
  fi
  if [[ -f "$worker_dir/plan.json" ]]; then
    args+=(--resume)
  fi
  "$EVAL_PYTHON" -u "${args[@]}" >"$RUN_DIR/logs/worker_$worker.log" 2>&1
}

run_worker 0 0 mmmu &
pids=("$!")
run_worker 1 1 hallusionbench &
pids+=("$!")
run_worker 2 2 refcoco_bbox_rec_testA refcoco_bbox_rec_testB &
pids+=("$!")
run_worker 3 3 refcocog_bbox_rec_test &
pids+=("$!")

terminate_workers() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  kill -TERM "$monitor_pid" 2>/dev/null || true
  wait || true
  exit 143
}
trap terminate_workers TERM INT

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
trap - TERM INT
kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
"$EVAL_PYTHON" scripts/eval/lmms_eval/summarize_qwen25_parallel.py "$RUN_DIR" --workers 4
exit "$failed"
