#!/bin/bash
#SBATCH --job-name=28-q25-dyn
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
: "${SEED_RUN_DIR:?Set SEED_RUN_DIR to the previous worker_*/ run}"
: "${REFERENCE_IDENTITY:?Set REFERENCE_IDENTITY}"
: "${OLD_MLP:?Set OLD_MLP}"
: "${VISUAL_SELECTOR:?Set VISUAL_SELECTOR}"
: "${IMAGE_SELECTOR:?Set IMAGE_SELECTOR}"
: "${QUESTION_SELECTOR:?Set QUESTION_SELECTOR}"
: "${PROTECTED_SELECTOR:?Set PROTECTED_SELECTOR}"

export WANDB_MODE=offline
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/shared" logs/slurm
[[ "$(nvidia-smi -L | wc -l)" -eq 4 ]] || { echo "Expected 4 allocated GPUs" >&2; exit 2; }

gpu_monitor="$RUN_DIR/logs/gpu_usage.log"
(while true; do
  date --iso-8601=seconds
  nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu,power.draw --format=csv,noheader
  sleep 60
done) >"$gpu_monitor" 2>&1 &
monitor_pid=$!

"$EVAL_PYTHON" - "$REFERENCE_IDENTITY" "$MODEL_PATH" "$RUN_DIR/shared" <<'PY'
import sys
from pathlib import Path
from scripts.eval.lmms_eval.qwen25_portable_queue import resolve_identity
resolve_identity(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
PY
LOCAL_IDENTITY="$RUN_DIR/shared/model_identity.json"
SHARED_CACHE="${CACHE_DIR:-$RUN_DIR/cache}"

run_worker() {
  local gpu="$1"
  local args=(scripts/eval/lmms_eval/qwen25_portable_queue.py
    --data-root "$DATA_ROOT" --model "$MODEL_PATH" --output "$RUN_DIR"
    --cache-dir "$SHARED_CACHE" --gpu "$gpu"
    --reference-identity "$LOCAL_IDENTITY"
    --checkpoint "old_mlp=$OLD_MLP"
    --checkpoint "visual_selector=$VISUAL_SELECTOR"
    --checkpoint "image_context_selector=$IMAGE_SELECTOR"
    --checkpoint "question_context_selector=$QUESTION_SELECTOR"
    --checkpoint "protected_selector=$PROTECTED_SELECTOR"
    --methods dense eadp prunesid mmtok old_mlp visual_selector image_context_selector question_context_selector protected_selector
    --tasks gqa scienceqa_img mmmu hallusionbench textvqa refcoco_bbox_rec_testA refcoco_bbox_rec_testB refcocog_bbox_rec_test vqav2_testdev
    --mmmu-single-image-only --chunk-size "${CHUNK_SIZE:-512}"
    --shared-workers 4 --worker-index "$gpu" --seed-from "$SEED_RUN_DIR" --resume)
  if [[ -n "${HALLUSION_IMAGE:-}" ]]; then args+=(--hallusion-image "$HALLUSION_IMAGE"); fi
  "$EVAL_PYTHON" -u "${args[@]}" >"$RUN_DIR/logs/worker_$gpu.log" 2>&1
}

pids=()
terminate_workers() {
  for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
  kill -TERM "$monitor_pid" 2>/dev/null || true
  wait || true
  exit 143
}
trap terminate_workers TERM INT

failed=0
for gpu in 0 1 2 3; do
  run_worker "$gpu" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then failed=1; fi
done
trap - TERM INT
kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true

"$EVAL_PYTHON" - "$RUN_DIR" <<'PY'
import json, os, pathlib, sys
root = pathlib.Path(sys.argv[1])
methods = ("dense", "eadp", "prunesid", "mmtok", "old_mlp", "visual_selector",
           "image_context_selector", "question_context_selector", "protected_selector")
tasks = ("gqa", "scienceqa_img", "mmmu", "hallusionbench", "textvqa",
         "refcoco_bbox_rec_testA", "refcoco_bbox_rec_testB", "refcocog_bbox_rec_test", "vqav2_testdev")
done = [(m, t) for m in methods for t in tasks if (root / "cells" / m / t / "complete.json").is_file()]
failed = [(m, t) for m in methods for t in tasks if (root / "cells" / m / t / "failed.json").is_file()]
state = "complete" if len(done) == len(methods) * len(tasks) else "incomplete"
payload = {"status": state, "completed_cells": len(done), "expected_cells": len(methods) * len(tasks),
           "failed_cells": failed, "methods": methods, "tasks": tasks,
           "pending_official_scoring": ["hallusionbench", "vqav2_testdev"],
           "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
tmp = root / "parallel_status.json.tmp"
tmp.write_text(json.dumps(payload, indent=2) + "\n")
os.replace(tmp, root / "parallel_status.json")
print(json.dumps(payload, indent=2))
if state != "complete" or failed:
    raise SystemExit(1)
PY
exit "$failed"
