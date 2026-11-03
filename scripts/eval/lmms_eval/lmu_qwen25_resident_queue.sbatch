#!/bin/bash
#SBATCH --job-name=28-q25-resident
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
cd "${SLURM_SUBMIT_DIR:?Submit from the project root}"
source scripts/eval/lmms_eval/lmu_qwen25_resident.env.sh
resolve_resident_selection
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
unset MLLM_EXTRA_SMOKE MLLM_TOPK_SHARED_GPU
mkdir -p "$RUN_DIR/logs" "$RUN_DIR/shared" "$CACHE_DIR"

"$EVAL_PYTHON" - "$REFERENCE_IDENTITY" "$MODEL_PATH" "$RUN_DIR/shared" <<'PY'
import sys
from pathlib import Path
from scripts.eval.lmms_eval.qwen25_portable_queue import resolve_identity
resolve_identity(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
PY
export REFERENCE_IDENTITY="$RUN_DIR/shared/model_identity.json"

# Respect Slurm's GPU visibility; local CUDA index is always zero in each worker.
IFS=',' read -r -a devices <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
[[ ${#devices[@]} -eq 4 ]] || { echo "Expected four allocated CUDA devices" >&2; exit 2; }
(while true; do
    date --iso-8601=seconds
    nvidia-smi --query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu --format=csv,noheader
    sleep 60
done) >>"$RUN_DIR/logs/gpu_usage.log" 2>&1 &
monitor_pid=$!
pids=()
stop_workers() {
    for pid in "${pids[@]}"; do kill -TERM "$pid" 2>/dev/null || true; done
    kill -TERM "$monitor_pid" 2>/dev/null || true
    wait || true
}
trap 'stop_workers; exit 143' TERM INT

for index in 0 1 2 3; do
    args=(scripts/eval/lmms_eval/qwen25_portable_queue.py
        --data-root "$DATA_ROOT" --model "$MODEL_PATH" --output "$RUN_DIR"
        --cache-dir "$CACHE_DIR" --reference-identity "$REFERENCE_IDENTITY"
        --experiment "$EXPERIMENT_CONFIG"
        --resident-model --reference-only --gpu 0
        --chunk-size "$CHUNK_SIZE" --shared-workers 4 --worker-index "$index"
        --mmmu-single-image-only --resume)
    read -r -a selected_methods <<< "$METHODS"
    read -r -a selected_tasks <<< "$TASKS"
    args+=(--methods "${selected_methods[@]}" --tasks "${selected_tasks[@]}")
    for method in "${selected_methods[@]}"; do
        if [[ -n "${METHOD_CHECKPOINTS[$method]:-}" ]]; then args+=(--checkpoint "$method=${METHOD_CHECKPOINTS[$method]}"); fi
    done
    if [[ " $METHODS " == *" importance_fps "* || " $METHODS " == *" importance_fps_pool "* ]]; then
        args+=(--scorer-checkpoint "$IMAGE_SELECTOR")
    fi
    IFS=':' read -r -a seed_dirs <<< "$SEED_RUN_DIRS"
    for previous in "${seed_dirs[@]}"; do
        [[ -n "$previous" ]] && args+=(--seed-from "$previous")
    done
    if [[ -n "${RESULT_BUNDLE:-}" ]]; then args+=(--result-bundle "$RESULT_BUNDLE"); fi
    if [[ -n "${HALLUSION_IMAGE:-}" ]]; then args+=(--hallusion-image "$HALLUSION_IMAGE"); fi
    if [[ -n "${SMOKE_LIMIT:-}" ]]; then args+=(--limit "$SMOKE_LIMIT"); fi
    CUDA_VISIBLE_DEVICES="${devices[$index]}" "$EVAL_PYTHON" -u "${args[@]}" >>"$RUN_DIR/logs/worker_$index.log" 2>&1 &
    pids+=("$!")
done

remaining=("${pids[@]}")
failed=0
while [[ ${#remaining[@]} -gt 0 ]]; do
    next=()
    for pid in "${remaining[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            next+=("$pid")
        elif ! wait "$pid"; then
            failed=1
        fi
    done
    if [[ "$failed" -ne 0 ]]; then stop_workers; break; fi
    remaining=("${next[@]}")
    [[ ${#remaining[@]} -eq 0 ]] || sleep 5
done
kill -TERM "$monitor_pid" 2>/dev/null || true
wait "$monitor_pid" 2>/dev/null || true
trap - TERM INT

"$EVAL_PYTHON" - "$RUN_DIR" "$failed" <<'PY'
import json, os, sys
from pathlib import Path
from scripts.eval.lmms_eval.qwen25_portable_queue import dump, report
root=Path(sys.argv[1]); plan=json.loads((root/'plan.json').read_text())
methods,tasks=[m for m in plan['methods'] if m!='dense'],plan['tasks']
requested=plan['requested_cells']
done=[(m,t) for m,t in requested if (root/'cells'/m/t/'complete.json').is_file()]
report(root,plan['methods'],tasks,requested)
status={'status':'complete' if len(done)==len(requested) and sys.argv[2]=='0' else 'incomplete',
        'completed_cells':len(done),'expected_cells':len(requested), 'requested_cells':requested,
        'slurm_job_id':os.environ.get('SLURM_JOB_ID'), 'methods':methods,'tasks':tasks}
dump(root/'parallel_status.json',status)
print(json.dumps(status,indent=2))
if status['status']!='complete': raise SystemExit(1)
PY
exit "$failed"
