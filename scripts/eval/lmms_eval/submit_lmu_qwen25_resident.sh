#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
source scripts/eval/lmms_eval/lmu_qwen25_resident.env.sh

jobs=$(squeue -h -u "${USER:?}" -o '%i %j' | awk '$2 ~ /^28-/')
if [[ -n "$jobs" ]]; then
    echo "An ID-28 job is pending/running; keeping the one-node quota:" >&2
    echo "$jobs" >&2
    exit 3
fi
resolve_resident_selection
required_paths=("$EVAL_PYTHON" "$DATA_ROOT" "$MODEL_PATH" "$REFERENCE_IDENTITY" "$EXPERIMENT_CONFIG")
for method in $METHODS; do
    if [[ -n "${METHOD_CHECKPOINTS[$method]:-}" ]]; then required_paths+=("${METHOD_CHECKPOINTS[$method]}"); fi
    if [[ "$method" == importance_fps || "$method" == importance_fps_pool ]]; then required_paths+=("$IMAGE_SELECTOR"); fi
done
for path in "${required_paths[@]}"; do
    [[ -e "$path" ]] || { echo "Missing required path: $path" >&2; exit 2; }
done
if [[ -n "$RESULT_BUNDLE" && ! -f "$RESULT_BUNDLE" ]]; then
    echo "Missing result reuse bundle: $RESULT_BUNDLE" >&2
    exit 2
fi
IFS=':' read -r -a seed_dirs <<< "$SEED_RUN_DIRS"
for previous in "${seed_dirs[@]}"; do
    [[ -z "$previous" || -d "$previous" ]] || { echo "Missing reuse directory: $previous" >&2; exit 2; }
done
if [[ " $TASKS " == *" vqav2_val "* || " $TASKS " == *" vqav2_testdev "* ]]; then
    echo "VQAv2 is excluded from this continuation suite." >&2
    exit 2
fi
mkdir -p logs/slurm "$RUN_DIR" "$CACHE_DIR"
bash -n scripts/eval/lmms_eval/lmu_qwen25_resident_queue.sbatch
echo "Run directory: $RUN_DIR"
echo "Methods: $METHODS"
echo "Tasks: $TASKS"
echo "Experiment: $EXPERIMENT_CONFIG ($REQUESTED_CELL_COUNT requested method/task cells)"
echo "One node, four resident workers, shared work queue, raw and normalized scores."
sbatch --export=ALL scripts/eval/lmms_eval/lmu_qwen25_resident_queue.sbatch
