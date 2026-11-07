#!/usr/bin/env bash
#SBATCH --job-name=28-local-fallback
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
: "${SLURM_JOB_ID:?Submit through Slurm; do not run inference on a login node}"
: "${CUDA_VISIBLE_DEVICES:?Slurm must expose allocated GPUs}"
FALLBACK_REPO="${FALLBACK_REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
FALLBACK_PYTHON="${FALLBACK_PYTHON:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp/bin/python}"
FALLBACK_PROFILE="${FALLBACK_PROFILE:-configs/workflow/profiles/lmu_local_fallback.json}"
FALLBACK_EXPERIMENT="${FALLBACK_EXPERIMENT:-configs/workflow/experiments/qwen25_local_fallback.json}"
FALLBACK_OUTPUT="${FALLBACK_OUTPUT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace/outputs/Fengyu/qwen25_keep15_local4_fallback_v1}"
read -r -a FALLBACK_METHOD_LIST <<< "${FALLBACK_METHODS:-importance_fps_balanced_local4_gap01_r15}"
FALLBACK_TASK_ARGS=()
if [[ -n "${FALLBACK_TASKS:-}" ]]; then
  read -r -a FALLBACK_TASK_LIST <<< "$FALLBACK_TASKS"
  FALLBACK_TASK_ARGS=(--tasks "${FALLBACK_TASK_LIST[@]}")
fi
export WANDB_MODE=offline HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
# Slurm owns these devices: do not inherit local workstation sharing caps.
unset MLLM_TOPK_SHARED_GPU MLLM_TOPK_NORMAL_GPU MLLM_TOPK_MEMORY_FRACTION MLLM_TOPK_DUTY_CYCLE
cd "$FALLBACK_REPO"
FALLBACK_RESUME=()
if [[ -f "$FALLBACK_OUTPUT/workflow.json" ]]; then FALLBACK_RESUME=(--resume); fi
"$FALLBACK_PYTHON" scripts/workflow.py prepare \
  --profile "$FALLBACK_PROFILE" --method-catalog configs/workflow/methods_local_assignment.json \
  --experiment "$FALLBACK_EXPERIMENT" \
  --methods "${FALLBACK_METHOD_LIST[@]}" "${FALLBACK_TASK_ARGS[@]}" --run-dir "$FALLBACK_OUTPUT" "${FALLBACK_RESUME[@]}"

FALLBACK_SMOKE="$FALLBACK_OUTPUT/smoke_${SLURM_JOB_ID}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}" \
"$FALLBACK_PYTHON" scripts/workflow.py smoke \
  --profile "$FALLBACK_PROFILE" --method-catalog configs/workflow/methods_local_assignment.json \
  --experiment "$FALLBACK_EXPERIMENT" \
  --methods "${FALLBACK_METHOD_LIST[@]}" "${FALLBACK_TASK_ARGS[@]}" --limit 2 --run-dir "$FALLBACK_SMOKE"
"$FALLBACK_PYTHON" - "$FALLBACK_OUTPUT" "$FALLBACK_SMOKE" <<'PY'
import json, pathlib, sys
full, smoke = [json.loads((pathlib.Path(p) / 'workflow.json').read_text()) for p in sys.argv[1:]]
if full['source_hashes'] != smoke['source_hashes'] or full['config'] != smoke['config']:
    raise SystemExit('Source or configuration changed between full preparation and smoke')
PY
echo "SMOKE_PASS; FULL_START gpus=$CUDA_VISIBLE_DEVICES output=$FALLBACK_OUTPUT"
exec "$FALLBACK_PYTHON" "$FALLBACK_OUTPUT/source/scripts/workflow.py" execute --run-dir "$FALLBACK_OUTPUT"
