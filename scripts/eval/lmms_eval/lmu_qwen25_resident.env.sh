#!/usr/bin/env bash
# Edit paths here, or override individual values before calling the submit script.
export REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
export EVAL_PYTHON="${EVAL_PYTHON:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/zfy-mllm-comp/bin/python}"
export WS="${WS:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace}"
export DATA_ROOT="${DATA_ROOT:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/mllm_dataset/lmms_lab_dataset}"
export MODEL_PATH="${MODEL_PATH:-$WS/ckpts/ckpts-lmm/Qwen2.5-VL-7B-Instruct}"
export CKPT_ROOT="${CKPT_ROOT:-$WS/ckpts/ckpts-lmm/Proj}"
export REFERENCE_IDENTITY="${REFERENCE_IDENTITY:-$CKPT_ROOT/model_identity.json}"
export OLD_MLP="${OLD_MLP:-$CKPT_ROOT/checkpoints/old_mlp}"
export VISUAL_SELECTOR="${VISUAL_SELECTOR:-$CKPT_ROOT/checkpoints/visual_selector}"
export IMAGE_SELECTOR="${IMAGE_SELECTOR:-$CKPT_ROOT/checkpoints/image_context_selector}"
export QUESTION_SELECTOR="${QUESTION_SELECTOR:-$CKPT_ROOT/checkpoints/question_context_selector}"
export PROTECTED_SELECTOR="${PROTECTED_SELECTOR:-$CKPT_ROOT/checkpoints/protected_selector}"
export RUN_DIR="${RUN_DIR:-$WS/outputs/qwen25_resident_bench_20261103_v2}"
export CACHE_DIR="${CACHE_DIR:-$WS/Fengyu/cache/qwen25_resident}"
export SEED_RUN_DIRS="${SEED_RUN_DIRS-$WS/outputs/qwen25_remaining_dynamic_testdev_20261101:$WS/outputs/qwen25_remaining_benchmarks_20261031}"
export CHUNK_SIZE="${CHUNK_SIZE:-1024}"
export EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG:-$REPO/configs/eval/lmms_eval/qwen25_r1t2_completion.json}"
export METHODS="${METHODS-}"
export RESULT_BUNDLE="${RESULT_BUNDLE-$REPO/artifacts/eval/qwen25_p6000_reuse_20261103.json.gz}"
export TASKS="${TASKS-}"

# Existing method bindings supply the algorithms; the JSON chooses their tasks.
declare -A METHOD_CHECKPOINTS=(
  [old_mlp]="$OLD_MLP" [importance_fps]="$OLD_MLP" [importance_fps_pool]="$OLD_MLP"
  [visual_selector]="$VISUAL_SELECTOR" [image_context_selector]="$IMAGE_SELECTOR"
  [question_context_selector]="$QUESTION_SELECTOR" [protected_selector]="$PROTECTED_SELECTOR"
)
resolve_resident_selection() {
  local selection
  selection=$(PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}" "$EVAL_PYTHON" - "$EXPERIMENT_CONFIG" "$METHODS" "$TASKS" <<'PY'
import sys
from scripts.eval.lmms_eval.qwen25_portable_queue import select_experiment
methods, tasks, cells = select_experiment(sys.argv[1], sys.argv[2].split() or None, sys.argv[3].split() or None)
if 'dense' in methods:
    raise ValueError('This continuation job reuses dense references; remove dense from method_tasks')
print(' '.join(methods))
print(' '.join(tasks))
print(len(cells))
PY
  ) || return $?
  METHODS=$(sed -n '1p' <<< "$selection")
  TASKS=$(sed -n '2p' <<< "$selection")
  REQUESTED_CELL_COUNT=$(sed -n '3p' <<< "$selection")
  export METHODS TASKS REQUESTED_CELL_COUNT
}
