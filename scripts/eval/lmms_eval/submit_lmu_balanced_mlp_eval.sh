#!/usr/bin/env bash
# Evaluate the bundled balanced step320 MLP; no retraining or HF token is needed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
BALANCED_REPO="$PWD"
export OLD_MLP="$BALANCED_REPO/artifacts/checkpoints/r1_balanced2560_step320"
export WS="${WS:-/hkfs/work/workspace/scratch/tum_yvc3016-comp/workspace}"
export RUN_DIR="${BALANCED_EVAL_RUN_DIR:-$WS/outputs/Fengyu/qwen25_r1_balanced_step320_github_eval_v1}"
export EXPERIMENT_CONFIG="$BALANCED_REPO/configs/workflow/experiments/r1_balanced_full_eval.json"
export METHODS=importance_fps
export TASKS=""
export CHUNK_SIZE="${CHUNK_SIZE:-512}"
export RESULT_BUNDLE="$BALANCED_REPO/artifacts/eval/qwen25_p6000_reuse_20261103.json.gz"

(cd "$OLD_MLP" && sha256sum -c SHA256SUMS)
echo "Evaluating bundled balanced R1 step320; existing base model and image-context scorer are reused."
exec bash scripts/eval/lmms_eval/submit_lmu_qwen25_resident.sh
