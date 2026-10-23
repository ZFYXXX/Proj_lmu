#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--plan-only" && "$1" != "--help" && "$1" != "-h" ) ]]; then
  printf 'Usage: bash %s [--plan-only]\n' "$0" >&2
  exit 2
fi
if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'HELP'
Run all 10 acoustic analyses on all 5,000 MMSU-final questions with two GPUs.
Every intervention preserves native token counts; no trained probe is required.
Shared inference: 56 distinct interventions + one native baseline = 285,000 generations.
Usage: bash run_all_mmsufinal_2gpu.sh [--plan-only]

Required: CHECKPOINT_PATH (full Qwen2.5-Omni checkpoint).
Optional: PYTHON_BIN, CUDA_VISIBLE_DEVICES (default 0,1), RUN_GROUP (new directory),
          MMSU_JSONL, MMSUFINAL_DATASET_DIR, MAIN_PROCESS_PORT,
          BOOTSTRAP_REPEATS, TOLERANCE_PP.
Outputs: shared records in RUN_GROUP; 10 views in RUN_GROUP/report/<experiment>/.
--plan-only prints the shared plan without loading weights or creating outputs.
Use run_experiment_2gpu.sh all --limit N for a small end-to-end smoke test.
HELP
  exit 0
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full Qwen2.5-Omni checkpoint}"
export PYTHON_BIN="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export RUN_GROUP="${RUN_GROUP:-$ROOT/runs/analysis/acoustic_mmsufinal_2gpu_$(date +%Y%m%d_%H%M%S)}"
export OUTPUT_DIR="$RUN_GROUP"

if [[ "${1:-}" == "--plan-only" ]]; then
  exec bash "$SCRIPT_DIR/run_experiment_2gpu.sh" all --plan-only
fi

mkdir -p -- "$(dirname -- "$RUN_GROUP")"
# Keep previous logs intact, just as launch.py refuses an existing run directory.
(set -o noclobber; : > "$RUN_GROUP.log")
bash "$SCRIPT_DIR/run_experiment_2gpu.sh" all 2>&1 | tee -a "$RUN_GROUP.log"
