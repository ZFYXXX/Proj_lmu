#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact full or merged Qwen2.5-Omni checkpoint}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONPATH="$ROOT:$ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_DIR="${RUN_DIR:-${ANALYSIS_RUN_DIR:-${OUTPUT_DIR:-$ROOT/runs/analysis/silent_redundancy_mmsu_4gpu_$(date +%Y%m%d_%H%M%S)}}}"

exec "$PYTHON_BIN" -m scripts.analysis.silent_redundancy_mmsu.run_suite \
  --checkpoint "$CHECKPOINT_PATH" \
  --run-dir "$RUN_DIR" \
  --num-processes 4 \
  --port-base "${PORT_BASE:-${MAIN_PROCESS_PORT:-29720}}" \
  "$@"
