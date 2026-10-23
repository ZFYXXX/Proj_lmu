#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPERIMENT="${1:-${EXPERIMENT:-observe}}"
if [[ $# -gt 0 ]]; then shift; fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES=2

# Keep this entry point at two workers even if the caller passes a process count.
# All experiment options and automatic reporting remain owned by the shared runner.
exec bash "$SCRIPT_DIR/run_experiment.sh" "$EXPERIMENT" "$@" --num-processes 2
