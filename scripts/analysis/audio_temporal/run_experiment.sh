#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 EXPERIMENT OUTPUT_DIR [launch options...]" >&2
  exit 2
fi
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "CHECKPOINT_PATH must point to the full or merged Qwen2.5-Omni checkpoint" >&2
  exit 2
fi

experiment_name="$1"
analysis_output="$2"
shift 2
python_command="${PYTHON_BIN:-python}"

"$python_command" -m scripts.analysis.audio_temporal.launch \
  --experiment "$experiment_name" \
  --checkpoint "$CHECKPOINT_PATH" \
  --output-dir "$analysis_output" \
  "$@"

"$python_command" -m scripts.analysis.audio_temporal.report \
  --run-dir "$analysis_output"
