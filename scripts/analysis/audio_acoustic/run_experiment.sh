#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT:$ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
PYTHON_BIN="${PYTHON_BIN:-python}"
EXPERIMENT="${1:-${EXPERIMENT:-observe}}"
if [[ $# -gt 0 ]]; then shift; fi
: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full Qwen2.5-Omni checkpoint}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR to a new run directory}"

"$PYTHON_BIN" -m scripts.analysis.audio_acoustic.launch \
  --experiment "$EXPERIMENT" --checkpoint "$CHECKPOINT_PATH" --output-dir "$OUTPUT_DIR" \
  --num-processes "${NUM_PROCESSES:-4}" --main-process-port "${MAIN_PROCESS_PORT:-29669}" "$@"
for argument in "$@"; do
  if [[ "$argument" == "--plan-only" ]]; then exit 0; fi
done
"$PYTHON_BIN" -m scripts.analysis.audio_acoustic.report --run-dir "$OUTPUT_DIR" \
  --bootstrap "${BOOTSTRAP_REPEATS:-1000}" --tolerance-pp "${TOLERANCE_PP:-1.0}"
