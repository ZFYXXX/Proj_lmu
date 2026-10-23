#!/usr/bin/env bash
# Full MMSU only. This file is an entry point; creating it does not start a job.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
cd "$project_root"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact full/merged 7B LoRA checkpoint}"
analysis_python="${PYTHON_BIN:-python}"
analysis_run="${ANALYSIS_RUN_DIR:-runs/analysis/audio_sink/mmsufinal_suite_$(date +%Y%m%d_%H%M%S)}"
analysis_plan="${analysis_run}.plan.json"
mkdir -p "$(dirname -- "$analysis_plan")"
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

"$analysis_python" -m scripts.analysis.audio_sink.suite_plan \
  --checkpoint "$CHECKPOINT_PATH" --output "$analysis_plan" \
  --families "${ATTENTION_FAMILIES:-range,head,sink,evidence}" \
  --layers "${ATTENTION_LAYERS:-all}" --sink-heads "${SINK_HEADS:-all}"

"$analysis_python" -m scripts.analysis.audio_sink.launch \
  --config "$script_dir/eval_mmsufinal.yaml" \
  --checkpoint "$CHECKPOINT_PATH" --mode suite --suite-plan "$analysis_plan" \
  --output-dir "$analysis_run" --num-processes "${NUM_PROCESSES:-4}" \
  --main-process-port "${MAIN_PROCESS_PORT:-29649}" \
  --example-doc-ids 0,625,1250,1875,2500,3125,3750,4375

"$analysis_python" -m scripts.analysis.audio_sink.suite_report \
  --run-dir "$analysis_run" --group-field audio_path --bootstrap 1000

# Reuse the suite's native-reference records; this does not run the model again.
"$analysis_python" -m scripts.analysis.audio_sink.report \
  --run-dir "$analysis_run" --group-field audio_path --bootstrap 1000
