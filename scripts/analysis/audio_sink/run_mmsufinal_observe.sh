#!/usr/bin/env bash
# Full MMSU, one native generation per sample, observational diagnostics only.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
cd "$project_root"

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to the exact full 7B checkpoint (base or merged LoRA)}"
: "${MMSUFINAL_DATASET_DIR:?Set MMSUFINAL_DATASET_DIR to the MMSU Parquet/audio-cache directory}"
analysis_python="${PYTHON_BIN:-python}"
analysis_run="${ANALYSIS_RUN_DIR:-runs/analysis/audio_sink/mmsufinal_7b_all_layers_observe_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$(dirname -- "$analysis_run")"
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

# Observe every encoder layer; the observer always includes every head.
# Read architecture only, without loading weights or using a calibration set.
analysis_layers="$("$analysis_python" - "$CHECKPOINT_PATH" <<'PY'
import json
from pathlib import Path
import sys

config = json.loads((Path(sys.argv[1]).expanduser() / 'config.json').read_text())
count = config['thinker_config']['audio_config']['encoder_layers']
print(','.join(map(str, range(count))))
PY
)"

printf 'MMSU observe: 5000 samples, one generation per sample; layers=%s (all heads).\n' "$analysis_layers"

"$analysis_python" -m scripts.analysis.audio_sink.launch \
  --config "$script_dir/eval_mmsufinal.yaml" \
  --checkpoint "$CHECKPOINT_PATH" --mode observe \
  --output-dir "$analysis_run" --num-processes "${NUM_PROCESSES:-4}" \
  --main-process-port "${MAIN_PROCESS_PORT:-29649}" \
  --layers "$analysis_layers" --query-block 128 --check-av-samples 1 \
  --example-doc-ids 0,1666,3333 --example-chunk 0 \
  --virtual-keep-ratio 0.1 --segments 10

# The report uses saved measurements; no second model pass or interventions.
"$analysis_python" -m scripts.analysis.audio_sink.report \
  --run-dir "$analysis_run" --group-field audio_path --bootstrap 1000

# Same saved observations, grouped by the official MMSU two-by-three taxonomy.
"$analysis_python" -m scripts.analysis.audio_sink.group_report \
  --run-dir "$analysis_run" --dataset-dir "$MMSUFINAL_DATASET_DIR"
