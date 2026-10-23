#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: bash scripts/analysis/audio_four_probes/run.sh [mmsufinal] [launch options]

Required: CHECKPOINT_PATH (full local Qwen2.5-Omni base or merged checkpoint)
Optional: PYTHON_BIN, CUDA_VISIBLE_DEVICES (default 0), ANALYSIS_RUN_DIR
Data overrides: MMSU_JSONL, MMSUFINAL_DATASET_DIR

Defaults: all 5,000 mmsufinal samples, all LLM layers/heads, all-layer Attention and FFN deletion,
          one full-answer generation per sample, four probes, raw data and reports.

Plan only:  run.sh --dry-run
GPU smoke:  run.sh --sample-limit 4 --layers 0,1
Full run:   run.sh
EOF
    exit 0 ;;
  mmsufinal) shift ;;
  ""|--*) ;;
  *) printf 'Four-probe analysis only supports mmsufinal; unsupported benchmark: %s\n' "$1" >&2; exit 2 ;;
esac

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full base or merged Qwen2.5-Omni checkpoint}"
analysis_python="${PYTHON_BIN:-python}"
run_root="${ANALYSIS_RUN_DIR:-$project_root/runs/analysis/audio_four_probes/$(date +%Y%m%d_%H%M%S)}"
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$project_root"

exec "$analysis_python" -m scripts.analysis.audio_four_probes.launch \
  --config "$script_dir/qwen25_omni_mmsufinal_four_probes.yaml" \
  --checkpoint "$CHECKPOINT_PATH" \
  --gpus "${CUDA_VISIBLE_DEVICES:-0}" \
  --output-dir "$run_root/mmsufinal" \
  "$@"
