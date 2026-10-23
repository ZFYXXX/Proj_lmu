#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"

case "${1:-}" in
  -h|--help)
    cat <<'EOF'
Usage: bash scripts/analysis/audio_four_probes/run_hierarchical_taylor.sh [mmsufinal] [launch options]

Required: CHECKPOINT_PATH (full local Qwen2.5-Omni checkpoint)
Optional: PYTHON_BIN, TAYLOR_GPU (default: first CUDA_VISIBLE_DEVICES ID, or 0),
          TAYLOR_RUN_DIR (default: ANALYSIS_RUN_DIR/hierarchical_taylor)
Data overrides: MMSU_JSONL, MMSUFINAL_DATASET_DIR

Defaults: all labeled samples, all LLM layers, audio-target hierarchical Taylor,
          one GPU, layer_batch_size=1, native task scores and ranks.
Only mmsufinal runs, covering all 5,000 rows.
Each sample uses teacher-forced answer CE and backward.
Full-answer baseline scoring is also retained. Causal validation and per-sample
neuron-level raw dumps are disabled in the YAML; task-level neuron scores are saved.

Plan only:  run_hierarchical_taylor.sh --dry-run
GPU smoke:  run_hierarchical_taylor.sh --sample-limit 1 --layers 0,1
Full run:   run_hierarchical_taylor.sh
EOF
    exit 0 ;;
  mmsufinal) shift ;;
  ""|--*) ;;
  *) printf 'Taylor only supports mmsufinal; unsupported benchmark: %s\n' "$1" >&2; exit 2 ;;
esac

: "${CHECKPOINT_PATH:?Set CHECKPOINT_PATH to a full base or merged Qwen2.5-Omni checkpoint}"
analysis_python="${PYTHON_BIN:-python}"
visible_gpus="${CUDA_VISIBLE_DEVICES:-0}"
taylor_gpu="${TAYLOR_GPU:-${visible_gpus%%,*}}"
analysis_root="${ANALYSIS_RUN_DIR:-$project_root/runs/analysis/audio_four_probes/$(date +%Y%m%d_%H%M%S)}"
run_root="${TAYLOR_RUN_DIR:-$analysis_root/hierarchical_taylor}"
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
cd "$project_root"

exec "$analysis_python" -m scripts.analysis.audio_four_probes.launch \
  --config "$script_dir/qwen25_omni_mmsufinal_hierarchical_taylor.yaml" \
  --checkpoint "$CHECKPOINT_PATH" \
  --gpus "$taylor_gpu" \
  --output-dir "$run_root/mmsufinal" \
  "$@"
