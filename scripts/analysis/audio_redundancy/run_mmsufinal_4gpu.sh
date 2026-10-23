#!/usr/bin/env bash
# All five experiments share baseline forwards and matched uniform controls.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "$script_dir/../../.." && pwd)"
python_command="${PYTHON_BIN:-python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export PYTHONFAULTHANDLER=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="$project_root:$project_root/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"
IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#gpu_ids[@]}" -ne 4 ]]; then
  echo "CUDA_VISIBLE_DEVICES 必须恰好指定四张不同 GPU，例如 0,1,2,3" >&2
  exit 2
fi
for ((i=0; i<4; i++)); do
  if [[ -z "${gpu_ids[$i]// /}" ]]; then
    echo "GPU ID 不能为空" >&2
    exit 2
  fi
  for ((j=0; j<i; j++)); do
    if [[ "${gpu_ids[$i]}" == "${gpu_ids[$j]}" ]]; then
      echo "GPU ID 不可重复" >&2
      exit 2
    fi
  done
done
if [[ -z "${CHECKPOINT_PATH:-}" ]]; then
  echo "请设置 CHECKPOINT_PATH 为完整或 merged Qwen2.5-Omni checkpoint" >&2
  exit 2
fi
# Keep the four-GPU contract; use the Python launcher for a one-GPU smoke run.
for argument in "$@"; do
  case "$argument" in
    --num-processes|--num-processes=*) echo "四卡脚本固定 --num-processes 4" >&2; exit 2 ;;
  esac
done
run_dir="${ANALYSIS_RUN_DIR:-$project_root/runs/analysis/audio_redundancy/all_4gpu_$(date +%Y%m%d_%H%M%S)}"
cd "$project_root"
exec "$python_command" -m scripts.analysis.audio_redundancy.launch \
  --experiments all --checkpoint "$CHECKPOINT_PATH" --output-dir "$run_dir" \
  --num-processes 4 --main-process-port "${MAIN_PROCESS_PORT:-29679}" "$@"
