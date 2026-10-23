#!/usr/bin/env bash
# Run the selected MMSU-final temporal analyses sequentially on exactly two GPUs.
# Provenance runs native LTBM once; similarity separately observes the baseline.
set -uo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="${PROJECT_ROOT:-$(cd -- "$script_dir/../../.." && pwd)}"

export PROJECT_ROOT="$project_root"
export DATASET_ROOT="${DATASET_ROOT:-/home/hk-project-p0022189/tum_yvc3016/haowu/workspace_compression/mllm_dataset/lmms_lab_dataset}"
export PYTHON_BIN="${PYTHON_BIN:-/home/hk-project-p0022189/tum_yvc3016/miniconda3/envs/lbw-comp-platform/bin/python3.10}"
export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$PROJECT_ROOT/runs/lmu/no_compression/lmu-7b-lora-all-20260906-123850/audio_7b_lora/v0-20260906-125716/checkpoint-4474-merged}"
export MMSUFINAL_DATASET_DIR="${MMSUFINAL_DATASET_DIR:-$DATASET_ROOT/mmsu}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export NUM_PROCESSES=2
export MAIN_PROCESS_PORT="${MAIN_PROCESS_PORT:-29649}"

export PATH="${PYTHON_BIN%/*}:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/frameworks/lmms-eval${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#/software/all/devel/cuda/12.4/lib64:}"
fi

IFS=',' read -r -a gpu_ids <<< "$CUDA_VISIBLE_DEVICES"
if [[ "${#gpu_ids[@]}" -ne 2 ]]; then
  echo "ERROR: CUDA_VISIBLE_DEVICES 必须恰好包含两张卡，例如 0,1；当前为 $CUDA_VISIBLE_DEVICES" >&2
  exit 2
fi

run_root="${ANALYSIS_RUN_DIR:-$PROJECT_ROOT/runs/analysis/audio_temporal/core_2gpu_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$run_root"
status_file="$run_root/status.txt"
: > "$status_file"

if [[ ! -f "$CHECKPOINT_PATH/config.json" ]]; then
  echo "ERROR: checkpoint 不完整，未找到 $CHECKPOINT_PATH/config.json" >&2
  exit 2
fi
if [[ ! -d "$MMSUFINAL_DATASET_DIR/data" ]]; then
  echo "ERROR: MMSU-final data 目录不存在：$MMSUFINAL_DATASET_DIR/data" >&2
  exit 2
fi
if [[ ! -f "$PROJECT_ROOT/third_party/frameworks/lmms-eval/lmms_eval/tasks/mmsufinal/mmsufinal.yaml" ]]; then
  echo "ERROR: 找不到 mmsufinal.yaml" >&2
  exit 2
fi

echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "MMSUFINAL_DATASET_DIR=$MMSUFINAL_DATASET_DIR"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "NUM_PROCESSES=$NUM_PROCESSES"
echo "RUN_ROOT=$run_root"
echo "计划：5 个任务，共 21 个完整 MMSU 等价遍历（105,000 次生成）"

failed=0
experiment_index=0

run_one() {
  local experiment="$1"
  shift

  local output_dir="$run_root/$experiment"
  local log_file="$run_root/${experiment}.log"
  local port=$((MAIN_PROCESS_PORT + experiment_index))
  experiment_index=$((experiment_index + 1))

  echo
  echo "============================================================"
  echo "开始：$experiment"
  echo "输出：$output_dir"
  echo "日志：$log_file"
  echo "端口：$port"
  echo "============================================================"

  bash "$script_dir/run_experiment.sh" \
    "$experiment" \
    "$output_dir" \
    --num-processes "$NUM_PROCESSES" \
    --main-process-port "$port" \
    --example-doc-ids 0,1,2 \
    "$@" \
    2>&1 | tee "$log_file"

  local status=${PIPESTATUS[0]}
  if [[ "$status" -eq 0 ]]; then
    echo "$experiment SUCCESS" | tee -a "$status_file"
  else
    echo "$experiment FAILED exit=$status" | tee -a "$status_file"
    failed=1
  fi
}

# 1 MMSU pass: native LTBM and actual output-token provenance.
run_one provenance \
  --ltbm-keep-ratio 0.5 \
  --ltbm-window-size 8

# 1 MMSU pass: baseline representation capture and multi-scale lag similarity.
run_one similarity \
  --lags-ms 40,80,160,320,640,1280,2560 \
  --region-lag-ms 80

# 5 MMSU-equivalent passes: baseline + four temporal-region substitutions.
run_one regions \
  --region-lag-ms 80 \
  --region-types silent,stable,transient,boundary \
  --replace-ratios 0.1

# 9 MMSU-equivalent passes: baseline + 4 block sizes x 2 predictors.
run_one predictability \
  --block-ms 40,80,160,320 \
  --predictors copy_left,linear \
  --replace-ratios 0.1

# 5 MMSU-equivalent passes: baseline + four length-preserving substitutions.
run_one substitution \
  --replace-ratios 0.1 \
  --window-ms 320 \
  --far-ms 2000

echo
echo "============================================================"
echo "双卡时间分析执行结束"
echo "结果目录：$run_root"
cat "$status_file"
echo "============================================================"

if [[ "$failed" -ne 0 ]]; then
  exit 1
fi
