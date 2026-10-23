#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
PYTHON_BIN=/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python
RUN_ROOT=${RUN_ROOT:-"$PROJECT_ROOT/outputs/eval/qwen25_vl/scienceqa_img_full_k64_20261017_v2"}
CONFIG_DIR=${CONFIG_DIR:-"$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_scienceqa_full"}
GPU_ID=${1:?usage: run_qwen25_scienceqa_k64.sh GPU_ID METHOD}
METHOD=${2:?usage: run_qwen25_scienceqa_k64.sh GPU_ID METHOD}
CONFIG="$CONFIG_DIR/$METHOD.yaml"
OUT="$RUN_ROOT/$METHOD"
LOG="$RUN_ROOT/logs/$METHOD.log"
TIME_LOG="$RUN_ROOT/logs/$METHOD.time.txt"
GPU_LOG="$RUN_ROOT/logs/$METHOD.gpu.csv"

if [[ ! -f "$CONFIG" ]]; then
  echo "Unknown method config: $CONFIG" >&2
  exit 2
fi
mkdir -p "$RUN_ROOT/logs"
if [[ -f "$OUT/result.json" ]] && grep -q '"status": "succeeded"' "$OUT/result.json"; then
  echo "Already succeeded: $METHOD"
  exit 0
fi

cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID"
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_CACHE="$RUN_ROOT/parquet-cache"
export HF_DATASETS_OFFLINE=1

/usr/bin/time -v -o "$TIME_LOG" \
  "$PYTHON_BIN" -m mllm_comp.cli.main eval --config "$CONFIG" \
  >"$LOG" 2>&1 &
RUN_PID=$!
while kill -0 "$RUN_PID" 2>/dev/null; do
  nvidia-smi --id="$GPU_ID" \
    --query-gpu=timestamp,memory.used,utilization.gpu \
    --format=csv,noheader,nounits >>"$GPU_LOG"
  sleep 1
done
wait "$RUN_PID"
