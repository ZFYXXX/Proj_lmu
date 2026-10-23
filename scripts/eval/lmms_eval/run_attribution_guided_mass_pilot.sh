#!/usr/bin/env bash
set -uo pipefail

PROJECT_ROOT=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
GPU_ID=${1:?usage: run_attribution_guided_mass_pilot.sh GPU_ID chartqa|textvqa}
DATASET=${2:?usage: run_attribution_guided_mass_pilot.sh GPU_ID chartqa|textvqa}
ROOT="$PROJECT_ROOT/outputs/eval/qwen25_vl/attribution_guided_mass_pilot"

case "$DATASET" in
  chartqa)
    CONFIGS=(
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_chartqa_full/baseline.yaml"
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_chartqa_full/feature_anchor_mean_k64.yaml"
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_chartqa_stage2/feature_anchor_mass_k64.yaml"
    )
    METHODS=(baseline feature_anchor_k64 feature_anchor_mass_k64)
    ;;
  textvqa)
    CONFIGS=(
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_textvqa_full_k64_20261018/baseline.yaml"
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_textvqa_full_k64_20261018/feature_anchor_mean_k64.yaml"
      "$PROJECT_ROOT/configs/eval/lmms_eval/qwen25_vl_chartqa_stage2/textvqa_feature_anchor_mass_k64.yaml"
    )
    METHODS=(baseline feature_anchor_k64 feature_anchor_mass_k64)
    ;;
  *) echo "Unknown dataset: $DATASET" >&2; exit 2 ;;
esac

cd "$PROJECT_ROOT"
source /home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/activate
export HF_HOME=/home/ZhouFengyu/.cache/huggingface
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU_ID"
mkdir -p "$ROOT/logs"

run_index=0
for method in "${METHODS[@]}"; do
  config="${CONFIGS[$run_index]}"
  run_index=$((run_index + 1))
  out="$ROOT/${DATASET}_${method}_n100"
  log="$ROOT/logs/${DATASET}_${method}_n100.log"
  if [[ -e "$out/result.json" ]]; then
    echo "SKIP $(date -Is) gpu=$GPU_ID dataset=$DATASET method=$method result_exists" | tee -a "$log"
    continue
  fi
  echo "START $(date -Is) gpu=$GPU_ID dataset=$DATASET method=$method limit=100" | tee "$log"
  if timeout 7200s python -m mllm_comp.cli.main eval \
    --config "$config" \
    --sample_limit 100 \
    --output_dir "$out" \
    --manifest "$out/manifest.json" >> "$log" 2>&1; then
    echo "FINISH $(date -Is) gpu=$GPU_ID dataset=$DATASET method=$method status=ok" | tee -a "$log"
  else
    status=$?
    echo "FINISH $(date -Is) gpu=$GPU_ID dataset=$DATASET method=$method status=$status" | tee -a "$log"
  fi
done
