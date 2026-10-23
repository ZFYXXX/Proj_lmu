#!/usr/bin/env bash
set -u

project_root=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
python_bin=/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python
config_root="$project_root/configs/eval/lmms_eval/qwen25_vl_textvqa_full_k64_20261018"
output_root="$project_root/outputs/eval/qwen25_vl/textvqa_val_full_k64_20261018_v3"
lane=${1:?usage: run_qwen25_vl_textvqa_full_20261018.sh gpu0|gpu1}

case "$lane" in
  gpu0) gpu_id=0; methods=(baseline grid_mean_k64 bifocal) ;;
  gpu1) gpu_id=1; methods=(feature_anchor_mean_k64 prumerge_plus_k64) ;;
  *) echo "Unknown lane: $lane" >&2; exit 2 ;;
esac

cd "$project_root"
mkdir -p "$output_root/logs"
for method in "${methods[@]}"; do
  run_dir="$output_root/$method"
  log_path="$output_root/logs/$method.log"
  echo "START $(date -Is) lane=$lane method=$method"
  if timeout 43200s env CUDA_VISIBLE_DEVICES="$gpu_id" "$python_bin" \
    -m mllm_comp.cli.main eval \
    --config "$config_root/$method.yaml" \
    --device cuda:0 \
    --output_dir "$run_dir" \
    --manifest "$run_dir/manifest.json" \
    >"$log_path" 2>&1; then
    echo "FINISH $(date -Is) lane=$lane method=$method status=succeeded"
  else
    status=$?
    echo "FINISH $(date -Is) lane=$lane method=$method status=$status"
  fi
done
