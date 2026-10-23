#!/usr/bin/env bash
set -euo pipefail

project_root=/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main
python_bin=/home/ZhouFengyu/fastv-repro/mllm-comp-310/bin/python
model_path=/home/ZhouFengyu/fastv-repro/models/Qwen2.5-VL-3B-Instruct
run_root="$project_root/outputs/train/qwen25_vl/stage2_pilot_20261018"

mkdir -p "$run_root/mean_only" "$run_root/meta_residual"
cd "$project_root"

nohup env CUDA_VISIBLE_DEVICES=0 "$python_bin" \
  -m scripts.train.qwen25_vl.train_feature_anchor_projector \
  --model "$model_path" \
  --output-dir "$run_root/mean_only" \
  --variant mean_only \
  --samples-per-source 256 \
  --gradient-accumulation 8 \
  --max-optimizer-steps 64 \
  --save-steps 16 \
  --learning-rate 3e-4 \
  --seed 42 \
  --device cuda:0 \
  >"$run_root/mean_only/launcher.log" 2>&1 &
mean_pid=$!

nohup env CUDA_VISIBLE_DEVICES=1 "$python_bin" \
  -m scripts.train.qwen25_vl.train_feature_anchor_projector \
  --model "$model_path" \
  --output-dir "$run_root/meta_residual" \
  --variant meta_residual \
  --samples-per-source 256 \
  --gradient-accumulation 8 \
  --max-optimizer-steps 64 \
  --save-steps 16 \
  --learning-rate 3e-4 \
  --seed 42 \
  --device cuda:0 \
  >"$run_root/meta_residual/launcher.log" 2>&1 &
meta_pid=$!

printf '%s\n' "$mean_pid" >"$run_root/mean_only/launcher.pid"
printf '%s\n' "$meta_pid" >"$run_root/meta_residual/launcher.pid"
printf 'mean_only pid=%s\nmeta_residual pid=%s\n' "$mean_pid" "$meta_pid"
