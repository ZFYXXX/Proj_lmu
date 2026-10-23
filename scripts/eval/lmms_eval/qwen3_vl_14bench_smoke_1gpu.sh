#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
run_stamp="${EVAL_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
output_root="${EVAL_OUTPUT_ROOT:-${root_dir}/outputs/qwen3_vl_14bench/smoke/${run_stamp}}"

tasks=(
  gqa
  vqav2_val
  ok_vqa_val2014
  textvqa_val
  docvqa_val
  ocrbench
  scienceqa_img
  ai2d
  mmstar
  refcoco_bbox_rec_testA
  refcoco_bbox_rec_testB
  refcoco+_bbox_rec_testA
  refcoco+_bbox_rec_testB
  refcocog_bbox_rec_test
)

for task in "${tasks[@]}"; do
  if [[ -f "${output_root}/${task}/manifest.json" ]]; then
    echo "[$(date -Is)] skip completed smoke task: ${task}"
    continue
  fi
  echo "[$(date -Is)] Qwen3-VL smoke task: ${task}"
  EVAL_LIMIT="${EVAL_LIMIT:-1}" \
  EVAL_OUTPUT_DIR="${output_root}/${task}" \
  bash "${script_dir}/qwen3_vl_14bench_1gpu.sh" --tasks "${task}" "$@"
done

echo "[$(date -Is)] all ${#tasks[@]} Qwen3-VL benchmark smoke runs passed"
echo "Results: ${output_root}"
