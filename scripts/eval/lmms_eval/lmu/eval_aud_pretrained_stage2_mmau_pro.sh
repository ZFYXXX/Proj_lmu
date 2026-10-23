#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"
run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}}"
run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/audio_pretrained}"

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-Omni-3B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

exec bash "${script_dir}/run_mmau_pro_stage.sh" \
  "${root_dir}" \
  "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud_mmau_pro.yaml" \
  "${run_root}/mmau_pro" \
  "${MMAU_PRO_MAIN_PROCESS_PORT:-29731}" \
  "$@"
