#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../../.." && pwd)"

# Accept a full checkpoint or an already merged model, as used by the YAML.
model_path="${CHECKPOINT_PATH:-${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-}}}"
if [[ -z "${model_path}" ]]; then
  printf 'set CHECKPOINT_PATH, MODEL_PATH, or QWEN25_OMNI_MODEL_PATH to the model directory\n' >&2
  exit 2
fi

run_tag="${RUN_TAG:-${SLURM_JOB_ID:-manual}-$(date +%Y%m%d-%H%M%S)}"
eval_name="${EVAL_NAME:-audio_eval}"
output_dir="${OUTPUT_DIR:-${MLLM_COMP_EVAL_RUN_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/${eval_name}/mmarfinal}}"

export QWEN25_OMNI_MODEL_PATH="${model_path}"
export MLLM_COMP_EVAL_RUN_DIR="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

cd "${root_dir}"
mkdir -p "${output_dir}"

printf 'starting task=mmarfinal model=%s gpus=%s output=%s\n' \
  "${QWEN25_OMNI_MODEL_PATH}" "${CUDA_VISIBLE_DEVICES}" "${output_dir}"

# Reuse the existing audio config and select the final task through the CLI.
# The platform validates GPU/process settings and launches Accelerate itself.
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${root_dir}/configs/eval/lmms_eval/lmu/eval_aud.yaml" \
  --tasks mmarfinal \
  --num_processes "${EVAL_NUM_PROCESSES:-4}" \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  --main_process_port "${MMARFINAL_MAIN_PROCESS_PORT:-29638}" \
  "$@"
