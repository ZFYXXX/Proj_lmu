#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
config="${root_dir}/configs/eval/lmms_eval/qwen25_omni_mmau_pro_1gpu.yaml"
output_dir="${OUTPUT_DIR:-${root_dir}/runs/qwen25_omni_mmau_pro_1gpu}"

if [[ ! -f "${config}" ]]; then
  printf 'Missing MMAU-Pro eval config: %s\n' "${config}" >&2
  exit 2
fi

export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-}}"
: "${QWEN25_OMNI_MODEL_PATH:?Set MODEL_PATH or QWEN25_OMNI_MODEL_PATH}"
: "${MMAU_PRO_QWEN_JUDGE_MODEL:?Set MMAU_PRO_QWEN_JUDGE_MODEL}"
: "${MMAU_PRO_NVEMBED_MODEL:?Set MMAU_PRO_NVEMBED_MODEL}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY="${MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY:-true}"
export MMAU_PRO_EVALUATOR_DEVICE="${MMAU_PRO_EVALUATOR_DEVICE:-auto}"
export MMAU_PRO_EVALUATOR_DTYPE="${MMAU_PRO_EVALUATOR_DTYPE:-auto}"
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS="${MMAU_PRO_JUDGE_MAX_NEW_TOKENS:-512}"
export MMAU_PRO_JUDGE_TEMPERATURE="${MMAU_PRO_JUDGE_TEMPERATURE:-0.1}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

mkdir -p "${output_dir}"
cd "${root_dir}"

exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${config}" \
  --compression_mode none \
  --target baseline \
  --num_processes 1 \
  --device cuda \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  "$@"
