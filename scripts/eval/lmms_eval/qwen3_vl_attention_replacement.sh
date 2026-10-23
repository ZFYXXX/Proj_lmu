#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/../../.." && pwd)"
strategy="${QWEN3_VL_REPLACEMENT_STRATEGY:-${QWEN3_VL_ATTENTION_REPLACEMENT:-${1:-attn_linear_attention}}}"
if [[ "${strategy}" == "attn_linear_attention" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_linear_attention.yaml"
elif [[ "${strategy}" == "attn_cnn" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_cnn_attention.yaml"
elif [[ "${strategy}" == "ffn_low_rank" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_ffn_low_rank.yaml"
elif [[ "${strategy}" == "attn_identity" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_identity_attention.yaml"
elif [[ "${strategy}" == "attn_lolcats_linear_window" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_lolcats_linear_window.yaml"
elif [[ "${strategy}" == "attn_nha" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_nha.yaml"
elif [[ "${strategy}" == "attn_sparse_window" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_sparse_window_attention.yaml"
elif [[ "${strategy}" == "ffn_identity" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_ffn_identity.yaml"
elif [[ "${strategy}" == "ffn_two_layer_mlp" ]]; then
  config="${root_dir}/configs/eval/lmms_eval/qwen3_vl_ffn_two_layer_mlp.yaml"
else
  echo "Unknown Qwen3-VL replacement strategy: ${strategy}" >&2
  exit 2
fi
if [[ $# -gt 0 ]]; then
  shift
fi

export QWEN3_VL_MODEL_PATH="${MODEL_PATH:-${QWEN3_VL_MODEL_PATH:-/home/WuHao/checkpoints/ckpts-vlm/Qwen3-VL-8B-Instruct}}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1

cd "${root_dir}"
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval --config "${config}" "$@"
