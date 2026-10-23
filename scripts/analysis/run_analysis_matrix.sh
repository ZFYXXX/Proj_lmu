#!/usr/bin/env bash
set -eu

# 无论从哪个目录调用脚本，都以仓库根目录解析配置和输出路径。
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
root_dir="$(cd -- "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/analysis/run_analysis_matrix.sh BACKBONE [analyze options]

BACKBONE:
  llava_onevision | qwen25_vl | qwen25_omni | qwen3_vl

Examples:
  bash scripts/analysis/run_analysis_matrix.sh qwen25_vl
  bash scripts/analysis/run_analysis_matrix.sh qwen25_omni --probe attention_flow
  bash scripts/analysis/run_analysis_matrix.sh qwen3_vl --probe hierarchical_taylor

Optional environment:
  MODEL_PATH               checkpoint override shared by every backbone
  LLAVA_ONEVISION_MODEL_PATH
  QWEN25_VL_MODEL_PATH
  QWEN25_OMNI_MODEL_PATH     Qwen Audio checkpoint fallback
  QWEN3_VL_MODEL_PATH       backbone-specific checkpoint fallback
  ANALYSIS_DEVICE           default: cuda:0
  ANALYSIS_OUTPUT_ROOT      default: outputs/analysis
  ANALYSIS_RUN_STAMP        default: current timestamp
  CUDA_VISIBLE_DEVICES      default: 0
  PYTHON_BIN                default: python
EOF
}

# 第一个位置参数用于选择基座；case 由 esac 结束。
case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  llava_onevision)
    backbone="$1"
    config="${root_dir}/examples/analysis/llava_onevision_analysis_matrix.yaml"
    checkpoint="${MODEL_PATH:-${LLAVA_ONEVISION_MODEL_PATH:-}}"
    ;;
  qwen25_vl)
    backbone="$1"
    config="${root_dir}/examples/analysis/qwen25_vl_analysis_matrix.yaml"
    checkpoint="${MODEL_PATH:-${QWEN25_VL_MODEL_PATH:-}}"
    ;;
  qwen25_omni)
    backbone="$1"
    config="${root_dir}/examples/analysis/qwen25_omni_audio_analysis_matrix.yaml"
    checkpoint="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-}}"
    ;;
  qwen3_vl)
    backbone="$1"
    config="${root_dir}/examples/analysis/qwen3_vl_analysis_matrix.yaml"
    checkpoint="${MODEL_PATH:-${QWEN3_VL_MODEL_PATH:-}}"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# 基座名已经写入 backbone，移除 $1；剩余参数原样转交给 analyze 命令。
shift

run_stamp="${ANALYSIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
output_root="${ANALYSIS_OUTPUT_ROOT:-${root_dir}/outputs/analysis}"
run_dir="${output_root}/${backbone}/${run_stamp}"

# 环境变量只提供默认值，调用方显式设置的值优先。
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# set -- 重新组装位置参数：先放脚本生成的公共参数，再追加用户参数。
if [ -n "${checkpoint}" ]; then
  set -- \
    --config "${config}" \
    --device "${ANALYSIS_DEVICE:-cuda:0}" \
    --output_dir "${run_dir}" \
    --manifest "${run_dir}/manifest.json" \
    --checkpoint "${checkpoint}" \
    "$@"
else
  set -- \
    --config "${config}" \
    --device "${ANALYSIS_DEVICE:-cuda:0}" \
    --output_dir "${run_dir}" \
    --manifest "${run_dir}/manifest.json" \
    "$@"
fi

cd "${root_dir}"
# exec 让 Python 进程替换当前 Shell，退出码和终止信号可直接传递给调用方。
exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main analyze "$@"
