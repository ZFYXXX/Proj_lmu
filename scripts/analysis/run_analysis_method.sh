#!/usr/bin/env bash
set -eu

# 无论从哪个目录调用脚本，都以仓库根目录解析配置和输出路径。
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
root_dir="$(cd -- "${script_dir}/../.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/analysis/run_analysis_method.sh METHOD [analyze options]

METHOD:
  representation_dynamics
  prediction_dynamics
  attention_flow
  residual_stream_decomposition
  layer_importance
  late_injection_early_exit
  hierarchical_taylor

Examples:
  bash scripts/analysis/run_analysis_method.sh representation_dynamics
  bash scripts/analysis/run_analysis_method.sh prediction_dynamics
  bash scripts/analysis/run_analysis_method.sh attention_flow
  bash scripts/analysis/run_analysis_method.sh residual_stream_decomposition
  bash scripts/analysis/run_analysis_method.sh layer_importance
  bash scripts/analysis/run_analysis_method.sh late_injection_early_exit
  bash scripts/analysis/run_analysis_method.sh hierarchical_taylor

Optional environment:
  ANALYSIS_BACKBONE        qwen25_vl (default) | qwen25_omni
  MODEL_PATH               checkpoint override
  QWEN25_VL_MODEL_PATH     checkpoint fallback
  QWEN25_OMNI_MODEL_PATH   Qwen Audio checkpoint fallback
  ANALYSIS_DEVICE          default: cuda:0
  ANALYSIS_OUTPUT_ROOT     default: outputs/analysis/methods
  ANALYSIS_RUN_STAMP       default: current timestamp
  CUDA_VISIBLE_DEVICES     default: 0
  PYTHON_BIN               default: python
EOF
}

# 第一个位置参数用于选择分析方法；case 由 esac 结束。
case "${1:-}" in
  -h|--help|help)
    usage
    exit 0
    ;;
  representation_dynamics|prediction_dynamics|attention_flow|residual_stream_decomposition|layer_importance|late_injection_early_exit|hierarchical_taylor)
    method="$1"
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

# 方法名已经写入 method，移除 $1；剩余参数原样转交给 analyze 命令。
shift

analysis_backbone="${ANALYSIS_BACKBONE:-qwen25_vl}"
case "${analysis_backbone}" in
  qwen25_vl)
    config="${root_dir}/examples/analysis/methods/qwen25_vl_${method}.yaml"
    checkpoint="${MODEL_PATH:-${QWEN25_VL_MODEL_PATH:-}}"
    ;;
  qwen25_omni)
    config="${root_dir}/examples/analysis/methods/qwen25_omni_audio_${method}.yaml"
    checkpoint="${MODEL_PATH:-${QWEN25_OMNI_MODEL_PATH:-}}"
    ;;
  *)
    echo "Unsupported ANALYSIS_BACKBONE: ${analysis_backbone}" >&2
    exit 2
    ;;
esac
run_stamp="${ANALYSIS_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
output_root="${ANALYSIS_OUTPUT_ROOT:-${root_dir}/outputs/analysis/methods}"
run_dir="${output_root}/${analysis_backbone}/${method}/${run_stamp}"

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
