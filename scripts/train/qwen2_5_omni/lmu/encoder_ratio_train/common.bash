# Shared launcher for one encoder method, model size, and target keep ratio.

run_encoder_ratio_train() {
  local method="$1"
  shift

  if (( $# < 2 )); then
    printf 'usage: run_%s.sh <3b|7b> <10|25|50|75> [mllm-comp train options]\n' "${method}" >&2
    return 2
  fi

  local model_size="$1" keep="$2"
  shift 2
  case "${model_size}" in
    3b) local default_model="Qwen/Qwen2.5-Omni-3B" model_offset=0 ;;
    7b) local default_model="Qwen/Qwen2.5-Omni-7B" model_offset=1 ;;
    *) printf 'model size must be 3b or 7b: %s\n' "${model_size}" >&2; return 2 ;;
  esac

  local ratio_offset
  case "${keep}" in
    10) ratio_offset=0 ;;
    25) ratio_offset=2 ;;
    50) ratio_offset=4 ;;
    75) ratio_offset=6 ;;
    *) printf 'target keep ratio must be 10, 25, 50, or 75: %s\n' "${keep}" >&2; return 2 ;;
  esac

  local port_base
  case "${method}" in
    encoder_topk) port_base=29670 ;;
    atome) port_base=29690 ;;
    progressive_fixed) port_base=29710 ;;
    *) printf 'unknown encoder method: %s\n' "${method}" >&2; return 2 ;;
  esac

  : "${DATASET_PATH:?set DATASET_PATH=/absolute/path/to/audio_train.jsonl}"

  local sweep_dir root_dir run_tag config_path run_root
  sweep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd -- "${sweep_dir}/../../../../.." && pwd)"
  config_path="${sweep_dir}/configs/${method}/keep${keep}/${model_size}_lora.yaml"
  run_tag="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/train/${run_tag}/audio_encoder_ratio}"

  export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${default_model}}"
  export MLLM_COMP_TRAIN_DATASET="${DATASET_PATH}"
  export MLLM_COMP_TRAIN_RUN_DIR="${run_root}/${method}/keep${keep}/${model_size}_lora"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  export NNODES="${NNODES:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="${MASTER_PORT:-$((port_base + ratio_offset + model_offset))}"
  export WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5_omni_lmu}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  cd "${root_dir}"
  exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main train \
    --config "${config_path}" "$@"
}
