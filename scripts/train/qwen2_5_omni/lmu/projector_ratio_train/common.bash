# Shared launch settings for one projector method and one 3B/7B LoRA run.

projector_train_setting_label() {
  local method="$1" setting="$2"
  case "${method}" in
    segmentwise_topk|ltbm)
      case "${setting}" in
        10|25|50|75) printf 'keep%s\n' "${setting}"; return 0 ;;
      esac
      printf '%s requires a keep ratio of 10, 25, 50, or 75: %s\n' "${method}" "${setting}" >&2
      ;;
    sa_map|unisamp|uniavg)
      case "${setting}" in
        10|25|50) printf 'keep%s\n' "${setting}"; return 0 ;;
      esac
      printf '%s requires a keep ratio of 10, 25, or 50: %s\n' "${method}" "${setting}" >&2
      ;;
    unseg)
      if [[ "${setting}" == natural ]]; then
        printf 'natural\n'; return 0
      fi
      printf 'UnSeg requires the natural setting, without a fixed keep ratio: %s\n' "${setting}" >&2
      ;;
    fastslm_hta)
      if [[ "${setting}" == default ]]; then
        printf 'default\n'; return 0
      fi
      printf 'FastSLM HTA requires the default setting (50/750 token retention): %s\n' "${setting}" >&2
      ;;
    *) printf 'unknown projector method: %s\n' "${method}" >&2 ;;
  esac
  return 2
}

run_projector_ratio_train() {
  local method="$1"
  shift
  if (( $# < 2 )); then
    printf 'usage: run_%s.sh <3b|7b> <setting> [mllm-comp train options]\n' "${method}" >&2
    return 2
  fi

  local model_size="$1" setting="$2"
  shift 2
  local default_model model_offset setting_label method_offset ratio_offset
  case "${model_size}" in
    3b) default_model="${MODEL_PATH_3B:-Qwen/Qwen2.5-Omni-3B}"; model_offset=0 ;;
    7b) default_model="${MODEL_PATH_7B:-Qwen/Qwen2.5-Omni-7B}"; model_offset=1 ;;
    *) printf 'model size must be 3b or 7b: %s\n' "${model_size}" >&2; return 2 ;;
  esac
  setting_label="$(projector_train_setting_label "${method}" "${setting}")" || return $?
  case "${method}" in
    segmentwise_topk) method_offset=0 ;;
    ltbm) method_offset=8 ;;
    sa_map) method_offset=16 ;;
    unseg) method_offset=24 ;;
    unisamp) method_offset=32 ;;
    uniavg) method_offset=40 ;;
    fastslm_hta) method_offset=48 ;;
  esac
  case "${setting}" in
    10|natural|default) ratio_offset=0 ;;
    25) ratio_offset=2 ;;
    50) ratio_offset=4 ;;
    75) ratio_offset=6 ;;
  esac

  : "${DATASET_PATH:?set DATASET_PATH=/absolute/path/to/audio_train.jsonl}"
  local sweep_dir root_dir config_path run_tag run_root base_port
  sweep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd -- "${sweep_dir}/../../../../.." && pwd)"
  config_path="${sweep_dir}/configs/${method}/${setting_label}/${model_size}_lora.yaml"
  run_tag="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/train/${run_tag}/audio_projector_ratio}"
  base_port="${MASTER_PORT_BASE:-${MASTER_PORT:-29540}}"

  export QWEN25_OMNI_MODEL_PATH="${MODEL_PATH:-${default_model}}"
  export MLLM_COMP_TRAIN_DATASET="${DATASET_PATH}"
  export MLLM_COMP_TRAIN_RUN_DIR="${run_root}/${method}/${setting_label}/${model_size}_lora"
  export PYTHONPATH="${root_dir}${PYTHONPATH:+:${PYTHONPATH}}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
  export NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
  export NNODES="${NNODES:-1}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
  export MASTER_PORT="$((base_port + method_offset + ratio_offset + model_offset))"
  export WANDB_PROJECT="${WANDB_PROJECT:-qwen2_5_omni_lmu}"
  export WANDB_MODE="${WANDB_MODE:-offline}"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

  printf 'config=%s\noutput=%s\nport=%s\n' "${config_path}" "${MLLM_COMP_TRAIN_RUN_DIR}" "${MASTER_PORT}"
  cd "${root_dir}"
  exec "${PYTHON_BIN:-python}" -m mllm_comp.cli.main train --config "${config_path}" "$@"
}
