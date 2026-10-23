# Shared launcher for the three encoder compression sweeps. Source from an entry SH.

run_encoder_ratio_sweep() {
  local method="$1"
  shift
  local sweep_dir root_dir workspace_root run_tag run_root
  sweep_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  root_dir="$(cd -- "${sweep_dir}/../../../../.." && pwd)"
  workspace_root="$(cd -- "${root_dir}/../.." && pwd)"
  run_tag="${RUN_TAG:-${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"
  run_root="${OUTPUT_DIR:-${root_dir}/runs/lmu/eval/${run_tag}/${EVAL_NAME:-audio_encoder_sweep_final}}/${method}"

  local port_base
  case "${method}" in
    encoder_topk) port_base=29634 ;;
    atome) port_base=29644 ;;
    progressive_fixed) port_base=29654 ;;
    *) printf 'unknown encoder method: %s\n' "${method}" >&2; return 2 ;;
  esac

  local plan_only=false cli_num_processes="" expect_num_processes=false arg
  for arg in "$@"; do
    if [[ "${expect_num_processes}" == true ]]; then
      cli_num_processes="${arg}"
      expect_num_processes=false
      continue
    fi
    case "${arg}" in
      --plan_only|--dry_run) plan_only=true ;;
      --num_processes) expect_num_processes=true ;;
      --num_processes=*) cli_num_processes="${arg#--num_processes=}" ;;
      --config|--config=*|--tasks|--tasks=*|--checkpoint|--checkpoint=*|--target|--target=*|--compression_mode|--compression_mode=*|--output_dir|--output_dir=*|--manifest|--manifest=*)
        printf '%s is owned by this sweep; use MODEL_PATH/ADAPTER_PATH, BENCHMARKS and OUTPUT_DIR instead\n' "${arg}" >&2
        return 2
        ;;
    esac
  done
  if [[ "${expect_num_processes}" == true ]]; then
    printf '%s requires a value\n' '--num_processes' >&2
    return 2
  fi

  local gpu_list="${CUDA_VISIBLE_DEVICES-0,1,2,3}" num_processes
  gpu_list="${gpu_list//[[:space:]]/}"
  if [[ -z "${gpu_list}" || "${gpu_list}" == ,* || "${gpu_list}" == *, || "${gpu_list}" == *,,* ]]; then
    printf 'invalid CUDA_VISIBLE_DEVICES: %s\n' "${gpu_list}" >&2
    return 2
  fi
  local -a gpu_ids benchmarks
  IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
  num_processes="${EVAL_NUM_PROCESSES:-${cli_num_processes:-${#gpu_ids[@]}}}"
  if [[ ! "${num_processes}" =~ ^[1-9][0-9]*$ ]] || (( num_processes > ${#gpu_ids[@]} )); then
    printf 'num_processes must be a positive integer <= %s visible GPUs, got: %s\n' "${#gpu_ids[@]}" "${num_processes}" >&2
    return 2
  fi

  read -r -a benchmarks <<< "${BENCHMARKS:-audiomarathon2 mmaunew3 mmsufinal mmarfinal mmau_pro}"
  local task has_audiomarathon=false has_mmau_pro=false
  for task in "${benchmarks[@]}"; do
    case "${task}" in
      audiomarathon2) has_audiomarathon=true ;;
      mmaunew3|mmsufinal|mmarfinal) ;;
      mmau_pro) has_mmau_pro=true ;;
      *) printf 'unknown benchmark: %s\n' "${task}" >&2; return 2 ;;
    esac
  done
  if [[ "${has_audiomarathon}" == true ]]; then
    export AUDIOMARATHON_TASK_DIR="${AUDIOMARATHON_TASK_DIR:-${workspace_root}/audiomarathon2}"
    export AUDIOMARATHON_ROOT="${AUDIOMARATHON_ROOT:-${DATASET_ROOT:-${workspace_root}/datasets}/AudioMarathon}"
    if [[ "${plan_only}" == false && ! -f "${AUDIOMARATHON_TASK_DIR}/audiomarathon.yaml" ]]; then
      printf 'AudioMarathon2 task package not found: %s; set AUDIOMARATHON_TASK_DIR\n' "${AUDIOMARATHON_TASK_DIR}" >&2
      return 2
    fi
  fi
  if [[ "${has_mmau_pro}" == true && "${plan_only}" == false ]]; then
    : "${MMAU_PRO_QWEN_JUDGE_MODEL:?set MMAU_PRO_QWEN_JUDGE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct}"
    : "${MMAU_PRO_NVEMBED_MODEL:?set MMAU_PRO_NVEMBED_MODEL=/absolute/path/to/NV-Embed-v2}"
  fi

  local adapter_path="${ADAPTER_PATH:-}"
  local direct_model_path="${CHECKPOINT_PATH:-${MODEL_PATH:-}}"
  local resolved_model_path
  if [[ -n "${adapter_path}" && -n "${direct_model_path}" ]]; then
    printf 'set ADAPTER_PATH or CHECKPOINT_PATH/MODEL_PATH, not both\n' >&2
    return 2
  fi
  if [[ -z "${adapter_path}" && -z "${direct_model_path}" ]]; then
    printf 'set ADAPTER_PATH, CHECKPOINT_PATH, or MODEL_PATH\n' >&2
    return 2
  fi
  if [[ -n "${adapter_path}" ]]; then
    if [[ "${plan_only}" == false && ! -d "${adapter_path}" ]]; then
      printf 'LoRA adapter does not exist: %s\n' "${adapter_path}" >&2
      return 2
    fi
    resolved_model_path="${MERGED_MODEL_PATH:-${adapter_path}-merged}"
    if [[ "${plan_only}" == false && ! -f "${resolved_model_path}/config.json" ]]; then
      "${SWIFT_BIN:-swift}" export --adapters "${adapter_path}" \
        --merge_lora true --output_dir "${resolved_model_path}"
    fi
  else
    resolved_model_path="${direct_model_path}"
  fi

  export QWEN25_OMNI_MODEL_PATH="${resolved_model_path}"
  export CUDA_VISIBLE_DEVICES="${gpu_list}"
  export MMAU_PRO_CUDA_VISIBLE_DEVICES="${MMAU_PRO_CUDA_VISIBLE_DEVICES:-${gpu_list}}"
  export MMAU_PRO_NUM_GPUS="${MMAU_PRO_NUM_GPUS:-${num_processes}}"
  export PYTHONUNBUFFERED=1
  export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
  export PYTHONPATH="${root_dir}:${root_dir}/third_party/frameworks/lmms-eval${PYTHONPATH:+:${PYTHONPATH}}"

  cd "${root_dir}"
  local keep config_path output_dir port task_processes task_gpus
  # Outer loop is intentional: finish all five benchmarks at one ratio first.
  for keep in 10 25 75 50; do
    for task in "${benchmarks[@]}"; do
      config_path="${sweep_dir}/configs/${method}/keep${keep}/${task}.yaml"
      output_dir="${run_root}/keep${keep}/${task}"
      task_processes="${num_processes}"
      task_gpus="${gpu_list}"
      case "${task}" in
        audiomarathon2) port="${AUDIOMARATHON_MAIN_PROCESS_PORT:-${port_base}}" ;;
        mmaunew3) port="${MMAUNEW3_MAIN_PROCESS_PORT:-$((port_base + 1))}" ;;
        mmsufinal) port="${MMSUFINAL_MAIN_PROCESS_PORT:-$((port_base + 2))}" ;;
        mmarfinal) port="${MMARFINAL_MAIN_PROCESS_PORT:-$((port_base + 3))}" ;;
        mmau_pro)
          port="${MMAU_PRO_MAIN_PROCESS_PORT:-$((port_base + 100))}"
          task_processes="${MMAU_PRO_NUM_GPUS}"
          task_gpus="${MMAU_PRO_CUDA_VISIBLE_DEVICES}"
          ;;
      esac
      printf 'starting method=%s target_keep=%s%% task=%s model=%s gpus=%s processes=%s output=%s\n' \
        "${method}" "${keep}" "${task}" "${QWEN25_OMNI_MODEL_PATH}" \
        "${task_gpus}" "${task_processes}" "${output_dir}"

      if [[ "${task}" == mmau_pro && "${plan_only}" == false ]]; then
        bash "${sweep_dir}/../run_mmau_pro_stage.sh" \
          "${root_dir}" "${config_path}" "${output_dir}" "${port}" \
          "$@" --num_processes "${task_processes}"
      else
        # Plan-only validates the native generation plan without loading judges
        # or creating the temporary MMAU-Pro evaluator overlay.
        if [[ "${plan_only}" == false ]]; then
          mkdir -p "${output_dir}"
        fi
        MLLM_COMP_EVAL_RUN_DIR="${output_dir}" CUDA_VISIBLE_DEVICES="${task_gpus}" \
          "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
          --config "${config_path}" "$@" \
          --num_processes "${task_processes}" \
          --output_dir "${output_dir}" \
          --manifest "${output_dir}/manifest.json" \
          --main_process_port "${port}"
      fi
    done
  done
}
