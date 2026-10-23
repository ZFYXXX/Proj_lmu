#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/../.." && pwd)"
python_bin="${PYTHON_BIN:-python}"
read -r -a pytest_flags <<< "${VALIDATION_PYTEST_FLAGS:--q}"

usage() {
  cat <<'EOF'
Usage: bash scripts/diagnostics/run_qwen25_omni_eight_method_validation.sh COMMAND TARGET

  oracle METHOD       Independent algorithm and native CPU checks
  gpu TARGET          Real-checkpoint inference (llm also checks training/reload)
  backward METHOD     Real audio gradients (audio_pyramid/spade use gpu llm)
  check TARGET|all    Run the applicable oracle, gpu and backward checks
  plan METHOD|all     Native Swift launch plan
  swift METHOD        Native Swift one-step training
  quality METHOD|baseline|all
  profile METHOD|baseline|all

METHOD: segmentwise | atome | ltbm | sa_map | encoder_topk | progressive |
        audio_pyramid | spade
TARGET: segmentwise | atome | extensions | llm

Every command runs independently; no saved gates or mandatory full regression.
Required: VALIDATION_ROOT (logs and reports)
GPU/Swift/eval: QWEN25_OMNI_MODEL_PATH and one CUDA_VISIBLE_DEVICES device
Plan/Swift: MLLM_COMP_TRAIN_DATASET
Optional: PYTHON_BIN, VALIDATION_PYTEST_FLAGS (-q), MLLM_COMP_TRAIN_RUN_DIR,
          SEGMENTWISE_KEEP_RATIO (0.25), SEGMENTWISE_RUN_TAG (r25),
          VALIDATION_MAX_NEW_TOKENS (16, shared by quality/profile)

Use a new VALIDATION_ROOT when code, model, data or generation settings change.
Logs and actual reports are evidence; successful commands do not certify quality
or full trainer resume. Run the matching baseline before comparing experiments.
EOF
}

fail() {
  printf 'FAIL | %s\n' "$*" >&2
  exit 1
}

command_ok() {
  printf 'COMMAND_OK | %s\n' "$*"
}

require_variable() {
  local name="$1"
  [[ -n "${!name:-}" ]] || fail "environment variable is required: ${name}"
}

prepare_validation_root() {
  require_variable VALIDATION_ROOT
  mkdir -p "${VALIDATION_ROOT}"
  VALIDATION_ROOT="$(cd -- "${VALIDATION_ROOT}" && pwd)"
  export VALIDATION_ROOT
  cd "${repo_root}"
}

require_model() {
  require_variable QWEN25_OMNI_MODEL_PATH
  [[ -d "${QWEN25_OMNI_MODEL_PATH}" ]] \
    || fail "QWEN25_OMNI_MODEL_PATH is not a directory: ${QWEN25_OMNI_MODEL_PATH}"
  QWEN25_OMNI_MODEL_PATH="$(cd -- "${QWEN25_OMNI_MODEL_PATH}" && pwd)"
  export QWEN25_OMNI_MODEL_PATH
}

require_gpu() {
  require_model
  require_variable CUDA_VISIBLE_DEVICES
  "${python_bin}" -c \
    "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; assert torch.cuda.device_count() == 1, f'expected one visible GPU, got {torch.cuda.device_count()}'; print(torch.cuda.get_device_name(0))"
}

require_training_data() {
  require_variable MLLM_COMP_TRAIN_DATASET
  [[ -f "${MLLM_COMP_TRAIN_DATASET}" ]] \
    || fail "MLLM_COMP_TRAIN_DATASET is not a file: ${MLLM_COMP_TRAIN_DATASET}"
  local dataset_dir dataset_name
  dataset_dir="$(cd -- "$(dirname -- "${MLLM_COMP_TRAIN_DATASET}")" && pwd)"
  dataset_name="$(basename -- "${MLLM_COMP_TRAIN_DATASET}")"
  MLLM_COMP_TRAIN_DATASET="${dataset_dir}/${dataset_name}"
  export MLLM_COMP_TRAIN_DATASET
  export MLLM_COMP_TRAIN_RUN_DIR="${MLLM_COMP_TRAIN_RUN_DIR:-${VALIDATION_ROOT}/05_swift}"
  export WANDB_MODE="${WANDB_MODE:-disabled}"
}

run_logged() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname -- "${log_path}")"
  printf 'RUN  | %s\n' "$*"
  "$@" 2>&1 | tee "${log_path}"
  command_ok "log=${log_path}"
}

run_pytest() {
  local log_path="$1"
  shift
  mkdir -p "$(dirname -- "${log_path}")"
  printf 'RUN  | pytest %s\n' "$*"
  "${python_bin}" -m pytest "${pytest_flags[@]}" "$@" 2>&1 | tee "${log_path}"
  command_ok "log=${log_path}"
}

validate_json() {
  local path="$1"
  [[ -f "${path}" ]] || fail "expected JSON report was not created: ${path}"
  JSON_PATH="${path}" "${python_bin}" -c \
    "import json, os, pathlib; p=pathlib.Path(os.environ['JSON_PATH']); json.loads(p.read_text(encoding='utf-8')); print(p)"
  command_ok "valid JSON=${path}"
}

require_new_report() {
  [[ ! -e "$1" ]] || fail "Report already exists: $1; choose a new VALIDATION_ROOT"
}

validate_eval_manifest() {
  local path="$1"
  MANIFEST_PATH="${path}" "${python_bin}" -c \
    "import json, math, os, pathlib; p=pathlib.Path(os.environ['MANIFEST_PATH']); d=json.loads(p.read_text(encoding='utf-8')); metrics=[*d.get('baseline_metrics', []), *d.get('compressed_metrics', [])]; assert metrics, 'evaluation manifest has no metrics'; values=[v for x in metrics for v in (x.get('metrics') or {}).values()]; assert values and all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values), 'evaluation metrics must be finite numeric values'; print(f'metric_records={len(metrics)} numeric_values={len(values)}')"
  command_ok "non-empty evaluation metrics=${path}"
}

validate_eval_protocol() {
  local output_dir="$1"
  local expected_mode="$2"
  local max_new_tokens="$3"
  PLUGIN_SPEC_PATH="${output_dir}/mllm_comp_lmms_plugin.json" \
  EXPECTED_RUNTIME_MODE="${expected_mode}" \
  EXPECTED_MAX_NEW_TOKENS="${max_new_tokens}" \
    "${python_bin}" -c \
    "import json, os, pathlib; p=pathlib.Path(os.environ['PLUGIN_SPEC_PATH']); d=json.loads(p.read_text(encoding='utf-8')); g=d.get('generation', {}); assert d.get('runtime_mode') == os.environ['EXPECTED_RUNTIME_MODE'], (d.get('runtime_mode'), os.environ['EXPECTED_RUNTIME_MODE']); assert g.get('max_new_tokens') == int(os.environ['EXPECTED_MAX_NEW_TOKENS']), g; assert g.get('use_cache') is True, g; print(f\"runtime_mode={d['runtime_mode']} generation={g}\")"
  command_ok "shared generation protocol=${output_dir}"
}

validate_profile_summary() {
  local path="$1"
  local target="$2"
  SUMMARY_PATH="${path}" PROFILE_TARGET="${target}" "${python_bin}" -c \
    "import functools, json, os, pathlib; p=pathlib.Path(os.environ['SUMMARY_PATH']); d=json.loads(p.read_text(encoding='utf-8')); get=lambda path: functools.reduce(lambda x,k: x[k], path.split('.'), d); numeric=lambda v: isinstance(v,(int,float)) and not isinstance(v,bool); assert d.get('runtime_modes') == ['profile'], d.get('runtime_modes'); traces=int(d.get('num_trace_records', 0)); timings=int(d.get('num_timing_records', 0)); assert traces > 0 and timings == traces, (traces,timings); required=['timing_seconds.generation.median','timing_seconds.prefill.median','llm_input_tokens.median','cuda_memory_bytes.peak_allocation_delta.median']; assert all(numeric(get(x)) for x in required), {x:get(x) for x in required}; target=os.environ['PROFILE_TARGET']; assert target in {'baseline','spade'} or (d.get('num_compressed_records') == traces and d.get('token_summary',{}).get('chronological_index_violations') == []), 'compressed token traces are incomplete'; print(f'trace_records={traces} timing_memory=OK')"
  command_ok "non-empty profile timing=${path}"
}

normalize_method() {
  case "${1:-}" in
    segmentwise|segmentwise_topk) printf 'segmentwise\n' ;;
    atome|a_tome|a-tome) printf 'atome\n' ;;
    ltbm) printf 'ltbm\n' ;;
    sa_map|samap|sa-map) printf 'sa_map\n' ;;
    encoder_topk|encoder-topk) printf 'encoder_topk\n' ;;
    progressive|progressive_pruning) printf 'progressive\n' ;;
    audio_pyramid|audio-pyramid) printf 'audio_pyramid\n' ;;
    spade|spade_operation_only|spade-operation-only) printf 'spade\n' ;;
    *) fail "unknown method: ${1:-<missing>}" ;;
  esac
}

method_plugin() {
  case "$1" in
    segmentwise) printf 'segmentwise_topk_audio\n' ;;
    atome) printf 'adjacent_token_merging_audio\n' ;;
    ltbm) printf 'local_temporal_bipartite_merging_audio\n' ;;
    sa_map) printf 'similarity_attention_merge_prune_audio\n' ;;
    encoder_topk) printf 'encoder_attention_topk_audio\n' ;;
    progressive) printf 'progressive_attention_pruning_audio\n' ;;
    audio_pyramid) printf 'audio_pyramid_attention_pruning\n' ;;
    spade) printf 'structured_llm_layer_pruning\n' ;;
    *) fail "no plugin mapping for method: $1" ;;
  esac
}

train_config() {
  case "$1" in
    segmentwise) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_segmentwise_topk.yaml' ;;
    atome) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_atome.yaml' ;;
    ltbm) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_ltbm.yaml' ;;
    sa_map) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_sa_map.yaml' ;;
    encoder_topk) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_encoder_topk.yaml' ;;
    progressive) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_progressive_pruning.yaml' ;;
    audio_pyramid) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_audio_pyramid.yaml' ;;
    spade) printf '%s\n' 'configs/train/qwen2_5_omni/train_aud_sft_spade_operation_only.yaml' ;;
    *) fail "no train config mapping for method: $1" ;;
  esac
}

eval_config() {
  case "$1" in
    baseline) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud.yaml' ;;
    segmentwise) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_segmentwise_topk_paper.yaml' ;;
    atome) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_atome.yaml' ;;
    ltbm) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_ltbm.yaml' ;;
    sa_map) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_sa_map.yaml' ;;
    encoder_topk) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_encoder_topk.yaml' ;;
    progressive) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_progressive_pruning.yaml' ;;
    audio_pyramid) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_aud_audio_pyramid.yaml' ;;
    spade) printf '%s\n' 'configs/eval/lmms_eval/qwen25_omni_spade_operation_only.yaml' ;;
    *) fail "no eval config mapping for method: $1" ;;
  esac
}

run_oracle() {
  prepare_validation_root
  local method
  method="$(normalize_method "${1:-}")"
  local unit_file unit_selector cpu_file cpu_selector

  case "${method}" in
    segmentwise)
      unit_file="tests/unit/test_segmentwise_topk.py"
      unit_selector=""
      cpu_file="tests/integration/test_qwen25_omni_audio_compression_cpu.py"
      cpu_selector="test_real_audio_encoder_hooks_prune_two_audio_units_before_projector"
      ;;
    atome)
      unit_file="tests/unit/test_adjacent_token_merging.py"
      unit_selector=""
      cpu_file="tests/integration/test_qwen25_omni_encoder_compression_cpu.py"
      cpu_selector="test_real_audio_encoder_runs_layerwise_atome_and_repairs_lengths"
      ;;
    ltbm)
      unit_file="tests/unit/test_projector_extension_methods.py"
      unit_selector="ltbm"
      cpu_file="tests/integration/test_qwen25_omni_audio_compression_cpu.py"
      cpu_selector="local_temporal_bipartite_merging_audio"
      ;;
    sa_map)
      unit_file="tests/unit/test_projector_extension_methods.py"
      unit_selector="sa_map"
      cpu_file="tests/integration/test_qwen25_omni_audio_compression_cpu.py"
      cpu_selector="similarity_attention_merge_prune_audio"
      ;;
    encoder_topk)
      unit_file="tests/unit/test_encoder_extension_methods.py"
      unit_selector="attention_topk"
      cpu_file="tests/integration/test_qwen25_omni_encoder_compression_cpu.py"
      cpu_selector="encoder_attention_topk_audio"
      ;;
    progressive)
      unit_file="tests/unit/test_encoder_extension_methods.py"
      unit_selector="progressive"
      cpu_file="tests/integration/test_qwen25_omni_encoder_compression_cpu.py"
      cpu_selector="progressive_attention_pruning_audio"
      ;;
    audio_pyramid)
      unit_file="tests/unit/test_llm_level_compression.py"
      unit_selector=""
      cpu_file="tests/integration/test_qwen25_omni_llm_token_cpu.py"
      cpu_selector="prefill_decode_and_checkpoint_backward"
      ;;
    spade)
      unit_file="tests/unit/test_operation_compression.py"
      unit_selector=""
      cpu_file="tests/integration/test_qwen25_omni_llm_token_cpu.py"
      cpu_selector="operation_rewrite_export_and_reload"
      ;;
  esac

  local method_dir="${VALIDATION_ROOT}/02_oracle/${method}"
  if [[ -n "${unit_selector}" ]]; then
    run_pytest "${method_dir}/unit.log" "${unit_file}" -k "${unit_selector}"
  else
    run_pytest "${method_dir}/unit.log" "${unit_file}"
  fi
  run_pytest "${method_dir}/cpu_integration.log" \
    "${cpu_file}" -k "${cpu_selector}"
  command_ok "oracle ${method}; evidence=${method_dir}"
}

run_gpu() {
  prepare_validation_root
  require_gpu
  local target="${1:-}"

  case "${target}" in
    segmentwise)

      export MLLM_COMP_RUN_QWEN25_OMNI_AUDIO_COMPRESSION_GPU_SMOKE=1
      export QWEN25_OMNI_AUDIO_COMPRESSION_SMOKE_REPORT="${VALIDATION_ROOT}/03_gpu/segmentwise/report.json"
      require_new_report "${QWEN25_OMNI_AUDIO_COMPRESSION_SMOKE_REPORT}"
      run_pytest "${VALIDATION_ROOT}/03_gpu/segmentwise/pytest.log" \
        tests/integration/test_qwen25_omni_audio_compression_gpu_smoke.py::test_qwen25_omni_audio_compression_identity_and_r25_gpu
      validate_json "${QWEN25_OMNI_AUDIO_COMPRESSION_SMOKE_REPORT}"

      ;;
    atome)

      export MLLM_COMP_RUN_QWEN25_OMNI_ATOME_GPU_SMOKE=1
      export QWEN25_OMNI_ATOME_SMOKE_REPORT="${VALIDATION_ROOT}/03_gpu/atome/report.json"
      require_new_report "${QWEN25_OMNI_ATOME_SMOKE_REPORT}"
      run_pytest "${VALIDATION_ROOT}/03_gpu/atome/pytest.log" \
        tests/integration/test_qwen25_omni_encoder_compression_gpu_smoke.py::test_qwen25_omni_atome_default_mode_checkpoint_gpu_smoke
      validate_json "${QWEN25_OMNI_ATOME_SMOKE_REPORT}"

      ;;
    extensions)
      export MLLM_COMP_RUN_QWEN25_OMNI_EXTENSION_GPU_SMOKE=1
      export QWEN25_OMNI_EXTENSION_SMOKE_REPORT_DIR="${VALIDATION_ROOT}/03_gpu/extensions"
      require_new_report "${QWEN25_OMNI_EXTENSION_SMOKE_REPORT_DIR}/eval.json"
      run_pytest "${VALIDATION_ROOT}/03_gpu/extensions/pytest.log" \
        tests/integration/test_qwen25_omni_extension_methods_gpu_smoke.py::test_four_extension_methods_generate_from_real_checkpoint
      validate_json "${QWEN25_OMNI_EXTENSION_SMOKE_REPORT_DIR}/eval.json"
      ;;
    llm)

      export MLLM_COMP_RUN_QWEN25_OMNI_LLM_LEVEL_GPU_SMOKE=1
      export QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR="${VALIDATION_ROOT}/03_gpu/llm"
      require_new_report "${QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR}/llm_level_eval_train_export_reload.json"
      run_pytest "${VALIDATION_ROOT}/03_gpu/llm/pytest.log" \
        tests/integration/test_qwen25_omni_llm_level_compression_gpu_smoke.py::test_llm_level_eval_train_and_operation_export_reload
      validate_json "${QWEN25_OMNI_LLM_LEVEL_SMOKE_REPORT_DIR}/llm_level_eval_train_export_reload.json"

      ;;
    *) fail "gpu target must be: segmentwise, atome, extensions, or llm" ;;
  esac
  command_ok "real checkpoint ${target}; smoke assertions passed"
}

run_backward() {
  prepare_validation_root
  require_gpu
  local method plugin
  method="$(normalize_method "${1:-}")"
  case "${method}" in
    audio_pyramid|spade)
      fail "${method} backward is part of: gpu llm"
      ;;
  esac
  plugin="$(method_plugin "${method}")"
  export MLLM_COMP_RUN_QWEN25_OMNI_COMPRESSED_TRAIN_GPU_SMOKE=1
  export QWEN25_OMNI_COMPRESSED_TRAIN_SMOKE_REPORT_DIR="${VALIDATION_ROOT}/04_backward"
  require_new_report "${QWEN25_OMNI_COMPRESSED_TRAIN_SMOKE_REPORT_DIR}/${plugin}.json"

  run_pytest "${VALIDATION_ROOT}/04_backward/${method}.log" \
    tests/integration/test_qwen25_omni_compressed_training_gpu_smoke.py \
    -k "${plugin}"
  validate_json "${QWEN25_OMNI_COMPRESSED_TRAIN_SMOKE_REPORT_DIR}/${plugin}.json"
  command_ok "${method}; finite loss and nonzero audio gradient assertions passed"
}

run_check() {
  local target="${1:-}"
  case "${target}" in
    segmentwise)
      run_oracle segmentwise
      run_gpu segmentwise
      run_backward segmentwise
      ;;
    atome)
      run_oracle atome
      run_gpu atome
      run_backward atome
      ;;
    llm)
      run_oracle audio_pyramid
      run_oracle spade
      run_gpu llm
      ;;
    extensions)
      local method
      for method in ltbm sa_map encoder_topk progressive; do
        run_oracle "${method}"
      done
      run_gpu extensions
      for method in ltbm sa_map encoder_topk progressive; do
        run_backward "${method}"
      done
      ;;
    all)
      run_check segmentwise
      run_check atome
      run_check llm
      run_check extensions
      ;;
    *) fail "check target must be: segmentwise, atome, llm, extensions, or all" ;;
  esac
  command_ok "development check ${target}"
}

run_plan_one() {
  local method="$1"
  local output="${VALIDATION_ROOT}/05_swift/plans/${method}.json"
  mkdir -p "$(dirname -- "${output}")"
  "${python_bin}" -m mllm_comp.cli.main train \
    --config "$(train_config "${method}")" --plan_only > "${output}"
  validate_json "${output}"
}

run_plan() {
  prepare_validation_root
  require_model
  require_training_data
  local target="${1:-}" method
  if [[ "${target}" == "all" ]]; then
    for method in segmentwise atome ltbm sa_map encoder_topk progressive audio_pyramid spade; do
      run_plan_one "${method}"
    done
  else
    run_plan_one "$(normalize_method "${target}")"
  fi
}

run_swift() {
  prepare_validation_root
  require_gpu
  require_training_data
  local method config output_dir manifest
  method="$(normalize_method "${1:-}")"
  config="$(train_config "${method}")"
  output_dir="${VALIDATION_ROOT}/05_swift/runs/${method}"
  manifest="${output_dir}/manifest.json"

  run_logged "${VALIDATION_ROOT}/05_swift/${method}.log" \
    "${python_bin}" -m mllm_comp.cli.main train \
    --config "${config}" \
    --max_steps 1 \
    --output_dir "${output_dir}" \
    --manifest "${manifest}"
  validate_json "${manifest}"
  command_ok "Swift 1-step ${method}; native command and Manifest completed"
}

run_eval_one() {
  local target="$1"
  local mode="$2"
  local config output_root output_dir manifest max_new_tokens
  config="$(eval_config "${target}")"
  max_new_tokens="${VALIDATION_MAX_NEW_TOKENS:-16}"
  if [[ "${mode}" == "profile" ]]; then
    output_root="${VALIDATION_ROOT}/07_profile"
  else
    output_root="${VALIDATION_ROOT}/06_quality"
  fi
  output_dir="${output_root}/${target}"
  manifest="${output_dir}/manifest.json"
  local command=(
    "${python_bin}" -m mllm_comp.cli.main eval
    --config "${config}"
    --num_processes 1
    --generation_max_new_tokens "${max_new_tokens}"
    --use_cache
    --output_dir "${output_dir}"
    --manifest "${manifest}"
  )
  if [[ "${mode}" == "profile" ]]; then
    command+=(--profile)
  fi
  run_logged "${output_root}/${target}.log" "${command[@]}"
  validate_json "${manifest}"
  validate_eval_manifest "${manifest}"
  if [[ "${mode}" == "profile" ]]; then
    validate_eval_protocol "${output_dir}" profile "${max_new_tokens}"
  else
    validate_eval_protocol "${output_dir}" default "${max_new_tokens}"
  fi
  if [[ "${mode}" == "profile" ]]; then
    run_logged "${output_root}/${target}_summary.log" \
      "${python_bin}" scripts/analysis/summarize_audio_compression.py \
      "${output_dir}" --output "${output_dir}/efficiency_summary.json"
    validate_json "${output_dir}/efficiency_summary.json"
    validate_profile_summary "${output_dir}/efficiency_summary.json" "${target}"
  fi
}

run_eval_group() {
  prepare_validation_root
  require_gpu
  export SEGMENTWISE_KEEP_RATIO="${SEGMENTWISE_KEEP_RATIO:-0.25}"
  export SEGMENTWISE_RUN_TAG="${SEGMENTWISE_RUN_TAG:-r25}"
  local max_new_tokens="${VALIDATION_MAX_NEW_TOKENS:-16}"
  [[ "${max_new_tokens}" =~ ^[1-9][0-9]*$ ]] \
    || fail "VALIDATION_MAX_NEW_TOKENS must be a positive integer"
  local mode="$1" target="${2:-}" method
  if [[ "${target}" == "all" ]]; then
    for method in baseline segmentwise atome ltbm sa_map encoder_topk progressive audio_pyramid spade; do
      run_eval_one "${method}" "${mode}"
    done
  elif [[ "${target}" == "baseline" ]]; then
    run_eval_one baseline "${mode}"
  else
    run_eval_one "$(normalize_method "${target}")" "${mode}"
  fi
  command_ok "${mode} ${target}; compare matching baseline and generation settings before a research claim"
}

main() {
  local command="${1:-help}"
  shift || true
  case "${command}" in
    help|-h|--help) usage ;;
    oracle) run_oracle "$@" ;;
    gpu) run_gpu "$@" ;;
    backward) run_backward "$@" ;;
    check) run_check "$@" ;;
    plan) run_plan "$@" ;;
    swift) run_swift "$@" ;;
    quality|profile) run_eval_group "${command}" "$@" ;;
    *) usage; fail "unknown command: ${command}" ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
