#!/usr/bin/env bash
set -euo pipefail

if (( $# < 4 )); then
  printf 'usage: %s ROOT_DIR CONFIG_PATH OUTPUT_DIR MAIN_PROCESS_PORT [EVAL_ARGS...]\n' "$0" >&2
  exit 2
fi

root_dir="$1"
config_path="$2"
output_dir="$3"
main_process_port="$4"
shift 4

: "${MMAU_PRO_QWEN_JUDGE_MODEL:?set MMAU_PRO_QWEN_JUDGE_MODEL=/absolute/path/to/Qwen2.5-7B-Instruct}"
: "${MMAU_PRO_NVEMBED_MODEL:?set MMAU_PRO_NVEMBED_MODEL=/absolute/path/to/NV-Embed-v2}"

# This variable is scoped to stage 2. It can select a different GPU set without
# changing the stage-1 CUDA_VISIBLE_DEVICES value used by the caller.
visible_devices="${MMAU_PRO_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-}}"
visible_devices="${visible_devices//[[:space:]]/}"
if [[ -z "${visible_devices}" ]]; then
  printf 'set MMAU_PRO_CUDA_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES to a comma-separated GPU list\n' >&2
  exit 2
fi

IFS=',' read -r -a device_ids <<< "${visible_devices}"
for device_id in "${device_ids[@]}"; do
  if [[ -z "${device_id}" ]]; then
    printf 'invalid MMAU-Pro GPU list: %s\n' "${visible_devices}" >&2
    exit 2
  fi
done

requested_count="${MMAU_PRO_NUM_GPUS:-${#device_ids[@]}}"
if [[ ! "${requested_count}" =~ ^[1-9][0-9]*$ ]]; then
  printf 'MMAU_PRO_NUM_GPUS must be a positive integer, got: %s\n' "${requested_count}" >&2
  exit 2
fi
if (( requested_count > ${#device_ids[@]} )); then
  printf 'MMAU_PRO_NUM_GPUS=%s exceeds the %s GPUs in %s\n' \
    "${requested_count}" "${#device_ids[@]}" "${visible_devices}" >&2
  exit 2
fi

selected_devices="${device_ids[0]}"
for ((index = 1; index < requested_count; index++)); do
  selected_devices+=",${device_ids[index]}"
done

export CUDA_VISIBLE_DEVICES="${selected_devices}"
export MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY="${MMAU_PRO_EVALUATOR_LOCAL_FILES_ONLY:-true}"
export MMAU_PRO_EVALUATOR_DEVICE="${MMAU_PRO_EVALUATOR_DEVICE:-auto}"
export MMAU_PRO_EVALUATOR_DTYPE="${MMAU_PRO_EVALUATOR_DTYPE:-auto}"
export MMAU_PRO_NVEMBED_DEVICE_MAP="${MMAU_PRO_NVEMBED_DEVICE_MAP:-balanced}"
# Preserve the official MMAU-Pro Qwen Judge decoding settings. The runtime
# overlay below partitions only Qwen Judge samples across the visible GPUs.
export MMAU_PRO_JUDGE_MAX_NEW_TOKENS="${MMAU_PRO_JUDGE_MAX_NEW_TOKENS:-512}"
export MMAU_PRO_JUDGE_TEMPERATURE="${MMAU_PRO_JUDGE_TEMPERATURE:-0.1}"

# Keep the stage-specific orchestration out of the base evaluator. For this run,
# use a temporary lmms-eval overlay. Qwen Judge uses one worker per GPU;
# NV-Embed uses one worker that sees every selected GPU and shards its model
# with the configured device map.
runtime_overlay="$(mktemp -d "${TMPDIR:-/tmp}/mmau-pro-lmms-eval.XXXXXX")"
cleanup() {
  rm -rf -- "${runtime_overlay}"
}
trap cleanup EXIT

cp -R "${root_dir}/third_party/frameworks/lmms-eval/lmms_eval" "${runtime_overlay}/lmms_eval"
runtime_utils="${runtime_overlay}/lmms_eval/tasks/mmau_pro/utils.py"
runtime_evaluator="${runtime_overlay}/lmms_eval/evaluator.py"
"${PYTHON_BIN:-python}" - "${runtime_utils}" "${runtime_evaluator}" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text(encoding="utf-8")
old_judge_loop = '''    model, tokenizer = _load_qwen_judge()
    try:
        for item in items:
'''
new_judge_loop = '''    model, tokenizer = _load_qwen_judge()
    try:
        for item_index, item in enumerate(items, start=1):
'''
old_judge_progress = '''            del generated, continuation, model_inputs
'''
new_judge_progress = '''            if item_index == 1 or item_index % 10 == 0 or item_index == len(items):
                eval_logger.info(
                    f"MMAU-Pro Qwen judge progress: {item_index}/{len(items)}; "
                    f"worker_pid={os.getpid()}"
                )
            del generated, continuation, model_inputs
'''
old_nvembed_loop = '''    model = _load_nvembed()
    try:
        for item in items:
'''
new_nvembed_loop = '''    model = _load_nvembed()
    try:
        for item_index, item in enumerate(items, start=1):
'''
old_nvembed_progress = '''            del prediction_embedding, choice_embeddings, similarities
'''
new_nvembed_progress = '''            if item_index == 1 or item_index % 50 == 0 or item_index == len(items):
                eval_logger.info(
                    f"MMAU-Pro NV-Embed progress: {item_index}/{len(items)}"
                )
            del prediction_embedding, choice_embeddings, similarities
'''
if (
    source.count(old_judge_loop) != 1
    or source.count(old_judge_progress) != 1
    or source.count(old_nvembed_loop) != 1
    or source.count(old_nvembed_progress) != 1
):
    raise RuntimeError(f"unexpected MMAU-Pro evaluator loop in {path}")
source = (
    source.replace(old_judge_loop, new_judge_loop)
    .replace(old_judge_progress, new_judge_progress)
    .replace(old_nvembed_loop, new_nvembed_loop)
    .replace(old_nvembed_progress, new_nvembed_progress)
)

function_start = source.index("def _run_isolated_evaluator(evaluator, items):")
function_end = source.index(
    "\n\n# ---------------------------------------------------------------------------\n"
    "# Official evaluator 1:",
    function_start,
)
parallel_evaluator = r'''def _run_isolated_evaluator(evaluator, items):
    """Run evaluator workers after the generation model has released its VRAM.

    Qwen Judge is data-parallel: split samples across one single-GPU worker per
    visible GPU. NV-Embed is model-parallel: launch one worker that sees every
    visible GPU and lets Transformers shard one model across them.
    """
    if not items:
        return []
    if evaluator not in {"qwen_judge", "nvembed"}:
        raise ValueError(f"Unknown MMAU-Pro evaluator worker: {evaluator!r}")

    _configure_evaluators()
    worker_path = Path(__file__).with_name("evaluator_worker.py")
    if not worker_path.is_file():
        raise FileNotFoundError(f"MMAU-Pro evaluator worker is missing: {worker_path}")

    raw_device = _clean_text(
        _EVALUATOR_CONFIG.get("evaluator_device") or "cuda"
    ).lower()
    visible_gpu_ids = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    data_parallel = (
        evaluator == "qwen_judge"
        and raw_device == "auto"
        and len(visible_gpu_ids) > 1
    )

    _clear_device_cache()
    if data_parallel:
        worker_count = min(len(visible_gpu_ids), len(items))
        item_shards = [items[index::worker_count] for index in range(worker_count)]
        eval_logger.info(
            f"Starting data-parallel MMAU-Pro {evaluator}: workers={worker_count}, "
            f"samples={len(items)}, visible_devices={','.join(visible_gpu_ids)}"
        )
        with tempfile.TemporaryDirectory(prefix=f"mmau_pro_{evaluator}_parallel_") as temp_dir:
            temp_path = Path(temp_dir)
            processes = []
            output_paths = []
            for worker_index, item_shard in enumerate(item_shards):
                input_path = temp_path / f"input_{worker_index}.json"
                output_path = temp_path / f"output_{worker_index}.json"
                payload = {
                    "config": _to_builtin(dict(_EVALUATOR_CONFIG)),
                    "items": _to_builtin(item_shard),
                }
                with input_path.open("w", encoding="utf-8") as handle:
                    json.dump(payload, handle, ensure_ascii=False)

                child_env = os.environ.copy()
                child_env["PYTHONUNBUFFERED"] = "1"
                child_env["CUDA_VISIBLE_DEVICES"] = visible_gpu_ids[worker_index]
                eval_logger.info(
                    f"Starting {evaluator} worker {worker_index + 1}/{worker_count}: "
                    f"CUDA_VISIBLE_DEVICES={visible_gpu_ids[worker_index]}, "
                    f"samples={len(item_shard)}"
                )
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            str(worker_path),
                            evaluator,
                            str(input_path),
                            str(output_path),
                        ],
                        env=child_env,
                        cwd=str(worker_path.parent),
                    )
                )
                output_paths.append(output_path)

            failures = []
            for worker_index, process in enumerate(processes):
                return_code = process.wait()
                if return_code != 0:
                    failures.append((worker_index, return_code))
            if failures:
                details = ", ".join(
                    f"worker={index}, exit={return_code}"
                    for index, return_code in failures
                )
                raise RuntimeError(f"Parallel MMAU-Pro {evaluator} failed: {details}")

            records = [None] * len(items)
            for worker_index, (item_shard, output_path) in enumerate(
                zip(item_shards, output_paths)
            ):
                if not output_path.is_file():
                    raise RuntimeError(
                        f"MMAU-Pro {evaluator} worker {worker_index} produced no output"
                    )
                with output_path.open("r", encoding="utf-8") as handle:
                    shard_records = json.load(handle)
                if not isinstance(shard_records, list) or len(shard_records) != len(item_shard):
                    raise RuntimeError(
                        f"MMAU-Pro {evaluator} worker {worker_index} returned an "
                        f"invalid record count"
                    )
                for shard_offset, record in enumerate(shard_records):
                    records[worker_index + shard_offset * worker_count] = record

        _clear_device_cache()
        if any(record is None for record in records):
            raise RuntimeError(f"Parallel MMAU-Pro {evaluator} lost one or more records")
        return records

    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"
    if raw_device == "cpu" or raw_device.startswith("cpu:"):
        child_env["CUDA_VISIBLE_DEVICES"] = ""
        child_env["NVIDIA_VISIBLE_DEVICES"] = "void"

    payload = {
        "config": _to_builtin(dict(_EVALUATOR_CONFIG)),
        "items": _to_builtin(items),
    }
    with tempfile.TemporaryDirectory(prefix=f"mmau_pro_{evaluator}_") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        output_path = Path(temp_dir) / "output.json"
        with input_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)

        eval_logger.info(
            f"Starting isolated MMAU-Pro {evaluator} worker: "
            f"device={raw_device}, "
            f"visible_devices={','.join(visible_gpu_ids) or 'none'}, "
            f"samples={len(items)}"
        )
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(worker_path),
                    evaluator,
                    str(input_path),
                    str(output_path),
                ],
                check=True,
                env=child_env,
                cwd=str(worker_path.parent),
            )
        finally:
            _clear_device_cache()

        if not output_path.is_file():
            raise RuntimeError(f"MMAU-Pro {evaluator} worker produced no output")
        with output_path.open("r", encoding="utf-8") as handle:
            records = json.load(handle)

    if not isinstance(records, list) or len(records) != len(items):
        record_count = len(records) if isinstance(records, list) else "not-a-list"
        raise RuntimeError(
            f"MMAU-Pro {evaluator} worker returned {record_count} records "
            f"for {len(items)} samples"
        )
    return records
'''
source = source[:function_start] + parallel_evaluator + source[function_end:]
path.write_text(source, encoding="utf-8")

# lmms-eval calls lm.clean() on every rank before aggregation, but the platform
# adapter keeps its model under a non-nn.Module bridge that the generic cleanup
# cannot see. Release that nested runtime on every rank before the existing
# distributed barrier, so evaluator workers can actually use all visible GPUs.
evaluator_path = Path(sys.argv[2])
evaluator_source = evaluator_path.read_text(encoding="utf-8")
old_rank_cleanup = '''    lm.clean()
    RANK = global_rank
'''
new_rank_cleanup = '''    lm.clean()
    platform_bridge = vars(lm).pop("_platform_bridge", None)
    if platform_bridge is not None:
        platform_runtime = getattr(platform_bridge, "runtime", None)
        if platform_runtime is not None:
            platform_runtime.model = None
            platform_runtime.processor = None
        platform_bridge.runtime = None
        platform_bridge.facade = None
        del platform_runtime, platform_bridge
        import gc

        gc.collect()
        torch.cuda.empty_cache()
        eval_logger.info(
            f"Rank {global_rank}: released nested MLLM-Comp generation runtime "
            "before MMAU-Pro evaluator aggregation"
        )
    RANK = global_rank
'''
if evaluator_source.count(old_rank_cleanup) != 1:
    raise RuntimeError(f"unexpected lmms-eval cleanup block in {evaluator_path}")
evaluator_path.write_text(
    evaluator_source.replace(old_rank_cleanup, new_rank_cleanup),
    encoding="utf-8",
)
PY

export PYTHONPATH="${runtime_overlay}${PYTHONPATH:+:${PYTHONPATH}}"

printf 'MMAU-Pro stage 2: CUDA_VISIBLE_DEVICES=%s, generation_processes=%s, Qwen-Judge workers=%s, NV-Embed workers=1, NV-Embed device_map=%s\n' \
  "${CUDA_VISIBLE_DEVICES}" "${requested_count}" "${requested_count}" "${MMAU_PRO_NVEMBED_DEVICE_MAP}"
printf 'MMAU-Pro Qwen judge: max_new_tokens=%s, temperature=%s\n' \
  "${MMAU_PRO_JUDGE_MAX_NEW_TOKENS}" "${MMAU_PRO_JUDGE_TEMPERATURE}"
mkdir -p "${output_dir}"
cd "${root_dir}"
MLLM_COMP_EVAL_RUN_DIR="${output_dir}" "${PYTHON_BIN:-python}" -m mllm_comp.cli.main eval \
  --config "${config_path}" \
  "$@" \
  --num_processes "${requested_count}" \
  --output_dir "${output_dir}" \
  --manifest "${output_dir}/manifest.json" \
  --main_process_port "${main_process_port}"
