from __future__ import annotations

import importlib
import json
import subprocess
import sys
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from scripts.diagnostics.qwen_video_checkpoint_common import (
    mrope_gather_exact,
    qwen25_video_input_record,
)
from scripts.diagnostics.video_checkpoint_common import (
    compare_generation,
    inference_main,
    load_generation_runtime,
)
from scripts.diagnostics.video_training_checkpoint_common import llava_training_fixture


@pytest.mark.parametrize(
    "representatives,result",
    [
        ([0, 2], [(0, 0, 0), (4, 5, 6)]),
        ([2, 0], [(4, 5, 6), (0, 0, 0)]),
    ],
    ids=["source-order", "explicit-reorder"],
)
def test_mrope_evidence_replays_representative_source_columns(
    representatives, result
) -> None:
    splice = {
        "source_visual_mrope": [(0, 0, 0), (1, 2, 3), (4, 5, 6)],
        "result_visual_mrope": result,
        "representative_indices": representatives,
    }

    assert mrope_gather_exact(splice) is True
    splice["result_visual_mrope"] = list(reversed(result))
    assert mrope_gather_exact(splice) is False


@pytest.fixture
def qwen25_native_input():
    state = SimpleNamespace(
        modality_token_spans=[SimpleNamespace(modality="video", start=11, end=1579)],
        video_grid_thw=[[8, 28, 28]],
        video_second_per_grid=[[0.5]],
        video_pixel_values=SimpleNamespace(shape=(6272, 1176)),
        labels=None,
        native_model_inputs={},
    )
    batch = SimpleNamespace(
        metadata={
            "spatial_merge_size": 2,
            "processor_output_contract": "native-video",
        }
    )
    return state, batch


def test_qwen25_video_input_record_keeps_native_layout_and_timing(qwen25_native_input):
    state, batch = qwen25_native_input
    assert qwen25_video_input_record(state, batch) == {
        "video_grid_thw": [8, 28, 28],
        "second_per_grid_ts": [0.5],
        "spatial_merge_size": 2,
        "source_grid_token_count": 1568,
        "video_span": {"start": 11, "end": 1579, "count": 1568},
        "pixel_values_videos_shape": [6272, 1176],
        "processor_output_contract": "native-video",
        "labels_absent_for_inference": True,
    }
    state.native_model_inputs["labels"] = [1]
    assert not qwen25_video_input_record(state, batch)["labels_absent_for_inference"]


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("modality_token_spans", [], "video span"),
        ("video_grid_thw", [[8, 28]], "video_grid_thw row"),
        ("video_grid_thw", [[7, 28, 28]], "span differs"),
        ("video_second_per_grid", [0.5, 0.5], "second_per_grid_ts"),
        ("video_second_per_grid", [float("nan")], "finite"),
    ],
)
def test_qwen25_video_input_record_rejects_incoherent_native_input(
    qwen25_native_input,
    field,
    value,
    error,
):
    state, batch = qwen25_native_input
    setattr(state, field, value)
    with pytest.raises(RuntimeError, match=error):
        qwen25_video_input_record(state, batch)


@pytest.mark.parametrize("failure", [None, "generation", "validation"])
def test_generation_comparison_uninstalls_after_success_or_failure(failure) -> None:
    events = []
    activation = SimpleNamespace(installed=False, calls=0)
    activation.runtime_metadata = lambda: {"calls": activation.calls}
    state = object()
    torch = SimpleNamespace(inference_mode=nullcontext)

    def install():
        events.append("install")
        activation.installed = True

    def uninstall():
        events.append("uninstall")
        activation.installed = False

    activation.install = install
    activation.uninstall = uninstall

    def generate(**inputs):
        assert inputs == {"torch": torch, "state": state, "max_new_tokens": 4}
        phase = "compressed" if activation.installed else "baseline"
        events.append(phase)
        if activation.installed:
            activation.calls += 1
            if failure == "generation":
                raise RuntimeError("generation failed")
        return {"phase": phase}

    expected = nullcontext() if failure is None else pytest.raises(RuntimeError)
    with expected:
        with compare_generation(
            activation=activation,
            generate=generate,
            torch=torch,
            state=state,
            max_new_tokens=4,
        ) as (baseline, compressed, before, after):
            assert activation.installed
            assert (baseline, compressed) == (
                {"phase": "baseline"},
                {"phase": "compressed"},
            )
            assert (before, after) == ({"calls": 0}, {"calls": 1})
            events.append("validation")
            if failure == "validation":
                raise RuntimeError("validation failed")

    assert not activation.installed
    assert events == ["baseline", "install", "compressed"] + (
        [] if failure == "generation" else ["validation"]
    ) + ["uninstall"]


@pytest.mark.parametrize(
    "dtype,device_map,metadata",
    [
        ("float16", "cuda:0", {"fastvid_siglip_path": "/local/siglip"}),
        ("bfloat16", {"": 0}, {"enable_talker": False, "speech_sample_rate": None}),
    ],
    ids=["llava", "qwen"],
)
def test_generation_load_keeps_backbone_dtype_device_and_runtime_metadata(
    dtype,
    device_map,
    metadata,
) -> None:
    checkpoint = Path("/local/checkpoint")
    processor, model = object(), Mock()
    facade = SimpleNamespace(
        processor_adapter=SimpleNamespace(load_processor=Mock(return_value=processor)),
        model_adapter=SimpleNamespace(load_model=Mock(return_value=model)),
    )
    torch = SimpleNamespace(device=Mock(return_value="logical-device"))
    loaded_processor, loaded_model, runtime = load_generation_runtime(
        checkpoint,
        facade=facade,
        torch=torch,
        dtype=dtype,
        device_map=device_map,
        metadata=metadata,
    )

    facade.processor_adapter.load_processor.assert_called_once_with(
        str(checkpoint),
        local_files_only=True,
    )
    facade.model_adapter.load_model.assert_called_once_with(
        str(checkpoint),
        train=False,
        local_files_only=True,
        torch_dtype=dtype,
        attn_implementation="sdpa",
        device_map=device_map,
    )
    model.eval.assert_called_once_with()
    torch.device.assert_called_once_with("cuda:0")
    assert loaded_processor is runtime.processor is processor
    assert loaded_model is runtime.model is model
    assert runtime.dtype == dtype
    assert runtime.device == "logical-device"
    assert runtime.metadata is metadata


_ALGORITHMS = (
    "dycoke",
    "dytok",
    "fastvid",
    "metok",
    "prunevid",
    "visionzip",
    "vqtoken",
)
_COMMANDS = [
    f"validate_{backbone}{'' if algorithm == 'vqtoken' else '_' + algorithm}_video_compression_checkpoint"
    for backbone in ("llava", "qwen25_vl")
    for algorithm in _ALGORITHMS
]


@pytest.fixture(
    params=_COMMANDS,
    ids=lambda name: name.removeprefix("validate_").removesuffix(
        "_video_compression_checkpoint"
    ),
)
def command(request):
    return importlib.import_module("scripts.diagnostics." + request.param)


def _arguments(command, output: Path, *, max_new_tokens: int = 4) -> list[str]:
    arguments = [
        "--checkpoint",
        str(output.parent / "checkpoint"),
        "--nextqa_parquet",
        str(output.parent / "nextqa.parquet"),
        "--video_root",
        str(output.parent / "videos"),
        "--output_dir",
        str(output),
        "--code_revision",
        "fixture-revision",
        "--physical_gpu",
        "0",
        "--max_new_tokens",
        str(max_new_tokens),
    ]
    options = {
        option
        for action in command._parser()._actions
        for option in action.option_strings
    }
    for option in ("--detector_checkpoint", "--detector_siglip", "--siglip_checkpoint"):
        if option in options:
            arguments.extend((option, str(output.parent / option[2:])))
    return arguments


@pytest.mark.parametrize("entry", ["shared", "qwen3"])
def test_inference_main_writes_report_and_samples_without_overwriting(
    tmp_path,
    capsys,
    monkeypatch,
    entry,
) -> None:
    output = tmp_path / "evidence"
    args = SimpleNamespace(output_dir=output)
    calls = []

    def run(args):
        calls.append(args)
        return {"status": "passed", "sample_count": 1}, [
            {"doc_id": 442, "text": "答案 D"}
        ]

    if entry == "qwen3":
        qwen3 = importlib.import_module(
            "scripts.diagnostics.validate_qwen3_vl_video_compression_checkpoint"
        )
        args.max_new_tokens, args.video_fps, args.video_max_pixels = 4, 1.0, 131072
        monkeypatch.setattr(qwen3, "_run", lambda args, profile=None: run(args))
        run_main = partial(qwen3._execute, args)
    else:
        run_main = partial(inference_main, args, run)

    assert run_main() == 0
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert report == {
        "status": "passed",
        "sample_count": 1,
        "artifacts": {
            "report": str(output / "report.json"),
            "lmms_eval_samples": str(output / "samples.jsonl"),
        },
    }
    assert json.loads(capsys.readouterr().out) == {
        "status": "passed", "report": str(output / "report.json"), "sample_count": 1,
    }
    assert json.loads((output / "samples.jsonl").read_text(encoding="utf-8")) == {
        "doc_id": 442,
        "text": "答案 D",
    }
    assert calls == [args]
    before = (output / "report.json").read_bytes()
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_main()
    assert len(calls) == 1
    assert (output / "report.json").read_bytes() == before


@pytest.mark.parametrize(
    "error", [RuntimeError("checkpoint unavailable"), KeyboardInterrupt()]
)
def test_inference_main_records_failure_and_reraises_same_exception(
    tmp_path,
    error,
) -> None:
    output = tmp_path / "failed"
    args = SimpleNamespace(
        output_dir=output, code_revision="fixture-revision", physical_gpu=0
    )

    def run(args):
        raise error

    with pytest.raises(type(error)) as caught:
        inference_main(args, run)
    assert caught.value is error
    report = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert set(report) == {
        "status",
        "error_type",
        "error",
        "command",
        "code_revision",
        "physical_gpu",
        "elapsed_seconds",
    }
    assert report["status"] == "failed"
    assert report["error_type"] == type(error).__name__
    assert report["error"] == str(error)
    assert report["code_revision"] == "fixture-revision"
    assert report["physical_gpu"] == 0
    assert not (output / "samples.jsonl").exists()


def test_commands_delegate_validated_arguments_and_reject_invalid_bound(
    command,
    tmp_path,
    monkeypatch,
) -> None:
    output = tmp_path / "invalid"
    calls = []

    def main(args, run):
        assert run is command._run
        assert args.output_dir == output
        assert args.max_new_tokens == 4
        assert args.code_revision == "fixture-revision"
        assert args.physical_gpu == 0
        calls.append(args)
        return 17

    assert command.inference_main is inference_main
    monkeypatch.setattr(command, "inference_main", main)
    assert command.main(_arguments(command, output)) == 17
    with pytest.raises(RuntimeError, match="max_new_tokens"):
        command.main(_arguments(command, output, max_new_tokens=1))
    assert not output.exists()
    assert len(calls) == 1


def test_shared_llava_training_fixture_preserves_test_data_boundary(
    tmp_path,
) -> None:
    for algorithm in ("dycoke", "dytok", "fastvid", "visionzip", "metok", "prunevid"):
        command = importlib.import_module(
            f"scripts.diagnostics.validate_llava_{algorithm}_video_training_checkpoint"
        )
        assert command._fixture is llava_training_fixture
    video = tmp_path / "3064389881.mp4"
    fixture, path = llava_training_fixture(
        document={"video": "3064389881", "prompt": "What happens?", "target": "D"},
        video=video,
        output_dir=tmp_path,
    )
    assert path == tmp_path / "fixture.json"
    assert (
        json.loads(path.read_text(encoding="utf-8"))
        == fixture
        == {
            "schema_version": 1,
            "kind": "nextqa_mc_test_derived_sft_code_correctness_fixture",
            "quality_or_training_data_claim": False,
            "task": "nextqa_mc_test",
            "split": "test",
            "doc_id": 442,
            "video_id": "3064389881",
            "video_path": str(video),
            "prompt": "What happens?",
            "target": "D",
            "video_preprocessing": {
                "decoded_frames": 32,
                "decode_policy": "uniform_local_decord",
            },
        }
    )


def test_common_modules_import_without_loading_command_entrypoints() -> None:
    script = """
import importlib, sys
for name in ('video_checkpoint_common', 'video_training_checkpoint_common',
             'qwen_video_checkpoint_common', 'qwen25_provider_video_checkpoint_common',
             'llava_provider_video_checkpoint_common', 'fastvid_video_checkpoint_common'):
    importlib.import_module('scripts.diagnostics.' + name)
assert not [name for name in sys.modules if name.startswith('scripts.diagnostics.validate_')]
"""
    subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True, text=True
    )


@pytest.mark.parametrize(
    "algorithm", ["dycoke", "dytok", "fastvid", "metok", "prunevid", "visionzip"]
)
def test_qwen_training_grid_records_keep_timing_conversion_and_validation(
    algorithm,
) -> None:
    command = importlib.import_module(
        f"scripts.diagnostics.validate_qwen25_{algorithm}_video_training_checkpoint"
    )
    state = SimpleNamespace(video_grid_thw=[[8, 28, 28]], video_second_per_grid=[[0.5]])
    batch = SimpleNamespace(metadata={"spatial_merge_size": 2})
    record = command._grid_record(state, batch)
    assert record["video_grid_thw"] == [8, 28, 28]
    assert record["spatial_merge_size"] == 2
    assert record["temporal_slice_count"] == 8
    assert record["per_temporal_slice_source_count"] == 196
    assert record["source_grid_token_count"] == 1568
    assert record["second_per_grid_ts"] == [0.5]
    state.video_second_per_grid = [0.0]
    with pytest.raises(RuntimeError, match="positive video timing"):
        command._grid_record(state, batch)


@pytest.mark.parametrize(
    "command_name,max_pixels",
    [
        ("qwen25_vl_fastvid", 128 * 28 * 28),
        ("qwen25_vl_metok", 128 * 28 * 28),
        ("qwen25_vl_prunevid", 128 * 28 * 28),
        ("qwen3_vl_fastvid", 128 * 32 * 32),
        ("qwen3_vl_metok", 128 * 32 * 32),
        ("qwen3_vl", 128 * 32 * 32),
    ],
)
def test_native_video_conversation_keeps_each_command_policy(
    command_name, max_pixels
) -> None:
    command = importlib.import_module(
        f"scripts.diagnostics.validate_{command_name}_video_compression_checkpoint"
    )
    prompt = "Question\nA. one\nB. two"
    video = Path("/dataset/sample.mp4")
    conversation = command._conversation(prompt=prompt, video=video)
    assert conversation[1]["content"][0] == {
        "type": "video",
        "video": str(video),
        "fps": 1.0,
        "max_pixels": max_pixels,
    }
    assert conversation[1]["content"][1]["text"] == prompt


def test_completed_cache_gate_requires_every_physical_layer_to_shrink() -> None:
    from scripts.diagnostics.video_method_checks import prunevid_cache_layers_exact

    cache = {
        "policy": "compact_prefix", "boundary_layer": 10, "boundary": "after_layer",
        "source_length": 7, "result_length": 4, "touched_layers": list(range(11)),
        "cache_type": "DynamicCache",
        "layers_before": [
            {"layer_index": i, "logical_length": 7, "physical_length": 7}
            for i in range(11)
        ],
        "layers_after": [
            {"layer_index": i, "logical_length": 4, "physical_length": 4}
            for i in range(11)
        ],
    }
    assert prunevid_cache_layers_exact(cache, source_length=7, result_length=4)
    cache["layers_after"][-1]["physical_length"] = 7
    assert not prunevid_cache_layers_exact(cache, source_length=7, result_length=4)
    cache["layers_after"][-1]["physical_length"] = 4
    cache["touched_layers"].pop()
    assert not prunevid_cache_layers_exact(cache, source_length=7, result_length=4)


def test_progressive_report_gate_rejects_a_missing_boundary() -> None:
    from scripts.diagnostics.validate_llava_metok_video_compression_checkpoint import (
        _pruning_chain_exact,
    )

    assert not _pruning_chain_exact([{}, {}], [{}, {}], [{}, {}], projector_count=4)


@pytest.mark.parametrize("token_count", [4, 3])
def test_dycoke_generation_still_requires_the_exact_decode_length(
    token_count, monkeypatch,
) -> None:
    from scripts.diagnostics import (
        validate_qwen25_vl_dycoke_video_compression_checkpoint as command,
    )

    report = {"generated_token_count": token_count}
    generate = Mock(return_value=report)
    monkeypatch.setattr(command._common, "generation_report", generate)
    run = partial(
        command._generation_report,
        facade=object(), runtime=object(), state=object(), batch=object(),
        audit=object(), torch=object(), max_new_tokens=4,
    )
    if token_count == 4:
        assert run() is report
    else:
        with pytest.raises(RuntimeError, match="exact bounded decode length"):
            run()
    assert generate.call_count == 1
    assert generate.call_args.kwargs["min_new_tokens"] == 4
    assert generate.call_args.kwargs["max_new_tokens"] == 4
