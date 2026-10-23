"""Opt-in LLaVA-OneVision compression forward/backward smoke on one GPU."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from tests.fixtures.native_image import training_backward as _run_training_backward

import pytest
from tests.fixtures.native_image import stage as _stage

import mllm_comp.backbones.llava_onevision.image_compression as compression_module
from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.vocabulary import ExecutionPhase
from tests.image_compression_semantic_audit import (
    DecoderPositionAuditRecorder,
    SemanticAuditRecorder,
)


pytestmark = pytest.mark.gpu


def _enabled() -> bool:
    return os.environ.get("MLLM_COMP_RUN_LLAVA_COMPRESSION_GPU_SMOKE") == "1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _move(value: Any, torch: Any) -> Any:
    if isinstance(value, list):
        return [_move(item, torch) for item in value]
    to = getattr(value, "to", None)
    if not callable(to):
        return value
    value = value.to(torch.device("cuda:0"))
    if value.is_floating_point():
        value = value.to(dtype=torch.float16)
    return value


def _assert_tome_reduces_every_vision_layer(
    model: Any,
    records: list[dict[str, Any]],
) -> None:
    vision_tower = model.get_vision_tower().vision_tower
    layer_count = len(vision_tower.vision_model.encoder.layers)
    layer_records = [item for item in records if "layer_index" in item]
    observed_layers = {int(item["layer_index"]) for item in layer_records}
    assert observed_layers == set(range(layer_count))
    assert all(
        int(item["source_count"]) - int(item["result_count"]) == 1
        for item in layer_records
    )
    permanent_records = [
        item
        for item in records
        if item.get("site") == "post_anyres_pack_pre_llm_splice_tome"
    ]
    assert permanent_records
    assert all(
        item["media_unit"] == "image_feature"
        and int(item["source_count"]) - int(item["result_count"]) == 1
        for item in permanent_records
    )


@pytest.mark.skipif(
    not _enabled(), reason="real LLaVA-OneVision compression smoke is opt-in"
)
def test_llava_onevision_three_image_compressors_real_forward_and_backward(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_gpu = os.environ.get("MLLM_COMP_EXPECTED_PHYSICAL_GPU", "1")
    assert expected_gpu in {"0", "1"}, "MLLM_COMP_EXPECTED_PHYSICAL_GPU must be 0 or 1."
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == expected_gpu, (
        "CUDA_VISIBLE_DEVICES must equal MLLM_COMP_EXPECTED_PHYSICAL_GPU "
        f"(expected {expected_gpu!r})."
    )
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    image_module = pytest.importorskip("PIL.Image")
    assert transformers.__version__ == "4.57.3"
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1

    model_path_value = os.environ.get("LLAVA_ONEVISION_MODEL_PATH")
    if not model_path_value:
        pytest.fail("LLAVA_ONEVISION_MODEL_PATH must point to a local checkpoint.")
    model_path = Path(model_path_value)
    assert model_path.is_dir()
    image_path = tmp_path / "llava_onevision_compression.png"
    image_module.new("RGB", (384, 256), color=(176, 48, 64)).save(image_path)

    facade = create_llava_onevision_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path), local_files_only=True
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
        device_map="cuda:0",
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device("cuda:0"),
        dtype=torch.float16,
        metadata={"enable_talker": False, "speech_sample_rate": None},
    )
    sample = LlavaOneVisionSample(
        sample_id="compression-forward",
        prompt="What is the dominant color?",
        images=(str(image_path),),
    )
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
    state = facade.state_builder.build(batch, phase="eval")
    model_inputs = {
        name: _move(value, torch)
        for name, value in state.native_model_inputs.items()
        if value is not None and name != "labels"
    }
    model_inputs.update({"use_cache": False, "return_dict": True})
    model.eval()

    semantic_audit = SemanticAuditRecorder()
    semantic_audit.install(monkeypatch, compression_module)
    position_audit = DecoderPositionAuditRecorder(model.get_model(), "text_1d")
    position_audit.install()
    baseline_position_mark = position_audit.mark()
    with torch.no_grad():
        baseline_output = model(**model_inputs)
    baseline_logits = getattr(baseline_output, "logits", None)
    assert baseline_logits is not None
    baseline_length = int(baseline_logits.shape[1])
    baseline_position_summary = position_audit.summary_since(baseline_position_mark)

    registry = builtin_multimodal_compression_contracts()
    cases = (
        ("tome", "tome.image.llava_onevision", {"r": 1, "output_r": 1}),
        ("visionzip", "visionzip.image.llava_onevision", {"budget": 32}),
        (
            "pyramid_drop",
            "pyramid_drop.image.llava_onevision",
            {"keep_ratio": 0.5, "num_stages": 4},
        ),
    )
    report: dict[str, Any] = {
        "environment": {
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "expected_physical_gpu": expected_gpu,
            "gpu_name": torch.cuda.get_device_name(0),
            "model_path": str(model_path),
            "config_sha256": _sha256(model_path / "config.json"),
        },
        "fixture_sha256": _sha256(image_path),
        "execution_phase": ExecutionPhase.FORWARD.value,
        "use_cache": False,
        "training_backward_verified": True,
        "baseline_logits_shape": [int(item) for item in baseline_logits.shape],
        "baseline_position_audit": baseline_position_summary,
        "cases": {},
    }
    for algorithm, binding_name, config in cases:
        binding = registry.binding(binding_name)
        activation = binding.executor_factory(
            model=model,
            runtime=runtime,
            facade=facade,
            stage=_stage(algorithm, binding.variant, config),
            binding=binding,
            binding_config=SimpleNamespace(config={}),
            execution_phase=ExecutionPhase.FORWARD,
        )
        activation.install()
        try:
            bridge = getattr(model, "mllm_compression_bridge")
            semantic_mark = semantic_audit.mark()
            position_mark = position_audit.mark()
            with torch.no_grad():
                output = model(**model_inputs)
            logits = getattr(output, "logits", None)
            assert logits is not None and int(logits.shape[0]) == 1
            assert int(logits.shape[1]) < baseline_length
            metadata = activation.runtime_metadata()
            records = metadata["layout_delta_records"]
            assert records, f"{algorithm} ran without publishing LayoutDelta evidence"
            assert any(
                int(item["result_count"]) < int(item["source_count"])
                for item in records
            ), f"{algorithm} did not reduce a token layout"
            if algorithm == "tome":
                _assert_tome_reduces_every_vision_layer(model, records)
            semantic_eval = semantic_audit.since(semantic_mark)
            assert semantic_eval[algorithm]
            position_eval = position_audit.summary_since(position_mark)
            training_semantic_mark = semantic_audit.mark()
            training_position_mark = position_audit.mark()
            training = _run_training_backward(model, model_inputs, torch)
            assert training["output_logits_shape"][1] < baseline_length
            report["cases"][algorithm] = {
                "binding": binding_name,
                "output_logits_shape": [int(item) for item in logits.shape],
                "activation": metadata,
                "provider": bridge.runtime_metadata(),
                "provider_auto_installed": True,
                "semantic_audit": {
                    "eval": semantic_eval,
                    "training": semantic_audit.since(training_semantic_mark),
                },
                "position_audit": {
                    "eval": position_eval,
                    "training": position_audit.summary_since(
                        training_position_mark
                    ),
                },
                "training_backward": training,
            }
        finally:
            activation.uninstall()
        assert not hasattr(model, "mllm_compression_bridge")

    combined = []
    try:
        for algorithm, binding_name, config in cases:
            binding = registry.binding(binding_name)
            activation = binding.executor_factory(
                model=model,
                runtime=runtime,
                facade=facade,
                stage=_stage(algorithm, binding.variant, config),
                binding=binding,
                binding_config=SimpleNamespace(config={}),
                execution_phase=ExecutionPhase.FORWARD,
            )
            activation.install()
            combined.append(activation)
        combined_semantic_mark = semantic_audit.mark()
        combined_position_mark = position_audit.mark()
        with torch.no_grad():
            output = model(**model_inputs)
        logits = getattr(output, "logits", None)
        assert logits is not None and int(logits.shape[0]) == 1
        assert int(logits.shape[1]) < baseline_length
        combined_metadata = [item.runtime_metadata() for item in combined]
        assert all(item["layout_delta_records"] for item in combined_metadata)
        _assert_tome_reduces_every_vision_layer(
            model,
            combined_metadata[0]["layout_delta_records"],
        )
        report["combined_three_stage"] = {
            "order": [item[0] for item in cases],
            "output_logits_shape": [int(item) for item in logits.shape],
            "activations": combined_metadata,
            "provider": getattr(model, "mllm_compression_bridge").runtime_metadata(),
            "semantic_audit": semantic_audit.since(combined_semantic_mark),
            "position_audit": position_audit.summary_since(
                combined_position_mark
            ),
        }
    finally:
        for activation in reversed(combined):
            activation.uninstall()
    assert not hasattr(model, "mllm_compression_bridge")
    position_audit.close()

    assert set(report["cases"]) == {"tome", "visionzip", "pyramid_drop"}
    report_path = Path(
        os.environ.get(
            "LLAVA_COMPRESSION_SMOKE_REPORT",
            tmp_path / "llava_onevision_image_compression_forward.json",
        )
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    assert report_path.is_file()
