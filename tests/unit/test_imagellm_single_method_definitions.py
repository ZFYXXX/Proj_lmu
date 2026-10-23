from __future__ import annotations

from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import CompressionMode
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.pipeline.stage import CompressionStageConfig


_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES = (
    _ROOT / "configs/eval/lmms_eval/qwen25_vl_imagellm_tome_forward_probe.yaml",
    _ROOT / "configs/eval/lmms_eval/qwen25_vl_imagellm_visionzip_forward_probe.yaml",
)


@pytest.mark.parametrize("path", _EXAMPLES, ids=lambda path: path.stem)
def test_imagellm_single_method_definition_is_valid(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("MLLM_COMP_IMAGELLM_PROBE_ROOT", "/runs/imagellm")
    config = load_config(path)

    assert len(config.pipeline.stages) == 1
    definition = resolve_compression_definition(
        config,
        compression_mode=config.eval.compression_mode,
        interface="eval",
        runtime_batch_size=config.eval.runtime_batch_size,
    )
    assert len(definition.stages) == 1


def test_imagellm_train_definition_accepts_train_step_verified_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("MLLM_COMP_IMAGELLM_PROBE_ROOT", "/runs/imagellm")
    config = load_config(_EXAMPLES[0])
    config.train.backend = "ms_swift"
    config.train.compression_mode = CompressionMode.COMPRESSED

    definition = resolve_compression_definition(
        config,
        compression_mode=config.train.compression_mode,
        interface="train",
    )

    assert len(definition.stages) == 1
    assert definition.stages[0].algorithm == "tome"


def test_d2pruner_train_definition_allows_unverified_research(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("MLLM_COMP_IMAGELLM_PROBE_ROOT", "/runs/imagellm")
    config = load_config(_EXAMPLES[0])
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("d2pruner.image.qwen25_vl")
    variant = registry.variant(binding.variant)
    algorithm = registry.algorithm(variant.algorithm)
    config.pipeline.stages = [
        CompressionStageConfig(
            name="d2pruner-image-train",
            placement=binding.ordered_targets()[0].placement,
            execution_phase="forward",
            plugin=algorithm.name,
            algorithm=algorithm.name,
            variant=variant.name,
            binding=binding.name,
            method=algorithm.method,
            object_family="tensor_slot",
            object_subtype="llm_hidden_state",
            modalities=set(variant.target_modalities),
            conditioning_modalities=set(variant.conditioning_modalities),
            preserved_modalities=set(variant.preserved_modalities),
            coupling_policy=variant.coupling_policy,
            media_alignment=variant.media_alignment,
            config=dict(variant.config_defaults),
        )
    ]
    config.train.backend = "ms_swift"
    config.train.compression_mode = CompressionMode.COMPRESSED

    definition = resolve_compression_definition(
        config,
        compression_mode=config.train.compression_mode,
        interface="train",
    )
    assert definition.stages[0].algorithm == "d2pruner"
    assert all(not unit.inference_active for unit in registry.support.find(algorithm="d2pruner"))


def test_train_step_verified_image_stage_builds_compressed_swift_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_VL_MODEL_PATH", "/models/qwen25-vl")
    monkeypatch.setenv("MLLM_COMP_IMAGELLM_PROBE_ROOT", "/runs/imagellm")
    config = load_config(_EXAMPLES[0])
    config.train.backend = "ms_swift"
    config.train.compression_mode = CompressionMode.COMPRESSED
    config.train.max_steps = 1
    config.train.args = {
        "datasets": ["/datasets/train.jsonl"],
        "trainable_branch": "language_model",
        "tuner_type": "full",
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }

    plan = MSSwiftTrainAdapter().build_launch_plan(config)

    assert plan.compression_training["enabled"] is True
    assert plan.compression_training["executor_family"] == "image_multimodal"
    assert plan.compression_training["backbone_name"] == "qwen25_vl"
    assert plan.environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"]

    from mllm_comp.core.support import SupportRegistry
    from mllm_comp.run_manifest import build_run_manifest

    registry = builtin_multimodal_compression_contracts()
    monkeypatch.setattr(registry, "support", SupportRegistry())
    resolve_compression_definition(config, compression_mode="compressed", interface="train")
    research_plan = MSSwiftTrainAdapter().build_launch_plan(config)
    assert research_plan.compression_training["enabled"] is True
    record = build_run_manifest(config).compression_stages[0]
    assert record.support_status == "undeclared"
    assert record.evidence == []

    config.pipeline.stages[0].execution_phase = "generate"
    with pytest.raises(BackendIntegrationError, match="execution_phase=forward"):
        MSSwiftTrainAdapter().build_launch_plan(config)
