"""Concise requests expand through the same typed configuration and registry."""

from copy import deepcopy
from pathlib import Path

import pytest

from mllm_comp.config import config_to_dict, load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import ConfigError
from mllm_comp.pipeline.definition import resolve_compression_definition


def _raw(stage, *, backbone="qwen25_omni", extensions=()):
    return {
        "experiment": {"name": "concise"},
        "backbone": {"name": backbone},
        "pipeline": {"extensions": list(extensions), "stages": [stage]},
        "train": {"backend": "ms_swift", "compression_mode": "compressed"},
    }


@pytest.mark.parametrize("kind", ["attention", "ffn"])
def test_existing_operation_recipe_and_saved_snapshot_use_the_same_binding(kind):
    from mllm_comp.pipeline.stage import CompressionStageConfig

    algorithm = f"qwen3_vl_{kind}_replacement"
    legacy_stage = {
        "name": "replace", "plugin": algorithm, "placement": "llm_level",
        "execution_phase": "prepare", "method": "operation_compression",
        "object_family": "module_unit", "modalities": ["text", "image"],
        "operation": {"type": "replace"},
        "schedule": {"targets": [{"site": "llm.layer[0]." + ("attn" if kind == "attention" else "ffn")}]},
        "config": {"strategy": "attn_identity" if kind == "attention" else "ffn_identity"},
    }
    recipe = _raw(legacy_stage, backbone="qwen3_vl")
    # This is the previously serialized shape, including null v2 identities
    # and the old schema's independent coupling default.
    saved = experiment_config_from_dict({"backbone": {"name": "qwen3_vl"}})
    saved.pipeline.stages = [CompressionStageConfig.from_dict(legacy_stage)]
    snapshot = config_to_dict(saved)
    for raw in (recipe, snapshot):
        migrated = experiment_config_from_dict(raw)
        definition = resolve_compression_definition(migrated, compression_mode="compressed")
        stage = definition.stages[0]
        assert stage.algorithm == algorithm
        assert stage.binding == f"{algorithm}.shared_llm.qwen3_vl"
        assert stage.coupling_policy.value == "synchronized"
        assert definition.projections[stage.name].algorithm_config == legacy_stage["config"]
        assert stage.schedule == legacy_stage["schedule"]
    assert "algorithm" not in legacy_stage
    assert snapshot["pipeline"]["stages"][0]["algorithm"] is None


def test_algorithm_request_matches_explicit_config_without_mutating_source():
    short = {
        "name": "reduce_audio", "algorithm": "segmentwise_topk_audio",
        "execution_phase": "forward", "config": {"keep_ratio": 0.5},
    }
    raw = _raw(short)
    before = deepcopy(raw)
    expanded = experiment_config_from_dict(raw)
    explicit = {
        **short,
        "plugin": "segmentwise_topk_audio",
        "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni",
        "placement": "projector_level",
        "method": "token_compression",
        "object_family": "tensor_slot",
        "target_modalities": ["audio"],
        "conditioning_modalities": [],
        "preserved_modalities": ["image", "text", "video"],
        "coupling_policy": "independent",
        "media_alignment": "temporal",
    }
    assert raw == before
    assert config_to_dict(expanded) == config_to_dict(experiment_config_from_dict(_raw(explicit)))
    definition = resolve_compression_definition(expanded, compression_mode="compressed", interface="train")
    assert definition.projections["reduce_audio"].algorithm_config["keep_ratio"] == 0.5


def test_exact_operation_binding_supplies_identity_family_and_unique_phase():
    config = experiment_config_from_dict(_raw({
        "name": "prepare_layers", "binding": "structured_llm_layer_pruning.audio.qwen25_omni",
    }))
    stage = config.pipeline.stages[0]
    assert stage.algorithm == stage.plugin == "structured_llm_layer_pruning"
    assert stage.variant == "structured_llm_layer_pruning.audio"
    assert stage.method.value == "operation_compression"
    assert stage.object_family.value == "module_unit"
    assert stage.execution_phase.value == "prepare"


@pytest.mark.parametrize("stage,backbone,message", [
    ({"name": "reduce", "algorithm": "segmentwise_topk_audio"}, "qwen25_omni", "execution_phase.*forward.*generate"),
    ({"name": "reduce", "algorithm": "visionzip", "execution_phase": "forward"}, "qwen25_vl", "visionzip.image.*visionzip.video"),
    ({"name": "reduce", "binding": "tome.image.qwen25_vl", "execution_phase": "forward"}, "qwen25_vl", "placement.*encoder_level.*projector_level"),
    ({"name": "reduce", "binding": "segmentwise_topk_audio.audio.qwen25_omni", "execution_phase": "forward"}, "qwen25_vl", "targets backbone 'qwen25_omni'"),
])
def test_ambiguous_or_wrong_backbone_choices_are_not_guessed(stage, backbone, message):
    with pytest.raises(ConfigError, match=message):
        experiment_config_from_dict(_raw(stage, backbone=backbone))


def test_explicit_modality_selects_the_real_variant():
    config = experiment_config_from_dict(_raw({
        "name": "reduce_image", "algorithm": "visionzip", "target_modalities": ["image"],
        "execution_phase": "forward",
    }, backbone="qwen25_vl"))
    assert config.pipeline.stages[0].binding == "visionzip.image.qwen25_vl"


def test_explicit_conflicting_fields_are_preserved_for_validation():
    config = experiment_config_from_dict(_raw({
        "name": "reduce_audio", "algorithm": "segmentwise_topk_audio",
        "plugin": "different_plugin", "execution_phase": "forward",
    }))
    assert config.pipeline.stages[0].plugin == "different_plugin"
    with pytest.raises(ConfigError, match="plugin 'different_plugin' does not match"):
        resolve_compression_definition(config, compression_mode="compressed", interface="train")


def test_compound_recipe_still_requires_an_explicit_placement():
    with pytest.raises(ConfigError, match="placement"):
        experiment_config_from_dict(_raw({
            "name": "compound", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
            "plugin": "tome", "execution_phase": "forward",
            "bindings": [{"binding": "tome.image.qwen25_vl"}],
        }, backbone="qwen25_vl"))


def test_extension_registry_is_shared_for_completion_and_not_reloaded_for_snapshot(monkeypatch):
    from examples.compression_extensions import first_k_audio as research

    register = research.register
    calls = []

    def counted_register(registry):
        calls.append(registry)
        register(registry)

    monkeypatch.setattr(research, "register", counted_register)
    raw = _raw({
        "name": "first", "algorithm": "researcher_first_k_audio", "execution_phase": "forward",
    }, extensions=["examples.compression_extensions.first_k_audio:register"])
    raw["pipeline"]["stages"].append({
        "name": "second", "binding": "researcher_first_k_audio.audio.qwen25_omni",
        "execution_phase": "forward",
    })
    config = experiment_config_from_dict(raw)
    assert len(calls) == 1
    assert {stage.variant for stage in config.pipeline.stages} == {"researcher_first_k_audio.audio"}
    snapshot = config_to_dict(config)
    assert config_to_dict(experiment_config_from_dict(snapshot)) == snapshot
    assert len(calls) == 1


def test_concise_yaml_example_uses_the_public_loader():
    path = Path(__file__).resolve().parents[2] / "examples/compression/qwen25_omni_audio_minimal.yaml"
    config = load_config(path)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    assert definition.projections["reduce_audio"].binding_names == (
        "segmentwise_topk_audio.audio.qwen25_omni",
    )
