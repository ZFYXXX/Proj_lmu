"""Selected Variant parameters reach the existing physical position rules."""

from dataclasses import replace

import pytest

from mllm_comp.compressors.image.decoder_registration import create_registered_decoder_selection
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.pipeline.stage import StageBindingConfig
from mllm_comp.pipeline.v2_validation import V2ContractValidationError, validate_v2_stage_contract
from mllm_comp import run_manifest


def _research_variant(algorithm, defaults, *, modality="image", validator=None):
    registry = builtin_multimodal_compression_contracts().clone()
    base = registry.variant(f"{algorithm}.{modality}")
    variant = replace(base, name=f"research_{algorithm}.{modality}",
                      config_defaults={**base.config_defaults, **defaults},
                      config_validator=validator)
    binding = replace(registry.binding(f"{algorithm}.{modality}.qwen3_vl"),
                      name=f"{variant.name}.qwen3_vl", variant=variant.name)
    registry.register_variant(variant)
    registry.register_binding(binding)
    config = experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl", "model_id": "fixture"},
        "pipeline": {"stages": [{
            "name": "research-position", "algorithm": algorithm,
            "variant": base.name, "binding": f"{base.name}.qwen3_vl",
            "placement": "llm_level" if modality == "image" else "projector_level",
            "execution_phase": "forward",
        }]},
    })
    stage = config.pipeline.stages[0]
    stage.variant, stage.binding = variant.name, binding.name
    return registry, config, stage


def _image_method(registry, stage):
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen3_vl")
    method, _, _, _ = create_registered_decoder_selection(
        stage=stage, binding=projection.bindings[0], contract_projection=projection,
        backbone="qwen3_vl", execution_phase="forward", binding_config=projection.selections[0],
    )
    method.resolve_depth(16)
    return projection, method


@pytest.mark.parametrize("algorithm,defaults,layers,edge,scores,indices", [
    ("fastv", {"prune_layer": 3}, [3], "after", [2], [4]),
    ("pyramid_drop", {"num_stages": 3, "layer_indices": (1, 3)}, [2, 4], "after", [1, 3], [3, 5]),
    ("btp", {"prune_before_layers": (2, 4, 6, 8)}, [2, 4, 6, 8], "before", [0, 2, 4], [2, 4, 6, 8]),
    ("halfv", {"ivr_layer_index": 1, "ssr_layer_index": 4}, [2, 5], "before", [0, 3], [2, 5]),
    ("transprune", {"accumulating_layers": (2, 3, 4), "prune_after_layers": (2, 4), "retain_ratios": (0.75, 0.5)}, [2, 4], "after", [1, 3], [3, 5]),
])
def test_registered_image_variant_defaults_match_method_positions_and_manifest(
    monkeypatch, algorithm, defaults, layers, edge, scores, indices,
):
    registry, config, stage = _research_variant(algorithm, defaults)
    # A new Variant name must also work at the method's unmodified boundary.
    _image_method(registry, stage)
    stage.operation = {"target_site": "llm_decoder_layer_output_image_tokens",
                       "layers": layers, "layer_edge": edge}
    projection, method = _image_method(registry, stage)
    target = projection.bindings[0].targets[0]
    assert list(method.score_layers) == scores
    if algorithm == "transprune":
        assert method.spec.retain_ratios == (0.75, 0.5)
    assert [point.index for point in method.positions] == indices
    assert target.metadata["application_positions"] == [point.as_dict() for point in method.positions]
    assert target.metadata.get("score_layers", [target.metadata.get("score_layer")]) == scores
    monkeypatch.setattr(run_manifest, "compression_contracts", lambda entries: registry)
    record = run_manifest.build_run_manifest(config).compression_stages[0]
    key = f"{stage.binding}:{target.name}"
    assert record.metadata["application_positions"][key] == [point.as_dict() for point in method.positions]
    assert record.metadata["score_layers"][key] == (scores[0] if algorithm == "fastv" else scores)
    assert stage.config == {}


@pytest.mark.parametrize("criterion,private,binding_override,expected", [
    ({}, {}, {}, 3),
    ({"prune_layer": 1}, {}, {}, 1),
    ({"prune_layer": 1}, {"prune_layer": 4}, {}, 4),
    ({"prune_layer": 1}, {"prune_layer": 4}, {"prune_layer": 5}, 5),
])
def test_image_position_defaults_preserve_legacy_and_binding_precedence(
    criterion, private, binding_override, expected,
):
    registry, _, stage = _research_variant("fastv", {"prune_layer": 3})
    stage.criterion, stage.config = criterion, private
    stage.bindings = [StageBindingConfig(binding=stage.binding, config=binding_override)]
    stage.binding = None
    stage.operation = {"target_site": "llm_decoder_layer_output_image_tokens", "layers": [5]}
    projection, method = _image_method(registry, stage)
    assert method.spec.prune_layer == expected
    assert projection.bindings[0].targets[0].metadata["score_layer"] == expected - 1
    assert stage.config == private and stage.criterion == criterion


def test_unknown_image_algorithm_requires_its_own_position_resolver():
    registry, _, stage = _research_variant("fastv", {"prune_layer": 3})
    algorithm = replace(registry.algorithm("fastv"), name="research_unknown")
    variant = replace(registry.variant(stage.variant), name="research_unknown.image", algorithm=algorithm.name)
    binding = replace(registry.binding(stage.binding), name="research_unknown.image.qwen3_vl", variant=variant.name)
    registry.register_algorithm(algorithm)
    registry.register_variant(variant)
    registry.register_binding(binding)
    stage.algorithm = stage.plugin = algorithm.name
    stage.variant, stage.binding = variant.name, binding.name
    with pytest.raises(V2ContractValidationError, match="provide a method-specific position_resolver"):
        validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen3_vl")


def test_video_position_projection_uses_the_selected_variant_without_promoting_executor(monkeypatch):
    from mllm_comp.backbones.prunevid_video import PruneVidVideoExecutor
    from mllm_comp.compressors.video.config import resolve_canonical_video_config
    from mllm_comp.core.errors import BackendIntegrationError

    # Video validators own their defaults and retain the canonical rule checks.
    def validate(raw):
        return resolve_canonical_video_config("prunevid", {"vtp_selected_layer": 2, **raw})

    registry, config, stage = _research_variant(
        "prunevid", {"vtp_selected_layer": 2}, modality="video", validator=validate,
    )
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen3_vl")
    target = next(target for target in projection.bindings[0].targets if target.name == "vtp_prefill_prune")
    assert projection.algorithm_config["vtp_selected_layer"] == 2
    assert target.metadata["score_layers"] == [2]
    assert target.metadata["application_positions"] == [{"boundary": "inter_layer", "index": 3}]
    monkeypatch.setattr(run_manifest, "compression_contracts", lambda entries: registry)
    record = run_manifest.build_run_manifest(config).compression_stages[0]
    assert record.metadata["score_layers"][f"{stage.binding}:vtp_prefill_prune"] == [2]
    with pytest.raises(BackendIntegrationError, match="requires binding"):
        PruneVidVideoExecutor(backbone="qwen3_vl", model=object(), stage=stage,
                              binding=projection.bindings[0], contract_projection=projection,
                              execution_phase="forward")
