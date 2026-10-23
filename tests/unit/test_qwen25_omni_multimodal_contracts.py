from __future__ import annotations

from dataclasses import replace
from itertools import permutations
from types import SimpleNamespace

import pytest

from mllm_comp.core.errors import InvariantError

from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.support import SupportStatus
from mllm_comp.core.vocabulary import (
    CompressionMethod, CompressionMode, ExecutionPhase, Modality, OperationKind,
)
from mllm_comp.registry.catalog import CapabilityKind, CapabilityStatus, get_capability_spec
from mllm_comp.backbones.qwen25_omni.v2_compression import (
    resolve_qwen25_omni_compression_execution,
    resolve_qwen25_omni_stage_execution,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.pipeline.v2_validation import (
    V2ContractValidationError,
    validate_v2_stage_contract,
)


_ACTIVE = {
    "segmentwise_topk_audio",
    "local_temporal_bipartite_merging_audio",
    "similarity_attention_merge_prune_audio",
    "adjacent_token_merging_audio",
    "encoder_attention_topk_audio",
    "progressive_attention_pruning_audio",
    "audio_pyramid_attention_pruning",
    "structured_llm_layer_pruning",
}
_TARGET = {
    "fastslm_hta_audio",
    "uniform_average_pooling_audio",
    "uniform_sampling_audio",
    "unsupervised_segmentation_audio",
    "fastadasp_weighted_token_merging",
    "headrouter_dynamic_audio_pruning",
}


def test_all_fourteen_audio_methods_have_exact_v2_layers() -> None:
    registry = builtin_multimodal_compression_contracts()

    for algorithm_name in _ACTIVE | _TARGET:
        algorithm = registry.algorithm(algorithm_name)
        variant = registry.variant(f"{algorithm_name}.audio")
        binding = registry.binding(f"{algorithm_name}.audio.qwen25_omni")

        assert algorithm.kernel().name == algorithm_name
        assert variant.algorithm == algorithm_name
        assert variant.target_modalities == {Modality.AUDIO}
        assert binding.backbone == "qwen25_omni"
        assert binding.variant == variant.name
        assert binding.target_modalities == {Modality.AUDIO}
        assert binding.executor_family in {
            "projector_token",
            "encoder_token",
            "llm_token",
            "llm_operation",
        }
        assert binding.metadata["required_runtime_context"]
        assert callable(binding.executor_factory)


def test_audio_support_units_are_phase_exact_and_preserve_status() -> None:
    registry = builtin_multimodal_compression_contracts()
    units = registry.support.find(backbone="qwen25_omni", target_modality="audio")

    assert len(units) == 27
    assert sum(unit.status is SupportStatus.ACTIVE for unit in units) == 15
    assert sum(unit.status is SupportStatus.DECLARED for unit in units) == 12
    for unit in units:
        if unit.identity.algorithm in _ACTIVE:
            assert unit.status is SupportStatus.ACTIVE
            assert not unit.missing_activation_evidence
        else:
            assert unit.identity.algorithm in _TARGET
            assert unit.status is SupportStatus.DECLARED
            assert unit.metadata["capability_status"] == "target"
            assert unit.inference_active is False

    spade = registry.support.find(algorithm="structured_llm_layer_pruning")
    assert len(spade) == 1
    assert spade[0].identity.phase is ExecutionPhase.PREPARE


def test_structural_pruning_declares_module_operations_without_token_units() -> None:
    registry = builtin_multimodal_compression_contracts()
    name = "structured_llm_layer_pruning"
    algorithm = registry.algorithm(name)
    assert algorithm.method is CompressionMethod.OPERATION_COMPRESSION
    assert algorithm.operations == {OperationKind.PRUNE}
    assert registry.variant(f"{name}.audio").token_unit is None
    spec = get_capability_spec(CapabilityKind.COMPRESSOR, name)
    assert spec.metadata["method"] == "operation_compression"
    assert spec.metadata["operations"] == ["prune"]


@pytest.mark.parametrize(
    ("algorithm", "phase", "wrong_method"),
    [
        ("audio_pyramid_attention_pruning", "generate", CompressionMethod.OPERATION_COMPRESSION),
        ("structured_llm_layer_pruning", "prepare", CompressionMethod.TOKEN_COMPRESSION),
    ],
)
def test_stage_method_must_match_the_selected_algorithm(
    algorithm, phase, wrong_method
) -> None:
    registry = builtin_multimodal_compression_contracts()
    stage = _representative_stage(algorithm, phase)
    validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")
    stage.method = wrong_method
    with pytest.raises(V2ContractValidationError, match="does not match algorithm"):
        validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")


def test_qwen_audio_catalog_prefers_v2_identity_without_promoting_targets() -> None:
    active = get_capability_spec(
        CapabilityKind.COMPRESSOR,
        "segmentwise_topk_audio",
    )
    target = get_capability_spec(CapabilityKind.COMPRESSOR, "fastslm_hta_audio")

    assert active.status is CapabilityStatus.ACTIVE
    assert active.activation_phase == "multimodal_compression_v2"
    assert active.metadata["contract_version"] == 2
    assert active.metadata["backbones"] == ["qwen25_omni"]
    assert active.metadata["modalities"] == ["audio"]
    assert active.metadata["executor_family"] == "projector_token"
    assert active.interfaces == {"analysis", "eval", "train"}
    assert target.status is CapabilityStatus.TARGET
    assert {
        unit["status"] for unit in target.metadata["support_units"]
    } == {"declared"}


def test_v2_execution_contract_builds_all_existing_plugins_specs_and_strategies() -> None:
    registry = builtin_multimodal_compression_contracts()
    for algorithm_name in _ACTIVE | _TARGET:
        binding = registry.binding(f"{algorithm_name}.audio.qwen25_omni")
        target = binding.targets[0]
        stage = CompressionStageConfig.from_dict(
            {
                "name": f"test_{algorithm_name}",
                "placement": target.placement.value,
                "execution_phase": sorted(
                    binding.supported_phases,
                    key=lambda item: item.value,
                )[0].value,
                "plugin": algorithm_name,
                "algorithm": algorithm_name,
                "variant": f"{algorithm_name}.audio",
                "binding": binding.name,
                "method": registry.algorithm(algorithm_name).method.value,
                "object_family": (
                    "module_unit"
                    if algorithm_name == "structured_llm_layer_pruning"
                    else "tensor_slot"
                ),
                "object_subtype": target.target_site,
                "modalities": ["audio"],
                "preserved_modalities": ["text", "image", "video"],
                "media_alignment": registry.variant(
                    f"{algorithm_name}.audio"
                ).media_alignment.value,
                "operation": {"target_site": target.target_site},
            }
        )
        execution = resolve_qwen25_omni_compression_execution(
            stage=stage,
            binding=binding,
        )
        plugin_config = stage.plugin_config()
        plugin_config["_stage_name"] = stage.name
        plugin_config["_execution_phase"] = stage.execution_phase.value

        assert execution.create_plugin().name == algorithm_name
        spec = execution.create_spec(plugin_config)
        assert execution.create_strategy(spec) is not None


def _hta_window_variant():
    registry = builtin_multimodal_compression_contracts().clone()
    original = registry.variant("fastslm_hta_audio.audio")
    variant = replace(
        original, name="research_window.audio",
        config_defaults={**original.config_defaults, "window_tokens": 150,
                         "stage_query_tokens": 4, "detail_recovery_depth": 0},
    )
    binding = replace(
        registry.binding("fastslm_hta_audio.audio.qwen25_omni"),
        name="research_window.audio.qwen25_omni", variant=variant.name,
    )
    registry.register_variant(variant)
    registry.register_binding(binding)
    stage = _representative_stage("fastslm_hta_audio", "forward")
    stage.variant, stage.binding = variant.name, binding.name
    # Valid with the registered window, but not the original 750-token window.
    stage.schedule = {"keep_ratio": 0.2}
    stage.config = {}
    return registry, stage


@pytest.mark.parametrize("explicit_window", [None, 100])
def test_audio_variant_defaults_reach_position_validation_and_plan(monkeypatch, explicit_window):
    from mllm_comp.compressors.base import CompressionContext
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.core.serialization import to_jsonable
    from mllm_comp.core.state import ForwardState, TokenSpan
    from mllm_comp.core.vocabulary import Placement
    from mllm_comp.pipeline import definition as definition_module
    from mllm_comp import run_manifest

    registry, stage = _hta_window_variant()
    if explicit_window is not None:
        stage.config["window_tokens"] = explicit_window
    expected_window = explicit_window or 150
    stage.placement = Placement.LLM_LEVEL
    stage.operation.update(target_site="llm_decoder_layer_output_audio_tokens", layers=[2])
    config = experiment_config_from_dict({
        "backbone": {"name": "qwen25_omni", "model_id": "Qwen/fixture"},
        "pipeline": {"stages": [to_jsonable(stage)]},
    })
    monkeypatch.setattr(definition_module, "compression_contracts", lambda entries: registry)
    monkeypatch.setattr(run_manifest, "compression_contracts", lambda entries: registry)
    definition = definition_module.resolve_compression_definition(
        config, compression_mode="compressed", interface="train",
    )
    execution = resolve_qwen25_omni_stage_execution(stage, registry=registry)
    assert definition.projections[stage.name].bindings[0].executor_family == "llm_token"
    assert execution.binding.executor_family == "llm_token"
    assert execution.create_spec(stage.plugin_config()).window_tokens == expected_window
    payload, spec = execution.prepare_config(stage.plugin_config())
    strategy = execution.create_strategy(spec, operation=stage.operation, stage=stage)
    plan = execution.create_plugin().plan(
        state=ForwardState(modality_token_spans=[TokenSpan(0, 0, 300, modality="audio")]),
        context=CompressionContext(phase="train", model=None), config=payload,
    )
    assert plan.token_actions[0].metadata["keep_count"] == 60
    assert plan.token_actions[0].metadata["window_tokens"] == expected_window
    assert strategy.rule.spec.window_tokens == expected_window
    assert strategy.output_length(300) == 60
    record = run_manifest.build_run_manifest(config).compression_stages[0]
    assert record.metadata["algorithm_config"]["window_tokens"] == expected_window
    assert record.target_sites == ["llm_decoder_layer_output_audio_tokens"]
    assert stage.config == ({} if explicit_window is None else {"window_tokens": 100})


def test_audio_variant_defaults_build_the_controller_hta_module(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_omni import audio_compression
    from mllm_comp.compressors.base import CompressionContext
    from mllm_comp.core.state import ForwardState, TokenSpan

    registry, stage = _hta_window_variant()
    monkeypatch.setattr(audio_compression, "_require_verified_transformers", lambda model: None)
    tower = torch.nn.Module()
    tower.proj = torch.nn.Linear(8, 8)
    tower.layers = []
    tower.config = SimpleNamespace(d_model=8, encoder_attention_heads=2)
    thinker = torch.nn.Module()
    thinker.audio_tower = tower
    controller = audio_compression.Qwen25OmniAudioCompressionController(
        SimpleNamespace(thinker=thinker), stage,
        execution=resolve_qwen25_omni_stage_execution(stage, registry=registry),
    )
    module = controller.strategy_runtime_module
    assert module is getattr(thinker, controller.strategy_runtime_module_name)
    assert module.window_tokens == 150
    values = torch.randn(300, 8, requires_grad=True)
    plan = controller.plugin.plan(
        state=ForwardState(modality_token_spans=[TokenSpan(0, 0, 300, modality="audio")]),
        context=CompressionContext(phase="train", model=None), config=controller.plugin_config,
    )
    keep_count = plan.token_actions[0].metadata["keep_count"]
    output, _, details = controller.strategy.compress(
        values, scores=None, keep_count=keep_count, item_index=0,
    )
    assert output.shape == (60, 8)
    assert details["window_lengths"] == [150, 150]
    output.square().mean().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in module.parameters())


@pytest.mark.parametrize("algorithm", ["uniform_sampling_audio", "uniform_average_pooling_audio"])
def test_audio_budget_sections_do_not_gain_conflicting_spec_defaults(algorithm):
    stage = _representative_stage(algorithm, "forward")
    stage.schedule = {"keep_ratio": 0.25}
    execution = resolve_qwen25_omni_stage_execution(stage)
    payload, spec = execution.prepare_config(stage.plugin_config())
    assert spec.compression_factor == 4
    assert "compression_factor" not in payload["config"]
    assert "keep_ratio" not in payload["config"]


class _FakeActivation:
    def __init__(self, execution: object) -> None:
        self.execution = execution
        self.installed = False

    def install(self) -> None:
        self.installed = True

    def uninstall(self) -> None:
        self.installed = False

    def runtime_metadata(self) -> dict[str, object]:
        return {
            "algorithm": self.execution.name,  # type: ignore[attr-defined]
            "executor_family": self.execution.executor_family,  # type: ignore[attr-defined]
        }


@pytest.mark.parametrize("boundary", ["post_layer", "post_attention_pre_ffn"])
def test_headrouter_metadata_uses_actual_boundary(monkeypatch, boundary: str) -> None:
    from mllm_comp.backbones.qwen25_omni import llm_token_compression as runtime

    monkeypatch.setattr(runtime, "_require_verified_transformers", lambda model: None)
    model = SimpleNamespace(thinker=SimpleNamespace(
        forward=lambda **kwargs: None,
        model=SimpleNamespace(layers=[SimpleNamespace(forward=lambda **kwargs: None) for _ in range(3)]),
    ))
    algorithm = "headrouter_dynamic_audio_pruning"
    site = (
        "llm_decoder_layer_output_audio_tokens" if boundary == "post_layer"
        else "llm_decoder_layer_post_attention_audio_tokens"
    )
    stage = CompressionStageConfig.from_dict({
        "name": "headrouter-boundary",
        "placement": "llm_level",
        "execution_phase": "generate",
        "plugin": algorithm,
        "algorithm": algorithm,
        "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni",
        "target_modalities": ["audio"],
        "operation": {"target_site": site},
    })
    metadata = runtime.Qwen25OmniLLMTokenCompressionController(model, stage).runtime_metadata()
    assert metadata["target_site"] == site
    assert metadata["compression_boundary"] == boundary
    assert ("compression_boundary_after_layer_zero_based" in metadata) == (boundary == "post_layer")
    assert metadata["qk_probe_layer_zero_based"] == 1


def _representative_stage(algorithm: str, phase: str) -> CompressionStageConfig:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding(f"{algorithm}.audio.qwen25_omni")
    variant = registry.variant(f"{algorithm}.audio")
    target = binding.targets[0]
    operation_type = {
        "segmentwise_topk_audio": "keep",
        "uniform_sampling_audio": "keep",
        "uniform_average_pooling_audio": "merge",
        "fastslm_hta_audio": "merge",
        "local_temporal_bipartite_merging_audio": "merge",
        "similarity_attention_merge_prune_audio": "merge_prune",
        "adjacent_token_merging_audio": "merge",
        "encoder_attention_topk_audio": "keep",
        "progressive_attention_pruning_audio": "keep",
        "audio_pyramid_attention_pruning": "keep",
        "structured_llm_layer_pruning": "prune",
    }[algorithm]
    return CompressionStageConfig.from_dict(
        {
            "name": f"stage_{algorithm}",
            "placement": target.placement.value,
            "execution_phase": phase,
            "plugin": algorithm,
            "algorithm": algorithm,
            "variant": variant.name,
            "binding": binding.name,
            "method": registry.algorithm(algorithm).method.value,
            "object_family": (
                "module_unit"
                if algorithm == "structured_llm_layer_pruning"
                else "tensor_slot"
            ),
            "object_subtype": (
                "transformer_layer"
                if algorithm == "structured_llm_layer_pruning"
                else target.target_site
            ),
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [
                modality.value for modality in variant.conditioning_modalities
            ],
            "preserved_modalities": [
                modality.value for modality in variant.preserved_modalities
            ],
            "coupling_policy": variant.coupling_policy.value,
            "media_alignment": variant.media_alignment.value,
            "operation": {
                "type": operation_type,
                "target_site": target.target_site,
                **(
                    {
                        "target_scope": "llm",
                        "target_kind": "layer",
                        "apply_mode": "structural_rewrite",
                        "shape_effect": "remove_module",
                    }
                    if algorithm == "structured_llm_layer_pruning"
                    else {}
                ),
            },
            "schedule": {"keep_ratio": 0.5, "merge_ratio": 0.1},
            "criterion": {"selection_source": "explicit"},
            "activation_scope": {"layers": [0]},
        }
    )


@pytest.mark.parametrize(
    ("algorithm", "module_name", "class_name"),
    [
        (
            "segmentwise_topk_audio",
            "mllm_comp.backbones.qwen25_omni.audio_compression",
            "Qwen25OmniAudioCompressionController",
        ),
        (
            "adjacent_token_merging_audio",
            "mllm_comp.backbones.qwen25_omni.encoder_compression",
            "Qwen25OmniEncoderCompressionController",
        ),
        (
            "audio_pyramid_attention_pruning",
            "mllm_comp.backbones.qwen25_omni.llm_token_compression",
            "Qwen25OmniLLMTokenCompressionController",
        ),
    ],
)
def test_representative_dynamic_paths_use_generic_v2_session(
    monkeypatch: pytest.MonkeyPatch,
    algorithm: str,
    module_name: str,
    class_name: str,
) -> None:
    module = __import__(module_name, fromlist=[class_name])
    created: list[_FakeActivation] = []

    def controller_factory(model: object, stage: object, **kwargs: object) -> object:
        del model, stage
        activation = _FakeActivation(kwargs["execution"])
        created.append(activation)
        return activation

    monkeypatch.setattr(module, class_name, controller_factory)
    stage = _representative_stage(algorithm, "generate")
    binding = builtin_multimodal_compression_contracts().binding(stage.binding or "")
    handles = {name: object() for name in binding.required_handles}
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(structural_handles=lambda model: handles),
    )
    runtime = SimpleNamespace(model=object(), metadata={})
    dispatcher = RuntimeCompressionDispatcher(
        contract_registry=builtin_multimodal_compression_contracts()
    )

    session = dispatcher.resolve(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.GENERATE,
    )

    assert session is not None
    assert created == []
    assert session.activations[0].binding == binding.name
    assert session.activations[0].metadata["executor_family"] == "v2_binding"
    session.install()
    assert len(created) == 1
    assert created[0].installed is True
    assert session.runtime_metadata()["activations"][0]["algorithm"] == algorithm


def test_spade_prepare_path_uses_v2_operation_activation() -> None:
    stage = _representative_stage("structured_llm_layer_pruning", "prepare")
    llm = SimpleNamespace(layers=[object(), object()])
    handles = {"llm": llm}

    class _OperationApplier:
        def apply(self, **kwargs: object) -> object:
            assert len(kwargs["actions"]) == 1  # type: ignore[arg-type]
            return SimpleNamespace(
                violations=(),
                metadata={"operation_count": 1},
            )

    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(structural_handles=lambda model: handles),
        operation_applier=_OperationApplier(),
    )
    runtime = SimpleNamespace(model=SimpleNamespace(training=False), metadata={})
    dispatcher = RuntimeCompressionDispatcher(
        contract_registry=builtin_multimodal_compression_contracts()
    )
    session = dispatcher.resolve(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.PREPARE,
    )

    assert session is not None
    assert session.activations[0].rollback_safe is False
    session.install()
    metadata = session.runtime_metadata()["activations"][0]
    assert metadata["algorithm"] == "structured_llm_layer_pruning"
    assert metadata["operation_count"] == 1
    assert metadata["irreversible"] is True


def test_research_position_reaches_executor_without_reusing_standard_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mllm_comp.backbones.qwen25_omni import llm_token_compression

    registry = builtin_multimodal_compression_contracts()
    stage = _representative_stage("audio_pyramid_attention_pruning", "generate")
    canonical = registry.binding(stage.binding)
    old_units = registry.support.find(algorithm=stage.algorithm)
    site = "llm_decoder_layer_post_attention_audio_tokens"
    stage.operation["target_site"] = site
    monkeypatch.setattr(
        llm_token_compression,
        "Qwen25OmniLLMTokenCompressionController",
        lambda model, stage, **kwargs: _FakeActivation(kwargs["execution"]),
    )
    session = RuntimeCompressionDispatcher(contract_registry=registry).resolve(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
        facade=SimpleNamespace(
            capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        ),
        runtime=SimpleNamespace(model=object(), metadata={}),
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.GENERATE,
    )
    session.install()
    record = session.activations[0]
    assert record.activation.execution.target_site == site
    assert record.activation.execution.compression_boundary == "post_attention_pre_ffn"
    assert record.metadata["support_status"] == "declared"
    assert record.metadata["development_runtime"] is True
    assert record.metadata["canonical_evidence_applicable"] is False
    assert record.metadata["resolved_target_sites"] == {"llm_token": site}
    assert registry.binding(stage.binding) is canonical
    assert canonical.targets[0].target_site == "llm_decoder_layer_output_audio_tokens"
    assert registry.support.find(algorithm=stage.algorithm) == old_units
    assert all(unit.status is SupportStatus.ACTIVE for unit in old_units)


@pytest.mark.parametrize("algorithm", [
    "segmentwise_topk_audio",
    "adjacent_token_merging_audio",
    "audio_pyramid_attention_pruning",
])
def test_unknown_research_position_fails_before_executor_creation(algorithm: str) -> None:
    stage = _representative_stage(algorithm, "generate")
    stage.operation["target_site"] = "nonexistent_boundary"
    with pytest.raises(V2ContractValidationError, match="nonexistent_boundary"):
        validate_v2_stage_contract(
            stage,
            registry=builtin_multimodal_compression_contracts(),
            backbone_name="qwen25_omni",
        )


@pytest.mark.parametrize("algorithm", [
    "segmentwise_topk_audio", "fastslm_hta_audio", "uniform_sampling_audio",
    "uniform_average_pooling_audio", "unsupervised_segmentation_audio",
    "local_temporal_bipartite_merging_audio", "similarity_attention_merge_prune_audio",
])
@pytest.mark.parametrize("operation_site, legacy_site", [
    (None, "audio_projector_output_post_projector"),
    ("audio_encoder_output_pre_projector", "audio_projector_output_post_projector"),
    ("audio_projector_output_post_projector", "audio_encoder_output_pre_projector"),
])
def test_projector_position_sources_agree_with_spec(algorithm, operation_site, legacy_site):
    registry = builtin_multimodal_compression_contracts()
    stage = CompressionStageConfig.from_dict({
        "name": "projector-position-sources",
        "placement": "projector_level",
        "execution_phase": "generate",
        "plugin": algorithm,
        "algorithm": algorithm,
        "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni",
        "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"],
        "media_alignment": "temporal",
        "operation": {} if operation_site is None else {"target_site": operation_site},
        "config": {"target_site": legacy_site},
    })
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=projection.bindings[0])
    spec = execution.create_spec(stage.plugin_config())
    assert spec.target_site == execution.target_site == (operation_site or legacy_site)


@pytest.mark.parametrize("operation, message", [
    ({"layers": [0, 1], "layer_edge": "before"}, "precedes scoring layer"),
    ({"layers": [3]}, "number of stages"),
    ({"layers": [2, 4]}, "outside"),
])
def test_encoder_invalid_application_is_rejected_before_execution(operation, message):
    from mllm_comp.backbones.qwen25_omni.encoder_compression import Qwen25OmniEncoderCompressionController

    stage = _representative_stage("adjacent_token_merging_audio", "generate")
    stage.activation_scope = {"layers": [1, 2]}
    stage.operation.update({"target_site": "audio_encoder_layer_output_post_block", **operation})
    model = SimpleNamespace(thinker=SimpleNamespace(audio_tower=SimpleNamespace(layers=[object()] * 4)))
    with pytest.raises((ValueError, InvariantError), match=message):
        Qwen25OmniEncoderCompressionController(model, stage)


def _temporal_stage(algorithm, *, llm):
    stage = _representative_stage(algorithm, "generate")
    stage.operation["type"] = "keep" if algorithm in {"uniform_sampling_audio", "segmentwise_topk_audio"} else "merge"
    stage.schedule = {"keep_ratio": 0.5} if algorithm in {"local_temporal_bipartite_merging_audio", "segmentwise_topk_audio"} else {"compression_factor": 2}
    if algorithm == "fastslm_hta_audio":
        stage.schedule = {"keep_ratio": 0.5}
        stage.config = {"window_tokens": 8, "output_query_tokens": 4}
        stage.criterion = {}
    if algorithm == "similarity_attention_merge_prune_audio":
        stage.operation["type"] = "merge_prune"
        stage.schedule = {"keep_ratio": 0.5}
    if llm:
        from mllm_comp.core.vocabulary import Placement
        stage.placement = Placement.LLM_LEVEL
        stage.operation.update(target_site="llm_decoder_layer_output_audio_tokens", layers=[0], layer_edge="before")
    return stage


@pytest.mark.parametrize("algorithm,score_source", [
    ("uniform_sampling_audio", None), ("uniform_average_pooling_audio", None),
    ("local_temporal_bipartite_merging_audio", None), ("fastslm_hta_audio", None),
    ("segmentwise_topk_audio", "uniform"), ("segmentwise_topk_audio", "random"),
])
@pytest.mark.parametrize("family", ["llm", "encoder"])
def test_temporal_rule_moves_to_layers_with_actual_contract(algorithm, score_source, family):
    stage = _temporal_stage(algorithm, llm=True)
    if score_source:
        stage.criterion = {"score_source": score_source}
    if family == "encoder":
        from mllm_comp.core.vocabulary import Placement
        stage.placement = Placement.ENCODER_LEVEL
        stage.operation["target_site"] = "audio_encoder_layer_output_post_block"
    registry = builtin_multimodal_compression_contracts()
    canonical = registry.binding(stage.binding)
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni", available_handles={"llm" if family == "llm" else "audio_encoder": object()})
    binding = projection.bindings[0]
    assert binding.targets[0].placement.value == f"{family}_level"
    assert binding.executor_family == f"{family}_token"
    assert binding.exclusive_runtime_slot == f"qwen25_omni.{family}_token"
    assert binding.metadata["position_changed"] is True
    assert registry.binding(stage.binding) is canonical
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), operation=stage.operation)
    assert strategy.requires_layer_input is False
    expected = 3 if algorithm == "uniform_sampling_audio" else 2
    if family == "llm":
        assert strategy.output_length(5) == expected
    else:
        assert strategy.output_lengths(0, [5], (0,)) == [expected]
    stage.operation.pop("layers")
    with pytest.raises(V2ContractValidationError, match="explicit operation.layers"):
        validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")


@pytest.mark.parametrize("both_llm", [False, True])
def test_moved_rule_conflicts_follow_actual_executor(both_llm, monkeypatch):
    from mllm_comp.backbones.qwen25_omni import audio_compression, llm_token_compression
    from mllm_comp.core.errors import BackendIntegrationError

    registry = builtin_multimodal_compression_contracts()
    stages = [_temporal_stage("uniform_sampling_audio", llm=True),
              _temporal_stage("uniform_average_pooling_audio", llm=both_llm)]
    for module, name in [(audio_compression, "Qwen25OmniAudioCompressionController"),
                         (llm_token_compression, "Qwen25OmniLLMTokenCompressionController")]:
        monkeypatch.setattr(module, name, lambda model, stage, **kwargs: _FakeActivation(kwargs["execution"]))
    dispatcher = RuntimeCompressionDispatcher(contract_registry=registry)
    kwargs = dict(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages)),
        facade=SimpleNamespace(capabilities=lambda: SimpleNamespace(name="qwen25_omni")),
        runtime=SimpleNamespace(model=object(), metadata={}), compression_mode=CompressionMode.COMPRESSED,
    )
    if both_llm:
        with pytest.raises(BackendIntegrationError, match="conflicts"):
            dispatcher.resolve(**kwargs)
    else:
        session = dispatcher.resolve(**kwargs)
        session.install()
        assert [item.activation.execution.executor_family for item in session.activations] == ["llm_token", "projector_token"]
        assert session.install_order == (stages[1].name, stages[0].name)
        session.uninstall()
        stages[1].depends_on = (stages[0].name,)
        with pytest.raises(BackendIntegrationError, match="dependency cycle"):
            dispatcher.resolve(**kwargs)


@pytest.mark.parametrize("config_order", list(permutations(range(3))))
def test_audio_stage_order_follows_resolved_positions(config_order, monkeypatch):
    from mllm_comp.backbones.qwen25_omni import (
        audio_compression, encoder_compression, llm_token_compression,
    )
    from mllm_comp.core.errors import BackendIntegrationError
    from mllm_comp.core.vocabulary import Placement

    encoder = _temporal_stage("uniform_average_pooling_audio", llm=True)
    encoder.name = "encoder"
    encoder.placement = Placement.ENCODER_LEVEL
    encoder.operation["target_site"] = "audio_encoder_layer_output_post_block"
    projector = _representative_stage("encoder_attention_topk_audio", "generate")
    projector.name = "projector"
    projector.placement = Placement.PROJECTOR_LEVEL
    projector.operation["target_site"] = "audio_projector_output_post_projector"
    llm = _temporal_stage("uniform_sampling_audio", llm=True)
    llm.name = "llm"
    ordered = [encoder, projector, llm]
    for module, name in [
        (audio_compression, "Qwen25OmniAudioCompressionController"),
        (encoder_compression, "Qwen25OmniEncoderCompressionController"),
        (llm_token_compression, "Qwen25OmniLLMTokenCompressionController"),
    ]:
        monkeypatch.setattr(module, name, lambda model, stage, **kwargs: _FakeActivation(kwargs["execution"]))
    dispatcher = RuntimeCompressionDispatcher(contract_registry=builtin_multimodal_compression_contracts())
    kwargs = dict(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[ordered[i] for i in config_order])),
        facade=SimpleNamespace(capabilities=lambda: SimpleNamespace(name="qwen25_omni")),
        runtime=SimpleNamespace(model=object(), metadata={}),
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.GENERATE,
    )
    session = dispatcher.resolve(**kwargs)
    assert session.install_order == ("encoder", "projector", "llm")
    session.install()
    assert {item.name: item.activation.execution.executor_family for item in session.activations} == {
        "encoder": "encoder_token", "projector": "projector_token", "llm": "llm_token",
    }
    session.uninstall()
    # A contradictory explicit dependency must fail before any installation.
    encoder.depends_on = ("llm",)
    with pytest.raises(BackendIntegrationError, match="dependency cycle"):
        dispatcher.resolve(**kwargs)
    assert all(not item.activation.installed for item in session.activations)


@pytest.mark.parametrize("algorithm", ["uniform_sampling_audio", "uniform_average_pooling_audio"])
@pytest.mark.parametrize("feature_length,factor", [(62, 2), (4, 1), (4, 2)])
def test_temporal_encoder_plan_uses_chunk_and_pooler_budget(algorithm, feature_length, factor):
    from mllm_comp.compressors.base import CompressionContext
    from mllm_comp.core.errors import InvariantError
    from mllm_comp.core.state import ForwardState, TokenSpan
    from mllm_comp.core.vocabulary import Placement

    stage = _temporal_stage(algorithm, llm=True)
    stage.placement = Placement.ENCODER_LEVEL
    stage.schedule = {"compression_factor": factor}
    stage.operation.update(target_site="audio_encoder_layer_output_post_block", layers=[0, 2])
    binding = builtin_multimodal_compression_contracts().binding(stage.binding)
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), operation=stage.operation)
    state = ForwardState(audio_feature_lengths=[feature_length], modality_token_spans=[
        TokenSpan(0, 1, feature_length // 4 + 1, modality="audio")
    ])
    plan = execution.create_plugin().plan(state=state, context=CompressionContext(phase="eval", model=None), config=stage.plugin_config())
    if feature_length == 4 and factor == 2:
        with pytest.raises(InvariantError, match="no pooled audio token"):
            strategy.adapt_plan(plan, state=state, audio_tower=SimpleNamespace(n_window=16))
        return
    plan = strategy.adapt_plan(plan, state=state, audio_tower=SimpleNamespace(n_window=16))
    metadata = plan.token_actions[0].metadata
    final = [2] if factor == 1 else ([4, 4] if algorithm == "uniform_sampling_audio" else [4, 3])
    assert plan.placement is Placement.ENCODER_LEVEL
    assert metadata["initial_chunk_lengths"] == ([2] if factor == 1 else [16, 15])
    assert metadata["final_chunk_lengths"] == final
    assert metadata["keep_count"] == sum(final) // 2
    assert plan.this_stage_metadata.stats["planned_audio_tokens"] == metadata["keep_count"]



@pytest.mark.parametrize("algorithm", ["segmentwise_topk_audio", "similarity_attention_merge_prune_audio"])
def test_attention_encoder_contract_keeps_fixed_score_source(algorithm):
    from mllm_comp.core.vocabulary import Placement
    registry = builtin_multimodal_compression_contracts()
    stage = _temporal_stage(algorithm, llm=False)
    canonical = registry.binding(stage.binding)
    stage.placement = Placement.ENCODER_LEVEL
    stage.operation.update(target_site="audio_encoder_layer_post_mhsa_pre_ffn", layers=[3])
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni", available_handles={"audio_encoder": object()})
    binding = projection.bindings[0]
    assert binding.metadata["position_changed"] is True
    assert registry.binding(stage.binding) is canonical
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), operation=stage.operation)
    assert strategy.requires_layer_input is True
    offset = 1 if algorithm == "segmentwise_topk_audio" else 2
    assert strategy.score_layer_index(3, 4) == 4 - offset
    assert strategy.score_layer_index(0, offset) == 0
    assert strategy.spec.score_source == ("encoder_last_layer_attention" if offset == 1 else "encoder_penultimate_layer_attention")
    # The same algorithm may occupy two different physical executors.
    projector = _temporal_stage(algorithm, llm=False)
    projector.name = "projector"
    for current in [stage, projector]:
        validate_v2_stage_contract(current, registry=registry, backbone_name="qwen25_omni", selected_stages=[stage, projector])



@pytest.mark.parametrize("algorithm", ["segmentwise_topk_audio", "similarity_attention_merge_prune_audio"])
def test_attention_llm_requires_actual_encoder_reference(algorithm):
    registry = builtin_multimodal_compression_contracts()
    stage = _temporal_stage(algorithm, llm=True)
    with pytest.raises(V2ContractValidationError, match="audio_encoder"):
        validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni", available_handles={"llm": object()})
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni", available_handles={"llm": object(), "audio_encoder": object()})
    binding = projection.bindings[0]
    assert binding.targets[0].required_handles == frozenset({"llm", "audio_encoder"})
    assert "encoder_attention_reference" in binding.metadata["required_runtime_context"]
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), operation=stage.operation)
    assert strategy.requires_layer_input is False
    offset = -1 if algorithm == "segmentwise_topk_audio" else -2
    assert strategy.encoder_attention_layer_offset == offset
    assert strategy.spec.score_source == ("encoder_last_layer_attention" if offset == -1 else "encoder_penultimate_layer_attention")



@pytest.mark.parametrize("family", ["encoder", "llm"])
def test_unseg_layers_keep_dynamic_budget_and_original_rule(family):
    site = "audio_encoder_layer_output_post_block" if family == "encoder" else "llm_decoder_layer_output_audio_tokens"
    handle = "audio_encoder" if family == "encoder" else "llm"
    stage = CompressionStageConfig.from_dict({
        "name": f"unseg-{family}", "placement": f"{family}_level", "execution_phase": "forward",
        "plugin": "unsupervised_segmentation_audio", "algorithm": "unsupervised_segmentation_audio",
        "variant": "unsupervised_segmentation_audio.audio", "binding": "unsupervised_segmentation_audio.audio.qwen25_omni",
        "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "merge", "target_site": site, "layers": [0], "layer_edge": "before"},
    })
    projection = validate_v2_stage_contract(stage, registry=builtin_multimodal_compression_contracts(),
        backbone_name="qwen25_omni", available_handles={handle: object()})
    binding = projection.bindings[0]
    assert binding.executor_family == f"{family}_token"
    assert binding.targets[0].required_handles == frozenset({handle})
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    strategy = execution.create_strategy(execution.create_spec(stage.plugin_config()), operation=stage.operation)
    if family == "encoder":
        assert strategy.output_lengths(0, [4, 3], (0,)) == [None, None]
    else:
        assert strategy.output_length(8) is None
    assert strategy.spec.budget_policy == "natural_peaks"
    assert strategy.requires_layer_input is False


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"])
def test_encoder_attention_rule_can_resolve_llm_position(algorithm):
    stage = _representative_stage(algorithm, "generate")
    from mllm_comp.core.vocabulary import Placement
    stage.placement = Placement.LLM_LEVEL
    stage.operation.update(target_site="llm_decoder_layer_output_audio_tokens", layers=[0], layer_edge="before")
    stage.activation_scope["layers"] = [2]
    registry = builtin_multimodal_compression_contracts()
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=registry.binding(stage.binding))
    assert execution.executor_family == "llm_token"
    assert execution.binding.canonical_executor_family == "encoder_token"
    assert execution.binding.metadata["position_changed"] is True
    assert execution.binding.targets[0].required_handles == frozenset({"llm", "audio_encoder"})
    assert projection is not None


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"])
@pytest.mark.parametrize("site", ["audio_encoder_output_pre_projector", "audio_projector_output_post_projector"])
def test_encoder_attention_rule_can_resolve_projector_position(algorithm, site):
    from mllm_comp.core.vocabulary import Placement

    stage = _representative_stage(algorithm, "generate")
    stage.placement = Placement.PROJECTOR_LEVEL
    stage.operation.update(target_site=site)
    stage.activation_scope["layers"] = [2, 0]
    registry = builtin_multimodal_compression_contracts()
    assert validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni") is not None
    execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=registry.binding(stage.binding))
    assert execution.executor_family == "projector_token"
    assert execution.binding.canonical_executor_family == "encoder_token"
    assert execution.binding.metadata["position_changed"] is True
    assert execution.binding.prepare_after_slots == ("qwen25_omni.encoder_token",)
    assert execution.binding.targets[0].required_handles == frozenset({"thinker", "audio_encoder", "audio_projector"})
