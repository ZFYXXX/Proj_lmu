from __future__ import annotations

from typing import Any

import pytest
import mllm_comp.compressors.registry as registry_module

from mllm_comp.compressors.registry import (
    CompressionMethodDescriptor,
    CompressionMethodRegistry,
    builtin_compression_methods,
    register_compression_method,
)
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.errors import RegistryError
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)
from mllm_comp.registry.catalog import (
    CapabilityKind,
    CapabilityStatus,
    get_capability_spec,
)
from mllm_comp.registry.compressors import ACTIVE_COMPRESSORS, get_compressor


_TARGET_CANDIDATES = frozenset(
    {
        "fastslm_hta_audio",
        "uniform_average_pooling_audio",
        "uniform_sampling_audio",
        "unsupervised_segmentation_audio",
        "fastadasp_weighted_token_merging",
        "headrouter_dynamic_audio_pruning",
    }
)


def test_all_qwen_audio_methods_share_one_v2_contract_source() -> None:
    registry = builtin_multimodal_compression_contracts()
    bindings = {
        binding.name: binding
        for binding in registry.bindings()
        if binding.backbone == "qwen25_omni"
        and Modality.AUDIO in binding.target_modalities
    }
    algorithms = {
        registry.variant(binding.variant).algorithm for binding in bindings.values()
    }

    assert algorithms == {
        "segmentwise_topk_audio",
        "fastslm_hta_audio",
        "uniform_average_pooling_audio",
        "uniform_sampling_audio",
        "unsupervised_segmentation_audio",
        "local_temporal_bipartite_merging_audio",
        "similarity_attention_merge_prune_audio",
        "adjacent_token_merging_audio",
        "encoder_attention_topk_audio",
        "progressive_attention_pruning_audio",
        "audio_pyramid_attention_pruning",
        "fastadasp_weighted_token_merging",
        "headrouter_dynamic_audio_pruning",
        "structured_llm_layer_pruning",
    }
    assert builtin_compression_methods().descriptors() == ()
    assert {
        binding.executor_family for binding in bindings.values()
    } == {
        "projector_token",
        "encoder_token",
        "llm_token",
        "llm_operation",
    }
    for binding in bindings.values():
        algorithm = registry.algorithm(registry.variant(binding.variant).algorithm)
        capability = get_capability_spec(CapabilityKind.COMPRESSOR, algorithm.name)
        assert capability.provider is algorithm.kernel
        assert capability.status.value == algorithm.metadata["capability_status"]
        assert capability.metadata["executor_family"] == binding.executor_family
        assert capability.metadata["target_site"] == binding.targets[0].target_site
        assert binding.metadata["required_runtime_context"]


def test_target_candidates_remain_explicit_without_entering_active_registry() -> None:
    registry = builtin_multimodal_compression_contracts()

    assert {
        name
        for name in _TARGET_CANDIDATES
        if get_capability_spec(CapabilityKind.COMPRESSOR, name).status
        is CapabilityStatus.TARGET
    } == _TARGET_CANDIDATES
    assert _TARGET_CANDIDATES.isdisjoint(ACTIVE_COMPRESSORS)
    for name in _TARGET_CANDIDATES:
        assert get_compressor(name) is registry.algorithm(name).kernel


def test_qwen_v2_binding_keeps_runtime_context_out_of_legacy_registry() -> None:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding(
        "audio_pyramid_attention_pruning.audio.qwen25_omni"
    )

    assert "prefill_decode_state" in binding.metadata["required_runtime_context"]
    assert builtin_compression_methods().names() == frozenset()


def test_public_registry_accepts_an_external_existing_family_method(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Spec:
        @classmethod
        def from_config(cls, config: Any) -> "Spec":
            del config
            return cls()

    class Plugin:
        name = "researcher_projector_method"

        def validate_config(self, config: Any) -> None:
            del config

        def plan(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class Strategy:
        def __init__(self, spec: Any) -> None:
            self.spec = spec

        score_source = "fixture"
        attention_layer_offset = None

        def runtime_metadata(self) -> dict[str, Any]:
            return {}

        def compress(self, *args: Any, **kwargs: Any) -> tuple[Any, Any, dict[str, Any]]:
            del args, kwargs
            return None, None, {}

    descriptor = CompressionMethodDescriptor(
        name="researcher_projector_method",
        method=CompressionMethod.TOKEN_COMPRESSION,
        placement=Placement.PROJECTOR_LEVEL,
        target_site="audio_encoder_output_pre_projector",
        executor_family="projector_token",
        plugin=Plugin,
        spec=Spec,
        strategy=Strategy,
        supported_phases=frozenset(
            {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}
        ),
        interfaces=frozenset({"eval", "train"}),
        required_context=frozenset({"audio_spans", "audio_projector"}),
        object_family=ObjectFamily.TENSOR_SLOT,
        object_subtype="audio_encoder_output_pre_projector",
        modalities=frozenset({Modality.AUDIO}),
        token_roles=frozenset({TokenRole.OBSERVATION}),
    )
    registry = CompressionMethodRegistry()
    registry.register(descriptor)

    assert registry.get(descriptor.name).create_spec({}).__class__ is Spec
    assert isinstance(descriptor.create_strategy(Spec()), Strategy)

    monkeypatch.setattr(
        registry_module,
        "_BUILTIN_REGISTRY",
        builtin_compression_methods().clone(),
    )
    register_compression_method(descriptor)
    capability = get_capability_spec(CapabilityKind.COMPRESSOR, descriptor.name)
    assert capability.provider is Plugin
    assert capability.metadata["executor_family"] == "projector_token"


def test_descriptor_rejects_a_new_target_hidden_inside_an_existing_family() -> None:
    class Spec:
        @classmethod
        def from_config(cls, config: Any) -> "Spec":
            del config
            return cls()

    class Plugin:
        name = "invalid_hidden_target"

        def validate_config(self, config: Any) -> None:
            del config

        def plan(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

    class Strategy:
        def __init__(self, spec: Any) -> None:
            del spec

    with pytest.raises(RegistryError, match="requires an executor-family contract"):
        CompressionMethodDescriptor(
            name=Plugin.name,
            method=CompressionMethod.TOKEN_COMPRESSION,
            placement=Placement.LLM_LEVEL,
            target_site="llm_secret_mid_block_site",
            executor_family="llm_token",
            plugin=Plugin,
            spec=Spec,
            strategy=Strategy,
            supported_phases=frozenset({ExecutionPhase.GENERATE}),
            interfaces=frozenset({"eval"}),
            required_context=frozenset({"llm_decoder_layers"}),
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype="llm_secret_mid_block_site",
        )


def test_v2_stage_contract_includes_prefill_decode_context() -> None:
    registry = builtin_multimodal_compression_contracts()
    algorithm = registry.algorithm("audio_pyramid_attention_pruning")
    binding = registry.binding(
        "audio_pyramid_attention_pruning.audio.qwen25_omni"
    )
    assert "prefill_decode_state" in binding.metadata["required_runtime_context"]
    assert algorithm.metadata["prefill_compression"] is True
    assert algorithm.metadata["decode_compression"] is False
