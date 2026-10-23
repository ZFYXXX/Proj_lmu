from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image.visionzip import visionzip_image
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.compressors.video import (
    DyCoKeVideoKernel,
    DyToKVideoKernel,
    FastVIDVideoKernel,
    LLaMAVIDVideoKernel,
    LongVUVideoKernel,
    METokVideoKernel,
    PruneVidVideoKernel,
    VIDEO_ALGORITHMS,
    VIDEO_BACKBONES,
    VQTokenVideoKernel,
    VisionZipVideoKernel,
    VideoTokenPayload,
    canonical_video_config_record,
    resolve_canonical_video_config,
    validate_video_payload_signals,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError
from mllm_comp.core.support import EvidenceKind, SupportStatus
from mllm_comp.core.vocabulary import (
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
)
from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.pipeline.v2_validation import (
    V2ContractValidationError,
    validate_v2_stage_contract,
)
from mllm_comp.registry.catalog import CapabilityKind, get_capability_spec


_EXPECTED_CONFIGS = {
    "metok": {
        "segmentation_method": "similarity",
        "num_segments": 4,
        "keyframe_method": "similarity",
        "keyframe_ratio": 0.4,
        "top_k_events": 3,
        "distinguish_important_events": True,
        "s1_important": 2,
        "s2_important": 4,
        "s1_normal": 4,
        "s2_normal": 8,
        "pool_mode": "bilinear",
        "non_keyframe_method": "adaptive_pool",
        "keyframe_pool_enable": False,
        "inside_compression": True,
        "pruning_layers": [3, 10, 19],
        "ratio": 0.76,
    },
    "prunevid": {
        "tau": 0.8,
        "cluster_ratio": 0.5,
        "temporal_ratio": 0.25,
        "k": 7,
        "min_tokens_for_cluster": 14,
        "enable_vtp": True,
        "vtp_selected_layer": 10,
        "vtp_alpha": 0.4,
    },
    "visionzip": {
        "dominant_num": 126,
        "contextual_num": 21,
        "attn_layer": -2,
        "hidden_state_layer": -2,
    },
    "fastvid": {
        "retention_ratio": 0.1,
        "dyseg_c": 8,
        "dyseg_tau": 0.9,
        "stprune_d": 0.4,
        "dtm_p": 4,
        "dtm_beta": 0.6,
    },
    "vqtoken": {
        "num_clusters": 32,
        "adaptive": False,
        "use_cross_attention": False,
    },
    "dytok": {
        "mode": "dynamic",
        "base_compressor": "visionzip",
        "base_config": {
            "dominant_num": 126,
            "contextual_num": 21,
            "attn_layer": -2,
            "hidden_state_layer": -2,
        },
        "llm_config": {"target_layers": list(range(16, 24))},
        "upper_limit": 196,
    },
    "longvu": {
        "window_size": 16,
        "threshold": 0.83,
        "max_frames": 400,
        "min_frames": 1,
        "dino_interp_size": 576,
        "vision_hidden_size": 1024,
        "query_num": 144,
        "connector_depth": 3,
    },
    "llamavid": {
        "num_query": 32,
        "bert_type": "qformer",
        "compress_type": "mean",
    },
    "dycoke": {
        "merging_ratio": 0.7,
        "decoder_prune_layer": 3,
        "kv_drop_ratio": 0.8,
        "attention_similarity_threshold": 0.9,
    },
}

_NON_CANONICAL = {
    "metok": {"num_segments": 8},
    "prunevid": {"tau": 0.7},
    "visionzip": {"dominant_num": 125},
    "fastvid": {"retention_ratio": 0.2},
    "vqtoken": {"adaptive": True},
    "dytok": {"mode": "static"},
    "longvu": {"window_size": 8},
    "llamavid": {"compress_type": "grid:2"},
    "dycoke": {"kv_drop_ratio": 0.7},
}

_CONDITIONING = {
    "metok": {Modality.TEXT},
    "prunevid": {Modality.TEXT},
    "visionzip": set(),
    "fastvid": set(),
    "vqtoken": set(),
    "dytok": {Modality.TEXT},
    "longvu": set(),
    "llamavid": {Modality.TEXT},
    "dycoke": {Modality.TEXT},
}


@dataclass(frozen=True)
class _TensorLike:
    shape: tuple[int, ...]


def _stage(
    algorithm: str,
    backbone: str,
    *,
    config: dict[str, object] | None = None,
    plugin: str | None = None,
    compound_config: dict[str, object] | None = None,
) -> CompressionStageConfig:
    raw: dict[str, object] = {
        "name": f"{algorithm}-video-{backbone}",
        "placement": "projector_level",
        "execution_phase": "generate",
        "plugin": plugin or algorithm,
        "algorithm": algorithm,
        "variant": f"{algorithm}.video",
        "target_modalities": ["video"],
        "conditioning_modalities": sorted(
            modality.value for modality in _CONDITIONING[algorithm]
        ),
        "preserved_modalities": ["text", "image", "audio"],
        "coupling_policy": "independent",
        "media_alignment": "spatiotemporal",
        "config": config or {},
    }
    binding_name = f"{algorithm}.video.{backbone}"
    if compound_config is None:
        raw["binding"] = binding_name
    else:
        raw["bindings"] = [
            {
                "binding": binding_name,
                "target_modalities": ["video"],
                "order": 0,
                "config": compound_config,
            }
        ]
    return CompressionStageConfig.from_dict(raw)


def test_registry_activates_only_exact_evidenced_video_cells() -> None:
    registry = builtin_multimodal_compression_contracts()
    variants = [
        variant
        for variant in registry.variants()
        if Modality.VIDEO in variant.target_modalities
    ]
    bindings = [
        binding
        for binding in registry.bindings()
        if Modality.VIDEO in binding.target_modalities
    ]
    units = registry.support.find(target_modality=Modality.VIDEO)

    assert {variant.name for variant in variants} == {
        f"{algorithm}.video" for algorithm in VIDEO_ALGORITHMS
    }
    assert len(variants) == 9
    assert len(bindings) == 28
    assert len(units) == 56
    assert {unit.identity.algorithm for unit in units} == set(VIDEO_ALGORITHMS)
    assert {unit.identity.backbone for unit in units} == {*VIDEO_BACKBONES, "qwen3_5"}
    assert {unit.identity.phase for unit in units} == {
        ExecutionPhase.FORWARD,
        ExecutionPhase.GENERATE,
    }
    active = registry.support.find(
        target_modality=Modality.VIDEO,
        status=SupportStatus.ACTIVE,
    )
    assert {unit.identity.key for unit in active} == {
        f"{algorithm}:{algorithm}.video:{backbone}:video:{phase}"
        for algorithm in _EXPECTED_CONFIGS
        for backbone in ("llava_onevision", "qwen25_vl", "qwen3_vl")
        for phase in ("forward", "generate")
    } | {"vqtoken:vqtoken.video:qwen3_5:video:generate"}
    for unit in active:
        assert unit.inference_active
        if unit.identity.phase is ExecutionPhase.FORWARD:
            assert unit.passing_evidence == {
                EvidenceKind.KERNEL,
                EvidenceKind.CPU_CONTRACT,
                EvidenceKind.REAL_CHECKPOINT,
                EvidenceKind.TRAIN_STEP,
                EvidenceKind.TRAIN_BACKWARD,
                EvidenceKind.CHECKPOINT_RELOAD,
            }
            assert unit.train_step_verified
            assert unit.train_verified
        else:
            assert unit.passing_evidence == {
                EvidenceKind.KERNEL,
                EvidenceKind.CPU_CONTRACT,
                EvidenceKind.REAL_CHECKPOINT,
            }
        assert unit.missing_activation_evidence == frozenset()

    qwen3_5 = registry.binding("vqtoken.video.qwen3_5")
    assert qwen3_5.targets[0].placement.value == "projector_level"
    assert qwen3_5.legalizer == "qwen3_5.video_sparse_mrope"
    assert qwen3_5.metadata["implementation_state"] == "implemented_candidate"


def test_active_video_evidence_artifact_matches_every_registered_checksum() -> None:
    registry = builtin_multimodal_compression_contracts()
    active = registry.support.find(
        target_modality=Modality.VIDEO,
        status=SupportStatus.ACTIVE,
    )
    assert len(active) == 55
    for unit in active:
        artifact = Path(unit.evidence[0].artifact or "")
        assert artifact.is_file()
        checksum = hashlib.sha256(artifact.read_bytes()).hexdigest()

        assert {record.artifact for record in unit.evidence} == {artifact.as_posix()}
        assert {record.checksum for record in unit.evidence} == {checksum}
        assert unit.train_verified is (unit.identity.phase is ExecutionPhase.FORWARD)


def test_catalog_projects_video_units_without_borrowing_image_activation() -> None:
    for algorithm in _EXPECTED_CONFIGS:
        spec = get_capability_spec(CapabilityKind.COMPRESSOR, algorithm)
        units = [
            item
            for item in spec.metadata["support_units"]
            if ":video:" in item["identity"]
        ]
        assert len(units) == 6
        assert {item["identity"] for item in units} == {
            f"{algorithm}:{algorithm}.video:{backbone}:video:{phase}"
            for backbone in ("llava_onevision", "qwen25_vl", "qwen3_vl")
            for phase in ("forward", "generate")
        }
        assert {item["status"] for item in units} == {"active"}


def test_video_variants_are_independent_and_preserve_image_text_audio() -> None:
    registry = builtin_multimodal_compression_contracts()

    for algorithm in VIDEO_ALGORITHMS:
        variant = registry.variant(f"{algorithm}.video")
        assert variant.algorithm == algorithm
        assert variant.target_modalities == {Modality.VIDEO}
        assert variant.conditioning_modalities == _CONDITIONING[algorithm]
        assert variant.preserved_modalities == {
            Modality.TEXT,
            Modality.IMAGE,
            Modality.AUDIO,
        }
        assert variant.coupling_policy is ModalityCouplingPolicy.INDEPENDENT
        assert variant.media_alignment is MediaAlignment.SPATIOTEMPORAL
        assert variant.token_unit.value == "feature_token"
        assert variant.metadata["scope"] == "video_only"
        assert variant.metadata["config_policy"] == "canonical_only"
        assert variant.name != "visionzip.image"


def test_all_nine_canonical_configs_match_the_frozen_reference_profile() -> None:
    registry = builtin_multimodal_compression_contracts()

    assert set(_EXPECTED_CONFIGS) == set(VIDEO_ALGORITHMS)
    for algorithm, expected in _EXPECTED_CONFIGS.items():
        assert canonical_video_config_record(algorithm) == expected
        variant = registry.variant(f"{algorithm}.video")
        assert canonical_video_config_record(variant.name) == expected
        with pytest.raises(TypeError):
            variant.config_defaults["new"] = 1  # type: ignore[index]


@pytest.mark.parametrize("algorithm", VIDEO_ALGORITHMS)
def test_canonical_config_rejects_unknown_fields_and_ablations(
    algorithm: str,
) -> None:
    with pytest.raises(ConfigError, match="unknown fields"):
        resolve_canonical_video_config(algorithm, {"not_canonical": 1})
    with pytest.raises(ConfigError, match="non-canonical"):
        resolve_canonical_video_config(algorithm, _NON_CANONICAL[algorithm])


def test_canonical_config_accepts_yaml_sequences_and_partial_nested_identity() -> None:
    metok = resolve_canonical_video_config(
        "metok.video", {"pruning_layers": [3, 10, 19]}
    )
    dytok = resolve_canonical_video_config(
        "dytok", {"base_config": {"dominant_num": 126}}
    )

    assert metok["pruning_layers"] == (3, 10, 19)
    assert dytok["base_config"]["contextual_num"] == 21
    with pytest.raises(ConfigError, match="non-canonical"):
        resolve_canonical_video_config(
            "dytok", {"llm_config": {"target_layers": [15, *range(17, 24)]}}
        )


@pytest.mark.parametrize(
    ("algorithm", "backbone"),
    [
        (algorithm, backbone)
        for algorithm in VIDEO_ALGORITHMS
        for backbone in VIDEO_BACKBONES
    ]
    + [("vqtoken", "qwen3_5")],
)
def test_all_bindings_validate_to_the_full_canonical_config(
    algorithm: str,
    backbone: str,
) -> None:
    registry = builtin_multimodal_compression_contracts()
    projection = validate_v2_stage_contract(
        _stage(algorithm, backbone),
        registry=registry,
        backbone_name=backbone,
    )

    assert projection.binding_names == (f"{algorithm}.video.{backbone}",)
    assert projection.variant.name == f"{algorithm}.video"
    assert dict(projection.algorithm_config) == dict(
        registry.variant(f"{algorithm}.video").config_defaults
    )


def test_video_stage_validation_rejects_plugin_alias_and_binding_overrides() -> None:
    registry = builtin_multimodal_compression_contracts()

    with pytest.raises(V2ContractValidationError, match="does not match algorithm"):
        validate_v2_stage_contract(
            _stage("vqtoken", "qwen25_vl", plugin="visionzip"),
            registry=registry,
            backbone_name="qwen25_vl",
        )
    with pytest.raises(V2ContractValidationError, match="unknown fields"):
        validate_v2_stage_contract(
            _stage(
                "vqtoken",
                "qwen25_vl",
                compound_config={"num_clusters": 32},
            ),
            registry=registry,
            backbone_name="qwen25_vl",
        )


def test_video_binding_structural_handles_fail_closed_when_inspected() -> None:
    registry = builtin_multimodal_compression_contracts()
    stage = _stage("visionzip", "qwen3_vl")
    binding = registry.binding("visionzip.video.qwen3_vl")

    with pytest.raises(V2ContractValidationError, match="missing structural handles"):
        validate_v2_stage_contract(
            stage,
            registry=registry,
            backbone_name="qwen3_vl",
            available_handles={},
        )
    projection = validate_v2_stage_contract(
        stage,
        registry=registry,
        backbone_name="qwen3_vl",
        available_handles={name: object() for name in binding.required_handles},
    )
    assert "vision_deepstack_projectors" in projection.required_handles


def test_binding_topology_legalizers_and_conflicts_are_exact() -> None:
    registry = builtin_multimodal_compression_contracts()
    compound = {"metok", "prunevid", "dytok", "dycoke"}
    legalizers = {
        "llava_onevision": "llava_onevision.video_projector_llm_splice",
        "qwen25_vl": "qwen25_vl.video_grid_thw_sparse_mrope",
        "qwen3_vl": "qwen3_vl.video_mrope_deepstack_visual_pos_masks",
    }

    for algorithm in VIDEO_ALGORITHMS:
        for backbone in VIDEO_BACKBONES:
            binding = registry.binding(f"{algorithm}.video.{backbone}")
            assert binding.supported_phases == {
                ExecutionPhase.FORWARD,
                ExecutionPhase.GENERATE,
            }
            assert callable(binding.executor_factory)
            assert binding.legalizer == legalizers[backbone]
            assert binding.is_compound is (algorithm in compound)
            assert binding.conflicts == {
                f"{other}.video.{backbone}"
                for other in VIDEO_ALGORITHMS
                if other != algorithm
            }
            assert all(
                target.target_modalities == {Modality.VIDEO}
                for target in binding.targets
            )
            assert not any(".image." in name for name in binding.conflicts)
            if backbone == "qwen25_vl":
                assert binding.metadata["qwen25_video_grid_thw"] is True
            if backbone == "qwen3_vl":
                assert binding.metadata["qwen3_deepstack_lockstep"] is True
                assert binding.metadata["qwen3_visual_pos_masks"] is True
            assert binding.metadata["implementation_state"] == "implemented_candidate"

    detection = registry.binding("dytok.video.qwen25_vl").ordered_targets()[0]
    assert detection.name == "detection_budget_provider"
    assert detection.metadata["paper_compression_stage"] is False
    assert detection.metadata["component_role"] == "capability_provider"


def test_forward_and_generate_support_units_have_phase_exact_evidence_gates() -> None:
    registry = builtin_multimodal_compression_contracts()
    forward = registry.support.find(
        variant="dycoke.video", phase=ExecutionPhase.FORWARD
    )
    generate = registry.support.find(
        variant="dycoke.video", phase=ExecutionPhase.GENERATE
    )

    assert len(forward) == len(generate) == 3
    assert EvidenceKind.TRAIN_STEP in forward[0].required_evidence
    assert EvidenceKind.TRAIN_BACKWARD in forward[0].required_evidence
    assert EvidenceKind.CHECKPOINT_RELOAD in forward[0].required_evidence
    assert EvidenceKind.REAL_CHECKPOINT in generate[0].required_evidence
    assert EvidenceKind.TRAIN_STEP not in generate[0].required_evidence


def test_implemented_kernels_and_support_gate_fail_before_false_execution() -> None:
    registry = builtin_multimodal_compression_contracts()
    assert registry.algorithm("visionzip").kernel is visionzip_image
    assert VisionZipVideoKernel().name == "visionzip"
    metok = registry.algorithm("metok")
    assert isinstance(metok.kernel, METokVideoKernel)
    assert metok.metadata["implementation_state"] == "implemented_candidate"
    fastvid = registry.algorithm("fastvid")
    assert isinstance(fastvid.kernel, FastVIDVideoKernel)
    assert fastvid.metadata["implementation_state"] == "implemented_candidate"
    prunevid = registry.algorithm("prunevid")
    assert isinstance(prunevid.kernel, PruneVidVideoKernel)
    assert prunevid.metadata["implementation_state"] == "implemented_candidate"
    vqtoken = registry.algorithm("vqtoken")
    assert isinstance(vqtoken.kernel, VQTokenVideoKernel)
    assert vqtoken.metadata["implementation_state"] == "implemented_candidate"
    longvu = registry.algorithm("longvu")
    assert isinstance(longvu.kernel, LongVUVideoKernel)
    assert longvu.metadata["implementation_state"] == "implemented_candidate"
    llamavid = registry.algorithm("llamavid")
    assert isinstance(llamavid.kernel, LLaMAVIDVideoKernel)
    assert llamavid.metadata["implementation_state"] == "implemented_candidate"
    dytok = registry.algorithm("dytok")
    assert isinstance(dytok.kernel, DyToKVideoKernel)
    assert dytok.metadata["implementation_state"] == "implemented_candidate"
    dycoke = registry.algorithm("dycoke")
    assert isinstance(dycoke.kernel, DyCoKeVideoKernel)
    assert dycoke.metadata["implementation_state"] == "implemented_candidate"
    stage = _stage("llamavid", "qwen3_vl")
    runtime = SimpleNamespace(model=object(), metadata={})
    facade = SimpleNamespace(capabilities=lambda: SimpleNamespace(name="qwen3_vl"))
    with pytest.raises(BackendIntegrationError, match="visual dimensions"):
        RuntimeCompressionDispatcher(contract_registry=registry).activate(
            config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
            facade=facade,
            runtime=runtime,
            compression_mode="compressed",
        )

    with pytest.raises(BackendIntegrationError, match="metok_siglip_path"):
        RuntimeCompressionDispatcher(contract_registry=registry).activate(
            config=SimpleNamespace(
                pipeline=SimpleNamespace(stages=[_stage("metok", "qwen3_vl")])
            ),
            facade=facade,
            runtime=runtime,
            compression_mode="compressed",
        )


def test_implemented_video_factories_build_declared_candidates_only() -> None:
    registry = builtin_multimodal_compression_contracts()
    for algorithm in (
        "dycoke",
        "dytok",
        "fastvid",
        "llamavid",
        "longvu",
        "metok",
        "prunevid",
        "visionzip",
        "vqtoken",
    ):
        for backbone in VIDEO_BACKBONES:
            binding = registry.binding(f"{algorithm}.video.{backbone}")
            executor = binding.executor_factory(
                model=object(),
                runtime=SimpleNamespace(),
                facade=SimpleNamespace(),
                stage=_stage(algorithm, backbone),
                binding=binding,
                binding_config=SimpleNamespace(config={}),
                execution_phase=ExecutionPhase.GENERATE,
            )

            metadata = executor.runtime_metadata()
            assert metadata["backbone"] == backbone
            assert metadata["variant"] == f"{algorithm}.video"
            assert metadata["installed"] is False
            assert metadata["projector_invocations"] == 0
            if algorithm == "metok":
                assert metadata["pruning_invocations"] == 0
                assert metadata["signal_provider"]["siglip_frozen"] is True
            elif algorithm == "fastvid":
                assert metadata["stages"] == ("dyseg", "stprune", "dtm")
                assert metadata["signal_provider"]["raw_stream_attached"] is True
            elif algorithm == "dytok":
                assert metadata["stages"] == (
                    "detection_budget_provider",
                    "dynamic_visionzip_reduce",
                )
                assert metadata["base_budget_per_temporal_slice"] == 147
                assert metadata["signal_provider"]["detector_frozen"] is True
            elif algorithm == "longvu":
                assert metadata["stages"] == (
                    "dinov2_frame_selection",
                    "sva_fusion",
                    "output_projection",
                )
                assert metadata["fusion_registered_pre_optimizer"] is False
            elif algorithm == "llamavid":
                assert metadata["stages"] == (
                    "bert_qformer_cross_attention",
                    "context_attention_pool",
                    "mean_content_pool",
                )
                assert metadata["qformer_registered_pre_optimizer"] is False
            elif algorithm == "prunevid":
                assert metadata["vtp_invocations"] == 0
            elif algorithm == "vqtoken":
                assert metadata["global_fixed_k"] == 32
            elif algorithm == "dycoke":
                assert metadata["decode_view_invocations"] == 0
                assert metadata["physical_cache_retained"] is True
            else:
                assert metadata["fixed_budget_per_temporal_slice"] == 147
                assert metadata["image_variant_shared"] is False
            evidenced = algorithm in {"vqtoken", "dycoke", "longvu", "llamavid"}
            assert metadata["support_activation_claimed"] is evidenced
            assert metadata["real_checkpoint_verified"] is evidenced
            assert (metadata.get("evidence_artifact") is not None) is evidenced


def test_runtime_signal_contract_uses_payload_builtins_and_provider_signals() -> None:
    vqtoken_payload = VideoTokenPayload(
        features=_TensorLike((4, 3)),
        media_ids=("video-0",),
        video_offsets=(0, 4),
        coordinates=((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1)),
    )
    available = validate_video_payload_signals("vqtoken.video", vqtoken_payload)
    assert {"projected_features", "video_boundaries", "coordinates"} <= available

    with pytest.raises(BackendIntegrationError, match="vision_encoder_attention"):
        validate_video_payload_signals("visionzip", vqtoken_payload)
    visionzip_payload = VideoTokenPayload(
        features=_TensorLike((4, 3)),
        media_ids=("video-0",),
        video_offsets=(0, 4),
        coordinates=vqtoken_payload.coordinates,
        signals={
            "vision_encoder_attention": object(),
            "vision_encoder_hidden_states": object(),
        },
    )
    validate_video_payload_signals("visionzip.video", visionzip_payload)


def test_video_registration_preserves_visionzip_image_defaults() -> None:
    registry = builtin_multimodal_compression_contracts()
    assert registry.variant("visionzip.image").config_defaults == {
        "budget": 32,
        "dominant_weight": 27,
        "contextual_weight": 5,
    }
    assert registry.variant("visionzip.image").target_modalities == {Modality.IMAGE}


@pytest.mark.parametrize("algorithm", VIDEO_ALGORITHMS)
def test_video_factory_consumes_resolved_config_without_reparsing(
    algorithm, monkeypatch
):
    from importlib import import_module

    registry = builtin_multimodal_compression_contracts()
    stage = _stage(algorithm, "qwen3_vl")
    projection = validate_v2_stage_contract(
        stage, registry=registry, backbone_name="qwen3_vl"
    )
    provider = import_module(f"mllm_comp.backbones.{algorithm}_video")

    def reparse(*args, **kwargs):
        pytest.fail("Video factory discarded its validated config")

    monkeypatch.setattr(provider, "_validated_config", reparse)
    binding = projection.bindings[0]
    executor = binding.executor_factory(
        model=SimpleNamespace(),
        runtime=SimpleNamespace(metadata={}),
        stage=stage,
        binding=binding,
        binding_config=projection.selections[0],
        execution_phase=ExecutionPhase.GENERATE,
        contract_projection=projection,
    )
    assert executor.config is projection.algorithm_config
    assert not executor.installed
