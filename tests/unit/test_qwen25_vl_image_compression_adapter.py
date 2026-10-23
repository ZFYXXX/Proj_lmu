"""CPU contract tests for the image-only Qwen2.5-VL compression bridge."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen25_vl import image_compression as qwen25_binding
from mllm_comp.backbones.qwen25_vl.image_compression import (
    QWEN25_VL_IMAGE_BINDING_CONTRACTS,
    Qwen25VLDecoderRepairPlan,
    Qwen25VLImageCompressionAdapter,
    Qwen25VLMRoPEResult,
    Qwen25VLPlaceholderRepairPlan,
    create_qwen25_vl_image_compression_executor,
)
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


class _Bridge:
    def __init__(self) -> None:
        self.events: list[tuple[str, Any]] = []
        self.executor: Any | None = None
        self.handle = object()

    def install_image_compression(self, executor: Any) -> object:
        self.events.append(("install", executor.contract.binding))
        self.executor = executor
        return self.handle

    def uninstall_image_compression(self, handle: object) -> None:
        assert handle is self.handle
        self.events.append(("uninstall", self.executor.contract.binding))
        self.executor = None


def _stage(algorithm: str, variant: str, **config: Any) -> SimpleNamespace:
    return SimpleNamespace(
        name=f"{algorithm}-image",
        algorithm=algorithm,
        variant=variant,
        target_modalities={Modality.IMAGE},
        config=config,
    )


def _resolve(
    binding_name: str,
    *,
    bridge: _Bridge | None = None,
    binding_config: dict[str, Any] | None = None,
    phase: ExecutionPhase = ExecutionPhase.GENERATE,
) -> Any:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding(binding_name)
    contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS[binding_name]
    runtime = SimpleNamespace(
        metadata=(
            {"qwen25_vl_image_compression_bridge": bridge} if bridge is not None else {}
        )
    )
    return Qwen25VLImageCompressionAdapter().resolve(
        model=SimpleNamespace(),
        runtime=runtime,
        stage=_stage(contract.algorithm, contract.variant),
        binding=binding,
        binding_config=SimpleNamespace(config=binding_config or {}),
        execution_phase=phase,
        facade=SimpleNamespace(),
    )


def test_qwen25_vl_exposes_exact_image_only_binding_contracts() -> None:
    assert set(QWEN25_VL_IMAGE_BINDING_CONTRACTS) == {
        "tome.image.qwen25_vl",
        "visionzip.image.qwen25_vl",
        "feature_anchor_mean.image.qwen25_vl",
        "grid_mean.image.qwen25_vl",
        "pyramid_drop.image.qwen25_vl",
        "divprune.image.qwen25_vl",
        "saint.image.qwen25_vl",
        "aim.image.qwen25_vl",
        "pact.image.qwen25_vl",
        "fastv.image.qwen25_vl",
        "sgl.image.qwen25_vl",
        "dart.image.qwen25_vl",
        "d2pruner.image.qwen25_vl",
        "sparsevlm.image.qwen25_vl",
        "finepruner.image.qwen25_vl",
        "btp.image.qwen25_vl",
        "halfv.image.qwen25_vl",
        "transprune.image.qwen25_vl",
        "mmtok.image.qwen25_vl",
        "vispruner.image.qwen25_vl",
        "zoo_prune.image.qwen25_vl",
        "nuwa.image.qwen25_vl",
        "restore.image.qwen25_vl",
        "prumerge.image.qwen25_vl",
        "gap.image.qwen25_vl",
        "feather.image.qwen25_vl",
        "ivc_prune.image.qwen25_vl",
        "bifocal.image.qwen25_vl",
    }
    assert {
        contract.callback for contract in QWEN25_VL_IMAGE_BINDING_CONTRACTS.values()
    } == {
        "encoder_dense_boundary",
        "encoder_progressive_prune",
        "post_visual_merger",
        "decoder_layer_loop",
    }
    assert all(
        contract.image_only and not contract.supports_video
        for contract in QWEN25_VL_IMAGE_BINDING_CONTRACTS.values()
    )
    assert QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "visionzip.image.qwen25_vl"
    ].target_sites == ("post_vision_merger_pre_llm_scatter",)
    assert QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "divprune.image.qwen25_vl"
    ].target_sites == ("post_vision_merger_pre_llm_scatter",)
    assert QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "mmtok.image.qwen25_vl"
    ].target_sites == ("post_vision_merger_pre_llm_scatter",)
    vispruner = QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "vispruner.image.qwen25_vl"
    ]
    assert vispruner.target_sites == ("post_vision_merger_pre_llm_scatter",)
    assert vispruner.requires_visual_metrics is True
    zoo_prune = QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "zoo_prune.image.qwen25_vl"
    ]
    assert zoo_prune.target_sites == ("post_vision_merger_pre_llm_scatter",)
    assert zoo_prune.requires_visual_metrics is False
    assert zoo_prune.metadata["training_free"] is True
    nuwa = QWEN25_VL_IMAGE_BINDING_CONTRACTS["nuwa.image.qwen25_vl"]
    assert nuwa.target_sites == (
        "post_vision_merger_pre_llm_scatter",
        "llm_decoder_pre_stage2_layer_image_tokens",
    )
    assert nuwa.requires_visual_metrics is True
    assert nuwa.metadata["composite_binding"] is True
    assert nuwa.metadata["cache_policy"] == "compact_prefix"
    assert nuwa.metadata["position_policy"] == "native_sparse_mrope_rpme"
    assert (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS[
            "divprune.image.qwen25_vl"
        ].requires_visual_metrics
        is False
    )
    assert (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS["fastv.image.qwen25_vl"].metadata[
            "cache_policy"
        ]
        == "compact_prefix"
    )
    sgl = QWEN25_VL_IMAGE_BINDING_CONTRACTS["sgl.image.qwen25_vl"]
    assert sgl.metadata["dual_model_lifecycle"] is True
    assert sgl.metadata["single_image_only"] is True
    assert (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS["dart.image.qwen25_vl"].metadata[
            "single_image_only"
        ]
        is True
    )
    assert (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS["d2pruner.image.qwen25_vl"].metadata[
            "requires_calibrated_bias_prior"
        ]
        is True
    )
    assert (
        QWEN25_VL_IMAGE_BINDING_CONTRACTS["sparsevlm.image.qwen25_vl"].metadata[
            "cache_policy"
        ]
        == "compact_prefix"
    )
    finepruner = QWEN25_VL_IMAGE_BINDING_CONTRACTS["finepruner.image.qwen25_vl"]
    assert finepruner.metadata["cache_policy"] == "layer_local"
    assert finepruner.metadata["attention_capture"] == "last_text_query_per_head"
    btp = QWEN25_VL_IMAGE_BINDING_CONTRACTS["btp.image.qwen25_vl"]
    assert btp.metadata["cache_policy"] == "compact_prefix"
    assert btp.metadata["position_policy"] == "preserve_original"
    assert btp.metadata["final_stage"] == "remove_remaining_visual_span"
    halfv = QWEN25_VL_IMAGE_BINDING_CONTRACTS["halfv.image.qwen25_vl"]
    assert halfv.metadata["cache_policy"] == "compact_prefix"
    assert halfv.metadata["position_policy"] == "preserve_original"
    assert halfv.metadata["single_image_only"] is True
    transprune = QWEN25_VL_IMAGE_BINDING_CONTRACTS[
        "transprune.image.qwen25_vl"
    ]
    assert transprune.callback == "decoder_layer_loop"
    assert transprune.metadata["cache_policy"] == "layer_local"
    assert transprune.metadata["position_policy"] == "contiguous"
    assert transprune.metadata["iga_source"] == "next_layer_post_image_text"
    assert transprune.metadata["single_image_only"] is True


def test_executor_install_and_uninstall_are_explicit_and_idempotent() -> None:
    bridge = _Bridge()
    executor = _resolve("visionzip.image.qwen25_vl", bridge=bridge)

    assert bridge.events == []
    assert executor.runtime_metadata()["checkpoint_evidence"] == "not_claimed"

    executor.install()
    executor.install()
    assert bridge.events == [("install", "visionzip.image.qwen25_vl")]
    assert bridge.executor is executor

    executor.uninstall()
    executor.uninstall()
    assert bridge.events[-1] == ("uninstall", "visionzip.image.qwen25_vl")
    assert executor.runtime_metadata()["installed"] is False


def test_nuwa_composite_executor_uses_one_provider_lifecycle() -> None:
    bridge = _Bridge()
    executor = _resolve(
        "nuwa.image.qwen25_vl",
        bridge=bridge,
        binding_config={
            "stage2_after_layer": 2,
            "semantic_hidden_start": 0,
            "semantic_hidden_end": 3,
        },
    )

    executor.install()

    assert bridge.events == [("install", "nuwa.image.qwen25_vl")]
    assert executor.contract.metadata["composite_binding"] is True
    assert executor.runtime_metadata()["algorithm"] == "nuwa"
    assert executor.runtime_metadata()["checkpoint_evidence"] == "not_claimed"

    executor.uninstall()
    assert bridge.events[-1] == ("uninstall", "nuwa.image.qwen25_vl")


def test_invalid_model_fails_when_pinned_provider_auto_install_is_impossible() -> None:
    executor = _resolve("tome.image.qwen25_vl")

    with pytest.raises(BackendIntegrationError, match="could not resolve.*install"):
        executor.install()


def test_divprune_invalid_config_fails_before_provider_install() -> None:
    bridge = _Bridge()

    with pytest.raises(BackendIntegrationError, match="Invalid DivPrune config"):
        _resolve(
            "divprune.image.qwen25_vl",
            bridge=bridge,
            binding_config={"retain_ratio": 0.0},
        )

    assert bridge.events == []


@pytest.mark.parametrize(
    "binding_name",
    [
        "d2pruner.image.qwen25_vl",
        "mmtok.image.qwen25_vl",
        "vispruner.image.qwen25_vl",
    ],
)
def test_explicit_config_typo_fails_before_provider_install(
    binding_name: str,
) -> None:
    bridge = _Bridge()
    with pytest.raises(BackendIntegrationError, match="rato"):
        _resolve(
            binding_name,
            bridge=bridge,
            binding_config={
                "pivot_rato"
                if binding_name.startswith("d2pruner")
                else "retain_rato": 0.5
            },
        )
    assert bridge.events == []


def test_adapter_rejects_video_as_a_separate_unsupported_unit() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "visionzip.image.qwen25_vl"
    )
    stage = _stage("visionzip", "visionzip.image")
    stage.target_modalities = {Modality.VIDEO}

    with pytest.raises(BackendIntegrationError, match="video requires a separate"):
        Qwen25VLImageCompressionAdapter().resolve(
            model=object(),
            runtime=SimpleNamespace(metadata={}),
            stage=stage,
            binding=binding,
            execution_phase=ExecutionPhase.GENERATE,
        )


def test_tome_callback_returns_dense_restore_before_the_spatial_merger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta = LayoutDelta.merge(
        4,
        [(0, 1)],
        media_unit=LayoutTokenUnit.IMAGE_PATCH,
    )
    calls: list[dict[str, Any]] = []

    def fake_tome(hidden_states: Any, keys: Any, **kwargs: Any) -> Any:
        calls.append({"hidden_states": hidden_states, "keys": keys, **kwargs})
        return SimpleNamespace(layout_delta=delta, hidden_states=["m", "c", "d"])

    monkeypatch.setattr(qwen25_binding, "tome_merge_image", fake_tome)
    executor = _resolve(
        "tome.image.qwen25_vl",
        bridge=_Bridge(),
        binding_config={"r": 1},
    )

    compressed = executor.encoder_dense_boundary("hidden", "keys")

    assert calls == [{"hidden_states": "hidden", "keys": "keys", "rules": {}, "r": 1}]
    assert compressed.dense_restore.restore(
        compressed.kernel_result.hidden_states,
        sequence_dim=0,
    ) == ["m", "m", "c", "d"]
    assert compressed.dense_restore.dense_boundary_delta.is_identity
    assert compressed.dense_restore.dense_boundary_delta.result_count == 4
    record = executor.runtime_metadata()["layout_delta_records"][0]
    assert record["old_to_new"] == [0, 0, 1, 2]
    assert len(record["mapping_sha256"]) == 64


def test_visionzip_repair_compacts_placeholders_then_requires_sparse_mrope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    delta = LayoutDelta.merge(
        4,
        [(0, 1), (2, 3)],
        media_unit="image_feature",
        source_layout={"image_grid_thw": (1, 2, 2)},
        result_layout={"sparse": True},
    )
    kernel_calls: list[dict[str, Any]] = []

    def fake_visionzip(*args: Any, **kwargs: Any) -> Any:
        kernel_calls.append({"args": args, **kwargs})
        return SimpleNamespace(layout_delta=delta, tokens=[1, 2])

    monkeypatch.setattr(qwen25_binding, "visionzip_image", fake_visionzip)
    executor = _resolve("visionzip.image.qwen25_vl", bridge=_Bridge())
    compressed = executor.post_visual_merger(
        "tokens",
        "attention",
        "keys",
        placeholder_indices=(1, 2, 3, 4),
        source_sequence_length=6,
        image_index=0,
    )
    requests: list[Any] = []

    def rebuild_mrope(request: Any) -> Qwen25VLMRoPEResult:
        requests.append(request)
        assert request.input_ids == [100, 999, 999, 101]
        assert request.image_grid_thw == [(1, 2, 2)]
        assert request.layout_delta is delta
        return Qwen25VLMRoPEResult(
            position_ids=[[0, 1, 3, 4]] * 3,
            rope_deltas=[1],
            metadata={"provider": "fake_sparse_mrope"},
        )

    repaired = compressed.prompt_repair.repair(
        input_ids=[100, 999, 999, 999, 999, 101],
        attention_mask=[1, 1, 1, 1, 1, 1],
        labels=[-100, -100, -100, -100, -100, 101],
        image_grid_thw=[(1, 2, 2)],
        mrope_rebuilder=rebuild_mrope,
    )

    assert repaired.input_ids == [100, 999, 999, 101]
    assert repaired.attention_mask == [1, 1, 1, 1]
    assert repaired.labels == [-100, -100, -100, 101]
    assert repaired.removed_placeholder_indices == (2, 4)
    assert repaired.metadata["sparse_mrope_rebuilt"] is True
    assert repaired.metadata["provider"] == "fake_sparse_mrope"
    assert len(requests) == 1
    assert kernel_calls[0]["media_unit"] == "image_feature"


def test_divprune_reuses_post_merger_prompt_legalizer_and_records_image_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_divprune(tokens: Any, **kwargs: Any) -> Any:
        assert tokens == "tokens"
        assert kwargs["spec"].target_count == 2
        assert kwargs["source_layout"]["image_index"] == 2
        assert kwargs["result_layout"]["image_index"] == 2
        delta = LayoutDelta.drop(
            4,
            [1, 3],
            media_unit="image_feature",
            source_layout=kwargs["source_layout"],
            result_layout=kwargs["result_layout"],
        )
        return SimpleNamespace(
            tokens=["a", "c"],
            layout_delta=delta,
            metadata={"effective_target_count": 2},
        )

    monkeypatch.setattr(qwen25_binding, "divprune_image", fake_divprune)
    bridge = _Bridge()
    executor = _resolve(
        "divprune.image.qwen25_vl",
        bridge=bridge,
        binding_config={"target_count": 2},
    )
    executor.install()
    compressed = executor.post_visual_merger(
        "tokens",
        None,
        None,
        placeholder_indices=(1, 2, 3, 4),
        source_sequence_length=6,
        image_index=2,
    )

    assert compressed.kernel_result.tokens == ["a", "c"]
    assert compressed.prompt_repair.layout_delta.source_layout["image_index"] == 2
    latest = executor.runtime_metadata()["latest_layout_delta"]
    assert latest["planned_count"] == 2
    assert latest["selected_indices"] == [0, 2]
    assert latest["image_index"] == 2
    executor.uninstall()
    assert bridge.events == [
        ("install", "divprune.image.qwen25_vl"),
        ("uninstall", "divprune.image.qwen25_vl"),
    ]


def test_zoo_prune_uses_native_merger_inputs_and_prompt_legalizer() -> None:
    torch = pytest.importorskip("torch")
    executor = _resolve(
        "zoo_prune.image.qwen25_vl",
        bridge=_Bridge(),
        binding_config={
            "target_count": 2,
            "num_directions": 2,
            "projector_proxy": "full",
            "direction_batch_size": 1,
        },
    )
    projector_layers = (
        torch.nn.Linear(3, 3),
        torch.nn.GELU(),
        torch.nn.Linear(3, 2),
    )
    projector_inputs = torch.tensor(
        [[1.0, 0.0, 0.2], [0.9, 0.1, -0.1], [0.0, 1.0, 0.3], [-1.0, 0.0, 0.4]]
    )
    tokens = projector_inputs
    for layer in projector_layers:
        tokens = layer(tokens)
    compressed = executor.post_visual_merger(
        tokens,
        None,
        None,
        placeholder_indices=(1, 2, 3, 4),
        source_sequence_length=6,
        image_index=0,
        projector_inputs=projector_inputs,
        projector_layers=projector_layers,
    )

    assert compressed.kernel_result.tokens.shape == (2, 2)
    assert compressed.prompt_repair.layout_delta.source_count == 4
    assert compressed.prompt_repair.layout_delta.result_count == 2
    assert compressed.kernel_result.metadata["low_rank_solver_used"] == "full"
    assert executor.runtime_metadata()["latest_layout_delta"]["image_index"] == 0


def test_placeholder_plan_rejects_dense_count_mismatch_and_no_mrope_rebuilder() -> None:
    delta = LayoutDelta.drop(
        4,
        [1, 3],
        media_unit="image_feature",
    )
    with pytest.raises(Exception, match="placeholder count"):
        Qwen25VLPlaceholderRepairPlan(
            layout_delta=delta,
            placeholder_indices=(1, 2, 3),
            source_sequence_length=5,
        )
    plan = Qwen25VLPlaceholderRepairPlan(
        layout_delta=delta,
        placeholder_indices=(1, 2, 3, 4),
        source_sequence_length=6,
    )
    with pytest.raises(BackendIntegrationError, match="M-RoPE rebuilder"):
        plan.repair(
            input_ids=[0, 9, 9, 9, 9, 1],
            attention_mask=None,
            labels=None,
            image_grid_thw=[(1, 2, 2)],
            mrope_rebuilder=None,  # type: ignore[arg-type]
        )


def test_placeholder_plan_uses_explicit_source_ordered_representatives() -> None:
    delta = LayoutDelta.from_mapping(
        source_count=6,
        old_to_new=(1, 1, 0, 0, 0, 1),
        result_count=2,
        media_unit="image_feature",
        result_layout={"representative_indices": (3, 5)},
        ordering_policy="explicit",
    )

    plan = Qwen25VLPlaceholderRepairPlan(
        layout_delta=delta,
        placeholder_indices=(1, 2, 3, 4, 5, 6),
        source_sequence_length=8,
    )

    assert plan.layout_delta.result_layout["representative_indices"] == (3, 5)


def test_pyramid_drop_decoder_plan_repairs_sequence_fields_and_requires_cache_hook() -> (
    None
):
    delta = LayoutDelta.drop(6, [1, 4], media_unit="llm_token")
    plan = Qwen25VLDecoderRepairPlan(layout_delta=delta, boundary_layer=7)

    with pytest.raises(BackendIntegrationError, match="KV-cache rebuilder"):
        plan.repair(
            attention_mask=[1] * 6,
            position_ids=list(range(6)),
            cache_position=list(range(6)),
            labels=list(range(6)),
            past_key_values="cache",
            execution_phase=ExecutionPhase.GENERATE,
        )

    cache_requests: list[Any] = []
    repaired = plan.repair(
        attention_mask=[1] * 6,
        position_ids=list(range(6)),
        cache_position=list(range(6)),
        labels=list(range(6)),
        past_key_values="cache",
        execution_phase=ExecutionPhase.GENERATE,
        cache_rebuilder=lambda request: cache_requests.append(request)
        or "sparse-cache",
    )

    assert repaired.attention_mask == [1, 1, 1, 1]
    assert repaired.position_ids == [0, 2, 3, 5]
    assert repaired.labels == [0, 2, 3, 5]
    assert repaired.past_key_values == "sparse-cache"
    assert cache_requests[0].boundary_layer == 7
    assert cache_requests[0].kept_sequence_indices == (0, 2, 3, 5)


def test_public_factory_matches_registry_executor_signature() -> None:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("pyramid_drop.image.qwen25_vl")
    contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS[binding.name]

    executor = create_qwen25_vl_image_compression_executor(
        model=object(),
        runtime=SimpleNamespace(metadata={}),
        facade=object(),
        stage=_stage(contract.algorithm, contract.variant, keep_ratio=0.5),
        binding=binding,
        binding_config=SimpleNamespace(config={"minimum_keep_tokens": 2}),
        execution_phase=ExecutionPhase.GENERATE,
    )

    assert executor.config == {"keep_ratio": 0.5, "minimum_keep_tokens": 2}
    assert executor.contract.binding == binding.name


def test_executor_filters_control_plane_schema_fields_before_kernel_calls() -> None:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("tome.image.qwen25_vl")
    contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS[binding.name]
    stage = _stage(contract.algorithm, contract.variant, r=8)
    stage.operation = {"type": "merge", "target_site": "vision_block"}
    stage.criterion = {
        "score_source": "encoder_attention_key",
        "ranking": "bipartite_soft_matching",
    }
    stage.schedule = {"granularity": "per_image", "budgets": 8}

    executor = create_qwen25_vl_image_compression_executor(
        model=object(),
        runtime=SimpleNamespace(metadata={}),
        facade=object(),
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(
            config={"dense_restore_at_encoder_boundary": True}
        ),
        execution_phase=ExecutionPhase.GENERATE,
    )

    assert executor.config == {"r": 8}


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_registered_decoder_method_keeps_its_identity_through_adapter_lifecycle(backbone, monkeypatch):
    from dataclasses import replace
    from mllm_comp.core.vocabulary import Placement
    from mllm_comp.pipeline.definition import resolve_compression_definition
    from examples.compression_extensions.image_selection import experiment_config
    from tests.unit import test_qwen3_vl_image_compression_adapter as qwen3_fixture
    from tests.unit import test_llava_onevision_image_compression_adapter as llava_fixture
    from examples.compression_extensions.image_selection import ALGORITHM, VARIANT, AlternatingImageMethod

    config = experiment_config(backbone)
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    stage = definition.stages[0]
    projection = definition.projections[stage.name]
    binding = projection.bindings[0]
    runtime = SimpleNamespace(metadata={})
    if backbone == "qwen25_vl":
        model = SimpleNamespace()
        runtime.metadata["qwen25_vl_image_compression_bridge"] = _Bridge()
    elif backbone == "qwen3_vl":
        model, _, _ = qwen3_fixture._model()
    else:
        model = llava_fixture._model(llava_fixture._Bridge())
    kwargs = dict(model=model, runtime=runtime, facade=None, stage=stage, binding=binding,
        binding_config=projection.selections[0], execution_phase="forward", contract_projection=projection)
    executor = binding.executor_factory(**kwargs)
    assert isinstance(executor.decoder_selection, AlternatingImageMethod)
    assert executor.config == {"stride": 2, "offset": 1, "apply_before_layer": 2}
    executor.install()
    try:
        metadata = executor.runtime_metadata()
        assert metadata["algorithm"] == ALGORITHM
        assert metadata["variant"] == VARIANT
        assert metadata["binding"] == binding.name
        assert metadata["score_layers"] == []
        assert metadata["cache_policy"] == "compact_prefix"
    finally:
        executor.uninstall()
    assert not executor.installed

    invalid = replace(binding, targets=(replace(binding.targets[0], placement=Placement.PROJECTOR_LEVEL),))
    with pytest.raises(BackendIntegrationError, match="one image-only LLM target"):
        binding.executor_factory(**{**kwargs, "binding": invalid})
    with pytest.raises(BackendIntegrationError, match="Unknown|no ImageLLM"):
        binding.executor_factory(**{**kwargs, "binding": replace(binding, strategy_factory=None)})
    if backbone in {"qwen25_vl", "qwen3_vl"}:
        monkeypatch.setattr(AlternatingImageMethod, "position_policy", "contiguous")
        contiguous = binding.executor_factory(**kwargs).decoder_selection
        assert contiguous.position_policy == "contiguous"
        assert contiguous.cache_policy.value == "compact_prefix"


@pytest.mark.parametrize("backbone,mode,boundary", [
    (backbone, mode, boundary)
    for backbone in ("qwen25_vl", "qwen3_vl", "llava_onevision")
    for mode, boundary in (("override", 1), ("override", 2), ("none", 2), ("omitted", 2))
    if backbone != "llava_onevision" or mode != "omitted"
])
def test_registered_decoder_preserves_legacy_parameters_and_binding_overrides(backbone, mode, boundary):
    from mllm_comp.pipeline.stage import CompressionStageConfig, StageBindingConfig

    binding = builtin_multimodal_compression_contracts().binding(f"fastv.image.{backbone}")
    stage = CompressionStageConfig.from_dict({
        "name": "prune", "algorithm": "fastv", "plugin": "fastv", "variant": "fastv.image",
        "binding": binding.name, "placement": "llm_level", "execution_phase": "forward",
        "target_modalities": ["image"], "criterion": {"prune_layer": 2},
        "schedule": {"prune_ratio": 0.25},
        "operation": {"target_site": "llm_decoder_layer_output_image_tokens",
                      "layers": [boundary], "layer_edge": "before"},
    })
    selection = StageBindingConfig(binding=binding.name, config={"prune_layer": 1})
    stage.binding, stage.bindings = None, [selection]
    model = SimpleNamespace(model=SimpleNamespace(
        visual=SimpleNamespace(deepstack_merger_list=()),
        language_model=SimpleNamespace(layers=[object()] * 4),
    ))
    kwargs = dict(model=model, runtime=SimpleNamespace(metadata={}), facade=None,
                  stage=stage, binding=binding, execution_phase="forward")
    if mode != "omitted":
        kwargs["binding_config"] = selection if mode == "override" else None
    executor = binding.executor_factory(**kwargs)
    expected = 1 if mode == "override" else 2
    assert executor.decoder_selection.spec.prune_layer == expected
    assert executor.decoder_selection.spec.prune_ratio == 0.25
    assert executor.decoder_selection.score_layers == (expected - 1,)
    assert [point.index for point in executor.decoder_selection.positions] == [boundary]
    if backbone != "qwen25_vl":
        assert executor.binding.targets[0].metadata["score_layer"] == expected - 1
        assert executor.binding.metadata["position_changed"] is (boundary != expected)
    assert stage.criterion == {"prune_layer": 2}
    assert stage.config == {}


def test_registered_pyramid_keeps_nondefault_legacy_schedule_and_budget():
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.definition import resolve_compression_definition

    config = experiment_config_from_dict({"backbone": {"name": "qwen25_vl"},
        "pipeline": {"stages": [{"name": "prune", "algorithm": "pyramid_drop",
            "execution_phase": "forward", "criterion": {"minimum_keep_tokens": 2},
            "schedule": {"layer_indices": [1, 3], "num_stages": 3},
            "config": {"keep_ratio": 0.75}}]}})
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="train")
    projection = definition.projections["prune"]
    binding = projection.bindings[0]
    kwargs = dict(model=SimpleNamespace(), runtime=SimpleNamespace(metadata={}), facade=None,
        stage=definition.stages[0], binding=binding, binding_config=projection.selections[0],
        execution_phase="forward", contract_projection=projection)
    method = binding.executor_factory(**kwargs).decoder_selection
    method.resolve_depth(6)
    assert method.score_layers == (1, 3)
    assert method.spec.minimum_keep_tokens == 2
    assert method.spec.keep_ratio == 0.75
    assert method.spec.num_stages == 3
    with pytest.raises(BackendIntegrationError, match="keep_rato"):
        binding.executor_factory(**{**kwargs, "binding_config": SimpleNamespace(config={"keep_rato": 0.1})})
