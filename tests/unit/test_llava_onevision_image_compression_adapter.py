"""CPU contract tests for the fail-closed LLaVA-OneVision image bridge."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from mllm_comp.compressors.image import decoder_selection

import pytest

from mllm_comp.backbones.llava_onevision import image_compression as adapter
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import LayoutDelta
from mllm_comp.core.vocabulary import ExecutionPhase


class _Handle:
    def __init__(self, bridge: "_Bridge", site: str, callback: Any) -> None:
        self.bridge = bridge
        self.site = site
        self.callback = callback
        self.removed = False

    def remove(self) -> None:
        if not self.removed:
            self.bridge.callbacks[self.site].remove(self.callback)
            self.removed = True


class _Bridge:
    api_version = adapter.LLAVA_ONEVISION_BRIDGE_API_VERSION
    source_route = adapter.LLAVA_ONEVISION_SOURCE_ROUTE

    def __init__(self, *, capabilities: dict[str, bool] | None = None) -> None:
        self.callbacks = {
            site: []
            for contract in adapter.LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS.values()
            for site in contract.sites
        }
        self.requires_packed_metrics: dict[str, bool] = {}
        self.capabilities = {
            "crop_isolated_encoder_state": True,
            "anyres_layout_metadata": True,
            "image_newline_indices": True,
            "packed_attention_and_keys": True,
            "multisource_embedding_payload": True,
            "visual_attention_crop_payload": True,
            "projector_input_and_layers": True,
            "nuwa_visual_source_payload": True,
            "decoder_loop_control": True,
            "sequence_layout_repair": True,
            "kv_cache_repair": True,
            **(capabilities or {}),
        }

    def available_sites(self) -> tuple[str, ...]:
        return tuple(self.callbacks)

    def register_callback(
        self,
        site: str,
        callback: Any,
        *,
        requires_packed_metrics: bool = False,
        decoder_selection=None,
    ) -> _Handle:
        self.callbacks[site].append(callback)
        self.requires_packed_metrics[site] = requires_packed_metrics
        return _Handle(self, site, callback)

    def emit(self, site: str, request: Any) -> Any:
        callbacks = self.callbacks[site]
        assert len(callbacks) == 1
        return callbacks[0](request)


def _model(bridge: _Bridge | None = None) -> Any:
    model = SimpleNamespace()
    if bridge is not None:
        model.mllm_compression_bridge = bridge
    return model


def _binding(name: str) -> Any:
    return builtin_multimodal_compression_contracts().binding(name)


def _stage(variant: str, **config: Any) -> Any:
    return SimpleNamespace(variant=variant, config=config)


def _factory(
    name: str,
    model: Any,
    *,
    config: dict[str, Any] | None = None,
    phase: ExecutionPhase = ExecutionPhase.GENERATE,
) -> Any:
    binding = _binding(name)
    return adapter.create_llava_onevision_image_compression_executor(
        model=model,
        runtime=SimpleNamespace(),
        facade=SimpleNamespace(),
        stage=_stage(binding.variant, **(config or {})),
        binding=binding,
        binding_config=None,
        execution_phase=phase,
    )


def test_binding_contracts_pin_exact_source_tree_sites_without_active_claim() -> None:
    contracts = adapter.LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS

    assert tuple(contracts) == (
        "tome.image.llava_onevision",
        "saint.image.llava_onevision",
        "aim.image.llava_onevision",
        "pact.image.llava_onevision",
        "visionzip.image.llava_onevision",
        "prumerge.image.llava_onevision",
        "gap.image.llava_onevision",
        "pyramid_drop.image.llava_onevision",
        "feather.image.llava_onevision",
        "ivc_prune.image.llava_onevision",
        "divprune.image.llava_onevision",
        "mmtok.image.llava_onevision",
        "vispruner.image.llava_onevision",
        "zoo_prune.image.llava_onevision",
        "nuwa.image.llava_onevision",
        "restore.image.llava_onevision",
        "fastv.image.llava_onevision",
        "sgl.image.llava_onevision",
        "dart.image.llava_onevision",
        "d2pruner.image.llava_onevision",
        "sparsevlm.image.llava_onevision",
        "finepruner.image.llava_onevision",
        "btp.image.llava_onevision",
        "halfv.image.llava_onevision",
        "transprune.image.llava_onevision",
    )
    assert contracts["tome.image.llava_onevision"].sites == (
        adapter.TOME_MERGE_SITE,
        adapter.TOME_RESTORE_SITE,
        adapter.TOME_PACK_SITE,
    )
    assert contracts["visionzip.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
    )
    assert contracts["pyramid_drop.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    assert contracts["divprune.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
    )
    assert contracts["mmtok.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
    )
    assert contracts["vispruner.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
    )
    assert contracts["zoo_prune.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
    )
    assert contracts["nuwa.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
        adapter.FASTV_SITE,
    )
    assert "build one-dimensional RPME positions" in contracts[
        "nuwa.image.llava_onevision"
    ].callbacks[0].required_repairs
    assert "compact every completed DynamicCache prefix layer" in contracts[
        "nuwa.image.llava_onevision"
    ].callbacks[1].required_repairs
    assert contracts["restore.image.llava_onevision"].sites == (
        adapter.VISIONZIP_SITE,
        adapter.RESTORE_CALIBRATION_SITE,
    )
    assert contracts["fastv.image.llava_onevision"].sites == (adapter.FASTV_SITE,)
    assert contracts["sgl.image.llava_onevision"].sites == (adapter.FASTV_SITE,)
    assert contracts["dart.image.llava_onevision"].sites == (adapter.FASTV_SITE,)
    assert contracts["d2pruner.image.llava_onevision"].sites == (adapter.FASTV_SITE,)
    assert contracts["sparsevlm.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    assert contracts["finepruner.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    assert contracts["btp.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    assert contracts["halfv.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    assert contracts["transprune.image.llava_onevision"].sites == (
        adapter.PYRAMID_DROP_SITE,
    )
    transprune_repairs = contracts[
        "transprune.image.llava_onevision"
    ].callbacks[0].required_repairs
    assert any("rebuild contiguous text positions" in item for item in transprune_repairs)
    assert any("layer-local DynamicCache" in item for item in transprune_repairs)
    assert (
        "allow the final stage to remove the complete visual span"
        in contracts["btp.image.llava_onevision"].callbacks[0].required_repairs
    )
    assert (
        "preserve completed shallow DynamicCache layer lengths"
        in contracts["finepruner.image.llava_onevision"].callbacks[0].required_repairs
    )
    assert "LLaVA-NeXT" in adapter.LLAVA_ONEVISION_SOURCE_ROUTE
    for contract in contracts.values():
        assert contract.image_only is True
        assert contract.source_route == adapter.LLAVA_ONEVISION_SOURCE_ROUTE


@pytest.mark.parametrize(
    ("binding", "phase"),
    (
        ("prumerge.image.llava_onevision", ExecutionPhase.FORWARD),
        ("gap.image.llava_onevision", ExecutionPhase.FORWARD),
        ("feather.image.llava_onevision", ExecutionPhase.GENERATE),
        ("ivc_prune.image.llava_onevision", ExecutionPhase.GENERATE),
    ),
)
def test_new_llava_factories_accept_framework_runtime_arguments(
    binding: str,
    phase: ExecutionPhase,
) -> None:
    activation = _factory(binding, _model(), phase=phase)
    assert activation.contract.binding == binding


def test_invalid_model_fails_when_pinned_provider_auto_install_is_impossible() -> None:
    activation = _factory(
        "visionzip.image.llava_onevision",
        _model(),
    )
    with pytest.raises(BackendIntegrationError, match="could not resolve.*install"):
        activation.install()

    bridge = _Bridge(capabilities={"image_newline_indices": False})
    activation = _factory(
        "visionzip.image.llava_onevision",
        _model(bridge),
    )
    with pytest.raises(BackendIntegrationError, match="image_newline_indices"):
        activation.install()
    assert all(not callbacks for callbacks in bridge.callbacks.values())


def test_nuwa_installs_one_composite_activation_at_both_native_sites() -> None:
    bridge = _Bridge()
    activation = _factory(
        "nuwa.image.llava_onevision",
        _model(bridge),
        config={
            "stage2_after_layer": 2,
            "semantic_hidden_start": 0,
            "semantic_hidden_end": 3,
        },
    )

    activation.install()

    assert len(bridge.callbacks[adapter.VISIONZIP_SITE]) == 1
    assert len(bridge.callbacks[adapter.FASTV_SITE]) == 1
    assert bridge.requires_packed_metrics[adapter.VISIONZIP_SITE] is False
    assert activation.runtime_metadata()["binding"] == (
        "nuwa.image.llava_onevision"
    )
    assert activation.runtime_metadata()["real_checkpoint_verified"] is False

    activation.uninstall()
    assert not bridge.callbacks[adapter.VISIONZIP_SITE]
    assert not bridge.callbacks[adapter.FASTV_SITE]


@pytest.mark.parametrize(
    ("config", "budget_policy"),
    [
        ({"tokens_per_crop": 2}, "tokens_per_crop"),
        ({"retain_ratio": 0.5}, "retain_ratio"),
    ],
)
def test_mmtok_selects_each_crop_then_legalizes_native_anyres_layout(
    config: dict[str, Any],
    budget_policy: str,
) -> None:
    torch = pytest.importorskip("torch")
    bridge = _Bridge()
    activation = _factory(
        "mmtok.image.llava_onevision",
        _model(bridge),
        config=config,
    )
    activation.install()
    projected = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]]]
    )
    tokens = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    response = bridge.emit(
        adapter.VISIONZIP_SITE,
        adapter.LlavaPackedImageRequest(
            invocation_id="mmtok-anyres-1",
            image_index=0,
            tokens=tokens,
            attention=None,
            keys=None,
            image_sizes=((32, 32),),
            modalities=("image",),
            newline_indices=(4,),
            anyres_layout={
                "packed_after_anyres": True,
                "pre_llm_splice": True,
                "image_index": 0,
                "packed_token_count": 5,
                "newline_indices": (4,),
                "crop_boundaries": ((0, 5),),
            },
            projected_crops=projected,
            vision_crops=projected.clone(),
            text_embeddings=torch.tensor([[1.0, 0.0]]),
            patches_per_side=2,
            patch_merge_type="spatial_unpad",
        ),
    )

    assert response.tokens.shape == (3, 2)
    assert response.layout_delta.source_count == 5
    assert response.layout_delta.result_count == 3
    assert response.layout_delta.protected_indices == (4,)
    assert response.newline_indices == (2,)
    assert len(response.metadata["selected_indices_per_crop"][0]) == 2
    assert response.metadata["budget_policy"] == budget_policy
    activation.uninstall()


def test_mmtok_rejects_conflicting_budget_keys_before_provider_install() -> None:
    bridge = _Bridge()
    with pytest.raises(BackendIntegrationError, match="accepts one budget"):
        _factory(
            "mmtok.image.llava_onevision",
            _model(bridge),
            config={"tokens_per_crop": 2, "retain_ratio": 0.5},
        )
    assert all(not callbacks for callbacks in bridge.callbacks.values())


def test_vispruner_selects_each_crop_then_legalizes_native_anyres_layout() -> None:
    torch = pytest.importorskip("torch")
    bridge = _Bridge()
    activation = _factory(
        "vispruner.image.llava_onevision",
        _model(bridge),
        config={"tokens_per_crop": 2},
    )
    activation.install()
    projected = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
    vision = torch.tensor(
        [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]]
    )
    attention = torch.tensor(
        [[[[0.1, 0.2, 0.3, 0.4]] * 4, [[0.4, 0.3, 0.2, 0.1]] * 4]]
    )
    response = bridge.emit(
        adapter.VISIONZIP_SITE,
        adapter.LlavaPackedImageRequest(
            invocation_id="vispruner-anyres-1",
            image_index=0,
            tokens=torch.arange(10, dtype=torch.float32).reshape(5, 2),
            attention=None,
            keys=None,
            image_sizes=((32, 32),),
            modalities=("image",),
            newline_indices=(4,),
            anyres_layout={
                "packed_after_anyres": True,
                "pre_llm_splice": True,
                "image_index": 0,
                "packed_token_count": 5,
                "newline_indices": (4,),
                "crop_boundaries": ((0, 5),),
            },
            projected_crops=projected,
            vision_crops=vision,
            vision_attentions=attention,
            patches_per_side=2,
            patch_merge_type="spatial_unpad",
        ),
    )

    assert response.layout_delta.source_count == 5
    assert response.layout_delta.result_count == 3
    assert response.layout_delta.protected_indices == (4,)
    assert response.newline_indices == (2,)
    assert len(response.metadata["selected_indices_per_crop"][0]) == 2
    assert response.metadata["attention_source"] == "siglip_final_mean_received"
    activation.uninstall()


def test_zoo_prune_uses_native_projector_and_legalizes_anyres_layout() -> None:
    torch = pytest.importorskip("torch")
    bridge = _Bridge()
    activation = _factory(
        "zoo_prune.image.llava_onevision",
        _model(bridge),
        config={
            "target_count": 2,
            "num_directions": 2,
            "projector_proxy": "full",
            "direction_batch_size": 1,
        },
    )
    activation.install()
    projector_layers = (
        torch.nn.Linear(2, 2),
        torch.nn.GELU(),
        torch.nn.Linear(2, 2),
    )
    vision = torch.tensor(
        [[[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]]
    )
    projected = vision
    for layer in projector_layers:
        projected = layer(projected)
    packed = torch.cat((projected[0], torch.zeros(1, 2)), dim=0)
    response = bridge.emit(
        adapter.VISIONZIP_SITE,
        adapter.LlavaPackedImageRequest(
            invocation_id="zoo-prune-anyres-1",
            image_index=0,
            tokens=packed,
            attention=None,
            keys=None,
            image_sizes=((32, 32),),
            modalities=("image",),
            newline_indices=(4,),
            anyres_layout={
                "packed_after_anyres": True,
                "pre_llm_splice": True,
                "image_index": 0,
                "packed_token_count": 5,
                "newline_indices": (4,),
                "crop_boundaries": ((0, 5),),
            },
            projected_crops=projected,
            vision_crops=vision,
            patches_per_side=2,
            patch_merge_type="spatial_unpad",
            projector_layers=projector_layers,
        ),
    )

    assert response.layout_delta.source_count == 5
    assert response.layout_delta.result_count == 3
    assert response.newline_indices == (2,)
    assert response.metadata["training_free"] is True
    assert response.metadata["low_rank_solver_used_per_crop"] == ["full"]
    assert len(response.metadata["selected_indices_per_crop"][0]) == 2
    activation.uninstall()


def test_decoder_config_typo_fails_before_provider_install() -> None:
    bridge = _Bridge()
    with pytest.raises(BackendIntegrationError, match="retain_rato"):
        _factory(
            "dart.image.llava_onevision",
            _model(bridge),
            config={"retain_rato": 0.5},
        )
    assert all(not callbacks for callbacks in bridge.callbacks.values())


def test_tome_hook_tracks_crop_layout_and_dense_restores_before_anyres(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _Bridge()
    activation = _factory(
        "tome.image.llava_onevision",
        _model(bridge),
        config={"r": 1},
    )
    delta = LayoutDelta.from_mapping(
        source_count=4,
        old_to_new={0: 0, 1: 1, 2: 1, 3: 2},
        media_unit="image_patch",
        result_layout={"ordering": "fixture"},
    )

    def fake_tome(hidden_states: Any, keys: Any, **kwargs: Any) -> Any:
        assert hidden_states == [[10], [20], [30], [40]]
        assert keys == "keys"
        assert kwargs["r"] == 1
        return SimpleNamespace(
            hidden_states=[[10], [25], [40]],
            token_sizes=[[1], [2], [1]],
            proportional_attention_bias="bias",
            layout_deltas=(delta,),
            source_groups=(((0,), (1, 2), (3,)),),
            metadata={"algorithm": "tome"},
        )

    monkeypatch.setattr(adapter, "tome_merge_image", fake_tome)
    activation.install()

    merged = bridge.emit(
        adapter.TOME_MERGE_SITE,
        adapter.LlavaEncoderMergeRequest(
            invocation_id="encode-1",
            hidden_states=[[10], [20], [30], [40]],
            keys="keys",
            layer_index=3,
            stream_ids=("sample-0/crop-0",),
        ),
    )
    assert merged.hidden_states == [[10], [25], [40]]
    assert merged.metadata["dense_restore_pending"] is True

    restored = bridge.emit(
        adapter.TOME_RESTORE_SITE,
        adapter.LlavaEncoderDenseRestoreRequest(
            invocation_id="encode-1",
            hidden_states=merged.hidden_states,
            stream_ids=("sample-0/crop-0",),
        ),
    )
    assert restored.hidden_states == [[10], [25], [25], [40]]
    assert restored.dense_to_compact == ((0, 1, 1, 2),)
    assert restored.source_token_count == 4
    assert restored.metadata["pre_anyres_shape_repaired"] is True
    metadata = activation.runtime_metadata()
    assert metadata["callback_counts"] == {
        adapter.TOME_MERGE_SITE: 1,
        adapter.TOME_RESTORE_SITE: 1,
        adapter.TOME_PACK_SITE: 0,
    }
    assert metadata["last_layout"]["dense_restored"] is True
    assert metadata["real_checkpoint_verified"] is False
    assert metadata["support_activation_claimed"] is False

    activation.uninstall()
    assert bridge.callbacks[adapter.TOME_MERGE_SITE] == []
    assert bridge.callbacks[adapter.TOME_RESTORE_SITE] == []


def test_visionzip_repairs_packed_count_and_newline_without_changing_batch_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _Bridge()
    activation = _factory(
        "visionzip.image.llava_onevision",
        _model(bridge),
        config={"budget": 3},
    )
    delta = LayoutDelta.from_mapping(
        source_count=4,
        old_to_new={0: 0, 1: 1, 2: 1, 3: 2},
        media_unit="image_patch",
    )

    def fake_visionzip(tokens: Any, attention: Any, keys: Any, **kwargs: Any) -> Any:
        assert tokens == [[0], [1], [2], [3]]
        assert attention == "attention"
        assert keys == "keys"
        assert kwargs["protected_indices"] == (3,)
        assert kwargs["budget"] == 3
        return SimpleNamespace(
            tokens=[[0], [1.5], [3]],
            layout_delta=delta,
            metadata={"algorithm": "visionzip"},
        )

    monkeypatch.setattr(adapter, "visionzip_image", fake_visionzip)
    activation.install()
    assert bridge.requires_packed_metrics[adapter.VISIONZIP_SITE] is True
    image_sizes = ((640, 480),)
    modalities = ("image",)
    response = bridge.emit(
        adapter.VISIONZIP_SITE,
        adapter.LlavaPackedImageRequest(
            invocation_id="pack-1",
            image_index=0,
            tokens=[[0], [1], [2], [3]],
            attention="attention",
            keys="keys",
            image_sizes=image_sizes,
            modalities=modalities,
            newline_indices=(3,),
            anyres_layout={
                "packed_after_anyres": True,
                "pre_llm_splice": True,
                "image_index": 0,
                "packed_token_count": 4,
                "newline_indices": (3,),
                "crop_boundaries": ((0, 4),),
            },
        ),
    )
    assert response.tokens == [[0], [1.5], [3]]
    assert response.image_sizes is image_sizes
    assert response.modalities is modalities
    assert response.newline_indices == (2,)
    assert response.anyres_layout["packed_token_count"] == 3
    assert response.metadata["image_newline_protected"] is True

    with pytest.raises(BackendIntegrationError, match="image-only"):
        bridge.emit(
            adapter.VISIONZIP_SITE,
            adapter.LlavaPackedImageRequest(
                invocation_id="pack-video",
                image_index=0,
                tokens=[[0], [1], [2], [3]],
                attention="attention",
                keys="keys",
                image_sizes=image_sizes,
                modalities=("video",),
                newline_indices=(3,),
                anyres_layout={
                    "packed_after_anyres": True,
                    "pre_llm_splice": True,
                    "image_index": 0,
                    "packed_token_count": 4,
                    "newline_indices": (3,),
                    "crop_boundaries": ((0, 4),),
                },
            ),
        )
    activation.uninstall()
    assert bridge.callbacks[adapter.VISIONZIP_SITE] == []


def test_divprune_selects_packed_features_and_preserves_newline_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bridge = _Bridge()
    activation = _factory(
        "divprune.image.llava_onevision",
        _model(bridge),
        config={"target_count": 3},
    )
    delta = LayoutDelta.drop(
        4,
        (1,),
        media_unit="image_feature",
        protected_indices=(3,),
    )

    def fake_divprune(tokens: Any, **kwargs: Any) -> Any:
        assert tokens == [[0], [1], [2], [3]]
        assert kwargs["spec"].target_count == 3
        assert kwargs["protected_indices"] == (3,)
        return SimpleNamespace(
            tokens=[[0], [2], [3]],
            layout_delta=delta,
            metadata={
                "algorithm": "divprune",
                "effective_target_count": 3,
            },
        )

    monkeypatch.setattr(adapter, "divprune_image", fake_divprune)
    activation.install()
    assert bridge.requires_packed_metrics[adapter.VISIONZIP_SITE] is False
    response = bridge.emit(
        adapter.VISIONZIP_SITE,
        adapter.LlavaPackedImageRequest(
            invocation_id="divprune-1",
            image_index=0,
            tokens=[[0], [1], [2], [3]],
            attention=None,
            keys=None,
            image_sizes=((640, 480),),
            modalities=("image",),
            newline_indices=(3,),
            anyres_layout={
                "packed_after_anyres": True,
                "pre_llm_splice": True,
                "image_index": 0,
                "packed_token_count": 4,
                "newline_indices": (3,),
                "crop_boundaries": ((0, 4),),
            },
        ),
    )

    assert response.tokens == [[0], [2], [3]]
    assert response.newline_indices == (2,)
    assert response.anyres_layout["packed_token_count"] == 3
    assert response.anyres_layout["crop_boundaries"] == ((0, 3),)
    assert response.layout_delta.result_layout == response.anyres_layout
    assert response.layout_delta.result_layout["newline_indices"] == (2,)
    assert response.metadata["algorithm"] == "divprune"
    metadata = activation.runtime_metadata()
    assert metadata["last_layout"]["planned_count"] == 3
    assert metadata["last_layout"]["selected_indices"] == [0, 2, 3]
    activation.uninstall()
    assert bridge.callbacks[adapter.VISIONZIP_SITE] == []


def test_saint_and_aim_preserve_llava_anyres_newline_layout() -> None:
    torch = pytest.importorskip("torch")
    tokens = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )
    layout = {
        "packed_after_anyres": True,
        "pre_llm_splice": True,
        "image_index": 0,
        "packed_token_count": 4,
        "newline_indices": (3,),
        "crop_boundaries": ((0, 4),),
    }
    request = adapter.LlavaPackedImageRequest(
        invocation_id="new-methods-1",
        image_index=0,
        tokens=tokens,
        attention=None,
        keys=tokens.unsqueeze(0).expand(2, -1, -1),
        image_sizes=((640, 480),),
        modalities=("image",),
        newline_indices=(3,),
        anyres_layout=layout,
    )
    for name, config in (
        ("saint.image.llava_onevision", {"target_count": 3, "group_size": 1}),
        ("aim.image.llava_onevision", {"target_count": 3}),
    ):
        bridge = _Bridge()
        activation = _factory(
            name,
            _model(bridge),
            config=config,
            phase=ExecutionPhase.FORWARD,
        )
        activation.install()
        response = bridge.emit(adapter.VISIONZIP_SITE, request)
        assert response.tokens.shape[0] == 3
        assert response.newline_indices == (2,)
        assert response.anyres_layout["crop_boundaries"] == ((0, 3),)
        assert response.metadata["image_newline_protected"] is True
        assert response.layout_delta.old_to_new[3] == 2
        activation.uninstall()


@pytest.mark.parametrize("identity", [False, True])
def test_pact_repairs_llava_sequence_and_returns_cluster_mass(identity) -> None:
    torch = pytest.importorskip("torch")
    bridge = _Bridge()
    activation = _factory(
        "pact.image.llava_onevision",
        _model(bridge),
        config={"reduction_layer": 2, "target_keep_ratio": 1.0 if identity else 0.5,
            "cutoff": 0.0 if identity else None},
    )
    activation.install()
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    query_states = torch.randn(2, 6, 2)
    key_states = torch.randn(2, 6, 2)
    source_mask = torch.ones(1, 6)
    source_positions = torch.arange(6).view(1, 6)

    def repair_layout(**kwargs: Any) -> adapter.LlavaDecoderLayoutRepair:
        delta = kwargs["layout_delta"]
        kept = kwargs["kept_indices"].long()
        source_images = kwargs["image_indices"].tolist()
        source_newlines = kwargs["newline_indices"]
        return adapter.LlavaDecoderLayoutRepair(
            attention_mask=source_mask.index_select(-1, kept),
            position_ids=source_positions.index_select(-1, kept),
            past_key_values=kwargs["past_key_values"],
            image_indices=torch.tensor(
                [delta.old_to_new[int(value)] for value in source_images],
                dtype=torch.long,
            ),
            newline_indices=tuple(
                delta.old_to_new[int(value)] for value in source_newlines
            ),
            sequence_length=delta.result_count,
            metadata={"cache_policy": "layer_local"},
        )

    response = bridge.emit(
        adapter.PYRAMID_DROP_SITE,
        adapter.LlavaPACTRequest(
            invocation_id="pact-1",
            layer_index=1,
            num_layers=4,
            hidden_states=hidden,
            image_indices=torch.tensor([1, 2, 3, 4]),
            query_states=query_states,
            key_states=key_states,
            clustering_keys=torch.eye(6) if identity else key_states.transpose(0, 1).flatten(1),
            attention_mask=source_mask,
            position_ids=source_positions,
            past_key_values=None,
            use_cache=False,
            image_sizes=((640, 480),),
            modalities=("image",),
            newline_indices=(4,),
            repair_layout=repair_layout,
        ),
    )
    assert response.dropped is not identity
    assert response.layer_index == 1
    assert response.hidden_states.shape[0] == response.token_sizes.shape[0]
    assert float(response.token_sizes.sum().item()) == 6.0
    assert response.newline_indices[0] < response.hidden_states.shape[0]
    assert response.metadata["image_newline_protected"] is True
    activation.uninstall()


def test_pyramid_drop_delegates_layer_local_cache_validation_to_shared_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch = pytest.importorskip("torch")
    bridge = _Bridge()
    activation = _factory(
        "pyramid_drop.image.llava_onevision",
        _model(bridge),
        config={"keep_ratio": 0.5, "num_stages": 4},
    )
    delta = LayoutDelta.from_mapping(
        source_count=4,
        old_to_new={0: 0, 1: None, 2: 1, 3: 2},
        media_unit="llm_token",
        protected_indices=(0, 2, 3),
    )
    kernel_calls: list[dict[str, Any]] = []

    def fake_pyramid(hidden_states: Any, **kwargs: Any) -> Any:
        kernel_calls.append(kwargs)
        assert kwargs["protected_indices"] == (2,)
        return SimpleNamespace(
            hidden_states=hidden_states[[0, 2, 3]],
            kept_indices=[0, 2, 3],
            dropped_indices=[1],
            kept_image_indices=[2],
            layout_delta=delta,
            metadata={"algorithm": "pyramid_drop"},
        )

    monkeypatch.setattr(decoder_selection, "pyramid_drop", fake_pyramid)
    activation.install()
    old_cache = object()
    repair_calls: list[dict[str, Any]] = []

    def repair_layout(**kwargs: Any) -> adapter.LlavaDecoderLayoutRepair:
        repair_calls.append(kwargs)
        return adapter.LlavaDecoderLayoutRepair(
            attention_mask=[1, 1, 1],
            position_ids=[0, 1, 2],
            past_key_values=old_cache,
            image_indices=[1],
            newline_indices=(1,),
            sequence_length=3,
            metadata={
                "provider_repair": "fake_qwen2_decoder_loop",
                "cache_policy": "layer_local",
                "cache_validation_owner": "shared_decoder_lifecycle",
            },
        )

    request = adapter.LlavaDecoderLayerRequest(
        invocation_id="decode-1",
        layer_index=0,
        num_layers=4,
        hidden_states=torch.arange(4).reshape(4, 1),
        image_indices=torch.tensor([1, 2]),
        instruction_query="query",
        image_keys="keys",
        attention_mask=[1, 1, 1, 1],
        position_ids=[0, 1, 2, 3],
        past_key_values=old_cache,
        use_cache=True,
        image_sizes=((640, 480),),
        modalities=("image",),
        newline_indices=(2,),
        repair_layout=repair_layout,
    )
    response = bridge.emit(adapter.PYRAMID_DROP_SITE, request)
    assert response.dropped is True
    assert response.hidden_states.tolist() == [[0], [2], [3]]
    assert response.attention_mask == [1, 1, 1]
    assert response.position_ids == [0, 1, 2]
    assert response.past_key_values is old_cache
    assert response.image_indices == [1]
    assert response.newline_indices == (1,)
    assert response.image_sizes is request.image_sizes
    assert response.modalities is request.modalities
    assert len(kernel_calls) == len(repair_calls) == 1
    assert response.metadata["mask_position_cache_repaired"] is True

    no_drop = bridge.emit(
        adapter.PYRAMID_DROP_SITE,
        adapter.LlavaDecoderLayerRequest(**{**request.__dict__, "layer_index": 3}),
    )
    assert no_drop.dropped is False
    assert no_drop.layout_delta is None
    assert len(kernel_calls) == 1

    activation.uninstall()
    assert bridge.callbacks[adapter.PYRAMID_DROP_SITE] == []


def test_generate_pyramid_install_requires_explicit_kv_cache_capability() -> None:
    bridge = _Bridge(capabilities={"kv_cache_repair": False})
    activation = _factory(
        "pyramid_drop.image.llava_onevision",
        _model(bridge),
    )
    with pytest.raises(BackendIntegrationError, match="kv_cache_repair"):
        activation.install()



def test_tome_public_callback_checks_the_final_nonzero_budget_source():
    binding = _binding("tome.image.llava_onevision")
    stage = _stage(binding.variant, r=0)
    stage.criterion = {"score_layer": 1}
    activation = adapter.create_llava_onevision_image_compression_executor(
        model=_model(), runtime=SimpleNamespace(), facade=SimpleNamespace(),
        stage=stage, binding=binding, binding_config=SimpleNamespace(config={"r": 1}),
        execution_phase=ExecutionPhase.FORWARD,
    )
    with pytest.raises(ValueError, match="future layer"):
        activation.merge_encoder_tokens(adapter.LlavaEncoderMergeRequest(
            invocation_id="direct", hidden_states=[[10], [20], [30], [40]],
            keys="unused", layer_index=0, stream_ids=("crop-0",),
        ))
    assert not activation.runtime_metadata()["layout_delta_records"]
