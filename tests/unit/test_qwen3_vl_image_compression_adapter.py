"""CPU/fake-model contracts for Qwen3-VL ImageLLM compression adapters."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen3_vl.compression_adapter import (
    QWEN3_VL_IMAGE_BINDING_CONTRACTS,
    Qwen3VLDivPruneActivation,
    Qwen3VLPyramidDropActivation,
    Qwen3VLToMeActivation,
    Qwen3VLVisionZipActivation,
    create_qwen3_vl_image_compression_executor,
)
from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


class _Handle:
    def __init__(self, remove: Any) -> None:
        self._remove = remove
        self.removed = False

    def remove(self) -> None:
        if self.removed:
            return
        self.removed = True
        self._remove()


class _VisualHookHost:
    def __init__(self, *, deepstack_count: int = 3) -> None:
        self.blocks = [object() for _ in range(8)]
        self.merger = object()
        self.deepstack_merger_list = [object() for _ in range(deepstack_count)]
        self.controllers: dict[str, Any] = {}
        self.registration_kwargs: dict[str, dict[str, Any]] = {}

    def register_encoder_dense_boundary_controller(
        self,
        controller: Any,
        **kwargs: Any,
    ) -> _Handle:
        return self._register("tome", controller, kwargs)

    def register_post_visual_merger_controller(
        self,
        controller: Any,
        **kwargs: Any,
    ) -> _Handle:
        return self._register("visionzip", controller, kwargs)

    def _register(
        self,
        name: str,
        controller: Any,
        kwargs: dict[str, Any],
    ) -> _Handle:
        assert name not in self.controllers
        self.controllers[name] = controller
        self.registration_kwargs[name] = dict(kwargs)
        return _Handle(lambda: self.controllers.pop(name, None))


class _DecoderLoopHookHost:
    def __init__(self, *, layer_count: int = 12) -> None:
        self.layers = [object() for _ in range(layer_count)]
        self.controller: Any | None = None
        self.registration_kwargs: dict[str, Any] = {}

    def register_decoder_layer_loop_controller(
        self,
        controller: Any,
        **kwargs: Any,
    ) -> _Handle:
        assert self.controller is None
        self.controller = controller
        self.registration_kwargs = dict(kwargs)
        return _Handle(lambda: setattr(self, "controller", None))


def _model(
    *,
    deepstack_count: int = 3,
    layer_count: int = 12,
) -> tuple[Any, _VisualHookHost, _DecoderLoopHookHost]:
    visual = _VisualHookHost(deepstack_count=deepstack_count)
    language_model = _DecoderLoopHookHost(layer_count=layer_count)
    model = SimpleNamespace(
        model=SimpleNamespace(
            visual=visual,
            language_model=language_model,
        )
    )
    return model, visual, language_model


def _binding(name: str, *, modality: Modality = Modality.IMAGE) -> Any:
    contract = QWEN3_VL_IMAGE_BINDING_CONTRACTS[name]
    return SimpleNamespace(
        name=name,
        backbone="qwen3_vl",
        variant=contract.variant,
        target_modalities=frozenset({modality}),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}),
    )


def _stage(
    contract_name: str,
    *,
    schedule: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    contract = QWEN3_VL_IMAGE_BINDING_CONTRACTS[contract_name]
    return SimpleNamespace(
        name=f"fixture-{contract.algorithm}",
        algorithm=contract.algorithm,
        variant=contract.variant,
        target_modalities={Modality.IMAGE},
        operation={},
        criterion={},
        schedule=dict(schedule or {}),
        config=dict(config or {}),
    )


def _executor(
    name: str,
    *,
    model: Any,
    phase: ExecutionPhase = ExecutionPhase.GENERATE,
    schedule: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    rules: dict[str, Any] | None = None,
) -> Any:
    stage = _stage(name, schedule=schedule, config=config)
    if rules:
        stage.operation["rules"] = rules
    return create_qwen3_vl_image_compression_executor(
        model=model,
        runtime=SimpleNamespace(metadata={}),
        facade=None,
        stage=stage,
        binding=_binding(name),
        binding_config=SimpleNamespace(config={}),
        execution_phase=phase,
    )


def test_binding_contracts_are_exact_image_only_and_do_not_claim_video() -> None:
    assert set(QWEN3_VL_IMAGE_BINDING_CONTRACTS) == {
        "tome.image.qwen3_vl",
        "visionzip.image.qwen3_vl",
        "pyramid_drop.image.qwen3_vl",
        "divprune.image.qwen3_vl",
        "saint.image.qwen3_vl",
        "aim.image.qwen3_vl",
        "pact.image.qwen3_vl",
        "fastv.image.qwen3_vl",
        "sgl.image.qwen3_vl",
        "dart.image.qwen3_vl",
        "d2pruner.image.qwen3_vl",
        "sparsevlm.image.qwen3_vl",
        "finepruner.image.qwen3_vl",
        "btp.image.qwen3_vl",
        "halfv.image.qwen3_vl",
        "transprune.image.qwen3_vl",
        "mmtok.image.qwen3_vl",
        "vispruner.image.qwen3_vl",
        "zoo_prune.image.qwen3_vl",
        "nuwa.image.qwen3_vl",
        "restore.image.qwen3_vl",
        "prumerge.image.qwen3_vl",
        "gap.image.qwen3_vl",
        "feather.image.qwen3_vl",
        "ivc_prune.image.qwen3_vl",
    }
    tome = QWEN3_VL_IMAGE_BINDING_CONTRACTS["tome.image.qwen3_vl"]
    visionzip = QWEN3_VL_IMAGE_BINDING_CONTRACTS["visionzip.image.qwen3_vl"]
    pyramid = QWEN3_VL_IMAGE_BINDING_CONTRACTS["pyramid_drop.image.qwen3_vl"]
    divprune = QWEN3_VL_IMAGE_BINDING_CONTRACTS["divprune.image.qwen3_vl"]
    fastv = QWEN3_VL_IMAGE_BINDING_CONTRACTS["fastv.image.qwen3_vl"]
    sgl = QWEN3_VL_IMAGE_BINDING_CONTRACTS["sgl.image.qwen3_vl"]
    dart = QWEN3_VL_IMAGE_BINDING_CONTRACTS["dart.image.qwen3_vl"]
    d2pruner = QWEN3_VL_IMAGE_BINDING_CONTRACTS["d2pruner.image.qwen3_vl"]
    sparsevlm = QWEN3_VL_IMAGE_BINDING_CONTRACTS["sparsevlm.image.qwen3_vl"]
    finepruner = QWEN3_VL_IMAGE_BINDING_CONTRACTS["finepruner.image.qwen3_vl"]
    btp = QWEN3_VL_IMAGE_BINDING_CONTRACTS["btp.image.qwen3_vl"]
    halfv = QWEN3_VL_IMAGE_BINDING_CONTRACTS["halfv.image.qwen3_vl"]
    transprune = QWEN3_VL_IMAGE_BINDING_CONTRACTS["transprune.image.qwen3_vl"]
    mmtok = QWEN3_VL_IMAGE_BINDING_CONTRACTS["mmtok.image.qwen3_vl"]
    vispruner = QWEN3_VL_IMAGE_BINDING_CONTRACTS["vispruner.image.qwen3_vl"]
    zoo_prune = QWEN3_VL_IMAGE_BINDING_CONTRACTS["zoo_prune.image.qwen3_vl"]
    nuwa = QWEN3_VL_IMAGE_BINDING_CONTRACTS["nuwa.image.qwen3_vl"]
    restore = QWEN3_VL_IMAGE_BINDING_CONTRACTS["restore.image.qwen3_vl"]

    assert tome.callback == "encoder_dense_boundary"
    assert tome.target_sites[-3:] == (
        "visual_pre_deepstack_mergers",
        "visual_output_pre_final_merger",
        "post_visual_mergers_pre_llm_scatter_tome",
    )
    assert visionzip.callback == "post_visual_merger_with_deepstack"
    assert visionzip.metadata["shared_main_deepstack_delta"] is True
    assert pyramid.callback == "decoder_layer_loop_after_last_deepstack"
    assert pyramid.metadata["requires_decoder_loop_control"] is True
    assert divprune.callback == "post_visual_merger_with_deepstack"
    assert divprune.metadata["shared_main_deepstack_delta"] is True
    assert divprune.requires_visual_metrics is False
    assert fastv.callback == "decoder_layer_loop"
    assert fastv.metadata["cache_policy"] == "compact_prefix"
    assert fastv.metadata["remaining_deepstack_synchronized"] is True
    assert sgl.callback == "decoder_layer_loop"
    assert sgl.metadata["dual_model_lifecycle"] is True
    assert sgl.metadata["single_image_only"] is True
    assert dart.callback == "decoder_layer_loop"
    assert dart.metadata["single_image_only"] is True
    assert d2pruner.callback == "decoder_layer_loop"
    assert d2pruner.metadata["requires_calibrated_bias_prior"] is True
    assert sparsevlm.callback == "decoder_layer_loop_after_deepstack"
    assert sparsevlm.metadata["cache_policy"] == "compact_prefix"
    assert sparsevlm.metadata["single_image_only"] is True
    assert finepruner.callback == "decoder_layer_loop_after_deepstack"
    assert finepruner.metadata["cache_policy"] == "layer_local"
    assert finepruner.metadata["attention_capture"] == "last_text_query_per_head"
    assert btp.callback == "decoder_layer_loop_after_deepstack"
    assert btp.metadata["cache_policy"] == "compact_prefix"
    assert btp.metadata["final_stage"] == "remove_remaining_visual_span"
    assert halfv.callback == "decoder_layer_loop_after_deepstack"
    assert halfv.metadata["cache_policy"] == "compact_prefix"
    assert halfv.metadata["position_policy"] == "preserve_original"
    assert halfv.metadata["remaining_deepstack_synchronized"] is True
    assert transprune.callback == "decoder_layer_loop_after_deepstack"
    assert transprune.metadata["cache_policy"] == "layer_local"
    assert transprune.metadata["position_policy"] == "contiguous"
    assert transprune.metadata["synchronize_remaining_deepstack"] is True
    assert transprune.metadata["iga_source"] == "next_layer_post_image_text"
    assert mmtok.callback == "post_visual_merger_with_deepstack"
    assert mmtok.requires_visual_metrics is False
    assert mmtok.metadata["shared_main_deepstack_delta"] is True
    assert vispruner.callback == "post_visual_merger_with_deepstack"
    assert vispruner.requires_visual_metrics is True
    assert vispruner.metadata["shared_main_deepstack_delta"] is True
    assert zoo_prune.callback == "post_visual_merger_with_deepstack"
    assert zoo_prune.requires_visual_metrics is False
    assert zoo_prune.metadata["shared_main_deepstack_delta"] is True
    assert zoo_prune.metadata["training_free"] is True
    assert nuwa.callback == "composite_post_merger_and_decoder_loop"
    assert nuwa.requires_visual_metrics is True
    assert nuwa.metadata["composite_binding"] is True
    assert nuwa.metadata["shared_main_deepstack_delta"] is True
    assert nuwa.metadata["cache_policy"] == "compact_prefix"
    assert nuwa.metadata["position_policy"] == "native_sparse_mrope_rpme"
    assert restore.callback == "composite_post_merger_and_decoder_loop"
    assert all(
        contract.image_only and not contract.supports_video
        for contract in QWEN3_VL_IMAGE_BINDING_CONTRACTS.values()
    )


def test_adapter_contract_sites_match_the_builtin_qwen3_registry_bindings() -> None:
    registry = builtin_multimodal_compression_contracts()

    for name, contract in QWEN3_VL_IMAGE_BINDING_CONTRACTS.items():
        binding = registry.binding(name)
        assert binding.backbone == "qwen3_vl"
        assert binding.variant == contract.variant
        assert (
            tuple(target.target_site for target in binding.ordered_targets())
            == contract.target_sites
        )
        assert callable(binding.executor_factory)


@pytest.mark.parametrize(
    "binding_name",
    [
        "dart.image.qwen3_vl",
        "mmtok.image.qwen3_vl",
        "vispruner.image.qwen3_vl",
        "zoo_prune.image.qwen3_vl",
    ],
)
def test_explicit_config_typo_fails_before_provider_install(
    binding_name: str,
) -> None:
    model, visual, language_model = _model()
    with pytest.raises(BackendIntegrationError, match="retain_rato"):
        _executor(
            binding_name,
            model=model,
            config={"retain_rato": 0.5},
        )
    assert visual.controllers == {}
    assert language_model.controller is None


def test_factories_install_and_uninstall_all_semantic_hook_contracts() -> None:
    model, visual, llm = _model(deepstack_count=3, layer_count=12)
    tome = _executor("tome.image.qwen3_vl", model=model)
    visionzip = _executor("visionzip.image.qwen3_vl", model=model)
    pyramid = _executor(
        "pyramid_drop.image.qwen3_vl",
        model=model,
        schedule={"layer_indices": [3, 6, 9], "num_stages": 4},
    )

    assert isinstance(tome, Qwen3VLToMeActivation)
    assert isinstance(visionzip, Qwen3VLVisionZipActivation)
    assert isinstance(pyramid, Qwen3VLPyramidDropActivation)
    assert visual.controllers == {}
    assert llm.controller is None

    tome.install()
    visionzip.install()
    pyramid.install()
    assert visual.controllers == {"tome": tome, "visionzip": visionzip}
    assert llm.controller is pyramid
    assert visual.registration_kwargs["tome"]["restore_before"] == (
        "deepstack_merger[0]",
        "deepstack_merger[1]",
        "deepstack_merger[2]",
        "final_merger",
    )
    assert visual.registration_kwargs["visionzip"] == {
        "target_modality": "image",
        "synchronize_deepstack": True,
        "requires_visual_metrics": True,
    }
    assert llm.registration_kwargs["requires_mask_position_repair"] is True
    assert llm.registration_kwargs["requires_kv_cache_repair"] is True

    # Installation is idempotent and every hook remains removable.
    tome.install()
    visionzip.install()
    pyramid.install()
    pyramid.uninstall()
    visionzip.uninstall()
    tome.uninstall()
    assert visual.controllers == {}
    assert llm.controller is None
    for activation in (tome, visionzip, pyramid):
        metadata = activation.runtime_metadata()
        assert metadata["installed"] is False
        assert metadata["target_modality"] == "image"
        assert metadata["real_checkpoint_verified"] is False
        assert metadata["support_lifecycle_mutated"] is False

    nuwa_model, nuwa_visual, nuwa_llm = _model(
        deepstack_count=3,
        layer_count=8,
    )
    nuwa = _executor(
        "nuwa.image.qwen3_vl",
        model=nuwa_model,
        config={
            "stage2_after_layer": 2,
            "semantic_hidden_start": 0,
            "semantic_hidden_end": 3,
        },
    )
    nuwa.install()
    assert nuwa_visual.controllers == {"visionzip": nuwa}
    assert nuwa_llm.controller is nuwa
    assert nuwa_visual.registration_kwargs["visionzip"] == {
        "target_modality": "image",
        "synchronize_deepstack": True,
        "requires_visual_metrics": True,
        "requires_semantic_hidden_range": True,
    }
    assert nuwa_llm.registration_kwargs["requires_kv_cache_repair"] is True
    nuwa.uninstall()
    assert nuwa_visual.controllers == {}
    assert nuwa_llm.controller is None

    div_model, div_visual, _ = _model(deepstack_count=2)
    divprune = _executor(
        "divprune.image.qwen3_vl",
        model=div_model,
        config={"target_count": 2},
    )
    assert isinstance(divprune, Qwen3VLDivPruneActivation)
    divprune.install()
    assert div_visual.controllers == {"visionzip": divprune}
    assert div_visual.registration_kwargs["visionzip"] == {
        "target_modality": "image",
        "synchronize_deepstack": True,
        "requires_visual_metrics": False,
    }
    divprune.uninstall()
    assert div_visual.controllers == {}


def test_invalid_model_fails_when_pinned_provider_auto_install_is_impossible() -> None:
    model, _, _ = _model()
    model.model.visual.register_post_visual_merger_controller = None
    activation = _executor("visionzip.image.qwen3_vl", model=model)

    with pytest.raises(
        BackendIntegrationError,
        match="transformers==4.57.3|Pinned Qwen3-VL bridge requires",
    ):
        activation.install()

    assert activation.installed is False


def test_nuwa_rolls_back_visual_registration_when_decoder_registration_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, visual, language_model = _model(deepstack_count=1, layer_count=8)
    activation = _executor(
        "nuwa.image.qwen3_vl",
        model=model,
        config={
            "stage2_after_layer": 2,
            "semantic_hidden_start": 0,
            "semantic_hidden_end": 3,
        },
    )

    def fail_decoder_registration(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise RuntimeError("decoder registration failed")

    monkeypatch.setattr(
        language_model,
        "register_decoder_layer_loop_controller",
        fail_decoder_registration,
    )
    with pytest.raises(RuntimeError, match="decoder registration failed"):
        activation.install()

    assert activation.installed is False
    assert visual.controllers == {}
    assert language_model.controller is None


def test_video_cannot_reuse_an_image_support_unit() -> None:
    model, _, _ = _model()
    binding = _binding("visionzip.image.qwen3_vl", modality=Modality.VIDEO)

    with pytest.raises(BackendIntegrationError, match="video requires.*separate"):
        create_qwen3_vl_image_compression_executor(
            model=model,
            stage=_stage("visionzip.image.qwen3_vl"),
            binding=binding,
            execution_phase=ExecutionPhase.GENERATE,
        )


def test_visionzip_replays_one_layout_delta_on_main_and_every_deepstack() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    activation = _executor(
        "visionzip.image.qwen3_vl",
        model=model,
        config={"budget": 2},
    )
    main = torch.tensor([[0.0, 0.0], [1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
    deepstack = (main + 10.0, main * 3.0)
    attention = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )
    keys = torch.tensor([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0], [0.2, 0.8]])

    result = activation.compress_image_features(
        main,
        deepstack,
        attention=attention,
        keys=keys,
        image_index=1,
    )

    assert result.main_image_features.shape[0] == 2
    assert [branch.shape[0] for branch in result.deepstack_image_features] == [
        2,
        2,
    ]
    assert all(delta is result.layout_delta for delta in result.branch_deltas)
    assert len(result.branch_deltas) == 3
    groups = [
        [
            source
            for source in range(result.layout_delta.source_count)
            if result.layout_delta.old_to_new[source] == target
        ]
        for target in range(result.layout_delta.result_count)
    ]
    for source_branch, compact_branch in zip(
        deepstack,
        result.deepstack_image_features,
    ):
        representatives = result.kernel_result.indices.tolist()
        contextual = set(result.kernel_result.metadata["contextual_indices"])
        expected = torch.stack(
            [
                (
                    source_branch[representative]
                    + source_branch[
                        [index for index in group if index != representative]
                    ].mean(dim=0)
                    if representative in contextual and len(group) > 1
                    else source_branch[representative]
                )
                for representative, group in zip(representatives, groups)
            ],
            dim=0,
        )
        torch.testing.assert_close(compact_branch, expected)
    assert result.metadata["shared_main_deepstack_delta"] is True

    with pytest.raises(BackendIntegrationError, match="image-only"):
        activation.compress_image_features(
            main,
            deepstack,
            attention=attention,
            keys=keys,
            image_index=1,
            target_modality=Modality.VIDEO,
        )


def test_divprune_replays_selected_indices_on_every_deepstack_branch() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    activation = _executor(
        "divprune.image.qwen3_vl",
        model=model,
        config={"target_count": 2},
    )
    main = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]])
    deepstack = (main + 10.0, main * 3.0)

    result = activation.compress_image_features(
        main,
        deepstack,
        attention=None,
        keys=None,
        image_index=2,
    )

    selected = result.kernel_result.indices
    torch.testing.assert_close(
        result.main_image_features, main.index_select(0, selected)
    )
    for source, compact in zip(deepstack, result.deepstack_image_features):
        torch.testing.assert_close(compact, source.index_select(0, selected))
    assert result.layout_delta.operation == "drop"
    assert result.metadata["selected_indices"] == selected.tolist()
    assert result.metadata["image_index"] == 2
    assert result.layout_delta.source_layout["image_index"] == 2
    assert result.layout_delta.result_layout["image_index"] == 2
    assert all(delta is result.layout_delta for delta in result.branch_deltas)


def test_saint_and_aim_share_one_layout_across_qwen3_deepstack() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    main = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )
    branches = (main + 10.0, main * 3.0)
    keys = main.unsqueeze(0).expand(2, -1, -1)

    saint = _executor(
        "saint.image.qwen3_vl",
        model=model,
        phase=ExecutionPhase.FORWARD,
        config={"target_count": 2, "group_size": 1},
    )
    saint_result = saint.compress_image_features(
        main,
        branches,
        keys=keys,
        image_index=0,
    )
    selected = saint_result.kernel_result.selected_indices
    for source, compact in zip(branches, saint_result.deepstack_image_features):
        torch.testing.assert_close(compact, source.index_select(0, selected))
    assert all(
        delta is saint_result.layout_delta for delta in saint_result.branch_deltas
    )

    aim = _executor(
        "aim.image.qwen3_vl",
        model=model,
        phase=ExecutionPhase.FORWARD,
        config={"target_count": 2},
    )
    aim_result = aim.compress_image_features(main, branches, image_index=0)
    groups = [
        [
            source
            for source, target in aim_result.layout_delta.old_to_new.items()
            if target == result_index
        ]
        for result_index in range(aim_result.layout_delta.result_count)
    ]
    for source, compact in zip(branches, aim_result.deepstack_image_features):
        expected = torch.stack([source[group].mean(dim=0) for group in groups])
        torch.testing.assert_close(compact, expected)
    assert all(delta is aim_result.layout_delta for delta in aim_result.branch_deltas)


def _unchanged_aim_scores(request: Any) -> Any:
    return request["scores"]


def test_aim_scorer_only_preserves_deepstack_values_and_gradients() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=1)
    main = torch.tensor([[2.0], [11.0], [4.0], [7.0], [9.0], [3.0], [14.0]])
    original = _executor(
        "aim.image.qwen3_vl", model=model, config={"target_count": 2}
    )
    scorer_only = _executor(
        "aim.image.qwen3_vl", model=model, config={"target_count": 2},
        rules={"scorer": {"callable": f"{__name__}:_unchanged_aim_scores"}},
    )
    branches = [main.clone().requires_grad_() for _ in range(2)]
    expected = original.compress_image_features(main, (branches[0],), image_index=0)
    actual = scorer_only.compress_image_features(main, (branches[1],), image_index=0)
    assert len(actual.kernel_result.merge_rounds) > 1
    assert actual.layout_delta.old_to_new == expected.layout_delta.old_to_new
    torch.testing.assert_close(actual.main_image_features, expected.main_image_features)
    torch.testing.assert_close(
        actual.deepstack_image_features[0], expected.deepstack_image_features[0]
    )
    # Unequal lineage sizes distinguish per-round means from final-member means.
    assert not torch.allclose(expected.main_image_features, expected.deepstack_image_features[0])
    actual.deepstack_image_features[0].square().sum().backward()
    expected.deepstack_image_features[0].square().sum().backward()
    torch.testing.assert_close(branches[1].grad, branches[0].grad)




def test_vispruner_replays_visual_cue_selection_on_every_deepstack_branch() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    activation = _executor(
        "vispruner.image.qwen3_vl",
        model=model,
        config={"target_count": 2},
    )
    main = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    deepstack = (main + 10.0, main * 3.0)
    vision_reference = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )
    attention = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.2, 0.3, 0.4],
        ]
    )

    result = activation.compress_image_features(
        main,
        deepstack,
        attention=attention,
        vision_reference=vision_reference,
        image_index=1,
    )

    selected = result.kernel_result.indices
    torch.testing.assert_close(
        result.main_image_features,
        main.index_select(0, selected),
    )
    for source, compact in zip(deepstack, result.deepstack_image_features):
        torch.testing.assert_close(compact, source.index_select(0, selected))
    assert result.layout_delta.operation == "drop"
    assert result.metadata["selected_indices"] == selected.tolist()
    assert all(delta is result.layout_delta for delta in result.branch_deltas)


def test_zoo_prune_replays_projector_selection_on_every_deepstack_branch() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    activation = _executor(
        "zoo_prune.image.qwen3_vl",
        model=model,
        config={
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
    main = projector_inputs
    for layer in projector_layers:
        main = layer(main)
    deepstack = (main + 10.0, main * 3.0)
    result = activation.compress_image_features(
        main,
        deepstack,
        projector_inputs=projector_inputs,
        projector_layers=projector_layers,
        image_index=0,
    )

    selected = result.kernel_result.indices
    torch.testing.assert_close(result.main_image_features, main[selected])
    for source, compact in zip(deepstack, result.deepstack_image_features):
        torch.testing.assert_close(compact, source[selected])
    assert result.metadata["low_rank_solver_used"] == "full"
    assert result.metadata["training_free"] is True
    assert all(delta is result.layout_delta for delta in result.branch_deltas)


def test_tome_dense_restore_recovers_native_qwen3_merger_cardinality() -> None:
    torch = pytest.importorskip("torch")
    model, _, _ = _model(deepstack_count=2)
    activation = _executor(
        "tome.image.qwen3_vl",
        model=model,
        config={"r": 1},
    )
    hidden = torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0], [6.0, 7.0]])
    keys = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])

    compact = activation.merge_encoder_tokens(hidden, keys)
    dense = activation.restore_dense_boundary(
        compact.hidden_states,
        compact.layout_delta,
    )

    assert compact.hidden_states.shape[0] == 3
    assert dense.shape[0] == hidden.shape[0]
    for source_index in range(hidden.shape[0]):
        result_index = compact.layout_delta.old_to_new[source_index]
        assert result_index is not None
        torch.testing.assert_close(
            dense[source_index],
            compact.hidden_states[result_index],
        )


def test_pyramid_drop_preserves_remaining_deepstack_at_early_boundaries() -> None:
    model, _, _ = _model(deepstack_count=3, layer_count=12)

    early = _executor(
        "pyramid_drop.image.qwen3_vl",
        model=model,
        schedule={"layer_indices": [1, 6, 9], "num_stages": 4},
    )
    assert early.application_positions[0].index == 2

    activation = _executor(
        "pyramid_drop.image.qwen3_vl",
        model=model,
        schedule={"layer_indices": [3, 6, 9], "num_stages": 4},
    )
    assert activation.last_deepstack_injection_layer == 2
    assert activation.first_drop_layer == 3
    assert activation.runtime_metadata()["first_drop_after_deepstack"] is True
