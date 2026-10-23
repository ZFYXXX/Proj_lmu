"""Pinned-provider lifecycle tests for the Qwen3-VL compression bridge."""

from __future__ import annotations

import inspect
import importlib
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen3_vl.compression_adapter import (
    QWEN3_VL_IMAGE_BINDING_CONTRACTS,
    create_qwen3_vl_image_compression_executor,
)
from mllm_comp.backbones.qwen3_vl import provider_bridge as provider
from mllm_comp.backbones._vision_attention import streaming_vision_attention
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import LayoutDelta, OrderingPolicy
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
def test_layout_composition_preserves_explicit_aim_order(backbone) -> None:
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    first = LayoutDelta.identity(4, media_unit="image_feature")
    aim_delta = LayoutDelta.from_mapping(
        source_count=4,
        result_count=2,
        old_to_new=(1, 0, 0, 1),
        media_unit="image_feature",
        result_layout={"representative_indices": (1, 0)},
        ordering_policy=OrderingPolicy.EXPLICIT,
    )

    composed = provider._compose_layout_mapping(first, aim_delta)

    assert composed.ordering_policy is OrderingPolicy.EXPLICIT
    assert tuple(composed.old_to_new.values()) == (1, 0, 0, 1)
    assert provider._layout_representatives(composed) == (1, 0)


class _Callable:
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return args[0] if args else None


class _Config:
    image_token_id = 10
    rope_scaling = {"mrope_section": [1, 0, 0]}


class Qwen3VLVisionModel:
    def __init__(self, *, deepstack_count: int = 3) -> None:
        self.blocks = [SimpleNamespace() for _ in range(8)]
        self.merger = _Callable()
        self.deepstack_visual_indexes = list(range(deepstack_count))
        self.deepstack_merger_list = [_Callable() for _ in range(deepstack_count)]
        self.patch_embed = _Callable()
        self.fast_pos_embed_interpolate = _Callable()
        self.rot_pos_emb = _Callable()
        self.spatial_merge_size = 2
        self.spatial_merge_unit = 4

    def forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        **kwargs: Any,
    ) -> Any:
        return hidden_states, []


class Qwen3VLTextModel:
    def __init__(self, *, layer_count: int = 12) -> None:
        self.layers = [SimpleNamespace() for _ in range(layer_count)]
        self.norm = _Callable()
        self.rotary_emb = _Callable()
        self.embed_tokens = _Callable()
        self.config = _Config()

    def _deepstack_process(
        self,
        hidden_states: Any,
        visual_pos_masks: Any,
        visual_embeds: Any,
    ) -> Any:
        return hidden_states

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        cache_position: Any = None,
        visual_pos_masks: Any = None,
        deepstack_visual_embeds: Any = None,
        **kwargs: Any,
    ) -> Any:
        return SimpleNamespace(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class Qwen3VLModel:
    def __init__(self, *, deepstack_count: int = 3, layer_count: int = 12) -> None:
        self.visual = Qwen3VLVisionModel(deepstack_count=deepstack_count)
        self.language_model = Qwen3VLTextModel(layer_count=layer_count)
        self.config = _Config()

    def get_image_features(self, pixel_values: Any, image_grid_thw: Any = None) -> Any:
        return pixel_values, []

    def get_video_features(
        self,
        pixel_values_videos: Any,
        video_grid_thw: Any = None,
    ) -> Any:
        return pixel_values_videos, []

    def get_placeholder_mask(
        self,
        input_ids: Any,
        inputs_embeds: Any,
        image_features: Any = None,
        video_features: Any = None,
    ) -> Any:
        return None, None

    def get_rope_index(self, *args: Any, **kwargs: Any) -> Any:
        return None

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> Any:
        return SimpleNamespace(last_hidden_state=inputs_embeds)


class Qwen3VLForConditionalGeneration:
    def __init__(self, *, deepstack_count: int = 3, layer_count: int = 12) -> None:
        self.model = Qwen3VLModel(
            deepstack_count=deepstack_count,
            layer_count=layer_count,
        )
        self.forward_calls: list[dict[str, Any]] = []

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        labels: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> Any:
        self.forward_calls.append(
            {
                "past_key_values": past_key_values,
                "labels": labels,
                **kwargs,
            }
        )
        return "dense-forward"

    def generate(self, *args: Any, **kwargs: Any) -> str:
        del args, kwargs
        return "dense-generate"


@pytest.fixture(autouse=True)
def _pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider,
        "_installed_transformers_version",
        lambda: provider.SUPPORTED_TRANSFORMERS_VERSION,
    )


def _binding(name: str) -> Any:
    contract = QWEN3_VL_IMAGE_BINDING_CONTRACTS[name]
    return SimpleNamespace(
        name=name,
        backbone="qwen3_vl",
        variant=contract.variant,
        target_modalities=frozenset({Modality.IMAGE}),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}),
    )


def _stage(
    name: str,
    *,
    schedule: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    contract = QWEN3_VL_IMAGE_BINDING_CONTRACTS[name]
    return SimpleNamespace(
        name=f"provider-{contract.algorithm}",
        algorithm=contract.algorithm,
        variant=contract.variant,
        target_modalities={Modality.IMAGE},
        operation={},
        criterion={},
        schedule=dict(schedule or {}),
        config=dict(config or {}),
    )


def _activation(
    model: Any,
    name: str,
    *,
    phase: ExecutionPhase = ExecutionPhase.FORWARD,
    schedule: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
) -> Any:
    return create_qwen3_vl_image_compression_executor(
        model=model,
        runtime=SimpleNamespace(
            metadata={},
            model_path_or_id="/fixtures/unit-test-model",
        ),
        facade=None,
        stage=_stage(name, schedule=schedule, config=config),
        binding=_binding(name),
        binding_config=SimpleNamespace(config={}),
        execution_phase=phase,
    )


@pytest.mark.parametrize("algorithm", ["dart", "d2pruner", "sparsevlm"])
def test_single_image_decoder_methods_reject_native_multi_image_before_selection(algorithm, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from tests.fixtures import native_image

    model = native_image.qwen_model("qwen3_vl").eval()
    configs = {
        "dart": {"prune_after_layer": 2, "target_visual_tokens": 2,
                 "pivot_image_tokens": 1, "pivot_text_tokens": 1},
        "sparsevlm": {"prune_layers": (1, 2, 3), "schedule_keep_tokens": (2, 1, 1),
                      "schedule_base_visual_tokens": 4, "scale_schedule_to_input": False},
    }
    if algorithm == "d2pruner":
        prior = tmp_path / "prior.pt"
        torch.save({"bias": torch.ones(4), "sample_count": 4,
                    "model_path": "/fixtures/unit-test-model", "scorer_layer": 1}, prior)
        config = {"prune_after_layer": 2, "retain_ratio": 0.5, "bias_prior_path": str(prior)}
    else:
        config = configs[algorithm]
    activation = _activation(model, f"{algorithm}.image.qwen3_vl",
                             phase=ExecutionPhase.GENERATE, config=config)
    activation.install()
    ids = torch.tensor([[1, 7, 10, 10, 10, 10, 8, 2, 7, 10, 10, 10, 10, 8, 3]])
    try:
        with torch.no_grad(), pytest.raises(BackendIntegrationError, match="single-image only"):
            model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True,
                  max_new_tokens=1, do_sample=False,
                  pixel_values=torch.randn(32, 12), image_grid_thw=torch.tensor([[1, 4, 4]] * 2))
        assert not activation.runtime_metadata().get("layout_delta_records")
    finally:
        activation.uninstall()


def test_bridge_install_is_idempotent_and_restores_every_instance_patch() -> None:
    model = Qwen3VLForConditionalGeneration()
    original_signature = inspect.signature(model.forward)
    original = {
        "model": model.forward.__func__,
        "generate": model.generate.__func__,
        "core": model.model.forward.__func__,
        "visual": model.model.visual.forward.__func__,
        "language": model.model.language_model.forward.__func__,
    }

    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)

    assert provider.install_qwen3_vl_transformers_4573_bridge(model) is bridge
    assert bridge.installed is True
    assert inspect.signature(model.forward) == original_signature
    assert callable(model.model.visual.register_encoder_dense_boundary_controller)
    assert callable(model.model.visual.register_post_visual_merger_controller)
    assert callable(model.model.language_model.register_decoder_layer_loop_controller)
    assert bridge.runtime_metadata()["real_checkpoint_verified"] is False

    provider.uninstall_qwen3_vl_transformers_4573_bridge(model)

    assert bridge.installed is False
    assert model.forward.__func__ is original["model"]
    assert model.generate.__func__ is original["generate"]
    assert model.model.forward.__func__ is original["core"]
    assert model.model.visual.forward.__func__ is original["visual"]
    assert model.model.language_model.forward.__func__ is original["language"]
    assert not hasattr(
        model.model.visual,
        "register_encoder_dense_boundary_controller",
    )
    assert not hasattr(model, "_mllm_comp_qwen3_vl_provider_bridge")


def test_bridge_registers_multiple_activations_and_removes_them_independently() -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    tome = _activation(model, "tome.image.qwen3_vl")
    visionzip = _activation(model, "visionzip.image.qwen3_vl")
    pyramid = _activation(
        model,
        "pyramid_drop.image.qwen3_vl",
        schedule={"layer_indices": [3, 6, 9], "num_stages": 4},
    )

    tome.install()
    visionzip.install()
    pyramid.install()
    metadata = bridge.runtime_metadata()
    assert metadata["registration_count"] == 3
    assert {item["algorithm"] for item in metadata["registrations"]} == {
        "tome",
        "visionzip",
        "pyramid_drop",
    }
    with pytest.raises(BackendIntegrationError, match="activations remain"):
        bridge.uninstall()

    visionzip.uninstall()
    assert bridge.runtime_metadata()["registration_count"] == 2
    pyramid.uninstall()
    tome.uninstall()
    assert bridge.runtime_metadata()["registration_count"] == 0
    bridge.uninstall()


def test_post_merger_registration_rejects_metric_demand_mismatch() -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    controller = SimpleNamespace(
        execution_phase=ExecutionPhase.FORWARD,
        requires_visual_metrics=False,
    )

    with pytest.raises(BackendIntegrationError, match="disagree"):
        model.model.visual.register_post_visual_merger_controller(
            controller,
            target_modality=Modality.IMAGE.value,
            synchronize_deepstack=True,
            requires_visual_metrics=True,
        )

    assert bridge.runtime_metadata()["registration_count"] == 0
    bridge.uninstall()


def test_generate_registration_is_decoder_only_and_other_cached_forward_fails() -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    generated = _activation(
        model,
        "pyramid_drop.image.qwen3_vl",
        phase=ExecutionPhase.GENERATE,
        schedule={"layer_indices": [3, 6, 9], "num_stages": 4},
    )

    generated.install()
    assert generated.installed is True
    assert bridge.runtime_metadata()["active_generate_supported"] is True
    assert bridge.runtime_metadata()["supported_phases"] == ["forward", "generate"]
    assert bridge.runtime_metadata()["use_cache_by_phase"] == {
        "forward": False,
        "generate": True,
    }
    generated.uninstall()

    forward = _activation(model, "visionzip.image.qwen3_vl")
    forward.install()
    with pytest.raises(BackendIntegrationError, match="cached generation"):
        model.forward(use_cache=True)
    with pytest.raises(BackendIntegrationError, match="cached generation"):
        model.forward(use_cache=False, past_key_values=object())
    assert model.forward(use_cache=False) == "dense-forward"

    forward.uninstall()
    bridge.uninstall()


@pytest.mark.parametrize("mixed", [False, True])
def test_image_controller_rejects_video_only_and_mixed_inputs(mixed: bool) -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    activation = _activation(model, "dart.image.qwen3_vl")
    activation.install()

    with pytest.raises(BackendIntegrationError, match="image-only"):
        model.forward(
            use_cache=False,
            pixel_values=object() if mixed else None,
            image_grid_thw=object() if mixed else None,
            pixel_values_videos=object(),
            video_grid_thw=object(),
        )

    activation.uninstall()
    bridge.uninstall()


def test_training_forward_rejects_provider_output_without_logits() -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    activation = _activation(model, "visionzip.image.qwen3_vl")
    activation.install()

    with pytest.raises(InvariantError, match="rank-3 logits"):
        model.forward(use_cache=False, labels=object())

    activation.uninstall()
    bridge.uninstall()


def test_force_uninstall_evicts_live_registration_handles_and_restores_model() -> None:
    model = Qwen3VLForConditionalGeneration()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    activation = _activation(model, "tome.image.qwen3_vl")
    activation.install()

    bridge.uninstall(force=True)

    assert bridge.installed is False
    assert not hasattr(
        model.model.visual,
        "register_encoder_dense_boundary_controller",
    )
    # The activation's handle is now an idempotent no-op.
    activation.uninstall()
    assert activation.installed is False


def test_version_and_structure_guards_fail_before_mutating_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Qwen3VLForConditionalGeneration()
    monkeypatch.setattr(
        provider,
        "_installed_transformers_version",
        lambda: "4.58.0",
    )

    with pytest.raises(BackendIntegrationError, match="source-pinned"):
        provider.install_qwen3_vl_transformers_4573_bridge(model)
    assert not hasattr(model, "_mllm_comp_qwen3_vl_provider_bridge")
    assert not hasattr(
        model.model.visual,
        "register_encoder_dense_boundary_controller",
    )

    monkeypatch.setattr(
        provider,
        "_installed_transformers_version",
        lambda: provider.SUPPORTED_TRANSFORMERS_VERSION,
    )
    model.model.visual.deepstack_merger_list.pop()
    with pytest.raises(BackendIntegrationError, match="counts differ"):
        provider.install_qwen3_vl_transformers_4573_bridge(model)
    assert not hasattr(model, "_mllm_comp_qwen3_vl_provider_bridge")


def test_tome_boundary_is_between_attention_residual_and_mlp() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn
    events: list[tuple[str, int]] = []

    class _QKV(nn.Linear):
        def forward(self, hidden_states: Any) -> Any:
            events.append(("attention", int(hidden_states.shape[0])))
            return super().forward(hidden_states)

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scaling = 2**-0.5
            self.qkv = _QKV(4, 12, bias=False)
            self.proj = nn.Identity()
            with torch.no_grad():
                self.qkv.weight.copy_(torch.eye(4).repeat(3, 1))

    class _MLP(nn.Module):
        def forward(self, hidden_states: Any) -> Any:
            events.append(("mlp", int(hidden_states.shape[0])))
            return torch.zeros_like(hidden_states)

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.attn = _Attention()
            self.mlp = _MLP()

        def forward(self, *args: Any, **kwargs: Any) -> Any:
            raise AssertionError("ToMe must split the provider block before its MLP")

    class _Merger(nn.Module):
        def forward(self, hidden_states: Any) -> Any:
            return hidden_states.reshape(-1, 4, 4).mean(1)

    model = Qwen3VLForConditionalGeneration(deepstack_count=0, layer_count=8)
    visual = model.model.visual
    visual.blocks = nn.ModuleList([_Block(), _Block()])
    visual.deepstack_visual_indexes = []
    visual.deepstack_merger_list = nn.ModuleList()
    visual.merger = _Merger()
    visual.patch_embed = nn.Identity()
    visual.fast_pos_embed_interpolate = lambda grid: torch.zeros(8, 4)
    visual.rot_pos_emb = lambda grid: torch.zeros(8, 1)
    visual.spatial_merge_size = 2
    visual.spatial_merge_unit = 4
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    activation = _activation(
        model,
        "tome.image.qwen3_vl",
        config={"r": 1},
    )
    activation.install()

    main, deepstack = bridge._execute_image_visual_forward(
        torch.arange(32, dtype=torch.float32).reshape(8, 4),
        torch.tensor([[1, 2, 4]]),
        {},
        tome_controllers=(activation,),
        visionzip_controllers=(),
    )

    assert events == [
        ("attention", 8),
        ("mlp", 7),
        ("attention", 7),
        ("mlp", 6),
    ]
    assert main.shape == (1, 4)
    assert deepstack == []
    plan = bridge._visual_plan.get()
    assert plan is not None
    assert plan.source_counts == (2,)
    assert plan.result_counts == (1,)
    assert plan.layout_deltas[0].media_unit == "image_feature"
    assert plan.layout_deltas[0].ordering_policy.value == "preserve_source"
    activation.uninstall()
    bridge.uninstall()


def test_tome_token_mass_is_added_to_subsequent_attention_logits() -> None:
    torch = pytest.importorskip("torch")
    query = torch.zeros(2, 1, 1)
    key = torch.zeros_like(query)
    value = torch.tensor([[[10.0]], [[0.0]]])

    dense_output, dense_received = streaming_vision_attention(
        query,
        key,
        value,
        scaling=1.0,
        token_sizes=None,
        capture_mean_received=True,
        query_chunk_size=1,
    )
    proportional_output, proportional_received = streaming_vision_attention(
        query,
        key,
        value,
        scaling=1.0,
        token_sizes=torch.tensor([[2.0], [1.0]]),
        capture_mean_received=True,
        query_chunk_size=1,
    )

    assert torch.allclose(dense_output, torch.full_like(dense_output, 5.0))
    assert dense_received is not None
    assert torch.allclose(dense_received, torch.tensor([0.5, 0.5]))
    assert torch.allclose(
        proportional_output,
        torch.full_like(proportional_output, 20.0 / 3.0),
    )
    assert proportional_received is not None
    assert torch.allclose(
        proportional_received,
        torch.tensor([2.0 / 3.0, 1.0 / 3.0]),
    )


def test_visionzip_captures_final_block_qkv_once_before_mlp() -> None:
    torch = pytest.importorskip("torch")
    nn = torch.nn

    class _CountingQKV(nn.Linear):
        def __init__(self) -> None:
            super().__init__(4, 12, bias=False)
            self.calls = 0
            with torch.no_grad():
                self.weight.copy_(torch.eye(4).repeat(3, 1))

        def forward(self, hidden_states: Any) -> Any:
            self.calls += 1
            return super().forward(hidden_states)

    class _Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.num_heads = 2
            self.scaling = 2**-0.5
            self.qkv = _CountingQKV()
            self.proj = nn.Identity()

    class _RecordingIdentity(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_input: Any | None = None

        def forward(self, hidden_states: Any) -> Any:
            self.last_input = hidden_states.detach().clone()
            return hidden_states

    class _Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm1 = nn.Identity()
            self.norm2 = nn.Identity()
            self.attn = _Attention()
            self.mlp = _RecordingIdentity()
            self.forward_calls = 0

        def forward(
            self,
            hidden_states: Any,
            cu_seqlens: Any,
            position_embeddings: Any,
            **kwargs: Any,
        ) -> Any:
            del cu_seqlens, position_embeddings, kwargs
            self.forward_calls += 1
            # A post-block QKV reconstruction observes this shifted state and
            # calls qkv a second time; the semantic bridge must do neither.
            self.attn.qkv(self.norm1(hidden_states))
            return hidden_states + 100.0

    class _Merger(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.hidden_size = 16
            self.use_postshuffle_norm = False
            self.norm = nn.Identity()
            self.linear_fc1 = nn.Linear(16, 16, bias=False)
            self.act_fn = nn.Identity()
            self.linear_fc2 = nn.Linear(16, 4, bias=False)
            with torch.no_grad():
                self.linear_fc1.weight.copy_(torch.eye(16))
                self.linear_fc2.weight.zero_()
                for output_index in range(4):
                    self.linear_fc2.weight[
                        output_index, output_index::4
                    ] = 0.25

        def forward(self, hidden_states: Any) -> Any:
            hidden_states = self.norm(hidden_states).reshape(-1, self.hidden_size)
            return self.linear_fc2(self.act_fn(self.linear_fc1(hidden_states)))

    model = Qwen3VLForConditionalGeneration(deepstack_count=1, layer_count=8)
    visual = model.model.visual
    block = _Block()
    visual.blocks = nn.ModuleList([block])
    visual.deepstack_visual_indexes = [0]
    visual.deepstack_merger_list = nn.ModuleList([_Merger()])
    visual.merger = _Merger()
    visual.patch_embed = nn.Identity()
    visual.fast_pos_embed_interpolate = lambda grid: torch.zeros(
        int(grid.prod(-1).sum().item()), 4
    )
    visual.rot_pos_emb = lambda grid: torch.zeros(int(grid.prod(-1).sum().item()), 1)
    visual.spatial_merge_size = 2
    visual.spatial_merge_unit = 4
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    activation = _activation(
        model,
        "visionzip.image.qwen3_vl",
        config={"budget": 2},
    )
    captured: dict[str, Any] = {}
    original_compress = activation.compress_after_visual_mergers

    def record_metrics(*args: Any, **kwargs: Any) -> Any:
        captured["attention"] = kwargs["attention"]
        captured["keys"] = kwargs["keys"]
        return original_compress(*args, **kwargs)

    activation.compress_after_visual_mergers = record_metrics
    activation.install()
    source = torch.arange(64, dtype=torch.float32).reshape(16, 4)

    main, deepstack = bridge._execute_image_visual_forward(
        source,
        torch.tensor([[1, 4, 4]]),
        {},
        tome_controllers=(),
        visionzip_controllers=(activation,),
    )

    expected_keys = (
        source.reshape(16, 2, 2).reshape(4, 4, 2, 2).mean(dim=1).permute(1, 0, 2)
    )
    assert block.forward_calls == 0
    assert block.attn.qkv.calls == 1
    assert torch.allclose(captured["keys"], expected_keys)
    assert captured["attention"].shape == (4, 4)
    assert captured["attention"].stride(0) == 0
    assert main.shape == (2, 4)
    assert deepstack[0].shape == (2, 4)
    activation.uninstall()

    divprune = _activation(
        model,
        "divprune.image.qwen3_vl",
        config={"target_count": 2},
    )
    divprune.install()
    two_image_source = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    div_main, div_deepstack = bridge._execute_image_visual_forward(
        two_image_source,
        torch.tensor([[1, 4, 4], [1, 4, 4]]),
        {},
        tome_controllers=(),
        visionzip_controllers=(divprune,),
    )
    # DivPrune needs only merged features, so the provider executes the native
    # visual block and does not enter the metric-capture path a second time.
    assert block.forward_calls == 1
    assert block.attn.qkv.calls == 2
    assert div_main.shape == (4, 4)
    assert div_deepstack[0].shape == (4, 4)
    records = divprune.runtime_metadata()["layout_delta_records"]
    assert [record["image_index"] for record in records] == [0, 1]
    assert all(record["source_count"] == 4 for record in records)
    assert all(record["result_count"] == 2 for record in records)
    assert all(record["operation"] == "drop" for record in records)
    assert [record["source_layout"]["image_index"] for record in records] == [0, 1]
    divprune.uninstall()

    mmtok = _activation(
        model,
        "mmtok.image.qwen3_vl",
        config={"target_count": 2},
    )
    captured_reference: dict[str, Any] = {}
    original_mmtok = mmtok.compress_after_visual_mergers

    def record_mmtok_reference(*args: Any, **kwargs: Any) -> Any:
        captured_reference["value"] = kwargs["vision_reference"].detach().clone()
        return original_mmtok(*args, **kwargs)

    mmtok.compress_after_visual_mergers = record_mmtok_reference
    mmtok.install()
    text_token = bridge._prompt_text_embeddings.set(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    )
    first_group = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ]
    )
    second_group = torch.tensor(
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, -1.0],
        ]
    )
    mmtok_source = torch.cat(
        (first_group, second_group, torch.eye(4), -torch.eye(4)), dim=0
    )
    try:
        mmtok_main, mmtok_deepstack = bridge._execute_image_visual_forward(
            mmtok_source,
            torch.tensor([[1, 4, 4]]),
            {},
            tome_controllers=(),
            visionzip_controllers=(mmtok,),
        )
    finally:
        bridge._prompt_text_embeddings.reset(text_token)
    assert mmtok_main.shape == (2, 4)
    assert mmtok_deepstack[0].shape == (2, 4)
    latest = mmtok.runtime_metadata()["layout_delta_records"][-1]
    assert latest["source_count"] == 4
    assert latest["result_count"] == 2
    assert latest["source_layout"]["vision_source"] == (
        "pre_merger_visual_features"
    )
    assert latest["shared_main_deepstack_delta"] is True
    reference = captured_reference["value"]
    assert reference.shape == (4, 16)
    torch.testing.assert_close(
        reference[0].reshape(4, 4).mean(dim=0),
        reference[1].reshape(4, 4).mean(dim=0),
    )
    assert not torch.equal(reference[0], reference[1])
    mmtok.uninstall()

    vispruner = _activation(
        model,
        "vispruner.image.qwen3_vl",
        config={"target_count": 2},
    )
    vispruner.install()
    vispruner_main, vispruner_deepstack = bridge._execute_image_visual_forward(
        source,
        torch.tensor([[1, 4, 4]]),
        {},
        tome_controllers=(),
        visionzip_controllers=(vispruner,),
    )
    assert vispruner_main.shape == (2, 4)
    assert vispruner_deepstack[0].shape == (2, 4)
    vispruner_record = vispruner.runtime_metadata()["layout_delta_records"][-1]
    assert vispruner_record["source_count"] == 4
    assert vispruner_record["result_count"] == 2
    assert vispruner_record["shared_main_deepstack_delta"] is True
    assert vispruner_record["source_layout"]["attention_source"] == (
        "final_visual_mean_received"
    )
    vispruner.uninstall()

    zoo_prune = _activation(
        model,
        "zoo_prune.image.qwen3_vl",
        config={
            "target_count": 2,
            "num_directions": 2,
            "direction_batch_size": 1,
            "projector_proxy": "full",
        },
    )
    captured_projector: dict[str, Any] = {}
    original_zoo_prune = zoo_prune.compress_after_visual_mergers

    def record_zoo_projector(*args: Any, **kwargs: Any) -> Any:
        captured_projector["inputs"] = kwargs["projector_inputs"].detach().clone()
        captured_projector["layers"] = kwargs["projector_layers"]
        return original_zoo_prune(*args, **kwargs)

    zoo_prune.compress_after_visual_mergers = record_zoo_projector
    zoo_prune.install()
    zoo_main, zoo_deepstack = bridge._execute_image_visual_forward(
        source,
        torch.tensor([[1, 4, 4]]),
        {},
        tome_controllers=(),
        visionzip_controllers=(zoo_prune,),
    )
    assert captured_projector["inputs"].shape == (4, 16)
    assert captured_projector["layers"] == (
        visual.merger.linear_fc1,
        visual.merger.act_fn,
        visual.merger.linear_fc2,
    )
    assert zoo_main.shape == (2, 4)
    assert zoo_deepstack[0].shape == (2, 4)
    zoo_record = zoo_prune.runtime_metadata()["layout_delta_records"][-1]
    assert zoo_record["source_count"] == 4
    assert zoo_record["result_count"] == 2
    assert zoo_record["shared_main_deepstack_delta"] is True
    assert zoo_record["source_layout"]["projector_input"] == (
        "native_final_merger_mlp_input"
    )
    zoo_prune.uninstall()

    nuwa = _activation(
        model,
        "nuwa.image.qwen3_vl",
        config={
            "stage1_total_tokens": 2,
            "stage2_after_layer": 3,
            "stage2_retain_ratio": 0.5,
            "region_size": 1,
            "top_n_per_region": 1,
            "semantic_hidden_start": 0,
            "semantic_hidden_end": 2,
        },
    )
    captured_nuwa_semantic: dict[str, Any] = {}
    original_nuwa = nuwa.compress_after_visual_mergers

    def record_nuwa_semantic(*args: Any, **kwargs: Any) -> Any:
        captured_nuwa_semantic["value"] = kwargs[
            "semantic_reference"
        ].detach().clone()
        return original_nuwa(*args, **kwargs)

    nuwa.compress_after_visual_mergers = record_nuwa_semantic
    nuwa.install()
    nuwa_main, nuwa_deepstack = bridge._execute_image_visual_forward(
        source,
        torch.tensor([[1, 4, 4]]),
        {},
        tome_controllers=(),
        visionzip_controllers=(nuwa,),
    )
    assert nuwa_main.shape == (2, 4)
    assert nuwa_deepstack[0].shape == (2, 4)
    nuwa_record = nuwa.runtime_metadata()["layout_delta_records"][-1]
    assert (nuwa_record["source_count"], nuwa_record["result_count"]) == (4, 2)
    assert nuwa_record["shared_main_deepstack_delta"] is True
    assert nuwa_record["result_layout"]["position_policy"] == (
        "native_sparse_mrope_rpme"
    )
    assert block.mlp.last_input is not None
    expected_nuwa_semantic = (
        source + 2.0 * block.mlp.last_input
    ) / 2.0
    torch.testing.assert_close(
        captured_nuwa_semantic["value"],
        expected_nuwa_semantic.reshape(4, 16),
    )
    nuwa.uninstall()

    restore = _activation(
        model,
        "restore.image.qwen3_vl",
        config={"target_count": 2},
    )
    restore.install()
    restore_runtime: dict[str, Any] = {}
    restore_token = bridge._restore_runtime.set(restore_runtime)
    try:
        restore_main, restore_deepstack = bridge._execute_image_visual_forward(
            source,
            torch.tensor([[1, 4, 4]]),
            {},
            tome_controllers=(),
            visionzip_controllers=(restore,),
        )
    finally:
        bridge._restore_runtime.reset(restore_token)
    assert restore_main.shape == (2, 4)
    assert restore_deepstack[0].shape == (2, 4)
    restore_record = restore.runtime_metadata()["layout_delta_records"][-1]
    assert (restore_record["source_count"], restore_record["result_count"]) == (
        4,
        2,
    )
    assert restore_record["shared_main_deepstack_delta"] is True
    assert sum(restore_record["result_layout"]["token_sizes"]) == 4
    restore_plan = bridge._visual_plan.get()
    assert restore_plan is not None
    assert tuple(
        restore_plan.layout_deltas[0].result_layout["representative_indices"]
    ) == tuple(restore_record["result_layout"]["representative_indices"])
    assert restore_runtime["source_visual_count"] == 4
    restore.uninstall()
    bridge.uninstall()
