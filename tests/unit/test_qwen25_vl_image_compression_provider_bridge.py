"""Pinned-provider and torch execution tests for the Qwen2.5-VL bridge."""

from __future__ import annotations

import inspect
import json
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from mllm_comp.backbones.qwen25_vl.image_compression import (  # noqa: E402
    QWEN25_VL_IMAGE_BINDING_CONTRACTS,
    create_qwen25_vl_image_compression_executor,
)
from mllm_comp.backbones.qwen25_vl import provider_bridge as provider  # noqa: E402
from mllm_comp.core.errors import BackendIntegrationError  # noqa: E402
from mllm_comp.core.vocabulary import ExecutionPhase, Modality  # noqa: E402


class _VisionAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_heads = 2
        self.scaling = 2**-0.5
        self.qkv = nn.Linear(4, 12, bias=False)
        with torch.no_grad():
            self.qkv.weight.copy_(torch.eye(4).repeat(3, 1))

    def forward(
        self,
        hidden_states: Any,
        cu_seqlens: Any,
        rotary_pos_emb: Any = None,
        position_embeddings: Any = None,
        **kwargs: Any,
    ) -> Any:
        del cu_seqlens, rotary_pos_emb, position_embeddings, kwargs
        return hidden_states * 0.01


class _VisionBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.Identity()
        self.norm2 = nn.Identity()
        self.attn = _VisionAttention()
        self.mlp = nn.Identity()


class _Merger(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_source_count: int | None = None
        self.hidden_size = 16
        self.ln_q = nn.Identity()
        first = nn.Linear(16, 16, bias=False)
        second = nn.Linear(16, 4, bias=False)
        with torch.no_grad():
            first.weight.copy_(torch.eye(16))
            second.weight.zero_()
            for output_index in range(4):
                second.weight[output_index, output_index::4] = 0.25
        self.mlp = nn.Sequential(first, nn.Identity(), second)

    def forward(self, hidden_states: Any) -> Any:
        self.last_source_count = int(hidden_states.shape[0])
        return self.mlp(self.ln_q(hidden_states).reshape(-1, self.hidden_size))


class Qwen2_5_VisionTransformerPretrainedModel(nn.Module):
    def __init__(self, *, block_count: int = 2) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_VisionBlock() for _ in range(block_count)])
        self.merger = _Merger()
        self.patch_embed = nn.Identity()
        self.fullatt_block_indexes = list(range(block_count))
        self.spatial_merge_size = 2
        self.spatial_merge_unit = 4
        self.dtype = torch.float32

    def rot_pos_emb(self, grid_thw: Any) -> Any:
        count = int(grid_thw.prod(-1).sum().item())
        return torch.zeros(count, 1, device=grid_thw.device)

    def get_window_index(self, grid_thw: Any) -> Any:
        feature_counts = (grid_thw.prod(-1) // self.spatial_merge_unit).tolist()
        total = int(sum(feature_counts))
        return torch.arange(total, device=grid_thw.device), [0, total * 4]

    def forward(
        self,
        hidden_states: Any,
        grid_thw: Any,
        **kwargs: Any,
    ) -> Any:
        del grid_thw, kwargs
        return self.merger(self.patch_embed(hidden_states))


class _TextAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.head_dim = 2
        self.num_key_value_groups = 2
        self.rope_scaling = {"mrope_section": [1, 0, 0]}
        self.q_proj = nn.Linear(4, 4, bias=False)
        self.k_proj = nn.Linear(4, 2, bias=False)
        with torch.no_grad():
            self.q_proj.weight.copy_(torch.eye(4))
            self.k_proj.weight.copy_(torch.eye(2, 4))

    def forward(self, hidden_states: Any, **kwargs: Any) -> tuple[Any]:
        del kwargs
        return (torch.zeros_like(hidden_states) + 0.005,)


class _TransitionMLP(nn.Module):
    def forward(self, hidden_states: Any) -> Any:
        return torch.zeros_like(hidden_states) + 0.005


class _Decoder(nn.Module):
    def __init__(
        self, layer_index: int, attention_type: str = "full_attention"
    ) -> None:
        super().__init__()
        self.layer_index = layer_index
        self.input_layernorm = nn.Identity()
        self.self_attn = _TextAttention()
        self.post_attention_layernorm = nn.Identity()
        self.mlp = _TransitionMLP()
        self.attention_type = attention_type

    def forward(
        self,
        hidden_states: Any,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        output_attentions: Any = None,
        use_cache: Any = None,
        cache_position: Any = None,
        position_embeddings: Any = None,
        **kwargs: Any,
    ) -> Any:
        residual = hidden_states
        attention_output = self.self_attn(self.input_layernorm(hidden_states))[0]
        hidden_states = residual + attention_output
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        del (
            attention_mask,
            position_ids,
            output_attentions,
            use_cache,
            cache_position,
            position_embeddings,
            kwargs,
        )
        return (hidden_states,)


class _Rotary(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("inv_freq", torch.tensor([1.0]))

    def forward(self, hidden_states: Any, position_ids: Any) -> Any:
        shape = (*position_ids.shape, 2)
        return hidden_states.new_ones(shape), hidden_states.new_zeros(shape)


class _TextConfig:
    use_cache = False
    output_attentions = False
    output_hidden_states = False
    use_return_dict = True
    rope_scaling = {"mrope_section": [1, 0, 0]}


class _BaseOutput:
    def __init__(
        self,
        *,
        last_hidden_state: Any,
        past_key_values: Any = None,
        hidden_states: Any = None,
        attentions: Any = None,
    ) -> None:
        self.last_hidden_state = last_hidden_state
        self.past_key_values = past_key_values
        self.hidden_states = hidden_states
        self.attentions = attentions

    def __getitem__(self, index: int) -> Any:
        assert index == 0
        return self.last_hidden_state


class Qwen2_5_VLTextModel(nn.Module):
    def __init__(self, *, layer_count: int = 8, sliding: bool = False) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                _Decoder(
                    index,
                    "sliding_attention" if sliding and index % 2 else "full_attention",
                )
                for index in range(layer_count)
            ]
        )
        self.norm = nn.Identity()
        self.rotary_emb = _Rotary()
        self.embed_tokens = nn.Embedding(32, 4)
        self.has_sliding_layers = sliding
        self.config = _TextConfig()
        self.last_forward: dict[str, Any] | None = None

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        output_attentions: Any = None,
        output_hidden_states: Any = None,
        return_dict: Any = None,
        cache_position: Any = None,
        **kwargs: Any,
    ) -> Any:
        del input_ids, kwargs
        self.last_forward = {
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "cache_position": cache_position,
        }
        return _BaseOutput(
            last_hidden_state=inputs_embeds,
            past_key_values=past_key_values,
        )


class _CoreConfig:
    image_token_id = 9
    video_token_id = 8
    output_attentions = False
    output_hidden_states = False


class _CoreOutput(_BaseOutput):
    def __init__(self, *, rope_deltas: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.rope_deltas = rope_deltas


class Qwen2_5_VLModel(nn.Module):
    def __init__(self, *, layer_count: int = 8, sliding: bool = False) -> None:
        super().__init__()
        self.visual = Qwen2_5_VisionTransformerPretrainedModel()
        self.language_model = Qwen2_5_VLTextModel(
            layer_count=layer_count,
            sliding=sliding,
        )
        self.config = _CoreConfig()
        self.rope_deltas = None
        self.test_rope_delta = 0

    def get_input_embeddings(self) -> Any:
        return self.language_model.embed_tokens

    def get_image_features(
        self,
        pixel_values: Any,
        image_grid_thw: Any = None,
    ) -> Any:
        embeds = self.visual(
            pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw
        )
        sizes = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(embeds, sizes)

    def get_video_features(
        self,
        pixel_values_videos: Any,
        video_grid_thw: Any = None,
    ) -> Any:
        embeds = self.visual(
            pixel_values_videos.type(self.visual.dtype), grid_thw=video_grid_thw
        )
        sizes = (video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2).tolist()
        return torch.split(embeds, sizes)

    def get_placeholder_mask(
        self,
        input_ids: Any,
        inputs_embeds: Any,
        image_features: Any = None,
        video_features: Any = None,
    ) -> Any:
        image = (
            (input_ids == self.config.image_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        video = (
            (input_ids == self.config.video_token_id)
            .unsqueeze(-1)
            .expand_as(inputs_embeds)
        )
        if (
            image_features is not None
            and inputs_embeds[image].numel() != image_features.numel()
        ):
            raise ValueError("image cardinality mismatch")
        if (
            video_features is not None
            and inputs_embeds[video].numel() != video_features.numel()
        ):
            raise ValueError("video cardinality mismatch")
        return image, video

    def get_rope_index(
        self,
        input_ids: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        second_per_grid_ts: Any = None,
        attention_mask: Any = None,
    ) -> Any:
        del image_grid_thw, video_grid_thw, second_per_grid_ts, attention_mask
        positions = (
            torch.arange(input_ids.shape[1], device=input_ids.device)
            + self.test_rope_delta
        ).clamp_min(0)
        positions = positions.view(1, 1, -1).expand(3, input_ids.shape[0], -1)
        rope = torch.full(
            (input_ids.shape[0], 1),
            self.test_rope_delta,
            dtype=torch.long,
            device=input_ids.device,
        )
        return positions, rope

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        use_cache: Any = None,
        output_attentions: Any = None,
        output_hidden_states: Any = None,
        return_dict: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        rope_deltas: Any = None,
        cache_position: Any = None,
        second_per_grid_ts: Any = None,
        **kwargs: Any,
    ) -> Any:
        del return_dict, kwargs
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            image = torch.cat(
                self.get_image_features(pixel_values, image_grid_thw), dim=0
            ).to(inputs_embeds)
            image_mask, _ = self.get_placeholder_mask(
                input_ids, inputs_embeds, image_features=image
            )
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image)
        if pixel_values_videos is not None:
            video = torch.cat(
                self.get_video_features(pixel_values_videos, video_grid_thw), dim=0
            ).to(inputs_embeds)
            _, video_mask = self.get_placeholder_mask(
                input_ids, inputs_embeds, video_features=video
            )
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video)
        if position_ids is None:
            position_ids, rope_deltas = self.get_rope_index(
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts=second_per_grid_ts,
                attention_mask=attention_mask,
            )
            self.rope_deltas = rope_deltas
        output = self.language_model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )
        return _CoreOutput(
            last_hidden_state=output.last_hidden_state,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            rope_deltas=self.rope_deltas,
        )


class Qwen2_5_VLForConditionalGeneration(nn.Module):
    def __init__(self, *, layer_count: int = 8, sliding: bool = False) -> None:
        super().__init__()
        self.model = Qwen2_5_VLModel(layer_count=layer_count, sliding=sliding)
        self.lm_head = nn.Linear(4, 32, bias=False)

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        position_ids: Any = None,
        past_key_values: Any = None,
        inputs_embeds: Any = None,
        labels: Any = None,
        use_cache: Any = None,
        output_attentions: Any = None,
        output_hidden_states: Any = None,
        pixel_values: Any = None,
        pixel_values_videos: Any = None,
        image_grid_thw: Any = None,
        video_grid_thw: Any = None,
        rope_deltas: Any = None,
        cache_position: Any = None,
        second_per_grid_ts: Any = None,
        **kwargs: Any,
    ) -> Any:
        del labels, kwargs
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
        )
        output.logits = self.lm_head(output.last_hidden_state)
        output.loss = None
        return output

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


def _activation(
    model: Any,
    binding_name: str,
    *,
    config: dict[str, Any] | None = None,
    schedule: dict[str, Any] | None = None,
    phase: ExecutionPhase = ExecutionPhase.FORWARD,
) -> Any:
    contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS[binding_name]
    binding = SimpleNamespace(
        name=binding_name,
        backbone="qwen25_vl",
        variant=contract.variant,
        target_modalities=frozenset({Modality.IMAGE}),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}),
        ordered_targets=lambda: (),
    )
    stage = SimpleNamespace(
        name=f"provider-{contract.algorithm}",
        algorithm=contract.algorithm,
        variant=contract.variant,
        target_modalities={Modality.IMAGE},
        operation={},
        criterion={},
        schedule=dict(schedule or {}),
        config=dict(config or {}),
    )
    return create_qwen25_vl_image_compression_executor(
        model=model,
        runtime=SimpleNamespace(
            metadata={},
            model_path_or_id="/fixtures/unit-test-model",
        ),
        facade=None,
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(config={}),
        execution_phase=phase,
    )


def _image_call(model: Any) -> Any:
    return model.forward(
        input_ids=torch.tensor([[1, 9, 9, 9, 9, 2]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        pixel_values=torch.arange(64, dtype=torch.float32).reshape(16, 4),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        use_cache=False,
    )


def _two_image_call(model: Any) -> Any:
    return model.forward(
        input_ids=torch.tensor([[1, 9, 9, 9, 9, 2, 9, 9, 9, 9, 3]]),
        attention_mask=torch.ones(1, 11, dtype=torch.long),
        pixel_values=torch.arange(128, dtype=torch.float32).reshape(32, 4),
        image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
        use_cache=False,
    )


def _training_image_call(model: Any) -> tuple[Any, Any]:
    pixels = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    pixels.requires_grad_(True)
    output = model.forward(
        input_ids=torch.tensor([[1, 9, 9, 9, 9, 2]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, -100, 2]]),
        pixel_values=pixels,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        use_cache=False,
    )
    return output, pixels


def test_install_is_idempotent_and_restores_all_instance_patches() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    original_signature = inspect.signature(model.forward)
    original = {
        "model": model.forward.__func__,
        "core": model.model.forward.__func__,
        "visual": model.model.visual.forward.__func__,
        "language": model.model.language_model.forward.__func__,
        "generate": model.generate.__func__,
    }

    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)

    assert provider.install_qwen25_vl_transformers_4573_bridge(model) is bridge
    assert bridge.installed
    assert inspect.signature(model.forward) == original_signature
    assert model._mllm_comp_qwen25_vl_image_compression_bridge is bridge
    assert bridge.runtime_metadata()["real_checkpoint_verified"] is False

    provider.uninstall_qwen25_vl_transformers_4573_bridge(model)

    assert model.forward.__func__ is original["model"]
    assert model.model.forward.__func__ is original["core"]
    assert model.model.visual.forward.__func__ is original["visual"]
    assert model.model.language_model.forward.__func__ is original["language"]
    assert model.generate.__func__ is original["generate"]
    assert not hasattr(model, "_mllm_comp_qwen25_vl_provider_bridge")
    assert not hasattr(model, "_mllm_comp_qwen25_vl_image_compression_bridge")


def test_multiple_activations_remove_independently_and_force_uninstall() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    first = _activation(model, "tome.image.qwen25_vl", config={"r": 1})
    second = _activation(model, "tome.image.qwen25_vl", config={"r": 2})
    visionzip = _activation(model, "visionzip.image.qwen25_vl", config={"budget": 2})
    first.install()
    second.install()
    visionzip.install()

    assert bridge.runtime_metadata()["registration_count"] == 3
    second.uninstall()
    assert bridge.runtime_metadata()["registration_count"] == 2
    with pytest.raises(BackendIntegrationError, match="activations remain"):
        bridge.uninstall()

    bridge.uninstall(force=True)
    first.uninstall()
    visionzip.uninstall()
    assert not bridge.installed


def test_generate_cache_video_and_implicit_cache_fail_closed() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, "visionzip.image.qwen25_vl", config={"budget": 2})
    activation.install()

    with pytest.raises(BackendIntegrationError, match="explicit argument"):
        model.forward(input_ids=torch.tensor([[1, 2]]))
    with pytest.raises(BackendIntegrationError, match="explicit argument"):
        model.forward(input_ids=torch.tensor([[1, 2]]), use_cache=True)
    with pytest.raises(BackendIntegrationError, match="cache state"):
        model.forward(
            input_ids=torch.tensor([[1, 2]]),
            use_cache=False,
            past_key_values=object(),
        )
    with pytest.raises(BackendIntegrationError, match="image-only"):
        model.forward(
            input_ids=torch.tensor([[1, 8, 2]]),
            use_cache=False,
            pixel_values_videos=torch.zeros(4, 4),
        )
    with pytest.raises(BackendIntegrationError, match="does not support generate"):
        model.generate(torch.tensor([[1, 2]]), use_cache=False)

    dense_text = model.forward(input_ids=torch.tensor([[1, 2]]), use_cache=False)
    assert dense_text.last_hidden_state.shape == (1, 2, 4)
    assert bridge.runtime_metadata()["supports_labels"] is True
    activation.uninstall()
    bridge.uninstall()


def test_version_class_signature_and_final_full_attention_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    monkeypatch.setattr(provider, "_installed_transformers_version", lambda: "4.58.0")
    with pytest.raises(BackendIntegrationError, match="source-pinned"):
        provider.install_qwen25_vl_transformers_4573_bridge(model)
    assert not hasattr(model, "_mllm_comp_qwen25_vl_provider_bridge")

    monkeypatch.setattr(
        provider,
        "_installed_transformers_version",
        lambda: provider.SUPPORTED_TRANSFORMERS_VERSION,
    )
    wrong = SimpleNamespace(
        model=model.model, forward=model.forward, generate=model.generate
    )
    with pytest.raises(BackendIntegrationError, match="requires model class"):
        provider.install_qwen25_vl_transformers_4573_bridge(wrong)

    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    model.model.visual.fullatt_block_indexes = [0]
    activation = _activation(model, "visionzip.image.qwen25_vl", config={"budget": 2})
    with pytest.raises(BackendIntegrationError, match="final visual block"):
        activation.install()
    divprune = _activation(
        model,
        "divprune.image.qwen25_vl",
        config={"target_count": 2},
    )
    divprune.install()
    divprune.uninstall()
    bridge.uninstall()


def test_torch_forward_executes_tome_restore_visionzip_and_sparse_mrope() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    tome = _activation(model, "tome.image.qwen25_vl", config={"r": 1})
    visionzip = _activation(model, "visionzip.image.qwen25_vl", config={"budget": 2})
    tome.install()
    visionzip.install()

    output = _image_call(model)

    assert output.last_hidden_state.shape == (1, 4, 4)
    assert model.model.visual.merger.last_source_count == 16
    forwarded = model.model.language_model.last_forward
    assert forwarded is not None
    positions = forwarded["position_ids"]
    assert positions.shape == (3, 1, 4)
    assert positions[0, 0, 0].item() == 0
    assert positions[0, 0, -1].item() == 5
    assert set(positions[0, 0, 1:-1].tolist()).issubset({1, 2, 3, 4})
    assert output.rope_deltas.item() == 2

    metadata = bridge.runtime_metadata()
    assert metadata["invocation_counts"] == {
        "encoder_dense_boundary": 3,
        "encoder_progressive_prune": 0,
        "post_visual_merger": 1,
        "decoder_layer_loop": 0,
    }
    assert metadata["latest_layout_deltas"]
    json.dumps(metadata["latest_layout_deltas"])
    latest = visionzip.runtime_metadata()["latest_layout_delta"]
    assert latest == {
        "source_count": 3,
        "result_count": 2,
        "media_unit": "image_feature",
        "operation": "merge",
        "site": "post_visual_merger",
        "invocation_id": "visionzip.image.qwen25_vl:1",
    }

    visionzip.uninstall()
    tome.uninstall()
    bridge.uninstall()


def test_torch_forward_executes_zoo_prune_with_native_merger_projector() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    model.eval()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(
        model,
        "zoo_prune.image.qwen25_vl",
        config={
            "target_count": 2,
            "num_directions": 2,
            "projector_proxy": "full",
            "direction_batch_size": 1,
        },
    )
    captured: dict[str, Any] = {}
    original = activation.post_visual_merger

    def capture_projector(*args: Any, **kwargs: Any) -> Any:
        captured["inputs"] = kwargs["projector_inputs"].detach().clone()
        captured["layers"] = tuple(kwargs["projector_layers"])
        return original(*args, **kwargs)

    activation.post_visual_merger = capture_projector
    activation.install()
    output = _image_call(model)

    assert output.last_hidden_state.shape == (1, 4, 4)
    assert captured["inputs"].shape == (4, 16)
    assert len(captured["layers"]) == 3
    record = activation.runtime_metadata()["layout_delta_records"][-1]
    assert record["source_layout"]["projector_input"] == (
        "native_merger_mlp_input"
    )
    assert record["source_count"] == 4
    assert record["result_count"] == 2
    assert bridge.runtime_metadata()["invocation_counts"]["post_visual_merger"] == 1

    activation.uninstall()
    bridge.uninstall()


def test_torch_forward_executes_mmtok_with_native_multisource_payload() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(
        model,
        "mmtok.image.qwen25_vl",
        config={"target_count": 2},
    )
    captured: dict[str, Any] = {}
    original_post_visual_merger = activation.post_visual_merger

    def capture_reference(*args: Any, **kwargs: Any) -> Any:
        captured["vision_reference"] = kwargs["vision_reference"].detach().clone()
        return original_post_visual_merger(*args, **kwargs)

    activation.post_visual_merger = capture_reference
    activation.install()
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
    pixels = torch.cat(
        (first_group, second_group, torch.eye(4), -torch.eye(4)), dim=0
    )
    output = model.forward(
        input_ids=torch.tensor([[1, 9, 9, 9, 9, 2]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        pixel_values=pixels,
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        use_cache=False,
    )

    assert output.last_hidden_state.shape == (1, 4, 4)
    forwarded = model.model.language_model.last_forward
    assert forwarded is not None
    assert forwarded["position_ids"].shape == (3, 1, 4)
    latest = activation.runtime_metadata()["latest_layout_delta"]
    assert latest["source_count"] == 4
    assert latest["result_count"] == 2
    assert latest["operation"] == "drop"
    record = activation.runtime_metadata()["layout_delta_records"][-1]
    assert record["source_layout"]["vision_source"] == (
        "pre_merger_2x2_mean_features"
    )
    assert len(latest["selected_indices"]) == 2
    assert bridge.runtime_metadata()["invocation_counts"]["post_visual_merger"] == 1
    reference = captured["vision_reference"]
    assert reference.shape == (4, 16)
    torch.testing.assert_close(
        reference[0].reshape(4, 4).mean(dim=0),
        reference[1].reshape(4, 4).mean(dim=0),
    )
    assert not torch.equal(reference[0], reference[1])

    activation.uninstall()
    bridge.uninstall()


def test_divprune_records_each_image_in_a_multi_image_prompt() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(
        model,
        "divprune.image.qwen25_vl",
        config={"target_count": 2},
    )
    activation.install()

    output = _two_image_call(model)

    assert output.last_hidden_state.shape == (1, 7, 4)
    records = activation.runtime_metadata()["layout_delta_records"]
    assert [record["image_index"] for record in records] == [0, 1]
    assert all(record["source_count"] == 4 for record in records)
    assert all(record["result_count"] == 2 for record in records)
    assert [record["source_layout"]["image_index"] for record in records] == [0, 1]
    latest = bridge.runtime_metadata()["latest_layout_deltas"]
    assert latest[0]["image_index"] == 1

    activation.uninstall()
    bridge.uninstall()


@pytest.mark.parametrize(
    "binding_name",
    ["dart.image.qwen25_vl", "d2pruner.image.qwen25_vl"],
)
def test_single_image_decoder_methods_reject_multi_image_before_selection(
    binding_name: str,
    tmp_path: Any,
) -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    config: dict[str, Any] = {"prune_after_layer": 2}
    if binding_name.startswith("d2pruner"):
        prior_path = tmp_path / "d2pruner-prior.pt"
        torch.save(
            {
                "bias": torch.ones(4),
                "sample_count": 4,
                "model_path": "/fixtures/unit-test-model",
                "scorer_layer": 1,
            },
            prior_path,
        )
        config["bias_prior_path"] = str(prior_path)
    activation = _activation(model, binding_name, config=config)
    activation.install()

    with pytest.raises(BackendIntegrationError, match="single-image only"):
        _two_image_call(model)

    assert bridge.runtime_metadata()["invocation_counts"]["decoder_layer_loop"] == 0
    activation.uninstall()
    bridge.uninstall()


def test_tome_permanently_compacts_post_merger_visual_prompt() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    tome = _activation(
        model,
        "tome.image.qwen25_vl",
        config={"r": 1, "output_r": 1},
    )
    tome.install()

    output = _image_call(model)

    # Four native 2x2-merger features become three actual LLM image tokens.
    assert output.last_hidden_state.shape == (1, 5, 4)
    assert model.model.visual.merger.last_source_count == 16
    latest = tome.runtime_metadata()["latest_layout_delta"]
    assert latest["source_count"] == 4
    assert latest["result_count"] == 3
    assert latest["media_unit"] == "image_feature"
    assert latest["site"] == "permanent_visual_output"
    record = tome.runtime_metadata()["layout_delta_records"][-1]
    assert record["ordering_policy"] == "preserve_source"
    representatives = [
        record["old_to_new"].index(result_index)
        for result_index in range(record["result_count"])
    ]
    assert representatives == sorted(representatives)

    tome.uninstall()
    bridge.uninstall()


def test_tome_and_visionzip_training_compact_labels_and_backpropagate() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    tome = _activation(
        model,
        "tome.image.qwen25_vl",
        config={"r": 1, "output_r": 1},
    )
    visionzip = _activation(
        model,
        "visionzip.image.qwen25_vl",
        config={"budget": 2},
    )
    tome.install()
    visionzip.install()

    output, pixels = _training_image_call(model)

    # Six dense prompt positions become two text + two visual positions.
    assert output.logits.shape[:2] == (1, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert pixels.grad is not None
    assert model.lm_head.weight.grad is not None

    visionzip.uninstall()
    tome.uninstall()
    bridge.uninstall()


def test_torch_forward_executes_pyramid_inside_decoder_and_rebuilds_masks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask_lengths: list[tuple[str, int]] = []

    def full_mask(**kwargs: Any) -> Any:
        length = int(kwargs["input_embeds"].shape[1])
        mask_lengths.append(("full", length))
        return torch.zeros(1, 1, length, length)

    def sliding_mask(**kwargs: Any) -> Any:
        length = int(kwargs["input_embeds"].shape[1])
        mask_lengths.append(("sliding", length))
        return torch.zeros(1, 1, length, length)

    monkeypatch.setattr(
        provider,
        "_transformers_decoder_helpers",
        lambda: (_BaseOutput, full_mask, sliding_mask),
    )
    model = Qwen2_5_VLForConditionalGeneration(sliding=True)
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    pyramid = _activation(
        model,
        "pyramid_drop.image.qwen25_vl",
        config={"keep_ratio": 0.5, "minimum_keep_tokens": 1},
        schedule={"layer_indices": [1, 3, 5], "num_stages": 4},
    )
    pyramid.install()

    output, pixels = _training_image_call(model)

    assert output.last_hidden_state.shape == (1, 3, 4)
    assert output.logits.shape[:2] == (1, 3)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert pixels.grad is not None
    assert output.rope_deltas.item() == 0
    assert bridge.runtime_metadata()["invocation_counts"]["decoder_layer_loop"] == 3
    assert {length for _, length in mask_lengths} == {6, 4, 3}
    assert {kind for kind, _ in mask_lengths} == {"full", "sliding"}
    latest = pyramid.runtime_metadata()["latest_layout_delta"]
    assert latest["source_count"] == 3
    assert latest["result_count"] == 3
    assert latest["operation"] == "identity"
    bridge_latest = bridge.runtime_metadata()["latest_layout_deltas"]
    assert bridge_latest[0]["boundary_layer"] == 6
    json.dumps(bridge.runtime_metadata())

    pyramid.uninstall()
    bridge.uninstall()


def test_generate_phase_executor_registers_only_at_decoder_site() -> None:
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    generated = _activation(
        model,
        "pyramid_drop.image.qwen25_vl",
        config={"keep_ratio": 0.5},
        schedule={"layer_indices": [1, 3, 5]},
        phase=ExecutionPhase.GENERATE,
    )

    generated.install()
    assert generated.installed
    metadata = bridge.runtime_metadata()
    assert metadata["registration_count"] == 1
    assert metadata["active_generate_supported"] is True
    assert metadata["supported_phases"] == ["forward", "generate"]
    assert metadata["use_cache_by_phase"] == {
        "forward": False,
        "generate": True,
    }
    generated.uninstall()
    bridge.uninstall()


def test_nuwa_semantic_reference_is_mean_of_requested_visual_states(monkeypatch):
    # Fixed linear visual blocks provide an independent closed-form reference;
    # native integration tests cover Nuwa's decoder/cache execution.
    model = Qwen2_5_VLForConditionalGeneration().eval()
    monkeypatch.setattr(provider, "_transformers_decoder_helpers",
        lambda: (_BaseOutput, lambda **kwargs: None, lambda **kwargs: None))
    activation = _activation(model, "nuwa.image.qwen25_vl", config={
        "stage1_total_tokens": 2, "stage2_after_layer": 2, "stage2_retain_ratio": 0.5,
        "region_size": 1, "top_n_per_region": 1,
        "semantic_hidden_start": 0, "semantic_hidden_end": 3,
    })
    captured = []
    original = activation.post_visual_merger

    def record(*args, **kwargs):
        captured.append(kwargs["vision_reference"].detach().clone())
        return original(*args, **kwargs)

    monkeypatch.setattr(activation, "post_visual_merger", record)
    activation.install()
    try:
        with torch.no_grad():
            _image_call(model)
        pixels = torch.arange(64, dtype=torch.float32).reshape(16, 4)
        expected = (pixels + pixels * 2.02 + pixels * (2.02**2)) / 3.0
        assert len(captured) == 1
        torch.testing.assert_close(captured[0], expected.reshape(4, 16))
    finally:
        activation.uninstall()
