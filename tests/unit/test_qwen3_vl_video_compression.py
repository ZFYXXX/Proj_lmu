"""Pinned-provider tests for Qwen3-VL video and DeepStack legalization."""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from tests.fixtures.qwen_video import source_call as _source_call

try:
    import torch
except ImportError:  # dependency-light control-plane environment
    torch = None  # type: ignore[assignment]

from mllm_comp.backbones.qwen3_vl import provider_bridge as image_provider
from mllm_comp.backbones.qwen3_vl.video_compression import (
    QWEN3_VL_VIDEO_SPLICE_SITE,
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
    _call_bound_method,
    install_qwen3_vl_video_bridge,
    legalize_qwen3_vl_video_prompt,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from tests.unit.test_qwen3_vl_image_compression_provider_bridge import (
    Qwen3VLForConditionalGeneration as StructuralQwen3VLModel,
)


requires_torch = pytest.mark.skipif(
    torch is None,
    reason="PyTorch is unavailable in this control-plane environment",
)


def test_bound_method_call_deduplicates_decorated_explicit_kwargs() -> None:
    def target(
        input_ids: Any = None,
        use_cache: Any = None,
        **kwargs: Any,
    ) -> tuple[Any, Any, dict[str, Any]]:
        return input_ids, use_cache, kwargs

    signature = inspect.signature(target)
    bound = signature.bind_partial(input_ids="prompt", use_cache=True)
    bound.arguments["kwargs"] = {"use_cache": True, "marker": "kept"}

    assert _call_bound_method(target, bound) == (
        "prompt",
        True,
        {"marker": "kept"},
    )


def _pin_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_provider,
        "_installed_transformers_version",
        lambda: image_provider.SUPPORTED_TRANSFORMERS_VERSION,
    )


def _compress_to_three(
    request: Qwen3VLPackedVideoRequest,
) -> Qwen3VLVideoCompressionResponse:
    groups = ((0, 1, 2), (3, 4), (5, 6, 7))
    representatives = (0, 4, 7)
    compressed = torch.stack(
        [request.payload.features[list(group)].mean(dim=0) for group in groups]
    )
    result = VideoCompressionResult.from_source_groups(
        request.payload,
        compressed,
        source_groups=(groups,),
        representative_indices=(representatives,),
        exact_budgets=(3,),
        metadata={"algorithm": "fixture.video"},
    )
    deepstack = tuple(
        torch.stack([branch[list(group)].mean(dim=0) for group in groups])
        for branch in request.deepstack_features
    )
    return Qwen3VLVideoCompressionResponse.from_result(
        request,
        result,
        deepstack_features=deepstack,
    )


def _tensor_request(
    features: Any,
    *,
    positions: tuple[int, ...] = tuple(range(1, 9)),
    prompt_input_ids: Any = None,
) -> Qwen3VLPackedVideoRequest:
    grid = Qwen3VLVideoGrid.from_native(
        (2, 4, 4),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    if prompt_input_ids is None:
        prompt_input_ids = torch.tensor([[1, *([8] * 8), 2]])
    payload = grid.payload(features, media_id="sample-7:video-0")
    return Qwen3VLPackedVideoRequest(
        invocation_id="video-splice-1",
        media_id="sample-7:video-0",
        payload=payload,
        deepstack_features=tuple(features + index + 1 for index in range(3)),
        grid=grid,
        source_placeholder_positions=positions,
        prompt_input_ids=prompt_input_ids,
        source_attention_mask=torch.ones_like(prompt_input_ids),
        source_video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )


def _build_torch_model() -> Any:
    if torch is None:
        raise RuntimeError("tensor helper requires PyTorch")
    nn = torch.nn

    class _Config:
        image_token_id = 9
        video_token_id = 8
        rope_scaling = {"mrope_section": [1, 0, 0]}

    class Qwen3VLVisionModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.ModuleList([nn.Identity() for _ in range(3)])
            self.merger = nn.Identity()
            self.deepstack_visual_indexes = [0, 1, 2]
            self.deepstack_merger_list = nn.ModuleList(
                [nn.Identity() for _ in range(3)]
            )
            self.patch_embed = nn.Identity()
            self.fast_pos_embed_interpolate = lambda grid: torch.zeros(
                int(grid.prod(-1).sum().item()), 4
            )
            self.rot_pos_emb = lambda grid: torch.zeros(
                int(grid.prod(-1).sum().item()), 1
            )
            self.spatial_merge_size = 2
            self.spatial_merge_unit = 4

        @property
        def dtype(self) -> Any:
            return torch.float32

        def forward(
            self,
            hidden_states: Any,
            grid_thw: Any,
            **kwargs: Any,
        ) -> Any:
            del grid_thw, kwargs
            merged = hidden_states.reshape(-1, 4, hidden_states.shape[-1]).mean(1)
            deepstack = [merged + float(index + 1) for index in range(3)]
            return merged, deepstack

    class Qwen3VLTextModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])
            self.norm = nn.Identity()
            self.rotary_emb = nn.Identity()
            self.embed_tokens = nn.Embedding(32, 4)
            self.config = _Config()
            self.last_forward: dict[str, Any] | None = None

        def _deepstack_process(
            self,
            hidden_states: Any,
            visual_pos_masks: Any,
            visual_embeds: Any,
        ) -> Any:
            output = hidden_states.clone()
            output[visual_pos_masks] = output[visual_pos_masks] + visual_embeds
            return output

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
            del input_ids, kwargs
            self.last_forward = {
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "inputs_embeds": inputs_embeds,
                "use_cache": use_cache,
                "cache_position": cache_position,
                "visual_pos_masks": visual_pos_masks,
                "deepstack_visual_embeds": deepstack_visual_embeds,
            }
            hidden = inputs_embeds
            for branch in deepstack_visual_embeds or ():
                hidden = self._deepstack_process(hidden, visual_pos_masks, branch)
            return SimpleNamespace(
                last_hidden_state=hidden,
                past_key_values=past_key_values,
            )

    class Qwen3VLModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.visual = Qwen3VLVisionModel()
            self.language_model = Qwen3VLTextModel()
            self.config = _Config()
            self.rope_deltas = None

        def get_input_embeddings(self) -> Any:
            return self.language_model.embed_tokens

        def get_image_features(
            self,
            pixel_values: Any,
            image_grid_thw: Any = None,
        ) -> Any:
            embeds, deepstack = self.visual(
                pixel_values.type(self.visual.dtype),
                grid_thw=image_grid_thw,
            )
            sizes = (
                image_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
            ).tolist()
            return torch.split(embeds, sizes), deepstack

        def get_video_features(
            self,
            pixel_values_videos: Any,
            video_grid_thw: Any = None,
        ) -> Any:
            embeds, deepstack = self.visual(
                pixel_values_videos.type(self.visual.dtype),
                grid_thw=video_grid_thw,
            )
            sizes = (
                video_grid_thw.prod(-1) // self.visual.spatial_merge_size**2
            ).tolist()
            return torch.split(embeds, sizes), deepstack

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
            if image_features is not None and (
                inputs_embeds[image].numel() != image_features.numel()
            ):
                raise ValueError("image cardinality mismatch")
            if video_features is not None and (
                inputs_embeds[video].numel() != video_features.numel()
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
                .view(1, 1, -1)
                .expand(3, input_ids.shape[0], -1)
            )
            rope = torch.zeros(
                (input_ids.shape[0], 1),
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
            pixel_values: Any = None,
            pixel_values_videos: Any = None,
            image_grid_thw: Any = None,
            video_grid_thw: Any = None,
            cache_position: Any = None,
            second_per_grid_ts: Any = None,
            use_cache: Any = None,
            rope_deltas: Any = None,
            **kwargs: Any,
        ) -> Any:
            del kwargs
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)
            visual_pos_masks = None
            deepstack_visual_embeds = None
            if pixel_values is not None:
                image_chunks, image_deepstack = self.get_image_features(
                    pixel_values,
                    image_grid_thw,
                )
                image = torch.cat(image_chunks, dim=0).to(inputs_embeds)
                image_mask, _ = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds,
                    image_features=image,
                )
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image)
                visual_pos_masks = image_mask[..., 0]
                deepstack_visual_embeds = image_deepstack
            if pixel_values_videos is not None:
                video_chunks, video_deepstack = self.get_video_features(
                    pixel_values_videos,
                    video_grid_thw,
                )
                video = torch.cat(video_chunks, dim=0).to(inputs_embeds)
                _, video_mask = self.get_placeholder_mask(
                    input_ids,
                    inputs_embeds,
                    video_features=video,
                )
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video)
                visual_pos_masks = video_mask[..., 0]
                deepstack_visual_embeds = video_deepstack
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
                cache_position=cache_position,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
            return SimpleNamespace(
                last_hidden_state=output.last_hidden_state,
                past_key_values=output.past_key_values,
                rope_deltas=self.rope_deltas,
            )

    class Qwen3VLForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Qwen3VLModel()
            self.lm_head = nn.Linear(4, 32, bias=False)

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
            second_per_grid_ts: Any = None,
            use_cache: Any = None,
            rope_deltas: Any = None,
            return_dict: Any = None,
            **kwargs: Any,
        ) -> Any:
            del labels, return_dict, kwargs
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                cache_position=cache_position,
                second_per_grid_ts=second_per_grid_ts,
                use_cache=use_cache,
                rope_deltas=rope_deltas,
            )
            output.logits = self.lm_head(output.last_hidden_state)
            output.loss = None
            return output

        def generate(self, *args: Any, **kwargs: Any) -> str:
            del args, kwargs
            return "dense-generate"

    return Qwen3VLForConditionalGeneration()


def test_install_is_idempotent_signature_preserving_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = StructuralQwen3VLModel()
    original_signature = inspect.signature(model.forward)
    original = {
        "model": model.forward.__func__,
        "core": model.model.forward.__func__,
        "video": model.model.get_video_features.__func__,
        "placeholder": model.model.get_placeholder_mask.__func__,
        "language": model.model.language_model.forward.__func__,
        "generate": model.generate.__func__,
    }

    bridge = install_qwen3_vl_video_bridge(model)

    assert install_qwen3_vl_video_bridge(model) is bridge
    assert bridge.available_sites() == (QWEN3_VL_VIDEO_SPLICE_SITE,)
    assert not bridge.patched
    handle = bridge.register_callback(
        QWEN3_VL_VIDEO_SPLICE_SITE,
        Qwen3VLVideoCompressionResponse.identity,
    )
    assert bridge.patched
    assert inspect.signature(model.forward) == original_signature
    assert bridge.runtime_metadata()["capabilities"]["deepstack_shared_layout"] is True
    assert bridge.runtime_metadata()["support_activation_claimed"] is False

    handle.remove()
    bridge.detach()

    assert model.forward.__func__ is original["model"]
    assert model.model.forward.__func__ is original["core"]
    assert model.model.get_video_features.__func__ is original["video"]
    assert model.model.get_placeholder_mask.__func__ is original["placeholder"]
    assert model.model.language_model.forward.__func__ is original["language"]
    assert model.generate.__func__ is original["generate"]
    assert not hasattr(model, "mllm_video_compression_bridge")


def test_image_and_video_provider_bridges_reject_both_install_orders(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    video_first = StructuralQwen3VLModel()
    video_bridge = install_qwen3_vl_video_bridge(video_first)
    with pytest.raises(BackendIntegrationError, match="image and video"):
        image_provider.install_qwen3_vl_transformers_4573_bridge(video_first)
    video_bridge.detach()

    image_first = StructuralQwen3VLModel()
    image_bridge = image_provider.install_qwen3_vl_transformers_4573_bridge(image_first)
    with pytest.raises(BackendIntegrationError, match="image and video"):
        install_qwen3_vl_video_bridge(image_first)
    image_bridge.uninstall()


@requires_torch
def test_prompt_legalizer_gathers_mrope_visual_masks_labels_cache_and_deepstack() -> (
    None
):
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    request = _tensor_request(features)
    response = _compress_to_three(request)
    inputs_embeds = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4)
    rotary = torch.tensor(
        [
            [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
            [[0, 10, 11, 12, 13, 14, 15, 16, 17, 9]],
            [[0, 20, 21, 22, 23, 24, 25, 26, 27, 9]],
            [[0, 30, 31, 32, 33, 34, 35, 36, 37, 9]],
        ],
        dtype=torch.long,
    )
    labels = torch.tensor([[-100, *([-100] * 8), 2]])
    selected = torch.tensor(
        [[False, True, False, False, False, True, False, False, True, False]]
    )

    splice = legalize_qwen3_vl_video_prompt(
        input_ids=request.prompt_input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        position_ids=rotary,
        cache_position=torch.arange(10),
        labels=labels,
        visual_pos_masks=selected,
        deepstack_visual_embeds=response.deepstack_features,
        request=request,
        response=response,
    )

    assert splice.inputs_embeds.shape == (1, 5, 4)
    assert splice.input_ids.tolist() == [[1, 8, 8, 8, 2]]
    assert splice.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert splice.labels.tolist() == [[-100, -100, -100, -100, 2]]
    assert splice.cache_position.tolist() == [0, 1, 2, 3, 4]
    assert splice.visual_pos_masks.tolist() == [[False, True, True, True, False]]
    assert all(branch.shape == (3, 4) for branch in splice.deepstack_visual_embeds)
    assert splice.result_visual_mrope == tuple(
        splice.source_visual_mrope[index] for index in (0, 4, 7)
    )
    assert splice.rope_deltas.tolist() == [[33]]
    record = splice.record()
    assert record["shared_main_deepstack_layout"] is True
    assert record["source_placeholder_spans"] == ((1, 9),)
    assert record["result_placeholder_spans"] == ((1, 4),)
    assert record["source_position_plane_count"] == 4
    assert record["result_position_plane_count"] == 4
    assert record["visual_pos_masks_result_count"] == 3
    assert record["attention_mask_compacted"] is True
    assert record["position_ids_compacted"] is True
    assert record["cache_position_dense"] is True
    assert record["labels_present"] is True
    assert record["training_labels_compacted"] is True
    assert record["visual_labels_ignored"] is True
    assert record["deepstack_result_shapes"] == ((3, 4), (3, 4), (3, 4))
    assert record["deepstack_group_count"] == 3


@requires_torch
def test_prompt_legalizer_preserves_interleaved_video_spans_and_text() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    input_ids = torch.tensor([[1, 8, 8, 3, 8, 8, 4, 8, 8, 5, 8, 8]])
    request = _tensor_request(
        features,
        positions=(1, 2, 4, 5, 7, 8, 10, 11),
        prompt_input_ids=input_ids,
    )
    response = _compress_to_three(request)
    inputs_embeds = torch.arange(48, dtype=torch.float32).reshape(1, 12, 4)
    rotary = torch.stack(
        [
            torch.arange(12),
            torch.arange(12) + 100,
            torch.arange(12) + 200,
        ]
    ).unsqueeze(1)
    selected = torch.zeros(1, 12, dtype=torch.bool)
    selected[0, [1, 7, 11]] = True

    splice = legalize_qwen3_vl_video_prompt(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(1, 12, dtype=torch.long),
        position_ids=rotary,
        cache_position=torch.arange(12),
        labels=None,
        visual_pos_masks=selected,
        deepstack_visual_embeds=response.deepstack_features,
        request=request,
        response=response,
    )

    assert splice.input_ids.tolist() == [[1, 8, 3, 4, 8, 5, 8]]
    assert splice.visual_pos_masks.tolist() == [
        [False, True, False, False, True, False, True]
    ]
    assert splice.cache_position.tolist() == list(range(7))
    assert splice.result_visual_mrope == tuple(
        splice.source_visual_mrope[index] for index in (0, 4, 7)
    )
    record = splice.record()
    assert record["source_placeholder_spans"] == (
        (1, 3),
        (4, 6),
        (7, 9),
        (10, 12),
    )
    assert record["result_placeholder_spans"] == ((1, 2), (4, 5), (6, 7))


@requires_torch
def test_bridge_compresses_video_main_deepstack_mask_and_preserves_text_image(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = _build_torch_model()
    bridge = install_qwen3_vl_video_bridge(model)
    requests: list[Qwen3VLPackedVideoRequest] = []

    def callback(request: Qwen3VLPackedVideoRequest) -> Any:
        requests.append(request)
        return _compress_to_three(request)

    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, callback)
    source = _source_call()
    original_grid = source["video_grid_thw"].clone()
    with bridge.request_context(
        {"sample_id": "sample-7", "video_media_ids": ("sample-7:video-0",)}
    ):
        output = model.forward(**source, use_cache=False)

    forwarded = model.model.language_model.last_forward
    assert output.last_hidden_state.shape == (1, 5, 4)
    assert output.logits.shape[:2] == (1, 5)
    assert output.rope_deltas.tolist() == [[5]]
    assert torch.equal(source["video_grid_thw"], original_grid)
    assert forwarded["visual_pos_masks"].tolist() == [[False, True, True, True, False]]
    assert len(forwarded["deepstack_visual_embeds"]) == 3
    assert all(
        branch.shape == (3, 4) for branch in forwarded["deepstack_visual_embeds"]
    )
    assert requests[0].media_id == "sample-7:video-0"
    latest = bridge.runtime_metadata()["latest_splice"]
    assert latest["result_placeholder_count"] == 3
    assert latest["deepstack_branch_count"] == 3

    text = model.forward(
        input_ids=torch.tensor([[1, 2]]),
        attention_mask=torch.ones(1, 2, dtype=torch.long),
        use_cache=False,
    )
    assert text.last_hidden_state.shape == (1, 2, 4)
    image = model.forward(
        input_ids=torch.tensor([[1, 9, 9, 9, 9, 2]]),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        pixel_values=torch.arange(64, dtype=torch.float32).reshape(16, 4),
        image_grid_thw=torch.tensor([[1, 4, 4]]),
        use_cache=False,
    )
    assert image.last_hidden_state.shape == (1, 6, 4)
    assert len(requests) == 1

    with pytest.raises(BackendIntegrationError, match="cannot mix image"):
        model.forward(
            **source,
            pixel_values=torch.zeros(16, 4),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
            use_cache=False,
        )

    handle.remove()
    bridge.detach()


@requires_torch
def test_bridge_training_compacts_labels_and_backpropagates_all_visual_streams(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = _build_torch_model()
    bridge = install_qwen3_vl_video_bridge(model)
    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, _compress_to_three)
    pixels = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    pixels.requires_grad_(True)
    labels = torch.tensor([[-100, *([-100] * 8), 2]])

    output = model.forward(
        **_source_call(labels=labels, pixels=pixels),
    )
    output.loss.backward()

    assert output.logits.shape[:2] == (1, 5)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert pixels.grad is not None
    assert torch.count_nonzero(pixels.grad).item() > 0

    handle.remove()
    bridge.detach()


@requires_torch
def test_generate_repairs_prefill_cache_and_dense_decode_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = _build_torch_model()

    def simulate_generate(
        generated_model: Any,
        *,
        input_ids: Any,
        attention_mask: Any,
        pixel_values_videos: Any,
        video_grid_thw: Any,
        second_per_grid_ts: Any,
        **kwargs: Any,
    ) -> Any:
        del kwargs
        source_length = int(input_ids.shape[1])
        prefill = generated_model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values_videos=pixel_values_videos,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=True,
            cache_position=torch.arange(source_length),
        )
        prefill_call = dict(generated_model.model.language_model.last_forward)
        rope_delta = int(generated_model.model.rope_deltas.reshape(-1)[0].item())
        dense_cache_position = torch.tensor([source_length])
        positions = torch.full((3, 1, 1), source_length + rope_delta)
        decode = generated_model.forward(
            input_ids=torch.tensor([[7]]),
            attention_mask=torch.ones(1, source_length + 1, dtype=torch.long),
            position_ids=positions,
            video_grid_thw=video_grid_thw,
            past_key_values=object(),
            use_cache=True,
            cache_position=dense_cache_position,
        )
        decode_call = dict(generated_model.model.language_model.last_forward)
        return SimpleNamespace(
            prefill=prefill,
            decode=decode,
            prefill_call=prefill_call,
            decode_call=decode_call,
        )

    model.generate = MethodType(simulate_generate, model)
    bridge = install_qwen3_vl_video_bridge(model)
    handle = bridge.register_callback(QWEN3_VL_VIDEO_SPLICE_SITE, _compress_to_three)

    output = model.generate(**_source_call())

    assert output.prefill_call["inputs_embeds"].shape == (1, 5, 4)
    assert output.prefill_call["cache_position"].tolist() == [0, 1, 2, 3, 4]
    assert output.prefill_call["visual_pos_masks"].shape == (1, 5)
    assert output.decode_call["attention_mask"].shape == (1, 6)
    assert output.decode_call["cache_position"].tolist() == [5]
    assert output.decode_call["position_ids"][:, 0, 0].tolist() == [10, 10, 10]
    assert output.decode_call["visual_pos_masks"] is None
    assert output.decode_call["deepstack_visual_embeds"] is None
    telemetry = bridge.runtime_metadata()["latest_generation"]
    assert telemetry["prefill_complete"] is True
    assert telemetry["source_sequence_length"] == 10
    assert telemetry["result_sequence_length"] == 5
    assert telemetry["removed_count"] == 5
    assert telemetry["decode_repairs"] == 1
    assert telemetry["last_decode_record"]["output_attention_length"] == 6

    handle.remove()
    bridge.detach()


@requires_torch
def test_bridge_rejects_main_only_callback_and_deepstack_layout_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = _build_torch_model()
    bridge = install_qwen3_vl_video_bridge(model)
    handle = bridge.register_callback(
        QWEN3_VL_VIDEO_SPLICE_SITE,
        lambda request: VideoCompressionResult.identity(request.payload),
    )
    with pytest.raises(BackendIntegrationError, match="must return"):
        model.forward(**_source_call(), use_cache=False)
    handle.remove()

    handle = bridge.register_callback(
        QWEN3_VL_VIDEO_SPLICE_SITE,
        lambda request: Qwen3VLVideoCompressionResponse.from_result(
            request,
            _compress_to_three(request).compression_result,
            deepstack_features=(torch.zeros(3, 4),) * 2,
        ),
    )
    with pytest.raises(InvariantError, match="branch count"):
        model.forward(**_source_call(), use_cache=False)
    handle.remove()
    bridge.detach()
