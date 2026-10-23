"""Pinned-provider tensor tests for Qwen2.5-VL video legalization."""

from __future__ import annotations

import inspect
from types import MethodType, SimpleNamespace
from typing import Any

import pytest
from tests.fixtures.qwen_video import source_call as _source_call

torch = pytest.importorskip("torch")

from mllm_comp.backbones.qwen25_vl import provider_bridge as image_provider
from mllm_comp.backbones.qwen25_vl.video_compression import (
    QWEN25_VL_VIDEO_SPLICE_SITE,
    Qwen25VLPackedVideoRequest,
    Qwen25VLVideoGrid,
    install_qwen25_vl_video_bridge,
    legalize_qwen25_vl_video_prompt,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from tests.unit.test_qwen25_vl_image_compression_provider_bridge import (
    Qwen2_5_VLForConditionalGeneration,
)


def _pin_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        image_provider,
        "_installed_transformers_version",
        lambda: image_provider.SUPPORTED_TRANSFORMERS_VERSION,
    )


def _compress_to_three(
    request: Qwen25VLPackedVideoRequest,
) -> VideoCompressionResult:
    groups = ((0, 1, 2), (3, 4), (5, 6, 7))
    representatives = (0, 4, 7)
    compressed = torch.stack(
        [request.payload.features[list(group)].mean(dim=0) for group in groups]
    )
    return VideoCompressionResult.from_source_groups(
        request.payload,
        compressed,
        source_groups=(groups,),
        representative_indices=(representatives,),
        exact_budgets=(3,),
        metadata={"algorithm": "fixture.video"},
    )


def _tensor_request(features: Any) -> Qwen25VLPackedVideoRequest:
    grid = Qwen25VLVideoGrid.from_native(
        (2, 4, 4),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    payload = grid.payload(features, media_id="sample-7:video-0")
    return Qwen25VLPackedVideoRequest(
        invocation_id="video-splice-1",
        media_id="sample-7:video-0",
        payload=payload,
        grid=grid,
        source_placeholder_positions=tuple(range(1, 9)),
        prompt_input_ids=torch.tensor([[1, *([8] * 8), 2]]),
        source_attention_mask=torch.ones(1, 10, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[2, 4, 4]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )


def test_install_is_idempotent_signature_preserving_and_reversible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = Qwen2_5_VLForConditionalGeneration()
    original_signature = inspect.signature(model.forward)
    original = {
        "model": model.forward.__func__,
        "core": model.model.forward.__func__,
        "video": model.model.get_video_features.__func__,
        "placeholder": model.model.get_placeholder_mask.__func__,
        "language": model.model.language_model.forward.__func__,
        "generate": model.generate.__func__,
    }

    bridge = install_qwen25_vl_video_bridge(model)

    assert install_qwen25_vl_video_bridge(model) is bridge
    assert bridge.available_sites() == (QWEN25_VL_VIDEO_SPLICE_SITE,)
    assert not bridge.patched
    handle = bridge.register_callback(
        QWEN25_VL_VIDEO_SPLICE_SITE,
        lambda request: VideoCompressionResult.identity(request.payload),
    )
    assert bridge.patched
    assert inspect.signature(model.forward) == original_signature
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


def test_prompt_legalizer_gathers_four_plane_mrope_masks_labels_and_cache() -> None:
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    request = _tensor_request(features)
    result = _compress_to_three(request)
    inputs_embeds = torch.arange(40, dtype=torch.float32).reshape(1, 10, 4)
    input_ids = request.prompt_input_ids
    physical = torch.arange(10, dtype=torch.long).reshape(1, 1, 10)
    rotary = torch.tensor(
        [
            [[0, 10, 11, 12, 13, 14, 15, 16, 17, 9]],
            [[0, 20, 21, 22, 23, 24, 25, 26, 27, 9]],
            [[0, 30, 31, 32, 33, 34, 35, 36, 37, 9]],
        ],
        dtype=torch.long,
    )
    positions = torch.cat((physical, rotary), dim=0)
    labels = torch.tensor([[-100, *([-100] * 8), 2]])

    splice = legalize_qwen25_vl_video_prompt(
        input_ids=input_ids,
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(1, 10, dtype=torch.long),
        position_ids=positions,
        cache_position=torch.arange(10),
        labels=labels,
        request=request,
        compression_result=result,
    )

    assert splice.inputs_embeds.shape == (1, 5, 4)
    assert splice.input_ids.tolist() == [[1, 8, 8, 8, 2]]
    assert splice.attention_mask.tolist() == [[1, 1, 1, 1, 1]]
    assert splice.labels.tolist() == [[-100, -100, -100, -100, 2]]
    assert splice.cache_position.tolist() == [0, 1, 2, 3, 4]
    assert splice.source_visual_mrope == tuple(
        tuple(int(value) for value in rotary[:, 0, index].tolist())
        for index in range(1, 9)
    )
    assert splice.result_visual_mrope == tuple(
        splice.source_visual_mrope[index] for index in (0, 4, 7)
    )
    assert splice.rope_deltas.tolist() == [[33]]
    assert splice.record()["source_video_grid_preserved"] is True


def test_prompt_legalizer_identity_is_exact_and_rejects_supervised_video_labels() -> (
    None
):
    features = torch.arange(32, dtype=torch.float32).reshape(8, 4)
    request = _tensor_request(features)
    identity = VideoCompressionResult.identity(request.payload)
    embeds = torch.randn(1, 10, 4)
    positions = torch.arange(10).reshape(1, 1, 10).expand(3, 1, 10)
    mask = torch.ones(1, 10, dtype=torch.long)
    labels = torch.tensor([[-100, *([-100] * 8), 2]])

    splice = legalize_qwen25_vl_video_prompt(
        input_ids=request.prompt_input_ids,
        inputs_embeds=embeds,
        attention_mask=mask,
        position_ids=positions,
        cache_position=None,
        labels=labels,
        request=request,
        compression_result=identity,
    )

    assert torch.equal(splice.input_ids, request.prompt_input_ids)
    assert torch.equal(splice.inputs_embeds, embeds)
    assert torch.equal(splice.attention_mask, mask)
    assert torch.equal(splice.position_ids, positions)
    assert torch.equal(splice.labels, labels)
    assert splice.rope_deltas.tolist() == [[0]]

    bad_labels = labels.clone()
    bad_labels[0, 3] = 7
    with pytest.raises(InvariantError, match="source video labels must be ignored"):
        legalize_qwen25_vl_video_prompt(
            input_ids=request.prompt_input_ids,
            inputs_embeds=embeds,
            attention_mask=mask,
            position_ids=positions,
            cache_position=None,
            labels=bad_labels,
            request=request,
            compression_result=identity,
        )


def test_bridge_compresses_only_video_and_preserves_source_grid_and_images(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = install_qwen25_vl_video_bridge(model)
    requests: list[Qwen25VLPackedVideoRequest] = []

    def callback(request: Qwen25VLPackedVideoRequest) -> VideoCompressionResult:
        requests.append(request)
        return _compress_to_three(request)

    handle = bridge.register_callback(QWEN25_VL_VIDEO_SPLICE_SITE, callback)
    source = _source_call()
    original_grid = source["video_grid_thw"].clone()
    with bridge.request_context(
        {"sample_id": "sample-7", "video_media_ids": ("sample-7:video-0",)}
    ):
        output = model.forward(**source, use_cache=False)

    assert output.last_hidden_state.shape == (1, 5, 4)
    assert output.logits.shape[:2] == (1, 5)
    assert output.rope_deltas.tolist() == [[5]]
    assert torch.equal(source["video_grid_thw"], original_grid)
    assert len(requests) == 1
    request = requests[0]
    assert request.media_id == "sample-7:video-0"
    assert request.grid.source_grid_thw == (2, 4, 4)
    assert request.payload.token_count == 8
    assert bridge.runtime_metadata()["latest_splice"]["result_placeholder_count"] == 3

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
    with pytest.raises(BackendIntegrationError, match="requires use_cache=False"):
        model.forward(**source, use_cache=True)

    handle.remove()
    bridge.detach()


@pytest.mark.parametrize("cache_kwargs", [{}, {"use_cache": None}, {"use_cache": False}],
    ids=["omitted", "none", "false"])
def test_bridge_training_compacts_labels_and_backpropagates_to_video_path(
    monkeypatch: pytest.MonkeyPatch,
    cache_kwargs: dict[str, Any],
) -> None:
    _pin_provider(monkeypatch)
    model = Qwen2_5_VLForConditionalGeneration()
    bridge = install_qwen25_vl_video_bridge(model)
    handle = bridge.register_callback(
        QWEN25_VL_VIDEO_SPLICE_SITE,
        _compress_to_three,
    )
    pixels = torch.arange(128, dtype=torch.float32).reshape(32, 4)
    pixels.requires_grad_(True)
    labels = torch.tensor([[-100, *([-100] * 8), 2]])

    output = model.forward(
        **_source_call(labels=labels, pixels=pixels),
        **cache_kwargs,
    )
    output.loss.backward()

    assert output.logits.shape[:2] == (1, 5)
    assert output.loss is not None and torch.isfinite(output.loss)
    assert pixels.grad is not None
    assert torch.count_nonzero(pixels.grad).item() > 0

    handle.remove()
    bridge.detach()


def test_generate_repairs_prefill_cache_and_each_dense_decode_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _pin_provider(monkeypatch)
    model = Qwen2_5_VLForConditionalGeneration()

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
        physical = dense_cache_position.reshape(1, 1, 1)
        rotary = torch.full((3, 1, 1), source_length + rope_delta)
        decode = generated_model.forward(
            input_ids=torch.tensor([[7]]),
            attention_mask=torch.ones(1, source_length + 1, dtype=torch.long),
            position_ids=torch.cat((physical, rotary), dim=0),
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
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
    bridge = install_qwen25_vl_video_bridge(model)
    handle = bridge.register_callback(
        QWEN25_VL_VIDEO_SPLICE_SITE,
        _compress_to_three,
    )

    output = model.generate(**_source_call())

    assert output.prefill_call["inputs_embeds"].shape == (1, 5, 4)
    assert output.prefill_call["cache_position"].tolist() == [0, 1, 2, 3, 4]
    assert output.decode_call["attention_mask"].shape == (1, 6)
    assert output.decode_call["cache_position"].tolist() == [5]
    assert output.decode_call["position_ids"][:, 0, 0].tolist() == [5, 10, 10, 10]
    telemetry = bridge.runtime_metadata()["latest_generation"]
    assert telemetry["prefill_complete"] is True
    assert telemetry["source_sequence_length"] == 10
    assert telemetry["result_sequence_length"] == 5
    assert telemetry["removed_count"] == 5
    assert telemetry["decode_repairs"] == 1
    assert telemetry["last_decode_record"]["output_attention_length"] == 6

    handle.remove()
    bridge.detach()
