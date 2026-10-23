"""Deterministic native video packing and splice contracts for LLaVA-OneVision."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.llava_onevision.video_compression import (
    LLAVA_ONEVISION_VIDEO_SPLICE_SITE,
    LlavaOneVisionPackedVideoRequest,
    LlavaOneVisionVideoPacking,
    PinnedLlavaOneVisionVideoBridge,
    legalize_llava_onevision_video_splice,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import InvariantError


try:
    import torch
except ImportError:  # pragma: no cover - exercised by the local control runtime
    torch = None

pytestmark = pytest.mark.skipif(
    torch is None,
    reason="the tensor bridge requires the existing PyTorch runtime",
)


def _prepared_fixture(
    packing: LlavaOneVisionVideoPacking,
    *,
    media_id: str = "clip-0",
    bad_visual_label: bool = False,
) -> tuple[
    tuple[Any, ...],
    LlavaOneVisionPackedVideoRequest,
    Any,
]:
    hidden_size = 4
    packed = torch.arange(
        packing.packed_token_count * hidden_size,
        dtype=torch.float32,
    ).reshape(packing.packed_token_count, hidden_size)
    prefix = torch.full((1, hidden_size), -1.0)
    suffix = torch.tensor(
        [[-2.0] * hidden_size, [-3.0] * hidden_size],
        dtype=torch.float32,
    )
    embeds = torch.cat((prefix, packed, suffix), dim=0).unsqueeze(0)
    sequence_length = int(embeds.shape[1])
    positions = torch.arange(sequence_length).unsqueeze(0)
    mask = torch.ones((1, sequence_length), dtype=torch.long)
    labels = torch.tensor(
        [[10, *([-100] * packing.packed_token_count), 11, 12]],
        dtype=torch.long,
    )
    if bad_visual_label:
        labels[0, 1] = 7
    payload = packing.payload(packed, media_id=media_id)
    request = LlavaOneVisionPackedVideoRequest(
        invocation_id="fixture-invocation",
        media_id=media_id,
        payload=payload,
        packing=packing,
        source_visual_span=(1, 1 + packing.packed_token_count),
        prompt_input_ids=torch.tensor([10, -200, 11, 12]),
        prepared_attention_mask=mask,
        prepared_position_ids=positions,
        prepared_labels=labels,
    )
    return (None, positions, mask, None, embeds, labels), request, packed


@pytest.mark.parametrize(
    ("merge_type", "newline", "expected_structural"),
    [
        ("flat", "one_token", 0),
        ("spatial_unpad", "grid", 4),
        ("spatial_unpad", "frame", 2),
        ("spatial_unpad", "one_token", 1),
        ("spatial_unpad", "no_token", 0),
        ("spatial", "one_token", 0),
    ],
)
def test_native_video_packing_identity_is_exact(
    merge_type: str,
    newline: str,
    expected_structural: int,
) -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=2,
        grid_height=2,
        grid_width=2,
        patch_merge_type=merge_type,
        newline_position=newline,
        frame_indices=(3, 9),
        timestamps_seconds=(0.3, 0.9),
    )
    prepared, request, _packed = _prepared_fixture(packing)

    result = VideoCompressionResult.identity(request.payload)
    splice = legalize_llava_onevision_video_splice(prepared, request, result)

    assert len(packing.patch_positions) == 8
    assert len(packing.structural_positions) == expected_structural
    assert [coordinate.frame_index for coordinate in packing.coordinates] == [
        3,
        3,
        3,
        3,
        9,
        9,
        9,
        9,
    ]
    assert torch.equal(splice.prepared[4], prepared[4])
    assert torch.equal(splice.prepared[2], prepared[2])
    assert torch.equal(splice.prepared[1], prepared[1])
    assert torch.equal(splice.prepared[5], prepared[5])
    assert splice.source_visual_span == splice.result_visual_span
    assert splice.sequence_representative_indices == tuple(
        range(int(prepared[4].shape[1]))
    )
    assert splice.structural_result_positions == packing.structural_positions
    assert splice.record()["compression_applied"] is False


def test_exact_budget_merge_preserves_newlines_and_repairs_sequence_layout() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=2,
        grid_height=2,
        grid_width=2,
        patch_merge_type="spatial_unpad",
        newline_position="frame",
    )
    prepared, request, _packed = _prepared_fixture(packing)
    groups = (((0, 1), (2, 3), (4, 5), (6, 7)),)
    representatives = ((0, 2, 4, 6),)
    compressed = torch.stack(
        [request.payload.features[list(group)].mean(dim=0) for group in groups[0]],
        dim=0,
    )
    result = VideoCompressionResult.from_source_groups(
        request.payload,
        compressed,
        source_groups=groups,
        representative_indices=representatives,
        exact_budgets=(4,),
        metadata={"algorithm": "fixture-merge"},
    )

    splice = legalize_llava_onevision_video_splice(prepared, request, result)
    _, positions, mask, _, embeds, labels = splice.prepared

    assert result.source_count == 8
    assert result.result_count == 4
    assert splice.source_visual_span == (1, 11)
    assert splice.result_visual_span == (1, 7)
    assert tuple(embeds.shape) == (1, 9, 4)
    assert tuple(mask.shape) == (1, 9)
    assert torch.equal(positions, torch.arange(9).unsqueeze(0))
    assert labels[0, 1:7].tolist() == [-100] * 6
    assert splice.structural_source_positions == (4, 9)
    assert splice.structural_result_positions == (2, 5)
    source_visual = prepared[4][0, 1:11]
    result_visual = embeds[0, 1:7]
    assert torch.equal(result_visual[2], source_visual[4])
    assert torch.equal(result_visual[5], source_visual[9])
    assert splice.record()["compression_applied"] is True
    assert splice.record()["result_patch_count"] == 4


def test_splice_preserves_gradient_path_through_merged_patches() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=1,
        grid_height=2,
        grid_width=2,
        patch_merge_type="flat",
        newline_position="no_token",
    )
    prepared, request, packed = _prepared_fixture(packing)
    packed.requires_grad_(True)
    # Rebuild the prepared/payload pair from the same differentiable leaf.
    embeds = torch.cat(
        (
            torch.full((1, 4), -1.0),
            packed,
            torch.full((2, 4), -2.0),
        ),
        dim=0,
    ).unsqueeze(0)
    prepared = (
        None,
        prepared[1],
        prepared[2],
        None,
        embeds,
        prepared[5],
    )
    payload = packing.payload(packed, media_id="clip-0")
    request = LlavaOneVisionPackedVideoRequest(
        invocation_id=request.invocation_id,
        media_id="clip-0",
        payload=payload,
        packing=packing,
        source_visual_span=request.source_visual_span,
        prompt_input_ids=request.prompt_input_ids,
        prepared_attention_mask=request.prepared_attention_mask,
        prepared_position_ids=request.prepared_position_ids,
        prepared_labels=request.prepared_labels,
    )
    groups = (((0, 1), (2, 3)),)
    compressed = torch.stack(
        [payload.features[list(group)].mean(dim=0) for group in groups[0]], dim=0
    )
    result = VideoCompressionResult.from_source_groups(
        payload,
        compressed,
        source_groups=groups,
        representative_indices=((0, 2),),
        exact_budgets=(2,),
    )

    splice = legalize_llava_onevision_video_splice(prepared, request, result)
    splice.prepared[4].sum().backward()

    assert packed.grad is not None
    assert torch.count_nonzero(packed.grad).item() == packed.numel()


def test_splice_rejects_changed_media_identity_and_nonignored_visual_labels() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=1,
        grid_height=2,
        grid_width=2,
        patch_merge_type="flat",
        newline_position="no_token",
    )
    prepared, request, packed = _prepared_fixture(packing)
    other_payload = packing.payload(packed, media_id="other-clip")
    with pytest.raises(InvariantError, match="media identity"):
        legalize_llava_onevision_video_splice(
            prepared,
            request,
            VideoCompressionResult.identity(other_payload),
        )

    prepared, request, _packed = _prepared_fixture(
        packing,
        bad_visual_label=True,
    )
    with pytest.raises(InvariantError, match="labels must be ignored"):
        legalize_llava_onevision_video_splice(
            prepared,
            request,
            VideoCompressionResult.identity(request.payload),
        )


class _FakeNativeVideoModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            model_type="llava_qwen",
            mm_spatial_pool_stride=2,
            mm_spatial_pool_mode="bilinear",
            mm_patch_merge_type="flat",
            mm_newline_position="no_token",
            add_faster_video=False,
        )
        self.calls: list[tuple[str, ...]] = []
        self._vision = SimpleNamespace(num_patches_per_side=4)

    def get_vision_tower(self) -> Any:
        return self._vision

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids: Any,
        position_ids: Any,
        attention_mask: Any,
        past_key_values: Any,
        labels: Any,
        images: Any,
        modalities: Any = ("image",),
        image_sizes: Any = None,
    ) -> tuple[Any, ...]:
        del image_sizes
        resolved_modalities = (
            (modalities,) if isinstance(modalities, str) else tuple(modalities)
        )
        self.calls.append(resolved_modalities)
        if int(input_ids.shape[1]) == 1 or past_key_values is not None:
            return (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                None,
                labels,
            )
        frame_count = int(images[0].shape[0])
        packed_count = frame_count * 4
        hidden_size = 4
        packed = torch.arange(
            packed_count * hidden_size,
            dtype=torch.float32,
        ).reshape(packed_count, hidden_size)
        prefix = torch.full((1, hidden_size), -1.0)
        suffix = torch.full((1, hidden_size), -2.0)
        embeds = torch.cat((prefix, packed, suffix), dim=0).unsqueeze(0)
        expanded_length = int(embeds.shape[1])
        expanded_positions = (
            torch.arange(expanded_length).unsqueeze(0)
            if position_ids is not None
            else None
        )
        expanded_mask = (
            torch.ones((1, expanded_length), dtype=attention_mask.dtype)
            if attention_mask is not None
            else None
        )
        expanded_labels = (
            torch.tensor([[10, *([-100] * packed_count), 11]])
            if labels is not None
            else None
        )
        return (
            None,
            expanded_positions,
            expanded_mask,
            None,
            embeds,
            expanded_labels,
        )


def test_instance_bridge_emits_payload_and_only_mutates_video_prefill() -> None:
    model = _FakeNativeVideoModel()
    bridge = PinnedLlavaOneVisionVideoBridge(
        model,
        _contract_verifier=lambda _model: None,
    )
    requests: list[LlavaOneVisionPackedVideoRequest] = []

    def compress(request: LlavaOneVisionPackedVideoRequest) -> VideoCompressionResult:
        requests.append(request)
        groups = tuple(
            (index, index + 1) for index in range(0, request.payload.token_count, 2)
        )
        features = torch.stack(
            [request.payload.features[list(group)].mean(dim=0) for group in groups],
            dim=0,
        )
        return VideoCompressionResult.from_source_groups(
            request.payload,
            features,
            source_groups=(groups,),
            representative_indices=(tuple(group[0] for group in groups),),
            exact_budgets=(len(groups),),
        )

    handle = bridge.register_callback(LLAVA_ONEVISION_VIDEO_SPLICE_SITE, compress)
    input_ids = torch.tensor([[10, -200, 11]])
    attention_mask = torch.ones_like(input_ids)
    position_ids = torch.arange(3).unsqueeze(0)
    labels = torch.tensor([[10, -100, 11]])
    images = [torch.zeros((2, 3, 8, 8))]
    with bridge.request_context(
        {
            "sample_id": "sample-7",
            "video": {
                "media_id": "sample-7:video-0",
                "frame_indices": (2, 8),
                "timestamps_seconds": (0.2, 0.8),
            },
        }
    ):
        output = model.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            None,
            labels,
            images,
            modalities=("video",),
            image_sizes=[(8, 8)],
        )

    assert len(requests) == 1
    request = requests[0]
    assert request.media_id == "sample-7:video-0"
    assert request.payload.token_count == 8
    assert [item.frame_index for item in request.payload.coordinates] == [
        2,
        2,
        2,
        2,
        8,
        8,
        8,
        8,
    ]
    assert tuple(output[4].shape) == (1, 6, 4)
    assert output[5][0, 1:5].tolist() == [-100] * 4
    assert torch.equal(output[1], torch.arange(6).unsqueeze(0))
    assert bridge.runtime_metadata()["callback_count"] == 1
    assert bridge.runtime_metadata()["latest_splice"]["result_patch_count"] == 4

    image_output = model.prepare_inputs_labels_for_multimodal(
        input_ids,
        position_ids,
        attention_mask,
        None,
        labels,
        images,
        modalities=("image",),
        image_sizes=[(8, 8)],
    )
    assert tuple(image_output[4].shape) == (1, 10, 4)
    decode = model.prepare_inputs_labels_for_multimodal(
        torch.tensor([[11]]),
        None,
        torch.ones((1, 1), dtype=torch.long),
        object(),
        None,
        images,
        modalities=("video",),
        image_sizes=[(8, 8)],
    )
    assert decode[0].tolist() == [[11]]
    assert bridge.runtime_metadata()["callback_count"] == 1

    handle.remove()
    assert bridge.patched is False
    assert "prepare_inputs_labels_for_multimodal" not in vars(model)
