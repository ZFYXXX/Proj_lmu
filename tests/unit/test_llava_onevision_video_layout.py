"""Framework-neutral tests for LLaVA video packing and splice planning."""

from __future__ import annotations

import pytest

np = pytest.importorskip("numpy")

from mllm_comp.backbones.llava_onevision.video_compression import (
    LlavaOneVisionPackedVideoRequest,
    LlavaOneVisionVideoPacking,
    plan_llava_onevision_video_splice,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import BackendIntegrationError, InvariantError


def _rows(count: int, width: int = 3) -> object:
    return np.arange(count * width, dtype="float32").reshape(count, width)


def _request(
    packing: LlavaOneVisionVideoPacking,
    *,
    media_id: str = "clip-0",
) -> LlavaOneVisionPackedVideoRequest:
    payload = packing.payload(
        _rows(packing.packed_token_count),
        media_id=media_id,
    )
    return LlavaOneVisionPackedVideoRequest(
        invocation_id="fixture",
        media_id=media_id,
        payload=payload,
        packing=packing,
        source_visual_span=(2, 2 + packing.packed_token_count),
        prompt_input_ids=(10, 11, -200, 12),
        prepared_attention_mask=None,
        prepared_position_ids=None,
        prepared_labels=None,
    )


@pytest.mark.parametrize(
    ("merge_type", "newline", "packed_count", "structural_positions"),
    [
        ("flat", "one_token", 8, ()),
        ("spatial_unpad", "grid", 12, (2, 5, 8, 11)),
        ("spatial_unpad", "frame", 10, (4, 9)),
        ("spatial_unpad", "one_token", 9, (8,)),
        ("spatial_unpad", "no_token", 8, ()),
        ("spatial", "one_token", 8, ()),
    ],
)
def test_native_packing_partitions_patches_and_newlines(
    merge_type: str,
    newline: str,
    packed_count: int,
    structural_positions: tuple[int, ...],
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
    request = _request(packing)

    assert packing.packed_token_count == packed_count
    assert packing.structural_positions == structural_positions
    assert len(packing.patch_positions) == 8
    assert request.payload.token_count == 8
    assert np.array_equal(
        request.payload.features,
        _rows(packed_count)[list(packing.patch_positions)],
    )
    assert [coordinate.frame_index for coordinate in request.payload.coordinates] == [
        3,
        3,
        3,
        3,
        9,
        9,
        9,
        9,
    ]
    assert [
        coordinate.timestamp_seconds for coordinate in request.payload.coordinates
    ] == [
        0.3,
        0.3,
        0.3,
        0.3,
        0.9,
        0.9,
        0.9,
        0.9,
    ]


def test_identity_plan_preserves_every_packed_and_text_position() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=2,
        grid_height=2,
        grid_width=2,
        patch_merge_type="spatial_unpad",
        newline_position="grid",
    )
    request = _request(packing)
    result = VideoCompressionResult.identity(request.payload)
    sequence_length = packing.packed_token_count + 4

    plan = plan_llava_onevision_video_splice(
        request,
        result,
        sequence_length=sequence_length,
    )

    assert plan.source_visual_span == plan.result_visual_span == (2, 14)
    assert plan.sequence_representative_indices == tuple(range(sequence_length))
    assert plan.structural_source_positions == (2, 5, 8, 11)
    assert plan.structural_result_positions == (2, 5, 8, 11)
    assert [kind for _position, kind, _index in plan.packed_items].count("patch") == 8
    assert [kind for _position, kind, _index in plan.packed_items].count(
        "structural"
    ) == 4


def test_exact_budget_plan_interleaves_anchors_with_unchanged_frame_newlines() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=2,
        grid_height=2,
        grid_width=2,
        patch_merge_type="spatial_unpad",
        newline_position="frame",
    )
    request = _request(packing)
    groups = (((0, 1), (2, 3), (4, 5), (6, 7)),)
    compressed = np.asarray(
        [
            [
                sum(request.payload.features[index][channel] for index in group)
                / len(group)
                for channel in range(3)
            ]
            for group in groups[0]
        ],
        dtype="float32",
    )
    result = VideoCompressionResult.from_source_groups(
        request.payload,
        compressed,
        source_groups=groups,
        representative_indices=((0, 2, 4, 6),),
        exact_budgets=(4,),
    )

    plan = plan_llava_onevision_video_splice(
        request,
        result,
        sequence_length=14,
    )

    assert plan.source_visual_span == (2, 12)
    assert plan.result_visual_span == (2, 8)
    assert plan.packed_items == (
        (0, "patch", 0),
        (2, "patch", 1),
        (4, "structural", 4),
        (5, "patch", 2),
        (7, "patch", 3),
        (9, "structural", 9),
    )
    assert plan.structural_result_positions == (2, 5)
    assert plan.sequence_representative_indices == (
        0,
        1,
        2,
        4,
        6,
        7,
        9,
        11,
        12,
        13,
    )


def test_splice_plan_rejects_out_of_order_representative_anchors() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=1,
        grid_height=2,
        grid_width=2,
        patch_merge_type="flat",
        newline_position="no_token",
    )
    request = _request(packing)
    result = VideoCompressionResult.from_source_groups(
        request.payload,
        _rows(2),
        source_groups=(((0, 3), (1, 2)),),
        representative_indices=((3, 1),),
        exact_budgets=(2,),
    )

    with pytest.raises(InvariantError, match="preserve source order"):
        plan_llava_onevision_video_splice(
            request,
            result,
            sequence_length=8,
        )


def test_splice_plan_rejects_media_drift_and_truncated_source_span() -> None:
    packing = LlavaOneVisionVideoPacking.from_native(
        frame_count=1,
        grid_height=2,
        grid_width=2,
        patch_merge_type="flat",
        newline_position="no_token",
    )
    request = _request(packing)
    other = _request(packing, media_id="other")
    with pytest.raises(InvariantError, match="media identity"):
        plan_llava_onevision_video_splice(
            request,
            VideoCompressionResult.identity(other.payload),
            sequence_length=8,
        )
    with pytest.raises(InvariantError, match="outside"):
        plan_llava_onevision_video_splice(
            request,
            VideoCompressionResult.identity(request.payload),
            sequence_length=5,
        )


def test_packing_fails_closed_for_non_native_modes_and_bad_timing() -> None:
    with pytest.raises(BackendIntegrationError, match="patch merge"):
        LlavaOneVisionVideoPacking.from_native(
            frame_count=1,
            grid_height=2,
            grid_width=2,
            patch_merge_type="unknown",
            newline_position="no_token",
        )
    with pytest.raises(BackendIntegrationError, match="newline"):
        LlavaOneVisionVideoPacking.from_native(
            frame_count=1,
            grid_height=2,
            grid_width=2,
            patch_merge_type="spatial",
            newline_position="unknown",
        )
    with pytest.raises(ValueError, match="timestamps_seconds"):
        LlavaOneVisionVideoPacking.from_native(
            frame_count=2,
            grid_height=2,
            grid_width=2,
            patch_merge_type="flat",
            newline_position="no_token",
            timestamps_seconds=(0.0,),
        )
