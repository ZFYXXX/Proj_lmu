"""Framework-neutral Qwen2.5-VL video grid and splice contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mllm_comp.backbones.qwen25_vl.video_compression import (
    Qwen25VLPackedVideoRequest,
    Qwen25VLVideoGrid,
    plan_qwen25_vl_video_splice,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import InvariantError


def _request(
    *,
    media_id: str = "sample-7:video-0",
    positions: tuple[int, ...] = tuple(range(2, 10)),
) -> Qwen25VLPackedVideoRequest:
    grid = Qwen25VLVideoGrid.from_native(
        (2, 4, 4),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    features = np.arange(grid.token_count * 4, dtype=np.float32).reshape(
        grid.token_count,
        4,
    )
    payload = grid.payload(features, media_id=media_id)
    return Qwen25VLPackedVideoRequest(
        invocation_id="video-splice-1",
        media_id=media_id,
        payload=payload,
        grid=grid,
        source_placeholder_positions=positions,
        prompt_input_ids=np.arange(12, dtype=np.int64).reshape(1, 12),
        source_attention_mask=np.ones((1, 12), dtype=np.int64),
        source_video_grid_thw=np.asarray([[2, 4, 4]], dtype=np.int64),
        source_second_per_grid_ts=np.asarray([0.5], dtype=np.float32),
    )


def _three_token_result(
    request: Qwen25VLPackedVideoRequest,
) -> VideoCompressionResult:
    groups = ((0, 1, 2), (3, 4), (5, 6, 7))
    representatives = (0, 4, 7)
    features = np.stack(
        [request.payload.features[list(group)].mean(axis=0) for group in groups]
    )
    return VideoCompressionResult.from_source_groups(
        request.payload,
        features,
        source_groups=(groups,),
        representative_indices=(representatives,),
        exact_budgets=(3,),
        metadata={"algorithm": "fixture.video"},
    )


def test_grid_flattens_native_post_merger_raster_order_and_timing() -> None:
    grid = Qwen25VLVideoGrid.from_native(
        (2, 4, 6),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )

    assert grid.source_grid_thw == (2, 4, 6)
    assert (grid.merged_height, grid.merged_width, grid.token_count) == (2, 3, 12)
    assert [
        (
            coordinate.t,
            coordinate.h,
            coordinate.w,
            coordinate.tubelet_index,
            coordinate.timestamp_seconds,
        )
        for coordinate in (
            grid.coordinates[0],
            grid.coordinates[3],
            grid.coordinates[6],
        )
    ] == [
        (0, 0, 0, 0, 0.0),
        (0, 1, 0, 0, 0.0),
        (1, 0, 0, 1, 0.5),
    ]


def test_payload_preserves_source_grid_instead_of_fabricating_result_grid() -> None:
    request = _request()

    assert request.payload.signals["qwen25_vl.source_video_grid_thw"] == (2, 4, 4)
    assert request.payload.signals["qwen25_vl.spatial_merge_size"] == 2
    assert request.payload.metadata["merged_grid_thw"] == (2, 2, 2)
    assert request.payload.metadata["grid_policy"] == (
        "preserve_source_grid_and_gather_sparse_mrope"
    )


@pytest.mark.parametrize(
    ("grid", "merge_size", "timing", "error"),
    [
        ((1, 4.0, 4), 2, 0.5, TypeError),
        ((1, 3, 4), 2, 0.5, ValueError),
        ((1, 4, 4), 0, 0.5, ValueError),
        ((1, 4, 4), 2, float("nan"), ValueError),
    ],
)
def test_grid_rejects_non_native_or_ambiguous_values(
    grid: tuple[object, object, object],
    merge_size: int,
    timing: float,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        Qwen25VLVideoGrid.from_native(
            grid,  # type: ignore[arg-type]
            spatial_merge_size=merge_size,
            second_per_grid_seconds=timing,
        )


def test_identity_plan_keeps_every_native_placeholder_and_grid_anchor() -> None:
    request = _request()
    result = VideoCompressionResult.identity(request.payload)

    plan = plan_qwen25_vl_video_splice(request, result, sequence_length=12)

    assert plan.source_visual_span == (2, 10)
    assert plan.result_visual_span == (2, 10)
    assert plan.retained_placeholder_positions == tuple(range(2, 10))
    assert plan.removed_placeholder_positions == ()
    assert plan.kept_sequence_indices == tuple(range(12))
    assert result.coordinates == request.payload.coordinates


def test_exact_budget_plan_gathers_sparse_source_anchors_in_result_order() -> None:
    request = _request()
    result = _three_token_result(request)

    plan = plan_qwen25_vl_video_splice(request, result, sequence_length=12)

    assert plan.representative_indices == (0, 4, 7)
    assert plan.retained_placeholder_positions == (2, 6, 9)
    assert plan.removed_placeholder_positions == (3, 4, 5, 7, 8)
    assert plan.result_placeholder_positions == (2, 3, 4)
    assert plan.kept_sequence_indices == (0, 1, 2, 6, 9, 10, 11)
    assert result.coordinates == tuple(
        request.payload.coordinates[index] for index in (0, 4, 7)
    )
    json.dumps(dict(result.metadata))


def test_request_rejects_noncontiguous_or_wrong_count_placeholder_spans() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        _request(positions=(2, 3, 4, 5, 7, 8, 9, 10))
    with pytest.raises(ValueError, match="must match the source grid"):
        _request(positions=tuple(range(2, 9)))


def test_plan_rejects_media_drift_and_out_of_prompt_source_span() -> None:
    request = _request()
    other = _request(media_id="other:video-0")
    changed_media = VideoCompressionResult.identity(other.payload)

    with pytest.raises(InvariantError, match="media identity"):
        plan_qwen25_vl_video_splice(
            request,
            changed_media,
            sequence_length=12,
        )
    with pytest.raises(InvariantError, match="outside the prompt"):
        plan_qwen25_vl_video_splice(
            request,
            VideoCompressionResult.identity(request.payload),
            sequence_length=9,
        )
