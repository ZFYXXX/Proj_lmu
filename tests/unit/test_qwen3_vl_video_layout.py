"""Framework-neutral Qwen3-VL video and DeepStack layout contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
    plan_qwen3_vl_video_splice,
)
from mllm_comp.compressors.video import VideoCompressionResult
from mllm_comp.core.errors import InvariantError


def _request(
    *,
    media_id: str = "sample-7:video-0",
    positions: tuple[int, ...] = tuple(range(2, 10)),
    deepstack_count: int = 3,
) -> Qwen3VLPackedVideoRequest:
    grid = Qwen3VLVideoGrid.from_native(
        (2, 4, 4),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    features = np.arange(grid.token_count * 4, dtype=np.float32).reshape(
        grid.token_count,
        4,
    )
    payload = grid.payload(features, media_id=media_id)
    deepstack = tuple(features + 100 * (index + 1) for index in range(deepstack_count))
    return Qwen3VLPackedVideoRequest(
        invocation_id="video-splice-1",
        media_id=media_id,
        payload=payload,
        deepstack_features=deepstack,
        grid=grid,
        source_placeholder_positions=positions,
        prompt_input_ids=np.arange(12, dtype=np.int64).reshape(1, 12),
        source_attention_mask=np.ones((1, 12), dtype=np.int64),
        source_video_grid_thw=np.asarray([[2, 4, 4]], dtype=np.int64),
        source_second_per_grid_ts=np.asarray([0.5], dtype=np.float32),
    )


def _three_token_response(
    request: Qwen3VLPackedVideoRequest,
) -> Qwen3VLVideoCompressionResponse:
    groups = ((0, 1, 2), (3, 4), (5, 6, 7))
    representatives = (0, 4, 7)
    features = np.stack(
        [request.payload.features[list(group)].mean(axis=0) for group in groups]
    )
    result = VideoCompressionResult.from_source_groups(
        request.payload,
        features,
        source_groups=(groups,),
        representative_indices=(representatives,),
        exact_budgets=(3,),
        metadata={"algorithm": "fixture.video"},
    )
    deepstack = tuple(
        np.stack([branch[list(group)].mean(axis=0) for group in groups])
        for branch in request.deepstack_features
    )
    return Qwen3VLVideoCompressionResponse.from_result(
        request,
        result,
        deepstack_features=deepstack,
    )


def test_grid_flattens_post_merger_raster_order_and_optional_timing() -> None:
    grid = Qwen3VLVideoGrid.from_native(
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
    untimed = Qwen3VLVideoGrid.from_native(
        (1, 4, 4),
        spatial_merge_size=2,
    )
    assert all(item.timestamp_seconds is None for item in untimed.coordinates)


def test_payload_freezes_source_grid_and_deepstack_policy() -> None:
    request = _request()

    assert request.payload.signals["qwen3_vl.source_video_grid_thw"] == (2, 4, 4)
    assert request.payload.signals["qwen3_vl.spatial_merge_size"] == 2
    assert request.payload.metadata["merged_grid_thw"] == (2, 2, 2)
    assert request.payload.metadata["grid_policy"] == (
        "preserve_source_grid_and_gather_sparse_mrope"
    )
    assert request.payload.metadata["deepstack_policy"] == (
        "shared_source_groups_all_branches"
    )


def test_identity_response_keeps_main_and_every_deepstack_branch_exact() -> None:
    request = _request()
    response = Qwen3VLVideoCompressionResponse.identity(request)

    plan = plan_qwen3_vl_video_splice(request, response, sequence_length=12)

    assert response.compression_result.features is request.payload.features
    assert response.deepstack_features == request.deepstack_features
    assert plan.retained_placeholder_positions == tuple(range(2, 10))
    assert plan.removed_placeholder_positions == ()
    assert plan.kept_sequence_indices == tuple(range(12))


def test_exact_budget_response_shares_one_layout_across_all_deepstack_branches() -> (
    None
):
    request = _request()
    response = _three_token_response(request)

    plan = plan_qwen3_vl_video_splice(request, response, sequence_length=12)

    assert plan.representative_indices == (0, 4, 7)
    assert plan.retained_placeholder_positions == (2, 6, 9)
    assert plan.removed_placeholder_positions == (3, 4, 5, 7, 8)
    assert plan.result_placeholder_positions == (2, 3, 4)
    assert plan.kept_sequence_indices == (0, 1, 2, 6, 9, 10, 11)
    assert response.synchronization_source_groups == (((0, 1, 2), (3, 4), (5, 6, 7)),)
    assert all(branch.shape == (3, 4) for branch in response.deepstack_features)
    assert response.compression_result.coordinates == tuple(
        request.payload.coordinates[index] for index in (0, 4, 7)
    )
    json.dumps(dict(response.compression_result.metadata))


def test_exact_budget_preserves_native_interleaved_video_placeholder_spans() -> None:
    request = _request(positions=(1, 2, 4, 5, 7, 8, 10, 11))
    response = _three_token_response(request)

    plan = plan_qwen3_vl_video_splice(request, response, sequence_length=12)

    assert plan.retained_placeholder_positions == (1, 7, 11)
    assert plan.removed_placeholder_positions == (2, 4, 5, 8, 10)
    assert plan.kept_sequence_indices == (0, 1, 3, 6, 7, 9, 11)
    assert plan.result_placeholder_positions == (1, 4, 6)
    assert plan.source_visual_span == (1, 12)
    assert plan.result_visual_span == (1, 7)


def test_response_rejects_deepstack_shape_branch_or_layout_drift() -> None:
    request = _request()
    response = _three_token_response(request)
    result = response.compression_result

    with pytest.raises(ValueError, match="must match main features"):
        Qwen3VLVideoCompressionResponse(
            compression_result=result,
            deepstack_features=(np.zeros((2, 4), dtype=np.float32),) * 3,
            synchronization_source_groups=result.source_groups,
        )
    with pytest.raises(ValueError, match="synchronization groups"):
        Qwen3VLVideoCompressionResponse(
            compression_result=result,
            deepstack_features=response.deepstack_features,
            synchronization_source_groups=(((0,), (1,), (2,)),),
        )
    with pytest.raises(InvariantError, match="branch count"):
        Qwen3VLVideoCompressionResponse.from_result(
            request,
            result,
            deepstack_features=response.deepstack_features[:2],
        )


def test_request_and_plan_reject_placeholder_or_media_drift() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        _request(positions=(2, 3, 4, 5, 7, 8, 10, 9))
    request = _request()
    other = _request(media_id="other:video-0")
    changed = Qwen3VLVideoCompressionResponse.identity(other)
    with pytest.raises(InvariantError, match="media identity"):
        plan_qwen3_vl_video_splice(request, changed, sequence_length=12)
    with pytest.raises(InvariantError, match="outside the prompt"):
        plan_qwen3_vl_video_splice(
            request,
            Qwen3VLVideoCompressionResponse.identity(request),
            sequence_length=9,
        )
