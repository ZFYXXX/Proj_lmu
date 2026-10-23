from __future__ import annotations

from dataclasses import dataclass, replace
import json

import pytest

from mllm_comp.core.layout import (
    AlignmentDelta,
    CouplingPolicy,
    layout_delta_record,
)
from mllm_comp.compressors.video import (
    VideoCompressionResult,
    VideoTokenCoordinate,
    VideoTokenPayload,
)


@dataclass(frozen=True)
class _TensorLike:
    shape: tuple[int, ...]


def _coordinates(*counts: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (temporal_index, 0, spatial_index)
        for count in counts
        for temporal_index in range(count)
        for spatial_index in (0,)
    )


def _payload(*, with_raw: bool = False) -> VideoTokenPayload:
    values: dict[str, object] = {
        "features": _TensorLike((7, 4)),
        "media_ids": ("video-a", "video-b"),
        "video_offsets": (0, 4, 7),
        "coordinates": _coordinates(4, 3),
        "token_mask": (True,) * 7,
        "signals": {"encoder_attention": object()},
        "metadata": {"provider": "unit-test"},
    }
    if with_raw:
        values.update(
            {
                "raw_features": _TensorLike((9, 6)),
                "raw_video_offsets": (0, 5, 9),
                "raw_coordinates": _coordinates(5, 4),
            }
        )
    return VideoTokenPayload(**values)  # type: ignore[arg-type]


def _compressed_result() -> VideoCompressionResult:
    return VideoCompressionResult.from_source_groups(
        _payload(),
        _TensorLike((4, 4)),
        source_groups=(
            ((0, 1), (3,)),
            ((0,), (1, 2)),
        ),
        representative_indices=((1, 3), (0, 2)),
        exact_budgets=(2, 2),
        metadata={"algorithm": "test.video"},
    )


def test_payload_freezes_media_boundaries_coordinates_and_signals() -> None:
    payload = _payload(with_raw=True)

    assert payload.video_count == 2
    assert payload.token_count == 7
    assert payload.raw_token_count == 9
    assert payload.video_span("video-a") == slice(0, 4)
    assert payload.video_span(1) == slice(4, 7)
    assert payload.raw_video_span("video-b") == slice(5, 9)
    assert payload.coordinates[4] == VideoTokenCoordinate(0, 0, 0)
    assert payload.coordinates[4].frame_index == 0
    assert payload.signals["encoder_attention"] is not None
    with pytest.raises(TypeError):
        payload.signals["new"] = object()  # type: ignore[index]
    with pytest.raises(TypeError):
        payload.metadata["provider"] = "changed"  # type: ignore[index]


def test_coordinate_supports_explicit_frame_tubelet_and_timing_identity() -> None:
    coordinate = VideoTokenCoordinate(
        2,
        3,
        4,
        frame_index=5,
        tubelet_index=1,
        timestamp_seconds=0.125,
    )

    assert (coordinate.t, coordinate.h, coordinate.w) == (2, 3, 4)
    assert coordinate.frame_index == 5
    assert coordinate.tubelet_index == 1
    assert coordinate.timestamp_seconds == 0.125
    with pytest.raises(ValueError, match="finite and non-negative"):
        VideoTokenCoordinate(0, 0, 0, timestamp_seconds=float("nan"))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"media_ids": ("same", "same")}, "unique"),
        ({"video_offsets": (0, 0, 7)}, "non-empty increasing"),
        ({"video_offsets": (0, 4, 6)}, "end at token count"),
        ({"coordinates": _coordinates(4, 2)}, "one entry per token"),
        ({"token_mask": (True,) * 6}, "one entry per projected token"),
        ({"signals": {"": object()}}, "non-empty strings"),
    ],
)
def test_payload_rejects_ambiguous_or_incomplete_boundaries(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "features": _TensorLike((7, 4)),
        "media_ids": ("video-a", "video-b"),
        "video_offsets": (0, 4, 7),
        "coordinates": _coordinates(4, 3),
        "token_mask": (True,) * 7,
    }
    values.update(overrides)
    with pytest.raises((TypeError, ValueError), match=message):
        VideoTokenPayload(**values)  # type: ignore[arg-type]


def test_raw_features_require_their_own_complete_layout() -> None:
    values = {
        "features": _TensorLike((7, 4)),
        "media_ids": ("video-a", "video-b"),
        "video_offsets": (0, 4, 7),
        "coordinates": _coordinates(4, 3),
    }

    with pytest.raises(ValueError, match="requires raw_features"):
        VideoTokenPayload(**values, raw_video_offsets=(0, 4, 7))
    with pytest.raises(ValueError, match=r"video_count \+ 1"):
        VideoTokenPayload(**values, raw_features=_TensorLike((9, 6)))


def test_identity_result_is_per_video_and_preserves_original_objects() -> None:
    payload = _payload()
    result = VideoCompressionResult.identity(
        payload, metadata={"algorithm": "identity.video"}
    )

    assert result.features is payload.features
    assert result.media_ids == payload.media_ids
    assert result.source_video_offsets == (0, 4, 7)
    assert result.video_offsets == (0, 4, 7)
    assert result.coordinates == payload.coordinates
    assert result.source_groups == (
        ((0,), (1,), (2,), (3,)),
        ((0,), (1,), (2,)),
    )
    assert result.source_anchors == ((0, 1, 2, 3), (0, 1, 2))
    assert result.global_source_anchors == ((0, 1, 2, 3), (4, 5, 6))
    assert all(delta.is_identity for delta in result.layout_deltas)
    assert dict(result.alignment_deltas[0].mapping) == {
        0: (0,),
        1: (1,),
        2: (2,),
        3: (3,),
    }
    assert not result.compression_applied
    assert result.metadata["exact_budgets"] == (4, 3)
    assert result.metadata["compression_applied"] is False


def test_exact_budget_result_tracks_drop_merge_anchors_and_sparse_coordinates() -> None:
    result = _compressed_result()

    assert result.source_count == 7
    assert result.result_count == 4
    assert result.video_offsets == (0, 2, 4)
    assert result.video_span("video-b") == slice(2, 4)
    assert result.source_video_span("video-b") == slice(4, 7)
    assert result.compression_applied
    assert result.metadata["exact_budgets"] == (2, 2)
    assert dict(result.layout_deltas[0].old_to_new) == {
        0: 0,
        1: 0,
        2: None,
        3: 1,
    }
    assert dict(result.layout_deltas[1].old_to_new) == {
        0: 0,
        1: 1,
        2: 1,
    }
    assert dict(result.alignment_deltas[0].mapping) == {
        0: (0,),
        1: (0,),
        2: (),
        3: (1,),
    }
    assert result.coordinates == (
        VideoTokenCoordinate(1, 0, 0),
        VideoTokenCoordinate(3, 0, 0),
        VideoTokenCoordinate(0, 0, 0),
        VideoTokenCoordinate(2, 0, 0),
    )
    assert result.layout_deltas[0].result_layout["coordinate_layout"] == (
        "explicit_sparse"
    )
    assert result.layout_deltas[0].result_layout["source_groups"] == (
        (0, 1),
        (3,),
    )
    assert result.layout_deltas[0].result_layout["coordinates"] == (
        (1, 0, 0, 1, None, None),
        (3, 0, 0, 3, None, None),
    )
    json.dumps(layout_delta_record(result.layout_deltas[0]))


def test_source_groups_cannot_overlap_reorder_or_cross_video_boundaries() -> None:
    payload = _payload()

    with pytest.raises(ValueError, match="crosses or exceeds"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 4)),
            source_groups=(((0, 4), (1,)), ((0,), (1,))),
            representative_indices=((0, 1), (0, 1)),
        )
    with pytest.raises(ValueError, match="disjoint"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 4)),
            source_groups=(((0, 1), (1, 2)), ((0,), (1,))),
            representative_indices=((0, 1), (0, 1)),
        )
    with pytest.raises(ValueError, match="preserve source order"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 4)),
            source_groups=(((2,), (0,)), ((0,), (1,))),
            representative_indices=((2, 0), (0, 1)),
        )


def test_source_anchor_and_exact_budget_are_strict_assertions() -> None:
    payload = _payload()

    with pytest.raises(ValueError, match="belong to its group"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 4)),
            source_groups=(((0, 1), (3,)), ((0,), (1, 2))),
            representative_indices=((2, 3), (0, 2)),
        )
    with pytest.raises(ValueError, match="exact budget does not match"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 4)),
            source_groups=(((0, 1), (3,)), ((0,), (1, 2))),
            representative_indices=((1, 3), (0, 2)),
            exact_budgets=(3, 2),
        )
    with pytest.raises(ValueError, match="token count does not match"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((5, 4)),
            source_groups=(((0, 1), (3,)), ((0,), (1, 2))),
            representative_indices=((1, 3), (0, 2)),
        )
    with pytest.raises(ValueError, match="hidden shape"):
        VideoCompressionResult.from_source_groups(
            payload,
            _TensorLike((4, 5)),
            source_groups=(((0, 1), (3,)), ((0,), (1, 2))),
            representative_indices=((1, 3), (0, 2)),
        )


def test_result_rejects_tampered_alignment_and_anchor_coordinates() -> None:
    result = _compressed_result()
    wrong_alignment = AlignmentDelta(
        source_stream="video:video-a:source",
        target_stream="video:video-a:compressed",
        source_count=4,
        target_count=2,
        mapping={0: (0,), 1: (0,), 2: (1,), 3: (1,)},
        coupling_policy=CouplingPolicy.PRESERVE,
    )

    with pytest.raises(ValueError, match="exactly reproduce"):
        replace(
            result,
            alignment_deltas=(wrong_alignment, result.alignment_deltas[1]),
        )
    with pytest.raises(ValueError, match="representative anchors"):
        replace(
            result,
            coordinates=(result.coordinates[1], *result.coordinates[1:]),
        )
    wrong_stream = AlignmentDelta(
        source_stream="video:other:source",
        target_stream="video:video-a:compressed",
        source_count=4,
        target_count=2,
        mapping=result.alignment_deltas[0].mapping,
        coupling_policy=CouplingPolicy.PRESERVE,
    )
    with pytest.raises(ValueError, match="streams must match"):
        replace(
            result,
            alignment_deltas=(wrong_stream, result.alignment_deltas[1]),
        )


def test_result_metadata_is_immutable() -> None:
    result = _compressed_result()

    with pytest.raises(TypeError):
        result.metadata["algorithm"] = "changed"  # type: ignore[index]
