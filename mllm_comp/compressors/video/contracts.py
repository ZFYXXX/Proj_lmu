"""Backbone-neutral contracts for spatiotemporal video compression.

The shared kernel boundary is a flattened ``[tokens, hidden]`` sequence with
explicit per-video offsets.  Algorithms still operate on one video at a time:
every result carries one local :class:`LayoutDelta` and one exact alignment
relation per media identity, so a source group can never cross a video
boundary accidentally.

This module deliberately does not import a tensor framework.  Backbones own
tensor construction and legalization; the contract validates shapes,
boundaries, coordinates, and source provenance only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.core.layout import (
    AlignmentDelta,
    CouplingPolicy,
    LayoutDelta,
    LayoutTokenUnit,
    OrderingPolicy,
)


@dataclass(frozen=True)
class VideoTokenCoordinate:
    """Immutable source coordinate for one flattened video token.

    ``t/h/w`` are the original temporal, row, and column coordinates.  A
    provider may identify ``t`` as a concrete frame, a tubelet, or both.  A
    plain ``(t, h, w)`` coordinate defaults to frame identity ``t``.
    """

    t: int
    h: int
    w: int
    frame_index: int | None = None
    tubelet_index: int | None = None
    timestamp_seconds: float | None = None

    def __post_init__(self) -> None:
        for name in ("t", "h", "w"):
            value = getattr(self, name)
            _require_non_negative_integer(name, value)
        for name in ("frame_index", "tubelet_index"):
            value = getattr(self, name)
            if value is not None:
                _require_non_negative_integer(name, value)
        if self.frame_index is None and self.tubelet_index is None:
            object.__setattr__(self, "frame_index", self.t)
        if self.timestamp_seconds is not None:
            value = self.timestamp_seconds
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("timestamp_seconds must be a finite number")
            if not isfinite(float(value)) or float(value) < 0:
                raise ValueError("timestamp_seconds must be finite and non-negative")
            object.__setattr__(self, "timestamp_seconds", float(value))

    def as_tuple(
        self,
    ) -> tuple[int, int, int, int | None, int | None, float | None]:
        """Return an immutable, JSON-serializable evidence representation."""

        return (
            self.t,
            self.h,
            self.w,
            self.frame_index,
            self.tubelet_index,
            self.timestamp_seconds,
        )


@dataclass(frozen=True)
class VideoTokenPayload:
    """Algorithm-facing features and provenance for independent videos.

    ``features`` is a flattened two-dimensional tensor-like object.  The
    half-open spans in ``video_offsets`` partition it exactly in ``media_ids``
    order.  ``raw_features`` may use a different tokenization, so it owns a
    separate set of offsets and coordinates while retaining the same media
    identities.

    Optional backbone/provider signals (encoder attention, hidden states,
    prompt context, detection output, or decoder/cache context) live in the
    immutable ``signals`` mapping.  Their semantic requirements are declared
    by the exact algorithm binding rather than guessed here.
    """

    features: Any
    media_ids: tuple[str, ...]
    video_offsets: tuple[int, ...]
    coordinates: tuple[VideoTokenCoordinate | Sequence[int], ...]
    raw_features: Any | None = None
    raw_video_offsets: tuple[int, ...] = ()
    raw_coordinates: tuple[VideoTokenCoordinate | Sequence[int], ...] = ()
    token_mask: Any | None = None
    signals: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        feature_shape = _feature_shape(self.features, "features")
        media_ids = _normalise_media_ids(self.media_ids)
        offsets = _normalise_offsets(
            "video_offsets",
            self.video_offsets,
            video_count=len(media_ids),
            token_count=feature_shape[0],
        )
        coordinates = _normalise_coordinates(
            "coordinates", self.coordinates, feature_shape[0]
        )

        if self.raw_features is None:
            if self.raw_video_offsets:
                raise ValueError(
                    "raw_video_offsets requires raw_features"
                )
            if self.raw_coordinates:
                raise ValueError("raw_coordinates requires raw_features")
            raw_offsets: tuple[int, ...] = ()
            raw_coordinates: tuple[VideoTokenCoordinate, ...] = ()
        else:
            raw_shape = _feature_shape(self.raw_features, "raw_features")
            raw_offsets = _normalise_offsets(
                "raw_video_offsets",
                self.raw_video_offsets,
                video_count=len(media_ids),
                token_count=raw_shape[0],
            )
            raw_coordinates = _normalise_coordinates(
                "raw_coordinates", self.raw_coordinates, raw_shape[0]
            )

        if self.token_mask is not None:
            mask_count = _flat_sequence_length(self.token_mask, "token_mask")
            if mask_count != feature_shape[0]:
                raise ValueError(
                    "token_mask must contain one entry per projected token"
                )

        object.__setattr__(self, "media_ids", media_ids)
        object.__setattr__(self, "video_offsets", offsets)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "raw_video_offsets", raw_offsets)
        object.__setattr__(self, "raw_coordinates", raw_coordinates)
        object.__setattr__(self, "signals", _freeze_named_mapping("signals", self.signals))
        object.__setattr__(self, "metadata", _freeze_mapping("metadata", self.metadata))

    @property
    def video_count(self) -> int:
        return len(self.media_ids)

    @property
    def token_count(self) -> int:
        return self.video_offsets[-1]

    @property
    def raw_token_count(self) -> int | None:
        return None if self.raw_features is None else self.raw_video_offsets[-1]

    def video_index(self, video: int | str) -> int:
        """Resolve one numeric index or immutable media identity."""

        if isinstance(video, bool):
            raise TypeError("video must be an integer index or media identity")
        if isinstance(video, int):
            if not 0 <= video < self.video_count:
                raise IndexError("video index is outside the payload")
            return video
        if isinstance(video, str):
            try:
                return self.media_ids.index(video)
            except ValueError as exc:
                raise KeyError(f"unknown media identity: {video!r}") from exc
        raise TypeError("video must be an integer index or media identity")

    def video_span(self, video: int | str) -> slice:
        index = self.video_index(video)
        return slice(self.video_offsets[index], self.video_offsets[index + 1])

    def raw_video_span(self, video: int | str) -> slice:
        if self.raw_features is None:
            raise ValueError("payload does not contain raw_features")
        index = self.video_index(video)
        return slice(
            self.raw_video_offsets[index], self.raw_video_offsets[index + 1]
        )


@dataclass(frozen=True)
class VideoCompressionResult:
    """Validated compressed features and exact per-video source provenance.

    Source indices in ``source_groups`` and ``representative_indices`` are
    local to their video.  Each output token has exactly one non-empty source
    group and one representative anchor contained in that group.  Its output
    coordinate is the original coordinate of that anchor; irregular results
    therefore remain explicit and are never advertised as a fabricated dense
    grid.
    """

    features: Any
    media_ids: tuple[str, ...]
    source_video_offsets: tuple[int, ...]
    video_offsets: tuple[int, ...]
    coordinates: tuple[VideoTokenCoordinate | Sequence[int], ...]
    layout_deltas: tuple[LayoutDelta, ...]
    alignment_deltas: tuple[AlignmentDelta, ...]
    source_groups: tuple[tuple[tuple[int, ...], ...], ...]
    representative_indices: tuple[tuple[int, ...], ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        feature_shape = _feature_shape(self.features, "features")
        media_ids = _normalise_media_ids(self.media_ids)
        layouts = tuple(self.layout_deltas)
        alignments = tuple(self.alignment_deltas)
        groups_by_video = tuple(tuple(groups) for groups in self.source_groups)
        representatives_by_video = tuple(
            tuple(indices) for indices in self.representative_indices
        )
        video_count = len(media_ids)
        for name, values in (
            ("layout_deltas", layouts),
            ("alignment_deltas", alignments),
            ("source_groups", groups_by_video),
            ("representative_indices", representatives_by_video),
        ):
            if len(values) != video_count:
                raise ValueError(f"{name} must contain one entry per video")
        if not all(isinstance(delta, LayoutDelta) for delta in layouts):
            raise TypeError("layout_deltas entries must be LayoutDelta instances")
        if not all(isinstance(delta, AlignmentDelta) for delta in alignments):
            raise TypeError(
                "alignment_deltas entries must be AlignmentDelta instances"
            )

        source_count = sum(delta.source_count for delta in layouts)
        source_offsets = _normalise_offsets(
            "source_video_offsets",
            self.source_video_offsets,
            video_count=video_count,
            token_count=source_count,
        )
        result_offsets = _normalise_offsets(
            "video_offsets",
            self.video_offsets,
            video_count=video_count,
            token_count=feature_shape[0],
        )
        coordinates = _normalise_coordinates(
            "coordinates", self.coordinates, feature_shape[0]
        )

        normalised_groups: list[tuple[tuple[int, ...], ...]] = []
        normalised_representatives: list[tuple[int, ...]] = []
        for video_index, media_id in enumerate(media_ids):
            layout = layouts[video_index]
            alignment = alignments[video_index]
            expected_source_count = (
                source_offsets[video_index + 1] - source_offsets[video_index]
            )
            expected_result_count = (
                result_offsets[video_index + 1] - result_offsets[video_index]
            )
            if layout.source_count != expected_source_count:
                raise ValueError(
                    "LayoutDelta source_count does not match its video boundary"
                )
            if layout.result_count != expected_result_count:
                raise ValueError(
                    "LayoutDelta result_count does not match its video boundary"
                )
            if layout.media_unit != LayoutTokenUnit.VIDEO_PATCH.value:
                raise ValueError("video LayoutDelta media_unit must be video_patch")
            if layout.ordering_policy is not OrderingPolicy.PRESERVE_SOURCE:
                raise ValueError("video outputs must preserve source-group order")
            source_coordinates = _normalise_coordinates(
                "LayoutDelta.coordinates",
                layout.coordinates,
                layout.source_count,
            )

            groups = _normalise_source_groups(
                groups_by_video[video_index],
                source_count=layout.source_count,
            )
            expected_groups = _layout_source_groups(layout)
            if groups != expected_groups:
                raise ValueError(
                    "source_groups must exactly match the LayoutDelta preimages"
                )
            representatives = _normalise_representatives(
                representatives_by_video[video_index], groups
            )

            if alignment.source_count != layout.source_count:
                raise ValueError(
                    "AlignmentDelta source_count does not match LayoutDelta"
                )
            if alignment.target_count != layout.result_count:
                raise ValueError(
                    "AlignmentDelta target_count does not match LayoutDelta"
                )
            if alignment.coupling_policy is not CouplingPolicy.PRESERVE:
                raise ValueError("video source alignment must use preserve coupling")
            expected_alignment = {
                source_index: (
                    ()
                    if layout.old_to_new[source_index] is None
                    else (int(layout.old_to_new[source_index]),)
                )
                for source_index in range(layout.source_count)
            }
            if dict(alignment.mapping) != expected_alignment:
                raise ValueError(
                    "AlignmentDelta must exactly reproduce LayoutDelta.old_to_new"
                )

            result_start = result_offsets[video_index]
            result_coordinates = coordinates[
                result_start : result_offsets[video_index + 1]
            ]
            anchored_coordinates = tuple(
                source_coordinates[index] for index in representatives
            )
            if result_coordinates != anchored_coordinates:
                raise ValueError(
                    "output coordinates must be gathered from representative anchors"
                )

            declared_media = layout.source_layout.get("media_id")
            if declared_media is not None and declared_media != media_id:
                raise ValueError("LayoutDelta source media_id does not match result")
            result_media = layout.result_layout.get("media_id")
            if result_media is not None and result_media != media_id:
                raise ValueError("LayoutDelta result media_id does not match result")
            source_stream = layout.source_layout.get("stream_id")
            result_stream = layout.result_layout.get("stream_id")
            if not isinstance(source_stream, str) or not source_stream:
                raise ValueError("video LayoutDelta must declare a source stream_id")
            if not isinstance(result_stream, str) or not result_stream:
                raise ValueError("video LayoutDelta must declare a result stream_id")
            if (
                alignment.source_stream != source_stream
                or alignment.target_stream != result_stream
            ):
                raise ValueError(
                    "AlignmentDelta streams must match the video LayoutDelta"
                )
            source_boundary = layout.source_layout.get("global_offset")
            expected_source_boundary = (
                source_offsets[video_index],
                source_offsets[video_index + 1],
            )
            if (
                source_boundary is not None
                and tuple(source_boundary) != expected_source_boundary
            ):
                raise ValueError(
                    "LayoutDelta source global_offset does not match result boundary"
                )
            result_boundary = layout.result_layout.get("global_offset")
            expected_result_boundary = (
                result_offsets[video_index],
                result_offsets[video_index + 1],
            )
            if (
                result_boundary is not None
                and tuple(result_boundary) != expected_result_boundary
            ):
                raise ValueError(
                    "LayoutDelta result global_offset does not match result boundary"
                )
            declared_groups = layout.result_layout.get("source_groups")
            if declared_groups is not None and tuple(
                tuple(group) for group in declared_groups
            ) != groups:
                raise ValueError(
                    "LayoutDelta result_layout source_groups do not match result"
                )
            declared_representatives = layout.result_layout.get(
                "representative_indices"
            )
            if declared_representatives is not None and tuple(
                declared_representatives
            ) != representatives:
                raise ValueError(
                    "LayoutDelta representative_indices do not match result"
                )
            declared_coordinates = layout.result_layout.get("coordinates")
            coordinate_records = tuple(
                coordinate.as_tuple() for coordinate in result_coordinates
            )
            if (
                declared_coordinates is not None
                and tuple(tuple(value) for value in declared_coordinates)
                != coordinate_records
            ):
                raise ValueError(
                    "LayoutDelta result coordinates do not match result anchors"
                )

            normalised_groups.append(groups)
            normalised_representatives.append(representatives)

        object.__setattr__(self, "media_ids", media_ids)
        object.__setattr__(self, "source_video_offsets", source_offsets)
        object.__setattr__(self, "video_offsets", result_offsets)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "layout_deltas", layouts)
        object.__setattr__(self, "alignment_deltas", alignments)
        object.__setattr__(self, "source_groups", tuple(normalised_groups))
        object.__setattr__(
            self, "representative_indices", tuple(normalised_representatives)
        )
        object.__setattr__(self, "metadata", _freeze_mapping("metadata", self.metadata))

    @classmethod
    def identity(
        cls,
        payload: VideoTokenPayload,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VideoCompressionResult":
        """Return the lossless, allocation-free identity result."""

        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        groups = []
        representatives = []
        budgets = []
        for video_index in range(payload.video_count):
            span = payload.video_span(video_index)
            count = span.stop - span.start
            groups.append(tuple((index,) for index in range(count)))
            representatives.append(tuple(range(count)))
            budgets.append(count)
        return cls.from_source_groups(
            payload,
            payload.features,
            source_groups=groups,
            representative_indices=representatives,
            exact_budgets=budgets,
            metadata=metadata,
        )

    @classmethod
    def from_source_groups(
        cls,
        payload: VideoTokenPayload,
        compressed_features: Any,
        *,
        source_groups: Iterable[Iterable[Iterable[int]]],
        representative_indices: Iterable[Iterable[int]],
        exact_budgets: Iterable[int] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> "VideoCompressionResult":
        """Assemble a result from explicit local groups and source anchors.

        ``exact_budgets`` is an assertion, not a best-effort request.  Every
        entry must equal the number of output groups for that video.  Callers
        must cap or reject an infeasible budget before constructing the result.
        """

        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        feature_shape = _feature_shape(compressed_features, "compressed_features")
        source_feature_shape = _feature_shape(payload.features, "payload.features")
        if feature_shape[1:] != source_feature_shape[1:]:
            raise ValueError(
                "compressed_features must preserve the projected hidden shape"
            )

        raw_groups = tuple(tuple(tuple(group) for group in groups) for groups in source_groups)
        raw_representatives = tuple(
            tuple(indices) for indices in representative_indices
        )
        if len(raw_groups) != payload.video_count:
            raise ValueError("source_groups must contain one entry per video")
        if len(raw_representatives) != payload.video_count:
            raise ValueError(
                "representative_indices must contain one entry per video"
            )

        groups_by_video: list[tuple[tuple[int, ...], ...]] = []
        representatives_by_video: list[tuple[int, ...]] = []
        for video_index in range(payload.video_count):
            span = payload.video_span(video_index)
            groups = _normalise_source_groups(
                raw_groups[video_index], source_count=span.stop - span.start
            )
            representatives = _normalise_representatives(
                raw_representatives[video_index], groups
            )
            groups_by_video.append(groups)
            representatives_by_video.append(representatives)

        if exact_budgets is None:
            budgets = tuple(len(groups) for groups in groups_by_video)
        else:
            budgets = tuple(exact_budgets)
            if len(budgets) != payload.video_count:
                raise ValueError("exact_budgets must contain one entry per video")
            for video_index, budget in enumerate(budgets):
                _require_positive_integer("exact_budgets", budget)
                source_count = (
                    payload.video_offsets[video_index + 1]
                    - payload.video_offsets[video_index]
                )
                if budget > source_count:
                    raise ValueError("exact budget cannot exceed source token count")
                if budget != len(groups_by_video[video_index]):
                    raise ValueError(
                        "exact budget does not match the number of source groups"
                    )

        result_offsets = [0]
        for groups in groups_by_video:
            result_offsets.append(result_offsets[-1] + len(groups))
        if result_offsets[-1] != feature_shape[0]:
            raise ValueError(
                "compressed_features token count does not match source groups"
            )

        layouts: list[LayoutDelta] = []
        alignments: list[AlignmentDelta] = []
        result_coordinates: list[VideoTokenCoordinate] = []
        for video_index, media_id in enumerate(payload.media_ids):
            source_span = payload.video_span(video_index)
            source_count = source_span.stop - source_span.start
            source_coordinates = payload.coordinates[source_span]
            groups = groups_by_video[video_index]
            representatives = representatives_by_video[video_index]
            mapping: dict[int, int | None] = {
                index: None for index in range(source_count)
            }
            for result_index, group in enumerate(groups):
                for source_index in group:
                    mapping[source_index] = result_index

            result_span = (result_offsets[video_index], result_offsets[video_index + 1])
            video_coordinates = tuple(
                source_coordinates[index] for index in representatives
            )
            result_coordinates.extend(video_coordinates)
            source_stream = f"video:{media_id}:source"
            result_stream = f"video:{media_id}:compressed"
            layout = LayoutDelta.from_mapping(
                source_count=source_count,
                old_to_new=mapping,
                result_count=len(groups),
                media_unit=LayoutTokenUnit.VIDEO_PATCH,
                coordinates=source_coordinates,
                source_layout={
                    "layout_id": source_stream,
                    "stream_id": source_stream,
                    "media_id": media_id,
                    "video_index": video_index,
                    "global_offset": (source_span.start, source_span.stop),
                    "media_unit": LayoutTokenUnit.VIDEO_PATCH.value,
                },
                result_layout={
                    "layout_id": result_stream,
                    "stream_id": result_stream,
                    "media_id": media_id,
                    "video_index": video_index,
                    "global_offset": result_span,
                    "media_unit": LayoutTokenUnit.VIDEO_PATCH.value,
                    "coordinate_layout": "explicit_sparse",
                    "coordinate_fields": (
                        "t",
                        "h",
                        "w",
                        "frame_index",
                        "tubelet_index",
                        "timestamp_seconds",
                    ),
                    "coordinates": tuple(
                        coordinate.as_tuple() for coordinate in video_coordinates
                    ),
                    "source_groups": groups,
                    "representative_indices": representatives,
                    "representative_policy": "explicit_source_anchor",
                },
            )
            alignment = AlignmentDelta(
                source_stream=source_stream,
                target_stream=result_stream,
                source_count=source_count,
                target_count=len(groups),
                mapping={
                    index: (
                        () if target is None else (int(target),)
                    )
                    for index, target in mapping.items()
                },
                coupling_policy=CouplingPolicy.PRESERVE,
            )
            layouts.append(layout)
            alignments.append(alignment)

        resolved_metadata = dict(metadata or {})
        resolved_metadata.update(
            {
                "sequence_contract": "independent_video_sequences",
                "exact_budgets": budgets,
                "source_count": payload.token_count,
                "result_count": feature_shape[0],
                "compression_applied": any(
                    not delta.is_identity for delta in layouts
                ),
            }
        )
        return cls(
            features=compressed_features,
            media_ids=payload.media_ids,
            source_video_offsets=payload.video_offsets,
            video_offsets=tuple(result_offsets),
            coordinates=tuple(result_coordinates),
            layout_deltas=tuple(layouts),
            alignment_deltas=tuple(alignments),
            source_groups=tuple(groups_by_video),
            representative_indices=tuple(representatives_by_video),
            metadata=resolved_metadata,
        )

    @property
    def video_count(self) -> int:
        return len(self.media_ids)

    @property
    def source_count(self) -> int:
        return self.source_video_offsets[-1]

    @property
    def result_count(self) -> int:
        return self.video_offsets[-1]

    @property
    def compression_applied(self) -> bool:
        return any(not delta.is_identity for delta in self.layout_deltas)

    @property
    def source_anchors(self) -> tuple[tuple[int, ...], ...]:
        """Per-video local source anchors, one for every output token."""

        return self.representative_indices

    @property
    def global_source_anchors(self) -> tuple[tuple[int, ...], ...]:
        return tuple(
            tuple(self.source_video_offsets[index] + value for value in anchors)
            for index, anchors in enumerate(self.representative_indices)
        )

    def video_span(self, video: int | str) -> slice:
        index = _resolve_video_index(self.media_ids, video)
        return slice(self.video_offsets[index], self.video_offsets[index + 1])

    def source_video_span(self, video: int | str) -> slice:
        index = _resolve_video_index(self.media_ids, video)
        return slice(
            self.source_video_offsets[index], self.source_video_offsets[index + 1]
        )


def _feature_shape(value: Any, name: str) -> tuple[int, ...]:
    shape = getattr(value, "shape", None)
    if shape is None:
        raise TypeError(f"{name} must be a tensor-like object with shape")
    try:
        result = tuple(int(size) for size in shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name}.shape must contain integer dimensions") from exc
    if len(result) != 2:
        raise ValueError(f"{name} must have shape [tokens, hidden]")
    if result[0] < 1 or result[1] < 1:
        raise ValueError(f"{name} dimensions must be positive")
    return result


def _flat_sequence_length(value: Any, name: str) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            dimensions = tuple(int(size) for size in shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name}.shape must contain integer dimensions") from exc
        if len(dimensions) != 1:
            raise ValueError(f"{name} must be a flat sequence")
        return dimensions[0]
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return len(value)
    raise TypeError(f"{name} must be a flat tensor-like object or sequence")


def _normalise_media_ids(values: Iterable[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("media_ids must be an iterable of strings")
    result = tuple(values)
    if not result:
        raise ValueError("media_ids must contain at least one video")
    if any(not isinstance(value, str) or not value.strip() for value in result):
        raise ValueError("media_ids entries must be non-empty strings")
    result = tuple(value.strip() for value in result)
    if len(set(result)) != len(result):
        raise ValueError("media_ids must be unique within a payload")
    return result


def _normalise_offsets(
    name: str,
    values: Iterable[int],
    *,
    video_count: int,
    token_count: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    result = tuple(values)
    if len(result) != video_count + 1:
        raise ValueError(f"{name} must contain video_count + 1 entries")
    for value in result:
        _require_non_negative_integer(name, value)
    if result[0] != 0 or result[-1] != token_count:
        raise ValueError(f"{name} must start at zero and end at token count")
    if any(left >= right for left, right in zip(result, result[1:])):
        raise ValueError(f"{name} must define non-empty increasing video spans")
    return result


def _normalise_coordinates(
    name: str,
    values: Iterable[VideoTokenCoordinate | Sequence[int]],
    count: int,
) -> tuple[VideoTokenCoordinate, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of video coordinates")
    raw = tuple(values)
    if len(raw) != count:
        raise ValueError(f"{name} must contain one entry per token")
    result = []
    for value in raw:
        if isinstance(value, VideoTokenCoordinate):
            result.append(value)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            coordinate = tuple(value)
            if len(coordinate) == 3:
                result.append(VideoTokenCoordinate(*coordinate))
                continue
        raise TypeError(
            f"{name} entries must be VideoTokenCoordinate or (t, h, w)"
        )
    return tuple(result)


def _normalise_source_groups(
    values: Iterable[Iterable[int]],
    *,
    source_count: int,
) -> tuple[tuple[int, ...], ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("source_groups must be an iterable of index groups")
    groups = []
    represented: set[int] = set()
    for raw_group in values:
        if isinstance(raw_group, (str, bytes)):
            raise TypeError("source group must be an iterable of integers")
        group = tuple(raw_group)
        if not group:
            raise ValueError("source groups must be non-empty")
        for index in group:
            _require_non_negative_integer("source_groups", index)
            if index >= source_count:
                raise ValueError(
                    "source group index crosses or exceeds its video boundary"
                )
        if tuple(sorted(group)) != group or len(set(group)) != len(group):
            raise ValueError(
                "each source group must contain unique indices in source order"
            )
        if represented.intersection(group):
            raise ValueError("source groups must be disjoint within a video")
        represented.update(group)
        groups.append(group)
    result = tuple(groups)
    if not result:
        raise ValueError("each video must produce at least one output token")
    first_indices = tuple(group[0] for group in result)
    if tuple(sorted(first_indices)) != first_indices:
        raise ValueError("source groups must preserve source order")
    return result


def _normalise_representatives(
    values: Iterable[int],
    groups: tuple[tuple[int, ...], ...],
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("representative_indices must be an iterable of integers")
    result = tuple(values)
    if len(result) != len(groups):
        raise ValueError(
            "representative_indices must contain one anchor per source group"
        )
    for index, group in zip(result, groups):
        _require_non_negative_integer("representative_indices", index)
        if index not in group:
            raise ValueError("each representative anchor must belong to its group")
    return result


def _layout_source_groups(delta: LayoutDelta) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(
            source_index
            for source_index in range(delta.source_count)
            if delta.old_to_new[source_index] == result_index
        )
        for result_index in range(delta.result_count)
    )


def _freeze_named_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    result = dict(_freeze_mapping(name, value))
    if any(not isinstance(key, str) or not key.strip() for key in result):
        raise ValueError(f"{name} keys must be non-empty strings")
    return MappingProxyType(result)


def _freeze_mapping(name: str, value: Mapping[str, Any]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _resolve_video_index(media_ids: tuple[str, ...], video: int | str) -> int:
    if isinstance(video, bool):
        raise TypeError("video must be an integer index or media identity")
    if isinstance(video, int):
        if not 0 <= video < len(media_ids):
            raise IndexError("video index is outside the result")
        return video
    if isinstance(video, str):
        try:
            return media_ids.index(video)
        except ValueError as exc:
            raise KeyError(f"unknown media identity: {video!r}") from exc
    raise TypeError("video must be an integer index or media identity")


def _require_non_negative_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} entries must be integers")
    if value < 0:
        raise ValueError(f"{name} entries must be non-negative")


def _require_positive_integer(name: str, value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} entries must be integers")
    if value < 1:
        raise ValueError(f"{name} entries must be positive")


__all__ = [
    "VideoCompressionResult",
    "VideoTokenCoordinate",
    "VideoTokenPayload",
]
