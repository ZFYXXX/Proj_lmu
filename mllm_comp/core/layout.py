"""Pure-Python contracts for token-layout and cross-stream alignment changes.

The contracts in this module deliberately describe *what* changed without
depending on a tensor library or on a specific backbone.  A ``LayoutDelta``
always belongs to one media-unit stream; callers represent multiple streams
with separate deltas and may only compose deltas from the same stream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from .vocabulary import AlignmentCouplingPolicy, Modality


class LayoutTokenUnit(str, Enum):
    """Common physical units represented by one layout entry.

    ``media_unit`` remains an open string in the contracts so that a backbone
    may introduce a more precise unit without changing this enum.
    """

    TOKEN = "token"
    IMAGE_PATCH = "image_patch"
    VIDEO_PATCH = "video_patch"
    VIDEO_FRAME = "video_frame"
    AUDIO_FRAME = "audio_frame"
    AUDIO_CHUNK = "audio_chunk"
    SCENE_POINT = "scene_point"


class OrderingPolicy(str, Enum):
    """Ordering guarantee made by a layout transformation."""

    PRESERVE_SOURCE = "preserve_source"
    EXPLICIT = "explicit"
    ARBITRARY = "arbitrary"


# Backwards-compatible module aliases for the first public v2 draft.  Prefer
# the explicit names in new code and documentation.
TokenUnit = LayoutTokenUnit
CouplingPolicy = AlignmentCouplingPolicy


def layout_delta_record(
    delta: "LayoutDelta",
    *,
    include_mapping: bool = True,
) -> dict[str, Any]:
    """Return a stable serializable audit record for one layout change.

    Runtime evidence keeps the exact ``old_to_new`` mapping, while the compact
    manifest projection may set ``include_mapping=False`` and retain the hash.
    This makes multi-stage layout changes reproducible without forcing every
    top-level manifest to embed a potentially large token map.
    """

    if not isinstance(delta, LayoutDelta):
        raise TypeError("delta must be a LayoutDelta")
    mapping = tuple(delta.old_to_new[index] for index in range(delta.source_count))
    import hashlib

    encoded_mapping = ",".join(
        "-" if target is None else str(target) for target in mapping
    ).encode("ascii")
    result: dict[str, Any] = {
        "source_count": delta.source_count,
        "result_count": delta.result_count,
        "compression_ratio": (
            1.0
            if delta.source_count == 0
            else delta.result_count / delta.source_count
        ),
        "operation": delta.operation,
        "media_unit": delta.media_unit,
        "ordering_policy": delta.ordering_policy.value,
        "dropped_count": len(delta.dropped_indices),
        "merge_group_count": len(delta.merge_groups),
        "merged_source_count": sum(len(group) for group in delta.merge_groups),
        "protected_count": len(delta.protected_indices),
        "mapping_sha256": hashlib.sha256(encoded_mapping).hexdigest(),
        "source_layout": dict(delta.source_layout),
        "result_layout": dict(delta.result_layout),
    }
    if include_mapping:
        result.update(
            {
                "old_to_new": list(mapping),
                "dropped_indices": list(delta.dropped_indices),
                "merge_groups": [list(group) for group in delta.merge_groups],
                "protected_indices": list(delta.protected_indices),
            }
        )
    return result


def layout_delta_summary(delta: "LayoutDelta") -> dict[str, Any]:
    """Return the compact, hash-bearing manifest view of ``delta``."""

    return layout_delta_record(delta, include_mapping=False)


def _require_count(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
    return value


def _normalise_index(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} entries must be integers")
    return value


def _normalise_index_tuple(name: str, values: Iterable[int]) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    result = tuple(_normalise_index(name, value) for value in values)
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must be in ascending source order")
    return result


def _normalise_media_unit(value: str | LayoutTokenUnit) -> str:
    if isinstance(value, LayoutTokenUnit):
        return value.value
    if not isinstance(value, str):
        raise TypeError("media_unit must be a string or TokenUnit")
    value = value.strip()
    if not value:
        raise ValueError("media_unit must not be empty")
    return value


def _normalise_layout_metadata(
    name: str,
    value: Mapping[str, Any],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return MappingProxyType(dict(value))


def _normalise_old_to_new(
    source_count: int,
    value: Mapping[int, int | None] | Sequence[int | None],
) -> Mapping[int, int | None]:
    if isinstance(value, Mapping):
        raw = dict(value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        raw = dict(enumerate(value))
    else:
        raise TypeError("old_to_new must be a mapping or an index sequence")

    expected_keys = set(range(source_count))
    if set(raw) != expected_keys:
        missing = sorted(expected_keys - set(raw))
        extra = sorted(set(raw) - expected_keys, key=repr)
        raise ValueError(
            "old_to_new must contain every source index exactly once "
            f"(missing={missing}, extra={extra})"
        )

    normalised: dict[int, int | None] = {}
    for old_index in range(source_count):
        new_index = raw[old_index]
        if new_index is not None:
            new_index = _normalise_index("old_to_new", new_index)
        normalised[old_index] = new_index
    return MappingProxyType(normalised)


def _preimages(
    old_to_new: Mapping[int, int | None],
    result_count: int,
) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = [[] for _ in range(result_count)]
    for old_index, new_index in old_to_new.items():
        if new_index is not None:
            groups[new_index].append(old_index)
    return tuple(tuple(group) for group in groups)


@dataclass(frozen=True)
class ModalityLayout:
    """Current physical layout of one named modality stream."""

    stream_id: str
    modality: Modality | str
    count: int
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN
    coordinates: tuple[Any, ...] = ()
    protected_indices: tuple[int, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.stream_id, str) or not self.stream_id.strip():
            raise ValueError("stream_id must be a non-empty string")
        count = _require_count("count", self.count)
        media_unit = _normalise_media_unit(self.media_unit)
        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != count:
            raise ValueError(
                "coordinates must be empty or contain one entry per layout item"
            )
        protected = _normalise_index_tuple(
            "protected_indices", self.protected_indices
        )
        if any(index < 0 or index >= count for index in protected):
            raise ValueError("protected_indices contains an index outside the layout")
        object.__setattr__(self, "stream_id", self.stream_id.strip())
        object.__setattr__(self, "modality", Modality(self.modality))
        object.__setattr__(self, "count", count)
        object.__setattr__(self, "media_unit", media_unit)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "protected_indices", protected)
        object.__setattr__(
            self,
            "metadata",
            _normalise_layout_metadata("metadata", self.metadata),
        )

    def apply(self, delta: "LayoutDelta") -> "ModalityLayout":
        """Advance the current layout after a validated same-stream delta."""

        if delta.source_count != self.count:
            raise ValueError(
                f"layout count mismatch: {self.count} != {delta.source_count}"
            )
        if delta.media_unit != self.media_unit:
            raise ValueError(
                "layout media_unit mismatch: "
                f"{self.media_unit!r} != {delta.media_unit!r}"
            )
        result_coordinates = delta.result_layout.get("coordinates", ())
        if not result_coordinates and self.coordinates:
            preimages = _preimages(delta.old_to_new, delta.result_count)
            result_coordinates = tuple(
                self.coordinates[group[0]] for group in preimages
            )
        result_protected = tuple(
            sorted(
                {
                    delta.old_to_new[index]
                    for index in self.protected_indices
                    if delta.old_to_new[index] is not None
                }
            )
        )
        metadata = dict(self.metadata)
        metadata.update(dict(delta.result_layout))
        return ModalityLayout(
            stream_id=self.stream_id,
            modality=self.modality,
            count=delta.result_count,
            media_unit=self.media_unit,
            coordinates=tuple(result_coordinates),
            protected_indices=result_protected,
            metadata=metadata,
        )


@dataclass(frozen=True)
class LayoutDelta:
    """A validated change from one layout to another for one media unit.

    ``old_to_new`` is the canonical representation.  A ``None`` result means
    that the source entry was dropped.  Multiple source indices mapped to the
    same result index form one merge group.  Consequently, ``kept_indices``
    describes every source entry whose information remains represented, not
    only the representative token chosen by a concrete implementation.

    ``coordinates`` are source-layout coordinates and, when provided, have one
    entry per source item.  Result coordinates belong in ``result_layout`` so
    that coordinate legalization can be backbone-specific.
    """

    source_count: int
    result_count: int
    kept_indices: tuple[int, ...] = ()
    dropped_indices: tuple[int, ...] = ()
    merge_groups: tuple[tuple[int, ...], ...] = ()
    old_to_new: Mapping[int, int | None] | Sequence[int | None] = field(
        default_factory=dict
    )
    media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN
    coordinates: tuple[Any, ...] = ()
    protected_indices: tuple[int, ...] = ()
    source_layout: Mapping[str, Any] = field(default_factory=dict)
    result_layout: Mapping[str, Any] = field(default_factory=dict)
    ordering_policy: OrderingPolicy | str = OrderingPolicy.PRESERVE_SOURCE

    def __post_init__(self) -> None:
        source_count = _require_count("source_count", self.source_count)
        result_count = _require_count("result_count", self.result_count)
        kept_indices = _normalise_index_tuple("kept_indices", self.kept_indices)
        dropped_indices = _normalise_index_tuple(
            "dropped_indices", self.dropped_indices
        )
        protected_indices = _normalise_index_tuple(
            "protected_indices", self.protected_indices
        )
        media_unit = _normalise_media_unit(self.media_unit)

        try:
            ordering_policy = OrderingPolicy(self.ordering_policy)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(policy.value for policy in OrderingPolicy)
            raise ValueError(f"ordering_policy must be one of: {allowed}") from exc

        old_to_new = _normalise_old_to_new(source_count, self.old_to_new)
        for old_index, new_index in old_to_new.items():
            if new_index is not None and not 0 <= new_index < result_count:
                raise ValueError(
                    f"old_to_new[{old_index}]={new_index} is outside result layout"
                )

        preimages = _preimages(old_to_new, result_count)
        empty_results = [index for index, group in enumerate(preimages) if not group]
        if empty_results:
            raise ValueError(
                "every result index must have a source preimage "
                f"(empty={empty_results})"
            )

        expected_kept = tuple(
            old_index
            for old_index in range(source_count)
            if old_to_new[old_index] is not None
        )
        expected_dropped = tuple(
            old_index
            for old_index in range(source_count)
            if old_to_new[old_index] is None
        )
        if kept_indices != expected_kept:
            raise ValueError(
                "kept_indices must exactly match non-null old_to_new entries"
            )
        if dropped_indices != expected_dropped:
            raise ValueError(
                "dropped_indices must exactly match null old_to_new entries"
            )

        merge_groups = tuple(
            _normalise_index_tuple("merge_groups", group)
            for group in self.merge_groups
        )
        expected_merges = tuple(group for group in preimages if len(group) > 1)
        if merge_groups != expected_merges:
            raise ValueError(
                "merge_groups must list every non-singleton old_to_new preimage "
                "in result order"
            )

        for index in (*kept_indices, *dropped_indices, *protected_indices):
            if not 0 <= index < source_count:
                raise ValueError(f"source index {index} is outside source layout")

        for protected_index in protected_indices:
            new_index = old_to_new[protected_index]
            if new_index is None:
                raise ValueError("a protected index cannot be dropped")
            if len(preimages[new_index]) != 1:
                raise ValueError("a protected index cannot be merged")

        coordinates = tuple(self.coordinates)
        if coordinates and len(coordinates) != source_count:
            raise ValueError(
                "coordinates must be empty or contain one entry per source item"
            )

        if ordering_policy is OrderingPolicy.PRESERVE_SOURCE:
            first_source_indices = tuple(group[0] for group in preimages)
            if tuple(sorted(first_source_indices)) != first_source_indices:
                raise ValueError(
                    "preserve_source ordering requires result groups in source order"
                )

        source_layout = _normalise_layout_metadata(
            "source_layout", self.source_layout
        )
        result_layout = _normalise_layout_metadata(
            "result_layout", self.result_layout
        )
        for name, layout in (
            ("source_layout", source_layout),
            ("result_layout", result_layout),
        ):
            declared_unit = layout.get("media_unit")
            if declared_unit is not None and _normalise_media_unit(declared_unit) != media_unit:
                raise ValueError(
                    f"{name}.media_unit does not match delta media_unit={media_unit!r}"
                )

        object.__setattr__(self, "source_count", source_count)
        object.__setattr__(self, "result_count", result_count)
        object.__setattr__(self, "kept_indices", kept_indices)
        object.__setattr__(self, "dropped_indices", dropped_indices)
        object.__setattr__(self, "merge_groups", merge_groups)
        object.__setattr__(self, "old_to_new", old_to_new)
        object.__setattr__(self, "media_unit", media_unit)
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "protected_indices", protected_indices)
        object.__setattr__(self, "source_layout", source_layout)
        object.__setattr__(self, "result_layout", result_layout)
        object.__setattr__(self, "ordering_policy", ordering_policy)

    @property
    def is_identity(self) -> bool:
        """Whether no entry was dropped, merged, or reordered."""

        return (
            self.source_count == self.result_count
            and not self.dropped_indices
            and not self.merge_groups
            and all(self.old_to_new[index] == index for index in range(self.source_count))
        )

    @property
    def operation(self) -> str:
        """Compact human-readable classification of this delta."""

        if self.is_identity:
            return "identity"
        if self.merge_groups and self.dropped_indices:
            return "merge_drop"
        if self.merge_groups:
            return "merge"
        if self.dropped_indices:
            return "drop"
        return "reorder"

    @classmethod
    def from_mapping(
        cls,
        *,
        source_count: int,
        old_to_new: Mapping[int, int | None] | Sequence[int | None],
        result_count: int | None = None,
        media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN,
        coordinates: Iterable[Any] = (),
        protected_indices: Iterable[int] = (),
        source_layout: Mapping[str, Any] | None = None,
        result_layout: Mapping[str, Any] | None = None,
        ordering_policy: OrderingPolicy | str = OrderingPolicy.PRESERVE_SOURCE,
    ) -> "LayoutDelta":
        """Build a delta and derive all redundant audit fields."""

        source_count = _require_count("source_count", source_count)
        mapping = _normalise_old_to_new(source_count, old_to_new)
        used_results = sorted(
            {new_index for new_index in mapping.values() if new_index is not None}
        )
        inferred_result_count = len(used_results)
        if used_results != list(range(inferred_result_count)):
            raise ValueError("old_to_new result indices must be contiguous from zero")
        if result_count is None:
            result_count = inferred_result_count
        elif result_count != inferred_result_count:
            raise ValueError(
                "result_count does not match the result indices used by old_to_new"
            )

        preimages = _preimages(mapping, result_count)
        return cls(
            source_count=source_count,
            result_count=result_count,
            kept_indices=tuple(
                index for index in range(source_count) if mapping[index] is not None
            ),
            dropped_indices=tuple(
                index for index in range(source_count) if mapping[index] is None
            ),
            merge_groups=tuple(group for group in preimages if len(group) > 1),
            old_to_new=mapping,
            media_unit=media_unit,
            coordinates=tuple(coordinates),
            protected_indices=tuple(protected_indices),
            source_layout=source_layout or {},
            result_layout=result_layout or {},
            ordering_policy=ordering_policy,
        )

    @classmethod
    def identity(
        cls,
        count: int,
        *,
        media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN,
        coordinates: Iterable[Any] = (),
        protected_indices: Iterable[int] = (),
        layout: Mapping[str, Any] | None = None,
    ) -> "LayoutDelta":
        """Create an identity delta."""

        layout = layout or {}
        return cls.from_mapping(
            source_count=count,
            old_to_new=tuple(range(count)),
            result_count=count,
            media_unit=media_unit,
            coordinates=coordinates,
            protected_indices=protected_indices,
            source_layout=layout,
            result_layout=layout,
        )

    @classmethod
    def drop(
        cls,
        source_count: int,
        dropped_indices: Iterable[int],
        *,
        media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN,
        coordinates: Iterable[Any] = (),
        protected_indices: Iterable[int] = (),
        source_layout: Mapping[str, Any] | None = None,
        result_layout: Mapping[str, Any] | None = None,
    ) -> "LayoutDelta":
        """Create a stable compaction that drops selected source entries."""

        source_count = _require_count("source_count", source_count)
        dropped = _normalise_index_tuple("dropped_indices", dropped_indices)
        dropped_set = set(dropped)
        if any(index < 0 or index >= source_count for index in dropped):
            raise ValueError("dropped_indices contains an index outside source layout")
        mapping: dict[int, int | None] = {}
        next_index = 0
        for old_index in range(source_count):
            if old_index in dropped_set:
                mapping[old_index] = None
            else:
                mapping[old_index] = next_index
                next_index += 1
        return cls.from_mapping(
            source_count=source_count,
            old_to_new=mapping,
            result_count=next_index,
            media_unit=media_unit,
            coordinates=coordinates,
            protected_indices=protected_indices,
            source_layout=source_layout,
            result_layout=result_layout,
        )

    @classmethod
    def merge(
        cls,
        source_count: int,
        merge_groups: Iterable[Iterable[int]],
        *,
        dropped_indices: Iterable[int] = (),
        media_unit: str | LayoutTokenUnit = LayoutTokenUnit.TOKEN,
        coordinates: Iterable[Any] = (),
        protected_indices: Iterable[int] = (),
        source_layout: Mapping[str, Any] | None = None,
        result_layout: Mapping[str, Any] | None = None,
    ) -> "LayoutDelta":
        """Create a source-ordered merge, optionally with independent drops."""

        source_count = _require_count("source_count", source_count)
        groups = tuple(
            _normalise_index_tuple("merge_groups", group) for group in merge_groups
        )
        if any(len(group) < 2 for group in groups):
            raise ValueError("merge groups must contain at least two indices")
        flattened = tuple(index for group in groups for index in group)
        if len(set(flattened)) != len(flattened):
            raise ValueError("merge groups must be disjoint")
        if any(index < 0 or index >= source_count for index in flattened):
            raise ValueError("merge_groups contains an index outside source layout")

        dropped = _normalise_index_tuple("dropped_indices", dropped_indices)
        if any(index < 0 or index >= source_count for index in dropped):
            raise ValueError("dropped_indices contains an index outside source layout")
        if set(flattened).intersection(dropped):
            raise ValueError("an index cannot be both merged and dropped")

        group_by_index: dict[int, tuple[int, ...]] = {
            index: group for group in groups for index in group
        }
        units: list[tuple[int, ...]] = []
        visited: set[int] = set()
        dropped_set = set(dropped)
        for old_index in range(source_count):
            if old_index in visited or old_index in dropped_set:
                continue
            unit = group_by_index.get(old_index, (old_index,))
            units.append(unit)
            visited.update(unit)
        units.sort(key=lambda group: group[0])

        mapping: dict[int, int | None] = {
            index: None for index in range(source_count)
        }
        for new_index, group in enumerate(units):
            for old_index in group:
                mapping[old_index] = new_index
        return cls.from_mapping(
            source_count=source_count,
            old_to_new=mapping,
            result_count=len(units),
            media_unit=media_unit,
            coordinates=coordinates,
            protected_indices=protected_indices,
            source_layout=source_layout,
            result_layout=result_layout,
        )

    def compose(self, following: "LayoutDelta") -> "LayoutDelta":
        """Compose ``self`` followed by ``following``.

        Layouts from different media units are intentionally not composable.
        This makes accidental image/audio/video cross-wiring fail immediately.
        """

        if not isinstance(following, LayoutDelta):
            raise TypeError("following must be a LayoutDelta")
        if self.media_unit != following.media_unit:
            raise ValueError(
                "cannot compose layouts from different media units: "
                f"{self.media_unit!r} != {following.media_unit!r}"
            )
        if self.result_count != following.source_count:
            raise ValueError(
                "layout boundary count mismatch: "
                f"{self.result_count} != {following.source_count}"
            )
        self_result_id = self.result_layout.get("layout_id")
        following_source_id = following.source_layout.get("layout_id")
        if (
            self_result_id is not None
            and following_source_id is not None
            and self_result_id != following_source_id
        ):
            raise ValueError(
                "layout boundary identity mismatch: "
                f"{self_result_id!r} != {following_source_id!r}"
            )

        mapping: dict[int, int | None] = {}
        for old_index in range(self.source_count):
            middle_index = self.old_to_new[old_index]
            mapping[old_index] = (
                None
                if middle_index is None
                else following.old_to_new[middle_index]
            )

        protected = set(self.protected_indices)
        first_preimages = _preimages(self.old_to_new, self.result_count)
        for middle_index in following.protected_indices:
            preimage = first_preimages[middle_index]
            if len(preimage) == 1:
                protected.add(preimage[0])

        ordering_policy = _compose_ordering_policy(
            self.ordering_policy, following.ordering_policy
        )
        return LayoutDelta.from_mapping(
            source_count=self.source_count,
            old_to_new=mapping,
            result_count=following.result_count,
            media_unit=self.media_unit,
            coordinates=self.coordinates,
            protected_indices=sorted(protected),
            source_layout=self.source_layout,
            result_layout=following.result_layout,
            ordering_policy=ordering_policy,
        )

    @classmethod
    def compose_all(cls, deltas: Iterable["LayoutDelta"]) -> "LayoutDelta":
        """Compose a non-empty ordered sequence of deltas."""

        iterator = iter(deltas)
        try:
            result = next(iterator)
        except StopIteration as exc:
            raise ValueError("compose_all requires at least one delta") from exc
        if not isinstance(result, cls):
            raise TypeError("compose_all entries must be LayoutDelta instances")
        for delta in iterator:
            result = result.compose(delta)
        return result


def _compose_ordering_policy(
    first: OrderingPolicy,
    second: OrderingPolicy,
) -> OrderingPolicy:
    if OrderingPolicy.ARBITRARY in (first, second):
        return OrderingPolicy.ARBITRARY
    if OrderingPolicy.EXPLICIT in (first, second):
        return OrderingPolicy.EXPLICIT
    return OrderingPolicy.PRESERVE_SOURCE


@dataclass(frozen=True)
class AlignmentDelta:
    """Validated many-to-many alignment relation between two streams.

    ``mapping`` maps source indices to ordered, unique target indices.  Missing
    source keys represent unknown/no declared alignment.  Composition performs
    ordinary relation composition and therefore also works for one-to-many and
    many-to-one media alignments.
    """

    source_stream: str
    target_stream: str
    source_count: int
    target_count: int
    mapping: Mapping[int, Sequence[int] | int] = field(default_factory=dict)
    coupling_policy: AlignmentCouplingPolicy | str = AlignmentCouplingPolicy.PRESERVE

    def __post_init__(self) -> None:
        if not isinstance(self.source_stream, str) or not self.source_stream.strip():
            raise ValueError("source_stream must be a non-empty string")
        if not isinstance(self.target_stream, str) or not self.target_stream.strip():
            raise ValueError("target_stream must be a non-empty string")
        source_stream = self.source_stream.strip()
        target_stream = self.target_stream.strip()
        source_count = _require_count("source_count", self.source_count)
        target_count = _require_count("target_count", self.target_count)
        try:
            coupling_policy = AlignmentCouplingPolicy(self.coupling_policy)
        except (TypeError, ValueError) as exc:
            allowed = ", ".join(policy.value for policy in AlignmentCouplingPolicy)
            raise ValueError(f"coupling_policy must be one of: {allowed}") from exc

        if not isinstance(self.mapping, Mapping):
            raise TypeError("mapping must be a mapping")
        normalised: dict[int, tuple[int, ...]] = {}
        for raw_source, raw_targets in self.mapping.items():
            source_index = _normalise_index("mapping source", raw_source)
            if not 0 <= source_index < source_count:
                raise ValueError(
                    f"mapping source index {source_index} is outside source stream"
                )
            if isinstance(raw_targets, int) and not isinstance(raw_targets, bool):
                targets = (raw_targets,)
            else:
                targets = _normalise_index_tuple("mapping targets", raw_targets)
            if any(index < 0 or index >= target_count for index in targets):
                raise ValueError(
                    f"mapping[{source_index}] contains an index outside target stream"
                )
            normalised[source_index] = targets

        normalised = dict(sorted(normalised.items()))
        if coupling_policy is AlignmentCouplingPolicy.INDEPENDENT and normalised:
            raise ValueError("independent streams cannot declare an alignment mapping")
        if coupling_policy is AlignmentCouplingPolicy.LOCKSTEP:
            expected = {index: (index,) for index in range(source_count)}
            if source_count != target_count or normalised != expected:
                raise ValueError(
                    "lockstep alignment requires equal counts and an identity mapping"
                )

        object.__setattr__(self, "source_stream", source_stream)
        object.__setattr__(self, "target_stream", target_stream)
        object.__setattr__(self, "source_count", source_count)
        object.__setattr__(self, "target_count", target_count)
        object.__setattr__(self, "mapping", MappingProxyType(normalised))
        object.__setattr__(self, "coupling_policy", coupling_policy)

    @classmethod
    def identity(
        cls,
        stream: str,
        count: int,
        *,
        target_stream: str | None = None,
    ) -> "AlignmentDelta":
        """Create a lockstep identity relation, optionally between named peers."""

        return cls(
            source_stream=stream,
            target_stream=target_stream or stream,
            source_count=count,
            target_count=count,
            mapping={index: (index,) for index in range(count)},
            coupling_policy=AlignmentCouplingPolicy.LOCKSTEP,
        )

    def compose(self, following: "AlignmentDelta") -> "AlignmentDelta":
        """Compose this source-to-middle relation with middle-to-target."""

        if not isinstance(following, AlignmentDelta):
            raise TypeError("following must be an AlignmentDelta")
        if self.target_stream != following.source_stream:
            raise ValueError(
                "alignment stream mismatch: "
                f"{self.target_stream!r} != {following.source_stream!r}"
            )
        if self.target_count != following.source_count:
            raise ValueError(
                "alignment boundary count mismatch: "
                f"{self.target_count} != {following.source_count}"
            )
        if self.coupling_policy is not following.coupling_policy:
            raise ValueError(
                "alignment coupling policies must match when composed: "
                f"{self.coupling_policy.value!r} != "
                f"{following.coupling_policy.value!r}"
            )

        if self.coupling_policy is AlignmentCouplingPolicy.INDEPENDENT:
            composed_mapping: dict[int, tuple[int, ...]] = {}
        else:
            composed_mapping = {}
            for source_index, middle_indices in self.mapping.items():
                targets: set[int] = set()
                for middle_index in middle_indices:
                    targets.update(following.mapping.get(middle_index, ()))
                composed_mapping[source_index] = tuple(sorted(targets))

        return AlignmentDelta(
            source_stream=self.source_stream,
            target_stream=following.target_stream,
            source_count=self.source_count,
            target_count=following.target_count,
            mapping=composed_mapping,
            coupling_policy=self.coupling_policy,
        )

    @classmethod
    def compose_all(cls, deltas: Iterable["AlignmentDelta"]) -> "AlignmentDelta":
        """Compose a non-empty ordered sequence of alignment deltas."""

        iterator = iter(deltas)
        try:
            result = next(iterator)
        except StopIteration as exc:
            raise ValueError("compose_all requires at least one delta") from exc
        if not isinstance(result, cls):
            raise TypeError("compose_all entries must be AlignmentDelta instances")
        for delta in iterator:
            result = result.compose(delta)
        return result


__all__ = [
    "AlignmentDelta",
    "AlignmentCouplingPolicy",
    "CouplingPolicy",
    "LayoutTokenUnit",
    "LayoutDelta",
    "layout_delta_record",
    "layout_delta_summary",
    "ModalityLayout",
    "OrderingPolicy",
    "TokenUnit",
]
