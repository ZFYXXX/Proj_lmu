from __future__ import annotations

import pytest

from mllm_comp.core.layout import (
    AlignmentDelta,
    CouplingPolicy,
    LayoutDelta,
    ModalityLayout,
    OrderingPolicy,
    TokenUnit,
    layout_delta_record,
    layout_delta_summary,
)


def test_identity_delta_is_complete_and_immutable() -> None:
    delta = LayoutDelta.identity(
        3,
        media_unit=TokenUnit.IMAGE_PATCH,
        coordinates=((0, 0), (0, 1), (1, 0)),
        protected_indices=(0,),
        layout={"layout_id": "vision-0", "media_unit": "image_patch"},
    )

    assert delta.is_identity
    assert delta.operation == "identity"
    assert delta.kept_indices == (0, 1, 2)
    assert delta.dropped_indices == ()
    assert delta.merge_groups == ()
    assert dict(delta.old_to_new) == {0: 0, 1: 1, 2: 2}
    assert delta.media_unit == "image_patch"


def test_layout_delta_audit_record_keeps_exact_map_and_compact_hash() -> None:
    delta = LayoutDelta.merge(
        5,
        ((1, 2),),
        dropped_indices=(4,),
        media_unit="image_feature",
        protected_indices=(0,),
        source_layout={"stage": "projector"},
        result_layout={"stage": "llm_input"},
    )

    exact = layout_delta_record(delta)
    compact = layout_delta_summary(delta)

    assert exact["old_to_new"] == [0, 1, 1, 2, None]
    assert exact["merge_groups"] == [[1, 2]]
    assert exact["operation"] == "merge_drop"
    assert compact["mapping_sha256"] == exact["mapping_sha256"]
    assert "old_to_new" not in compact
    assert compact["source_count"] == 5
    assert compact["result_count"] == 3
    assert delta.ordering_policy is OrderingPolicy.PRESERVE_SOURCE
    with pytest.raises(TypeError):
        delta.old_to_new[0] = None  # type: ignore[index]
    with pytest.raises(TypeError):
        delta.source_layout["layout_id"] = "mutated"  # type: ignore[index]


def test_modality_layout_applies_drop_and_carries_coordinates() -> None:
    layout = ModalityLayout(
        stream_id="image:0",
        modality="image",
        count=4,
        media_unit="image_patch",
        coordinates=((0, 0), (0, 1), (1, 0), (1, 1)),
        protected_indices=(0,),
        metadata={"layout_id": "before"},
    )
    delta = LayoutDelta.drop(
        4,
        (2,),
        media_unit="image_patch",
        protected_indices=(0,),
        result_layout={"layout_id": "after"},
    )

    result = layout.apply(delta)

    assert result.count == 3
    assert result.coordinates == ((0, 0), (0, 1), (1, 1))
    assert result.protected_indices == (0,)
    assert result.metadata["layout_id"] == "after"


def test_modality_layout_rejects_cross_stream_physical_unit() -> None:
    layout = ModalityLayout("image:0", "image", 2, "image_patch")

    with pytest.raises(ValueError, match="media_unit mismatch"):
        layout.apply(LayoutDelta.identity(2, media_unit="audio_frame"))


def test_drop_delta_derives_kept_dropped_and_mapping() -> None:
    delta = LayoutDelta.drop(
        6,
        (1, 4),
        media_unit="audio_frame",
        protected_indices=(0, 5),
        source_layout={"layout_id": "audio-before"},
        result_layout={"layout_id": "audio-after"},
    )

    assert delta.operation == "drop"
    assert delta.source_count == 6
    assert delta.result_count == 4
    assert delta.kept_indices == (0, 2, 3, 5)
    assert delta.dropped_indices == (1, 4)
    assert dict(delta.old_to_new) == {
        0: 0,
        1: None,
        2: 1,
        3: 2,
        4: None,
        5: 3,
    }


def test_drop_rejects_protected_or_out_of_range_indices() -> None:
    with pytest.raises(ValueError, match="protected index cannot be dropped"):
        LayoutDelta.drop(4, (1,), protected_indices=(1,))
    with pytest.raises(ValueError, match="outside source layout"):
        LayoutDelta.drop(4, (4,))
    with pytest.raises(ValueError, match="ascending source order"):
        LayoutDelta.drop(4, (2, 1))


def test_merge_delta_supports_non_adjacent_groups_and_independent_drop() -> None:
    delta = LayoutDelta.merge(
        7,
        ((0, 4), (2, 5)),
        dropped_indices=(6,),
        media_unit=TokenUnit.VIDEO_PATCH,
    )

    assert delta.operation == "merge_drop"
    assert delta.result_count == 4
    assert delta.kept_indices == (0, 1, 2, 3, 4, 5)
    assert delta.dropped_indices == (6,)
    assert delta.merge_groups == ((0, 4), (2, 5))
    assert dict(delta.old_to_new) == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 0,
        5: 2,
        6: None,
    }


def test_merge_rejects_overlap_singletons_and_protected_members() -> None:
    with pytest.raises(ValueError, match="at least two"):
        LayoutDelta.merge(4, ((0,),))
    with pytest.raises(ValueError, match="disjoint"):
        LayoutDelta.merge(4, ((0, 1), (1, 2)))
    with pytest.raises(ValueError, match="both merged and dropped"):
        LayoutDelta.merge(4, ((0, 1),), dropped_indices=(1,))
    with pytest.raises(ValueError, match="protected index cannot be merged"):
        LayoutDelta.merge(4, ((0, 2),), protected_indices=(2,))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kept_indices": (0,)}, "kept_indices"),
        ({"dropped_indices": (1,)}, "dropped_indices"),
        ({"merge_groups": ()}, "merge_groups"),
        ({"result_count": 3}, "outside result layout|preimage"),
    ],
)
def test_direct_construction_rejects_inconsistent_redundant_fields(
    overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "source_count": 3,
        "result_count": 2,
        "kept_indices": (0, 1, 2),
        "dropped_indices": (),
        "merge_groups": ((0, 1),),
        "old_to_new": {0: 0, 1: 0, 2: 1},
    }
    values.update(overrides)
    with pytest.raises(ValueError, match=message):
        LayoutDelta(**values)  # type: ignore[arg-type]


def test_mapping_must_cover_source_and_use_contiguous_results() -> None:
    with pytest.raises(ValueError, match="every source index"):
        LayoutDelta.from_mapping(source_count=3, old_to_new={0: 0, 1: 1})
    with pytest.raises(ValueError, match="contiguous"):
        LayoutDelta.from_mapping(source_count=2, old_to_new=(0, 2))
    with pytest.raises(ValueError, match="does not match"):
        LayoutDelta.from_mapping(
            source_count=2, old_to_new=(0, 1), result_count=1
        )


def test_reorder_requires_an_explicit_non_preserving_policy() -> None:
    with pytest.raises(ValueError, match="result groups in source order"):
        LayoutDelta.from_mapping(source_count=3, old_to_new=(1, 0, 2))

    delta = LayoutDelta.from_mapping(
        source_count=3,
        old_to_new=(1, 0, 2),
        ordering_policy=OrderingPolicy.EXPLICIT,
    )
    assert delta.operation == "reorder"
    assert delta.ordering_policy is OrderingPolicy.EXPLICIT


def test_coordinates_and_layout_metadata_are_validated() -> None:
    with pytest.raises(ValueError, match="one entry per source"):
        LayoutDelta.identity(2, coordinates=((0, 0),))
    with pytest.raises(ValueError, match="does not match delta"):
        LayoutDelta.identity(
            2,
            media_unit="image_patch",
            layout={"media_unit": "audio_frame"},
        )


def test_layout_composition_retains_endpoints_and_builds_full_preimage() -> None:
    first = LayoutDelta.drop(
        6,
        (1,),
        media_unit="image_patch",
        coordinates=((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)),
        protected_indices=(0,),
        source_layout={"layout_id": "raw"},
        result_layout={"layout_id": "encoded"},
    )
    second = LayoutDelta.merge(
        5,
        ((1, 3),),
        dropped_indices=(4,),
        media_unit="image_patch",
        protected_indices=(2,),
        source_layout={"layout_id": "encoded"},
        result_layout={"layout_id": "llm-input"},
    )

    composed = first.compose(second)

    assert composed.operation == "merge_drop"
    assert composed.source_count == 6
    assert composed.result_count == 3
    assert composed.dropped_indices == (1, 5)
    assert composed.merge_groups == ((2, 4),)
    assert dict(composed.old_to_new) == {
        0: 0,
        1: None,
        2: 1,
        3: 2,
        4: 1,
        5: None,
    }
    assert composed.protected_indices == (0, 3)
    assert composed.coordinates == first.coordinates
    assert composed.source_layout["layout_id"] == "raw"
    assert composed.result_layout["layout_id"] == "llm-input"


def test_composition_rejects_later_mutation_of_an_earlier_protected_token() -> None:
    first = LayoutDelta.identity(3, protected_indices=(0,))
    following = LayoutDelta.merge(3, ((0, 2),))

    with pytest.raises(ValueError, match="protected index cannot be merged"):
        first.compose(following)


def test_composition_validates_boundary_count_and_layout_identity() -> None:
    with pytest.raises(ValueError, match="boundary count mismatch"):
        LayoutDelta.identity(3).compose(LayoutDelta.identity(2))

    first = LayoutDelta.identity(2, layout={"layout_id": "a"})
    following = LayoutDelta.identity(2, layout={"layout_id": "b"})
    with pytest.raises(ValueError, match="identity mismatch"):
        first.compose(following)


def test_media_units_are_isolated_and_can_be_tracked_independently() -> None:
    image = LayoutDelta.drop(4, (1,), media_unit="image_patch")
    audio = LayoutDelta.drop(4, (2,), media_unit="audio_frame")
    per_unit = {delta.media_unit: delta for delta in (image, audio)}

    assert per_unit["image_patch"].dropped_indices == (1,)
    assert per_unit["audio_frame"].dropped_indices == (2,)
    with pytest.raises(ValueError, match="different media units"):
        image.compose(audio)


def test_compose_all_is_ordered_and_requires_a_delta() -> None:
    first = LayoutDelta.drop(4, (1,), media_unit="image_patch")
    second = LayoutDelta.drop(3, (2,), media_unit="image_patch")
    assert LayoutDelta.compose_all((first, second)).dropped_indices == (1, 3)
    with pytest.raises(ValueError, match="at least one"):
        LayoutDelta.compose_all(())


def test_alignment_delta_normalises_and_freezes_many_to_many_mapping() -> None:
    delta = AlignmentDelta(
        source_stream="image_patch",
        target_stream="llm_visual_token",
        source_count=3,
        target_count=4,
        mapping={0: (0, 1), 2: 3},
        coupling_policy="preserve",
    )

    assert delta.coupling_policy is CouplingPolicy.PRESERVE
    assert dict(delta.mapping) == {0: (0, 1), 2: (3,)}
    with pytest.raises(TypeError):
        delta.mapping[0] = (2,)  # type: ignore[index]


def test_alignment_validation_catches_invalid_relations() -> None:
    with pytest.raises(ValueError, match="outside source stream"):
        AlignmentDelta("a", "b", 1, 1, {1: (0,)})
    with pytest.raises(ValueError, match="outside target stream"):
        AlignmentDelta("a", "b", 1, 1, {0: (1,)})
    with pytest.raises(ValueError, match="independent streams"):
        AlignmentDelta(
            "a", "b", 1, 1, {0: (0,)}, CouplingPolicy.INDEPENDENT
        )
    with pytest.raises(ValueError, match="lockstep alignment"):
        AlignmentDelta("a", "b", 2, 2, {0: (0,)}, CouplingPolicy.LOCKSTEP)


def test_alignment_composition_performs_relation_composition() -> None:
    first = AlignmentDelta(
        "patch",
        "projected",
        2,
        3,
        {0: (0, 1), 1: (2,)},
        CouplingPolicy.PRESERVE,
    )
    second = AlignmentDelta(
        "projected",
        "llm",
        3,
        2,
        {0: (0,), 1: (1,), 2: (1,)},
        CouplingPolicy.PRESERVE,
    )

    composed = first.compose(second)

    assert composed.source_stream == "patch"
    assert composed.target_stream == "llm"
    assert composed.source_count == 2
    assert composed.target_count == 2
    assert dict(composed.mapping) == {0: (0, 1), 1: (1,)}


def test_alignment_identity_and_compose_all() -> None:
    first = AlignmentDelta.identity("a", 3, target_stream="b")
    second = AlignmentDelta.identity("b", 3, target_stream="c")

    composed = AlignmentDelta.compose_all((first, second))

    assert composed.source_stream == "a"
    assert composed.target_stream == "c"
    assert dict(composed.mapping) == {0: (0,), 1: (1,), 2: (2,)}
    with pytest.raises(ValueError, match="at least one"):
        AlignmentDelta.compose_all(())


def test_alignment_composition_rejects_incompatible_boundaries() -> None:
    first = AlignmentDelta("a", "b", 1, 1, {0: (0,)})
    wrong_stream = AlignmentDelta("c", "d", 1, 1, {0: (0,)})
    with pytest.raises(ValueError, match="stream mismatch"):
        first.compose(wrong_stream)

    wrong_count = AlignmentDelta("b", "d", 2, 1, {0: (0,), 1: (0,)})
    with pytest.raises(ValueError, match="boundary count mismatch"):
        first.compose(wrong_count)

    wrong_policy = AlignmentDelta(
        "b", "d", 1, 1, {}, CouplingPolicy.CASCADE
    )
    with pytest.raises(ValueError, match="policies must match"):
        first.compose(wrong_policy)
