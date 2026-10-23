from collections import Counter

import pytest

from mllm_comp.analysis.common.visual_token_discovery import (
    joint_deletion_plan,
    size_matched_random_units,
    save_region_map,
)


def test_random_units_exactly_match_variable_token_budget_and_size_histogram() -> None:
    members = ((0,), (1,), (2, 3), (4, 5), (6, 7, 8), (9, 10, 11))
    selected = (0, 2, 4)
    for seed in ("11", "23", "47"):
        random_units = size_matched_random_units(members, selected, seed=seed)
        assert random_units == size_matched_random_units(members, selected, seed=seed)
        assert len(set(random_units)) == len(selected)
        assert Counter(len(members[i]) for i in random_units) == Counter({1: 1, 2: 1, 3: 1})
        assert sum(len(members[i]) for i in random_units) == 6
    with pytest.raises(ValueError, match="unique"):
        size_matched_random_units(members, (0, 0), seed="11")


def test_joint_plan_measures_low_and_high_oracle_with_budget_matched_controls() -> None:
    members = tuple(tuple(range(i * 2, i * 2 + 2)) for i in range(8))
    scores = {"feature_diversity": list(range(8)), "text_visual_attention_layer_0_prompt": list(reversed(range(8)))}
    plan = joint_deletion_plan(effects=list(range(8)), members=members, scores=scores,
                               fractions=[0.25, 0.5, 0.75], seeds=[11, 23, 47], sample_id="test")
    assert len(plan) == 48
    assert next(x for x in plan if x["ranking"] == "singleton_oracle_abs_low")["units"] == (0, 1)
    assert next(x for x in plan if x["ranking"] == "singleton_oracle_high")["units"] == (6, 7)
    for condition in plan:
        assert len(condition["units"]) == round(8 * condition["fraction_units"])
        assert len(set(condition["units"])) == len(condition["units"])


def test_joint_rankings_share_exact_area_quota_and_abs_oracle_excludes_large_negative_effect() -> None:
    members = ((0,), (1,), (2,), (3,), (4, 5), (6, 7), (8, 9), (10, 11))
    scores = {"feature_diversity": list(range(8)), "text_visual_attention_layer_0_prompt": list(reversed(range(8)))}
    plan = joint_deletion_plan(effects=[-100, .01, .02, 1, -100, .01, .02, 1], members=members,
                               scores=scores, fractions=[.5], seeds=[11, 23, 47], sample_id="test")
    for condition in plan:
        assert Counter(len(members[i]) for i in condition["units"]) == {1: 2, 2: 2}
    oracle = next(x for x in plan if x["ranking"] == "singleton_oracle_abs_low")
    assert oracle["units"] == (1, 2, 5, 6)


def test_region_map_projects_grid_cells_to_normalized_image_coordinates(tmp_path) -> None:
    pytest.importorskip("matplotlib")
    from PIL import Image
    result = save_region_map(
        media=[Image.new("RGB", (200, 100))], members=((0, 1), (2, 3)),
        effects=[-.2, .5], height=2, width=2, path=str(tmp_path / "regions.png"), title="test",
    )
    assert result["unit_boxes_xyxy_normalized"] == [[0, 0, 1, .5], [0, .5, 1, 1]]
    assert result["largest_positive_unit"] == 1
    assert (tmp_path / "regions.png").exists()
