import pytest

from mllm_comp.compressors.image.per_image_budget import image_spans_and_budgets


def test_two_images_are_capped_independently():
    mask = [False] + [True] * 144 + [False, False] + [True] * 4 + [False]
    assert image_spans_and_budgets(mask, [[1, 24, 24], [1, 4, 4]], merge_size=2) == [
        (1, 145, 128), (147, 151, 4)
    ]


def test_text_only_and_invalid_layout():
    assert image_spans_and_budgets([False, False], None, merge_size=2) == []
    with pytest.raises(ValueError, match="spans do not match"):
        image_spans_and_budgets([False, True], None, merge_size=2)
    with pytest.raises(ValueError, match="post-merger grid"):
        image_spans_and_budgets([True], [[1, 4, 4]], merge_size=2)
