from __future__ import annotations

import pytest

from mllm_comp.compressors.image.mmtok import (
    MMTokSpec,
    greedy_mmtok_coverage_indices,
    mmtok_image_tokens,
)


torch = pytest.importorskip("torch")


def test_greedy_coverage_selects_complementary_columns() -> None:
    coverage = torch.tensor([[1.0, 0.8, 0.0], [0.0, 0.1, 0.9]])

    selected = greedy_mmtok_coverage_indices(coverage, 2)

    assert selected.tolist() == [0, 2]


def test_joint_coverage_returns_source_order_and_exact_layout_delta() -> None:
    projected = torch.tensor(
        [[1.0, 0.0], [0.95, 0.05], [0.0, 1.0], [-1.0, 0.0]]
    )
    vision = projected.clone()
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    output = mmtok_image_tokens(
        projected,
        vision,
        text,
        spec=MMTokSpec(target_count=2),
        source_layout={"image_index": 0},
    )

    assert output.indices.tolist() == sorted(output.indices.tolist())
    assert len(set(output.selection_order.tolist())) == 2
    torch.testing.assert_close(output.tokens, projected.index_select(0, output.indices))
    assert output.layout_delta.kept_indices == tuple(output.indices.tolist())
    assert output.layout_delta.operation == "drop"
    assert output.layout_delta.source_layout["image_index"] == 0
    assert output.metadata["text_token_count"] == 2
    assert output.metadata["coverage_compute_dtype"] == "float32"


def test_dynamic_ratio_uses_floor_and_minimum_one() -> None:
    spec = MMTokSpec(retain_ratio=0.2)

    assert spec.resolve_target_count(107) == 21
    assert spec.resolve_target_count(3) == 1


def test_identity_budget_skips_coverage_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_greedy(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("identity selection must not run the greedy kernel")

    monkeypatch.setattr(
        "mllm_comp.compressors.image.mmtok.greedy_mmtok_coverage_indices",
        unexpected_greedy,
    )
    projected = torch.randn(4, 3)

    output = mmtok_image_tokens(
        projected,
        torch.randn(4, 5),
        torch.randn(2, 3),
        spec=MMTokSpec(target_count=4),
    )

    assert output.indices.tolist() == [0, 1, 2, 3]
    assert output.tokens is projected
    assert output.layout_delta.is_identity


def test_bfloat16_inputs_match_float32_selection() -> None:
    projected = torch.tensor(
        [[1.0, 0.0], [0.7, 0.3], [0.0, 1.0], [-1.0, 0.0]]
    )
    vision = torch.tensor(
        [[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [-1.0, 0.0]]
    )
    text = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    spec = MMTokSpec(target_count=2)

    expected = mmtok_image_tokens(projected, vision, text, spec=spec)
    actual = mmtok_image_tokens(
        projected.bfloat16(), vision.bfloat16(), text.bfloat16(), spec=spec
    )

    assert actual.indices.tolist() == expected.indices.tolist()


@pytest.mark.parametrize(
    "kwargs",
    (
        {"retain_ratio": 0.0},
        {"target_count": 0},
        {"alpha": -0.1},
        {"text_temperature": 0.0},
        {"vision_temperature": float("nan")},
    ),
)
def test_invalid_spec_is_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        MMTokSpec(**kwargs)  # type: ignore[arg-type]


def test_multisource_shape_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="token counts"):
        mmtok_image_tokens(
            torch.randn(3, 4),
            torch.randn(2, 5),
            torch.randn(2, 4),
        )
