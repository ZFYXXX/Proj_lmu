from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.fixed_visual_merge import (  # noqa: E402
    FixedVisualMergeSpec,
    fixed_visual_merge,
)
from mllm_comp.compressors.image.stage2_projector import (  # noqa: E402
    Stage2ProjectorSpec,
    create_stage2_projector,
    load_stage2_projector,
    save_stage2_projector,
)


@pytest.mark.parametrize(
    "variant", ["mean_only", "meta", "dispersion", "meta_residual"]
)
def test_zero_initialization_is_exact_stage1_mean_and_backpropagates(variant: str) -> None:
    tokens = torch.randn(80, 16, requires_grad=True)
    spec = FixedVisualMergeSpec(partition="feature_anchor", target_count=64)
    baseline = fixed_visual_merge(tokens, grid_height=8, grid_width=10, spec=spec)
    projector = create_stage2_projector(
        Stage2ProjectorSpec(hidden_size=16, variant=variant, bottleneck_dim=4)
    )
    result = fixed_visual_merge(
        tokens,
        grid_height=8,
        grid_width=10,
        spec=spec,
        projector=projector,
    )
    torch.testing.assert_close(result.tokens, baseline.tokens, rtol=0, atol=0)
    result.tokens.square().mean().backward()
    assert any(
        parameter.grad is not None and torch.count_nonzero(parameter.grad).item() > 0
        for parameter in projector.parameters()
    )
    assert result.metadata["aggregation"] == f"stage2_{variant}"


def test_stage2_projector_checkpoint_round_trip(tmp_path) -> None:
    spec = Stage2ProjectorSpec(
        hidden_size=8, variant="meta_residual", bottleneck_dim=4
    )
    first = create_stage2_projector(spec)
    with torch.no_grad():
        next(first.parameters()).add_(1)
    checkpoint = tmp_path / "projector.safetensors"
    save_stage2_projector(first, checkpoint)
    second = create_stage2_projector(spec)
    load_stage2_projector(second, checkpoint)
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right)

