from __future__ import annotations

import pytest

from mllm_comp.analysis.probes import (
    AttentionFlow,
    HierarchicalTaylor,
    LateInjectionEarlyExit,
    PredictionDynamics,
    RepresentationDynamics,
    ResidualStreamDecomposition,
)


@pytest.mark.parametrize(
    ("probe", "config"),
    [
        (RepresentationDynamics(), {"layers": [True]}),
        (RepresentationDynamics(), {"save_raw": "false"}),
        (LateInjectionEarlyExit(), {"early_exit_thresholds": [True]}),
        (ResidualStreamDecomposition(), {"collect_residual_stats": "false"}),
        (ResidualStreamDecomposition(), {"residual_atol": "0.002"}),
        (PredictionDynamics(), {"top_k": True}),
        (AttentionFlow(), {"token_groups": "vision"}),
        (AttentionFlow(), {"mass_conservation_tolerance": "0.002"}),
        (AttentionFlow(), {"layer_batch_size": True}),
        (HierarchicalTaylor(), {"layer_batch_size": True}),
    ],
)
def test_builtin_analysis_probes_reject_ambiguous_scalar_types(
    probe: object,
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        probe.validate_config(config)  # type: ignore[attr-defined]
