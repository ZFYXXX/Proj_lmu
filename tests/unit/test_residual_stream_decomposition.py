from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.analysis.common.residual import (
    aggregate_residual_samples,
    residual_vector_metrics,
)
from mllm_comp.analysis.probes import ResidualStreamDecomposition


def test_residual_probe_accepts_audio_token_group() -> None:
    ResidualStreamDecomposition().validate_config(
        {"token_groups": ["all", "text", "audio"], "operations": []}
    )


def test_residual_vector_metrics_match_update_definitions() -> None:
    metrics = residual_vector_metrics(
        [[3.0, 4.0], [1.0, 0.0]],
        [[6.0, 8.0], [1.0, 1.0]],
    )

    assert metrics is not None
    assert metrics["input_norm"] == pytest.approx(3.0)
    assert metrics["output_norm"] == pytest.approx((10.0 + 2**0.5) / 2.0)
    assert metrics["update_norm"] == pytest.approx(3.0)
    assert metrics["residual_ratio"] == pytest.approx(1.0)
    assert metrics["cosine_to_input"] == pytest.approx((1.0 + 2**-0.5) / 2.0)


def test_residual_aggregation_keeps_observation_and_causal_views_distinct() -> None:
    samples = [
        {
            "residual_stats": {
                "0": {
                    "attn": {"vision": {"update_norm": 2.0}},
                    "ffn": {"vision": {"update_norm": 4.0}},
                }
            },
            "causal": {
                "0": {
                    "remove_attn_update": {
                        "status": "finite",
                        "decision": {"js": 0.2},
                        "token_groups": {"vision": {"js": 0.1}},
                    }
                }
            },
        },
        {
            "residual_stats": {
                "0": {
                    "attn": {"vision": {"update_norm": 4.0}},
                    "ffn": {"vision": {"update_norm": 8.0}},
                }
            },
            "causal": {
                "0": {
                    "remove_attn_update": {
                        "status": "finite",
                        "decision": {"js": 0.4},
                        "token_groups": {"vision": {"js": 0.3}},
                    }
                }
            },
        },
    ]

    aggregate = aggregate_residual_samples(
        iter(samples),
        residual_metrics=["update_norm"],
        distribution_metrics=["js"],
        token_groups=["vision"],
    )

    assert aggregate["residual_stats"]["0"]["attn"]["vision"][
        "update_norm"
    ] == pytest.approx(3.0)
    assert aggregate["residual_stats"]["0"]["ffn"]["vision"][
        "update_norm"
    ] == pytest.approx(6.0)
    assert aggregate["causal"]["0"]["remove_attn_update"]["decision"][
        "js"
    ] == pytest.approx(0.3)


def test_residual_probe_writes_separate_candidate_views(
    tmp_path: Path,
) -> None:
    attention = SimpleNamespace(
        num_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        config=SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            hidden_size=4,
        ),
    )
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    handles = {
        "llm": SimpleNamespace(),
        "llm.layer[0]": layer,
        "llm.layer[0].attn": attention,
        "llm.layer[0].ffn": layer.mlp,
        "llm.final_norm": object(),
        "lm_head": object(),
    }
    backbone = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_vl"),
        model_adapter=SimpleNamespace(structural_handles=lambda _model: handles),
    )
    runtime = SimpleNamespace(
        model=object(),
        model_path_or_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="revision",
    )
    session = ResidualStreamDecomposition().open(
        backbone=backbone,
        runtime=runtime,
        config={
            "name": "residual_stream_decomposition",
            "layers": [0],
            "operations": [],
            "metrics": ["update_norm"],
            "token_groups": ["vision"],
            "save_raw": True,
        },
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    session.records.add_success(
        {
            "sample_id": "sample",
            "task_name": "task",
            "residual_stats": {
                "0": {
                    "attn": {"vision": {"update_norm": 2.0}},
                    "ffn": {"vision": {"update_norm": 1.0}},
                }
            },
            "causal": {},
        }
    )

    record = session.finish()
    session.close()

    assert record.summary["finite_aggregate_values"] == 2
    assert set(record.artifact_paths) == {"aggregate", "per_sample"}
    artifact = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    assert artifact["payload"]["residual_stats"]["0"]["attn"]["vision"][
        "update_norm"
    ] == 2.0
    assert artifact["payload"]["causal"] == {}
    raw = json.loads(
        Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
    )
    assert raw["payload"]["samples"][0]["sample_id"] == "sample"
