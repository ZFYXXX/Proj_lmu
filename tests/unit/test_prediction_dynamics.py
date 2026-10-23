from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.analysis.bindings import (
    AnalysisBackboneBinding,
    AttentionGeometry,
    DecoderLayerBinding,
)
from mllm_comp.analysis.common.prediction import (
    aggregate_prediction_samples,
    prediction_dynamics_metrics,
)
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes import PredictionDynamics
from mllm_comp.analysis.probes.prediction_dynamics import _PredictionBoundaryCapture


def test_prediction_metrics_preserve_direction_and_final_token_rank() -> None:
    report = prediction_dynamics_metrics(
        [0.0, 4.0, 1.0],
        [5.0, 0.0, 1.0],
        top_k=2,
    )

    assert report["metrics"]["kl_final_to_layer"] > 0.0
    assert report["metrics"]["kl_layer_to_final"] > 0.0
    assert report["metrics"]["top1_matches_final"] == 0.0
    assert report["metrics"]["rank_of_final_top1"] == 3.0
    assert report["layer_top1"]["token_id"] == 1
    assert report["final_top1"]["token_id"] == 0
    assert [item["token_id"] for item in report["top_tokens"]] == [1, 2]

    final = prediction_dynamics_metrics(
        [5.0, 0.0, 1.0],
        [5.0, 0.0, 1.0],
        top_k=1,
    )
    assert final["metrics"]["js"] == pytest.approx(0.0, abs=1e-12)
    assert final["metrics"]["top1_matches_final"] == 1.0
    assert final["metrics"]["rank_of_final_top1"] == 1.0


def test_prediction_aggregation_excludes_raw_top_tokens() -> None:
    aggregate = aggregate_prediction_samples(
        iter(
            [
                {"layers": {"0": {"metrics": {"js": 0.2}}}},
                {"layers": {"0": {"metrics": {"js": 0.4}}}},
            ]
        ),
        metrics=["js"],
    )

    assert aggregate["layers"]["0"]["js"] == pytest.approx(0.3)
    assert aggregate["layers"]["0"]["js_finite_sample_count"] == 2
    assert "top_tokens" not in aggregate["layers"]["0"]


def test_prediction_probe_writes_first_answer_trajectory_artifact(
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
        processor=None,
        model_path_or_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="revision",
    )
    session = PredictionDynamics().open(
        backbone=backbone,
        runtime=runtime,
        config={
            "name": "prediction_dynamics",
            "layers": [0],
            "metrics": ["js", "top1_matches_final"],
            "top_k": 2,
            "save_raw": True,
        },
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    session.records.add_success(
        {
            "sample_id": "sample",
            "task_name": "task",
            "layers": {
                "0": {
                    "metrics": {"js": 0.1, "top1_matches_final": 1.0},
                    "top_tokens": [{"token_id": 1, "token": "x", "prob": 0.5}],
                }
            },
        }
    )

    record = session.finish()
    session.close()

    assert record.summary["trajectory"] == "prefill_first_answer_token_logit_lens_v1"
    assert set(record.artifact_paths) == {"aggregate", "per_sample"}
    artifact = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    assert artifact["payload"]["layers"]["0"]["js"] == 0.1
    assert artifact["payload"]["trajectory"] == (
        "prefill_first_answer_token_logit_lens_v1"
    )
    raw = json.loads(
        Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
    )
    assert raw["payload"]["samples"][0]["sample_id"] == "sample"


def test_prediction_boundary_includes_provider_post_layer_injection() -> None:
    torch = pytest.importorskip("torch")

    class Add(torch.nn.Module):
        def __init__(self, value: float) -> None:
            super().__init__()
            self.value = value

        def forward(self, hidden: object) -> object:
            return hidden + self.value

    layer0 = Add(1.0)
    layer1 = Add(2.0)
    final_norm = Add(0.0)
    bindings = (
        DecoderLayerBinding(0, layer0, object(), object()),
        DecoderLayerBinding(1, layer1, object(), object()),
    )
    binding = AnalysisBackboneBinding(
        backbone_name="qwen3_vl",
        layers=bindings,
        final_norm=final_norm,
        lm_head=object(),
        attention_geometry=AttentionGeometry(2, 1, 2, 4),
    )
    capture = _PredictionBoundaryCapture(binding, bindings)
    hidden = torch.zeros((1, 3, 4))
    with HookSession() as hooks:
        capture.install(hooks)
        hidden = layer0(hidden)
        hidden = hidden + 10.0  # provider DeepStack injection between layers
        hidden = layer1(hidden)
        final_norm(hidden)
    values = capture.finish(sequence_length=3)

    assert torch.equal(values[0], torch.full((1, 3, 4), 11.0))
    assert torch.equal(values[1], torch.full((1, 3, 4), 13.0))
    assert capture.evidence()["qwen3_deepstack_semantics"].startswith(
        "captured_before_next_decoder_layer"
    )
