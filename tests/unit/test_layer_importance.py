from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.analysis.common.distribution import (
    IMPORTANCE_RANKING_DIRECTIONS,
    UNAVAILABLE_REASON_EXAMPLE_LIMIT,
    aggregate_layer_importance_samples,
    categorical_distribution_metrics,
)
from mllm_comp.analysis.common.interventions import (
    install_attention_edge_block,
    install_attention_head_ablation,
    install_ffn_neuron_ablation,
    install_ffn_update_skip,
    install_layer_update_skip,
)
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes.layer_importance import (
    LayerImportance,
    build_intervention_specs,
)


def test_categorical_distribution_metrics_preserve_kl_js_semantics() -> None:
    identical = categorical_distribution_metrics([4.0, 0.0], [4.0, 0.0])
    changed = categorical_distribution_metrics([4.0, 0.0], [0.0, 4.0])

    assert identical["kl_pq"] == pytest.approx(0.0, abs=1e-12)
    assert identical["kl_qp"] == pytest.approx(0.0, abs=1e-12)
    assert identical["js"] == pytest.approx(0.0, abs=1e-12)
    assert identical["top1_changed"] == 0.0
    assert changed["kl_pq"] > 0.0
    assert changed["kl_qp"] > 0.0
    assert 0.0 < changed["js"] < changed["kl_pq"]
    assert changed["top1_changed"] == 1.0
    assert changed["changed_prob_on_base_top1"] < identical["base_top1_prob"]


def test_layer_importance_aggregation_skips_absent_values() -> None:
    samples = [
        {
            "layers": {
                "0": {
                    "layer:vision": {
                        "status": "finite",
                        "decision": {"js": 0.2},
                        "token_groups": {
                            "vision": {"js": 0.4},
                            "text": {"js": None},
                        },
                    },
                    "layer:system": {
                        "status": "unavailable",
                        "reason": "system absent",
                    },
                }
            }
        },
        {
            "layers": {
                "0": {
                    "layer:vision": {
                        "status": "finite",
                        "decision": {"js": 0.4},
                        "token_groups": {
                            "vision": {"js": 0.8},
                            "text": {"js": 0.3},
                        },
                    }
                }
            }
        },
    ]

    result = aggregate_layer_importance_samples(
        iter(samples),
        metrics=["js"],
        token_groups=["text", "vision"],
    )

    vision = result["layers"]["0"]["layer:vision"]
    assert vision["decision"]["js"] == pytest.approx(0.3)
    assert vision["decision"]["js_finite_sample_count"] == 2
    assert vision["token_groups"]["vision"]["js"] == pytest.approx(0.6)
    assert vision["token_groups"]["text"]["js"] == pytest.approx(0.3)
    assert vision["token_groups"]["text"]["js_finite_sample_count"] == 1
    system = result["layers"]["0"]["layer:system"]
    assert system["status"] == "unavailable"
    assert system["unavailable_reasons"] == ["system absent"]


def test_layer_importance_bounds_dynamic_unavailable_reason_examples() -> None:
    result = aggregate_layer_importance_samples(
        (
            {
                "layers": {
                    "0": {
                        "layer:vision": {
                            "status": "unavailable",
                            "reason": f"dynamic-shape-{index}",
                        }
                    }
                }
            }
            for index in range(UNAVAILABLE_REASON_EXAMPLE_LIMIT + 5)
        ),
        metrics=["js"],
        token_groups=["vision"],
    )

    report = result["layers"]["0"]["layer:vision"]
    assert report["unavailable_sample_count"] == UNAVAILABLE_REASON_EXAMPLE_LIMIT + 5
    assert len(report["unavailable_reasons"]) == UNAVAILABLE_REASON_EXAMPLE_LIMIT
    assert report["unavailable_reasons_truncated"] is True
    assert report["unavailable_reason_example_limit"] == (
        UNAVAILABLE_REASON_EXAMPLE_LIMIT
    )


def test_intervention_specs_keep_target_to_source_attention_semantics() -> None:
    specs = build_intervention_specs(
        operations=["layer", "layer_strict", "attn", "ffn"],
        target_groups=["vision"],
        attention_edges=["t2v", "v2t"],
    )

    assert [spec.result_key for spec in specs] == [
        "layer:vision",
        "layer_strict:vision",
        "attn:t2v",
        "attn:v2t",
        "ffn:vision",
    ]
    text_to_vision = specs[2]
    assert text_to_vision.target_group == "text"
    assert text_to_vision.source_group == "vision"


def test_layer_importance_supports_audio_targets_and_edges() -> None:
    plugin = LayerImportance()
    plugin.validate_config(
        {
            "operations": ["layer", "attn"],
            "target_token_types": ["audio", "text"],
            "attn_edges": ["a2a", "a2t", "t2a", "t2t"],
        }
    )
    specs = build_intervention_specs(
        operations=["attn"],
        target_groups=["audio"],
        attention_edges=["a2a", "a2t", "t2a", "t2t"],
    )

    assert [(spec.target_group, spec.source_group) for spec in specs] == [
        ("audio", "audio"),
        ("audio", "text"),
        ("text", "audio"),
        ("text", "text"),
    ]


def test_layer_importance_ranking_metric_has_explicit_direction() -> None:
    LayerImportance().validate_config(
        {
            "operations": ["layer"],
            "target_token_types": ["vision"],
            "metrics": ["changed_prob_on_base_top1"],
            "ranking_metric": "changed_prob_on_base_top1",
        }
    )
    assert IMPORTANCE_RANKING_DIRECTIONS["changed_prob_on_base_top1"] is False

    with pytest.raises(ValueError, match="no unambiguous importance direction"):
        LayerImportance().validate_config(
            {
                "operations": ["layer"],
                "target_token_types": ["vision"],
                "metrics": ["base_entropy"],
                "ranking_metric": "base_entropy",
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        {"layers": [True]},
        {"operations": "layer"},
        {"save_raw": "false"},
    ],
)
def test_layer_importance_rejects_ambiguous_config_types(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        LayerImportance().validate_config(config)


def test_layer_importance_writes_ranked_candidate_artifact(
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
            _attn_implementation="flash_attention_2",
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
    session = LayerImportance().open(
        backbone=backbone,
        runtime=runtime,
        config={
            "name": "layer_importance",
            "layers": [0],
            "operations": ["layer"],
            "target_token_types": ["vision"],
            "metrics": ["js"],
            "ranking_metric": "js",
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
                    "layer:vision": {
                        "status": "finite",
                        "decision": {"js": 0.25},
                        "token_groups": {
                            "system": {"js": None},
                            "text": {"js": 0.1},
                            "vision": {"js": 0.4},
                        },
                    }
                }
            },
        }
    )
    session.records.add_skipped(
        {
            "sample_id": "skipped",
            "task_name": "task-all-skipped",
            "reason": "fixture skip",
        }
    )

    record = session.finish()
    session.close()

    assert record.sample_count == 1
    assert record.summary["finite_aggregate_values"] > 0
    assert set(record.artifact_paths) == {"aggregate", "per_sample"}
    artifact = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    ranked = artifact["payload"]["task_rank_aggregation"]["components"]
    assert artifact["payload"]["task_rank_aggregation"]["task_count"] == 1
    assert artifact["payload"]["task_aggregates"]["task-all-skipped"] == {
        "sample_count": 0,
        "layers": {},
        "skipped_sample_count": 1,
    }
    assert ranked["layer:0/layer:vision"]["raw_scores_by_task"] == {
        "task": 0.25
    }
    raw = json.loads(
        Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
    )
    assert raw["payload"]["samples"][0]["sample_id"] == "sample"
    assert raw["payload"]["skipped_samples"][0]["sample_id"] == "skipped"


def test_torch_semantic_interventions_are_reversible_and_exact() -> None:
    torch = pytest.importorskip("torch")

    class AddLayer(torch.nn.Module):
        def forward(self, hidden_states: Any) -> Any:
            return hidden_states + 2.0

    class DoubleFFN(torch.nn.Module):
        def forward(self, hidden_states: Any) -> Any:
            return hidden_states * 2.0

    class Attention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen_mask = None

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any = None,
            attention_mask: Any = None,
        ) -> Any:
            del position_embeddings
            self.seen_mask = attention_mask
            return hidden_states, None

    layout = TokenGroupLayout(
        labels_by_sample=(("image", "text", "text"),),
        evidence={},
    )
    hidden = torch.ones((1, 3, 2))
    layer = AddLayer()
    with HookSession() as hooks:
        install_layer_update_skip(
            hooks,
            layer=layer,
            layout=layout,
            target_group="vision",
        )
        changed = layer(hidden)
    assert torch.equal(changed[:, 0], hidden[:, 0])
    assert torch.equal(changed[:, 1:], hidden[:, 1:] + 2.0)
    assert torch.equal(layer(hidden), hidden + 2.0)

    ffn = DoubleFFN()
    with HookSession() as hooks:
        install_ffn_update_skip(
            hooks,
            ffn=ffn,
            layout=layout,
            target_group="vision",
        )
        changed_update = ffn(hidden)
    assert torch.equal(changed_update[:, 0], torch.zeros_like(hidden[:, 0]))
    assert torch.equal(changed_update[:, 1:], hidden[:, 1:] * 2.0)

    attention = Attention()
    minimum = torch.finfo(torch.float32).min
    causal = torch.full((1, 1, 3, 3), minimum)
    causal[0, 0] = torch.triu(
        torch.full((3, 3), minimum),
        diagonal=1,
    )
    evidence: dict[str, Any] = {}
    with HookSession() as hooks:
        install_attention_edge_block(
            hooks,
            attention=attention,
            layout=layout,
            source_group="vision",
            target_group="text",
            evidence=evidence,
        )
        attention(
            hidden_states=hidden,
            position_embeddings=None,
            attention_mask=causal,
        )
    assert attention.seen_mask[0, 0, 1, 0] == minimum
    assert attention.seen_mask[0, 0, 2, 0] == minimum
    assert evidence["newly_blocked_edge_count"] == 2

    degenerate_attention = Attention()
    degenerate_evidence: dict[str, Any] = {}
    with HookSession() as hooks:
        install_attention_edge_block(
            hooks,
            attention=degenerate_attention,
            layout=layout,
            source_group="vision",
            target_group="vision",
            evidence=degenerate_evidence,
        )
        degenerate_attention(
            hidden_states=hidden,
            position_embeddings=None,
            attention_mask=causal,
        )
    assert degenerate_attention.seen_mask[0, 0, 0, 0] == 0
    assert degenerate_attention.seen_mask[0, 0, 0, 1] == minimum
    assert degenerate_evidence["degenerate_query_head_count"] == 1
    assert degenerate_evidence["newly_blocked_edge_count"] == 0
    probabilities = torch.softmax(degenerate_attention.seen_mask[0, 0, 0], dim=-1)
    assert probabilities[0] == pytest.approx(1.0)
    assert probabilities[1:].sum() == pytest.approx(0.0)

    per_head_attention = Attention()
    with HookSession() as hooks:
        install_attention_edge_block(
            hooks,
            attention=per_head_attention,
            layout=layout,
            source_group="vision",
            target_group="text",
            head_indices=[1],
            num_heads=2,
        )
        per_head_attention(
            hidden_states=hidden,
            position_embeddings=None,
            attention_mask=causal,
        )
    assert tuple(per_head_attention.seen_mask.shape) == (1, 2, 3, 3)
    assert per_head_attention.seen_mask[0, 0, 1, 0] == 0
    assert per_head_attention.seen_mask[0, 1, 1, 0] == minimum

    output_projection = torch.nn.Linear(4, 4, bias=False)
    with torch.no_grad():
        output_projection.weight.copy_(torch.eye(4))
    head_input = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
    with HookSession() as hooks:
        install_attention_head_ablation(
            hooks,
            output_projection=output_projection,
            head_indices=[1],
            num_heads=2,
            head_dim=2,
        )
        head_changed = output_projection(head_input)
    assert torch.equal(head_changed[..., :2], head_input[..., :2])
    assert torch.equal(head_changed[..., 2:], torch.zeros_like(head_changed[..., 2:]))

    class DownProjection(torch.nn.Module):
        in_features = 3

        def __init__(self) -> None:
            super().__init__()
            self.seen = None

        def forward(self, value: object) -> object:
            self.seen = value
            return value

    down = DownProjection()
    neuron_input = torch.ones((1, 2, 3))
    with HookSession() as hooks:
        install_ffn_neuron_ablation(
            hooks,
            down_projection=down,
            neuron_indices=[1],
        )
        neuron_changed = down(neuron_input)
    assert torch.equal(neuron_changed[..., 1], torch.zeros_like(neuron_changed[..., 1]))
    assert torch.equal(neuron_changed[..., [0, 2]], neuron_input[..., [0, 2]])
