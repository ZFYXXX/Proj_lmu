from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.analysis.common.attention_flow import (
    aggregate_attention_flow_samples,
    attention_head_metrics,
    compute_attention_flow_layer,
)
from mllm_comp.analysis.bindings import DecoderLayerBinding, layer_batches
from mllm_comp.analysis.common.attention_weights import AttentionWeightCapture
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes import AttentionFlow
from mllm_comp.core.errors import BackendIntegrationError


def test_attention_head_metrics_conserve_direct_modality_mass() -> None:
    metrics = attention_head_metrics(
        [
            [1.0, 0.0, 0.0],
            [0.25, 0.75, 0.0],
            [0.2, 0.3, 0.5],
        ],
        labels=["image", "text", "text"],
        decision_position=2,
    )

    assert metrics["vision_to_vision_mass"] == pytest.approx(1.0)
    assert metrics["text_to_vision_mass"] == pytest.approx(0.225)
    assert metrics["decision_to_vision_mass"] == pytest.approx(0.2)
    assert metrics["decision_to_text_mass"] == pytest.approx(0.8)
    assert metrics["decision_to_system_mass"] is None
    assert metrics["decision_to_vision_entropy"] == pytest.approx(0.0)
    assert metrics["decision_to_vision_entropy_norm"] == pytest.approx(0.0)

    with pytest.raises(ValueError, match="not conserved"):
        attention_head_metrics(
            [[0.5, 0.0], [0.0, 1.0]],
            labels=["image", "text"],
            decision_position=1,
        )


def test_attention_entropy_excludes_queries_that_cannot_reach_vision() -> None:
    metrics = attention_head_metrics(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0, 0.0],
            [0.2, 0.2, 0.6, 0.0, 0.0],
            [0.1, 0.1, 0.4, 0.4, 0.0],
            [0.0, 0.0, 0.5, 0.5, 0.0],
        ],
        labels=["system", "text", "image", "image", "text"],
        decision_position=4,
    )

    assert metrics["text_to_vision_entropy"] == pytest.approx(math.log(2.0))
    assert metrics["text_to_vision_entropy_norm"] == pytest.approx(1.0)

    unreachable = attention_head_metrics(
        [[1.0, 0.0], [0.5, 0.5]],
        labels=["text", "image"],
        decision_position=0,
    )
    assert unreachable["text_to_vision_entropy"] is None
    assert unreachable["text_to_vision_entropy_norm"] is None


def test_attention_flow_emits_audio_routing_and_entropy_metrics() -> None:
    metrics = attention_head_metrics(
        [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [0.2, 0.3, 0.5],
        ],
        labels=["audio", "audio", "text"],
        decision_position=2,
        groups=["text", "audio"],
    )

    assert metrics["audio_to_audio_mass"] == pytest.approx(1.0)
    assert metrics["text_to_audio_mass"] == pytest.approx(0.5)
    assert metrics["decision_to_audio_mass"] == pytest.approx(0.5)
    assert metrics["decision_to_audio_entropy"] == pytest.approx(
        -(0.4 * math.log(0.4) + 0.6 * math.log(0.6))
    )
    assert "decision_to_vision_mass" not in metrics
    AttentionFlow().validate_config({"token_groups": ["text", "audio"]})


def test_attention_mass_tolerance_accounts_for_weight_dtype_precision() -> None:
    torch = pytest.importorskip("torch")
    layout = TokenGroupLayout(
        labels_by_sample=(("image", "text"),),
        evidence={},
    )
    quantized = torch.tensor(
        [[[[0.5, 0.49609375], [0.5, 0.49609375]]]],
        dtype=torch.bfloat16,
    )

    report = compute_attention_flow_layer(
        quantized,
        layout=layout,
        decision_position=1,
        mass_tolerance=2e-3,
    )

    evidence = report["mass_conservation"]
    assert evidence["max_abs_row_sum_error"] == pytest.approx(0.00390625)
    assert evidence["configured_tolerance"] == pytest.approx(0.002)
    assert evidence["tolerance"] == pytest.approx(torch.finfo(torch.bfloat16).eps)
    assert evidence["weight_dtype"] == "bfloat16"

    invalid = torch.tensor(
        [[[[0.5, 0.4765625], [0.5, 0.4765625]]]],
        dtype=torch.bfloat16,
    )
    with pytest.raises(BackendIntegrationError, match="not conserved"):
        compute_attention_flow_layer(
            invalid,
            layout=layout,
            decision_position=1,
            mass_tolerance=2e-3,
        )


def test_attention_flow_aggregation_preserves_heads_and_layer_mean() -> None:
    aggregate = aggregate_attention_flow_samples(
        iter(
            [
            {
                "layers": {
                    "0": {
                        "mean": {"decision_to_vision_mass": 0.2},
                        "mass_conservation": {
                            "passed": True,
                            "max_abs_row_sum_error": 0.003,
                            "tolerance": 0.0078125,
                            "configured_tolerance": 0.002,
                            "dtype_epsilon": 0.0078125,
                            "weight_dtype": "bfloat16",
                            "valid_query_count": 5,
                        },
                        "heads": {
                            "0": {"decision_to_vision_mass": 0.1},
                            "1": {"decision_to_vision_mass": 0.3},
                        },
                    }
                }
            },
            {
                "layers": {
                    "0": {
                        "mean": {"decision_to_vision_mass": 0.4},
                        "mass_conservation": {
                            "passed": True,
                            "max_abs_row_sum_error": 0.004,
                            "tolerance": 0.0078125,
                            "configured_tolerance": 0.002,
                            "dtype_epsilon": 0.0078125,
                            "weight_dtype": "bfloat16",
                            "valid_query_count": 7,
                        },
                        "heads": {
                            "0": {"decision_to_vision_mass": 0.2},
                            "1": {"decision_to_vision_mass": 0.6},
                        },
                    }
                }
            },
            ]
        ),
        metrics=["decision_to_vision_mass"],
    )

    assert aggregate["layers"]["0"]["mean"][
        "decision_to_vision_mass"
    ] == pytest.approx(0.3)
    assert aggregate["layers"]["0"]["heads"]["1"][
        "decision_to_vision_mass"
    ] == pytest.approx(0.45)
    assert aggregate["layers"]["0"]["mass_conservation"] == {
        "contract": "dtype_aware_valid_query_row_mass_v1",
        "passed": True,
        "sample_count": 2,
        "max_abs_row_sum_error": pytest.approx(0.004),
        "configured_tolerance": pytest.approx(0.002),
        "weight_dtypes": ["bfloat16"],
        "dtype_epsilon_by_weight_dtype": {
            "bfloat16": pytest.approx(0.0078125),
        },
        "effective_tolerance_by_weight_dtype": {
            "bfloat16": pytest.approx(0.0078125),
        },
        "valid_query_count_min": 5,
        "valid_query_count_max": 7,
    }


def test_attention_flow_aggregation_requires_mass_conservation_evidence() -> None:
    with pytest.raises(
        BackendIntegrationError,
        match="requires passed mass-conservation evidence",
    ):
        aggregate_attention_flow_samples(
            [
                {
                    "layers": {
                        "0": {
                            "mean": {"decision_to_vision_mass": 0.2},
                            "heads": {},
                        }
                    }
                }
            ],
            metrics=["decision_to_vision_mass"],
        )


def test_attention_flow_rejects_metric_without_required_token_groups() -> None:
    with pytest.raises(ValueError, match="were not selected"):
        AttentionFlow().validate_config(
            {
                "metrics": ["decision_to_vision_mass"],
                "token_groups": ["text"],
            }
        )


@pytest.mark.parametrize(
    "config",
    [
        {"metrics": "decision_to_vision_mass"},
        {"metrics": ["decision_to_vision_mass"], "layers": [True]},
        {"metrics": ["decision_to_vision_mass"], "save_raw": "false"},
    ],
)
def test_attention_flow_rejects_ambiguous_config_types(
    config: dict[str, object],
) -> None:
    with pytest.raises(ValueError):
        AttentionFlow().validate_config(config)


def test_attention_flow_defaults_to_single_layer_capture_batches() -> None:
    layers = tuple(
        DecoderLayerBinding(index, object(), object(), object())
        for index in range(3)
    )

    batches = layer_batches(layers, 1)

    assert [[layer.index for layer in batch] for batch in batches] == [
        [0],
        [1],
        [2],
    ]


def test_semantic_attention_hook_captures_weights_when_outer_model_drops_them() -> None:
    class Handle:
        def __init__(self, module: object) -> None:
            self.module = module

        def remove(self) -> None:
            self.module.callback = None

    class Weight:
        shape = (1, 2, 3, 3)

        def detach(self) -> "Weight":
            return self

    class Attention:
        callback = None

        def register_forward_hook(self, callback: object) -> Handle:
            self.callback = callback
            return Handle(self)

        def forward(self) -> object:
            output = (object(), Weight())
            if self.callback is not None:
                self.callback(self, (), output)
            return output[0]  # provider's outer decoder intentionally drops weights

    attention = Attention()
    layer = DecoderLayerBinding(0, object(), attention, object())
    capture = AttentionWeightCapture([layer])
    with HookSession() as hooks:
        capture.install(hooks)
        outer_output = attention.forward()
        values = capture.finish()

    assert outer_output is not values[0]
    assert values[0].shape == (1, 2, 3, 3)
    assert attention.callback is None


def test_semantic_attention_hook_fails_closed_when_module_omits_weights() -> None:
    class Handle:
        def __init__(self, module: object) -> None:
            self.module = module

        def remove(self) -> None:
            self.module.callback = None

    class Attention:
        callback = None

        def register_forward_hook(self, callback: object) -> Handle:
            self.callback = callback
            return Handle(self)

        def forward(self) -> object:
            output = (object(), None)
            if self.callback is not None:
                self.callback(self, (), output)
            return output[0]

    attention = Attention()
    capture = AttentionWeightCapture(
        [DecoderLayerBinding(0, object(), attention, object())]
    )
    with HookSession() as hooks:
        capture.install(hooks)
        attention.forward()
        with pytest.raises(BackendIntegrationError, match="no explicit weight tensor"):
            capture.finish()

    assert attention.callback is None


def _fixture_backbone(attention_backend: str = "eager") -> tuple[object, object]:
    attention = SimpleNamespace(
        num_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        config=SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            hidden_size=4,
            _attn_implementation=attention_backend,
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
    return backbone, runtime


def test_attention_flow_requires_eager_weights() -> None:
    backbone, runtime = _fixture_backbone("sdpa")

    with pytest.raises(BackendIntegrationError, match="requires exact eager"):
        AttentionFlow().open(
            backbone=backbone,
            runtime=runtime,
            config={"name": "attention_flow", "layers": [0]},
            phase="pre_compression",
            output_dir="unused",
        )


@pytest.mark.parametrize("save_raw", [False, True])
def test_attention_flow_default_aggregate_keeps_capture_evidence(
    tmp_path: Path,
    save_raw: bool,
) -> None:
    backbone, runtime = _fixture_backbone()
    config = {
        "name": "attention_flow",
        "layers": [0],
        "metrics": ["decision_to_vision_mass"],
        "token_groups": ["system", "text", "vision"],
    }
    if save_raw:
        config["save_raw"] = True
    session = AttentionFlow().open(
        backbone=backbone,
        runtime=runtime,
        config=config,
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    session._record_success(
        {
            "sample_id": "sample",
            "task_name": "task",
            "capture_execution": {
                "contract": "bounded_selected_layer_batches_v1",
                "layer_batch_size": 1,
                "capture_pass_count": 1,
                "selected_layer_count": 1,
            },
            "layers": {
                "0": {
                    "mean": {"decision_to_vision_mass": 0.3},
                    "mass_conservation": {
                        "passed": True,
                        "max_abs_row_sum_error": 0.0001,
                        "tolerance": 0.002,
                        "configured_tolerance": 0.002,
                        "dtype_epsilon": 1.1920928955078125e-07,
                        "weight_dtype": "float32",
                        "valid_query_count": 3,
                    },
                    "heads": {
                        "0": {"decision_to_vision_mass": 0.1},
                        "1": {"decision_to_vision_mass": 0.5},
                    },
                }
            },
        }
    )

    record = session.finish()
    session.close()

    assert record.summary["routing_semantics"] == "direct_attention_mass_no_rollout_v1"
    assert set(record.artifact_paths) == (
        {"aggregate", "per_sample"} if save_raw else {"aggregate"}
    )
    artifact = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    assert artifact["payload"]["layers"]["0"]["heads"]["1"][
        "decision_to_vision_mass"
    ] == 0.5
    assert artifact["payload"]["capture_execution"] == {
        "contract": "bounded_selected_layer_batches_v1",
        "layer_batch_size": 1,
        "capture_pass_count": 1,
        "selected_layer_count": 1,
        "successful_sample_count": 1,
    }
    assert artifact["payload"]["routing_semantics"] == (
        "direct_attention_mass_no_rollout_v1"
    )
    assert artifact["payload"]["layers"]["0"]["mass_conservation"] == {
        "contract": "dtype_aware_valid_query_row_mass_v1",
        "passed": True,
        "sample_count": 1,
        "max_abs_row_sum_error": pytest.approx(0.0001),
        "configured_tolerance": pytest.approx(0.002),
        "weight_dtypes": ["float32"],
        "dtype_epsilon_by_weight_dtype": {
            "float32": pytest.approx(1.1920928955078125e-07),
        },
        "effective_tolerance_by_weight_dtype": {
            "float32": pytest.approx(0.002),
        },
        "valid_query_count_min": 3,
        "valid_query_count_max": 3,
    }
    if save_raw:
        raw = json.loads(
            Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
        )
        assert raw["payload"]["samples"][0]["sample_id"] == "sample"
