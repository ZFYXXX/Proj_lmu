from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.analysis.probes import LateInjectionEarlyExit
from mllm_comp.analysis.probes.late_injection_early_exit import (
    TimingExperiment,
    _aggregate_timing_samples,
    _blocks_layer,
    _reinjection_bundle,
    _timing_experiments,
)


def test_timing_thresholds_preserve_before_and_after_layer_semantics() -> None:
    early = TimingExperiment("early_exit", 1)
    late = TimingExperiment("late_injection", 2)

    assert [_blocks_layer(early, layer) for layer in range(4)] == [
        False,
        False,
        True,
        True,
    ]
    assert [_blocks_layer(late, layer) for layer in range(4)] == [
        True,
        True,
        False,
        False,
    ]
    experiments = _timing_experiments(
        layer_count=4,
        config={"layers": [0, 2]},
    )
    assert [(item.experiment_type, item.threshold) for item in experiments] == [
        ("early_exit", 0),
        ("early_exit", 2),
        ("late_injection", 1),
        ("late_injection", 3),
    ]


def test_timing_probe_accepts_audio_as_explicit_target_modality() -> None:
    LateInjectionEarlyExit().validate_config(
        {"target_modality": "audio", "layers": [0]}
    )

    with pytest.raises(ValueError, match="target_modality"):
        LateInjectionEarlyExit().validate_config({"target_modality": "video"})


def test_timing_aggregation_keeps_experiment_type_and_threshold_axes() -> None:
    aggregate = _aggregate_timing_samples(
        iter(
            [
            {
                "experiments": {
                    "early_exit_0": {
                        "status": "finite",
                        "decision": {"js": 0.2},
                        "token_groups": {"vision": {"js": 0.1}},
                    },
                    "late_injection_1": {
                        "status": "finite",
                        "decision": {"js": 0.4},
                        "token_groups": {"vision": {"js": 0.3}},
                    },
                }
            },
            {
                "experiments": {
                    "early_exit_0": {
                        "status": "finite",
                        "decision": {"js": 0.6},
                        "token_groups": {"vision": {"js": 0.5}},
                    },
                    "late_injection_1": {
                        "status": "finite",
                        "decision": {"js": 0.8},
                        "token_groups": {"vision": {"js": 0.7}},
                    },
                }
            },
            ]
        ),
        metrics=["js"],
    )

    assert aggregate["early_exit"]["0"]["decision"]["js"] == pytest.approx(0.4)
    assert aggregate["late_injection"]["1"]["decision"]["js"] == pytest.approx(0.6)


def _fixture() -> tuple[object, object]:
    def attention() -> object:
        return SimpleNamespace(
            num_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            config=SimpleNamespace(
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=2,
                hidden_size=4,
                _attn_implementation="sdpa",
            ),
        )

    attention0 = attention()
    attention1 = attention()
    layer0 = SimpleNamespace(self_attn=attention0, mlp=object())
    layer1 = SimpleNamespace(self_attn=attention1, mlp=object())
    projectors = [object(), object()]
    handles = {
        "llm": SimpleNamespace(),
        "llm.layer[0]": layer0,
        "llm.layer[0].attn": attention0,
        "llm.layer[0].ffn": layer0.mlp,
        "llm.layer[1]": layer1,
        "llm.layer[1].attn": attention1,
        "llm.layer[1].ffn": layer1.mlp,
        "llm.final_norm": object(),
        "lm_head": object(),
        "vision_deepstack_projectors": projectors,
    }
    backbone = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen3_vl"),
        model_adapter=SimpleNamespace(structural_handles=lambda _model: handles),
    )
    runtime = SimpleNamespace(
        model=object(),
        model_path_or_id="Qwen/Qwen3-VL-2B-Instruct",
        revision="revision",
    )
    return backbone, runtime


def test_timing_probe_writes_explicit_qwen3_injection_contract(
    tmp_path: Path,
) -> None:
    backbone, runtime = _fixture()
    session = LateInjectionEarlyExit().open(
        backbone=backbone,
        runtime=runtime,
        config={
            "name": "late_injection_early_exit",
            "layers": [0],
            "metrics": ["js"],
            "save_raw": True,
        },
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    session.records.add_success(
        {
            "sample_id": "sample",
            "task_name": "task",
            "experiments": {
                "early_exit_0": {
                    "status": "finite",
                    "decision": {"js": 0.2},
                    "token_groups": {
                        "system": {"js": None},
                        "text": {"js": 0.1},
                        "vision": {"js": 0.3},
                    },
                },
                "late_injection_1": {
                    "status": "finite",
                    "decision": {"js": 0.4},
                    "token_groups": {
                        "system": {"js": None},
                        "text": {"js": 0.2},
                        "vision": {"js": 0.5},
                    },
                },
            },
        }
    )

    record = session.finish()
    session.close()

    assert record.summary["experiment_count"] == 2
    assert set(record.artifact_paths) == {"aggregate", "per_sample"}
    artifact = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    contract = artifact["payload"]["injection_contract"]["qwen3_deepstack"]
    assert contract["projector_count"] == 2
    assert contract["mapping"] == "branch_i_is_added_after_decoder_layer_i"
    assert artifact["payload"]["late_injection"]["1"]["decision"]["js"] == 0.4
    raw = json.loads(
        Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
    )
    assert raw["payload"]["samples"][0]["sample_id"] == "sample"


def test_qwen3_reinjection_bundle_adds_only_branches_before_threshold() -> None:
    torch = pytest.importorskip("torch")
    base = torch.ones((1, 2, 3))
    features = (torch.full_like(base, 2.0), torch.full_like(base, 4.0))

    threshold_one = _reinjection_bundle(base, features, threshold=1)
    threshold_two = _reinjection_bundle(base, features, threshold=2)

    assert torch.equal(threshold_one, torch.full_like(base, 3.0))
    assert torch.equal(threshold_two, torch.full_like(base, 7.0))
