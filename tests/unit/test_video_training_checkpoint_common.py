from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.diagnostics import video_training_checkpoint_common as target
from scripts.diagnostics.video_method_checks import dycoke_forward_runtime_checks
from tests.fixtures.video_reports import dycoke_training_metadata


class _Parameter:
    ndim = 1

    def __init__(self) -> None:
        self.requires_grad = False

    def is_floating_point(self) -> bool:
        return True

    def requires_grad_(self, value: bool):
        self.requires_grad = value
        return self


def test_select_final_norm_weight_freezes_every_other_parameter() -> None:
    chosen = _Parameter()
    other = _Parameter()
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            (("model.language_model.norm.weight", chosen), ("other", other))
        )
    )
    facade = SimpleNamespace(
        model_adapter=SimpleNamespace(
            structural_handles=lambda _model: {
                "llm.final_norm": SimpleNamespace(weight=chosen)
            }
        )
    )

    selected = target.select_final_norm_weight(facade, model)

    assert selected == [("model.language_model.norm.weight", chosen)]
    assert chosen.requires_grad is True
    assert other.requires_grad is False


def test_target_parameter_checks_require_one_nonzero_updated_target() -> None:
    checks = target.target_parameter_checks(
        [
            {
                "gradient_present": True,
                "gradient_finite": True,
                "gradient_nonzero": True,
                "max_parameter_delta": 0.015625,
            }
        ]
    )

    assert all(checks.values())
    assert target.target_parameter_checks([])["exact_single_intended_target"] is False


def test_append_jsonl_persists_before_later_gate_failure(tmp_path: Path) -> None:
    path = tmp_path / "progress.jsonl"
    target.append_jsonl(path, {"step": 1, "checks": {"passed": False}})

    assert '"step": 1' in path.read_text(encoding="utf-8")
    assert '"passed": false' in path.read_text(encoding="utf-8")


def _passing_step(step: int) -> dict[str, object]:
    return {"step": step, "checks": {"passed": True}}


def test_validate_step_records_requires_exact_five_contiguous_passes() -> None:
    target.validate_step_records(
        [_passing_step(step) for step in range(1, target.OPTIMIZER_STEPS + 1)]
    )

    with pytest.raises(RuntimeError, match="exactly 5"):
        target.validate_step_records([_passing_step(1)])
    with pytest.raises(RuntimeError, match="not contiguous"):
        target.validate_step_records([_passing_step(step) for step in (1, 2, 4, 4, 5)])
    with pytest.raises(RuntimeError, match="failed exact checks"):
        target.validate_step_records(
            [
                *[_passing_step(step) for step in range(1, 5)],
                {"step": 5, "checks": {"passed": False}},
            ]
        )


def test_validate_step_records_accepts_amended_three_step_cells() -> None:
    records = [_passing_step(step) for step in range(1, 4)]

    target.validate_step_records(records, optimizer_steps=3)

    with pytest.raises(RuntimeError, match="exactly 3"):
        target.validate_step_records(records[:2], optimizer_steps=3)
    with pytest.raises(ValueError, match="positive"):
        target.validate_step_records([], optimizer_steps=0)


# These are the existing admitted cells, not a generated support matrix.
# Estimates are ordered LLaVA / Qwen2.5-VL / Qwen3-VL.
_TRAINING_CELLS = (
    ("dycoke", (32768, 32768, 32768)),
    ("dytok", (49152, 49152, 32768)),
    ("fastvid", (49152, 49152, 49152)),
    ("metok", (32768, 32768, 32768)),
    ("prunevid", (49152, 32768, 32768)),
    ("visionzip", (49152, 49152, 49152)),
    ("vqtoken", (49152, 32768, 32768)),
)


@pytest.mark.parametrize(
    "algorithm,backbone,estimate",
    [
        pytest.param(algorithm, backbone, estimate, id=f"{algorithm}-{backbone}")
        for algorithm, estimates in _TRAINING_CELLS
        for backbone, estimate in zip(("llava", "qwen25", "qwen3"), estimates)
    ],
)
def test_training_command_keeps_exact_stage_and_admission_defaults(
    algorithm, backbone, estimate
) -> None:
    command = importlib.import_module(
        f"scripts.diagnostics.validate_{backbone}_{algorithm}_video_training_checkpoint"
    )
    arguments = [
        "--checkpoint", "checkpoint", "--nextqa_parquet", "test.parquet",
        "--video_root", "videos", "--output_dir", "output",
        "--code_revision", "revision", "--physical_gpu", "1",
    ]
    if algorithm == "dytok":
        arguments.extend(("--detector_checkpoint", "detector", "--detector_siglip", "siglip"))
    elif algorithm == "metok" or (algorithm == "fastvid" and backbone == "llava"):
        arguments.extend(("--siglip_checkpoint", "siglip"))
    args = command._parser().parse_args(arguments)
    assert args.estimated_peak_mib == estimate
    assert args.safety_margin_mib == 16384
    assert args.learning_rate == 0.01
    assert args.seed == 42
    assert args.checkpoint == Path("checkpoint")
    assert args.physical_gpu == 1

    stage = command._stage()
    model = "llava_onevision" if backbone == "llava" else f"{backbone}_vl"
    assert stage.algorithm == algorithm
    assert stage.variant == f"{algorithm}.video"
    assert stage.binding == f"{algorithm}.video.{model}"
    assert stage.execution_phase.value == "forward"
    assert {item.value for item in stage.target_modalities} == {"video"}
    expected_conditioning = {"text"} if algorithm in {"dycoke", "dytok", "metok", "prunevid"} else set()
    assert {item.value for item in stage.conditioning_modalities} == expected_conditioning
    assert stage.config == {}


def test_canonical_layout_removes_only_volatile_provenance() -> None:
    value = {
        "invocation_id": "run-1",
        "source_count": 64,
        "nested": {"event_index": 2, "result_count": 32},
    }
    assert target.canonical(value) == {
        "nested": {"result_count": 32}, "source_count": 64,
    }


@pytest.mark.parametrize("resolved", [{"retain": 4}, {"retain": 5}])
def test_training_activation_keeps_binding_forward_phase_and_canonical_gate(
    monkeypatch, resolved
) -> None:
    calls = []
    activation = SimpleNamespace(runtime_metadata=lambda: {"canonical_config": resolved})
    binding = SimpleNamespace(executor_factory=lambda **kwargs: calls.append(kwargs) or activation)
    bindings = []
    registry = SimpleNamespace(binding=lambda name: bindings.append(name) or binding)
    monkeypatch.setattr(target, "builtin_multimodal_compression_contracts", lambda: registry)
    stage = SimpleNamespace(binding="custom.video.qwen3_vl", algorithm="custom")
    arguments = dict(facade=object(), model=object(), runtime=object(), stage=stage)
    if resolved == {"retain": 4}:
        assert target.training_activation(**arguments, canonical_config={"retain": 4}) is activation
    else:
        with pytest.raises(RuntimeError, match="differs from canonical"):
            target.training_activation(**arguments, canonical_config={"retain": 4})
    assert bindings == [stage.binding]
    assert all(calls[0][key] is value for key, value in arguments.items())
    assert calls[0]["binding"] is binding
    assert calls[0]["binding_config"].config == {}
    assert calls[0]["execution_phase"].value == "forward"


def test_lightweight_checkpoint_roundtrip_preserves_scope_and_exact_target(tmp_path) -> None:
    import json

    torch = pytest.importorskip("torch")
    pytest.importorskip("safetensors.torch")
    model = torch.nn.LayerNorm(3)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([0.5, 1.5, 2.0]))
    scope = {
        "algorithm": "vqtoken", "variant": "vqtoken.video",
        "backbone": "llava_onevision", "binding": "vqtoken.video.llava_onevision",
    }
    saved = target.save_lightweight_checkpoint(
        tmp_path / "checkpoint-5", selected=[("weight", model.weight)], scope=scope,
        canonical_config={"num_clusters": 32}, base_checkpoint=tmp_path / "base",
        code_revision="fixture-revision", optimizer={"name": "AdamW", "lr": 0.01}, torch=torch,
    )
    config = json.loads(Path(saved["config_path"]).read_text(encoding="utf-8"))
    state = json.loads(Path(saved["training_state_path"]).read_text(encoding="utf-8"))
    assert all(config[key] == value for key, value in scope.items())
    assert config["format"] == "mllm_comp_selected_parameter_checkpoint"
    assert config["trainable_parameter_names"] == ["weight"]
    assert state["global_step"] == 5 and state["fixture_doc_id"] == 442
    fresh = torch.nn.LayerNorm(3)
    result = target.load_lightweight_checkpoint(fresh, saved, torch=torch)
    assert result["all_exact"] and result["parameter_records"][0]["exact_reload"]
    assert torch.equal(fresh.weight, model.weight)
    with pytest.raises(RuntimeError, match="shape changed"):
        target.load_lightweight_checkpoint(torch.nn.LayerNorm(4), saved, torch=torch)


@pytest.mark.parametrize(
    "method,cache,expected",
    [
        (
            "metok",
            {
                "policy": "compact_prefix",
                "boundary": "before_layer",
                "boundary_layer": 3,
                "source_length": 237,
                "result_length": 195,
                "touched_layers": [0, 1, 2],
                "cache_type": "DynamicCache",
            },
            {"boundary_layer": 3, "source_length": 237, "result_length": 195},
        ),
        (
            "prunevid",
            {
                "policy": "compact_prefix",
                "source_length": 390,
                "result_length": 240,
                "touched_layers": list(range(11)),
            },
            {"source_length": 390, "result_length": 240},
        ),
    ],
)
def test_training_commands_reject_inference_cache_claim(method, cache, expected) -> None:
    for backbone in ("qwen25", "qwen3"):
        command = importlib.import_module(
            f"scripts.diagnostics.validate_{backbone}_{method}_video_training_checkpoint"
        )
        assert command._training_cache_exact is target.training_cache_exact
    assert not target.training_cache_exact(cache, **expected)


def test_dycoke_forward_runtime_forbids_generation_cache_controller() -> None:
    for backbone in ("llava", "qwen25"):
        command = importlib.import_module(
            f"scripts.diagnostics.validate_{backbone}_dycoke_video_training_checkpoint"
        )
        assert command._forward_runtime_checks is dycoke_forward_runtime_checks
    checks = dycoke_forward_runtime_checks(dycoke_training_metadata())
    assert all(checks.values())
    assert checks["decode_view_not_invoked"]
    assert checks["decode_cache_controller_not_installed"]


@pytest.mark.parametrize(
    "field,value,failed_check",
    [
        ("execution_phase", "generate", "forward_phase_exact"),
        ("decode_view_invocations", 1, "decode_view_not_invoked"),
        ("live_cache_controllers", 1, "decode_cache_controller_not_installed"),
    ],
)
def test_dycoke_forward_runtime_rejects_generation_state(field, value, failed_check) -> None:
    metadata = dycoke_training_metadata()
    metadata[field] = value
    assert dycoke_forward_runtime_checks(metadata)[failed_check] is False
