"""Restore actual artifact kinds without reapplying materialized compression."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp import experiments
from mllm_comp.cli.main import main
from mllm_comp.config import config_to_dict, load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import ConfigError
from mllm_comp.core.vocabulary import CompressionMode, ExecutionPhase
from mllm_comp.integrations.native_cli import NativeCLIPlan
from mllm_comp.restoration import restore_config


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _checkpoint(directory: Path, *, operation: bool = False, state: bool = False) -> Path:
    _write(directory / "config.json", {
        "model_type": "qwen2_5_omni_thinker",
        **({"mllm_comp_operations": [{
            "stage_name": "prune", "source_num_hidden_layers": 4,
            "result_num_hidden_layers": 3, "pruned_source_layers": [1],
            "old_to_new_layer_mapping": {"0": 0, "1": None, "2": 1, "3": 2},
        }]} if operation else {}),
    })
    (directory / "model.safetensors").write_bytes(b"weights are loaded by the native runtime")
    if state:
        _write(directory / "trainer_state.json", {"global_step": 2})
    return directory


def _stage(name: str, *, operation: bool = False) -> dict:
    if not operation:
        recipe = load_config(
            Path(__file__).resolve().parents[2] / "configs/train/qwen2_5_omni/train_aud_sft_fastslm_hta.yaml"
        )
        stage = config_to_dict(recipe)["pipeline"]["stages"][0]
        stage["name"] = name
        return stage
    return {
        "name": name, "plugin": "structured_llm_layer_pruning",
        "algorithm": "structured_llm_layer_pruning", "variant": "structured_llm_layer_pruning.audio",
        "placement": "llm_level", "execution_phase": "prepare",
        "method": "operation_compression", "binding": "structured_llm_layer_pruning.audio.qwen25_omni",
        "config": {}, "operation": {"target_site": "llm.layers"},
    }


def _run(directory: Path, checkpoint: Path | None, *, stages=(), task="train", status="succeeded") -> Path:
    config = experiment_config_from_dict({
        "experiment": {"name": "restore", "run_id": "old-run"},
        "backbone": {"name": "qwen25_omni", "model_id": "Qwen/original", "config": {"enable_talker": False}},
        "pipeline": {"stages": list(stages)},
        "train": {
            "backend": "ms_swift", "compression_mode": "compressed" if stages else "none",
            "checkpoint_in": "Qwen/initial-training-model", "resume": "old-state",
            "args": {"datasets": ["fixture"], "tuner_type": "full", "resume_only_model": True},
        },
        "eval": {"tasks": ["mmau_test_mini"], "args": {"diagnostic_trace_path": "old-run/diagnostic.jsonl"}},
        "analysis": {"tasks": ["mmau_test_mini"]},
        "output": {"run_dir": str(directory), "manifest_path": str(directory / "manifest.json")},
    })
    _write(directory / "effective_config.json", config_to_dict(config))
    _write(directory / "result.json", {
        "task": task, "status": status,
        "output": {"checkpoint_path": str(checkpoint)} if checkpoint else {},
    })
    return directory


@pytest.mark.parametrize("task", ["eval", "analyze"])
def test_full_weights_restore_clears_output_identity_without_mutating_snapshot(tmp_path, task):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint)
    before = (run / "effective_config.json").read_bytes()
    config = restore_config(run, task=task)
    assert config.backbone.model_path == str(checkpoint)
    assert config.output.run_dir is None and config.output.manifest_path is None
    assert config.experiment.run_id is None and config.train.resume is None
    assert "diagnostic_trace_path" not in config.eval.args
    assert (run / "effective_config.json").read_bytes() == before
    assert config.path == str(run / "effective_config.json")


def test_thinker_only_checkpoint_does_not_request_absent_talker(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint)
    snapshot = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    snapshot["backbone"]["config"]["enable_talker"] = True
    _write(run / "effective_config.json", snapshot)
    assert restore_config(run, task="eval").backbone.config["enable_talker"] is False


def test_operation_eval_does_not_apply_materialized_stage_again(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint", operation=True)
    hta = _stage("hta")
    hta["depends_on"] = ["prune"]
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True), hta])
    config = restore_config(run, task="eval")
    assert config.pipeline.stages[0].enabled is False
    assert config.pipeline.stages[0].execution_phase is ExecutionPhase.PREPARE
    assert config.pipeline.stages[1].enabled is True
    assert config.pipeline.stages[1].execution_phase is ExecutionPhase.GENERATE
    assert config.pipeline.stages[1].depends_on == []
    assert config.pipeline.stages[1].config["checkpoint_path"] == str(checkpoint)
    assert config.pipeline.stages[1].config["window_tokens"] == 750
    assert config.eval.compression_mode is CompressionMode.COMPRESSED


@pytest.mark.parametrize("recorded", [False, True])
def test_operation_method_can_be_absent_in_an_effective_snapshot(tmp_path, recorded):
    checkpoint = _checkpoint(tmp_path / "checkpoint", operation=recorded)
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True)])
    snapshot = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    snapshot["pipeline"]["stages"][0]["method"] = None
    _write(run / "effective_config.json", snapshot)
    if recorded:
        assert restore_config(run, task="eval").pipeline.stages[0].enabled is False
    else:
        with pytest.raises(ConfigError, match="does not record materialized"):
            restore_config(run, task="eval")


@pytest.mark.parametrize("task", ["train", "analyze"])
def test_restore_selects_forward_for_training_and_analysis(tmp_path, task):
    checkpoint = _checkpoint(tmp_path / "checkpoint", operation=True, state=True)
    hta = _stage("hta")
    hta["execution_phase"] = "generate"
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True), hta])
    config = restore_config(run, task=task)
    assert config.pipeline.stages[0].execution_phase is ExecutionPhase.PREPARE
    assert config.pipeline.stages[1].execution_phase is ExecutionPhase.FORWARD


def test_training_hta_restores_to_a_valid_native_eval_plan(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("hta")])
    config = restore_config(run, task="eval")
    result = experiments.run_experiment(config, task="eval", plan_only=True)
    assert config.pipeline.stages[0].execution_phase is ExecutionPhase.GENERATE
    assert result.status == "planned", result.error


def test_restore_forward_probe_is_explicit_and_keeps_real_binding_limits(tmp_path):
    recipe = load_config(
        Path(__file__).resolve().parents[2] / "configs/eval/lmms_eval/qwen25_vl_imagellm_tome_forward_probe.yaml"
    )
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    _write(checkpoint / "config.json", {"model_type": "qwen2_5_vl"})
    run = _run(tmp_path / "run", checkpoint)
    recipe.backbone.model_path = "Qwen/original"
    recipe.eval.checkpoint_path = "Qwen/original"
    recipe.train.compression_mode = CompressionMode.COMPRESSED
    _write(run / "effective_config.json", config_to_dict(recipe))
    config = restore_config(run, task="eval")
    assert config.pipeline.stages[0].execution_phase is ExecutionPhase.FORWARD
    result = experiments.run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error

    # SAINT's encoder binding supports only forward. Ordinary generation must
    # expose that contract limit instead of silently choosing a probe.
    recipe.eval.args.pop("forward_only_probe")
    snapshot = config_to_dict(recipe)
    snapshot["pipeline"]["stages"] = [{
        "name": "saint", "algorithm": "saint", "binding": "saint.image.qwen25_vl",
        "execution_phase": "forward", "config": {},
    }]
    _write(run / "effective_config.json", snapshot)
    config = restore_config(run, task="eval")
    assert config.pipeline.stages[0].execution_phase is ExecutionPhase.GENERATE
    result = experiments.run_experiment(config, task="eval", plan_only=True)
    assert result.status == "failed"
    assert "phase" in result.error.message


def test_operation_only_restore_keeps_compressed_metric_identity(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint", operation=True)
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True)])
    config = restore_config(run, task="eval")
    assert config.eval.compression_mode is CompressionMode.NONE
    assert config.eval.target.value == "compressed"


def test_legacy_operation_export_metadata_is_read_without_rewriting(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    _write(checkpoint / "operation_compression_manifest.json", {
        "operation_compression": {"stage_name": "prune", "result_num_hidden_layers": 3},
    })
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True)])
    assert restore_config(run, task="eval").pipeline.stages[0].enabled is False


def test_missing_operation_evidence_does_not_guess_whether_to_prune(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint, stages=[_stage("prune", operation=True)])
    with pytest.raises(ConfigError, match="does not record materialized"):
        restore_config(run, task="eval")


def test_training_restore_keeps_original_model_and_modules_for_swift(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint", operation=True, state=True)
    stages = [_stage("prune", operation=True), _stage("hta")]
    run = _run(tmp_path / "run", checkpoint, stages=stages)
    config = restore_config(run, task="train")
    assert config.backbone.model_id == "Qwen/original"
    assert config.backbone.model_path is None
    assert config.train.checkpoint_in == "Qwen/initial-training-model"
    assert config.train.resume == str(checkpoint)
    assert config.train.args["resume_only_model"] is False
    assert all(stage.enabled for stage in config.pipeline.stages)
    assert "checkpoint_path" not in config.pipeline.stages[1].config


def test_hta_module_restore_keeps_backbone_and_exact_position(tmp_path):
    module = tmp_path / "hta.pt"
    module.write_bytes(b"native HTA loader owns state-dict validation")
    stage = _stage("hta")
    stage["operation"]["target_site"] = "audio_projector_output_post_projector"
    run = _run(tmp_path / "run", None, stages=[stage])
    config = restore_config(run, task="eval", checkpoint=module)
    restored = config.pipeline.stages[0]
    assert config.backbone.model_id == "Qwen/original"
    assert config.backbone.model_path == "Qwen/initial-training-model"
    assert config.eval.checkpoint_path == "Qwen/initial-training-model"
    assert restored.config["checkpoint_path"] == str(module)
    assert restored.operation == stage["operation"]
    assert restored.config["stage_query_tokens"] == 80
    assert restored.config["checkpoint_strict"] is True
    with pytest.raises(ConfigError, match="native Swift training state"):
        restore_config(run, task="train", checkpoint=module)


def test_single_hta_file_does_not_fill_multiple_different_module_slots(tmp_path):
    module = tmp_path / "hta.safetensors"
    module.write_bytes(b"one module")
    run = _run(tmp_path / "run", None, stages=[_stage("first"), _stage("second")])
    with pytest.raises(ConfigError, match="exactly one"):
        restore_config(run, task="eval", checkpoint=module)


@pytest.mark.parametrize("source_task,task,expected", [
    ("train", "analyze", "Qwen/initial-training-model"),
    ("eval", "analyze", "Qwen/evaluated-model"),
    ("analyze", "eval", "Qwen/original"),
])
def test_hta_module_uses_source_tasks_selected_model(tmp_path, source_task, task, expected):
    module = tmp_path / "hta.pt"
    module.write_bytes(b"module")
    run = _run(tmp_path / "run", None, stages=[_stage("hta")], task=source_task)
    snapshot = json.loads((run / "effective_config.json").read_text(encoding="utf-8"))
    snapshot["eval"]["checkpoint_path"] = "Qwen/evaluated-model"
    snapshot["analysis" if source_task == "analyze" else source_task]["compression_mode"] = "compressed"
    _write(run / "effective_config.json", snapshot)
    config = restore_config(run, task=task, checkpoint=module)
    assert config.backbone.model_path == expected
    if task == "eval":
        assert config.eval.checkpoint_path == expected


@pytest.mark.parametrize("source_task,status", [("eval", "succeeded"), ("train", "failed")])
def test_no_produced_checkpoint_never_falls_back_to_input_model(tmp_path, source_task, status):
    run = _run(tmp_path / "run", None, task=source_task, status=status)
    _write(run / "manifest.json", {"checkpoint_path": "Qwen/original"})
    with pytest.raises(ConfigError, match="did not record a produced"):
        restore_config(run, task="eval")


def test_recorded_checkpoint_can_survive_a_later_artifact_write_failure(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint, status="failed")
    assert restore_config(run, task="eval").backbone.model_path == str(checkpoint)


def test_selected_norm_diagnostic_is_not_a_peft_or_full_checkpoint(tmp_path):
    checkpoint = tmp_path / "selected-norm"
    _write(checkpoint / "adapter_config.json", {"format": "mllm_comp_selected_parameter_checkpoint"})
    run = _run(tmp_path / "run", checkpoint)
    for task in ("train", "eval"):
        with pytest.raises(ConfigError, match="diagnostic final-norm"):
            restore_config(run, task=task)


def test_peft_only_supports_native_training_resume(tmp_path):
    checkpoint = tmp_path / "adapter"
    _write(checkpoint / "adapter_config.json", {"peft_type": "LORA"})
    _write(checkpoint / "trainer_state.json", {"global_step": 2})
    (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
    run = _run(tmp_path / "run", checkpoint)
    assert restore_config(run, task="train").train.resume == str(checkpoint)
    with pytest.raises(ConfigError, match="do not load or merge PEFT"):
        restore_config(run, task="eval")


def test_missing_model_shards_and_missing_training_state_are_explicit(tmp_path):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "run", checkpoint)
    with pytest.raises(ConfigError, match="trainer_state.json"):
        restore_config(run, task="train")
    (checkpoint / "model.safetensors").unlink()
    _write(checkpoint / "model.safetensors.index.json", {"weight_map": {"weight": "shard.safetensors"}})
    with pytest.raises(ConfigError, match="missing weight shards"):
        restore_config(run, task="eval")


def test_restore_cli_and_python_use_same_config_and_new_run_boundary(tmp_path, monkeypatch, capsys):
    checkpoint = _checkpoint(tmp_path / "checkpoint")
    run = _run(tmp_path / "old", checkpoint)
    observed = []

    def plan(config, **kwargs):
        observed.append(deepcopy(config_to_dict(config)))
        return NativeCLIPlan(["native"], {}, str(tmp_path), [])

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        build_launch_plan=plan, dependency_report=lambda: {},
    ))
    restored = restore_config(run, task="eval")
    restored.output.run_dir = str(tmp_path / "new")
    restored.experiment.run_id = "next"
    expected = experiments.run_experiment(restored, task="eval", plan_only=True)
    assert expected.status == "planned", expected.error
    assert main([
        "eval", "--from_run", str(run), "--output_dir", str(tmp_path / "new"),
        "--run_id", "next", "--plan_only",
    ]) == 0
    assert observed[0] == observed[1]
    assert json.loads(capsys.readouterr().out) == expected.to_dict()
    assert not (tmp_path / "new").exists()


def test_restore_cli_checkpoint_selects_hta_weights_without_replacing_base(tmp_path, monkeypatch, capsys):
    from mllm_comp.cli import _common

    module = tmp_path / "hta.pt"
    module.write_bytes(b"module")
    run = _run(tmp_path / "run", None, stages=[_stage("hta")])
    observed = []

    def execute(config, **kwargs):
        observed.append(config)
        return experiments.ExperimentResult(task="eval")

    monkeypatch.setattr(_common, "run_experiment", execute)
    assert main(["eval", "--from_run", str(run), "--checkpoint", str(module), "--sample_limit", "1"]) == 0
    assert observed[0].backbone.model_path == "Qwen/initial-training-model"
    assert observed[0].pipeline.stages[0].config["checkpoint_path"] == str(module)
    assert observed[0].eval.sample_limit == 1


def test_restore_cli_error_is_structured_and_config_sources_are_exclusive(tmp_path, capsys):
    run = _run(tmp_path / "run", None, status="failed")
    assert main(["eval", "--from_run", str(run)]) == 2
    assert json.loads(capsys.readouterr().out)["error"]["stage"] == "configuration"
    with pytest.raises(SystemExit):
        main(["eval", "--from_run", str(run), "--config", "recipe.yaml"])
