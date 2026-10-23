"""The Python and CLI experiment boundary owns output identity and outcomes."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp import experiments
from mllm_comp.analysis.definition import AnalysisOverrides
from mllm_comp.analysis.runtime import _require_fresh_output_paths
from mllm_comp.config import config_to_dict, load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord, BackendRecord, MetricRecord
from mllm_comp.integrations.native_cli import NativeCLIPlan, NativeCLIProcessError


def _config():
    return experiment_config_from_dict({
        "experiment": {"name": "python-experiment"},
        "backbone": {"name": "qwen25_vl", "model_path": "fixture-model"},
        "eval": {"tasks": ["mme"]},
        "train": {"args": {"datasets": ["fixture"]}, "resume": "previous/checkpoint-9"},
        "analysis": {
            "tasks": ["mme"],
            "probes": [{"name": "representation_dynamics", "layers": [0]}],
        },
    }, path="source.yaml")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@pytest.mark.parametrize("task", ["eval", "train", "analyze"])
def test_execution_records_effective_config_before_runtime_and_uses_exact_paths(
    task, tmp_path, monkeypatch,
):
    monkeypatch.chdir(tmp_path)
    config = _config()
    config.output.run_dir = "exact"
    config.output.manifest_path = "reports/custom-manifest.json"
    original = config_to_dict(config)
    config.metadata["unserializable_runtime"] = object()
    observed = []

    def execute(config, *args, **kwargs):
        assert config.output.run_dir == str(tmp_path / "exact")
        running = _read(Path(config.output.run_dir) / "result.json")
        assert running["status"] == "running"
        assert running["returncode"] is None
        assert not Path(config.output.manifest_path).exists()
        snapshot = load_config(Path(config.output.run_dir) / "effective_config.json")
        assert config_to_dict(snapshot) == config_to_dict(config)
        assert snapshot.experiment.run_id
        observed.append(config)
        if task == "analyze":
            _require_fresh_output_paths(config)
            assert config.analysis.probes[0]["layers"] == [0, 1]
            assert args[0].tasks == config.analysis.tasks == ["gqa"]
            return SimpleNamespace(
                records=[AnalysisRecord(
                    name="representation_dynamics", phase="pre_compression",
                    method="representation_dynamics", sample_count=1,
                    summary={"skipped_samples": 0}, artifact_paths={},
                )],
                backend=BackendRecord(name="lmms_eval", version="fixture"),
                sample_count=1, compression_runtime_metadata=None,
            )
        if task == "train":
            assert config.train.resume == "previous/checkpoint-9"
            return SimpleNamespace(
                backend=BackendRecord(name="ms_swift", version="fixture"),
                checkpoint_path="actual/swift-v1/checkpoint-10",
                train_log_path="actual/swift-v1/logging.jsonl", metrics={"step": 10},
            )
        return SimpleNamespace(
            backend=BackendRecord(name="lmms_eval", version="fixture"),
            metrics=[], raw_output_path="native/results.json",
        )

    monkeypatch.setattr(experiments, "execute_analysis", execute)
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        evaluate=execute, train=execute,
    ))
    result = experiments.run_experiment(
        config, task=task,
        analysis_overrides=(AnalysisOverrides(layers="0,1", tasks="gqa") if task == "analyze" else None),
    )
    assert len(observed) == 1
    assert observed[0] is not config
    assert result.status == "succeeded" and result.returncode == 0
    assert result.native_returncode == (None if task == "analyze" else 0)
    assert result.paths.run_dir == str(tmp_path / "exact")
    assert result.paths.manifest == str(tmp_path / "reports/custom-manifest.json")
    assert all(Path(value).is_absolute() for value in result.to_dict()["paths"].values())
    assert _read(result.paths.result) == result.to_dict()
    assert _read(result.paths.manifest)["effective_config"] == _read(result.paths.effective_config)
    assert config_to_dict(config) == original
    assert config.experiment.run_id is None
    if task == "train":
        assert result.output["checkpoint_path"] == "actual/swift-v1/checkpoint-10"


def test_repeated_python_runs_generate_independent_directories(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config()
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        evaluate=lambda **kwargs: SimpleNamespace(
            backend=None, metrics=[], raw_output_path="native/results.json",
        ),
    ))
    results = [experiments.run_experiment(config, task="eval") for _ in range(2)]
    assert all(result.status == "succeeded" for result in results)
    assert results[0].run_id != results[1].run_id
    assert results[0].paths.run_dir != results[1].paths.run_dir
    for result in results:
        assert Path(result.paths.run_dir) == tmp_path / "runs/python-experiment" / result.run_id
        assert Path(result.paths.manifest) == Path(result.paths.run_dir) / "manifest.json"
    assert config.output.run_dir is None and config.output.manifest_path is None


@pytest.mark.parametrize("task", ["eval", "train", "analyze"])
def test_baseline_records_only_executed_stages_and_keeps_idle_recipe(task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    raw = config_to_dict(_config())
    raw["pipeline"] = {"stages": [{
        "name": "idle-fastv", "plugin": "fastv", "algorithm": "fastv",
        "variant": "fastv.image", "binding": "fastv.image.qwen25_vl",
        "placement": "llm_level", "execution_phase": "forward",
        "target_modalities": ["image"], "config": {"prune_layer": -1},
        "schedule": {"keep_ratio": "${UNUSED_KEEP_RATIO}"},
    }]}
    config = experiment_config_from_dict(raw)
    execution = SimpleNamespace(
        backend=None, metrics=[], raw_output_path="native/results.json",
        checkpoint_path="native/checkpoint", train_log_path=None,
        records=[], sample_count=1, compression_runtime_metadata=None,
    )
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        evaluate=lambda **kwargs: execution, train=lambda **kwargs: execution,
    ))
    monkeypatch.setattr(experiments, "execute_analysis", lambda *args, **kwargs: execution)
    result = experiments.run_experiment(config, task=task)
    assert result.status == "succeeded", result.error
    manifest = _read(result.paths.manifest)
    assert manifest["compressor_name"] == "none"
    assert manifest["compression_stages"] == []
    assert manifest["effective_config"]["pipeline"]["stages"][0]["config"]["prune_layer"] == -1
    assert manifest["effective_config"]["pipeline"]["stages"][0]["schedule"]["keep_ratio"] == "${UNUSED_KEEP_RATIO}"


@pytest.mark.parametrize("task", ["eval", "train", "analyze"])
def test_plan_only_is_read_only_and_never_executes(task, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = _config()
    config.experiment.run_id = "chosen-id"
    config.output.run_dir = "preview"
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        build_launch_plan=lambda *args, **kwargs: NativeCLIPlan(["native"], {}, ".", []),
        dependency_report=lambda: {"versions": {"native": "fixture"}},
    ))
    result = experiments.run_experiment(config, task=task, plan_only=True)
    assert result.status == "planned" and result.returncode is None
    assert result.native_returncode is None
    assert ("definition" if task == "analyze" else "plan") in result.output
    snapshot = result.output["effective_config"]
    assert snapshot["output"]["run_dir"] == result.paths.run_dir
    assert snapshot["experiment"]["run_id"] == "chosen-id"
    assert snapshot["analysis" if task == "analyze" else task]["backend"] == (
        "ms_swift" if task == "train" else "lmms_eval"
    )
    assert not (tmp_path / "preview").exists()
    assert result.run_id == config.experiment.run_id == "chosen-id"
    assert result.paths.run_dir == str(tmp_path / "preview")
    assert all(Path(value).is_absolute() for value in result.to_dict()["paths"].values())
    assert config.output.run_dir == "preview"


def test_baseline_plan_keeps_unused_task_and_compression_templates(tmp_path):
    raw = config_to_dict(_config())
    raw["train"]["args"]["datasets"] = ["${TRAIN_DATA}"]
    raw["analysis"]["tasks"] = ["${ANALYSIS_TASK}"]
    raw["data"]["dataset_id"] = "${UNUSED_DATASET}"
    raw["pipeline"]["stages"] = [{
        "name": "fastv", "algorithm": "fastv", "binding": "fastv.image.qwen25_vl",
        "execution_phase": "generate", "config": {"prune_layer": "${PRUNE_LAYER}"},
    }]
    config = experiment_config_from_dict(raw)
    config.output.run_dir = str(tmp_path / "baseline")
    result = experiments.run_experiment(config, task="eval", plan_only=True)
    assert result.status == "planned", result.error
    snapshot = result.output["effective_config"]
    assert snapshot["train"]["args"]["datasets"] == ["${TRAIN_DATA}"]
    assert snapshot["pipeline"]["stages"][0]["config"]["prune_layer"] == "${PRUNE_LAYER}"
    assert not Path(result.paths.run_dir).exists()


@pytest.mark.parametrize("task", ["eval", "train"])
@pytest.mark.parametrize("default_model", [None, "${DEFAULT_MODEL}"])
def test_plan_uses_task_checkpoint_without_requiring_default_model(task, default_model, tmp_path):
    config = _config()
    config.backbone.model_path = default_model
    config.output.run_dir = str(tmp_path / task)
    if task == "eval":
        config.eval.checkpoint_path = "chosen-model"
    else:
        config.train.checkpoint_in = "chosen-model"
        config.train.args["datasets"] = "chosen-dataset"
        config.data.dataset_id = "${UNUSED_DATASET}"
    result = experiments.run_experiment(config, task=task, plan_only=True)
    assert result.status == "planned", result.error
    plan = result.output["plan"]
    assert (plan.checkpoint_path if task == "eval" else plan.model_path) == "chosen-model"
    assert config.backbone.model_path == default_model


@pytest.mark.parametrize("task,field", [
    ("eval", "eval.tasks"), ("train", "train.args.datasets"), ("analyze", "analysis.tasks"),
])
def test_missing_task_data_reports_its_field_before_output_creation(task, field, tmp_path):
    config = _config()
    config.output.run_dir = str(tmp_path / task)
    if task == "train":
        config.train.args.pop("datasets")
    else:
        getattr(config, "analysis" if task == "analyze" else task).tasks = []
    result = experiments.run_experiment(config, task=task)
    assert result.status == "failed"
    assert result.error.stage == "configuration"
    assert field in result.error.message
    assert not Path(config.output.run_dir).exists()


def test_active_compression_placeholder_reports_original_stage_field(tmp_path):
    raw = config_to_dict(_config())
    raw["eval"]["compression_mode"] = "compressed"
    raw["pipeline"]["stages"] = [
        {
            "name": "unused", "algorithm": "fastv", "binding": "fastv.image.qwen25_vl",
            "execution_phase": "generate", "enabled": False,
            "config": {"prune_layer": "${UNUSED_LAYER}"},
        },
        {
            "name": "active", "algorithm": "fastv", "binding": "fastv.image.qwen25_vl",
            "execution_phase": "generate", "config": {"prune_layer": "${PRUNE_LAYER}"},
        },
    ]
    config = experiment_config_from_dict(raw)
    config.output.run_dir = str(tmp_path / "run")
    result = experiments.run_experiment(config, task="eval", plan_only=True)
    assert result.status == "failed"
    assert "pipeline.stages[1].config.prune_layer: ${PRUNE_LAYER}" in result.error.message
    assert "UNUSED_LAYER" not in result.error.message
    assert not Path(config.output.run_dir).exists()


@pytest.mark.parametrize("collision", ["run", "manifest", "reserved", "reserved-parent"])
def test_output_collisions_do_not_execute_or_overwrite(collision, tmp_path, monkeypatch):
    config = _config()
    run_dir = tmp_path / "run"
    manifest = tmp_path / "custom.json"
    config.output.run_dir = str(run_dir)
    if collision == "run":
        run_dir.mkdir()
        (run_dir / "previous.txt").write_text("keep", encoding="utf-8")
    elif collision == "manifest":
        manifest.write_text("keep", encoding="utf-8")
    elif collision == "reserved":
        manifest = run_dir / "result.json"
    else:
        manifest = run_dir / "effective_config.json" / "manifest.json"
    config.output.manifest_path = str(manifest)
    monkeypatch.setattr(experiments, "get_integration", lambda name: pytest.fail("Must not launch"))
    result = experiments.run_experiment(config, task="eval")
    assert result.status == "failed" and result.returncode == 2
    assert result.error.stage == "prepare_outputs"
    assert not (run_dir / "result.json").exists()
    if collision == "run":
        assert (run_dir / "previous.txt").read_text(encoding="utf-8") == "keep"
    elif collision == "manifest":
        assert manifest.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("error", [
    NativeCLIProcessError("lmms_eval", -15),
    BackendIntegrationError("native result missing"),
    RuntimeError("unexpected implementation failure"),
])
def test_execution_failure_preserves_native_code_and_unexpected_traceback(error, tmp_path, monkeypatch):
    config = _config()
    config.output.run_dir = str(tmp_path / "run")

    def execute(**kwargs):
        raise error

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(evaluate=execute))
    if isinstance(error, RuntimeError):
        with pytest.raises(RuntimeError, match="unexpected implementation failure") as raised:
            experiments.run_experiment(config, task="eval")
        assert raised.value is error
    else:
        result = experiments.run_experiment(config, task="eval")
        assert result.status == "failed"
    payload = _read(tmp_path / "run" / "result.json")
    assert payload["status"] == "failed"
    assert payload["error"] == {"type": type(error).__name__, "stage": "execute", "message": str(error)}
    assert payload["native_returncode"] == (-15 if isinstance(error, NativeCLIProcessError) else None)
    expected_code = (
        143 if isinstance(error, NativeCLIProcessError)
        else 1 if isinstance(error, RuntimeError) else 2
    )
    assert payload["returncode"] == expected_code
    assert (tmp_path / "run" / "effective_config.json").is_file()
    assert not (tmp_path / "run" / "manifest.json").exists()


def test_manifest_failure_is_recorded_after_native_success(tmp_path, monkeypatch):
    config = _config()
    config.output.run_dir = str(tmp_path / "run")
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        evaluate=lambda **kwargs: SimpleNamespace(backend=None, metrics=[], raw_output_path="native.json"),
    ))

    def fail(*args, **kwargs):
        raise OSError("artifact write failed")

    monkeypatch.setattr(experiments, "build_run_manifest", fail)
    result = experiments.run_experiment(config, task="eval")
    assert result.status == "failed" and result.returncode == 2
    assert result.native_returncode == 0
    assert result.error.stage == "write_artifacts"
    assert _read(result.paths.result) == result.to_dict()


def test_config_snapshot_omits_provenance_and_round_trips_stages(tmp_path):
    config = load_config("tests/fixtures/qwen25_omni_compressed_train.yaml")
    config.raw["unresolved_provenance"] = "${ORIGINAL_ONLY}"
    config.metadata["runtime_object"] = object()
    payload = config_to_dict(config)
    assert not {"raw", "path", "metadata"} & payload.keys()
    path = tmp_path / "effective_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert config_to_dict(load_config(path)) == payload
    config.pipeline.stages[0].config["new_runtime_value"] = 1
    assert "new_runtime_value" not in payload["pipeline"]["stages"][0]["config"]


@pytest.mark.parametrize("task", ["eval", "analyze"])
def test_reloaded_snapshot_retains_explicit_runtime_device(task, tmp_path, monkeypatch):
    config = _config()
    config.backbone.config["device"] = "cuda:7"
    config.output.run_dir = str(tmp_path / "first")
    devices = []

    def execute(config, *args, device=None, **kwargs):
        devices.append(device)
        assert config.backbone.config["device"] == device
        return SimpleNamespace(
            backend=None, metrics=[], raw_output_path="native.json",
            records=[], sample_count=0, compression_runtime_metadata=None,
        )

    monkeypatch.setattr(experiments, "execute_analysis", execute)
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(evaluate=execute))
    first = experiments.run_experiment(config, task=task, device="cpu")
    assert first.status == "succeeded", first.error
    snapshot = load_config(first.paths.effective_config, overrides={
        "experiment": {"run_id": None},
        "output": {"run_dir": str(tmp_path / "second"), "manifest_path": None},
    })
    second = experiments.run_experiment(snapshot, task=task)
    assert second.status == "succeeded", second.error
    assert devices == ["cpu", "cpu"]
    assert config.backbone.config["device"] == "cuda:7"


@pytest.mark.parametrize("artifact_name", [
    "mllm_comp_lmms_plugin.json", "compression_runtime.json",
])
def test_manifest_does_not_replace_a_new_backend_artifact(artifact_name, tmp_path, monkeypatch):
    config = _config()
    run_dir = tmp_path / "run"
    artifact = run_dir / artifact_name
    config.output.run_dir = str(run_dir)
    config.output.manifest_path = str(artifact)
    native_payload = {"backend_artifact": artifact_name}

    def execute(**kwargs):
        artifact.write_text(json.dumps(native_payload), encoding="utf-8")
        return SimpleNamespace(backend=None, metrics=[], raw_output_path="native.json")

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(evaluate=execute))
    result = experiments.run_experiment(config, task="eval")
    assert result.status == "failed" and result.returncode == 2
    assert result.native_returncode == 0
    assert result.error.stage == "write_artifacts"
    assert result.error.type == "FileExistsError"
    assert str(artifact) in result.error.message
    assert _read(artifact) == native_payload
    assert _read(result.paths.result) == result.to_dict()


def test_recorded_runs_compare_from_another_cwd_with_matching_cli_and_api(tmp_path, monkeypatch, capsys):
    from mllm_comp.cli.main import main
    from mllm_comp.experiment_tools import compare_runs

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(
        evaluate=lambda **kwargs: SimpleNamespace(
            backend=None, metrics=[MetricRecord("mme", "test", {"accuracy": 0.7}, 2)],
            raw_output_path="native/results.json",
        ),
    ))
    runs = []
    for name in ("left", "right"):
        config = _config()
        config.output.run_dir = f"runs/{name}"
        config.output.manifest_path = f"reports/{name}.json"
        result = experiments.run_experiment(config, task="eval")
        assert result.status == "succeeded", result.error
        assert _read(result.paths.effective_config)["output"] == {
            "run_dir": str(tmp_path / "runs" / name),
            "manifest_path": str(tmp_path / "reports" / f"{name}.json"),
        }
        runs.append(result.paths.run_dir)

    elsewhere = tmp_path / "unrelated-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    expected = compare_runs(*runs)
    assert expected["metrics"][0]["absolute_delta"] == 0.0
    assert all(Path(expected[side]["manifest_path"]).is_absolute() for side in ("left", "right"))
    assert main(["compare", *runs, "--output", "comparison.json"]) == 0
    assert json.loads(capsys.readouterr().out) == expected
    assert _read("comparison.json") == expected
