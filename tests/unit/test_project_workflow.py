import json
from pathlib import Path
import sys

import pytest

from mllm_comp.workflow.catalog import DEFAULT_CATALOG, load_catalog, method_options
from mllm_comp.workflow import runner
from scripts.eval.lmms_eval.qwen25_portable_queue import select_experiment


def settings(tmp_path, mapping=None, reference_only=True):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps({"schema_version": 1, "python": sys.executable,
        "data_root": str(tmp_path / "data"), "model": str(tmp_path / "model"),
        "checkpoint_root": str(tmp_path / "heads"), "output_root": str(tmp_path / "runs"),
        "cache_dir": str(tmp_path / "cache"), "gpus": ["0"],
        "slurm": {"job_name": "28-workflow", "account": "project", "partition": "accelerated",
                  "gpus": 4, "cpus": 32, "time": "23:50:00"}}))
    experiment = tmp_path / "experiment.json"
    experiment.write_text(json.dumps({"schema_version": 1, "run_id": "example", "reference_only": reference_only,
        "method_tasks": mapping or {"importance_fps": ["ai2d"], "importance_fps_pool": ["mmstar"]}}))
    return profile, experiment


def test_sparse_selection_and_custom_alias_need_no_python_edit(tmp_path):
    catalog = load_catalog()
    catalog["new_pool_recipe"] = {**catalog["importance_fps_pool"], "options": {
        **catalog["importance_fps_pool"]["options"], "spatial_beta": .2}}
    path = tmp_path / "methods.json"
    path.write_text(json.dumps({"schema_version": 1, "methods": catalog}))
    profile, experiment = settings(tmp_path, {"new_pool_recipe": ["ai2d", "mmstar"]})
    config = runner.configuration(profile, experiment, catalog_path=path, tasks=["mmstar"])
    assert config["requested_cells"] == [["new_pool_recipe", "mmstar"]]
    assert config["checkpoints"]["new_pool_recipe"].endswith("/heads/checkpoints/old_mlp")
    assert select_experiment(experiment, tasks=["mmstar"], catalog_path=path)[2] == config["requested_cells"]
    spec, options = method_options("new_pool_recipe", method_checkpoint="/mlp", scorer_checkpoint="/scorer",
                                   identity_path="/identity", catalog_path=path)
    assert spec["algorithm"] == "importance_anchor_adapter"
    assert options["pooling"] == "importance" and options["spatial_beta"] == .2
    assert options["scorer_path"] == "/scorer"


def test_catalog_methods_pass_the_platform_config_loader(tmp_path):
    from scripts.eval.lmms_eval.p6000_dataset_smoke import model_config
    from mllm_comp.config.loader import load_config
    from types import SimpleNamespace
    import yaml
    for name in load_catalog():
        cfg = model_config(SimpleNamespace(model="/model", method=name, method_checkpoint="/head",
                           scorer_checkpoint="/scorer", identity_path="/identity"))
        path = tmp_path / f"{name}.yaml"
        path.write_text(yaml.safe_dump(cfg))
        assert load_config(path) is not None


def test_invalid_selection_and_missing_required_scorer_fail(tmp_path):
    profile, experiment = settings(tmp_path)
    with pytest.raises(ValueError, match="absent"):
        runner.configuration(profile, experiment, tasks=["gqa"])
    with pytest.raises(ValueError, match="scorer_checkpoint"):
        method_options("importance_fps", method_checkpoint="/mlp", identity_path="/identity")
    experiment.write_text(json.dumps({"schema_version": 1, "method_tasks": {"dense": ["ai2d"]}, "reference_only": True}))
    with pytest.raises(ValueError, match="Dense"):
        runner.configuration(profile, experiment)


def small_snapshot(tmp_path, monkeypatch, mapping=None):
    root = tmp_path / "repo"
    (root / "configs/workflow").mkdir(parents=True)
    (root / "configs/workflow/methods.json").write_bytes(DEFAULT_CATALOG.read_bytes())
    (root / "pyproject.toml").write_text("# tiny source fixture\n")
    queue = root / "scripts/eval/lmms_eval/qwen25_portable_queue.py"
    queue.parent.mkdir(parents=True)
    queue.write_text("# queue fixture\n")
    monkeypatch.setattr(runner, "ROOT", root)
    profile, experiment = settings(tmp_path, mapping or {"dense": ["ai2d"]}, reference_only=False)
    config = runner.configuration(profile, experiment, catalog_path=root / "configs/workflow/methods.json")
    return root, queue, config


def test_snapshot_resumes_only_same_source_and_retains_separate_smoke_plan(tmp_path, monkeypatch):
    root, queue, config = small_snapshot(tmp_path, monkeypatch)
    output = tmp_path / "output"
    saved = runner.freeze(config, output, limit=2)
    command = runner.queue_command(saved, output, 0, 1)
    assert command[command.index("--limit") + 1] == "2"
    assert "--result-bundle" not in command and "--seed-from" not in command
    assert runner.freeze(config, output, limit=2, resume=True)["signature"] == saved["signature"]
    with pytest.raises(ValueError, match="Output exists"):
        runner.freeze(config, output, resume=True)
    queue.write_text("# changed implementation\n")
    with pytest.raises(ValueError, match="source/config changed"):
        runner.freeze(config, output, limit=2, resume=True)
    assert (output / "source" / queue.relative_to(root)).read_text() == "# queue fixture\n"


def test_lmu_job_requests_one_node_four_gpus_no_mail_and_frozen_source(tmp_path):
    profile, experiment = settings(tmp_path)
    config = runner.configuration(profile, experiment)
    job = runner.slurm_script({"config": config, "source_root": "/frozen/source"}, "/workspace/results")
    for field in ("--job-name=28-", "--nodes=1", "--gres=gpu:4", "--cpus-per-task=32", '--mail-type="NONE"',
                  "--time=23:50:00", "/frozen/source/scripts/workflow.py execute"):
        assert field in job
    config["profile"]["slurm"]["time"] = "24:00:00"
    with pytest.raises(ValueError, match="below 24"):
        runner.slurm_script({"config": config, "source_root": "/frozen"}, "/output")


def test_prepare_only_writes_a_job_for_manual_submission(tmp_path, monkeypatch):
    profile, experiment = settings(tmp_path)
    output = tmp_path / "manual_run"
    output.mkdir()
    monkeypatch.setattr(runner, "check_paths", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "freeze", lambda config, *args, **kwargs: {
        "config": config, "source_root": "/frozen/source", "signature": "fixture"})
    def refuse_external_command(*args, **kwargs):
        raise AssertionError("Preparing a script must not query or submit jobs")
    monkeypatch.setattr(runner.subprocess, "check_output", refuse_external_command)
    assert runner.main(["prepare", "--profile", str(profile), "--experiment", str(experiment),
                        "--run-dir", str(output)]) == 0
    assert (output / "job.sbatch").is_file()
    assert runner.read(output / "prepared_job.json")["submitted"] is False
    assert not (output / "submission.json").exists()


def test_actual_supervisor_launches_four_workers_and_preserves_gpu_mapping(tmp_path, monkeypatch):
    root, queue, config = small_snapshot(tmp_path, monkeypatch,
        {"dense": ["ai2d", "mmstar", "pope", "chartqa"]})
    queue.write_text('''import json, os, pathlib, sys
args=sys.argv
def arg(name): return args[args.index(name)+1]
root=pathlib.Path(arg("--output")); index=int(arg("--worker-index")); workers=int(arg("--shared-workers"))
experiment=json.loads(pathlib.Path(arg("--experiment")).read_text())
cells=[(m,t) for m,ts in experiment["method_tasks"].items() for t in ts]
for position,(method,task) in enumerate(cells):
    if position % workers == index:
        cell=root/"cells"/method/task; cell.mkdir(parents=True,exist_ok=True)
        (cell/"complete.json").write_text(json.dumps({"gpu":os.environ["CUDA_VISIBLE_DEVICES"]}))
''')
    output = tmp_path / "results"
    runner.freeze(config, output, limit=1)
    monkeypatch.setenv("SLURM_JOB_ID", "cpu-test")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b,GPU-c,GPU-d")
    monkeypatch.setattr(runner.shutil, "which", lambda name: None)
    assert runner.execute(output) == 0
    state = runner.read(output / "workflow_status.json")
    assert state["completed"] == state["expected"] == 4 and state["status"] == "complete"
    assert state["worker_exit_codes"] == [0, 0, 0, 0]
    assigned = {runner.read(path)["gpu"] for path in output.glob("cells/*/*/complete.json")}
    assert assigned == {"GPU-a", "GPU-b", "GPU-c", "GPU-d"}


def test_old_queue_outputs_can_be_inspected_without_workflow_metadata(tmp_path):
    (tmp_path / "plan.json").write_text(json.dumps({"requested_cells": [["old_mlp", "ai2d"]]}))
    assert runner.inspect_run(tmp_path)["pending_cells"] == [["old_mlp", "ai2d"]]
