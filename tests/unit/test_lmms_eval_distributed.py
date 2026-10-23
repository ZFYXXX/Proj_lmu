"""Distributed launch contracts for the lmms-eval integration."""

from __future__ import annotations

from pathlib import Path
import json
import sys
from typing import Any

import pytest

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.config import config_to_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.integrations.lmms_eval import LMMSEvalAdapter
from mllm_comp.integrations.native_cli import NativeCLIResult
from mllm_comp.run_manifest import build_run_manifest


def _config(
    tmp_path: Path,
    *,
    num_processes: int = 1,
    launcher: str = "accelerate",
    main_process_port: int | None = None,
    runtime_mode: str = "default",
) -> object:
    eval_section: dict[str, object] = {
        "tasks": ["mme"],
        "num_processes": num_processes,
        "distributed_launcher": launcher,
        "runtime_mode": runtime_mode,
    }
    if main_process_port is not None:
        eval_section["main_process_port"] = main_process_port
    return experiment_config_from_dict(
        {
            "experiment": {"name": "distributed-eval", "seed": 42},
            "backbone": {"name": "qwen25_omni", "model_id": "Qwen/fixture"},
            "eval": eval_section,
            "output": {
                "run_dir": str(tmp_path / "run"),
                "manifest_path": str(tmp_path / "run" / "manifest.json"),
            },
        },
        path=str(tmp_path / "eval.yaml"),
    )


def test_single_process_keeps_direct_native_cli(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    plan = LMMSEvalAdapter().build_launch_plan(_config(tmp_path))

    assert plan.argv == plan.lmms_eval_argv
    assert plan.argv[:3] == [sys.executable, "-m", "lmms_eval"]
    assert plan.num_processes == 1
    assert plan.plugin_spec["expected_world_size"] == 1
    assert plan.plugin_spec["distributed_launcher"] is None


def test_launch_plan_embeds_python_config_without_requiring_source_yaml(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.path = None
    config.backbone.config["device"] = "cpu"
    config.eval.generation["max_new_tokens"] = 7
    config.backbone.config["python_override"] = {"enabled": True}
    config.pipeline.extensions = ["research.example:register"]
    plan = LMMSEvalAdapter().build_launch_plan(config)
    assert plan.plugin_spec["schema_version"] == 2
    assert plan.plugin_spec["config_path"] is None
    assert plan.plugin_spec["effective_config"] == config_to_dict(config)
    assert plan.plugin_spec["device"] == "cpu"
    assert plan.lmms_eval_argv[plan.lmms_eval_argv.index("--device") + 1] == "cpu"
    assert not (tmp_path / "run").exists()
    config.backbone.config["python_override"]["enabled"] = False
    assert plan.plugin_spec["effective_config"]["backbone"]["config"]["python_override"]["enabled"] is True


def test_accelerate_launch_wraps_only_the_lmms_eval_child(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    plan = LMMSEvalAdapter().build_launch_plan(
        _config(
            tmp_path,
            num_processes=2,
            launcher="accelerate",
            main_process_port=29601,
        )
    )

    assert plan.argv[:8] == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
        "--multi_gpu",
        "--num_processes",
        "2",
        "--main_process_port",
        "29601",
    ]
    assert plan.argv[8:10] == ["-m", "lmms_eval"]
    assert plan.argv[10:] == plan.lmms_eval_argv[3:]
    assert plan.argv.count("--tasks") == 1
    assert plan.plugin_spec["expected_world_size"] == 2
    assert plan.plugin_spec["device"] is None
    assert plan.plugin_spec["distributed_launcher"] == "accelerate"
    assert plan.environment == {"CUDA_VISIBLE_DEVICES": "2,3"}


def test_torchrun_launch_uses_upstream_module_form(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-a,GPU-b")
    plan = LMMSEvalAdapter().build_launch_plan(
        _config(
            tmp_path,
            num_processes=2,
            launcher="torchrun",
            main_process_port=29602,
        )
    )

    assert plan.argv[:7] == [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_port",
        "29602",
    ]
    assert plan.argv[7:9] == ["-m", "lmms_eval"]
    assert plan.argv[9:] == plan.lmms_eval_argv[3:]
    assert plan.distributed_launcher == "torchrun"


def test_distributed_parent_launches_once_and_records_manifest_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    calls: list[Any] = []

    def command_runner(plan: Any) -> NativeCLIResult:
        calls.append(plan)
        result_dir = Path(plan.output_path) / "fixture"
        result_dir.mkdir(parents=True)
        trace_path = Path(plan.plugin_spec["diagnostic_trace_path"])
        for rank in range(2):
            trace_path.with_name(
                f"{trace_path.stem}.rank{rank}{trace_path.suffix}"
            ).write_text(
                "" if rank == 0 else '{"sample_id":"fixture"}\n',
                encoding="utf-8",
            )
        (result_dir / "distributed_results.json").write_text(
            json.dumps(
                {
                    "results": {"mme": {"mme_perception_score,none": 100.0}},
                    "configs": {"mme": {"test_split": "test"}},
                    "n-samples": {"mme": {"original": 4, "effective": 4}},
                }
            ),
            encoding="utf-8",
        )
        return NativeCLIResult(returncode=0)

    config = _config(tmp_path, num_processes=2, runtime_mode="profile")
    result = LMMSEvalAdapter(command_runner=command_runner).evaluate(config=config)

    assert len(calls) == 1
    assert result.backend is not None
    assert result.backend.args["num_processes"] == 2
    assert result.backend.args["distributed_launcher"] == "accelerate"
    assert result.backend.args["argv"][:3] == [
        sys.executable,
        "-m",
        "accelerate.commands.launch",
    ]
    assert result.backend.args["lmms_eval_argv"][:3] == [
        sys.executable,
        "-m",
        "lmms_eval",
    ]
    assert result.metrics[0].num_samples == 4
    assert len(result.backend.args["diagnostic_trace_paths"]) == 2
    assert result.backend.args["diagnostic_trace_paths"][0].endswith(
        "baseline_diagnostic.rank0.jsonl"
    )
    assert len(result.backend.args["diagnostic_pretty_paths"]) == 2
    assert result.backend.args["diagnostic_pretty_paths"][0].endswith(
        "baseline_diagnostic.rank0.pretty.json"
    )
    pretty_records = [
        json.loads(Path(path).read_text(encoding="utf-8"))
        for path in result.backend.args["diagnostic_pretty_paths"]
    ]
    assert pretty_records == [[], [{"sample_id": "fixture"}]]
    manifest = build_run_manifest(config, eval_backend=result.backend)
    assert len(manifest.trace_paths) == 4
    assert manifest.trace_paths["lmms_eval_diagnostic_pretty_0"].endswith(
        "baseline_diagnostic.rank0.pretty.json"
    )


@pytest.mark.parametrize("num_processes", [0, -1])
def test_distributed_eval_rejects_invalid_process_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    num_processes: int,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(BackendIntegrationError, match="at least 1"):
        LMMSEvalAdapter().build_launch_plan(
            _config(tmp_path, num_processes=num_processes)
        )


def test_distributed_eval_rejects_fixed_device(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    with pytest.raises(BackendIntegrationError, match="Do not pass --device"):
        LMMSEvalAdapter().build_launch_plan(
            _config(tmp_path, num_processes=2),
            device="cuda:0",
        )


def test_distributed_eval_rejects_too_few_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with pytest.raises(BackendIntegrationError, match="exceeds CUDA_VISIBLE_DEVICES"):
        LMMSEvalAdapter().build_launch_plan(
            _config(tmp_path, num_processes=2)
        )


@pytest.mark.parametrize("port", [0, 65536])
def test_distributed_eval_rejects_invalid_rendezvous_port(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    port: int,
) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(BackendIntegrationError, match="between 1 and 65535"):
        LMMSEvalAdapter().build_launch_plan(
            _config(tmp_path, num_processes=2, main_process_port=port)
        )
