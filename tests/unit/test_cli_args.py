"""CLI parsers accept the unified task-interface arguments."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp import experiments
from mllm_comp.cli import analyze, eval as eval_command, main as cli_main, train
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord, BackendRecord
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.integrations.native_cli import NativeCLIPlan, NativeCLIProcessError


_RUNNABLE_FIXTURE = "tests/fixtures/minimal_runnable_config.yaml"
_ANALYSIS_FIXTURE = "examples/analysis/qwen25_vl_analysis_matrix.yaml"


@pytest.mark.parametrize("command", [analyze, eval_command, train])
@pytest.mark.parametrize("complete", [True, False], ids=["complete", "missing-output"])
def test_cli_validates_effective_values_after_overrides(
    command: Any,
    complete: bool,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    for name in (
        "CLI_TEST_MODEL", "CLI_TEST_OUTPUT", "CLI_TEST_MODE", "CLI_TEST_BATCH",
        "CLI_TEST_TASKS", "CLI_TEST_WORKERS", "CLI_TEST_RUNTIME_MODE",
        "CLI_TEST_PHASE", "CLI_TEST_LAYERS",
    ):
        monkeypatch.delenv(name, raising=False)
    raw = {
        "experiment": {"name": "cli-overrides"},
        "backbone": {"name": "qwen25_vl", "model_path": "${CLI_TEST_MODEL}"},
        "analysis": {
            "tasks": ["mme"],
            "probes": [{"name": "representation_dynamics", "layers": [0]}],
        },
        "eval": {"tasks": ["mme"]},
        "train": {"args": {"datasets": ["fixture"]}},
        "output": {
            "run_dir": "${CLI_TEST_OUTPUT}",
            "manifest_path": "${CLI_TEST_OUTPUT}/manifest.json",
        },
    }
    if command is eval_command:
        raw["eval"]["checkpoint_path"] = "${CLI_TEST_MODEL}"
    elif command is train:
        raw["train"]["checkpoint_in"] = "${CLI_TEST_MODEL}"
    if command is train:
        raw["eval"]["tasks"] = ["${UNUSED_EVAL_TASK}"]
    else:
        raw["train"]["args"]["datasets"] = ["${UNUSED_TRAIN_DATA}"]
    section = "analysis" if command is analyze else command.__name__.rsplit(".", 1)[1]
    raw[section]["compression_mode"] = "${CLI_TEST_MODE}"
    if command is train:
        raw[section]["runtime_mode"] = "${CLI_TEST_RUNTIME_MODE}"
    else:
        raw[section].update(
            runtime_batch_size="${CLI_TEST_BATCH}",
            tasks="${CLI_TEST_TASKS}",
        )
    if command is eval_command:
        raw[section]["num_processes"] = "${CLI_TEST_WORKERS}"
    if command is analyze:
        raw[section]["phase"] = "${CLI_TEST_PHASE}"
        raw[section]["probes"][0]["layers"] = "${CLI_TEST_LAYERS}"
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    captured: dict[str, Any] = {}

    class RuntimeReached(Exception):
        """Stop the test exactly where real execution would begin."""

    def execute(config: Any, *args: Any, **kwargs: Any) -> Any:
        captured["config"] = config
        raise RuntimeReached

    class Adapter:
        def build_launch_plan(self, config: Any, **kwargs: Any) -> Any:
            return execute(config)

    if command is analyze:
        monkeypatch.setattr(experiments, "execute_analysis", execute)
    else:
        monkeypatch.setattr(experiments, "get_integration", lambda name: Adapter)
    argv = [
        "--config", str(path), "--checkpoint", "/models/override",
        "--compression_mode", "none",
    ]
    if command is train:
        argv += ["--runtime_mode", "default"]
    else:
        argv += ["--runtime_batch_size", "1", "--tasks", "mme"]
    if command is eval_command:
        argv += ["--num_processes", "1"]
    if command is analyze:
        argv += ["--phase", "pre_compression", "--layers", "0"]
    if complete:
        argv += [
            "--output_dir", str(tmp_path / "run"),
            "--manifest", str(tmp_path / "run" / "manifest.json"),
        ]
    if command is not analyze:
        argv.append("--plan_only")

    if complete:
        with pytest.raises(RuntimeReached):
            command.main(argv)
        config = captured["config"]
        assert config.backbone.model_path == "/models/override"
        assert config.output.run_dir == str(tmp_path / "run")
        assert config.raw["backbone"]["model_path"] == "${CLI_TEST_MODEL}"
        assert str(getattr(config, section).compression_mode) == "none"
        if command is eval_command:
            assert config.eval.num_processes == 1
        if command is analyze:
            assert config.analysis.probes[0]["layers"] == [0]
    else:
        assert command.main(argv) == 2
        assert "CLI_TEST_OUTPUT" in capsys.readouterr().err
        assert not captured


@pytest.mark.parametrize(
    ("level", "expected"),
    [("debug", ["debug message", "error message"]), ("ERROR", ["error message"])],
)
def test_cli_log_level_controls_emitted_python_logs(
    level: str,
    expected: list[str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.NOTSET)
    logger = logging.getLogger("mllm_comp.cli.test_logging")

    class Adapter:
        def build_launch_plan(self, config: Any, **kwargs: Any) -> NativeCLIPlan:
            logger.debug("debug message")
            logger.error("error message")
            return NativeCLIPlan([], {}, ".", [])

        def dependency_report(self) -> dict[str, Any]:
            return {"versions": {}}

    monkeypatch.setattr(experiments, "get_integration", lambda name: Adapter)
    assert eval_command.main(
        ["--config", _RUNNABLE_FIXTURE, "--log_level", level, "--plan_only"]
    ) == 0
    assert [record.message for record in caplog.records if record.name == logger.name] == expected


def test_analyze_accepts_analysis_overrides_and_writes_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    captured: dict[str, Any] = {}

    def execute(config: Any, definition: Any, **kwargs: Any) -> Any:
        captured["config"] = config
        captured["definition"] = definition
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            records=[
                AnalysisRecord(
                    name="representation_dynamics",
                    phase="pre_compression",
                    method="representation_dynamics",
                    summary={"successful_samples": 1, "skipped_samples": 0},
                    artifact_paths={"aggregate": "aggregate.json"},
                    sample_count=1,
                )
            ],
            backend=BackendRecord(name="lmms_eval", version="fixture"),
            sample_count=1,
            compression_runtime_metadata=None,
        )

    monkeypatch.setattr(experiments, "execute_analysis", execute)
    output_dir = tmp_path / "analyze"
    manifest = tmp_path / "manifest.json"

    assert analyze.main(
        [
            "--config",
            _ANALYSIS_FIXTURE,
            "--compression_mode",
            "none",
            "--tasks",
            "vqav2,gqa",
            "--sample_limit",
            "4",
            "--probe",
            "representation_dynamics",
            "--phase",
            "pre_compression",
            "--layers",
            "0,8,16",
            "--metrics",
            "cosine",
            "--save_raw",
            "--runtime_batch_size",
            "1",
            "--checkpoint",
            "/checkpoints/Qwen2.5-VL-3B-Instruct",
            "--output_dir",
            str(output_dir),
            "--manifest",
            str(manifest),
        ]
    ) == 0
    assert captured["definition"].tasks == ["vqav2", "gqa"]
    assert captured["definition"].probes[0]["layers"] == [0, 8, 16]
    assert captured["config"].backbone.model_path == (
        "/checkpoints/Qwen2.5-VL-3B-Instruct"
    )
    assert captured["kwargs"] == {"device": None}
    assert manifest.is_file()
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["invariants_passed"] is True
    assert payload["model_path_or_id"] == "/checkpoints/Qwen2.5-VL-3B-Instruct"
    assert payload["checkpoint_path"] == "/checkpoints/Qwen2.5-VL-3B-Instruct"


@pytest.mark.parametrize("command", [analyze, eval_command, train])
def test_cli_rejects_empty_checkpoint_override(command: Any) -> None:
    with pytest.raises(SystemExit):
        command.main(
            [
                "--config",
                _ANALYSIS_FIXTURE,
                "--checkpoint",
                "   ",
            ]
        )


def test_analysis_invariants_fail_when_any_probe_skips_a_sample() -> None:
    result = SimpleNamespace(
        sample_count=2,
        records=[
            AnalysisRecord(
                name="representation_dynamics",
                phase="pre_compression",
                method="representation_dynamics",
                summary={"successful_samples": 1, "skipped_samples": 1},
                artifact_paths={"aggregate": "aggregate.json"},
                sample_count=1,
            )
        ],
    )

    assert experiments.analysis_invariants_passed(result) is False


def test_analysis_invariants_fail_when_skip_evidence_is_missing() -> None:
    result = SimpleNamespace(
        sample_count=1,
        records=[
            AnalysisRecord(
                name="representation_dynamics",
                phase="pre_compression",
                method="representation_dynamics",
                summary={"successful_samples": 1},
                artifact_paths={"aggregate": "aggregate.json"},
                sample_count=1,
            )
        ],
    )

    assert experiments.analysis_invariants_passed(result) is False


def test_eval_rejects_compression_without_enabled_stage() -> None:
    assert eval_command.main(
        ["--config", _RUNNABLE_FIXTURE, "--compression_mode", "compressed", "--plan_only"]
    ) == 2


def test_train_rejects_compression_without_enabled_stage() -> None:
    assert train.main(
        ["--config", _RUNNABLE_FIXTURE, "--compression_mode", "compressed", "--plan_only"]
    ) == 2


@pytest.mark.parametrize("command", [eval_command.main, train.main])
def test_preflight_only_is_not_a_public_command_mode(command: Any) -> None:
    with pytest.raises(SystemExit):
        command(["--config", _RUNNABLE_FIXTURE, "--preflight_only"])


def test_train_accepts_ms_swift_plan_overrides(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert (
        train.main(
            [
                "--config",
                "tests/fixtures/qwen25_omni_compressed_train.yaml",
                "--compression_mode",
                "compressed",
                "--backend",
                "ms_swift",
                "--checkpoint",
                "checkpoints/base",
                "--artifact_format",
                "transformers",
                "--max_steps",
                "2",
                "--debug",
                "--output_dir",
                "runs/train",
                "--manifest",
                "runs/manifest.json",
                "--plan_only",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["output"]["plan"]["model_path"] == "checkpoints/base"


def test_train_success_records_passed_swift_boundaries(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Adapter:
        def train(self, **kwargs: object) -> object:
            return SimpleNamespace(
                backend=BackendRecord(name="ms_swift", version="fixture"),
                checkpoint_path="runs/train/v0/checkpoint-1",
                train_log_path="runs/train/v0/logging.jsonl",
                metrics={"global_step": 1.0},
            )

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: _Adapter())
    manifest_path = tmp_path / "manifest.json"
    assert train.main([
        "--config", _RUNNABLE_FIXTURE,
        "--output_dir", str(tmp_path / "run"), "--manifest", str(manifest_path),
    ]) == 0
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["invariants_passed"] is True
    assert manifest["invariants_run"] == [
        "MSSwiftSftArguments",
        "ThinkerOnlyTrainingBoundary",
    ]


@pytest.mark.parametrize(
    ("backbone", "boundary"),
    [
        ("qwen25_vl", "Qwen25VLLanguageModelTrainingBoundary"),
        ("qwen3_vl", "Qwen3VLLanguageModelTrainingBoundary"),
        ("llava_onevision", "LlavaOneVisionLanguageModelTrainingBoundary"),
    ],
)
def test_train_manifest_uses_image_language_model_boundary_name(
    backbone: str,
    boundary: str,
) -> None:
    assert experiments._training_boundary_invariants(backbone) == [
        "MSSwiftSftArguments",
        boundary,
    ]


def test_train_manifest_rejects_unmapped_backbone_boundary() -> None:
    with pytest.raises(ValueError, match="No train Manifest boundary mapping"):
        experiments._training_boundary_invariants("unknown_backbone")


def test_eval_accepts_generation_overrides(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        def evaluate(self, **kwargs: object) -> object:
            seen.update(kwargs)
            return SimpleNamespace(
                backend=BackendRecord(name="lmms_eval", version="fixture"),
                metrics=[],
                raw_output_path="runs/eval/raw.json",
            )

    monkeypatch.setattr(
        experiments,
        "get_integration",
        lambda name: lambda: _Adapter(),
    )

    assert (
        eval_command.main(
            [
                "--config",
                "tests/fixtures/qwen25_omni_compressed_train.yaml",
                "--compression_mode",
                "compressed",
                "--checkpoint",
                "runs/checkpoint-final",
                "--target",
                "compressed",
                "--tasks",
                "vqav2, gqa, ",
                "--sample_limit",
                "5",
                "--runtime_batch_size",
                "1",
                "--generation_max_new_tokens",
                "64",
                "--use_cache",
                "--temperature",
                "0.0",
                "--top_p",
                "1.0",
                "--output_dir",
                str(tmp_path / "run"),
                "--manifest",
                str(tmp_path / "manifest.json"),
            ]
        )
        == 0
    )
    config = seen["config"]
    assert config.eval.tasks == ["vqav2", "gqa"]
    assert config.eval.generation == {
        "max_new_tokens": 64,
        "use_cache": True,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    assert "backbone" not in seen
    assert "runtime" not in seen
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["invariants_run"] == [
        "Qwen25OmniNativeValidator.validate_state",
        "Qwen25OmniNativeValidator.validate_generation",
        "LMMSEvalModelProtocol",
    ]


def test_eval_can_explicitly_disable_generation_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        def build_launch_plan(self, config: object, **kwargs: object) -> NativeCLIPlan:
            seen.update({"config": config, **kwargs})
            return NativeCLIPlan(
                argv=["lmms_eval"],
                environment={},
                cwd=".",
                expected_output_paths=[],
            )

        def dependency_report(self) -> dict[str, object]:
            return {"versions": {}}

    monkeypatch.setattr(
        experiments,
        "get_integration",
        lambda name: lambda: _Adapter(),
    )

    assert (
        eval_command.main(
            ["--config", _RUNNABLE_FIXTURE, "--no-use_cache", "--plan_only"]
        )
        == 0
    )
    assert seen["config"].eval.generation["use_cache"] is False


@pytest.mark.parametrize(
    ("mode_args", "expected_mode"),
    [
        (["--runtime_mode", "debug"], RuntimeMode.DEBUG),
        (["--debug"], RuntimeMode.DEBUG),
        (["--profile"], RuntimeMode.PROFILE),
    ],
)
def test_eval_runtime_mode_cli_overrides_reach_launch_plan(
    monkeypatch: pytest.MonkeyPatch,
    mode_args: list[str],
    expected_mode: RuntimeMode,
) -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        def build_launch_plan(self, config: object, **kwargs: object) -> NativeCLIPlan:
            seen.update({"config": config, **kwargs})
            return NativeCLIPlan(
                argv=["lmms_eval"],
                environment={},
                cwd=".",
                expected_output_paths=[],
            )

        def dependency_report(self) -> dict[str, object]:
            return {"versions": {"lmms_eval": "fixture"}}

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: _Adapter())

    assert (
        eval_command.main(
            ["--config", _RUNNABLE_FIXTURE, *mode_args, "--plan_only"]
        )
        == 0
    )
    config = seen["config"]
    assert getattr(config, "eval").runtime_mode is expected_mode


def test_eval_accepts_distributed_launch_overrides(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}

    class _Adapter:
        def build_launch_plan(self, config: object, **kwargs: object) -> NativeCLIPlan:
            seen.update({"config": config, **kwargs})
            return NativeCLIPlan(
                argv=["accelerate", "launch", "-m", "lmms_eval"],
                environment={},
                cwd=".",
                expected_output_paths=[],
            )

        def dependency_report(self) -> dict[str, object]:
            return {"versions": {"lmms_eval": "fixture"}}

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: _Adapter())

    assert (
        eval_command.main(
            [
                "--config",
                _RUNNABLE_FIXTURE,
                "--num_processes",
                "4",
                "--distributed_launcher",
                "torchrun",
                "--main_process_port",
                "29603",
                "--plan_only",
            ]
        )
        == 0
    )
    config = seen["config"]
    assert config.eval.num_processes == 4
    assert str(config.eval.distributed_launcher) == "torchrun"
    assert config.eval.main_process_port == 29603


def test_eval_reports_invalid_distributed_plan_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _Adapter:
        def build_launch_plan(self, config: object, **kwargs: object) -> object:
            del config, kwargs
            raise BackendIntegrationError("distributed contract rejected")

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: _Adapter())

    assert (
        eval_command.main(
            ["--config", _RUNNABLE_FIXTURE, "--num_processes", "2", "--plan_only"]
        )
        == 2
    )
    captured = capsys.readouterr()
    assert captured.err.strip() == "distributed contract rejected"
    assert "Traceback" not in captured.err


def test_eval_manifest_uses_qwen3_vl_validator_names() -> None:
    assert experiments._native_validator_invariants("qwen3_vl") == [
        "Qwen3VLNativeValidator.validate_state",
        "Qwen3VLNativeValidator.validate_generation",
        "LMMSEvalModelProtocol",
    ]


def test_eval_manifest_uses_qwen3_5_validator_names() -> None:
    assert experiments._native_validator_invariants("qwen3_5") == [
        "Qwen3_5NativeValidator.validate_state",
        "Qwen3_5NativeValidator.validate_generation",
        "LMMSEvalModelProtocol",
    ]


def test_eval_manifest_uses_llava_onevision_validator_names() -> None:
    assert experiments._native_validator_invariants("llava_onevision") == [
        "LlavaOneVisionNativeValidator.validate_state",
        "LlavaOneVisionNativeValidator.validate_generation",
        "LMMSEvalModelProtocol",
    ]


def test_eval_manifest_rejects_unmapped_backbone_validator() -> None:
    with pytest.raises(ValueError, match="No eval Manifest validator mapping"):
        experiments._native_validator_invariants("unknown_backbone")


def test_eval_propagates_native_signal_exit_without_loading_a_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _Adapter:
        def evaluate(self, **kwargs: object) -> object:
            assert "backbone" not in kwargs and "runtime" not in kwargs
            raise NativeCLIProcessError("lmms-eval", -15)

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: _Adapter())
    assert (
        eval_command.main(
            [
                "--config",
                _RUNNABLE_FIXTURE,
                "--checkpoint",
                "Qwen/fixture",
                "--output_dir", str(tmp_path / "run"),
                "--manifest", str(tmp_path / "manifest.json"),
                "--tasks",
                "mme",
            ]
        )
        == 143
    )


def test_real_command_rejects_unresolved_config_before_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("QWEN25_OMNI_MODEL_PATH", raising=False)
    assert eval_command.main([
        "--config", "configs/eval/lmms_eval/qwen25_omni_img.yaml", "--plan_only",
    ]) == 2


@pytest.mark.parametrize("content", ["eval: [", '{"eval": {"num_processes": null}}'])
def test_cli_reports_config_parse_errors_as_structured_failures(content, tmp_path, capsys):
    path = tmp_path / "invalid.yaml"
    path.write_text(content, encoding="utf-8")
    assert eval_command.main(["--config", str(path), "--plan_only"]) == 2
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert result["status"] == "failed"
    assert result["error"]["stage"] == "configuration"
    assert result["error"]["type"] == "ConfigError"
    assert result["paths"] is None
    assert "Traceback" not in captured.err


def test_top_level_cli_reraises_unexpected_errors_after_recording(tmp_path, monkeypatch):
    error = NotImplementedError("unfinished backend implementation")

    def execute(**kwargs):
        raise error

    monkeypatch.setattr(experiments, "get_integration", lambda name: lambda: SimpleNamespace(evaluate=execute))
    run_dir = tmp_path / "run"
    with pytest.raises(NotImplementedError) as raised:
        cli_main.main([
            "eval", "--config", _RUNNABLE_FIXTURE,
            "--output_dir", str(run_dir), "--manifest", str(run_dir / "manifest.json"),
        ])
    assert raised.value is error
    payload = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert payload["returncode"] == 1
    assert payload["error"]["type"] == "NotImplementedError"
