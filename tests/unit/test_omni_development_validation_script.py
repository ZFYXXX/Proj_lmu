"""Exercise diagnostic command routing and failure propagation without a GPU."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/diagnostics/run_qwen25_omni_eight_method_validation.sh"
pytestmark = pytest.mark.skipif(
    os.name == "nt" or shutil.which("bash") is None,
    reason="The diagnostic launcher runs in the Linux validation environment",
)


def _run(commands: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", "-c", 'source "$1"\n' + commands, "diagnostic-test", str(SCRIPT)],
        env={
            **os.environ,
            "VALIDATION_ROOT": str(tmp_path),
            "VALIDATION_MAX_NEW_TOKENS": "16",
        },
        capture_output=True,
        text=True,
        check=False,
    )


def test_oracle_runs_only_its_method_without_saved_gates(tmp_path: Path) -> None:
    result = _run(
        'run_pytest() { shift; printf "TEST %s\\n" "$*"; }\n' "main oracle segmentwise",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    commands = [line for line in result.stdout.splitlines() if line.startswith("TEST ")]
    assert len(commands) == 2
    assert commands[0] == "TEST tests/unit/test_segmentwise_topk.py"
    assert (
        "test_real_audio_encoder_hooks_prune_two_audio_units_before_projector"
        in commands[1]
    )


def test_check_stops_at_the_first_failed_stage(tmp_path: Path) -> None:
    result = _run(
        "run_oracle() { return 23; }\n"
        "run_gpu() { echo unexpected-gpu; }\n"
        "run_backward() { echo unexpected-backward; }\n"
        "main check segmentwise",
        tmp_path,
    )
    assert result.returncode == 23
    assert "unexpected" not in result.stdout


def test_gpu_command_cannot_reuse_an_old_report(tmp_path: Path) -> None:
    report = tmp_path / "03_gpu/segmentwise/report.json"
    report.parent.mkdir(parents=True)
    report.write_text("old evidence", encoding="utf-8")
    result = _run(
        'require_gpu() { :; }\n'
        'run_pytest() { echo unexpected-run; }\n'
        'main gpu segmentwise',
        tmp_path,
    )
    assert result.returncode != 0
    assert "Report already exists" in result.stderr
    assert "unexpected-run" not in result.stdout
    assert report.read_text(encoding="utf-8") == "old evidence"


@pytest.mark.parametrize("mode", ["quality", "profile"])
def test_experiments_keep_native_generation_and_profile_commands(
    mode: str, tmp_path: Path
) -> None:
    result = _run(
        "require_gpu() { :; }\n"
        'run_logged() { shift; printf "RUN %s\\n" "$*"; }\n'
        "validate_json() { :; }\n"
        "validate_eval_manifest() { :; }\n"
        'validate_eval_protocol() { printf "PROTOCOL %s\\n" "$*"; }\n'
        "validate_profile_summary() { :; }\n"
        f"main {mode} segmentwise",
        tmp_path,
    )
    assert result.returncode == 0, result.stderr
    assert "-m mllm_comp.cli.main eval" in result.stdout
    assert "--generation_max_new_tokens 16 --use_cache" in result.stdout
    assert ("--profile" in result.stdout) == (mode == "profile")
    assert ("summarize_audio_compression.py" in result.stdout) == (mode == "profile")
    expected_mode = "profile" if mode == "profile" else "default"
    assert f" {expected_mode} 16" in result.stdout
