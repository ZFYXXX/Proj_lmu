"""Capability discovery and run comparison expose consistent CLI results."""

from __future__ import annotations

import json

import pytest

from mllm_comp.cli import compare, discover
from mllm_comp.core.errors import ConfigError


def test_main_discovery_reports_the_same_facts_as_the_public_api(capsys) -> None:
    from mllm_comp.cli.main import main
    from mllm_comp.experiment_tools import discover_capabilities

    expected = discover_capabilities(modality="audio", phase="generate", name="segmentwise_topk_audio")
    assert expected["items"]
    assert main([
        "discover", "--modality", "audio", "--phase", "generate", "--name", "segmentwise_topk_audio",
    ]) == 0
    assert json.loads(capsys.readouterr().out) == expected


def test_discover_forwards_filters_and_extensions(monkeypatch, capsys) -> None:
    captured = {}
    result = {"kind": "compressor", "capabilities": [{"name": "研究算法"}]}

    def inspect(**kwargs):
        captured.update(kwargs)
        return result

    monkeypatch.setattr(discover, "discover_capabilities", inspect)
    assert discover.main([
        "--kind", "compressor", "--name", "my_algorithm",
        "--method", "token_compression", "--variant", "my_variant",
        "--binding", "my_binding", "--backbone", "qwen25_omni",
        "--modality", "audio", "--phase", "forward",
        "--extension", "research.first", "--extension", "research.second",
    ]) == 0
    assert captured == {
        "kind": "compressor", "name": "my_algorithm",
        "method": "token_compression", "variant": "my_variant",
        "binding": "my_binding", "backbone": "qwen25_omni",
        "modality": "audio", "phase": "forward",
        "extensions": ("research.first", "research.second"),
    }
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert output.err == ""


def test_discover_defaults_to_compressors(monkeypatch, capsys) -> None:
    captured = {}

    def inspect(**kwargs):
        captured.update(kwargs)
        return {"capabilities": []}

    monkeypatch.setattr(discover, "discover_capabilities", inspect)
    assert discover.main([]) == 0
    assert captured == {
        "kind": "compressor", "name": None, "method": None,
        "variant": None, "binding": None, "backbone": None,
        "modality": None, "phase": None, "extensions": (),
    }
    assert json.loads(capsys.readouterr().out) == {"capabilities": []}


@pytest.mark.parametrize("option", ["--kind", "--method", "--modality", "--phase"])
def test_discover_rejects_unknown_enum_values(option, monkeypatch, capsys) -> None:
    def unexpected_call(**kwargs):
        pytest.fail("Invalid CLI choices must not reach discovery.")

    monkeypatch.setattr(discover, "discover_capabilities", unexpected_call)
    with pytest.raises(SystemExit) as exc_info:
        discover.main([option, "unknown"])
    assert exc_info.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_compare_forwards_paths_and_writes_json(monkeypatch, tmp_path, capsys) -> None:
    captured = []
    result = {"metrics": [{"metric": "accuracy", "delta": 0.1}], "note": "比较"}
    output_path = tmp_path / "reports" / "comparison.json"

    def inspect(left, right):
        captured.append((left, right))
        return result

    monkeypatch.setattr(compare, "compare_runs", inspect)
    assert compare.main(["runs/left", "runs/right", "--output", str(output_path)]) == 0
    assert captured == [("runs/left", "runs/right")]
    assert json.loads(output_path.read_text(encoding="utf-8")) == result
    output = capsys.readouterr()
    assert json.loads(output.out) == result
    assert output.err == ""


def test_compare_without_output_does_not_write(monkeypatch, capsys) -> None:
    monkeypatch.setattr(compare, "compare_runs", lambda left, right: {"metrics": []})

    def unexpected_write(*args, **kwargs):
        pytest.fail("Writing comparison output requires --output.")

    monkeypatch.setattr(compare, "atomic_write_json", unexpected_write)
    assert compare.main(["runs/left", "runs/right"]) == 0
    assert json.loads(capsys.readouterr().out) == {"metrics": []}


@pytest.mark.parametrize(
    ("command", "api_name", "argv"),
    [(discover, "discover_capabilities", []), (compare, "compare_runs", ["left", "right"])],
)
@pytest.mark.parametrize("error_type", [ConfigError, OSError, ValueError])
def test_expected_failures_are_structured(
    command, api_name, argv, error_type, monkeypatch, capsys,
) -> None:
    def fail(*args, **kwargs):
        raise error_type("Cannot inspect this input.")

    monkeypatch.setattr(command, api_name, fail)
    assert command.main(argv) == 2
    output = capsys.readouterr()
    assert json.loads(output.out) == {
        "status": "failed",
        "error": {"type": error_type.__name__, "message": "Cannot inspect this input."},
    }
    assert output.err.strip() == "Cannot inspect this input."


def test_compare_output_failure_is_reported(monkeypatch, capsys) -> None:
    monkeypatch.setattr(compare, "compare_runs", lambda left, right: {"metrics": []})

    def fail(*args, **kwargs):
        raise OSError("Cannot write comparison.json.")

    monkeypatch.setattr(compare, "atomic_write_json", fail)
    assert compare.main(["left", "right", "--output", "comparison.json"]) == 2
    output = capsys.readouterr()
    assert json.loads(output.out)["error"] == {
        "type": "OSError", "message": "Cannot write comparison.json.",
    }
    assert "Cannot write comparison.json." in output.err


@pytest.mark.parametrize(
    ("command", "api_name", "argv"),
    [(discover, "discover_capabilities", []), (compare, "compare_runs", ["left", "right"])],
)
def test_unexpected_failures_keep_the_traceback(
    command, api_name, argv, monkeypatch, capsys,
) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("Implementation bug.")

    monkeypatch.setattr(command, api_name, fail)
    with pytest.raises(RuntimeError, match="Implementation bug"):
        command.main(argv)
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err == ""
