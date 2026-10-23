"""Read-only discovery and comparison of real framework artifact shapes."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from mllm_comp.config import config_to_dict
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.manifest import MetricRecord
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.experiment_tools import compare_runs, discover_capabilities


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _manifest(*, mode: str = "none", accuracy: float = 0.7) -> dict:
    config = experiment_config_from_dict({
        "experiment": {"name": "comparison", "seed": 42},
        "backbone": {"name": "qwen25_omni", "model_path": "/models/qwen", "config": {"device": "cuda:0"}},
        "eval": {
            "backend": "lmms_eval", "tasks": ["mmau_test_mini"], "compression_mode": mode,
            "target": "baseline" if mode == "none" else "compressed",
        },
    })
    record = to_jsonable(MetricRecord("mmau_test_mini", "test", {"accuracy": accuracy}, 2))
    return {
        "run_id": "fixture", "backbone_name": "qwen25_omni",
        "model_path_or_id": "/models/qwen", "checkpoint_path": "/models/qwen",
        "source_commit": "fixture-commit", "source_dirty": False,
        "source_tree_hash": "recorded-runtime-source", "source_scope": "runtime_sources_v1",
        "eval_tasks": ["mmau_test_mini"], "trace_paths": {},
        "baseline_metrics": [record] if mode == "none" else [],
        "compressed_metrics": [record] if mode == "compressed" else [],
        "effective_config": config_to_dict(config),
        "eval_backend": {"name": "lmms_eval", "version": "0.7.2", "args": {
            "model_name": "qwen2_5_omni", "num_processes": 1, "distributed_launcher": None,
            "versions": {"transformers": "4.57.3", "torch": "2.8.0"},
            "lmms_eval_argv": ["python", "-m", "lmms_eval", "--batch_size", "1", "--seed", "42"],
            "plugin_spec": {
                "device": "cuda:0", "expected_world_size": 1, "generation": {"max_new_tokens": 32},
                "system_prompt": None, "use_audio_in_video": False,
                "runtime_mode": "profile", "compression_mode": mode,
            },
        }},
    }


def _run(directory: Path, manifest: dict | None, **result_overrides: object) -> Path:
    result = {
        "run_id": directory.name, "task": "eval", "status": "succeeded",
        "returncode": 0, "native_returncode": 0,
        "paths": {"run_dir": str(directory), "manifest": str(directory / "manifest.json")},
        **result_overrides,
    }
    _write(directory / "result.json", result)
    if manifest is not None:
        _write(directory / "manifest.json", manifest)
    return directory


def _profile(directory: Path, mode: str) -> dict:
    manifest = _manifest(mode=mode, accuracy=0.7 if mode == "none" else 0.68)
    directory.mkdir(parents=True, exist_ok=True)
    trace = directory / "diagnostic.jsonl"
    metadata = {
        "runtime_mode": "profile", "compression_mode": mode,
        "timing": {
            "generation_seconds": 1.0 if mode == "none" else 0.8,
            "prefill_seconds": 0.4 if mode == "none" else 0.2,
            "llm_input_tokens": 100 if mode == "none" else 60,
            "cuda_peak_memory_delta_bytes": 800 if mode == "none" else 500,
        },
    }
    if mode == "compressed":
        metadata["compression"] = {
            "original_audio_tokens": 100, "kept_audio_tokens": 25,
            "items": [{"audio_unit_index": 0, "indices_chronological": True}],
        }
    record = {
        "runtime_environment": {
            "device": "cuda:0", "gpu_name": "fixture-gpu",
            "packages": {"transformers": "4.57.3", "torch": "2.8.0"},
        },
        "generation_metadata": metadata,
    }
    trace.write_text("\n".join(json.dumps(record) for _ in range(2)) + "\n", encoding="utf-8")
    manifest["trace_paths"] = {"lmms_eval_diagnostic_0": str(trace)}
    return manifest


@pytest.mark.parametrize("modality", ["image", "video", "audio"])
def test_discovery_filters_real_bindings_and_exact_evidence(modality: str) -> None:
    all_items = discover_capabilities(modality=modality, phase="generate")["items"]
    assert all_items
    item = all_items[0]
    filtered = discover_capabilities(
        name=item["algorithm"], method=item["method"], variant=item["variant"],
        binding=item["binding"], backbone=item["backbone"], modality=modality, phase="generate",
    )["items"]
    assert filtered == [item]
    assert modality in item["target_modalities"]
    assert isinstance(item["config_defaults"], dict)
    for support in item["support"]:
        assert support["binding"] == item["binding"]
        assert support["target_modality"] == modality
        assert support["phase"] == "generate"
        assert "evidence" in support
    json.dumps(filtered, allow_nan=False)


def test_discovery_imports_explicit_extension_without_constructing_or_publishing(monkeypatch) -> None:
    from examples.compression_extensions.first_k_audio import FirstKStrategy
    from mllm_comp.compressors.projector_level import SegmentwiseTopKAudio

    def forbidden(*args, **kwargs):
        raise AssertionError("Discovery must not construct kernels or strategies")

    monkeypatch.setattr(SegmentwiseTopKAudio, "__init__", forbidden)
    monkeypatch.setattr(FirstKStrategy, "__init__", forbidden)
    name = "researcher_first_k_audio"
    assert discover_capabilities(name=name)["items"] == []
    items = discover_capabilities(
        name=name, phase="generate", extensions=["examples.compression_extensions.first_k_audio:register"],
    )["items"]
    assert len(items) == 1
    assert items[0]["capability_status"] == "target"
    assert items[0]["executor_available"] is True
    assert items[0]["support"] == []
    json.dumps(items, allow_nan=False)
    assert discover_capabilities(name=name)["items"] == []


@pytest.mark.parametrize("kind", ["backbone", "analysis", "integration"])
def test_discovery_catalog_is_json_serializable(kind: str) -> None:
    result = discover_capabilities(kind=kind)
    assert result["items"]
    json.dumps(result, allow_nan=False)
    name = result["items"][0]["name"]
    assert discover_capabilities(kind=kind, name=name)["items"] == [result["items"][0]]
    with pytest.raises(ValueError, match="require kind"):
        discover_capabilities(kind=kind, phase="generate")


def test_discovery_does_not_advertise_unimplemented_quantization_bindings() -> None:
    assert discover_capabilities(method="quantization")["items"] == []


def test_operation_replacements_expose_prepare_bindings_without_invented_evidence() -> None:
    name = "qwen3_vl_attention_replacement"
    result = discover_capabilities(name=name, backbone="qwen3_vl", phase="prepare")
    assert result["catalog_only"] == []
    items = {item["binding"]: item for item in result["items"]}
    assert set(items) == {f"{name}.shared_llm.qwen3_vl", f"{name}.image_llm.qwen3_vl"}
    for item in items.values():
        assert item["algorithm"] == name
        assert item["method"] == "operation_compression"
        assert item["capability_status"] == "active"
        assert all(row["status"] == "declared" and not row["evidence"] for row in item["support"])
    image_item = items[f"{name}.image_llm.qwen3_vl"]
    assert image_item["supported_rules"] == []
    assert len(image_item["support"]) == 1
    item = items[f"{name}.shared_llm.qwen3_vl"]
    assert item["supported_rules"] == ["factory"]
    assert item["token_unit"] is None
    assert len(item["support"]) == 2
    assert all(row["status"] == "declared" and not row["evidence"] for row in item["support"])
    assert discover_capabilities(name=name, phase="generate")["items"] == []



def test_compare_metrics_missing_and_zero_are_distinct(tmp_path: Path) -> None:
    left_manifest = _manifest(accuracy=0.0)
    right_manifest = _manifest(mode="compressed", accuracy=0.1)
    right_manifest["compressed_metrics"][0]["metrics"]["extra"] = 0.0
    report = compare_runs(_run(tmp_path / "left", left_manifest), _run(tmp_path / "right", right_manifest))
    accuracy, extra = report["metrics"]
    assert accuracy["absolute_delta"] == pytest.approx(0.1)
    assert accuracy["relative_delta"] is None
    assert extra["left"] is None and extra["left_present"] is False
    assert extra["right"] == 0 and extra["right_present"] is True
    assert extra["absolute_delta"] is None
    assert report["protocol"]["matched"] is True
    assert report["audio_efficiency"]["status"] == "unavailable"
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("change,field", [
    ("device", "backbone.config.device"), ("checkpoint", "checkpoint"),
    ("generation", "eval.generation.max_new_tokens"), ("samples", "sample_counts"),
    ("data", "data.dataset_commit"),
])
def test_compare_records_material_protocol_differences(tmp_path: Path, change: str, field: str) -> None:
    before = _manifest()
    after = deepcopy(before)
    if change == "device":
        after["effective_config"]["backbone"]["config"]["device"] = "cpu"
    elif change == "checkpoint":
        after["checkpoint_path"] = "/models/other"
    elif change == "generation":
        after["effective_config"]["eval"]["generation"]["max_new_tokens"] = 64
    elif change == "samples":
        after["baseline_metrics"][0]["num_samples"] = 1
    elif change == "data":
        after["effective_config"]["data"]["dataset_commit"] = "different"
    report = compare_runs(_run(tmp_path / "left", before), _run(tmp_path / "right", after))
    assert report["protocol"]["matched"] is False
    assert field in {item["field"] for item in report["protocol"]["differences"]}


def test_compare_ignores_output_identity_but_reports_pipeline_changes(tmp_path: Path) -> None:
    before = _manifest()
    after = deepcopy(before)
    after["effective_config"]["output"] = {"run_dir": "elsewhere", "manifest_path": "other.json"}
    after["effective_config"]["experiment"]["run_id"] = "different"
    after["effective_config"]["pipeline"]["extensions"] = ["research.method:register"]
    report = compare_runs(_run(tmp_path / "left", before), _run(tmp_path / "right", after))
    assert [item["field"] for item in report["config_differences"]] == ["pipeline.extensions"]
    assert report["protocol"]["matched"] is True


def test_failed_run_without_manifest_retains_failure_and_snapshot(tmp_path: Path) -> None:
    left = _run(tmp_path / "failed", None, status="failed", returncode=143, native_returncode=-15,
                error={"type": "BackendExit", "message": "terminated"})
    _write(left / "effective_config.json", _manifest()["effective_config"])
    right = _run(tmp_path / "right", _manifest())
    report = compare_runs(left, right)
    assert report["left"]["native_returncode"] == -15
    assert report["left"]["missing"] == ["manifest"]
    assert report["protocol"]["missing"]["left"] == ["manifest"]
    assert report["metrics"][0]["left"] is None
    assert report["config_differences"] == []


def test_compare_legacy_default_and_recorded_absolute_external_manifest(tmp_path: Path) -> None:
    left = tmp_path / "legacy"
    _write(left / "manifest.json", _manifest())
    external = tmp_path / "external.json"
    _write(external, _manifest())
    right = _run(tmp_path / "right", None, paths={"manifest": str(external)})
    report = compare_runs(left, right)
    assert report["left"]["status"] == "unknown"
    assert report["right"]["manifest_path"] == str(external)


def test_relative_default_is_within_supplied_run_but_custom_path_is_not_guessed(tmp_path: Path) -> None:
    left = _run(tmp_path / "left", _manifest(), paths={
        "run_dir": "runs/left", "manifest": "runs/left/manifest.json",
    })
    right = _run(tmp_path / "right", _manifest())
    assert compare_runs(left, right)["left"]["manifest_path"] == str(left / "manifest.json")
    _run(left, _manifest(), paths={"run_dir": "runs/left", "manifest": "reports/custom.json"})
    with pytest.raises(ValueError, match="relative"):
        compare_runs(left, right)


@pytest.mark.parametrize("missing", ["source_tree_hash", "num_samples"])
def test_incomplete_legacy_protocol_never_claims_match(tmp_path: Path, missing: str) -> None:
    manifest = _manifest()
    if missing == "source_tree_hash":
        del manifest["source_tree_hash"]
        expected = "source_tree_hash"
    else:
        manifest["baseline_metrics"][0]["num_samples"] = None
        expected = "sample_counts.mmau_test_mini.test"
    report = compare_runs(_run(tmp_path / "left", manifest), _run(tmp_path / "right", manifest))
    assert report["protocol"]["matched"] is None
    assert expected in report["protocol"]["missing"]["left"]


@pytest.mark.parametrize("reverse", [False, True])
def test_audio_profile_reuses_strict_comparator(tmp_path: Path, reverse: bool) -> None:
    baseline = _run(tmp_path / "baseline", _profile(tmp_path / "baseline", "none"))
    compressed = _run(tmp_path / "compressed", _profile(tmp_path / "compressed", "compressed"))
    report = compare_runs(compressed, baseline) if reverse else compare_runs(baseline, compressed)
    audio = report["audio_efficiency"]
    assert audio["status"] == "compared", audio
    assert audio["baseline"] == ("right" if reverse else "left")
    assert audio["comparison"]["efficiency_metrics"]["timing_seconds.prefill.median"]["baseline_over_compressed"] == 2


def test_audio_protocol_mismatch_never_produces_efficiency_ratios(tmp_path: Path) -> None:
    before = _profile(tmp_path / "baseline", "none")
    after = _profile(tmp_path / "compressed", "compressed")
    after["eval_backend"]["args"]["plugin_spec"]["generation"]["max_new_tokens"] = 64
    report = compare_runs(_run(tmp_path / "baseline", before), _run(tmp_path / "compressed", after))
    assert report["audio_efficiency"]["status"] == "unavailable"
    assert "different evaluation protocols" in report["audio_efficiency"]["reason"]
    assert "comparison" not in report["audio_efficiency"]
    assert report["protocol"]["matched"] is False
    assert "native_evaluation.generation.max_new_tokens" in {
        item["field"] for item in report["protocol"]["differences"]
    }


def test_train_manifest_without_eval_backend_has_no_audio_efficiency(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["eval_backend"] = None
    manifest["train_backend"] = {"name": "ms_swift", "version": "3", "args": {}}
    report = compare_runs(
        _run(tmp_path / "left", manifest, task="train", output={"metrics": {"loss": 1.0}}),
        _run(tmp_path / "right", manifest, task="train", output={"metrics": {"loss": 0.5}}),
    )
    loss = next(item for item in report["metrics"] if item["metric"] == "loss")
    assert loss["absolute_delta"] == -0.5
    assert report["audio_efficiency"]["status"] == "unavailable"


def test_legacy_summary_script_keeps_positional_and_output_contract(tmp_path: Path, monkeypatch) -> None:
    from mllm_comp.analysis.audio_summary import _read_records, summarize_records

    script = Path(__file__).resolve().parents[2] / "scripts/analysis/summarize_audio_compression.py"
    spec = importlib.util.spec_from_file_location("legacy_audio_summary", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = _profile(tmp_path / "profile", "compressed")
    trace = manifest["trace_paths"]["lmms_eval_diagnostic_0"]
    output = tmp_path / "summary.json"
    monkeypatch.setattr("sys.argv", [str(script), trace, "--output", str(output)])
    assert module.main() == 0
    expected = summarize_records(_read_records([Path(trace)]))
    expected["source_files"] = [trace]
    assert json.loads(output.read_text(encoding="utf-8")) == expected


def test_legacy_comparison_script_keeps_named_inputs_and_source_evidence(tmp_path: Path) -> None:
    from mllm_comp.analysis.audio_summary import _read_records, summarize_records

    script = Path(__file__).resolve().parents[2] / "scripts/analysis/compare_audio_compression_runs.py"
    spec = importlib.util.spec_from_file_location("legacy_audio_comparison", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv = []
    for label, mode in (("baseline", "none"), ("compressed", "compressed")):
        manifest = _profile(tmp_path / label, mode)
        trace = manifest["trace_paths"]["lmms_eval_diagnostic_0"]
        efficiency = summarize_records(_read_records([Path(trace)]))
        efficiency["source_files"] = [trace]
        for kind, payload in (
            ("manifest", manifest), ("efficiency", efficiency),
            ("result", {"results": {"mmau": {"accuracy": 0.7}}}),
        ):
            path = tmp_path / f"{label}_{kind}.json"
            _write(path, payload)
            argv.extend([f"--{label}_{kind}", str(path)])
    output = tmp_path / "comparison.json"
    assert module.main([*argv, "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["evaluation_protocol"]["matched"] is True
    assert len(report["sources"]) == 6
    assert all(Path(item["path"]).is_absolute() and len(item["sha256"]) == 64 for item in report["sources"].values())


def test_analysis_preserves_summaries_without_inventing_metric_semantics(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["baseline_metrics"] = []
    manifest["eval_backend"] = None
    manifest["analysis_records"] = [{
        "name": "entropy", "phase": "pre_compression", "method": "entropy",
        "summary": {"per_layer": [0.5, 0.4]}, "sample_count": 2, "artifact_paths": {},
    }]
    report = compare_runs(
        _run(tmp_path / "left", manifest, task="analyze"),
        _run(tmp_path / "right", manifest, task="analyze"),
    )
    assert report["metrics"] == []
    assert report["analysis_records"]["left"] == manifest["analysis_records"]
    assert report["protocol"]["matched"] is None
    assert "runtime_versions" in report["protocol"]["missing"]["left"]


def test_discovery_exposes_rules_from_the_selected_variant() -> None:
    items = discover_capabilities(name="researcher_ltbm_audio", phase="generate",
        extensions=["examples.compression_extensions.ltbm_audio:register"])["items"]
    assert len(items) == 1
    assert items[0]["supported_rules"] == ["aggregator", "grouper"]
    replaced = discover_capabilities(name="researcher_first_k_audio", phase="generate",
        extensions=["examples.compression_extensions.first_k_audio:register"])["items"]
    assert replaced[0]["supported_rules"] == []
