"""Tests for the joint quality-efficiency comparison report."""

from __future__ import annotations

import pytest

from mllm_comp.analysis import audio_comparison as module


def _manifest(trace_path: str, *, max_new_tokens: int = 32) -> dict:
    return {
        "backbone_name": "qwen25_omni",
        "checkpoint_path": "/models/qwen25-omni",
        "source_commit": "0123456789abcdef",
        "source_dirty": True,
        "source_tree_hash": "a" * 64,
        "eval_tasks": ["mmau_test_mini"],
        "trace_paths": {"lmms_eval_diagnostic_0": trace_path},
        "eval_backend": {
            "name": "lmms_eval",
            "version": "0.7.2",
            "args": {
                "model_name": "qwen2_5_omni",
                "num_processes": 1,
                "distributed_launcher": None,
                "versions": {
                    "lmms_eval": "0.7.2",
                    "accelerate": "1.12.0",
                    "transformers": "4.57.3",
                    "torch": "2.8.0",
                },
                "lmms_eval_argv": [
                    "python",
                    "-m",
                    "lmms_eval",
                    "--batch_size",
                    "1",
                    "--seed",
                    "42",
                ],
                "plugin_spec": {
                    "device": "cuda:0",
                    "expected_world_size": 1,
                    "generation": {"max_new_tokens": max_new_tokens},
                    "system_prompt": None,
                    "use_audio_in_video": False,
                },
            },
        },
    }


def _efficiency(mode: str, trace_path: str) -> dict:
    payload = {
        "compression_modes": [mode],
        "num_trace_records": 2,
        "source_files": [trace_path],
        "runtime_environments": [
            {
                "packages": {"transformers": "4.57.3", "torch": "2.8.0"},
                "cuda_runtime": "12.8",
                "gpu_name": "fixture-gpu",
            }
        ],
        "timing_seconds": {
            "generation": {"median": 1.0},
            "prefill": {"median": 0.4},
        },
        "llm_input_tokens": {"median": 100},
        "cuda_memory_bytes": {"peak_allocation_delta": {"median": 800}},
    }
    if mode == "compressed":
        payload.update(
            {
                "num_compressed_records": 2,
                "token_summary": {"chronological_index_violations": []},
            }
        )
    return payload


def test_compare_runs_joins_common_quality_and_efficiency() -> None:
    baseline_efficiency = {
        "compression_modes": ["none"],
        "num_trace_records": 10,
        "timing_seconds": {
            "generation": {"median": 1.0},
            "prefill": {"median": 0.4},
        },
        "llm_input_tokens": {"median": 100},
        "cuda_memory_bytes": {"peak_allocation_delta": {"median": 800}},
    }
    compressed_efficiency = {
        "compression_modes": ["compressed"],
        "num_trace_records": 10,
        "num_compressed_records": 10,
        "timing_seconds": {
            "generation": {"median": 0.8},
            "prefill": {"median": 0.2},
            "attention_score": {"median": 0.01},
        },
        "llm_input_tokens": {"median": 60},
        "cuda_memory_bytes": {"peak_allocation_delta": {"median": 500}},
        "token_summary": {
            "original_audio_tokens": 1000,
            "kept_audio_tokens": 250,
            "achieved_keep_ratio": 0.25,
            "chronological_index_violations": [],
        },
    }

    result = module.compare_runs(
        baseline_result={"results": {"mmau": {"accuracy": 0.7}}},
        compressed_result={"results": {"mmau": {"accuracy": 0.68}}},
        baseline_efficiency=baseline_efficiency,
        compressed_efficiency=compressed_efficiency,
    )

    quality = result["quality_metrics"]["mmau.accuracy"]
    assert quality["absolute_delta"] == pytest.approx(-0.02)
    prefill = result["efficiency_metrics"]["timing_seconds.prefill.median"]
    assert prefill["baseline_over_compressed"] == 2.0
    assert result["token_totals"]["achieved_keep_ratio"] == 0.25
    assert result["compression_only_metrics"]["attention_score_seconds_median"] == 0.01
    assert result["evaluation_protocol"] is None


def test_compare_runs_requires_matching_manifest_protocol_and_trace_sources() -> None:
    baseline_trace = "/runs/baseline_diagnostic.jsonl"
    compressed_trace = "/runs/compression_diagnostic.jsonl"

    result = module.compare_runs(
        baseline_result={"results": {"mmau": {"accuracy": 0.7}}},
        compressed_result={"results": {"mmau": {"accuracy": 0.68}}},
        baseline_efficiency=_efficiency("none", baseline_trace),
        compressed_efficiency=_efficiency("compressed", compressed_trace),
        baseline_manifest=_manifest(baseline_trace),
        compressed_manifest=_manifest(compressed_trace),
    )

    assert result["evaluation_protocol"]["matched"] is True
    assert result["evaluation_protocol"]["identity"]["eval_tasks"] == [
        "mmau_test_mini"
    ]

    with pytest.raises(ValueError, match="different evaluation protocols"):
        module.compare_runs(
            baseline_result={"results": {"mmau": {"accuracy": 0.7}}},
            compressed_result={"results": {"mmau": {"accuracy": 0.68}}},
            baseline_efficiency=_efficiency("none", baseline_trace),
            compressed_efficiency=_efficiency("compressed", compressed_trace),
            baseline_manifest=_manifest(baseline_trace),
            compressed_manifest=_manifest(
                compressed_trace,
                max_new_tokens=64,
            ),
        )


def test_scoped_source_comparison_keeps_code_and_protocol_checks() -> None:
    baseline = _manifest("baseline.jsonl")
    compressed = _manifest("compressed.jsonl")
    baseline["source_scope"] = compressed["source_scope"] = "runtime_sources_v1"
    compressed["source_commit"] = "different-docs-commit"
    compressed["source_dirty"] = False
    expected = module._evaluation_protocol(baseline, label="baseline")
    assert module._evaluation_protocol(compressed, label="compressed") == expected
    compressed["source_tree_hash"] = "b" * 64
    assert module._evaluation_protocol(compressed, label="compressed") != expected
    compressed["source_tree_hash"] = baseline["source_tree_hash"]
    compressed["eval_backend"]["args"]["plugin_spec"]["generation"]["max_new_tokens"] = 64
    assert module._evaluation_protocol(compressed, label="compressed") != expected
    legacy = _manifest("old.jsonl")
    assert module._evaluation_protocol(legacy, label="legacy") != expected


def test_efficiency_trace_matching_excludes_compression_details_and_pretty_reports() -> None:
    manifest = _manifest("diagnostic.jsonl")
    manifest["trace_paths"].update({
        "compression_runtime_0": "compression.json",
        "compression_requests_0": "compression.requests.jsonl",
        "lmms_eval_diagnostic_pretty_0": "diagnostic.txt",
    })
    assert module._manifest_trace_paths(
        manifest, _efficiency("compressed", "diagnostic.jsonl"), label="compressed",
    ) == ["diagnostic.jsonl"]


def test_compare_runs_rejects_no_common_quality_metric() -> None:

    with pytest.raises(ValueError, match="share no numeric metrics"):
        module.compare_runs(
            baseline_result={"results": {"a": {"score": 1}}},
            compressed_result={"results": {"b": {"score": 1}}},
            baseline_efficiency={
                "compression_modes": ["none"],
                "num_trace_records": 1,
                "timing_seconds": {
                    "generation": {"median": 1},
                    "prefill": {"median": 1},
                },
                "llm_input_tokens": {"median": 1},
                "cuda_memory_bytes": {
                    "peak_allocation_delta": {"median": 1}
                },
            },
            compressed_efficiency={
                "compression_modes": ["compressed"],
                "num_trace_records": 1,
                "num_compressed_records": 1,
                "timing_seconds": {
                    "generation": {"median": 1},
                    "prefill": {"median": 1},
                },
                "llm_input_tokens": {"median": 1},
                "cuda_memory_bytes": {
                    "peak_allocation_delta": {"median": 1}
                },
                "token_summary": {"chronological_index_violations": []},
            },
        )
