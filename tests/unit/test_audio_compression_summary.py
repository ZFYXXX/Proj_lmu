"""Tests for the audio-compression diagnostic summarizer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mllm_comp.analysis import audio_summary as module


def test_summary_aggregates_tokens_latency_and_order() -> None:
    records = [
        {
            "sample_id": "a",
            "runtime_environment": {
                "device": "cuda:0",
                "gpu_name": "fixture-gpu",
                "packages": {"transformers": "4.57.3", "torch": "2.8.0"},
            },
            "generation_metadata": {
                "runtime_mode": "profile",
                "compression_mode": "compressed",
                "compression": {
                    "original_audio_tokens": 100,
                    "kept_audio_tokens": 25,
                    "attention_score_seconds": 0.01,
                    "items": [{"audio_unit_index": 0, "indices_chronological": True}],
                },
                "timing": {
                    "generation_seconds": 1.0,
                    "prefill_seconds": 0.2,
                    "ttft_seconds": 0.25,
                    "decode_seconds_per_token": 0.05,
                    "llm_input_tokens": 40,
                    "cuda_memory_allocated_before_bytes": 1000,
                    "cuda_max_memory_allocated_bytes": 1400,
                    "cuda_peak_memory_delta_bytes": 400,
                },
            },
        },
        {
            "sample_id": "b",
            "runtime_environment": {
                "device": "cuda:1",
                "gpu_name": "fixture-gpu",
                "packages": {"transformers": "4.57.3", "torch": "2.8.0"},
            },
            "generation_metadata": {
                "runtime_mode": "profile",
                "compression_mode": "compressed",
                "compression": {
                    "original_audio_tokens": 80,
                    "kept_audio_tokens": 20,
                    "attention_score_seconds": 0.03,
                    "items": [{"audio_unit_index": 0, "indices_chronological": True}],
                },
                "timing": {
                    "generation_seconds": 2.0,
                    "prefill_seconds": 0.4,
                    "ttft_seconds": 0.5,
                    "decode_seconds_per_token": 0.07,
                    "llm_input_tokens": 35,
                    "cuda_memory_allocated_before_bytes": 1000,
                    "cuda_max_memory_allocated_bytes": 1300,
                    "cuda_peak_memory_delta_bytes": 300,
                },
            },
        },
    ]

    summary = module.summarize_records(records)

    assert summary["token_summary"]["original_audio_tokens"] == 180
    assert summary["token_summary"]["kept_audio_tokens"] == 45
    assert summary["token_summary"]["achieved_keep_ratio"] == 0.25
    assert summary["num_timing_records"] == 2
    assert summary["compression_modes"] == ["compressed"]
    assert summary["runtime_modes"] == ["profile"]
    assert summary["timing_seconds"]["prefill"]["median"] == pytest.approx(0.3)
    assert summary["timing_seconds"]["decode_per_token"]["median"] == pytest.approx(0.06)
    assert summary["llm_input_tokens"]["median"] == 37.5
    assert summary["cuda_memory_bytes"]["peak_allocation_delta"]["median"] == 350
    assert summary["token_summary"]["chronological_index_violations"] == []
    assert len(summary["runtime_environments"]) == 2


def test_summary_rejects_mixed_baseline_and_compressed_records() -> None:
    records = [
        {
            "generation_metadata": {
                "runtime_mode": "profile",
                "compression_mode": "none",
                "timing": {},
            }
        },
        {
            "generation_metadata": {
                "runtime_mode": "profile",
                "compression_mode": "compressed",
                "timing": {},
            }
        },
    ]

    with pytest.raises(ValueError, match="mix baseline and compressed"):
        module.summarize_records(records)


def test_summary_rejects_debug_records() -> None:

    with pytest.raises(ValueError, match="runtime_mode=profile"):
        module.summarize_records(
            [{"generation_metadata": {"runtime_mode": "debug", "timing": None}}]
        )


def test_run_directory_uses_plugin_world_size_and_ignores_stale_ranks(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "compression_diagnostic.jsonl"
    (tmp_path / "mllm_comp_lmms_plugin.json").write_text(
        json.dumps(
            {
                "diagnostic_trace_path": str(trace),
                "expected_world_size": 2,
                "runtime_mode": "profile",
            }
        ),
        encoding="utf-8",
    )
    expected = []
    for rank in range(3):
        ranked = trace.with_name(f"{trace.stem}.rank{rank}{trace.suffix}")
        ranked.write_text("{}\n", encoding="utf-8")
        if rank < 2:
            expected.append(ranked)

    assert module._resolve_paths([str(tmp_path)]) == expected


def test_run_directory_rejects_missing_expected_rank_trace(
    tmp_path: Path,
) -> None:
    trace = tmp_path / "baseline_diagnostic.jsonl"
    (tmp_path / "mllm_comp_lmms_plugin.json").write_text(
        json.dumps(
            {
                "diagnostic_trace_path": str(trace),
                "expected_world_size": 2,
                "runtime_mode": "profile",
            }
        ),
        encoding="utf-8",
    )
    trace.with_name(f"{trace.stem}.rank0{trace.suffix}").write_text(
        "{}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing expected diagnostic"):
        module._resolve_paths([str(tmp_path)])
