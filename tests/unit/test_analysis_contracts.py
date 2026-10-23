from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Iterator
import weakref

import pytest

import mllm_comp.analysis.common.streaming as streaming_module
from mllm_comp.analysis.artifacts import (
    ANALYSIS_ARTIFACT_SCHEMA_VERSION,
    make_analysis_record,
    write_analysis_artifact,
    write_analysis_sample_artifact,
    write_streaming_analysis_result,
)
from mllm_comp.analysis.common.rank import aggregate_task_percentiles
from mllm_comp.analysis.common.streaming import StreamingAnalysisRecords
from mllm_comp.analysis.definition import (
    AnalysisOverrides,
    resolve_analysis_definition,
)
from mllm_comp.config.loader import load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import ConfigError
from mllm_comp.integrations.lmms_eval.adapter import _analysis_target
from mllm_comp.run_manifest import build_run_manifest


def _config() -> object:
    return experiment_config_from_dict(
        {
            "experiment": {"name": "analysis-contract"},
            "backbone": {
                "name": "qwen25_vl",
                "model_id": "Qwen/Qwen2.5-VL-3B-Instruct",
                "model_revision": "revision",
            },
            "pipeline": {"stages": []},
            "analysis": {
                "backend": "lmms_eval",
                "tasks": ["configured-task"],
                "sample_limit": 8,
                "probes": [
                    {
                        "name": "representation_dynamics",
                        "layers": [1],
                        "metrics": ["cosine"],
                        "token_groups": ["text", "vision"],
                    }
                ],
            },
            "train": {},
            "eval": {
                "tasks": ["eval-task"],
                "checkpoint_path": "different-eval-checkpoint",
            },
            "data": {"dataset_id": "fixture"},
            "output": {
                "run_dir": "runs/analysis-contract",
                "manifest_path": "runs/analysis-contract/manifest.json",
            },
        }
    )


def test_analysis_cli_overrides_win_without_mutating_config() -> None:
    config = _config()
    result = resolve_analysis_definition(
        config,
        AnalysisOverrides(
            tasks="cli-task",
            sample_limit=2,
            layers="0,2",
            metrics="cosine",
            save_raw=True,
        ),
    )

    assert result.tasks == ["cli-task"]
    assert result.sample_limit == 2
    assert result.probes[0]["layers"] == [0, 2]
    assert result.probes[0]["metrics"] == ["cosine"]
    assert result.probes[0]["save_raw"] is True
    assert config.analysis.tasks == ["configured-task"]
    assert config.analysis.probes[0]["layers"] == [1]


@pytest.mark.parametrize("sample_limit", [0, -1, True])
def test_analysis_definition_rejects_nonpositive_or_boolean_sample_limit(
    sample_limit: object,
) -> None:
    with pytest.raises(ConfigError, match="sample_limit"):
        resolve_analysis_definition(
            _config(),
            AnalysisOverrides(sample_limit=sample_limit),  # type: ignore[arg-type]
        )


def test_analysis_definition_rejects_duplicate_logical_probe_names() -> None:
    config = _config()
    config.analysis.probes.append(
        {
            "name": "representation_dynamics",
            "method": "representation_dynamics",
            "metrics": ["cosine"],
        }
    )

    with pytest.raises(ConfigError, match="duplicated"):
        resolve_analysis_definition(config, AnalysisOverrides())


def test_analysis_definition_rejects_ambiguous_enabled_value() -> None:
    config = _config()
    config.analysis.probes[0]["enabled"] = "false"

    with pytest.raises(ConfigError, match="enabled must be a boolean"):
        resolve_analysis_definition(config, AnalysisOverrides())


def test_analysis_definition_rejects_unsupported_backbone() -> None:
    config = _config()
    config.backbone.name = "unsupported_model"

    with pytest.raises(ConfigError, match="Unknown backbone capability"):
        resolve_analysis_definition(config, AnalysisOverrides())


def test_compressed_analysis_rejects_an_empty_pipeline() -> None:
    with pytest.raises(ConfigError, match="at least one enabled pipeline stage"):
        resolve_analysis_definition(
            _config(),
            AnalysisOverrides(compression_mode="compressed"),
        )


def test_artifact_envelope_and_record_are_manifest_ready(
    tmp_path: Path,
) -> None:
    artifact_path = write_analysis_artifact(
        output_dir=tmp_path,
        filename="representation_dynamics.aggregate.json",
        artifact_kind="aggregate",
        probe_name="representation_dynamics",
        method="representation_dynamics",
        method_version="1.0",
        phase="pre_compression",
        backbone_name="qwen25_vl",
        checkpoint="Qwen/Qwen2.5-VL-3B-Instruct",
        checkpoint_revision="revision",
        capability_status="active",
        payload={"layers": {"0": {"vision": {"cosine": 0.5}}}},
        metadata={"task_count": 1},
    )
    payload = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
    assert payload["schema_version"] == ANALYSIS_ARTIFACT_SCHEMA_VERSION
    assert payload["artifact_kind"] == "aggregate"
    assert payload["backbone"]["revision"] == "revision"
    assert payload["probe"]["capability_status"] == "active"
    assert "candidate_validation" not in payload["probe"]

    record = make_analysis_record(
        name="representation_dynamics",
        phase="pre_compression",
        method="representation_dynamics",
        summary={"successful_samples": 1, "skipped_samples": 0},
        artifact_paths={"aggregate": artifact_path},
        sample_count=1,
    )
    manifest = build_run_manifest(_config(), analysis_records=[record])

    assert manifest.analysis_records == [record]
    assert manifest.eval_tasks == ["configured-task"]
    assert manifest.model_path_or_id == "Qwen/Qwen2.5-VL-3B-Instruct"


def test_artifact_writer_rejects_path_escape(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="safe .json basename"):
        write_analysis_artifact(
            output_dir=tmp_path,
            filename="../escape.json",
            artifact_kind="raw",
            probe_name="probe",
            method="method",
            method_version=None,
            phase="pre_compression",
            backbone_name="qwen25_vl",
            checkpoint="checkpoint",
            checkpoint_revision=None,
            payload={},
        )


def test_streaming_records_bound_examples_and_preserve_raw_artifact(
    tmp_path: Path,
) -> None:
    class SumAccumulator:
        def __init__(self) -> None:
            self.count = 0
            self.total = 0.0

        def add(self, sample: dict[str, object]) -> None:
            self.count += 1
            self.total += float(sample["value"])

        def result(self) -> dict[str, object]:
            return {"sample_count": self.count, "total": self.total}

    records = StreamingAnalysisRecords(
        output_dir=tmp_path,
        prefix="streaming-contract",
        save_raw=True,
        accumulator_factory=SumAccumulator,
        aggregate_by_task=True,
        skipped_example_limit=2,
    )
    try:
        for index in range(6):
            records.add_success(
                {
                    "sample_id": f"sample-{index}",
                    "task_name": f"task-{index % 2}",
                    "value": index,
                }
            )
        for index in range(5):
            records.add_skipped(
                {
                    "sample_id": f"skipped-{index}",
                    "task_name": "task-0" if index < 4 else "task-skipped",
                    "reason": f"reason-{index}",
                }
            )

        assert records.overall_result() == {"sample_count": 6, "total": 15.0}
        assert records.task_results() == {
            "task-0": {
                "sample_count": 3,
                "total": 6.0,
                "skipped_sample_count": 4,
            },
            "task-1": {
                "sample_count": 3,
                "total": 9.0,
                "skipped_sample_count": 0,
            },
            "task-skipped": {
                "sample_count": 0,
                "total": 0.0,
                "skipped_sample_count": 1,
            },
        }
        assert records.skipped_evidence() == {
            "skipped_samples": [
                {
                    "sample_id": "skipped-0",
                    "task_name": "task-0",
                    "reason": "reason-0",
                },
                {
                    "sample_id": "skipped-1",
                    "task_name": "task-0",
                    "reason": "reason-1",
                },
            ],
            "skipped_sample_count": 5,
            "skipped_samples_truncated": True,
            "skipped_sample_counts_by_task": {
                "task-0": 4,
                "task-skipped": 1,
            },
        }
        evidence = records.execution_evidence()
        assert evidence["retained_successful_sample_reports"] == 0
        assert evidence["raw_records_disk_spooled"] is True

        record = write_streaming_analysis_result(
            output_dir=tmp_path,
            filename_stem="streaming",
            probe_name="renamed_streaming",
            method="streaming",
            method_version="1.0.0",
            phase="pre_compression",
            backbone_name="qwen25_vl",
            checkpoint="checkpoint",
            checkpoint_revision="revision",
            capability_status="external",
            records=records,
            aggregate=records.overall_result(),
            summary={"total": 15.0},
            aggregate_metadata={"intervention_count": 3},
        )
        assert record.name == "renamed_streaming" and record.method == "streaming"
        assert record.sample_count == 6
        assert record.summary == {
            "successful_samples": 6, "skipped_samples": 5, "total": 15.0,
            "aggregation_contract": "online_global_and_task_accumulators_v1",
        }
        assert record.artifact_paths == {
            "aggregate": str(tmp_path / "streaming.aggregate.json"),
            "per_sample": str(tmp_path / "streaming.raw.json"),
        }
        aggregate = json.loads(
            Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
        )
        assert aggregate["payload"] == {"sample_count": 6, "total": 15.0}
        assert aggregate["metadata"] == {
            "successful_samples": 6, "skipped_samples": 5, "intervention_count": 3,
        }
        payload = json.loads(
            Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
        )
        assert payload["probe"] == aggregate["probe"] == {
            "name": "renamed_streaming", "method": "streaming",
            "method_version": "1.0.0", "capability_status": "external",
        }
        assert payload["metadata"] == {"sample_count": 6}
        assert [item["sample_id"] for item in payload["payload"]["samples"]] == [
            f"sample-{index}" for index in range(6)
        ]
        assert len(payload["payload"]["skipped_samples"]) == 5
    finally:
        sample_spool = records._sample_spool
        skipped_spool = records._skipped_spool
        records.close()

    assert sample_spool is not None and not sample_spool.path.exists()
    assert skipped_spool is not None and not skipped_spool.path.exists()


def test_streaming_records_do_not_spool_when_raw_output_is_disabled(
    tmp_path: Path,
) -> None:
    class CountAccumulator:
        def __init__(self) -> None:
            self.count = 0

        def add(self, _sample: dict[str, object]) -> None:
            self.count += 1

        def result(self) -> dict[str, int]:
            return {"sample_count": self.count}

    output_dir = tmp_path / "no-raw"
    records = StreamingAnalysisRecords(
        output_dir=output_dir,
        prefix="no-raw",
        save_raw=False,
        accumulator_factory=CountAccumulator,
        aggregate_by_task=False,
        skipped_example_limit=1,
    )
    try:
        class Marker:
            pass

        marker = Marker()
        marker_reference = weakref.ref(marker)
        marker_sample = {
            "sample_id": "marker",
            "task_name": "task",
            "value": marker,
        }
        records.add_success(marker_sample)
        del marker_sample
        del marker
        gc.collect()
        assert marker_reference() is None

        for index in range(100):
            records.add_success(
                {
                    "sample_id": str(index),
                    "task_name": "task",
                    "value": "x" * 1000,
                }
            )
        records.add_skipped({"sample_id": "a", "reason": "first"})
        records.add_skipped({"sample_id": "b", "reason": "second"})

        assert records.overall_result() == {"sample_count": 101}
        assert len(records.skipped_examples) == 1
        assert records.skipped_count == 2
        assert not output_dir.exists()
        assert records.execution_evidence()["raw_records_disk_spooled"] is False
    finally:
        records.close()


def test_streaming_record_construction_rolls_back_first_spool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    created: list[object] = []
    calls = 0

    class FakeSpool:
        def __init__(self, _output_dir: object, *, prefix: str) -> None:
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("second spool failed")
            self.prefix = prefix
            self.closed = False
            created.append(self)

        def close(self) -> None:
            self.closed = True

    monkeypatch.setattr(streaming_module, "JsonlRecordSpool", FakeSpool)

    with pytest.raises(OSError, match="second spool failed"):
        StreamingAnalysisRecords(
            output_dir=tmp_path,
            prefix="rollback",
            save_raw=True,
            accumulator_factory=lambda: None,
            aggregate_by_task=False,
        )

    assert len(created) == 1
    assert created[0].closed is True


def test_streaming_record_close_attempts_both_spools_and_can_retry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class CountAccumulator:
        def add(self, _sample: dict[str, object]) -> None:
            pass

        def result(self) -> dict[str, int]:
            return {"sample_count": 0}

    records = StreamingAnalysisRecords(
        output_dir=tmp_path,
        prefix="retry-close",
        save_raw=True,
        accumulator_factory=CountAccumulator,
        aggregate_by_task=False,
    )
    sample_spool = records._sample_spool
    skipped_spool = records._skipped_spool
    assert sample_spool is not None and skipped_spool is not None
    original_close = sample_spool.close
    calls = 0

    def fail_once() -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("sample cleanup failed")
        original_close()

    monkeypatch.setattr(sample_spool, "close", fail_once)

    with pytest.raises(OSError, match="sample cleanup failed"):
        records.close()

    assert sample_spool.path.exists()
    assert not skipped_spool.path.exists()
    records.close()
    assert not sample_spool.path.exists()


def test_streaming_artifact_failure_removes_unpublished_temp_file(
    tmp_path: Path,
) -> None:
    def failing_samples() -> Iterator[dict[str, str]]:
        yield {"sample_id": "first"}
        raise RuntimeError("sample stream failed")

    target = tmp_path / "failed.raw.json"
    with pytest.raises(RuntimeError, match="sample stream failed"):
        write_analysis_sample_artifact(
            output_dir=tmp_path,
            filename=target.name,
            artifact_kind="per_sample",
            probe_name="streaming",
            method="streaming",
            method_version="1.0.0",
            phase="pre_compression",
            backbone_name="qwen25_vl",
            checkpoint="checkpoint",
            checkpoint_revision="revision",
            samples=failing_samples(),
            skipped_samples=(),
        )

    assert not target.exists()
    assert list(tmp_path.glob("*.tmp")) == []


def test_task_percentiles_preserve_raw_scores_and_average_ties() -> None:
    result = aggregate_task_percentiles(
        {
            "task-a": {"a": 1.0, "b": 2.0, "c": 2.0},
            "task-b": {"a": 3.0, "b": 1.0, "c": float("nan")},
        }
    )

    assert result["components"]["a"]["raw_scores_by_task"] == {
        "task-a": 1.0,
        "task-b": 3.0,
    }
    assert result["components"]["b"]["percentiles_by_task"]["task-a"] == (
        2.5 / 3.0
    )
    assert result["components"]["c"]["task_count"] == 1
    assert result["skipped"] == [
        {
            "task": "task-b",
            "component": "c",
            "reason": "non_finite_raw_score",
        }
    ]


def test_lmms_target_contract_does_not_choose_from_multiple_answers() -> None:
    class _Task:
        def doc_to_target(self, _doc: object) -> list[str]:
            return ["first", "second"]

    target, reason = _analysis_target(_Task(), {}, task_name="fixture")

    assert target is None
    assert reason is not None and "non-string" in reason


def test_textvqa_analysis_target_uses_deterministic_plurality_reference() -> None:
    class _Task:
        def doc_to_target(self, _doc: object) -> str:
            return "answer"  # broken upstream config value for this dataset

    target, reason = _analysis_target(
        _Task(),
        {"answers": ["Dakota", "dakota", "nous les gosses", "DAKOTA!"]},
        task_name="textvqa_val",
    )

    assert target == "Dakota"
    assert reason is None


@pytest.mark.parametrize(
    "path",
    sorted(Path("examples/analysis").rglob("*.yaml")),
    ids=lambda path: path.as_posix(),
)
def test_analysis_examples_resolve(path: Path) -> None:
    definition = resolve_analysis_definition(load_config(path), AnalysisOverrides())
    assert definition.probes


def test_analysis_matrix_supports_single_probe_override() -> None:
    definition = resolve_analysis_definition(
        load_config("examples/analysis/qwen25_vl_analysis_matrix.yaml"),
        AnalysisOverrides(probes=["hierarchical_taylor"]),
    )
    assert [probe["name"] for probe in definition.probes] == ["hierarchical_taylor"]
