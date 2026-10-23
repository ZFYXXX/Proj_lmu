"""Bounded-memory lifecycle helpers for sample-oriented analysis Probes."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterator, Mapping, Protocol

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.serialization import to_jsonable


SKIPPED_SAMPLE_EXAMPLE_LIMIT = 20
STREAMING_AGGREGATION_CONTRACT = "online_global_and_task_accumulators_v1"


class SampleAccumulator(Protocol):
    def add(self, sample: Mapping[str, Any]) -> None: ...

    def result(self) -> dict[str, Any]: ...


class JsonlRecordSpool:
    """Append JSON records to a temporary file and iterate one record at a time."""

    def __init__(self, output_dir: str | Path, *, prefix: str) -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=directory,
            prefix=f".{prefix}.",
            suffix=".jsonl.tmp",
            delete=False,
        ) as handle:
            self.path = Path(handle.name)
        self._count = 0
        self._closed = False

    def append(self, record: Mapping[str, Any]) -> None:
        self.append_serialized(self.serialize(record))

    @staticmethod
    def serialize(record: Mapping[str, Any]) -> str:
        try:
            return json.dumps(
                to_jsonable(dict(record)),
                ensure_ascii=False,
                sort_keys=True,
            )
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(
                f"Analysis record is not JSON serializable: {exc}"
            ) from exc

    def append_serialized(self, serialized: str) -> None:
        if self._closed:
            raise BackendIntegrationError("Cannot append to a closed analysis spool.")
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(serialized)
            handle.write("\n")
        self._count += 1

    def __len__(self) -> int:
        return self._count

    def __iter__(self) -> Iterator[dict[str, Any]]:
        if self._closed:
            raise BackendIntegrationError("Cannot read a closed analysis spool.")
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise BackendIntegrationError(
                        "Analysis spool record must be a JSON object "
                        f"(line {line_number})."
                    )
                yield value

    def close(self) -> None:
        if self._closed:
            return
        self.path.unlink(missing_ok=True)
        self._closed = True


class StreamingAnalysisRecords:
    """Maintain online aggregate state and optional disk-backed raw records."""

    def __init__(
        self,
        *,
        output_dir: str | Path,
        prefix: str,
        save_raw: bool,
        accumulator_factory: Callable[[], SampleAccumulator],
        aggregate_by_task: bool,
        skipped_example_limit: int = SKIPPED_SAMPLE_EXAMPLE_LIMIT,
    ) -> None:
        if skipped_example_limit < 0:
            raise ValueError("skipped_example_limit cannot be negative.")
        self._accumulator_factory = accumulator_factory
        self._overall = accumulator_factory()
        self._task_accumulators: dict[str, SampleAccumulator] = {}
        self._task_skipped_counts: dict[str, int] = {}
        self._aggregate_by_task = aggregate_by_task
        self._successful_count = 0
        self._skipped_count = 0
        self._skipped_example_limit = skipped_example_limit
        self._skipped_examples: list[dict[str, Any]] = []
        self._save_raw = save_raw
        self._sample_spool: JsonlRecordSpool | None = None
        self._skipped_spool: JsonlRecordSpool | None = None
        if save_raw:
            sample_spool = JsonlRecordSpool(
                output_dir,
                prefix=f"{prefix}.samples",
            )
            try:
                skipped_spool = JsonlRecordSpool(
                    output_dir,
                    prefix=f"{prefix}.skipped",
                )
            except Exception:
                sample_spool.close()
                raise
            self._sample_spool = sample_spool
            self._skipped_spool = skipped_spool
        self._closed = False

    @property
    def successful_count(self) -> int:
        return self._successful_count

    @property
    def skipped_count(self) -> int:
        return self._skipped_count

    @property
    def save_raw(self) -> bool:
        return self._save_raw

    @property
    def skipped_examples(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._skipped_examples)

    def add_success(self, sample: Mapping[str, Any]) -> None:
        self._ensure_open()
        task_value = sample.get("task_name")
        task_name = "" if task_value is None else str(task_value)
        if self._aggregate_by_task and not task_name:
            raise BackendIntegrationError(
                "Task-level analysis aggregation requires a non-empty task_name."
            )
        serialized = (
            self._sample_spool.serialize(sample)
            if self._sample_spool is not None
            else None
        )
        self._overall.add(sample)
        if self._aggregate_by_task:
            accumulator = self._task_accumulator(task_name)
            accumulator.add(sample)
        if self._sample_spool is not None and serialized is not None:
            self._sample_spool.append_serialized(serialized)
        self._successful_count += 1

    def add_skipped(self, record: Mapping[str, Any]) -> None:
        self._ensure_open()
        normalized = to_jsonable(dict(record))
        if not isinstance(normalized, dict):  # pragma: no cover - dict stays dict
            raise BackendIntegrationError("Skipped sample evidence must be an object.")
        serialized = (
            self._skipped_spool.serialize(normalized)
            if self._skipped_spool is not None
            else None
        )
        if len(self._skipped_examples) < self._skipped_example_limit:
            self._skipped_examples.append(normalized)
        task_value = normalized.get("task_name")
        task_name = "" if task_value is None else str(task_value)
        if self._aggregate_by_task and task_name:
            self._task_accumulator(task_name)
            self._task_skipped_counts[task_name] = (
                self._task_skipped_counts.get(task_name, 0) + 1
            )
        if self._skipped_spool is not None and serialized is not None:
            self._skipped_spool.append_serialized(serialized)
        self._skipped_count += 1

    def overall_result(self) -> dict[str, Any]:
        self._ensure_open()
        return self._overall.result()

    def task_results(self) -> dict[str, dict[str, Any]]:
        self._ensure_open()
        results: dict[str, dict[str, Any]] = {}
        for task in sorted(self._task_accumulators):
            result = self._task_accumulators[task].result()
            result["skipped_sample_count"] = self._task_skipped_counts.get(task, 0)
            results[task] = result
        return results

    def skipped_evidence(self) -> dict[str, Any]:
        return {
            "skipped_samples": list(self._skipped_examples),
            "skipped_sample_count": self._skipped_count,
            "skipped_samples_truncated": (
                self._skipped_count > len(self._skipped_examples)
            ),
            "skipped_sample_counts_by_task": {
                task: self._task_skipped_counts[task]
                for task in sorted(self._task_skipped_counts)
            },
        }

    def execution_evidence(self) -> dict[str, Any]:
        return {
            "contract": STREAMING_AGGREGATION_CONTRACT,
            "task_aggregation": self._aggregate_by_task,
            "retained_successful_sample_reports": 0,
            "skipped_sample_example_limit": self._skipped_example_limit,
            "raw_records_disk_spooled": self._save_raw,
        }

    def iter_raw_samples(self) -> Iterator[dict[str, Any]]:
        if self._sample_spool is None:
            raise BackendIntegrationError(
                "Raw sample iteration requires save_raw: true."
            )
        yield from self._sample_spool

    def iter_raw_skipped(self) -> Iterator[dict[str, Any]]:
        if self._skipped_spool is None:
            raise BackendIntegrationError(
                "Raw skipped-sample iteration requires save_raw: true."
            )
        yield from self._skipped_spool

    def close(self) -> None:
        if self._closed:
            return
        errors: list[Exception] = []
        for spool in (self._sample_spool, self._skipped_spool):
            if spool is None:
                continue
            try:
                spool.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]
        self._closed = True

    def _task_accumulator(self, task_name: str) -> SampleAccumulator:
        accumulator = self._task_accumulators.get(task_name)
        if accumulator is None:
            accumulator = self._accumulator_factory()
            self._task_accumulators[task_name] = accumulator
        return accumulator

    def _ensure_open(self) -> None:
        if self._closed:
            raise BackendIntegrationError("Analysis records are already closed.")


__all__ = [
    "JsonlRecordSpool",
    "SKIPPED_SAMPLE_EXAMPLE_LIMIT",
    "STREAMING_AGGREGATION_CONTRACT",
    "SampleAccumulator",
    "StreamingAnalysisRecords",
]
