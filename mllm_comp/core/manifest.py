"""Reproducibility manifest records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from .actions import OperationExportRecord


@dataclass
class CompressionStageRecord:
    """Manifest-facing identity of one logical compression stage."""

    stage_name: str
    plugin: str
    algorithm: str
    algorithm_version: str | None
    variant: str | None
    bindings: list[str]
    backbone: str
    placement: str
    execution_phase: str
    target_modalities: list[str]
    conditioning_modalities: list[str]
    preserved_modalities: list[str]
    coupling_policy: str
    media_alignment: str
    target_sites: list[str] = field(default_factory=list)
    support_status: str | None = None
    evidence: list[dict[str, Any]] = field(default_factory=list)
    layout_delta_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendRecord:
    name: str
    version: str
    config_path: str | None = None
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricRecord:
    task: str
    split: str | None
    metrics: dict[str, float]
    num_samples: int | None = None


@dataclass
class AnalysisRecord:
    name: str
    phase: Literal["pre_compression", "post_compression"]
    method: str
    summary: dict[str, Any]
    artifact_paths: dict[str, str]
    sample_count: int | None = None


@dataclass
class CompressionRunManifest:
    run_id: str
    created_at: str
    config_path: str
    config_hash: str
    source_commit: str | None
    env_lockfile_hash: str | None

    backbone_name: str
    backbone_version: str | None
    backbone_code_origin: str
    backbone_source_ref: str
    model_path_or_id: str

    compressor_name: str
    compressor_version: str | None
    compressor_config: dict[str, Any]

    train_backend: BackendRecord | None
    eval_backend: BackendRecord | None
    dataset_id: str | None
    dataset_commit: str | None
    eval_tasks: list[str]

    # V2 keeps the legacy single-compressor projection above while recording
    # each logical stage and exact support claim independently.
    schema_version: int = 2
    compression_stages: list[CompressionStageRecord] = field(default_factory=list)

    baseline_metrics: list[MetricRecord] = field(default_factory=list)
    compressed_metrics: list[MetricRecord] = field(default_factory=list)
    analysis_records: list[AnalysisRecord] = field(default_factory=list)
    operation_export_records: list[OperationExportRecord] = field(default_factory=list)
    per_stage_stats: dict[str, dict[str, Any]] = field(default_factory=dict)
    invariants_passed: bool = False
    invariants_run: list[str] = field(default_factory=list)

    artifacts_dir: str = ""
    checkpoint_path: str | None = None
    trace_paths: dict[str, str] = field(default_factory=dict)
    source_dirty: bool | None = None
    source_tree_hash: str | None = None
    source_scope: str | None = None
    # Loader-compatible values after overrides; provenance remains in config_path.
    effective_config: dict[str, Any] = field(default_factory=dict)
