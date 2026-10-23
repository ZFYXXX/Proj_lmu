"""Create command-level manifests from effective experiment configuration."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, TYPE_CHECKING

from mllm_comp.compressors.contracts import CompressionContractRegistry
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.compressors.rules import legacy_rule_specs, recipe_changed, resolve_rule_binding, rule_specs
from mllm_comp.config import ExperimentConfig, config_to_dict
from mllm_comp.core.manifest import (
    AnalysisRecord,
    BackendRecord,
    CompressionRunManifest,
    CompressionStageRecord,
    MetricRecord,
)
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.vocabulary import CompressionMode, EvalTarget
from mllm_comp.registry.catalog import CapabilityKind, get_capability_spec

if TYPE_CHECKING:
    from mllm_comp.pipeline.definition import CompressionDefinition
    from mllm_comp.pipeline.v2_validation import V2ContractProjection


def _effective_config_payload(config: ExperimentConfig) -> dict[str, Any]:
    # Preserve path in the historical identity hash, while sharing the same
    # reloadable configuration fields with the experiment entry point.
    return {**config_to_dict(config), "path": config.path}


def _config_hash(effective_config: Mapping[str, Any]) -> str:
    payload = json.dumps(
        effective_config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


SOURCE_SCOPE = "runtime_sources_v1"
_SOURCE_SUFFIXES = frozenset({
    ".py", ".json", ".yaml", ".yml", ".toml", ".txt", ".sh",
    ".jinja", ".jinja2", ".patch", ".diff",
})


def _source_git(root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )


def _is_source_file(raw: bytes) -> bool:
    return bool(raw) and (
        Path(raw.decode("utf-8", errors="surrogateescape")).suffix in _SOURCE_SUFFIXES
        or raw == b".gitmodules"
    )


def _hash_source_files(root: Path, paths: tuple[str, ...], digest: Any) -> bool:
    """Read only runtime source/resource files in the specified Git paths."""

    files = _source_git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard", "--", *paths)
    status = _source_git(root, "status", "--porcelain", "-z", "--no-renames", "--untracked-files=all", "--ignore-submodules=all", "--", *paths)
    if files.returncode or status.returncode:
        raise OSError("Cannot identify runtime sources")
    selected = {
        raw for raw in files.stdout.split(b"\0") if _is_source_file(raw)
    }
    for raw in sorted(selected):
        path = root / raw.decode("utf-8", errors="surrogateescape")
        # A deletion has the same effective contents before and after staging.
        if not path.is_symlink() and not path.is_file():
            continue
        digest.update(raw + b"\0")
        if path.is_symlink():
            digest.update(b"SYMLINK\0" + str(path.readlink()).encode("utf-8", errors="surrogateescape"))
        elif path.is_file():
            digest.update(b"FILE\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
        digest.update(b"\0")
    return any(_is_source_file(entry[3:]) for entry in status.stdout.split(b"\0") if entry)


def _git_source_identity(
    config: ExperimentConfig,
    *,
    backend: str | None = None,
) -> tuple[str | None, bool | None, str | None]:
    """Identify the executing package and applicable vendored runtime sources.

    Config location, documentation, tests and run artifacts do not define code
    identity. The effective configuration is already recorded separately.
    """

    root = Path(__file__).resolve().parents[1]
    # An installed package must not accidentally identify an enclosing unrelated
    # repository (or the repository containing an external experiment config).
    if not (root / ".git").exists():
        return None, None, None
    try:
        commit_result = _source_git(root, "rev-parse", "HEAD")
        if commit_result.returncode:
            return None, None, None
        digest = hashlib.sha256(SOURCE_SCOPE.encode("ascii") + b"\0")
        dirty = _hash_source_files(
            root, ("mllm_comp", "scripts", "patches", "pyproject.toml", ".gitmodules"), digest,
        )
        vendors = []
        if config.backbone.name == "llava_onevision":
            vendors.append(("third_party/backbones/LLaVA-NeXT", "llava"))
        if (backend or config.eval.backend) == "lmms_eval":
            vendors.append(("third_party/frameworks/lmms-eval", "lmms_eval"))
        if (backend or config.train.backend) == "ms_swift":
            vendors.append(("third_party/frameworks/ms-swift", "swift"))
        for relative, package in vendors:
            vendor = root / relative
            digest.update(relative.encode("utf-8") + b"\0")
            if (vendor / ".git").exists():
                dirty |= _hash_source_files(
                    vendor, (package, "pyproject.toml", "setup.py", "setup.cfg", "requirements"), digest,
                )
            else:
                # Uninitialized vendors retain their pinned Git identity; no
                # filesystem walk or model/data reads are needed.
                pinned = _source_git(root, "ls-tree", "HEAD", "--", relative)
                if pinned.returncode:
                    return None, None, None
                digest.update(b"UNINITIALIZED\0" + pinned.stdout)
        commit = commit_result.stdout.decode("ascii", errors="replace").strip()
        return commit or None, dirty, digest.hexdigest()
    except OSError:
        return None, None, None


def _compression_identity(
    config: ExperimentConfig,
    definition: CompressionDefinition | None = None,
) -> tuple[str, dict[str, Any]]:
    enabled = (
        definition.stages if definition is not None
        else [stage for stage in config.pipeline.stages if stage.enabled]
    )
    if enabled:
        return "+".join(stage.plugin for stage in enabled), {
            "stages": [to_jsonable(stage) for stage in enabled]
        }
    modes = {definition.compression_mode} if definition is not None else {
        config.analysis.compression_mode,
        config.train.compression_mode,
        config.eval.compression_mode,
    }
    if CompressionMode.COMPRESSED in modes:
        return "external_exported_checkpoint", {}
    return "none", {}


def _compression_stage_records(
    config: ExperimentConfig,
    *,
    runtime_compression: Mapping[str, Any] | None = None,
    definition: CompressionDefinition | None = None,
) -> list[CompressionStageRecord]:
    """Build additive v2 records without changing the legacy projection."""

    records: list[CompressionStageRecord] = []
    stages = (
        definition.stages if definition is not None
        else tuple(stage for stage in config.pipeline.stages if stage.enabled)
    )
    if not stages:
        return records
    registry = (
        definition.registry if definition is not None
        else compression_contracts(config.pipeline.extensions)
    )
    layout_by_stage = _runtime_layout_by_stage(runtime_compression)
    for stage in stages:
        bindings = []
        if stage.binding:
            bindings.append(stage.binding)
        bindings.extend(
            item.binding for item in sorted(stage.bindings, key=lambda value: value.order)
        )
        target_site = stage.operation.get("target_site")
        target_sites = [target_site] if isinstance(target_site, str) and target_site else []
        algorithm_version = None
        support_status = None
        evidence: list[dict[str, Any]] = []
        support_metadata: dict[str, Any] = {}
        if stage.uses_v2_contract:
            (
                algorithm_version,
                target_sites,
                support_status,
                evidence,
                support_metadata,
            ) = _v2_stage_support_projection(
                stage=stage,
                backbone=config.backbone.name,
                binding_names=bindings,
                registry=registry,
                projection=(definition.projections[stage.name] if definition is not None else None),
            )
        else:
            capability = get_capability_spec(
                CapabilityKind.COMPRESSOR,
                stage.plugin,
            )
            support_status = capability.status.value
            support_metadata["capability_status"] = capability.status.value
            rules = legacy_rule_specs(stage)
            if rules:
                support_status = "declared"
                support_metadata.update(
                    rules=rules, rules_changed=True, development_runtime=True,
                    canonical_evidence_applicable=False, support_claim_is_exact=False,
                )
        records.append(
            CompressionStageRecord(
                stage_name=stage.name,
                plugin=stage.plugin,
                algorithm=stage.algorithm or stage.plugin,
                algorithm_version=algorithm_version,
                variant=stage.variant,
                bindings=bindings,
                backbone=config.backbone.name,
                placement=stage.placement.value,
                execution_phase=stage.execution_phase.value,
                target_modalities=sorted(
                    item.value for item in stage.target_modalities
                ),
                conditioning_modalities=sorted(
                    item.value for item in stage.conditioning_modalities
                ),
                preserved_modalities=sorted(
                    item.value for item in stage.preserved_modalities
                ),
                coupling_policy=stage.coupling_policy.value,
                media_alignment=stage.media_alignment.value,
                target_sites=target_sites,
                support_status=support_status,
                evidence=evidence,
                layout_delta_summary=layout_by_stage.get(stage.name, {}),
                metadata={
                    "legacy_descriptor": not stage.uses_v2_contract,
                    "depends_on": list(stage.depends_on),
                    **support_metadata,
                },
            )
        )
    return records


def _runtime_layout_by_stage(
    runtime_compression: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Group compact runtime LayoutDelta evidence by logical stage name."""

    if not isinstance(runtime_compression, Mapping):
        return {}
    summary = runtime_compression.get("layout_delta_summary", {})
    raw_records = summary.get("records", ()) if isinstance(summary, Mapping) else ()
    if not isinstance(raw_records, (list, tuple)):
        return {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for raw in raw_records:
        if not isinstance(raw, Mapping):
            continue
        stage_name = raw.get("stage_name")
        if not isinstance(stage_name, str) or not stage_name:
            continue
        grouped.setdefault(stage_name, []).append(dict(raw))
    return {
        stage_name: {
            "record_count": len(records),
            "records": records,
        }
        for stage_name, records in grouped.items()
    }


def _runtime_compression_from_backends(
    *,
    train_backend: BackendRecord | None,
    eval_backend: BackendRecord | None,
) -> Mapping[str, Any] | None:
    """Read already-validated runtime evidence attached by integrations."""

    candidates: list[Any] = []
    for backend in (train_backend, eval_backend):
        if backend is None:
            continue
        direct = backend.args.get("compression_runtime")
        if isinstance(direct, Mapping):
            candidates.append(direct)
        manifests = backend.args.get("compression_manifests", ())
        if isinstance(manifests, (list, tuple)):
            for payload in manifests:
                if isinstance(payload, Mapping):
                    runtime = payload.get("runtime")
                    if isinstance(runtime, Mapping):
                        candidates.append(runtime)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Distributed workers retain their own exact activation records.  Keep
    # them as worker-qualified records instead of pretending their token maps
    # form one sequential layout.
    merged_records: list[dict[str, Any]] = []
    for worker_index, candidate in enumerate(candidates):
        summary = candidate.get("layout_delta_summary", {})
        records = summary.get("records", ()) if isinstance(summary, Mapping) else ()
        if not isinstance(records, (list, tuple)):
            continue
        for raw in records:
            if isinstance(raw, Mapping):
                merged_records.append({"worker_index": worker_index, **dict(raw)})
    return {
        "distributed_worker_count": len(candidates),
        "layout_delta_summary": {
            "record_count": len(merged_records),
            "records": merged_records,
        },
    }


def _v2_stage_support_projection(
    *,
    stage: Any,
    backbone: str,
    binding_names: list[str],
    registry: CompressionContractRegistry,
    projection: V2ContractProjection | None = None,
) -> tuple[str | None, list[str], str, list[dict[str, Any]], dict[str, Any]]:
    """Project exact registry claims into a reproducibility manifest.

    Missing support is recorded rather than silently upgraded.  This makes a
    manifest useful during research runs without borrowing verification
    evidence from another physical binding.
    """

    algorithm_version = (projection.algorithm if projection is not None else registry.algorithm(stage.algorithm)).version
    target_sites: list[str] = []
    legalizers: dict[str, str] = {}
    support_keys: list[str] = []
    statuses: list[str] = []
    evidence: list[dict[str, Any]] = []
    application_positions: dict[str, Any] = {}
    score_layers: dict[str, Any] = {}
    key_projection_layers: dict[str, Any] = {}
    variant = projection.variant if projection is not None else registry.variant(stage.variant)
    rules = rule_specs(stage, supported_rules=variant.supported_rules)
    changed = False
    resolved_bindings = projection.bindings if projection is not None else (
        registry.binding(name) for name in binding_names
    )
    for binding in resolved_bindings:
        if projection is None and binding.position_resolver is not None:
            binding = binding.position_resolver(
                stage=stage, binding=binding, algorithm=registry.algorithm(stage.algorithm),
                variant=variant,
            )
        if projection is None:
            binding = resolve_rule_binding(binding=binding, rules=rules)
        binding_changed = recipe_changed(binding)
        changed = changed or binding_changed
        legalizers[binding.name] = binding.legalizer
        for target in binding.ordered_targets():
            if target.target_site not in target_sites:
                target_sites.append(target.target_site)
            if "score_layer" in target.metadata:
                score_layers[f"{binding.name}:{target.name}"] = target.metadata["score_layer"]
            elif "score_layers" in target.metadata:
                score_layers[f"{binding.name}:{target.name}"] = target.metadata["score_layers"]
            if "key_projection_layers" in target.metadata:
                key_projection_layers[f"{binding.name}:{target.name}"] = target.metadata["key_projection_layers"]
            if "application_positions" in target.metadata:
                application_positions[f"{binding.name}:{target.name}"] = target.metadata["application_positions"]
        for modality in sorted(binding.target_modalities, key=lambda item: item.value):
            matches = registry.support.find(
                algorithm=stage.algorithm,
                variant=stage.variant,
                backbone=backbone,
                target_modality=modality,
                phase=stage.execution_phase,
            )
            if len(matches) != 1 or matches[0].binding != binding.name:
                statuses.append("undeclared")
                support_keys.append(
                    ":".join(
                        (
                            stage.algorithm,
                            stage.variant,
                            backbone,
                            modality.value,
                            stage.execution_phase.value,
                        )
                    )
                )
                continue
            unit = matches[0]
            support_keys.append(unit.identity.key)
            statuses.append("declared" if binding_changed else unit.status.value)
            for record in unit.evidence:
                evidence.append(
                    {
                        "support_unit": unit.identity.key,
                        "kind": record.kind.value,
                        "passed": record.passed,
                        "artifact": record.artifact,
                        "source_commit": record.source_commit,
                        "checksum": record.checksum,
                        "recorded_at": record.recorded_at,
                        "metadata": to_jsonable(record.metadata),
                        **(
                            {"canonical_evidence_applicable": False}
                            if binding_changed else {}
                        ),
                    }
                )
    unique_statuses = sorted(set(statuses))
    support_status = (
        unique_statuses[0]
        if len(unique_statuses) == 1
        else "mixed[" + ",".join(unique_statuses) + "]"
    )
    return (
        algorithm_version,
        target_sites,
        support_status,
        evidence,
        {
            "support_unit_keys": support_keys,
            "binding_legalizers": legalizers,
            "support_claim_is_exact": not changed,
            "algorithm_config": dict(
                projection.algorithm_config if projection is not None
                else variant.resolve_config(stage.config)
            ),
            **({"rules": rules, "rules_changed": True} if rules else {}),
            **({"application_positions": application_positions} if application_positions else {}),
            **({"score_layers": score_layers} if score_layers else {}),
            **({"key_projection_layers": key_projection_layers} if key_projection_layers else {}),
            **(
                {
                    "development_runtime": True,
                    "canonical_evidence_applicable": False,
                }
                if changed else {}
            ),
        },
    )


_DATASET_ENV_PLACEHOLDER = re.compile(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}")


def _dataset_identity(
    config: ExperimentConfig,
    eval_backend: BackendRecord | None,
) -> tuple[str | None, str | None]:
    """Return only dataset identity that the platform can state reliably."""

    if eval_backend is not None and eval_backend.name == "lmms_eval":
        # lmms-eval task YAML owns dataset_path and split. The platform neither
        # resolves nor passes config.data.dataset_id to the native evaluator, so
        # copying it into an eval manifest would claim an unverified data source.
        return None, None
    dataset_id = config.data.dataset_id
    dataset_commit = config.data.dataset_commit
    # Unused task templates may remain in effective_config. They are not
    # resolved dataset provenance and must not appear as an actual identity.
    if not dataset_id or _DATASET_ENV_PLACEHOLDER.search(dataset_id):
        return None, None
    if dataset_commit and _DATASET_ENV_PLACEHOLDER.search(dataset_commit):
        dataset_commit = None
    return dataset_id, dataset_commit


def build_run_manifest(
    config: ExperimentConfig,
    *,
    train_backend: BackendRecord | None = None,
    eval_backend: BackendRecord | None = None,
    metrics: list[MetricRecord] | None = None,
    analysis_records: list[AnalysisRecord] | None = None,
    checkpoint_path: str | None = None,
    invariants_passed: bool = False,
    invariants_run: list[str] | None = None,
    compression_runtime_metadata: Mapping[str, Any] | None = None,
    compression_definition: CompressionDefinition | None = None,
) -> CompressionRunManifest:
    """Build a command record using its pre-execution compression definition.

    Without a definition, retain the standalone API's historical recipe
    projection. Experiment commands supply one to record only executed stages.
    The effective_config snapshot always retains the full recipe.
    """

    backbone_spec = get_capability_spec(CapabilityKind.BACKBONE, config.backbone.name)
    profile = backbone_spec.profile
    if analysis_records is not None:
        model_path = (
            config.backbone.model_path
            or config.backbone.model_id
            or ""
        )
    elif train_backend is not None:
        model_path = (
            config.train.checkpoint_in
            or config.backbone.model_path
            or config.backbone.model_id
            or ""
        )
    else:
        model_path = (
            config.eval.checkpoint_path
            or config.backbone.model_path
            or config.backbone.model_id
            or ""
        )
    compressor_name, compressor_config = _compression_identity(config, compression_definition)
    dataset_id, dataset_commit = _dataset_identity(config, eval_backend)
    actual_backend = train_backend or eval_backend
    source_backend = actual_backend.name if actual_backend is not None else (
        (config.analysis.backend or "lmms_eval") if analysis_records is not None
        else config.eval.backend or config.train.backend
    )
    source_commit, source_dirty, source_tree_hash = _git_source_identity(
        config, backend=source_backend,
    )
    runtime_compression = compression_runtime_metadata or (
        _runtime_compression_from_backends(
            train_backend=train_backend,
            eval_backend=eval_backend,
        )
    )
    trace_paths: dict[str, str] = {}
    if train_backend is not None:
        compression_training = train_backend.args.get("compression_training", {})
        evidence_paths = (
            compression_training.get("evidence_paths", [])
            if isinstance(compression_training, dict)
            else []
        )
        if isinstance(evidence_paths, list):
            trace_paths.update(
                {
                    f"compression_training_evidence_{index}": path
                    for index, path in enumerate(evidence_paths)
                    if isinstance(path, str) and path
                }
            )
        pretty_paths = (
            compression_training.get("diagnostic_pretty_paths", [])
            if isinstance(compression_training, dict)
            else []
        )
        if isinstance(pretty_paths, list):
            trace_paths.update(
                {
                    f"compression_training_pretty_{index}": path
                    for index, path in enumerate(pretty_paths)
                    if isinstance(path, str) and path
                }
            )
    if eval_backend is not None:
        diagnostic_paths = eval_backend.args.get("diagnostic_trace_paths", [])
        if isinstance(diagnostic_paths, list):
            trace_paths.update(
                {
                    f"lmms_eval_diagnostic_{index}": path
                    for index, path in enumerate(diagnostic_paths)
                    if isinstance(path, str) and path
                }
            )
        pretty_paths = eval_backend.args.get("diagnostic_pretty_paths", [])
        if isinstance(pretty_paths, list):
            trace_paths.update(
                {
                    f"lmms_eval_diagnostic_pretty_{index}": path
                    for index, path in enumerate(pretty_paths)
                    if isinstance(path, str) and path
                }
            )
        compression_manifest_paths = eval_backend.args.get(
            "compression_manifest_paths", []
        )
        if isinstance(compression_manifest_paths, list):
            trace_paths.update(
                {
                    f"compression_runtime_{index}": path
                    for index, path in enumerate(compression_manifest_paths)
                    if isinstance(path, str) and path
                }
            )
        for index, payload in enumerate(eval_backend.args.get("compression_manifests", ())):
            trace = payload.get("trace", {})
            if isinstance(trace, dict) and isinstance(trace.get("path"), str):
                trace_paths[f"compression_requests_{index}"] = trace["path"]
    effective_config = _effective_config_payload(config)
    result = CompressionRunManifest(
        run_id=config.experiment.run_id or config.experiment.name,
        created_at=datetime.now(timezone.utc).isoformat(),
        config_path=config.path or "",
        config_hash=_config_hash(effective_config),
        source_commit=source_commit,
        env_lockfile_hash=None,
        backbone_name=config.backbone.name,
        backbone_version=config.backbone.model_revision,
        backbone_code_origin=str(getattr(profile, "code_origin", "unknown")),
        backbone_source_ref=str(getattr(profile, "source_ref", "unknown")),
        model_path_or_id=model_path,
        compressor_name=compressor_name,
        compressor_version=None,
        compressor_config=compressor_config,
        train_backend=train_backend,
        eval_backend=eval_backend,
        dataset_id=dataset_id,
        dataset_commit=dataset_commit,
        eval_tasks=list(
            config.analysis.tasks if analysis_records is not None else config.eval.tasks
        ),
        compression_stages=_compression_stage_records(
            config,
            runtime_compression=runtime_compression,
            definition=compression_definition,
        ),
        invariants_passed=invariants_passed,
        invariants_run=list(invariants_run or []),
        artifacts_dir=config.output.run_dir,
        checkpoint_path=checkpoint_path,
        trace_paths=trace_paths,
        source_dirty=source_dirty,
        source_tree_hash=source_tree_hash,
        source_scope=SOURCE_SCOPE if source_tree_hash else None,
        effective_config={
            name: value for name, value in effective_config.items() if name != "path"
        },
    )
    result.analysis_records = list(analysis_records or [])
    result.per_stage_stats = {
        record.stage_name: dict(record.layout_delta_summary)
        for record in result.compression_stages
        if record.layout_delta_summary
    }
    metric_values = list(metrics or [])
    target = config.eval.target
    if target is EvalTarget.COMPRESSED or config.eval.compression_mode is CompressionMode.COMPRESSED:
        result.compressed_metrics = metric_values
    else:
        result.baseline_metrics = metric_values
    return result
