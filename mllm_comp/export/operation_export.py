"""Checkpoint export for applied structural operation compression."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from shutil import copy2
from typing import Any

from mllm_comp.core.actions import OperationApplyResult
from mllm_comp.core.errors import InvariantError


def export_operation_actions(
    *,
    model: Any,
    output_dir: str | Path,
    apply_result: OperationApplyResult,
    processor: Any | None = None,
    safe_serialization: bool = True,
    source_artifact_dir: str | Path | None = None,
    auxiliary_artifacts: Sequence[str] = (),
) -> dict[str, Any]:
    """Save a reloadable pruned checkpoint and a content-addressed manifest."""

    if apply_result.violations:
        raise InvariantError("Cannot export an invalid operation apply result.")
    if apply_result.metadata.get("dry_run"):
        raise InvariantError("Cannot export a dry-run operation result.")
    save = getattr(model, "save_pretrained", None)
    if not callable(save):
        raise InvariantError("Operation export requires model.save_pretrained().")
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    save(destination, safe_serialization=safe_serialization)
    if processor is not None:
        save_processor = getattr(processor, "save_pretrained", None)
        if callable(save_processor):
            save_processor(destination)
    copied_auxiliary_artifacts = _copy_auxiliary_artifacts(
        destination=destination,
        source_artifact_dir=source_artifact_dir,
        auxiliary_artifacts=auxiliary_artifacts,
    )
    artifact_hashes = {
        str(path.relative_to(destination)): _sha256(path)
        for path in sorted(destination.rglob("*"))
        if path.is_file() and path.name != "operation_compression_manifest.json"
    }
    manifest = {
        "schema_version": 1,
        "operation_compression": dict(apply_result.metadata),
        "actions": [asdict(item) for item in apply_result.export_records],
        "auxiliary_artifacts": copied_auxiliary_artifacts,
        "artifact_hashes": artifact_hashes,
    }
    manifest_raw = json.dumps(
        manifest,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )
    manifest_path = destination / "operation_compression_manifest.json"
    manifest_path.write_text(manifest_raw + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    manifest["manifest_payload_sha256"] = hashlib.sha256(
        manifest_raw.encode("utf-8")
    ).hexdigest()
    manifest["manifest_sha256"] = _sha256(manifest_path)
    for record in apply_result.export_records:
        record.artifact_path = str(destination)
    return manifest


def _copy_auxiliary_artifacts(
    *,
    destination: Path,
    source_artifact_dir: str | Path | None,
    auxiliary_artifacts: Sequence[str],
) -> list[str]:
    requested = [str(item) for item in auxiliary_artifacts]
    if not requested:
        return []
    if source_artifact_dir is None:
        raise InvariantError(
            "Operation export requires source_artifact_dir when auxiliary "
            "artifacts are requested."
        )
    source_root = Path(source_artifact_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise InvariantError(
            f"Operation export source artifact directory does not exist: "
            f"{source_root}."
        )
    copied: list[str] = []
    for raw_name in requested:
        relative = Path(raw_name)
        if not raw_name or relative.is_absolute() or ".." in relative.parts:
            raise InvariantError(
                f"Unsafe operation-export auxiliary artifact path {raw_name!r}."
            )
        source = (source_root / relative).resolve()
        try:
            source.relative_to(source_root)
        except ValueError as exc:
            raise InvariantError(
                f"Auxiliary artifact {raw_name!r} escapes its source directory."
            ) from exc
        if not source.is_file():
            raise InvariantError(
                f"Required operation-export auxiliary artifact is missing: {source}."
            )
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        copy2(source, target)
        copied.append(relative.as_posix())
    return copied


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["export_operation_actions"]
