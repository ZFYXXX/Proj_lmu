"""Evidence integrity and discovery independent of historical numeric snapshots."""

import json
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest

from mllm_comp.compressors.video import evidence as evidence_module
from mllm_comp.compressors.video.config import canonical_video_config_record
from mllm_comp.compressors.video.registry import video_bindings, video_support_units
from mllm_comp.core.errors import RegistryError
from mllm_comp.core.support import EvidenceKind, SupportStatus


def test_exact_evidence_points_to_passed_artifact_with_matching_identity_and_config() -> None:
    for unit in video_support_units():
        artifact = Path(unit.evidence[0].artifact)
        detail = json.loads(artifact.read_text(encoding="utf-8"))
        assert detail["status"] == "passed"
        assert detail["support_unit"] == {
            "algorithm": unit.identity.algorithm,
            "variant": unit.identity.variant,
            "backbone": unit.identity.backbone,
            "target_modality": "video",
            "phase": unit.identity.phase.value,
            "binding": unit.binding,
        }
        assert detail["canonical_config"] == canonical_video_config_record(unit.identity.algorithm)
        assert unit.metadata["evidence_artifact"] == str(artifact.as_posix())
        assert all(record.has_activation_provenance for record in unit.evidence)
        assert all(not record.metadata for record in unit.evidence)


def test_missing_cell_cannot_borrow_another_phase_or_backbones_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = dict(evidence_module.VIDEO_EVIDENCE)
    entries.pop(("longvu", "qwen3_vl", "forward"))
    monkeypatch.setattr(evidence_module, "VIDEO_EVIDENCE", entries)
    units = video_support_units()
    missing = next(unit for unit in units if unit.identity.key == "longvu:longvu.video:qwen3_vl:video:forward")
    assert missing.status is SupportStatus.DECLARED
    assert not missing.inference_active
    assert not missing.train_step_verified and not missing.train_verified
    assert len([unit for unit in units if unit.status is SupportStatus.ACTIVE]) == 53
    binding = next(binding for binding in video_bindings() if binding.name == missing.binding)
    assert binding.metadata["verified_phases"] == ("generate",)
    assert "forward_evidence_artifact" not in binding.metadata


def test_incomplete_training_evidence_cannot_claim_active_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = dict(evidence_module.VIDEO_EVIDENCE)
    key = ("longvu", "qwen3_vl", "forward")
    checksum, records = entries[key]
    entries[key] = (checksum, tuple(record for record in records if record[0] is not EvidenceKind.TRAIN_BACKWARD))
    monkeypatch.setattr(evidence_module, "VIDEO_EVIDENCE", entries)
    with pytest.raises(RegistryError, match="train_backward"):
        video_support_units()


def test_package_only_discovery_needs_no_checkout_artifacts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    archive = tmp_path / "platform.zip"
    with zipfile.ZipFile(archive, "w") as package:
        for path in (root / "mllm_comp").rglob("*.py"):
            package.write(path, path.relative_to(root).as_posix())
    code = "\n".join([
        "import sys",
        f"sys.path.insert(0, {str(archive)!r})",
        "from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts",
        "registry = builtin_multimodal_compression_contracts()",
        "units = registry.support.find(target_modality='video', status='active')",
        "assert len(units) == 54",
        "assert sum(unit.train_verified for unit in units) == 27",
        "assert all(not unit.missing_activation_evidence for unit in units)",
        "print('54 active video cells discovered from packaged Python only')",
    ])
    result = subprocess.run([sys.executable, "-I", "-c", code], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "54 active video cells" in result.stdout
