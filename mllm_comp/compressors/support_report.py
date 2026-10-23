"""Human- and machine-readable projections of exact compression support."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from mllm_comp.core.support import EvidenceKind, SupportUnit
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


@dataclass(frozen=True)
class SupportMatrixRow:
    """One exact matrix cell; never a backbone/modality set product."""

    algorithm: str
    variant: str
    backbone: str
    target_modality: str
    phase: str
    binding: str
    status: str
    passing_evidence: tuple[str, ...]
    missing_evidence: tuple[str, ...]
    target_sites: tuple[str, ...]
    legalizer: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def support_matrix_rows(
    registry: Any,
    *,
    target_modality: Modality | str | None = None,
    phase: ExecutionPhase | str | None = None,
) -> tuple[SupportMatrixRow, ...]:
    """Return deterministic rows from registered exact support units."""

    units = registry.support.find(
        target_modality=target_modality,
        phase=phase,
    )
    rows = [_support_matrix_row(unit, registry.binding(unit.binding)) for unit in units]
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                row.algorithm,
                row.backbone,
                row.target_modality,
                row.phase,
            ),
        )
    )


def _support_matrix_row(unit: SupportUnit, binding: Any) -> SupportMatrixRow:
    """Project one exact support unit for both matrix reports and discovery."""
    return SupportMatrixRow(
        algorithm=unit.identity.algorithm,
        variant=unit.identity.variant,
        backbone=unit.identity.backbone,
        target_modality=unit.identity.target_modality.value,
        phase=unit.identity.phase.value,
        binding=binding.name,
        status=unit.status.value,
        passing_evidence=tuple(
            kind.value for kind in EvidenceKind if kind in unit.passing_evidence
        ),
        missing_evidence=tuple(
            kind.value for kind in EvidenceKind if kind in unit.missing_activation_evidence
        ),
        target_sites=tuple(target.target_site for target in binding.ordered_targets()),
        legalizer=binding.legalizer,
    )


def render_support_matrix_markdown(rows: tuple[SupportMatrixRow, ...]) -> str:
    """Render a compact table suitable for a validation report."""

    header = (
        "| Algorithm | Backbone | Modality | Phase | Status | Evidence | Binding |\n"
        "|---|---|---|---|---|---|---|"
    )
    body = []
    for row in rows:
        evidence = ", ".join(row.passing_evidence) or "none"
        body.append(
            "| "
            + " | ".join(
                (
                    row.algorithm,
                    row.backbone,
                    row.target_modality,
                    row.phase,
                    row.status,
                    evidence,
                    row.binding,
                )
            )
            + " |"
        )
    return header + ("\n" + "\n".join(body) if body else "") + "\n"


def builtin_imagellm_rows(
    *,
    phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
) -> tuple[SupportMatrixRow, ...]:
    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )

    return support_matrix_rows(
        builtin_multimodal_compression_contracts(),
        target_modality=Modality.IMAGE,
        phase=phase,
    )


__all__ = [
    "SupportMatrixRow",
    "builtin_imagellm_rows",
    "render_support_matrix_markdown",
    "support_matrix_rows",
]
