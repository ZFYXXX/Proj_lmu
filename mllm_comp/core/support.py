"""Evidence-bearing support lifecycle for compression capabilities.

Support is intentionally recorded per concrete algorithm/variant/backbone/
modality/phase tuple.  This prevents a registry from accidentally turning a
set of supported backbones and a set of modalities into an unverified Cartesian
product.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Iterable

from .errors import RegistryError
from .vocabulary import ExecutionPhase, Modality, StringEnum


class SupportStatus(StringEnum):
    """Lifecycle states for one concrete support unit."""

    DECLARED = "declared"
    KERNEL_VERIFIED = "kernel_verified"
    CPU_CONTRACT_VERIFIED = "cpu_contract_verified"
    REAL_CHECKPOINT_VERIFIED = "real_checkpoint_verified"
    TRAIN_VERIFIED = "train_verified"
    ACTIVE = "active"


class EvidenceKind(StringEnum):
    """Stable evidence categories used by support promotion and manifests."""

    KERNEL = "kernel"
    CPU_CONTRACT = "cpu_contract"
    REAL_CHECKPOINT = "real_checkpoint"
    TRAIN_STEP = "train_step"
    TRAIN_BACKWARD = "train_backward"
    CHECKPOINT_RELOAD = "checkpoint_reload"
    QUALITY_BASELINE = "quality_baseline"
    EFFICIENCY_BASELINE = "efficiency_baseline"


_STATUS_EVIDENCE: dict[SupportStatus, frozenset[EvidenceKind]] = {
    SupportStatus.DECLARED: frozenset(),
    SupportStatus.KERNEL_VERIFIED: frozenset({EvidenceKind.KERNEL}),
    SupportStatus.CPU_CONTRACT_VERIFIED: frozenset(
        {EvidenceKind.KERNEL, EvidenceKind.CPU_CONTRACT}
    ),
    SupportStatus.REAL_CHECKPOINT_VERIFIED: frozenset(
        {
            EvidenceKind.KERNEL,
            EvidenceKind.CPU_CONTRACT,
            EvidenceKind.REAL_CHECKPOINT,
        }
    ),
    SupportStatus.TRAIN_VERIFIED: frozenset(
        {
            EvidenceKind.KERNEL,
            EvidenceKind.CPU_CONTRACT,
            EvidenceKind.REAL_CHECKPOINT,
            EvidenceKind.TRAIN_BACKWARD,
            EvidenceKind.CHECKPOINT_RELOAD,
        }
    ),
}

_STATUS_ORDER: tuple[SupportStatus, ...] = (
    SupportStatus.DECLARED,
    SupportStatus.KERNEL_VERIFIED,
    SupportStatus.CPU_CONTRACT_VERIFIED,
    SupportStatus.REAL_CHECKPOINT_VERIFIED,
    SupportStatus.TRAIN_VERIFIED,
    SupportStatus.ACTIVE,
)


@dataclass(frozen=True)
class EvidenceRecord:
    """One reproducible validation record.

    ``artifact`` may point at a manifest, raw result, or test report.  Evidence
    without a passing result can be retained for diagnosis but never satisfies
    an activation gate.
    """

    kind: EvidenceKind
    passed: bool
    artifact: str | None = None
    source_commit: str | None = None
    checksum: str | None = None
    recorded_at: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", EvidenceKind(self.kind))
        if not isinstance(self.passed, bool):
            raise RegistryError("Evidence passed must be a bool.")
        for name in ("artifact", "source_commit", "checksum", "recorded_at"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip()
            ):
                raise RegistryError(
                    f"Evidence {name} must be a non-empty string when provided."
                )

    @property
    def has_activation_provenance(self) -> bool:
        """Whether this record can authorize an executable support claim."""

        return all(
            isinstance(value, str) and bool(value.strip())
            for value in (self.artifact, self.source_commit, self.checksum)
        )


@dataclass(frozen=True)
class SupportUnitId:
    """Identity of one independently verified support claim."""

    algorithm: str
    variant: str
    backbone: str
    target_modality: Modality
    phase: ExecutionPhase

    def __post_init__(self) -> None:
        for name in ("algorithm", "variant", "backbone"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise RegistryError(f"Support unit {name} must be non-empty.")
        object.__setattr__(self, "target_modality", Modality(self.target_modality))
        object.__setattr__(self, "phase", ExecutionPhase(self.phase))

    @property
    def key(self) -> str:
        return ":".join(
            (
                self.algorithm,
                self.variant,
                self.backbone,
                self.target_modality.value,
                self.phase.value,
            )
        )


@dataclass(frozen=True)
class SupportUnit:
    """Lifecycle and evidence for one concrete support claim."""

    identity: SupportUnitId
    binding: str
    status: SupportStatus = SupportStatus.DECLARED
    evidence: tuple[EvidenceRecord, ...] = ()
    required_evidence: frozenset[EvidenceKind] = frozenset(
        {
            EvidenceKind.KERNEL,
            EvidenceKind.CPU_CONTRACT,
            EvidenceKind.REAL_CHECKPOINT,
        }
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.binding, str) or not self.binding.strip():
            raise RegistryError("Support unit binding must be non-empty.")
        object.__setattr__(self, "status", SupportStatus(self.status))
        object.__setattr__(
            self,
            "required_evidence",
            frozenset(EvidenceKind(item) for item in self.required_evidence),
        )
        self._validate_status(self.status)

    @property
    def passing_evidence(self) -> frozenset[EvidenceKind]:
        return frozenset(item.kind for item in self.evidence if item.passed)

    @property
    def missing_evidence(self) -> frozenset[EvidenceKind]:
        required = (
            self.required_evidence
            if self.status is SupportStatus.ACTIVE
            else _STATUS_EVIDENCE[self.status]
        )
        return required - self.passing_evidence

    @property
    def missing_activation_evidence(self) -> frozenset[EvidenceKind]:
        """Evidence still required before this exact claim may be active."""

        return self.required_evidence - self.passing_evidence

    @property
    def inference_active(self) -> bool:
        """Whether the exact configuration has active engineering support.

        ``ACTIVE`` records an inference support claim, not permission to run. Training
        evidence is independently observable through :attr:`train_verified`
        and is required only when a support unit explicitly includes the
        training evidence kinds in ``required_evidence``.
        """

        return self.status is SupportStatus.ACTIVE

    @property
    def train_verified(self) -> bool:
        """Whether backward and checkpoint-reload evidence both pass."""

        required = {
            EvidenceKind.TRAIN_BACKWARD,
            EvidenceKind.CHECKPOINT_RELOAD,
        }
        return required <= self.passing_evidence

    @property
    def train_step_verified(self) -> bool:
        """Whether a reproducible compression-aware optimizer step passes.

        This is intentionally independent from :attr:`train_verified`.  A
        one-step smoke records a successful training step without
        claiming the stronger checkpoint/reload lifecycle.
        """

        return any(
            record.kind is EvidenceKind.TRAIN_STEP
            and record.passed
            and record.has_activation_provenance
            for record in self.evidence
        )

    def with_evidence(self, *records: EvidenceRecord) -> "SupportUnit":
        """Return a new unit with records replaced by kind, then appended."""

        merged = {item.kind: item for item in self.evidence}
        for record in records:
            if not isinstance(record, EvidenceRecord):
                raise RegistryError("Support evidence must be EvidenceRecord instances.")
            merged[record.kind] = record
        ordered = tuple(merged[kind] for kind in EvidenceKind if kind in merged)
        return replace(self, evidence=ordered)

    def advance(self, status: SupportStatus | str) -> "SupportUnit":
        """Advance to a verified state without allowing evidence-free claims."""

        target = SupportStatus(status)
        if _STATUS_ORDER.index(target) < _STATUS_ORDER.index(self.status):
            raise RegistryError(
                f"Cannot move support unit {self.identity.key!r} backwards from "
                f"{self.status.value!r} to {target.value!r}."
            )
        if target is SupportStatus.ACTIVE:
            required = self.required_evidence
        else:
            required = _STATUS_EVIDENCE[target]
        missing = required - self.passing_evidence
        if missing:
            raise RegistryError(
                f"Cannot advance support unit {self.identity.key!r} to "
                f"{target.value!r}; missing passing evidence "
                f"{sorted(item.value for item in missing)}."
            )
        if target is SupportStatus.ACTIVE:
            self._validate_activation_provenance(required)
        return replace(self, status=target)

    def _validate_status(self, status: SupportStatus) -> None:
        if status is SupportStatus.ACTIVE:
            required = self.required_evidence
        else:
            required = _STATUS_EVIDENCE[status]
        missing = required - self.passing_evidence
        if missing:
            raise RegistryError(
                f"Support unit {self.identity.key!r} declares status "
                f"{status.value!r} without passing evidence "
                f"{sorted(item.value for item in missing)}."
            )
        if status is SupportStatus.ACTIVE:
            self._validate_activation_provenance(required)

    def _validate_activation_provenance(
        self,
        required: frozenset[EvidenceKind],
    ) -> None:
        records = {item.kind: item for item in self.evidence if item.passed}
        incomplete = {
            kind: tuple(
                name
                for name in ("artifact", "source_commit", "checksum")
                if not isinstance(getattr(records[kind], name), str)
                or not getattr(records[kind], name).strip()
            )
            for kind in required
            if kind in records and not records[kind].has_activation_provenance
        }
        if incomplete:
            details = ", ".join(
                f"{kind.value}={list(fields)}"
                for kind, fields in sorted(
                    incomplete.items(), key=lambda item: item[0].value
                )
            )
            raise RegistryError(
                f"Support unit {self.identity.key!r} cannot become active; "
                "passing activation evidence lacks reproducible provenance "
                f"({details})."
            )


class SupportRegistry:
    """Registry keyed by exact support-unit identity, never by set products."""

    def __init__(self, units: Iterable[SupportUnit] = ()) -> None:
        self._units: dict[str, SupportUnit] = {}
        for unit in units:
            self.register(unit)

    def register(self, unit: SupportUnit) -> None:
        key = unit.identity.key
        if key in self._units:
            raise RegistryError(f"Duplicate support unit {key!r}.")
        self._units[key] = unit

    def update(self, unit: SupportUnit) -> None:
        key = unit.identity.key
        if key not in self._units:
            raise RegistryError(f"Unknown support unit {key!r}.")
        self._units[key] = unit

    def get(self, identity: SupportUnitId) -> SupportUnit:
        try:
            return self._units[identity.key]
        except KeyError as exc:
            raise RegistryError(f"Unknown support unit {identity.key!r}.") from exc

    def find(
        self,
        *,
        algorithm: str | None = None,
        variant: str | None = None,
        backbone: str | None = None,
        target_modality: Modality | str | None = None,
        phase: ExecutionPhase | str | None = None,
        status: SupportStatus | str | None = None,
    ) -> tuple[SupportUnit, ...]:
        modality_value = (
            None if target_modality is None else Modality(target_modality)
        )
        phase_value = None if phase is None else ExecutionPhase(phase)
        status_value = None if status is None else SupportStatus(status)
        return tuple(
            unit
            for unit in self._units.values()
            if (algorithm is None or unit.identity.algorithm == algorithm)
            and (variant is None or unit.identity.variant == variant)
            and (backbone is None or unit.identity.backbone == backbone)
            and (
                modality_value is None
                or unit.identity.target_modality is modality_value
            )
            and (phase_value is None or unit.identity.phase is phase_value)
            and (status_value is None or unit.status is status_value)
        )

    def active(self) -> tuple[SupportUnit, ...]:
        return self.find(status=SupportStatus.ACTIVE)


__all__ = [
    "EvidenceKind",
    "EvidenceRecord",
    "SupportRegistry",
    "SupportStatus",
    "SupportUnit",
    "SupportUnitId",
]
