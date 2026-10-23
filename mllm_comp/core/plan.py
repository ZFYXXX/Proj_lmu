"""Compression plan contract."""

from __future__ import annotations

from dataclasses import dataclass, field

from .actions import (
    LegalizationPolicy,
    OperationAction,
    OperationCompositionPolicy,
    TokenAction,
)
from .metadata import StageMetadata
from .vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)


@dataclass
class CompressionPlan:
    """Bridge between plugin intent and pipeline execution."""

    stage_name: str
    placement: Placement
    execution_phase: ExecutionPhase
    modalities: set[Modality]
    token_roles: set[TokenRole]
    method: CompressionMethod
    object_family: ObjectFamily
    object_subtype: str | None

    token_actions: list[TokenAction] = field(default_factory=list)
    legalization_policy: LegalizationPolicy | None = None
    operation_actions: list[OperationAction] = field(default_factory=list)
    operation_composition_policy: OperationCompositionPolicy = field(
        default_factory=OperationCompositionPolicy
    )
    this_stage_metadata: StageMetadata | None = None

    def __post_init__(self) -> None:
        self.placement = Placement(self.placement)
        self.execution_phase = ExecutionPhase(self.execution_phase)
        self.modalities = {Modality(value) for value in self.modalities}
        self.token_roles = {TokenRole(value) for value in self.token_roles}
        self.method = CompressionMethod(self.method)
        self.object_family = ObjectFamily(self.object_family)
