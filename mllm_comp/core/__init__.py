"""Stable public contracts shared by token and operation compression."""

from .actions import (
    LegalizationPolicy,
    OperationAction,
    OperationApplyResult,
    OperationExportRecord,
    TokenAction,
)
from .manifest import (
    AnalysisRecord,
    BackendRecord,
    CompressionRunManifest,
    CompressionStageRecord,
    MetricRecord,
)
from .layout import (
    AlignmentCouplingPolicy,
    AlignmentDelta,
    LayoutDelta,
    LayoutTokenUnit,
    ModalityLayout,
    OrderingPolicy,
    layout_delta_record,
    layout_delta_summary,
)
from .metadata import StageMetadata
from .plan import CompressionPlan
from .state import ForwardState, TokenSpan
from .support import (
    EvidenceKind,
    EvidenceRecord,
    SupportRegistry,
    SupportStatus,
    SupportUnit,
    SupportUnitId,
)
from .vocabulary import (
    AnalysisPhase,
    CompressionTokenUnit,
    CompressionMethod,
    CompressionMode,
    EvalTarget,
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
    ObjectFamily,
    Placement,
    TokenRole,
)

__all__ = [
    "AnalysisRecord",
    "AnalysisPhase",
    "AlignmentCouplingPolicy",
    "AlignmentDelta",
    "BackendRecord",
    "CompressionPlan",
    "CompressionTokenUnit",
    "CompressionMethod",
    "CompressionMode",
    "ExecutionPhase",
    "EvidenceKind",
    "EvidenceRecord",
    "EvalTarget",
    "CompressionRunManifest",
    "CompressionStageRecord",
    "ForwardState",
    "LegalizationPolicy",
    "LayoutDelta",
    "LayoutTokenUnit",
    "layout_delta_record",
    "layout_delta_summary",
    "MediaAlignment",
    "MetricRecord",
    "Modality",
    "ModalityLayout",
    "ModalityCouplingPolicy",
    "ObjectFamily",
    "OperationAction",
    "OperationApplyResult",
    "OperationExportRecord",
    "OrderingPolicy",
    "StageMetadata",
    "SupportRegistry",
    "SupportStatus",
    "SupportUnit",
    "SupportUnitId",
    "Placement",
    "TokenAction",
    "TokenRole",
    "TokenSpan",
]
