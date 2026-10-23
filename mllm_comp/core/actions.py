"""Compression action contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Tensor = Any


@dataclass
class TokenAction:
    """Requested token or feature-slot transformation."""

    op: Literal["keep", "drop", "merge", "pool", "reorder", "reweight"]
    target_indices: Tensor
    merge_groups: Tensor | None = None
    weights: Tensor | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class LegalizationPolicy:
    """How the backbone legalizer repairs fields derived from token layout."""

    position_policy: Literal[
        "compact",
        "preserve_absolute",
        "m_rope_rebuild",
        "backbone_default",
    ]
    mask_policy: Literal["rebuild_from_tokens", "preserve", "backbone_default"]
    label_policy: Literal["rebuild_ignore_index", "preserve", "backbone_default"]
    coordinate_policy: Literal["preserve", "drop_removed", "backbone_default"] | None = None


@dataclass
class AttentionRouteSpec:
    """Query-to-key/value traffic route for route-level attention compression."""

    route_name: Literal["v2v", "v2t", "t2v", "t2t", "custom"]
    target_group: Literal["vision", "text", "audio", "scene_3d", "all"]
    source_group: Literal["vision", "text", "audio", "scene_3d", "all"]
    target_role: Literal["instruction", "observation", "answer", "any"] = "any"
    source_role: Literal["instruction", "observation", "answer", "any"] = "any"
    selector_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationCompositionPolicy:
    """Composition policy for operation actions in one plan."""

    apply_order: Literal["dependency_order", "config_order"] = "dependency_order"
    conflict_policy: Literal["error", "allow_disjoint"] = "error"
    validate_after_each_action: bool = True


@dataclass
class OperationAction:
    """Static computation-path compression action."""

    action_id: str
    op: Literal["skip", "mask", "prune", "replace"]
    target_kind: Literal["layer", "attention", "ffn", "head", "channel", "expert", "module", "projector"]
    target_scope: Literal["encoder", "projector", "llm"]
    module_path: str
    apply_mode: Literal["runtime_gate", "static_mask", "structural_rewrite"]
    shape_effect: Literal[
        "preserve_shape",
        "preserve_io_shape",
        "remove_module",
        "change_num_heads",
        "change_hidden_channels",
        "change_num_experts",
    ]
    original_shape: tuple | None
    pruned_shape: tuple | None
    strategy_name: str | None = None
    target_indices: Tensor | list[int] | None = None
    attention_routes: list[AttentionRouteSpec] = field(default_factory=list)
    coupled_module_paths: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)
    composition_group: str | None = None
    requires_checkpoint_export: bool = True
    export_signature: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class OperationExportRecord:
    """Manifest-facing record for exported operation-compression edits."""

    action_id: str
    op: str
    target_kind: str
    target_scope: str
    module_path: str
    attention_routes: list[dict[str, Any]]
    target_indices: list[int] | None
    original_shape: tuple | None
    pruned_shape: tuple | None
    export_signature: str
    artifact_path: str | None = None


@dataclass
class OperationApplyResult:
    """Result of validating or applying operation-compression actions."""

    applied_actions: list[OperationAction] = field(default_factory=list)
    installed_handles: list[str] = field(default_factory=list)
    violations: list[Any] = field(default_factory=list)
    reversible: bool = True
    export_records: list[OperationExportRecord] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
