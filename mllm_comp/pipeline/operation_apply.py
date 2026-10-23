"""Operation-action appliers.

The Qwen applier understands semantic structural handles, never paper or
plugin names.  The first active operation family removes complete Thinker LLM
layers and rewrites the provider ModuleList into a reloadable checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

from mllm_comp.core.actions import (
    OperationAction,
    OperationApplyResult,
    OperationExportRecord,
)
from mllm_comp.core.errors import InvariantError, NotActivatedError
from mllm_comp.core.invariants import InvariantViolation


_LLM_LAYER_PATH = re.compile(r"^llm\.layer\[(\d+)\]$")


class BaseOperationApplier:
    """Base class for backbone-specific structural application."""

    def validate(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
    ) -> list[InvariantViolation]:
        del model, actions, structural_handles
        raise NotActivatedError("This backbone has no operation applier.")

    def apply(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
        stage_name: str,
        dry_run: bool = False,
    ) -> OperationApplyResult:
        del model, actions, structural_handles, stage_name, dry_run
        raise NotActivatedError("This backbone has no operation applier.")


class Qwen25OmniOperationApplier(BaseOperationApplier):
    """Apply semantic full-layer rewrites to the Qwen Thinker LLM."""

    def validate(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
    ) -> list[InvariantViolation]:
        del model
        violations: list[InvariantViolation] = []
        llm = structural_handles.get("llm")
        layers = getattr(llm, "layers", None)
        if layers is None:
            return [
                _violation(
                    "operation_llm_handle_required",
                    "structural_handles.llm",
                    "module with layers",
                    None,
                )
            ]
        num_layers = len(layers)
        indices: list[int] = []
        action_ids: set[str] = set()
        for action in actions:
            if action.action_id in action_ids:
                violations.append(
                    _violation(
                        "operation_action_id_unique",
                        "action_id",
                        "unique",
                        action.action_id,
                    )
                )
            action_ids.add(action.action_id)
            expected_fields = {
                "op": (action.op, "prune"),
                "target_kind": (action.target_kind, "layer"),
                "target_scope": (action.target_scope, "llm"),
                "apply_mode": (action.apply_mode, "structural_rewrite"),
                "shape_effect": (action.shape_effect, "remove_module"),
                "requires_checkpoint_export": (
                    action.requires_checkpoint_export,
                    True,
                ),
            }
            for field_name, (actual, expected) in expected_fields.items():
                if actual != expected:
                    violations.append(
                        _violation(
                            "operation_layer_rewrite_contract",
                            field_name,
                            expected,
                            actual,
                        )
                    )
            match = _LLM_LAYER_PATH.fullmatch(action.module_path)
            if match is None:
                violations.append(
                    _violation(
                        "operation_semantic_layer_path",
                        "module_path",
                        "llm.layer[index]",
                        action.module_path,
                    )
                )
                continue
            index = int(match.group(1))
            target_indices = list(action.target_indices or [])
            if target_indices != [index]:
                violations.append(
                    _violation(
                        "operation_target_index_matches_path",
                        "target_indices",
                        [index],
                        target_indices,
                    )
                )
            if not 0 <= index < num_layers:
                violations.append(
                    _violation(
                        "operation_layer_index_bounds",
                        "module_path",
                        f"0 <= index < {num_layers}",
                        index,
                    )
                )
            indices.append(index)
        duplicates = sorted({item for item in indices if indices.count(item) > 1})
        if duplicates:
            violations.append(
                _violation(
                    "operation_layer_indices_unique",
                    "target_indices",
                    "unique",
                    duplicates,
                )
            )
        valid_indices = {item for item in indices if 0 <= item < num_layers}
        if num_layers and len(valid_indices) >= num_layers:
            violations.append(
                _violation(
                    "operation_preserve_one_layer",
                    "target_indices",
                    f"fewer than {num_layers} layers",
                    sorted(valid_indices),
                )
            )
        return violations

    def apply(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
        stage_name: str,
        dry_run: bool = False,
    ) -> OperationApplyResult:
        violations = self.validate(
            model=model,
            structural_handles=structural_handles,
            actions=actions,
        )
        if violations:
            return OperationApplyResult(violations=violations, reversible=True)
        llm = structural_handles["llm"]
        layers = llm.layers
        original_count = len(layers)
        prune_indices = sorted(
            int(_LLM_LAYER_PATH.fullmatch(action.module_path).group(1))
            for action in actions
        )
        prune_set = set(prune_indices)
        old_to_new: dict[int, int | None] = {}
        next_index = 0
        for old_index in range(original_count):
            if old_index in prune_set:
                old_to_new[old_index] = None
            else:
                old_to_new[old_index] = next_index
                next_index += 1
        parameter_count_before = _parameter_count(model)
        metadata = {
            "stage_name": stage_name,
            "target_site": "llm.layers",
            "source_num_hidden_layers": original_count,
            "pruned_source_layers": prune_indices,
            "old_to_new_layer_mapping": {
                str(key): value for key, value in old_to_new.items()
            },
            "result_num_hidden_layers": original_count - len(prune_indices),
            "apply_mode": "structural_rewrite",
            "requires_checkpoint_export": bool(actions),
            "identity": not actions,
            "dry_run": bool(dry_run),
        }
        export_records = [
            _export_record(action, old_to_new=old_to_new)
            for action in actions
        ]
        if dry_run or not actions:
            metadata["parameter_count_before"] = parameter_count_before
            metadata["parameter_count_after"] = parameter_count_before
            metadata["parameter_delta"] = 0
            if not dry_run:
                _record_materialized_operation(model, metadata)
            return OperationApplyResult(
                applied_actions=[] if dry_run else list(actions),
                installed_handles=[],
                reversible=True,
                export_records=export_records,
                metadata=metadata,
            )

        applied_stages = set(getattr(model, "_mllm_comp_operation_stages", ()))
        applied_stages.update(
            item["stage_name"]
            for item in getattr(getattr(model, "config", None), "mllm_comp_operations", ())
        )
        if stage_name in applied_stages:
            raise InvariantError(
                f"Operation stage {stage_name!r} has already been applied."
            )
        torch = _torch()
        retained = [
            layer for index, layer in enumerate(layers) if index not in prune_set
        ]
        retained_source_indices = [
            index for index in range(original_count) if index not in prune_set
        ]
        llm.layers = torch.nn.ModuleList(retained)
        for new_index, layer in enumerate(llm.layers):
            attention = getattr(layer, "self_attn", None)
            if attention is not None and hasattr(attention, "layer_idx"):
                attention.layer_idx = new_index
        _update_num_hidden_layers(
            model,
            llm,
            len(retained),
            source_num_layers=original_count,
            retained_source_indices=retained_source_indices,
        )
        parameter_count_after = _parameter_count(model)
        metadata.update(
            {
                "parameter_count_before": parameter_count_before,
                "parameter_count_after": parameter_count_after,
                "parameter_delta": parameter_count_after - parameter_count_before,
                "parameter_reduction": parameter_count_before - parameter_count_after,
                "state_dict_layer_indices_contiguous": _state_dict_layers_contiguous(
                    llm, len(retained)
                ),
                "dynamic_cache_compatible": True,
                "gradient_checkpointing_compatible": True,
                "config_layer_types_reindexed": _config_layer_types_match(
                    model, llm, len(retained)
                ),
            }
        )
        applied_stages.add(stage_name)
        model._mllm_comp_operation_stages = tuple(sorted(applied_stages))
        _record_materialized_operation(model, metadata)
        return OperationApplyResult(
            applied_actions=list(actions),
            installed_handles=["llm.layers"],
            violations=[],
            reversible=False,
            export_records=export_records,
            metadata=metadata,
        )


def _record_materialized_operation(model: Any, metadata: Mapping[str, Any]) -> None:
    """Keep applied structure in both composite and Thinker-only checkpoints."""
    config = getattr(model, "config", None)
    history = list(getattr(config, "mllm_comp_operations", ()))
    history.append(dict(metadata))
    model._mllm_comp_operation_manifests = history
    for owner in (config, getattr(getattr(model, "thinker", None), "config", None)):
        if owner is not None:
            owner.mllm_comp_operations = history


def _configuration_candidates(model: Any, llm: Any) -> list[Any]:
    candidates = [
        getattr(llm, "config", None),
        getattr(getattr(model, "thinker", None), "config", None),
        getattr(
            getattr(getattr(model, "thinker", None), "config", None),
            "text_config",
            None,
        ),
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "thinker_config", None),
        getattr(
            getattr(getattr(model, "config", None), "thinker_config", None),
            "text_config",
            None,
        ),
    ]
    unique: list[Any] = []
    observed: set[int] = set()
    for config in candidates:
        for candidate in (config, getattr(config, "text_config", None)):
            if candidate is None or id(candidate) in observed:
                continue
            observed.add(id(candidate))
            unique.append(candidate)
    return unique


def _update_num_hidden_layers(
    model: Any,
    llm: Any,
    value: int,
    *,
    source_num_layers: int,
    retained_source_indices: list[int],
) -> None:
    candidates = _configuration_candidates(model, llm)
    for config in candidates:
        if hasattr(config, "num_hidden_layers"):
            config.num_hidden_layers = value
        layer_types = getattr(config, "layer_types", None)
        if isinstance(layer_types, (list, tuple)) and len(layer_types) == source_num_layers:
            filtered = [layer_types[index] for index in retained_source_indices]
            config.layer_types = (
                tuple(filtered) if isinstance(layer_types, tuple) else filtered
            )


def _config_layer_types_match(model: Any, llm: Any, value: int) -> bool:
    observed = [
        getattr(config, "layer_types")
        for config in _configuration_candidates(model, llm)
        if getattr(config, "layer_types", None) is not None
    ]
    return all(len(item) == value for item in observed)


def _state_dict_layers_contiguous(llm: Any, num_layers: int) -> bool:
    keys = list(llm.state_dict())
    observed = {
        int(match.group(1))
        for key in keys
        if (match := re.match(r"^layers\.(\d+)\.", key)) is not None
    }
    return observed == set(range(num_layers))


def _parameter_count(model: Any) -> int:
    parameters = getattr(model, "parameters", None)
    if not callable(parameters):
        return 0
    return sum(int(item.numel()) for item in parameters())


def _export_record(
    action: OperationAction,
    *,
    old_to_new: Mapping[int, int | None],
) -> OperationExportRecord:
    payload = {
        "action_id": action.action_id,
        "module_path": action.module_path,
        "target_indices": list(action.target_indices or []),
        "old_to_new": {str(key): value for key, value in old_to_new.items()},
    }
    export_signature = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    action.export_signature = export_signature
    return OperationExportRecord(
        action_id=action.action_id,
        op=action.op,
        target_kind=action.target_kind,
        target_scope=action.target_scope,
        module_path=action.module_path,
        attention_routes=[],
        target_indices=list(action.target_indices or []),
        original_shape=action.original_shape,
        pruned_shape=action.pruned_shape,
        export_signature=export_signature,
    )


def _violation(
    name: str,
    field: str,
    expected: Any,
    actual: Any,
) -> InvariantViolation:
    return InvariantViolation(
        name=name,
        field=field,
        expected=expected,
        actual=actual,
    )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InvariantError("Structural operation compression requires PyTorch.") from exc
    return torch


__all__ = ["BaseOperationApplier", "Qwen25OmniOperationApplier"]
