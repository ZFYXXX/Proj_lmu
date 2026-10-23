"""Invariant checks for forward state and operation plans."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .actions import OperationCompositionPolicy
from .state import ForwardState, TokenSpan


@dataclass
class InvariantViolation:
    """Actionable invariant failure."""

    name: str
    field: str
    expected: Any
    actual: Any
    sample_id: str | int | None = None
    stage_name: str | None = None


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return tuple(int(dim) for dim in shape)
    except TypeError:
        return None


def _has_nan(value: Any) -> bool:
    isnan = getattr(value, "isnan", None)
    if not callable(isnan):
        return False
    try:
        result = isnan().any()
        item = getattr(result, "item", None)
        return bool(item() if callable(item) else result)
    except Exception:
        return False


class ForwardStateInvariant:
    """Minimal Phase 1a invariant checks over available fields."""

    def check(self, state: ForwardState) -> list[InvariantViolation]:
        violations: list[InvariantViolation] = []
        batch_size: int | None = None
        seq_len: int | None = None

        for field_name in ["input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"]:
            value = getattr(state, field_name)
            shape = _shape(value)
            if shape is None:
                continue
            if len(shape) < 2:
                violations.append(
                    InvariantViolation(
                        name="rank_at_least_2",
                        field=field_name,
                        expected="rank >= 2",
                        actual=shape,
                    )
                )
                continue
            if batch_size is None:
                batch_size = shape[0]
            elif shape[0] != batch_size:
                violations.append(
                    InvariantViolation(
                        name="batch_consistency",
                        field=field_name,
                        expected=batch_size,
                        actual=shape[0],
                    )
                )
            if field_name in {"input_ids", "inputs_embeds", "attention_mask", "position_ids", "labels"}:
                current_seq_len = shape[1]
                if seq_len is None:
                    seq_len = current_seq_len
                elif current_seq_len != seq_len:
                    violations.append(
                        InvariantViolation(
                            name="sequence_length_consistency",
                            field=field_name,
                            expected=seq_len,
                            actual=current_seq_len,
                        )
                    )

        violations.extend(
            _check_token_spans(
                state=state,
                spans=state.modality_token_spans,
                field="modality_token_spans",
                violation_prefix="modality_span",
                batch_size=batch_size,
                seq_len=seq_len,
            )
        )

        for field_name in ["inputs_embeds", "hidden_states"]:
            if _has_nan(getattr(state, field_name)):
                violations.append(
                    InvariantViolation(
                        name="no_nan",
                        field=field_name,
                        expected=False,
                        actual=True,
                    )
                )

        label_shape = _shape(state.labels)
        if label_shape is not None:
            carrier_name = "inputs_embeds" if state.inputs_embeds is not None else "input_ids"
            carrier_shape = _shape(getattr(state, carrier_name))
            if carrier_shape is None:
                violations.append(
                    InvariantViolation(
                        name="label_alignment",
                        field="labels",
                        expected=f"aligned with {carrier_name}",
                        actual="no token carrier present",
                    )
                )
            elif len(label_shape) >= 2 and len(carrier_shape) >= 2 and label_shape[:2] != carrier_shape[:2]:
                violations.append(
                    InvariantViolation(
                        name="label_alignment",
                        field="labels",
                        expected=carrier_shape[:2],
                        actual=label_shape[:2],
                    )
                )

        return violations


def _check_token_spans(
    *,
    state: ForwardState,
    spans: list[TokenSpan],
    field: str,
    violation_prefix: str,
    batch_size: int | None,
    seq_len: int | None,
) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []
    previous_span: TokenSpan | None = None
    spans_by_sample: dict[int, list[TokenSpan]] = {}

    for span in spans:
        if span.sample_index < 0 or (batch_size is not None and span.sample_index >= batch_size):
            violations.append(
                InvariantViolation(
                    name=f"{violation_prefix}_sample_index_bounds",
                    field=field,
                    expected=(
                        f"0 <= sample_index < {batch_size}"
                        if batch_size is not None
                        else "sample_index >= 0"
                    ),
                    actual=span.sample_index,
                    sample_id=_sample_id_for(state, span.sample_index),
                )
            )
        if state.sample_ids is not None and span.sample_index >= len(state.sample_ids):
            violations.append(
                InvariantViolation(
                    name=f"{violation_prefix}_sample_id_known",
                    field=field,
                    expected=f"sample_index < len(sample_ids) == {len(state.sample_ids)}",
                    actual=span.sample_index,
                )
            )

        if previous_span is not None and (span.sample_index, span.start) < (
            previous_span.sample_index,
            previous_span.start,
        ):
            violations.append(
                InvariantViolation(
                    name=f"{violation_prefix}_sorted_by_sample_and_start",
                    field=field,
                    expected="nondecreasing order by (sample_index, start)",
                    actual={
                        "previous": {
                            "sample_index": previous_span.sample_index,
                            "start": previous_span.start,
                        },
                        "current": {
                            "sample_index": span.sample_index,
                            "start": span.start,
                        },
                    },
                    sample_id=_sample_id_for(state, span.sample_index),
                )
            )
        previous_span = span
        spans_by_sample.setdefault(span.sample_index, []).append(span)

        if seq_len is not None and (span.start < 0 or span.end < span.start or span.end > seq_len):
            violations.append(
                InvariantViolation(
                    name=f"{violation_prefix}_bounds",
                    field=field,
                    expected=f"0 <= start <= end <= {seq_len}",
                    actual={"start": span.start, "end": span.end},
                    sample_id=_sample_id_for(state, span.sample_index),
                )
            )

    for sample_index, sample_spans in spans_by_sample.items():
        ordered_spans = sorted(sample_spans, key=lambda item: (item.start, item.end))
        for previous, current in zip(ordered_spans, ordered_spans[1:]):
            if current.start < previous.end:
                violations.append(
                    InvariantViolation(
                        name=f"{violation_prefix}_non_overlapping",
                        field=field,
                        expected="spans within the same sample do not overlap",
                        actual={
                            "sample_index": sample_index,
                            "previous": {"start": previous.start, "end": previous.end},
                            "current": {"start": current.start, "end": current.end},
                        },
                        sample_id=_sample_id_for(state, sample_index),
                    )
                )

    return violations


def _sample_id_for(state: ForwardState, sample_index: int) -> str | int | None:
    if sample_index < 0 or state.sample_ids is None or sample_index >= len(state.sample_ids):
        return None
    return state.sample_ids[sample_index]


class OperationPlanInvariant:
    """Shallow operation-plan checks until the first operation smoke is active."""

    def check(
        self,
        *,
        actions: list[Any],
        structural_handles: Any,
        composition_policy: OperationCompositionPolicy | None = None,
    ) -> list[InvariantViolation]:
        del composition_policy
        violations: list[InvariantViolation] = []
        seen_ids: set[str] = set()
        available = set(getattr(structural_handles, "paths", lambda: [])())
        all_ids = {getattr(action, "action_id", None) for action in actions if getattr(action, "action_id", None)}
        dependency_graph = {
            getattr(action, "action_id", None): list(getattr(action, "depends_on", []) or [])
            for action in actions
            if getattr(action, "action_id", None)
        }
        for action in actions:
            action_id = getattr(action, "action_id", None)
            if not action_id:
                violations.append(
                    InvariantViolation(
                        name="operation_action_id_required",
                        field="action_id",
                        expected="non-empty string",
                        actual=action_id,
                    )
                )
            elif action_id in seen_ids:
                violations.append(
                    InvariantViolation(
                        name="operation_action_id_unique",
                        field="action_id",
                        expected="unique",
                        actual=action_id,
                    )
                )
            if action_id:
                seen_ids.add(action_id)

            module_path = getattr(action, "module_path", None)
            if available and module_path not in available:
                violations.append(
                    InvariantViolation(
                        name="operation_module_path_known",
                        field="module_path",
                        expected=sorted(available),
                        actual=module_path,
                    )
                )

            if getattr(action, "requires_checkpoint_export", False) and not getattr(
                action, "export_signature", None
            ):
                violations.append(
                    InvariantViolation(
                        name="operation_export_signature_required",
                        field="export_signature",
                        expected="non-empty when requires_checkpoint_export=True",
                        actual=getattr(action, "export_signature", None),
                    )
                )

            target_kind = getattr(action, "target_kind", None)
            if target_kind in {"head", "channel", "expert"} and not getattr(
                action, "coupled_module_paths", None
            ):
                violations.append(
                    InvariantViolation(
                        name="operation_coupled_modules_required",
                        field="coupled_module_paths",
                        expected="non-empty for width-wise pruning",
                        actual=[],
                    )
                )

            valid_routes = {"v2v", "v2t", "t2v", "t2t", "custom"}
            for route in getattr(action, "attention_routes", []) or []:
                route_name = getattr(route, "route_name", None)
                if route_name not in valid_routes:
                    violations.append(
                        InvariantViolation(
                            name="operation_attention_route_valid",
                            field="attention_routes.route_name",
                            expected=sorted(valid_routes),
                            actual=route_name,
                        )
                    )

            for dependency in getattr(action, "depends_on", []) or []:
                if dependency not in all_ids:
                    violations.append(
                        InvariantViolation(
                            name="operation_dependency_known",
                            field="depends_on",
                            expected="known action_id",
                            actual=dependency,
                        )
                    )

        visiting: set[str] = set()
        visited: set[str] = set()

        def has_cycle(action_id: str) -> bool:
            if action_id in visiting:
                return True
            if action_id in visited:
                return False
            visiting.add(action_id)
            for dependency in dependency_graph.get(action_id, []):
                if dependency in dependency_graph and has_cycle(dependency):
                    return True
            visiting.remove(action_id)
            visited.add(action_id)
            return False

        for action_id in dependency_graph:
            if has_cycle(action_id):
                violations.append(
                    InvariantViolation(
                        name="operation_dependency_acyclic",
                        field="depends_on",
                        expected="acyclic dependency graph",
                        actual=dependency_graph,
                    )
                )
                break
        return violations
