"""Generic structured LLM-layer pruning and SPADE Operation-Only profile.

This is intentionally the operation-only subset used by the SPADE paper: the
layer selection is supplied explicitly or loaded as a precomputed result.  It
does not implement WLI analysis, teachers, distillation, head pruning or FFN
channel pruning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import OperationAction
from mllm_comp.compressors.audio_rule_utils import audio_rule_metadata
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    ObjectFamily,
    Placement,
)


@dataclass(frozen=True)
class StructuredLLMLayerPruningSpec:
    target_layers: tuple[int, ...] = ()
    selection_source: str = "explicit"
    target_site: str = "llm.layers"
    target_scope: str = "llm"
    target_kind: str = "layer"
    apply_mode: str = "structural_rewrite"
    shape_effect: str = "remove_module"

    @classmethod
    def from_config(
        cls, config: Mapping[str, Any]
    ) -> "StructuredLLMLayerPruningSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        activation_scope = _mapping(
            config.get("activation_scope"), "activation_scope"
        )
        legacy = _mapping(config.get("config"), "config")
        selection_source = str(
            _first_present(
                criterion.get("selection_source"),
                legacy.get("selection_source"),
                cls.selection_source,
            )
        )
        layers = _first_present(
            activation_scope.get("layers"),
            criterion.get("precomputed_layers"),
            legacy.get("target_layers"),
            cls.target_layers,
        )
        if not isinstance(layers, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in layers
        ):
            raise ValueError("Structured layer pruning requires integer target layers.")
        spec = cls(
            target_layers=tuple(int(item) for item in layers),
            selection_source=selection_source,
            target_site=str(
                _first_present(
                    operation.get("target_site"),
                    legacy.get("target_site"),
                    cls.target_site,
                )
            ),
            target_scope=str(
                _first_present(operation.get("target_scope"), cls.target_scope)
            ),
            target_kind=str(
                _first_present(operation.get("target_kind"), cls.target_kind)
            ),
            apply_mode=str(
                _first_present(operation.get("apply_mode"), cls.apply_mode)
            ),
            shape_effect=str(
                _first_present(operation.get("shape_effect"), cls.shape_effect)
            ),
        )
        spec.validate()
        operation_type = operation.get("type")
        if operation_type not in {None, "prune"}:
            raise ValueError("Structured layer pruning requires operation.type=prune.")
        return spec

    def validate(self) -> None:
        if self.selection_source not in {"explicit", "precomputed"}:
            raise ValueError(
                "selection_source must be 'explicit' or 'precomputed'; online WLI "
                "selection is outside SPADE Operation-Only."
            )
        if tuple(sorted(set(self.target_layers))) != self.target_layers:
            raise ValueError("target layers must be unique and strictly increasing.")
        if self.target_layers and self.target_layers[0] < 0:
            raise ValueError("target layers must be non-negative.")
        expected = {
            "target_site": (self.target_site, "llm.layers"),
            "target_scope": (self.target_scope, "llm"),
            "target_kind": (self.target_kind, "layer"),
            "apply_mode": (self.apply_mode, "structural_rewrite"),
            "shape_effect": (self.shape_effect, "remove_module"),
        }
        invalid = [name for name, (value, wanted) in expected.items() if value != wanted]
        if invalid:
            raise ValueError(
                "Structured layer pruning received unsupported operation fields: "
                f"{invalid}."
            )


class StructuredLLMLayerPruningStrategy:
    def __init__(self, spec: StructuredLLMLayerPruningSpec) -> None:
        self.spec = spec

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "target_site": self.spec.target_site,
            "selection_source": self.spec.selection_source,
            "target_layers": list(self.spec.target_layers),
            "apply_mode": self.spec.apply_mode,
            "shape_effect": self.spec.shape_effect,
            "requires_checkpoint_export": True,
            "scope_note": "SPADE Operation-Only; no WLI analysis or distillation",
            **audio_rule_metadata(self),
            **({"target_layers": list(self._selected_layers), "selection_source": "custom_decision"}
               if hasattr(self, "_selected_layers") else {}),
        }

    def plan_operations(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
    ) -> list[OperationAction]:
        llm = structural_handles.get("llm")
        layers = getattr(llm, "layers", None)
        if layers is None:
            raise InvariantError(
                "Structured layer pruning requires the semantic handle 'llm'."
            )
        num_layers = len(layers)
        if any(index >= num_layers for index in self.spec.target_layers):
            raise InvariantError(
                f"Target layers {self.spec.target_layers} exceed num_layers={num_layers}."
            )
        if len(self.spec.target_layers) >= num_layers and num_layers > 0:
            raise InvariantError("Structured layer pruning cannot remove every LLM layer.")
        selected = self.spec.target_layers
        decision = getattr(self, "rules", {}).get("decision")
        if decision is not None:
            selected = decision({"model": model, "layers": layers, "num_layers": num_layers,
                "target_layers": tuple(selected), "remove_count": len(selected)})
            if (not isinstance(selected, (tuple, list)) or len(selected) != len(self.spec.target_layers)
                    or any(isinstance(i, bool) or not isinstance(i, int) or not 0 <= i < num_layers for i in selected)
                    or len(set(selected)) != len(selected)):
                raise InvariantError("SPADE decision must return unique in-range layers with the configured removal count.")
            selected = tuple(sorted(selected))
            self._selected_layers = selected
        final_layers = num_layers - len(selected)
        return [
            OperationAction(
                action_id=f"prune_llm_layer_{index}",
                op="prune",
                target_kind="layer",
                target_scope="llm",
                module_path=f"llm.layer[{index}]",
                apply_mode="structural_rewrite",
                shape_effect="remove_module",
                original_shape=(num_layers,),
                pruned_shape=(final_layers,),
                strategy_name="structured_llm_layer_pruning",
                target_indices=[index],
                requires_checkpoint_export=True,
                metadata={
                    "selection_source": self.spec.selection_source if decision is None else "custom_decision",
                    "semantic_collection": self.spec.target_site,
                    "source_layer_index": index,
                },
            )
            for index in selected
        ]


class StructuredLLMLayerPruning:
    name = "structured_llm_layer_pruning"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        StructuredLLMLayerPruningSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del state
        spec = StructuredLLMLayerPruningSpec.from_config(config)
        strategy = context.extra.get("audio_operation_strategy") or StructuredLLMLayerPruningStrategy(spec)
        actions = strategy.plan_operations(
            model=context.model,
            structural_handles=context.structural_handles,
        )
        stage_name = str(config.get("_stage_name") or self.name)
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.LLM_LEVEL,
            execution_phase=ExecutionPhase(
                config.get("_execution_phase", ExecutionPhase.PREPARE)
            ),
            modalities=set(),
            token_roles=set(),
            method=CompressionMethod.OPERATION_COMPRESSION,
            object_family=ObjectFamily.MODULE_UNIT,
            object_subtype="transformer_layer",
            operation_actions=actions,
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                stats={
                    **strategy.runtime_metadata(),
                    "planned_operation_count": len(actions),
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("StructuredLLMLayerPruning has no trainable state")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_present(*values: Any) -> Any:
    return next((item for item in values if item is not None), None)


__all__ = [
    "StructuredLLMLayerPruning",
    "StructuredLLMLayerPruningSpec",
    "StructuredLLMLayerPruningStrategy",
]
