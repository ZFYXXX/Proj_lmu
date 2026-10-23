"""Install Qwen3-VL operation-compression replacements."""

from __future__ import annotations

from copy import deepcopy
from functools import partial
import re
from collections.abc import Callable, Mapping, MutableMapping
from typing import Any

from mllm_comp.compressors.llm_level.qwen3_vl_attention_replacement import (
    Qwen3VLAttentionReplacement,
)
from mllm_comp.compressors.llm_level.qwen3_vl_ffn_replacement import Qwen3VLFFNReplacement
from mllm_comp.core.actions import OperationAction, OperationApplyResult
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.invariants import InvariantViolation

_MODULE_PATH = re.compile(r"^llm\.layer\[(\d+)\]\.(attn|ffn)$")
_METHODS_BY_KIND = {
    "attention": Qwen3VLAttentionReplacement,
    "ffn": Qwen3VLFFNReplacement,
}


class Qwen3VLOperationApplier:
    """Apply shape-preserving attention and FFN replacements to Qwen3-VL layers."""

    def validate(
        self,
        *,
        model: Any,
        structural_handles: Mapping[str, Any],
        actions: list[OperationAction],
    ) -> list[InvariantViolation]:
        del model
        violations: list[InvariantViolation] = []
        for action in actions:
            if action.op != "replace":
                violations.append(
                    self._violation(action, "operation_kind", "replace", action.op)
                )
            if (
                action.target_kind not in _METHODS_BY_KIND
                or action.target_scope != "llm"
            ):
                violations.append(
                    self._violation(
                        action,
                        "operation_target",
                        "target_kind=attention or ffn,target_scope=llm",
                        f"target_kind={action.target_kind},target_scope={action.target_scope}",
                    )
                )
            if action.apply_mode != "structural_rewrite":
                violations.append(
                    self._violation(
                        action,
                        "operation_apply_mode",
                        "structural_rewrite",
                        action.apply_mode,
                    )
                )
            if action.shape_effect not in {"preserve_shape", "preserve_io_shape"}:
                violations.append(
                    self._violation(
                        action,
                        "operation_shape_effect",
                        "preserve_shape or preserve_io_shape",
                        action.shape_effect,
                    )
                )
            match = _MODULE_PATH.fullmatch(action.module_path)
            if match is None:
                violations.append(
                    self._violation(
                        action,
                        "operation_module_path",
                        "llm.layer[<index>].attn or llm.layer[<index>].ffn",
                        action.module_path,
                    )
                )
                continue
            path_kind = "attention" if match.group(2) == "attn" else "ffn"
            if action.target_kind != path_kind:
                violations.append(
                    self._violation(
                        action,
                        "operation_target_path_kind",
                        path_kind,
                        action.target_kind,
                    )
                )
            layer_path = f"llm.layer[{match.group(1)}]"
            if (
                action.module_path not in structural_handles
                or layer_path not in structural_handles
            ):
                violations.append(
                    self._violation(
                        action,
                        "operation_module_path_known",
                        "attention and parent layer structural handles",
                        action.module_path,
                    )
                )
            strategy = action.strategy_name or action.metadata.get("strategy")
            method = _METHODS_BY_KIND.get(action.target_kind)
            valid_strategies = method.strategies if method is not None else frozenset()
            if strategy not in valid_strategies:
                violations.append(
                    self._violation(
                        action,
                        "operation_replacement_strategy",
                        sorted(valid_strategies),
                        strategy,
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
        factory: Callable[..., Any] | None = None,
        factory_spec: Mapping[str, Any] | None = None,
    ) -> OperationApplyResult:
        violations = self.validate(
            model=model,
            structural_handles=structural_handles,
            actions=actions,
        )
        if violations:
            details = "; ".join(
                f"{item.name}: expected {item.expected!r}, got {item.actual!r}"
                for item in violations
            )
            raise InvariantError(
                f"Qwen3-VL operation stage {stage_name!r} is invalid: {details}"
            )

        result = OperationApplyResult(reversible=True)
        for action in actions:
            if dry_run:
                result.applied_actions.append(action)
                continue
            path_match = _MODULE_PATH.fullmatch(action.module_path)
            layer_index = int(path_match.group(1))  # type: ignore[union-attr]
            layer_path = f"llm.layer[{layer_index}]"
            parent_layer = structural_handles[layer_path]
            attribute_name = "self_attn" if action.target_kind == "attention" else "mlp"
            original_module = getattr(parent_layer, attribute_name, None)
            if original_module is None:
                raise BackendIntegrationError(
                    f"Qwen3-VL layer {layer_index} has no {attribute_name} module to replace."
                )
            requested_strategy = str(
                action.strategy_name or action.metadata.get("strategy")
            )
            installed_strategy = getattr(
                original_module,
                "mllm_comp_replacement_strategy",
                getattr(original_module, "strategy_name", None),
            )
            if installed_strategy is not None:
                installed_factory = getattr(original_module, "mllm_comp_factory_spec", None)
                installed_config = getattr(original_module, "mllm_comp_replacement_config", None)
                if (installed_strategy != requested_strategy or installed_factory != factory_spec
                        or installed_config != action.metadata):
                    raise BackendIntegrationError(
                        f"{action.module_path} already has replacement {installed_strategy!r}; "
                        f"cannot install {requested_strategy!r} with a different strategy, factory or configuration."
                    )
                replacement = original_module
            else:
                method = _METHODS_BY_KIND[action.target_kind]
                build_default = partial(
                    method.build_replacement,
                    original_module,
                    strategy=requested_strategy,
                    config=action.metadata,
                )
                if factory is None:
                    replacement = build_default()
                else:
                    from torch.nn import Module

                    replacement = factory({
                        "original_module": original_module,
                        "strategy": requested_strategy,
                        "target_kind": action.target_kind,
                        "module_path": action.module_path,
                        "config": dict(action.metadata),
                        "default_factory": build_default,
                    })
                    if not isinstance(replacement, Module):
                        raise BackendIntegrationError("The replacement factory must return a torch.nn.Module.")
                    if replacement is original_module:
                        raise BackendIntegrationError("The replacement factory must return a new module, not original_module.")
                    replacement.train(original_module.training)
                    replacement.mllm_comp_factory_spec = deepcopy(factory_spec)
                replacement.mllm_comp_replacement_strategy = requested_strategy
                # Planners materialize defaults in the Action. Preserve that
                # effective configuration, independently of its stage name.
                replacement.mllm_comp_replacement_config = deepcopy(action.metadata)
                replacement.mllm_comp_action_id = action.action_id
                setattr(parent_layer, attribute_name, replacement)
            if isinstance(structural_handles, MutableMapping):
                structural_handles[action.module_path] = replacement
            result.applied_actions.append(action)
            result.installed_handles.append(action.module_path)
        result.metadata.update(
            {
                "stage_name": stage_name,
                "operation_count": len(result.applied_actions),
                "installed_handles": list(result.installed_handles),
            }
        )
        return result

    @staticmethod
    def _violation(
        action: OperationAction,
        name: str,
        expected: Any,
        actual: Any,
    ) -> InvariantViolation:
        return InvariantViolation(
            name=name,
            field="operation_action",
            expected=expected,
            actual=actual,
            stage_name=action.action_id,
        )


__all__ = ["Qwen3VLOperationApplier"]
