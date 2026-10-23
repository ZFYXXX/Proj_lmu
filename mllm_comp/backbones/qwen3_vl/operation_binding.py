"""Runtime-session binding for Qwen3-VL prepare-time operation replacement."""

from __future__ import annotations

import re
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.rules import rule_specs, load_rule_specs
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import ExecutionPhase, Placement


_MODULE_PATH = re.compile(r"^llm\.layer\[(\d+)\]\.(attn|ffn)$")


@dataclass(frozen=True)
class _ModuleSnapshot:
    path: str
    parent: Any
    attribute: str
    original: Any


@dataclass
class _PreparedQwen3VLOperationActivation:
    """Plan and apply one operation stage with an exact structural rollback."""

    model: Any
    stage: Any
    facade: Any
    plugin: Any
    runtime_phase: ExecutionPhase
    config: dict[str, Any]
    supported_rules: frozenset[str]
    plan: Any | None = field(default=None, init=False)
    apply_result: Any | None = field(default=None, init=False)
    _handles: Mapping[str, Any] | None = field(default=None, init=False)
    _snapshots: list[_ModuleSnapshot] = field(default_factory=list, init=False)
    _applied: bool = field(default=False, init=False)
    _rules: dict[str, Any] = field(default_factory=dict, init=False)
    _rule_specs: dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        try:
            self._rule_specs = rule_specs(self.stage, supported_rules=self.supported_rules)
            self._rules = load_rule_specs(self._rule_specs)
        except ConfigError as exc:
            raise BackendIntegrationError(str(exc)) from exc

    def install(self) -> None:
        if self._applied:
            return
        operation_applier = getattr(self.facade, "operation_applier", None)
        if operation_applier is None:
            raise BackendIntegrationError(
                "Qwen3-VL operation replacement requires an OperationApplier."
            )
        handles = self.facade.model_adapter.structural_handles(self.model)
        if not isinstance(handles, Mapping):
            raise BackendIntegrationError(
                "Qwen3-VL structural_handles() must return a mapping."
            )
        config = self.config
        phase = "train" if bool(getattr(self.model, "training", False)) else "eval"
        plan = self.plugin.plan(
            state=ForwardState(),
            context=CompressionContext(
                phase=phase,
                model=self.model,
                structural_handles=handles,
                extra={
                    "stage_name": self.stage.name,
                    "runtime_phase": self.runtime_phase.value,
                },
            ),
            config=config,
        )
        if plan.stage_name != self.stage.name:
            raise InvariantError(
                f"Qwen3-VL plugin {self.stage.plugin!r} planned stage "
                f"{plan.stage_name!r}, expected {self.stage.name!r}."
            )
        if plan.placement is not Placement.LLM_LEVEL:
            raise InvariantError(
                f"Qwen3-VL operation stage {self.stage.name!r} planned placement "
                f"{plan.placement.value!r}, expected 'llm_level'."
            )
        if plan.execution_phase is not ExecutionPhase.PREPARE:
            raise InvariantError(
                f"Qwen3-VL operation stage {self.stage.name!r} must plan "
                "execution_phase='prepare'."
            )
        if plan.token_actions or not plan.operation_actions:
            raise InvariantError(
                f"Qwen3-VL operation stage {self.stage.name!r} must emit only "
                "operation actions."
            )

        snapshots = [
            self._snapshot(action.module_path, handles)
            for action in plan.operation_actions
        ]
        self._handles = handles
        self._snapshots = snapshots
        self.plan = plan
        custom = {}
        if self._rules:
            custom = {"factory": self._rules["factory"],
                      "factory_spec": self._rule_specs["factory"]}
            for action in plan.operation_actions:
                action.metadata.update(rules=self._rule_specs, rules_changed=True,
                                       canonical_evidence_applicable=False)
        self.apply_result = operation_applier.apply(
            model=self.model,
            structural_handles=handles,
            actions=plan.operation_actions,
            stage_name=self.stage.name,
            **custom,
        )
        self._applied = True

    def uninstall(self) -> None:
        if not self._snapshots:
            self._applied = False
            return
        for snapshot in reversed(self._snapshots):
            setattr(snapshot.parent, snapshot.attribute, snapshot.original)
            if isinstance(self._handles, MutableMapping):
                self._handles[snapshot.path] = snapshot.original
        self._snapshots.clear()
        self._handles = None
        self._applied = False

    def runtime_metadata(self) -> dict[str, Any]:
        plan_metadata = getattr(self.plan, "this_stage_metadata", None)
        stats = dict(getattr(plan_metadata, "stats", {}) or {})
        apply_metadata = dict(getattr(self.apply_result, "metadata", {}) or {})
        return {
            "prepared": self._applied,
            "reversible": True,
            "plugin": self.stage.plugin,
            "stage_name": self.stage.name,
            "execution_phase": ExecutionPhase.PREPARE.value,
            "runtime_phase": self.runtime_phase.value,
            "operation_count": len(getattr(self.plan, "operation_actions", ()) or ()),
            "installed_handles": list(
                getattr(self.apply_result, "installed_handles", ()) or ()
            ),
            "stats": stats,
            **apply_metadata,
            **({"rules": self._rule_specs, "rules_changed": True,
                "canonical_evidence_applicable": False, "development_runtime": True,
                "support_status": "declared"} if self._rule_specs else {}),
        }

    @staticmethod
    def _snapshot(path: str, handles: Mapping[str, Any]) -> _ModuleSnapshot:
        match = _MODULE_PATH.fullmatch(path)
        if match is None:
            raise BackendIntegrationError(
                f"Unsupported Qwen3-VL operation target {path!r}."
            )
        parent_path = f"llm.layer[{match.group(1)}]"
        try:
            parent = handles[parent_path]
        except KeyError as exc:
            raise BackendIntegrationError(
                f"Qwen3-VL structural handle {parent_path!r} is missing."
            ) from exc
        attribute = "self_attn" if match.group(2) == "attn" else "mlp"
        original = getattr(parent, attribute, None)
        if original is None:
            raise BackendIntegrationError(
                f"Qwen3-VL target {path!r} has no module to replace."
            )
        return _ModuleSnapshot(path, parent, attribute, original)


def create_qwen3_operation_executor(
    *, model: Any, stage: Any, facade: Any, binding: Any,
    contract_projection: Any, execution_phase: ExecutionPhase | str, **_: Any,
) -> _PreparedQwen3VLOperationActivation:
    """Construct the selected operation method after earlier prepare stages."""
    phase = ExecutionPhase(execution_phase)
    if phase is not ExecutionPhase.PREPARE or stage.placement is not Placement.LLM_LEVEL:
        raise BackendIntegrationError(
            "Qwen3-VL module replacement requires llm_level preparation."
        )
    plugin = binding.strategy_factory()
    config = stage.plugin_config()
    config["config"] = dict(contract_projection.algorithm_config)
    return _PreparedQwen3VLOperationActivation(
        model=model, stage=stage, facade=facade, plugin=plugin,
        runtime_phase=phase, config=config,
        supported_rules=contract_projection.variant.supported_rules,
    )


__all__ = ["create_qwen3_operation_executor"]
