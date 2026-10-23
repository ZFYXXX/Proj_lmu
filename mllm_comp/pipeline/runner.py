"""Reference token-plan execution; production activation lives in runtime_session."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.errors import InvariantError, RegistryError
from mllm_comp.core.invariants import ForwardStateInvariant
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.state import ForwardState
from mllm_comp.pipeline.stage import CompressionStageConfig


class CompressionPipeline:
    """Reference runner for token plans, caller-supplied legalization and invariants.

    Its documented import and apply contract remain available. Production
    commands install backbone executors through RuntimeCompressionDispatcher
    and CompressionSession, which own their model-hook lifecycle.
    """

    def apply(
        self,
        *,
        state: ForwardState,
        model: Any | None,
        backbone: Any,
        stages: list[CompressionStageConfig],
        registry: Mapping[str, Any],
        phase: Literal["dry_run", "analyze", "train", "eval"],
    ) -> tuple[ForwardState, list[StageMetadata]]:
        current = state
        metadata: list[StageMetadata] = []
        completed: set[str] = set()
        handles = (
            backbone.model_adapter.structural_handles(model)
            if model is not None
            else {}
        )
        for stage in stages:
            if not stage.enabled:
                continue
            missing_dependencies = sorted(set(stage.depends_on) - completed)
            if missing_dependencies:
                raise InvariantError(
                    f"Stage {stage.name!r} has unmet dependencies {missing_dependencies}."
                )
            provider = registry.get(stage.plugin)
            if provider is None:
                raise RegistryError(
                    f"Compressor {stage.plugin!r} is not available in the runtime registry."
                )
            plugin = provider() if isinstance(provider, type) else provider
            plugin_config = stage.plugin_config()
            plugin_config["_stage_name"] = stage.name
            plugin.validate_config(plugin_config)
            plan = plugin.plan(
                state=current,
                context=CompressionContext(
                    phase=phase,
                    model=model,
                    structural_handles=handles,
                ),
                config=plugin_config,
            )
            if plan.stage_name != stage.name or plan.placement is not stage.placement:
                raise InvariantError(
                    f"Compressor {stage.plugin!r} returned a plan that does not match "
                    f"stage {stage.name!r}."
                )
            if plan.operation_actions:
                raise InvariantError(
                    "CompressionPipeline token execution cannot apply operation_actions; "
                    "use an activated backbone OperationApplier."
                )
            if plan.token_actions:
                if plan.legalization_policy is None:
                    raise InvariantError(
                        f"Stage {stage.name!r} emitted token actions without a legalization policy."
                    )
                current = backbone.legalizer.apply(
                    current,
                    token_actions=plan.token_actions,
                    policy=plan.legalization_policy,
                    stage_name=stage.name,
                )
            violations = ForwardStateInvariant().check(current)
            if violations:
                details = "; ".join(
                    f"{item.name}[{item.field}] expected={item.expected!r} "
                    f"actual={item.actual!r}"
                    for item in violations
                )
                raise InvariantError(
                    f"Compression stage {stage.name!r} violated forward-state invariants: "
                    f"{details}"
                )
            stage_metadata = plan.this_stage_metadata or StageMetadata(
                stage_name=stage.name
            )
            current.stage_history[stage.name] = stage_metadata
            metadata.append(stage_metadata)
            completed.add(stage.name)
        return current, metadata
