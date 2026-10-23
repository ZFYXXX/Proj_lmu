"""Qwen2.5-Omni executors selected from multimodal v2 contracts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.layer_positions import resolve_layer_positions
from mllm_comp.compressors.contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    CompressionContractRegistry,
    ModalityVariantDescriptor,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, RegistryError
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import ExecutionPhase, Placement, RuntimeMode

_QWEN25_OMNI_AUXILIARY_ARTIFACTS = ("spk_dict.pt",)

_LLM_TOKEN_BOUNDARIES = {
    "llm_decoder_layer_output_audio_tokens": "post_layer",
    "llm_decoder_layer_post_attention_audio_tokens": "post_attention_pre_ffn",
}
_ENCODER_TOKEN_BOUNDARIES = {
    "audio_encoder_layer_output_post_block": "post_layer",
    "audio_encoder_layer_post_mhsa_pre_ffn": "post_attention_pre_ffn",
}
_PROJECTOR_TOKEN_BOUNDARIES = {
    "audio_encoder_output_pre_projector": "pre_projector",
    "audio_projector_output_post_projector": "post_projector",
}


def _resolve_plugin_config(
    config: Mapping[str, Any], variant: ModalityVariantDescriptor,
) -> dict[str, Any]:
    """Resolve only the variant's private parameters, preserving other sections."""

    return {**config, "config": dict(variant.resolve_config(config.get("config", {})))}


def resolve_qwen25_omni_position(
    *, stage: Any, binding: BackbonePlacementBinding,
    algorithm: AlgorithmDescriptor | None = None,
    variant: ModalityVariantDescriptor | None = None,
) -> BackbonePlacementBinding:
    """Resolve a research position without modifying its standard binding."""

    if algorithm is None or variant is None:
        from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts

        registry = builtin_multimodal_compression_contracts()
        variant = variant or registry.variant(binding.variant)
        algorithm = algorithm or registry.algorithm(variant.algorithm)
    plugin_config = _resolve_plugin_config(stage.plugin_config(), variant)
    canonical = binding
    requested = getattr(stage, "operation", {}).get("target_site")
    if requested is None:
        requested = plugin_config["config"].get("target_site")
    target = binding.targets[0]
    requested = target.target_site if requested is None else requested
    if not isinstance(requested, str) or not requested:
        raise ValueError("operation.target_site must be a non-empty string.")
    family = binding.executor_family
    operation = getattr(stage, "operation", {})
    moving_projector_to_layers = family == "projector_token" and requested in (_LLM_TOKEN_BOUNDARIES | _ENCODER_TOKEN_BOUNDARIES)
    if moving_projector_to_layers:
        spec = algorithm.metadata["spec_factory"].from_config(plugin_config)
        rule = binding.strategy_factory(spec)
        requested_family = "llm_token" if requested in _LLM_TOKEN_BOUNDARIES else "encoder_token"
        moving_projector_to_layers = requested_family in getattr(rule, "layer_families", ())
    if moving_projector_to_layers:
        is_llm = requested in _LLM_TOKEN_BOUNDARIES
        family = "llm_token" if is_llm else "encoder_token"
        if "layers" not in operation:
            raise ValueError("Moving a temporal resampling rule to layers requires explicit operation.layers.")
        target = replace(
            target, placement=Placement.LLM_LEVEL if is_llm else Placement.ENCODER_LEVEL,
            required_handles=frozenset((({"llm", "audio_encoder"} if rule.attention_layer_offset is not None else {"llm"}) if is_llm else {"audio_encoder"})),
        )
        binding = replace(binding,
            executor_family=family,
            exclusive_runtime_slot=f"qwen25_omni.{family}",
            prepare_after_slots=("qwen25_omni.encoder_token", "qwen25_omni.projector_token") if is_llm else (),
            canonical_executor_family="projector_token",
            metadata={**binding.metadata,
            "position_policy": "preserve_absolute" if is_llm else "compact_chunk_rebuild_mrope",
            "decode_policy": "prefill_only" if is_llm else "native",
            "required_runtime_context": (
                ("audio_spans", "llm_decoder_layers", "prefill_decode_state", "absolute_mrope_positions")
                + (("encoder_attention_reference",) if rule.attention_layer_offset is not None else ())
                if is_llm else ("audio_spans", "audio_encoder_layers")
            ),
        }, legalizer="qwen25_omni.audio_llm_prefill_cache_labels" if is_llm else "qwen25_omni.audio_encoder_chunk_pooler_mrope")
    if family == "encoder_token" and requested in (_LLM_TOKEN_BOUNDARIES | _PROJECTOR_TOKEN_BOUNDARIES):
        from .audio_method_adapters import EncoderAttentionTokenStrategy

        spec = algorithm.metadata["spec_factory"].from_config(plugin_config)
        rule = binding.strategy_factory(spec)
        EncoderAttentionTokenStrategy.validate_rule(rule)
        is_llm = requested in _LLM_TOKEN_BOUNDARIES
        if is_llm and "layers" not in operation:
            raise ValueError("Moving an encoder rule to the LLM requires explicit operation.layers.")
        family = "llm_token" if is_llm else "projector_token"
        target = replace(target, placement=Placement.LLM_LEVEL if is_llm else Placement.PROJECTOR_LEVEL,
                         required_handles=frozenset({"llm", "audio_encoder"} if is_llm else {"thinker", "audio_encoder", "audio_projector"}))
        binding = replace(binding,
            executor_family=family,
            canonical_executor_family="encoder_token",
            exclusive_runtime_slot=f"qwen25_omni.{family}",
            prepare_after_slots=("qwen25_omni.encoder_token", "qwen25_omni.projector_token") if is_llm else ("qwen25_omni.encoder_token",),
            metadata={**binding.metadata,
            "position_policy": "preserve_absolute" if is_llm else "compact",
            "decode_policy": "prefill_only" if is_llm else "native",
            "required_runtime_context": ("audio_spans", "llm_decoder_layers", "prefill_decode_state", "absolute_mrope_positions", "encoder_attention_reference")
                if is_llm else ("audio_spans", "audio_encoder", "audio_projector", "encoder_attention_reference"),
        }, legalizer="qwen25_omni.audio_llm_prefill_cache_labels" if is_llm else "qwen25_omni.audio_projector_placeholder_mrope")
    layer_override = "layers" in operation or "layer_edge" in operation
    boundaries = {
        "llm_token": _LLM_TOKEN_BOUNDARIES,
        "encoder_token": _ENCODER_TOKEN_BOUNDARIES,
        "projector_token": _PROJECTOR_TOKEN_BOUNDARIES,
    }.get(family, {})
    if layer_override and family not in {"llm_token", "encoder_token"}:
        raise ValueError(f"Application-layer overrides are not implemented for {family!r} yet.")
    if requested == target.target_site and not layer_override:
        return binding
    if requested not in boundaries:
        raise ValueError(
            f"Binding {binding.name!r} has no implemented execution boundary "
            f"for {requested!r}."
        )
    metadata = dict(target.metadata)
    position_changed = requested != target.target_site or bool(binding.metadata.get("position_changed"))
    if layer_override:
        # Full validation against model depth and the rule's number of stages
        # happens when the native controller resolves its layer stack.
        if "layers" in operation:
            positions = resolve_layer_positions(operation, (), boundary=boundaries[requested])
            metadata["application_positions"] = [p.as_dict() for p in positions]
            if binding.canonical_executor_family in {"projector_token", "encoder_token"}:
                # These rule coordinates are not LLM application coordinates.
                # There is no layer coordinate in the standard projector recipe.
                rule_layers = ()
            else:
                spec = algorithm.metadata["spec_factory"].from_config(plugin_config)
                if family == "encoder_token":
                    rule_layers = tuple(sorted(spec.layer_indices)) or tuple(
                        spec.first_layer + i * spec.layer_interval for i in range(len(positions))
                    )
                elif hasattr(spec, "stage_layers"):
                    rule_layers = spec.stage_layers
                elif hasattr(spec, "probe_layer"):
                    rule_layers = (spec.probe_layer,)
                else:
                    # FastAdaSP constant/decay schedules are consecutive. The
                    # controller validates their actual count against model depth;
                    # any valid same-length prefix here denotes the same schedule.
                    count = 1 if spec.task_mode == "sparse" else len(positions)
                    rule_layers = tuple(range(spec.operation_layer, spec.operation_layer + count))
            if rule_layers:
                default_positions = resolve_layer_positions(
                    {}, rule_layers, boundary=boundaries[canonical.targets[0].target_site]
                )
                position_changed = position_changed or positions != default_positions
            else:
                position_changed = True
        else:
            raise ValueError("operation.layer_edge requires explicit operation.layers.")
    return replace(
        binding,
        targets=(replace(target, target_site=requested, metadata=metadata),),
        metadata={**binding.metadata, "position_changed": position_changed},
    )

_REQUIRED_STRATEGY_METHODS: Mapping[str, tuple[str, ...]] = {
    "projector_token": ("runtime_metadata", "compress"),
    "encoder_token": (
        "active_layer_indices",
        "runtime_metadata",
        "enabled",
        "output_lengths",
        "probe",
        "compress",
        "replay",
    ),
    "llm_token": (
        "stage_layer_indices",
        "output_length",
        "runtime_metadata",
    ),
    "llm_operation": ("plan_operations", "runtime_metadata"),
}


@dataclass(frozen=True)
class Qwen25OmniCompressionExecution:
    """Resolved v2 layers consumed by one Qwen executor capability."""

    algorithm: AlgorithmDescriptor
    variant: ModalityVariantDescriptor
    binding: BackbonePlacementBinding
    spec_factory: type[Any]
    strategy_factory: Callable[[Any], Any]

    @property
    def name(self) -> str:
        return self.algorithm.name

    @property
    def executor_family(self) -> str:
        family = self.binding.executor_family
        if not isinstance(family, str) or family not in _REQUIRED_STRATEGY_METHODS:
            raise RegistryError(
                f"Binding {self.binding.name!r} has no supported Qwen executor "
                "capability."
            )
        return family

    @property
    def target_site(self) -> str:
        if len(self.binding.targets) != 1:
            raise RegistryError(
                f"Qwen audio binding {self.binding.name!r} must have one target."
            )
        return self.binding.targets[0].target_site

    @property
    def compression_boundary(self) -> str:
        return {
            **_LLM_TOKEN_BOUNDARIES,
            **_ENCODER_TOKEN_BOUNDARIES,
            **_PROJECTOR_TOKEN_BOUNDARIES,
        }[self.target_site]

    @property
    def required_context(self) -> frozenset[str]:
        values = self.binding.metadata.get("required_runtime_context", ())
        if not isinstance(values, (list, tuple, set, frozenset)) or any(
            not isinstance(value, str) or not value for value in values
        ):
            raise RegistryError(
                f"Binding {self.binding.name!r} has invalid runtime context."
            )
        return frozenset(values)

    def create_plugin(self) -> Any:
        plugin = self.algorithm.kernel()
        if getattr(plugin, "name", None) != self.algorithm.name:
            raise RegistryError(
                f"Algorithm {self.algorithm.name!r} created a mismatched plugin."
            )
        for method_name in ("validate_config", "plan"):
            if not callable(getattr(plugin, method_name, None)):
                raise RegistryError(
                    f"Algorithm {self.algorithm.name!r} plugin is missing "
                    f"{method_name}()."
                )
        return plugin

    def prepare_config(self, config: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
        """Return one effective payload and its spec for both plan and strategy."""

        factory = getattr(self.spec_factory, "from_config", None)
        if not callable(factory):
            raise RegistryError(
                f"Algorithm {self.algorithm.name!r} has no config normalizer."
            )
        effective = _resolve_plugin_config(config, self.variant)
        return effective, factory(effective)

    def create_spec(self, config: Mapping[str, Any]) -> Any:
        return self.prepare_config(config)[1]

    def create_strategy(self, spec: Any, *, operation: Mapping[str, Any] | None = None,
                        stage: Any = None) -> Any:
        specs = {}
        if stage is not None:
            from mllm_comp.compressors.rules import load_rule_specs, rule_specs

            try:
                specs = rule_specs(stage, supported_rules=self.variant.supported_rules)
                rules = load_rule_specs(specs)
            except ConfigError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        strategy = self.strategy_factory(spec)
        if specs:
            strategy.rules = rules
            strategy.rule_specs = specs
            if "scorer" in rules:
                strategy.layer_score_source = "custom"
        if self.binding.canonical_executor_family == "projector_token":
            from .audio_method_adapters import TemporalResamplingEncoderStrategy, TemporalResamplingLLMStrategy

            adapter = TemporalResamplingLLMStrategy if self.executor_family == "llm_token" else TemporalResamplingEncoderStrategy
            strategy = adapter(strategy, tuple((operation or {})["layers"]))
        elif self.binding.canonical_executor_family == "encoder_token":
            from .audio_method_adapters import EncoderAttentionTokenStrategy
            strategy = EncoderAttentionTokenStrategy(strategy, self.binding.targets[0].placement)
        required = _REQUIRED_STRATEGY_METHODS[self.executor_family]
        if self.executor_family == "encoder_token" and not getattr(strategy, "requires_layer_input", True):
            required = tuple(name for name in required if name != "probe")
        if self.executor_family == "llm_token":
            if getattr(strategy, "requires_layer_input", True):
                required += ("score_audio_tokens",)
            # Match the controller's value path. Needing Q/K scores does not
            # imply selection: FastAdaSP scores and then merges values.
            required += (("merge_audio_tokens", "replay_merge_audio_tokens")
                         if callable(getattr(strategy, "merge_audio_tokens", None))
                         else ("select_indices",))
        missing = [
            method_name
            for method_name in required
            if not callable(getattr(strategy, method_name, None))
        ]
        if self.executor_family == "projector_token":
            missing.extend(
                name
                for name in ("score_source", "attention_layer_offset")
                if not hasattr(strategy, name)
            )
        if missing:
            raise RegistryError(
                f"Algorithm {self.algorithm.name!r} does not implement the "
                f"{self.executor_family} executor contract: {sorted(set(missing))}."
            )
        return strategy


def resolve_qwen25_omni_compression_execution(
    *,
    stage: Any,
    binding: BackbonePlacementBinding,
    algorithm: AlgorithmDescriptor | None = None,
    variant: ModalityVariantDescriptor | None = None,
) -> Qwen25OmniCompressionExecution:
    """Resolve executable objects only through the selected v2 identity."""

    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )

    if binding.backbone != "qwen25_omni":
        raise BackendIntegrationError(
            f"Binding {binding.name!r} is not a Qwen2.5-Omni binding."
        )
    if algorithm is None or variant is None:
        registry = builtin_multimodal_compression_contracts()
        variant = registry.variant(binding.variant)
        algorithm = registry.algorithm(variant.algorithm)
    if binding.variant != variant.name or variant.algorithm != algorithm.name:
        raise BackendIntegrationError(
            f"Binding {binding.name!r}, variant {variant.name!r} and algorithm "
            f"{algorithm.name!r} do not form one exact v2 contract."
        )
    if stage.algorithm != algorithm.name or stage.variant != variant.name:
        raise BackendIntegrationError(
            f"Stage {stage.name!r} identity does not match binding {binding.name!r}."
        )
    if stage.plugin != algorithm.name:
        raise BackendIntegrationError(
            f"Stage {stage.name!r} must use plugin={algorithm.name!r}; plugin is "
            "an algorithm record in v2 and cannot select a legacy method."
        )
    binding = resolve_qwen25_omni_position(
        stage=stage, binding=binding, algorithm=algorithm, variant=variant,
    )
    from mllm_comp.compressors.rules import resolve_rule_binding, rule_specs
    try:
        binding = resolve_rule_binding(
            binding=binding, rules=rule_specs(stage, supported_rules=variant.supported_rules),
        )
    except ConfigError as exc:
        raise BackendIntegrationError(str(exc)) from exc
    strategy_factory = binding.strategy_factory
    if not callable(strategy_factory):
        raise BackendIntegrationError(
            f"Binding {binding.name!r} has no Qwen executor strategy factory."
        )
    spec_factory = algorithm.metadata.get("spec_factory")
    if not isinstance(spec_factory, type) or not callable(
        getattr(spec_factory, "from_config", None)
    ):
        raise BackendIntegrationError(
            f"Algorithm {algorithm.name!r} has no v2 config spec factory."
        )
    return Qwen25OmniCompressionExecution(
        algorithm=algorithm,
        variant=variant,
        binding=binding,
        spec_factory=spec_factory,
        strategy_factory=strategy_factory,
    )


def resolve_qwen25_omni_stage_execution(
    stage: Any,
    *, registry: CompressionContractRegistry | None = None,
) -> Qwen25OmniCompressionExecution:
    """Resolve a stage exclusively through its exact v2 binding identity."""

    binding_name = getattr(stage, "binding", None)
    if not binding_name:
        raise BackendIntegrationError(
            "Qwen2.5-Omni compression requires algorithm, variant and binding."
        )
    from mllm_comp.compressors.multimodal_registry import (
        builtin_multimodal_compression_contracts,
    )

    try:
        registry = registry or builtin_multimodal_compression_contracts()
        binding = registry.binding(binding_name)
        variant = registry.variant(binding.variant)
        algorithm = registry.algorithm(variant.algorithm)
    except RegistryError as exc:
        raise BackendIntegrationError(str(exc)) from exc
    return resolve_qwen25_omni_compression_execution(
        stage=stage, binding=binding, algorithm=algorithm, variant=variant,
    )


def create_qwen25_omni_compression_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: BackbonePlacementBinding,
    binding_config: Any,
    execution_phase: ExecutionPhase | str,
    contract_projection: Any | None = None,
) -> Any:
    """Create one of four Qwen executors from an exact v2 binding."""

    del binding_config
    execution = resolve_qwen25_omni_compression_execution(
        stage=stage,
        binding=binding,
        algorithm=(
            getattr(contract_projection, "algorithm", None)
            if contract_projection is not None
            else None
        ),
        variant=(
            getattr(contract_projection, "variant", None)
            if contract_projection is not None
            else None
        ),
    )
    phase = ExecutionPhase(execution_phase)
    if phase not in binding.supported_phases or stage.execution_phase is not phase:
        raise BackendIntegrationError(
            f"Stage {stage.name!r} cannot run binding {binding.name!r} in "
            f"phase={phase.value!r}."
        )
    runtime_mode = RuntimeMode(runtime.metadata.get("runtime_mode", RuntimeMode.DEFAULT.value))
    if execution.executor_family == "llm_operation":
        return _PreparedQwen25OmniOperationActivation(
            model=model,
            stage=stage,
            facade=facade,
            execution=execution,
            phase=runtime.metadata.get(
                "phase", "train" if bool(getattr(model, "training", False)) else "eval",
            ),
        )
    return _create_qwen25_omni_dynamic_controller(
        model=model,
        stage=stage,
        execution=execution,
        runtime_mode=runtime_mode,
    )


def create_qwen25_omni_dynamic_controller(
    *,
    model: Any,
    stage: Any,
    binding: BackbonePlacementBinding,
    runtime_mode: RuntimeMode | str = RuntimeMode.DEFAULT,
    execution: Qwen25OmniCompressionExecution | None = None,
) -> Any:
    """Create a dynamic controller from an exact v2 binding."""

    if execution is None:
        execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
    if execution.executor_family == "llm_operation":
        raise BackendIntegrationError(
            f"{execution.name!r} is a prepare-time operation, not a dynamic controller."
        )
    return _create_qwen25_omni_dynamic_controller(
        model=model,
        stage=stage,
        execution=execution,
        runtime_mode=RuntimeMode(runtime_mode),
    )


def _create_qwen25_omni_dynamic_controller(
    *,
    model: Any,
    stage: Any,
    execution: Qwen25OmniCompressionExecution,
    runtime_mode: RuntimeMode,
) -> Any:
    common = {
        "profile_enabled": runtime_mode is RuntimeMode.PROFILE,
        "execution": execution,
    }
    family = execution.executor_family
    if family == "projector_token":
        from .audio_compression import Qwen25OmniAudioCompressionController

        return Qwen25OmniAudioCompressionController(model, stage, **common)
    if family == "encoder_token":
        from .encoder_compression import Qwen25OmniEncoderCompressionController

        return Qwen25OmniEncoderCompressionController(
            model,
            stage,
            debug_enabled=runtime_mode is RuntimeMode.DEBUG,
            **common,
        )
    if family == "llm_token":
        from .llm_token_compression import Qwen25OmniLLMTokenCompressionController

        return Qwen25OmniLLMTokenCompressionController(
            model,
            stage,
            debug_enabled=runtime_mode is RuntimeMode.DEBUG,
            **common,
        )
    raise BackendIntegrationError(f"Unsupported Qwen executor family {family!r}.")


def apply_qwen25_omni_operation_compression(
    model: Any,
    stage: Any,
    *,
    binding: BackbonePlacementBinding | None = None,
    execution: Qwen25OmniCompressionExecution | None = None,
    model_adapter: Any | None = None,
    operation_applier: Any | None = None,
    phase: str = "eval",
) -> tuple[Any, Any]:
    """Plan and apply one v2 prepare-time operation before runner/optimizer."""

    if execution is None:
        if binding is None:
            binding_name = getattr(stage, "binding", None)
            if not binding_name:
                raise BackendIntegrationError(
                    "Qwen operation compression requires an exact v2 binding."
                )
            from mllm_comp.compressors.multimodal_registry import (
                builtin_multimodal_compression_contracts,
            )

            try:
                binding = builtin_multimodal_compression_contracts().binding(
                    binding_name
                )
            except RegistryError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        execution = resolve_qwen25_omni_compression_execution(
            stage=stage,
            binding=binding,
        )
    if execution.executor_family != "llm_operation":
        raise BackendIntegrationError(
            f"{execution.name!r} is not a prepare-time LLM operation method."
        )
    if model_adapter is None or operation_applier is None:
        from .implementation import Qwen25OmniModelAdapter
        from mllm_comp.pipeline.operation_apply import Qwen25OmniOperationApplier

        model_adapter = model_adapter or Qwen25OmniModelAdapter()
        operation_applier = operation_applier or Qwen25OmniOperationApplier()
    handles = model_adapter.structural_handles(model)
    plugin_config = stage.plugin_config()
    plugin_config["_stage_name"] = stage.name
    plugin_config["_execution_phase"] = stage.execution_phase.value
    plugin_config, spec = execution.prepare_config(plugin_config)
    plan = execution.create_plugin().plan(
        state=ForwardState(),
        context=CompressionContext(
            phase=phase,
            model=model,
            structural_handles=handles,
            extra={"lifecycle": "post_load_pre_runner_pre_optimizer",
                   "audio_operation_strategy": execution.create_strategy(spec, stage=stage)},
        ),
        config=plugin_config,
    )
    result = operation_applier.apply(
        model=model,
        structural_handles=handles,
        actions=plan.operation_actions,
        stage_name=plan.stage_name,
    )
    violations = tuple(getattr(result, "violations", ()))
    if violations:
        details = [
            {
                "name": item.name,
                "field": item.field,
                "expected": item.expected,
                "actual": item.actual,
            }
            for item in violations
        ]
        raise BackendIntegrationError(
            f"Operation compression validation failed: {details}."
        )
    if plan.this_stage_metadata.stats.get("rules_changed"):
        rule_metadata = {key: plan.this_stage_metadata.stats[key] for key in (
            "rules", "rules_changed", "canonical_evidence_applicable", "development_runtime")}
        result.metadata.update(rule_metadata)
        for manifest in getattr(model, "_mllm_comp_operation_manifests", ()):
            if manifest.get("stage_name") == stage.name:
                manifest.update(rule_metadata)
    return plan, result


@dataclass
class _PreparedQwen25OmniOperationActivation:
    """Prepare-time v2 structural rewrite with no false rollback promise."""

    model: Any
    stage: Any
    facade: Any
    execution: Qwen25OmniCompressionExecution
    phase: str
    rollback_safe: bool = field(default=False, init=False)
    plan: Any | None = field(default=None, init=False)
    apply_result: Any | None = field(default=None, init=False)
    _applied: bool = field(default=False, init=False)

    def install(self) -> None:
        if self._applied:
            return
        operation_applier = getattr(self.facade, "operation_applier", None)
        if operation_applier is None:
            raise BackendIntegrationError(
                f"Backbone {self.facade.capabilities().name!r} has no operation "
                "applier."
            )
        self.plan, self.apply_result = apply_qwen25_omni_operation_compression(
            self.model,
            self.stage,
            execution=self.execution,
            model_adapter=self.facade.model_adapter,
            operation_applier=operation_applier,
            phase=self.phase,
        )
        self._applied = True

    def uninstall(self) -> None:
        return None

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = (
            {}
            if self.apply_result is None
            else dict(getattr(self.apply_result, "metadata", {}))
        )
        return {
            "prepared": self._applied,
            "irreversible": True,
            "algorithm": self.execution.name,
            "binding": self.execution.binding.name,
            **metadata,
        }


def export_qwen25_omni_operation_checkpoint(
    *,
    model: Any,
    processor: Any,
    output_dir: str | Path,
    apply_result: Any,
    source_model_dir: str | Path,
) -> dict[str, Any]:
    """Export a reloadable Thinker-only artifact after a v2 rewrite."""

    from mllm_comp.export.operation_export import export_operation_actions

    if bool(getattr(model, "has_talker", False)):
        raise BackendIntegrationError(
            "Qwen LLM-operation export is Thinker-only; disable Talker before "
            "export so the artifact cannot imply unsupported speech weights."
        )
    model_config = getattr(model, "config", None)
    if model_config is not None and hasattr(model_config, "enable_audio_output"):
        model_config.enable_audio_output = False
    apply_result.metadata.update(
        {
            "artifact_scope": "thinker_only",
            "enable_audio_output": False,
        }
    )
    return export_operation_actions(
        model=model,
        processor=processor,
        output_dir=output_dir,
        apply_result=apply_result,
        source_artifact_dir=source_model_dir,
        auxiliary_artifacts=_QWEN25_OMNI_AUXILIARY_ARTIFACTS,
    )


__all__ = [
    "Qwen25OmniCompressionExecution",
    "apply_qwen25_omni_operation_compression",
    "create_qwen25_omni_compression_executor",
    "create_qwen25_omni_dynamic_controller",
    "export_qwen25_omni_operation_checkpoint",
    "resolve_qwen25_omni_compression_execution",
    "resolve_qwen25_omni_stage_execution",
]
