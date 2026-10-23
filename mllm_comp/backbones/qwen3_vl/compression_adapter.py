"""Qwen3-VL image compression activations and algorithm-to-provider calls.

Physical declarations live in ``compressors.image.contracts.qwen3_vl`` and
are re-exported here. Activations bind shared algorithms to the native
provider's encoder, projector and LLM boundaries. The provider owns masks,
positions, caches and synchronized DeepStack layouts; request state and
checkpoint replay use the shared image lifecycle.

Image and video have separate adapters. Exact support evidence belongs to
the registry, not to the existence of an activation class.
"""

from __future__ import annotations

from mllm_comp.compressors.image.positions import (
    tome_encoder_score_layer,
    tome_encoder_operation,
    tome_position_metadata,
    tome_output_site,
)

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, is_dataclass, replace
from typing import Any

from mllm_comp.backbones._decoder_attention import (
    DecoderAttentionImplementationOverride,
)
from mllm_comp.compressors.image.pyramid_drop import (
    PyramidDropSchedule,
)
from mllm_comp.compressors.image.divprune import (
    DivPruneOutput,
    DivPruneSpec,
    divprune_image,
)
from mllm_comp.compressors.image.saint import (
    SAINTOutput,
    SAINTSpec,
    saint_image_tokens,
)
from mllm_comp.compressors.image.aim import (
    AIMOutput,
    AIMSpec,
    aim_image_tokens,
    replay_aim_image_tokens,
)
from mllm_comp.compressors.image.mmtok import (
    MMTokOutput,
    MMTokSpec,
    merge_mmtok_executor_config,
    mmtok_image_tokens,
)
from mllm_comp.compressors.image.vispruner import (
    VisPrunerOutput,
    VisPrunerSpec,
    merge_vispruner_executor_config,
    vispruner_image_tokens,
)
from mllm_comp.compressors.image.zoo_prune import (
    ZOOProjectorProxyCache,
    ZOOPruneOutput,
    ZOOPruneSpec,
    merge_zoo_prune_executor_config,
    zoo_prune_image_tokens,
)
from mllm_comp.compressors.image.nuwa import (
    NuwaSpec,
    NuwaStage1Result,
    merge_nuwa_executor_config,
    nuwa_stage1_image_tokens,
)
from mllm_comp.compressors.image.restore import (
    RestoreOutput,
    RestoreSpec,
    merge_restore_executor_config,
    restore_image_tokens,
    replay_restore_values,
)
from mllm_comp.compressors.image.decoder_selection import (
    SGLDecoderMethod,
    NuwaDecoderMethod,
    FEATHERDecoderMethod,
    FinePrunerDecoderMethod,
    SparseVLMDecoderMethod,
    PACTDecoderMethod,
    IVCPruneDecoderMethod,
    DARTDecoderMethod,
    D2PrunerDecoderMethod,
    FastVDecoderMethod,
    PyramidDecoderMethod,
    BTPDecoderMethod,
    HalfVDecoderMethod,
    TransPruneDecoderMethod,
    execute_decoder_selection,
)
from mllm_comp.compressors.image.decoder_registration import (
    create_registered_decoder_selection,
    prepare_decoder_selection,
    resolve_decoder_selection_depth,
)
from mllm_comp.compressors.image.sgl import (
    SGLGuideController,
    SGLSpec,
)
from mllm_comp.compressors.image.decoder_config import (
    merge_decoder_executor_config,
)
from mllm_comp.compressors.image.tome import (
    ToMeImageResult,
    tome_merge_image,
    tome_preserve_source_order,
)
from mllm_comp.compressors.image.visionzip import (
    VisionZipOutput,
    visionzip_image,
    visionzip_merge_source_groups,
)
from mllm_comp.compressors.image.prumerge import (
    PruMergeOutput,
    PruMergeSpec,
    prumerge_image_tokens,
    replay_prumerge_values,
)
from mllm_comp.compressors.image.gap import GAPOutput, GAPSpec, gap_image_tokens
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.layout import LayoutDelta, layout_delta_record
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


from mllm_comp.compressors.image.contracts.qwen3_vl import (
    QWEN3_VL_TOME_BINDING,
    QWEN3_VL_VISIONZIP_BINDING,
    QWEN3_VL_PYRAMID_DROP_BINDING,
    QWEN3_VL_DIVPRUNE_BINDING,
    QWEN3_VL_SAINT_BINDING,
    QWEN3_VL_AIM_BINDING,
    QWEN3_VL_PACT_BINDING,
    QWEN3_VL_MMTOK_BINDING,
    QWEN3_VL_VISPRUNER_BINDING,
    QWEN3_VL_ZOO_PRUNE_BINDING,
    QWEN3_VL_NUWA_BINDING,
    QWEN3_VL_RESTORE_BINDING,
    QWEN3_VL_FASTV_BINDING,
    QWEN3_VL_SGL_BINDING,
    QWEN3_VL_DART_BINDING,
    QWEN3_VL_D2PRUNER_BINDING,
    QWEN3_VL_SPARSEVLM_BINDING,
    QWEN3_VL_FINEPRUNER_BINDING,
    QWEN3_VL_BTP_BINDING,
    QWEN3_VL_HALFV_BINDING,
    QWEN3_VL_TRANSPRUNE_BINDING,
    QWEN3_VL_PRUMERGE_BINDING,
    QWEN3_VL_GAP_BINDING,
    QWEN3_VL_FEATHER_BINDING,
    QWEN3_VL_IVC_PRUNE_BINDING,
    _TOME_VARIANT,
    _VISIONZIP_VARIANT,
    _PYRAMID_DROP_VARIANT,
    _DIVPRUNE_VARIANT,
    _SAINT_VARIANT,
    _AIM_VARIANT,
    _PACT_VARIANT,
    _MMTOK_VARIANT,
    _VISPRUNER_VARIANT,
    _ZOO_PRUNE_VARIANT,
    _NUWA_VARIANT,
    _RESTORE_VARIANT,
    _FASTV_VARIANT,
    _SGL_VARIANT,
    _DART_VARIANT,
    _D2PRUNER_VARIANT,
    _SPARSEVLM_VARIANT,
    _FINEPRUNER_VARIANT,
    _BTP_VARIANT,
    _HALFV_VARIANT,
    _TRANSPRUNE_VARIANT,
    _PRUMERGE_VARIANT,
    _GAP_VARIANT,
    _FEATHER_VARIANT,
    _IVC_PRUNE_VARIANT,
    _CONTRACT_VERSION,
    Qwen3VLImageBindingContract,
    QWEN3_VL_IMAGE_BINDING_CONTRACTS,
)


@dataclass(frozen=True)
class Qwen3VLVisionZipResult:
    """Synchronized Qwen3-VL main/DeepStack VisionZip result.

    ``branch_deltas`` contains the exact same ``LayoutDelta`` object for the
    main branch followed by each DeepStack branch.  This makes accidental
    branch-local token selection machine-checkable with ``is`` rather than
    merely comparing equal mappings.
    """

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: VisionZipOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = 1 + len(self.deepstack_image_features)
        if len(self.branch_deltas) != expected:
            raise InvariantError(
                "Qwen3-VL VisionZip must publish one delta reference per main/"
                "DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL VisionZip main and DeepStack branches must share the "
                "same LayoutDelta object."
            )


@dataclass(frozen=True)
class Qwen3VLDivPruneResult:
    """One DivPrune selection replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: DivPruneOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = 1 + len(self.deepstack_image_features)
        if len(self.branch_deltas) != expected:
            raise InvariantError(
                "Qwen3-VL DivPrune must publish one delta reference per main/"
                "DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL DivPrune main and DeepStack branches must share the "
                "same LayoutDelta object."
            )


@dataclass(frozen=True)
class Qwen3VLSAINTResult:
    """One SAINT selection replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: SAINTOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.branch_deltas) != 1 + len(self.deepstack_image_features):
            raise InvariantError(
                "Qwen3-VL SAINT must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL SAINT main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLAIMResult:
    """One AIM merge mapping replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: AIMOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.branch_deltas) != 1 + len(self.deepstack_image_features):
            raise InvariantError(
                "Qwen3-VL AIM must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL AIM main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLMMTokResult:
    """One MMTok selection replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: MMTokOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = 1 + len(self.deepstack_image_features)
        if len(self.branch_deltas) != expected:
            raise InvariantError(
                "Qwen3-VL MMTok must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL MMTok main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLVisPrunerResult:
    """One VisPruner selection replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: VisPrunerOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = 1 + len(self.deepstack_image_features)
        if len(self.branch_deltas) != expected:
            raise InvariantError(
                "Qwen3-VL VisPruner must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL VisPruner main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLZOOPruneResult:
    """One ZOO-Prune selection replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: ZOOPruneOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        expected = 1 + len(self.deepstack_image_features)
        if len(self.branch_deltas) != expected:
            raise InvariantError(
                "Qwen3-VL ZOO-Prune must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL ZOO-Prune main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLNuwaStage1Result:
    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: NuwaStage1Result
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.branch_deltas) != 1 + len(self.deepstack_image_features):
            raise InvariantError(
                "Qwen3-VL Nüwa must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL Nüwa main and DeepStack branches must share one delta."
            )


@dataclass(frozen=True)
class Qwen3VLRestoreResult:
    """RESTORE anchor merge replayed on main and every DeepStack branch."""

    main_image_features: Any
    deepstack_image_features: tuple[Any, ...]
    layout_delta: LayoutDelta
    branch_deltas: tuple[LayoutDelta, ...]
    kernel_result: RestoreOutput
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.branch_deltas) != 1 + len(self.deepstack_image_features):
            raise InvariantError(
                "Qwen3-VL RESTORE must publish one delta per main/DeepStack branch."
            )
        if any(delta is not self.layout_delta for delta in self.branch_deltas):
            raise InvariantError(
                "Qwen3-VL RESTORE branches must share the exact merge delta."
            )


class _Qwen3VLImageActivation:
    """Common idempotent lifecycle for semantic Qwen3-VL hook hosts."""

    algorithm: str
    variant: str
    binding_name: str
    target_site: str

    def _register_decoder_selection(self):
        register = getattr(self.language_model, "register_decoder_layer_loop_controller", None)
        if not callable(register):
            raise BackendIntegrationError("Decoder selection requires register_decoder_layer_loop_controller().")
        method = self.decoder_selection
        return (register(self, schedule=tuple(point.index for point in method.positions),
            target_modality=Modality.IMAGE.value,
            last_deepstack_injection_layer=self.last_deepstack_injection_layer,
            requires_mask_position_repair=True,
            requires_kv_cache_repair=self.execution_phase is ExecutionPhase.GENERATE,
            cache_policy=method.cache_policy.value, synchronize_remaining_deepstack=True),)

    def apply_decoder_selection(self, selection, point, hidden_states, *, image_indices):
        result = execute_decoder_selection(selection, point, hidden_states, image_indices=image_indices,
            media_unit="llm_token",
            source_layout={"backbone": "qwen3_vl", "application_position": point.as_dict()},
            result_layout={"backbone": "qwen3_vl", "application_position": point.as_dict(),
                "target_site": self.decoder_selection.target_site,
                "position_policy": self.decoder_selection.position_policy})
        self._record_layout_delta(result.layout_delta, site=self.decoder_selection.target_site,
            decoder_layer=point.index, application_position=point.as_dict())
        return result


    def __init__(
        self,
        *,
        model: Any,
        runtime: Any = None,
        facade: Any = None,
        stage: Any = None,
        binding: Any = None,
        binding_config: Any = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
        contract_projection: Any = None,
    ) -> None:
        self.model = model
        self.runtime = runtime
        self.facade = facade
        self.stage = stage
        from mllm_comp.compressors.rules import (
            load_rule_specs,
            resolve_rule_binding,
            builtin_rule_specs,
        )

        registered = contract_projection is not None and binding.strategy_factory is not None
        if registered:
            self.algorithm = contract_projection.algorithm.name
            self.variant = contract_projection.variant.name
            self.binding_name = binding.name
            (self.decoder_selection, registered_config, self._rules, self._rule_specs) = create_registered_decoder_selection(
                stage=stage, binding=binding, contract_projection=contract_projection,
                backbone="qwen3_vl", execution_phase=execution_phase, binding_config=binding_config)
        else:
            try:
                self._rule_specs = builtin_rule_specs(stage)
                self._rules = load_rule_specs(self._rule_specs)
            except ConfigError as exc:
                raise BackendIntegrationError(f"Invalid image compression rules: {exc}") from exc
        if self.algorithm in {"visionzip", "prumerge", "gap", "divprune", "aim", "vispruner", "saint", "mmtok", "zoo_prune", "nuwa", "restore"}:
            from mllm_comp.compressors.image.positions import image_projector_sites
            try:
                self.projector_target_sites = image_projector_sites(stage, self.algorithm, "qwen3_vl")
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid image projector position: {exc}") from exc
            self.target_site = self.projector_target_sites[0]
        if not registered and getattr(binding, "position_resolver", None) is not None:
            try:
                binding = binding.position_resolver(stage=stage, binding=binding, binding_config=binding_config)
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid image compression config: {exc}") from exc
        if is_dataclass(binding):
            try:
                binding = resolve_rule_binding(binding=binding, rules=self._rule_specs)
            except ConfigError as exc:
                raise BackendIntegrationError(f"Invalid image compression rules: {exc}") from exc
        self.binding = binding
        self.binding_config = binding_config
        self.execution_phase = ExecutionPhase(execution_phase)
        if registered:
            self.config = registered_config
        elif self.algorithm == "pyramid_drop":
            # Older direct callers keep physical targets/rules in operation.
            # Only method parameters from those sections belong in its Spec.
            self.config = {key: value for key, value in _merged_executor_config(stage, binding_config).items()
                           if key in PyramidDecoderMethod.legacy_config_keys}
            self.config.update(dict(getattr(stage, "config", {}) or {}))
            self.config.update(dict(getattr(binding_config, "config", {}) or {}))
        elif self.algorithm in {
            "pact",
            "fastv",
            "sgl",
            "dart",
            "d2pruner",
            "sparsevlm",
            "finepruner",
            "btp",
            "halfv",
            "transprune",
            "feather",
            "ivc_prune",
        }:
            try:
                self.config = merge_decoder_executor_config(
                    self.algorithm,
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif self.algorithm == "mmtok":
            try:
                self.config = merge_mmtok_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif self.algorithm == "vispruner":
            try:
                self.config = merge_vispruner_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif self.algorithm == "zoo_prune":
            try:
                self.config = merge_zoo_prune_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif self.algorithm == "nuwa":
            try:
                self.config = merge_nuwa_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        elif self.algorithm == "restore":
            try:
                self.config = merge_restore_executor_config(
                    stage=stage,
                    binding_config=binding_config,
                )
            except ValueError as exc:
                raise BackendIntegrationError(str(exc)) from exc
        else:
            self.config = _merged_executor_config(stage, binding_config)
        self._handles: list[Any] = []
        self._layout_delta_records: list[dict[str, Any]] = []
        self._layout_record_count = 0
        self._owns_provider_bridge = False
        self._installed = False
        if binding is not None:
            _validate_image_binding(
                binding,
                expected_variant=self.variant,
                expected_binding=self.binding_name,
                execution_phase=self.execution_phase,
            )
        if stage is not None:
            _validate_image_stage(
                stage,
                expected_algorithm=self.algorithm,
                expected_variant=self.variant,
            )

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        """Install only against a source bridge with the semantic endpoint."""

        if self._installed:
            return
        prepare_decoder_selection(getattr(self, "decoder_selection", None), self.runtime)
        installed: list[Any] = []
        owns_provider_bridge = False
        try:
            if not self._semantic_endpoint_available():
                from .provider_bridge import install_qwen3_vl_transformers_4573_bridge

                install_qwen3_vl_transformers_4573_bridge(self.model)
                owns_provider_bridge = True
            installed.extend(self._register_semantic_hooks())
            for handle in installed:
                _require_removable_handle(handle, algorithm=self.algorithm)
        except BaseException:
            for handle in reversed(installed):
                remove = getattr(handle, "remove", None)
                if callable(remove):
                    remove()
            if owns_provider_bridge:
                from .provider_bridge import (
                    uninstall_qwen3_vl_transformers_4573_bridge,
                )

                uninstall_qwen3_vl_transformers_4573_bridge(
                    self.model,
                    force=True,
                )
            raise
        self._handles = installed
        self._owns_provider_bridge = owns_provider_bridge
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        first_error: BaseException | None = None
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:  # best effort across compound sites
                if first_error is None:
                    first_error = exc
        self._handles.clear()
        owns_provider_bridge = self._owns_provider_bridge
        self._owns_provider_bridge = False
        self._installed = False
        if owns_provider_bridge:
            from .provider_bridge import uninstall_qwen3_vl_transformers_4573_bridge

            uninstall_qwen3_vl_transformers_4573_bridge(self.model)
        if first_error is not None:
            raise first_error

    def _semantic_endpoint_available(self) -> bool:
        endpoints = {
            "tome": (
                getattr(self, "visual", None),
                "register_encoder_dense_boundary_controller",
            ),
            "visionzip": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "prumerge": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "gap": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "divprune": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "saint": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "aim": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "mmtok": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "vispruner": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "zoo_prune": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "nuwa": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "restore": (
                getattr(self, "visual", None),
                "register_post_visual_merger_controller",
            ),
            "pyramid_drop": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
            "fastv": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
            "sgl": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
            "btp": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
            "halfv": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
            "transprune": (
                getattr(self, "language_model", None),
                "register_decoder_layer_loop_controller",
            ),
        }
        owner, name = endpoints[self.algorithm]
        return callable(getattr(owner, name, None))

    def runtime_metadata(self) -> dict[str, Any]:
        """Return diagnostics without representing a support unit as active."""

        return {
            "contract_version": _CONTRACT_VERSION,
            "backbone": "qwen3_vl",
            **({"rules": self._rule_specs, "rules_changed": True, "canonical_evidence_applicable": False, "development_runtime": True} if self._rules else {}),
            "algorithm": self.algorithm,
            "variant": self.variant,
            "binding": self.binding_name,
            "target_modality": Modality.IMAGE.value,
            "video_support_unit": "separate_and_not_implemented_here",
            "target_site": self.target_site,
            **({"projector_target_sites": list(self.projector_target_sites)} if hasattr(self, "projector_target_sites") else {}),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "semantic_hook_count": len(self._handles),
            "layout_record_count": self._layout_record_count,
            "layout_delta_records": [dict(item) for item in self._layout_delta_records],
            "real_checkpoint_verified": False,
            "support_lifecycle_mutated": False,
        }

    def _register_semantic_hooks(self) -> Sequence[Any]:
        raise NotImplementedError

    def clear_diagnostic_records(self) -> None:
        """Start a request without retaining previous diagnostic layouts."""

        self._layout_delta_records.clear()

    def _record_layout_delta(
        self,
        delta: LayoutDelta,
        *,
        site: str,
        **metadata: Any,
    ) -> None:
        record = layout_delta_record(delta)
        record.update(
            {
                "invocation_index": self._layout_record_count,
                "site": site,
                **({"projector_target_sites": list(self.projector_target_sites)} if hasattr(self, "projector_target_sites") else {}),
                **metadata,
            }
        )
        self._layout_delta_records.append(record)
        self._layout_record_count += 1


class Qwen3VLRegisteredDecoderSelectionActivation(_Qwen3VLImageActivation):
    """Install a selected method through Qwen's existing layout and cache repair."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError("Qwen3-VL decoder selection requires language_model.layers.")
        if getattr(self, "decoder_selection", None) is None:
            self.decoder_selection = self.method_factory(stage=self.stage, config=self.config,
                rules=self._rules, backbone="qwen3_vl")
        resolve_decoder_selection_depth(self.decoder_selection, len(layers))
        self.spec = self.decoder_selection.spec
        self.target_site = self.decoder_selection.target_site
        self.application_positions = self.decoder_selection.positions
        self.last_deepstack_injection_layer = len(tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1

        self._decoder_attention_override = (
            DecoderAttentionImplementationOverride(self.model, require_configs=False)
            if getattr(self.decoder_selection, "requires_eager_attention", False) else None)

    def install(self):
        if self.installed:
            return
        if self._decoder_attention_override is not None:
            self._decoder_attention_override.install()
        try:
            super().install()
        except BaseException:
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.restore()
            raise

    def uninstall(self):
        try:
            super().uninstall()
        finally:
            if self._decoder_attention_override is not None:
                self._decoder_attention_override.restore()

    def _semantic_endpoint_available(self):
        return callable(getattr(self.language_model, "register_decoder_layer_loop_controller", None))

    def _register_semantic_hooks(self):
        return self._register_decoder_selection()

    def runtime_metadata(self):
        return {**super().runtime_metadata(),
            "application_positions": [point.as_dict() for point in self.decoder_selection.positions],
            "score_layers": list(self.decoder_selection.score_layers),
            "cache_policy": self.decoder_selection.cache_policy.value,
            "position_policy": self.decoder_selection.position_policy,
            **({"decoder_attention_override": self._decoder_attention_override.metadata()}
               if self._decoder_attention_override is not None else {})}


class Qwen3VLToMeActivation(_Qwen3VLImageActivation):
    """Encoder ToMe plus permanent post-merger prompt compaction.

    Required hook-host method on ``model.model.visual``::

        register_encoder_dense_boundary_controller(
            controller,
            *,
            target_modality="image",
            restore_before=("deepstack_merger[0]", ..., "final_merger"),
        ) -> removable_handle

    The source bridge owns extraction of attention keys at each encoder block
    and calls :meth:`merge_encoder_tokens`.  It must call
    :meth:`restore_dense_boundary` before every named merger.
    """

    algorithm = "tome"
    variant = _TOME_VARIANT
    binding_name = QWEN3_VL_TOME_BINDING
    target_site = "encoder_dense_boundary"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError(
                "Qwen3-VL ToMe requires visual.merger for final dense restore."
            )
        self.restore_before = tuple(
            f"deepstack_merger[{index}]" for index in range(len(self.deepstack_mergers))
        ) + ("final_merger",)
        self.r = _integer_config(self.config, "r", default=1, minimum=0)
        self.output_target_site = tome_output_site(self.stage, "qwen3_vl")
        self.output_r = _integer_config(
            self.config,
            "output_r",
            default=self.r,
            minimum=0,
        )

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_encoder_dense_boundary_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL ToMe requires a verified visual source bridge exposing "
                "register_encoder_dense_boundary_controller(); generic output "
                "hooks cannot preserve encoder attention/window contracts."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                restore_before=self.restore_before,
            ),
        )

    def merge_encoder_tokens(
        self,
        hidden_states: Any,
        keys: Any,
        *,
        r: int | None = None,
        layer_index: int | None = None,
        position: Any | None = None,
        token_sizes: Any | None = None,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
    ) -> ToMeImageResult:
        """Run the backbone-neutral ToMe kernel for one encoder boundary."""

        result = tome_merge_image(
            hidden_states,
            keys,
            rules={name.removeprefix("encoder_"): rule for name, rule in self._rules.items() if name.startswith("encoder_")}, r=self.r if r is None else r,
            token_sizes=token_sizes,
            protected_indices=protected_indices,
            coordinates=coordinates,
        )
        for stream_index, delta in enumerate(result.layout_deltas):
            self._record_layout_delta(
                delta,
                site=tome_encoder_operation(self.stage, "qwen3_vl")["target_site"],
                layer_index=layer_index,
                score_layer=tome_encoder_score_layer(self.stage, "qwen3_vl") if "score_layer" in (getattr(self.stage, "criterion", {}) or {}) else layer_index,
                **({"application_position": position.as_dict()} if position is not None else {}),
                stream_index=stream_index,
            )
        return result

    def merge_output_tokens(
        self,
        hidden_states: Any,
        keys: Any,
        *,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
    ) -> ToMeImageResult:
        """Permanently merge values at the configured Projector boundary."""

        result = tome_preserve_source_order(
            tome_merge_image(
                hidden_states,
                keys.to(hidden_states.device),
                rules={name.removeprefix("output_"): rule for name, rule in self._rules.items() if name.startswith("output_")}, r=self.output_r,
                protected_indices=protected_indices,
                coordinates=coordinates,
                media_unit="image_feature",
            )
        )
        for stream_index, delta in enumerate(result.layout_deltas):
            self._record_layout_delta(
                delta,
                site=self.output_target_site if self.output_target_site.startswith("pre_") else "post_visual_merger_permanent",
                target_site=self.output_target_site,
                projector_position="before" if self.output_target_site.startswith("pre_") else "after",
                stream_index=stream_index,
            )
        return result

    def restore_dense_boundary(
        self,
        compact_hidden_states: Any,
        layout_delta: LayoutDelta | Sequence[LayoutDelta],
        *,
        token_dimension: int = -2,
    ) -> Any:
        """Broadcast compact ToMe groups back to the original patch layout.

        Dense restoration is intentionally value-preserving rather than an
        inverse merge: every source position receives the value of its merged
        result group.  Qwen3-VL's spatial mergers and DeepStack taps therefore
        continue to see their native dense cardinality.
        """

        torch = _torch()
        if not isinstance(compact_hidden_states, torch.Tensor):
            raise TypeError("compact_hidden_states must be a torch.Tensor.")
        if isinstance(layout_delta, LayoutDelta):
            deltas = (layout_delta,)
        else:
            deltas = tuple(layout_delta)
        if not deltas:
            raise ValueError("At least one ToMe LayoutDelta is required.")
        for delta in deltas:
            if delta.dropped_indices:
                raise InvariantError(
                    "Dense ToMe restoration cannot reconstruct independently "
                    "dropped source tokens."
                )
            if delta.result_count != int(compact_hidden_states.shape[token_dimension]):
                raise InvariantError(
                    "ToMe compact tensor cardinality does not match LayoutDelta."
                )

        if len(deltas) == 1:
            indices = _dense_restore_indices(
                deltas[0],
                device=compact_hidden_states.device,
            )
            return compact_hidden_states.index_select(token_dimension, indices)

        if compact_hidden_states.ndim < 3 or len(deltas) != int(
            compact_hidden_states.shape[0]
        ):
            raise InvariantError(
                "Batched ToMe dense restoration requires one LayoutDelta per "
                "leading batch item."
            )
        if token_dimension not in (-2, compact_hidden_states.ndim - 2):
            raise ValueError(
                "Batched dense restoration currently requires token_dimension=-2."
            )
        rows = [
            compact_hidden_states[index].index_select(
                -2,
                _dense_restore_indices(delta, device=compact_hidden_states.device),
            )
            for index, delta in enumerate(deltas)
        ]
        return torch.stack(rows, dim=0)

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            **tome_position_metadata(self.stage, "qwen3_vl"),
            **({"canonical_evidence_applicable": False} if self._rules else {}),
            "target_site": tome_encoder_operation(self.stage, "qwen3_vl")["target_site"],
            "callback_site": self.target_site,
            "dense_restore": True,
            "restore_before": list(self.restore_before),
            "deepstack_synchronized": True,
            "r": self.r,
            "output_r": self.output_r,
            "permanent_llm_token_reduction": self.output_r > 0,
        }


class Qwen3VLVisionZipActivation(_Qwen3VLImageActivation):
    """Broad-projector VisionZip for the main and all DeepStack branches.

    Required hook-host method on ``model.model.visual``::

        register_post_visual_merger_controller(
            controller,
            *,
            target_modality="image",
            synchronize_deepstack=True,
        ) -> removable_handle

    The bridge calls :meth:`compress_image_features` after the final merger and
    all DeepStack mergers, before placeholder scatter/injection.
    """

    algorithm = "visionzip"
    variant = _VISIONZIP_VARIANT
    binding_name = QWEN3_VL_VISIONZIP_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL VisionZip requires visual.merger.")
        self.budget = _integer_config(self.config, "budget", default=32, minimum=1)
        self.dominant_weight = _integer_config(
            self.config,
            "dominant_weight",
            default=27,
            minimum=1,
        )
        self.contextual_weight = _integer_config(
            self.config,
            "contextual_weight",
            default=5,
            minimum=1,
        )

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL VisionZip requires a verified visual source bridge "
                "exposing register_post_visual_merger_controller(); a generic "
                "visual hook cannot distinguish image and video calls safely."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=True,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any,
        keys: Any,
        budget: int | None = None,
        attention_mode: str = "auto",
        cls_index: int | None = None,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLVisionZipResult:
        """Compress main image features and replay its delta on every tap."""

        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL VisionZip received a DeepStack feature count that does "
                "not match visual.deepstack_merger_list."
            )
        kernel = visionzip_image(
            main_image_features,
            attention,
            keys,
            rules=self._rules, budget=self.budget if budget is None else budget,
            attention_mode=attention_mode,  # type: ignore[arg-type]
            cls_index=cls_index,
            protected_indices=protected_indices,
            dominant_weight=self.dominant_weight,
            contextual_weight=self.contextual_weight,
            coordinates=coordinates,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
            },
        )
        shared_delta = kernel.layout_delta
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
            image_index=image_index,
        )
        compressed_deepstack = tuple(
            _visionzip_by_layout_delta(
                features,
                shared_delta,
                rules=self._rules, representatives=tuple(int(index) for index in kernel.indices.tolist()),
                contextual_indices=tuple(kernel.metadata["contextual_indices"]),
            )
            for features in branches
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLVisionZipResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "deepstack_branch_count": len(compressed_deepstack),
                "shared_main_deepstack_delta": True,
            },
        )

    # Semantic hook hosts use this descriptive callback name.
    compress_after_visual_mergers = compress_image_features

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "budget": self.budget,
            "dominant_weight": self.dominant_weight,
            "contextual_weight": self.contextual_weight,
            "deepstack_branch_count": len(self.deepstack_mergers),
            "shared_main_deepstack_delta": True,
        }


class _Qwen3VLAttentionVisualActivation(_Qwen3VLImageActivation):
    """Shared final-vision-attention adapter for PruMerge and GAP."""

    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if self.algorithm == "prumerge":
            values = {
                name: self.config[name]
                for name in PruMergeSpec.__dataclass_fields__
                if name in self.config
            }
            try:
                self.spec = PruMergeSpec(**values)
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid PruMerge config: {exc}") from exc
        else:
            values = {
                name: self.config[name]
                for name in GAPSpec.__dataclass_fields__
                if name in self.config
            }
            try:
                self.spec = GAPSpec(**values)
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid GAP config: {exc}") from exc

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual, "register_post_visual_merger_controller", None
        )
        if not callable(register):
            raise BackendIntegrationError(
                f"Qwen3-VL {self.algorithm} requires the verified post-merger bridge."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=True,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any,
        keys: Any,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
        **kwargs: Any,
    ) -> Qwen3VLVisionZipResult:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        if kwargs:
            raise BackendIntegrationError(
                f"Qwen3-VL {self.algorithm} callback-time overrides are unsupported."
            )
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                f"Qwen3-VL {self.algorithm} DeepStack branch count changed."
            )

        def run(features: Any) -> PruMergeOutput | GAPOutput:
            common = {
                "protected_indices": protected_indices,
                "coordinates": coordinates,
                "media_unit": "image_feature",
                "source_layout": {
                    "backbone": "qwen3_vl",
                    "image_index": image_index,
                    "importance_source": "final_visual_mean_received",
                },
                "result_layout": {
                    "backbone": "qwen3_vl",
                    "image_index": image_index,
                    "target_site": self.target_site,
                    "position_policy": "preserve_original",
                },
            }
            if self.algorithm == "prumerge":
                return prumerge_image_tokens(
                    features, attention, keys, rules=self._rules, spec=self.spec, **common
                )
            return gap_image_tokens(features, attention, rules=self._rules, spec=self.spec, **common)

        kernel = run(main_image_features)
        if self.algorithm == "gap":
            branch_results = tuple(replace(kernel, tokens=features.index_select(
                0, kernel.indices.to(device=features.device))) for features in branches)
        elif self._rules:
            branch_results = tuple(replace(kernel, tokens=replay_prumerge_values(
                features, kernel, rules=self._rules)) for features in branches)
        else:
            branch_results = tuple(run(features) for features in branches)
        if any(
            result.layout_delta.old_to_new != kernel.layout_delta.old_to_new
            for result in branch_results
        ):
            raise InvariantError(
                f"Qwen3-VL {self.algorithm} selected different DeepStack layouts."
            )
        delta = kernel.layout_delta
        self._record_layout_delta(
            delta,
            site=self.target_site,
            image_index=image_index,
            shared_main_deepstack_delta=True,
        )
        return Qwen3VLVisionZipResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=tuple(result.tokens for result in branch_results),
            layout_delta=delta,
            branch_deltas=(delta,) * (1 + len(branch_results)),
            kernel_result=kernel,  # type: ignore[arg-type]
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "image_index": image_index,
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features


class Qwen3VLPruMergeActivation(_Qwen3VLAttentionVisualActivation):
    algorithm = "prumerge"
    variant = _PRUMERGE_VARIANT
    binding_name = QWEN3_VL_PRUMERGE_BINDING
    target_site = "post_visual_merger_with_deepstack"


class Qwen3VLGAPActivation(_Qwen3VLAttentionVisualActivation):
    algorithm = "gap"
    variant = _GAP_VARIANT
    binding_name = QWEN3_VL_GAP_BINDING
    target_site = "post_visual_merger_with_deepstack"


class Qwen3VLDivPruneActivation(_Qwen3VLImageActivation):
    """Layer-0 DivPrune synchronized across main and DeepStack features."""

    algorithm = "divprune"
    variant = _DIVPRUNE_VARIANT
    binding_name = QWEN3_VL_DIVPRUNE_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL DivPrune requires visual.merger.")
        self.spec = _divprune_spec(self.config)

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL DivPrune requires a verified visual source bridge "
                "exposing register_post_visual_merger_controller()."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=False,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any | None = None,
        keys: Any | None = None,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLDivPruneResult:
        del attention, keys
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL DivPrune received a DeepStack feature count that does "
                "not match visual.deepstack_merger_list."
            )
        kernel = divprune_image(
            main_image_features,
            rules=self._rules, spec=self.spec,
            protected_indices=protected_indices,
            coordinates=coordinates,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "position_policy": "sparse_mrope_rebuild",
            },
        )
        shared_delta = kernel.layout_delta
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
            planned_count=kernel.metadata["effective_target_count"],
            selected_indices=list(shared_delta.kept_indices),
            image_index=image_index,
        )
        compressed_deepstack = tuple(
            _select_by_layout_delta(features, shared_delta) for features in branches
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLDivPruneResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "selected_indices": list(shared_delta.kept_indices),
                "deepstack_branch_count": len(compressed_deepstack),
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "retain_ratio": float(self.spec.retain_ratio),
            "target_count": self.spec.target_count,
            "prune_layer": self.spec.prune_layer,
            "distance": self.spec.distance,
            "selection": self.spec.selection,
            "deepstack_branch_count": len(self.deepstack_mergers),
            "shared_main_deepstack_delta": True,
            "requires_visual_metrics": False,
        }


class Qwen3VLSAINTActivation(_Qwen3VLImageActivation):
    """SAINT at selected merger sides with shared DeepStack selection."""

    algorithm = "saint"
    variant = _SAINT_VARIANT
    binding_name = QWEN3_VL_SAINT_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL SAINT requires visual.merger.")
        self.spec = _saint_spec(self.config)

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual, "register_post_visual_merger_controller", None
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL SAINT requires the verified post-merger source bridge."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=True,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any | None = None,
        keys: Any,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLSAINTResult:
        del attention
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL SAINT DeepStack feature count does not match the model."
            )
        metric_keys = keys
        if (
            getattr(keys, "ndim", None) == 3
            and int(keys.shape[0]) != int(main_image_features.shape[0])
            and int(keys.shape[1]) == int(main_image_features.shape[0])
        ):
            metric_keys = keys.permute(1, 0, 2).contiguous()
        kernel = saint_image_tokens(
            main_image_features,
            metric_keys,
            rules=self._rules, spec=self.spec,
            protected_indices=protected_indices,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "adaptation": "native_spatial_groups_shared_deepstack_selection",
            },
        )
        shared_delta = kernel.layout_delta
        compressed_branches = tuple(
            _select_by_layout_delta(value, shared_delta) for value in branches
        )
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            image_index=image_index,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
        )
        return Qwen3VLSAINTResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_branches,
            layout_delta=shared_delta,
            branch_deltas=(shared_delta,) * (1 + len(compressed_branches)),
            kernel_result=kernel,
            metadata={
                **dict(kernel.metadata),
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features


class Qwen3VLAIMActivation(_Qwen3VLImageActivation):
    """AIM at the selected merger sides, sharing main clusters with every tap."""

    algorithm = "aim"
    variant = _AIM_VARIANT
    binding_name = QWEN3_VL_AIM_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL AIM requires visual.merger.")
        self.spec = _aim_spec(self.config)

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual, "register_post_visual_merger_controller", None
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL AIM requires the verified post-merger source bridge."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=False,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any | None = None,
        keys: Any | None = None,
        protected_indices: Iterable[int] = (),
        coordinates: Iterable[Any] = (),
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLAIMResult:
        del attention, keys, coordinates
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL AIM DeepStack feature count does not match the model."
            )
        kernel = aim_image_tokens(
            main_image_features,
            rules=self._rules, spec=self.spec,
            protected_indices=protected_indices,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
            },
        )
        shared_delta = kernel.layout_delta
        # Keep the original DeepStack rule: means of final source members.
        # Main AIM instead averages current nodes at each merge iteration.
        # Scoring or grouping rules do not replace this value aggregation.
        compressed_branches = tuple(
            replay_aim_image_tokens(value, kernel, rules=self._rules) if "aggregator" in self._rules
            else _mean_by_layout_delta(value, shared_delta) for value in branches
        )
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            image_index=image_index,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
        )
        return Qwen3VLAIMResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_branches,
            layout_delta=shared_delta,
            branch_deltas=(shared_delta,) * (1 + len(compressed_branches)),
            kernel_result=kernel,
            metadata={
                **dict(kernel.metadata),
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features


class Qwen3VLMMTokActivation(_Qwen3VLImageActivation):
    """MMTok coverage selection synchronized across main and DeepStack."""

    algorithm = "mmtok"
    variant = _MMTOK_VARIANT
    binding_name = QWEN3_VL_MMTOK_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL MMTok requires visual.merger.")
        self.spec = _mmtok_spec(self.config)

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL MMTok requires a verified visual source bridge "
                "exposing register_post_visual_merger_controller()."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=False,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        vision_reference: Any,
        text_embeddings: Any,
        image_index: int,
        projected_reference: Any | None = None,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLMMTokResult:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL MMTok received a DeepStack feature count that does "
                "not match visual.deepstack_merger_list."
            )
        if projected_reference is not None and (
            getattr(main_image_features, "ndim", None) != 2
            or int(main_image_features.shape[0]) != int(projected_reference.shape[0])
        ):
            raise InvariantError("Projector values and fixed scoring reference must share token rows.")
        scoring_features = main_image_features if projected_reference is None else projected_reference
        kernel = mmtok_image_tokens(
            scoring_features,
            vision_reference.to(device=scoring_features.device),
            text_embeddings.to(device=scoring_features.device),
            rules=self._rules, spec=self.spec,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
                "vision_source": "pre_merger_visual_features",
                "text_source": "prompt_input_embeddings",
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "position_policy": "sparse_mrope_rebuild",
            },
        )
        if projected_reference is not None or kernel.tokens.device != main_image_features.device:
            kernel = replace(kernel, tokens=main_image_features.index_select(
                0, kernel.indices.to(main_image_features.device),
            ))
        shared_delta = kernel.layout_delta
        compressed_deepstack = tuple(
            _select_by_layout_delta(features, shared_delta) for features in branches
        )
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            requires_full_projected_scoring_reference=True,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
            planned_count=kernel.metadata["effective_target_count"],
            selected_indices=list(shared_delta.kept_indices),
            image_index=image_index,
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLMMTokResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "selected_indices": list(shared_delta.kept_indices),
                "deepstack_branch_count": len(compressed_deepstack),
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "retain_ratio": float(self.spec.retain_ratio),
            "target_count": self.spec.target_count,
            "alpha": float(self.spec.alpha),
            "text_temperature": float(self.spec.text_temperature),
            "vision_temperature": float(self.spec.vision_temperature),
            "deepstack_branch_count": len(self.deepstack_mergers),
            "shared_main_deepstack_delta": True,
            "requires_visual_metrics": False,
            "requires_full_projected_scoring_reference": True,
        }


class Qwen3VLVisPrunerActivation(_Qwen3VLImageActivation):
    """Visual-cue selection synchronized across main and DeepStack branches."""

    algorithm = "vispruner"
    variant = _VISPRUNER_VARIANT
    binding_name = QWEN3_VL_VISPRUNER_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError(
                "Qwen3-VL VisPruner requires visual.merger."
            )
        self.spec = _vispruner_spec(self.config)

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL VisPruner requires a verified visual source bridge "
                "exposing register_post_visual_merger_controller()."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=True,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any,
        vision_reference: Any,
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLVisPrunerResult:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL VisPruner received a DeepStack feature count that does "
                "not match visual.deepstack_merger_list."
            )
        kernel = vispruner_image_tokens(
            main_image_features,
            vision_reference.to(device=main_image_features.device),
            attention.to(device=main_image_features.device),
            rules=self._rules, spec=self.spec,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
                "attention_source": "final_visual_mean_received",
                "selection_feature_source": "pre_merger_visual_features",
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "position_policy": "sparse_mrope_rebuild",
            },
        )
        shared_delta = kernel.layout_delta
        compressed_deepstack = tuple(
            _select_by_layout_delta(features, shared_delta) for features in branches
        )
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
            planned_count=kernel.metadata["effective_target_count"],
            selected_indices=list(shared_delta.kept_indices),
            image_index=image_index,
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLVisPrunerResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "selected_indices": list(shared_delta.kept_indices),
                "deepstack_branch_count": len(compressed_deepstack),
                "shared_main_deepstack_delta": True,
            },
        )

    compress_after_visual_mergers = compress_image_features

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "target_count": self.spec.target_count,
            "important_ratio": float(self.spec.important_ratio),
            "removal_step": self.spec.removal_step,
            "deepstack_branch_count": len(self.deepstack_mergers),
            "shared_main_deepstack_delta": True,
            "requires_visual_metrics": True,
        }


class Qwen3VLZOOPruneActivation(_Qwen3VLImageActivation):
    """Projector-sensitivity selection synchronized across DeepStack."""

    algorithm = "zoo_prune"
    variant = _ZOO_PRUNE_VARIANT
    binding_name = QWEN3_VL_ZOO_PRUNE_BINDING
    target_site = "post_visual_merger_with_deepstack"
    requires_visual_metrics = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, _ = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError(
                "Qwen3-VL ZOO-Prune requires visual.merger."
            )
        self.spec = _zoo_prune_spec(self.config)
        self._proxy_cache = ZOOProjectorProxyCache()

    def uninstall(self) -> None:
        try:
            super().uninstall()
        finally:
            self._proxy_cache.clear()

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        if not callable(register):
            raise BackendIntegrationError(
                "Qwen3-VL ZOO-Prune requires a verified visual source bridge "
                "exposing register_post_visual_merger_controller()."
            )
        return (
            register(
                self,
                target_modality=Modality.IMAGE.value,
                synchronize_deepstack=True,
                requires_visual_metrics=False,
            ),
        )

    def compress_image_features(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        projector_inputs: Any,
        projector_layers: Sequence[Any],
        image_index: int,
        projected_reference: Any | None = None,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLZOOPruneResult:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL ZOO-Prune received a DeepStack feature count that "
                "does not match visual.deepstack_merger_list."
            )
        if projected_reference is not None and (
            getattr(main_image_features, "ndim", None) != 2
            or int(main_image_features.shape[0]) != int(projected_reference.shape[0])
        ):
            raise InvariantError("Projector values and fixed scoring reference must share token rows.")
        scoring_features = main_image_features if projected_reference is None else projected_reference
        scoring_features = scoring_features.to(device=projector_inputs.device)
        kernel = zoo_prune_image_tokens(
            scoring_features,
            projector_inputs,
            projector_layers,
            rules=self._rules, spec=self.spec,
            proxy_cache=self._proxy_cache,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
                "projector_input": "native_final_merger_mlp_input",
                "selection_scope": "per_image",
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.target_site,
                "position_policy": "sparse_mrope_rebuild",
            },
        )
        if projected_reference is not None or kernel.tokens.device != main_image_features.device:
            kernel = replace(kernel, tokens=main_image_features.index_select(
                0, kernel.indices.to(main_image_features.device),
            ))
        shared_delta = kernel.layout_delta
        compressed_deepstack = tuple(
            _select_by_layout_delta(features, shared_delta) for features in branches
        )
        self._record_layout_delta(
            shared_delta,
            site=self.target_site,
            requires_full_projected_scoring_reference=True,
            deepstack_branch_count=len(branches),
            shared_main_deepstack_delta=True,
            planned_count=kernel.metadata["effective_target_count"],
            selected_indices=list(shared_delta.kept_indices),
            low_rank_solver_used=kernel.metadata["low_rank_solver_used"],
            image_index=image_index,
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLZOOPruneResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": self.algorithm,
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "selected_indices": list(shared_delta.kept_indices),
                "low_rank_solver_used": kernel.metadata[
                    "low_rank_solver_used"
                ],
                "deepstack_branch_count": len(compressed_deepstack),
                "shared_main_deepstack_delta": True,
                "training_free": True,
            },
        )

    compress_after_visual_mergers = compress_image_features

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "retention_ratio": float(self.spec.retention_ratio),
            "target_count": self.spec.target_count,
            "num_directions": self.spec.num_directions,
            "step_size": float(self.spec.step_size),
            "projector_proxy": self.spec.projector_proxy,
            "low_rank_rank": self.spec.low_rank_rank,
            "deepstack_branch_count": len(self.deepstack_mergers),
            "shared_main_deepstack_delta": True,
            "requires_visual_metrics": False,
            "requires_full_projected_scoring_reference": True,
            "training_free": True,
            "single_image_only": True,
        }


class Qwen3VLNuwaActivation(_Qwen3VLImageActivation):
    """Nüwa composite controller for visual Stage 1 and decoder Stage 2."""

    algorithm = "nuwa"
    variant = _NUWA_VARIANT
    binding_name = QWEN3_VL_NUWA_BINDING
    target_site = "post_visual_merger_and_decoder_stage2"
    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        if getattr(self.visual, "merger", None) is None:
            raise BackendIntegrationError("Qwen3-VL Nüwa requires visual.merger.")
        try:
            self.spec = NuwaSpec(**dict(self.config))
            self.decoder_selection = NuwaDecoderMethod(
                stage=self.stage,
                spec=self.spec,
                rules=self._rules,
                backbone="qwen3_vl",
                num_layers=len(self.language_model.layers),
            )
            self.application_position = self.decoder_selection.positions[0]
            self.decoder_target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid Nüwa config: {exc}") from exc
        self.last_deepstack_injection_layer = len(self.deepstack_mergers) - 1

    def _semantic_endpoint_available(self) -> bool:
        return callable(
            getattr(self.visual, "register_post_visual_merger_controller", None)
        ) and callable(
            getattr(
                self.language_model,
                "register_decoder_layer_loop_controller",
                None,
            )
        )

    def _register_semantic_hooks(self):
        visual_handle = self.visual.register_post_visual_merger_controller(
            self,
            target_modality=Modality.IMAGE.value,
            synchronize_deepstack=True,
            requires_visual_metrics=True,
            requires_semantic_hidden_range=True,
        )
        try:
            decoder_handles = self._register_decoder_selection()
        except BaseException:
            visual_handle.remove()
            raise
        return (visual_handle, *decoder_handles)

    def compress_after_visual_mergers(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any,
        keys: Any,
        semantic_reference: Any,
        feature_grid: Sequence[int],
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLNuwaStage1Result:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL Nüwa DeepStack feature count differs from native mergers."
            )
        if (
            not isinstance(feature_grid, Sequence)
            or len(feature_grid) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 1
                for value in feature_grid
            )
        ):
            raise BackendIntegrationError(
                "Qwen3-VL Nüwa requires an exact post-merger (height, width)."
            )
        kernel = nuwa_stage1_image_tokens(
            main_image_features,
            semantic_reference,
            attention,
            keys,
            rules=self._rules, grid_height=int(feature_grid[0]),
            grid_width=int(feature_grid[1]),
            spec=self.spec,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
                "stage": "stage1",
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.projector_target_sites[0],
                "position_policy": "native_sparse_mrope_rpme",
            },
        )
        weights = kernel.aggregation_weights
        compressed_deepstack = tuple(
            weights.to(device=branch.device, dtype=branch.dtype) @ branch
            for branch in branches
        )
        shared_delta = kernel.layout_delta
        self._record_layout_delta(
            shared_delta,
            site=self.projector_target_sites[0],
            stage="stage1",
            image_index=image_index,
            deepstack_branch_count=len(branches),
            selected_indices=list(shared_delta.kept_indices),
            shared_main_deepstack_delta=True,
            weighted_deepstack_aggregation=True,
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLNuwaStage1Result(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": "nuwa",
                "stage": "stage1",
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "shared_main_deepstack_delta": True,
                "weighted_deepstack_aggregation": True,
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "composite_binding": True,
            "application_position": self.application_position.as_dict(),
            "decoder_target_site": self.decoder_target_site, "score_layer": self.spec.stage2_score_layer,
            "stage2_after_layer": self.spec.stage2_after_layer,
            "stage2_retain_ratio": self.spec.stage2_retain_ratio,
            "cache_policy": "compact_prefix",
            "position_policy": "native_sparse_mrope_rpme",
            "deepstack_branch_count": len(self.deepstack_mergers),
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLRestoreActivation(_Qwen3VLImageActivation):
    """RESTORE visual anchor merge plus per-layer attention calibration."""

    algorithm = "restore"
    variant = _RESTORE_VARIANT
    binding_name = QWEN3_VL_RESTORE_BINDING
    target_site = "post_visual_merger_and_decoder_attention"
    requires_visual_metrics = True

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        self.deepstack_mergers = tuple(
            getattr(self.visual, "deepstack_merger_list", ()) or ()
        )
        values = dict(self.config)
        tokens_per_crop = values.pop("tokens_per_crop", None)
        if tokens_per_crop is not None:
            if "target_count" in values:
                raise BackendIntegrationError(
                    "RESTORE accepts either target_count or tokens_per_crop, not both."
                )
            values["target_count"] = tokens_per_crop
        try:
            self.spec = RestoreSpec(**values)
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid RESTORE config: {exc}") from exc

    def _semantic_endpoint_available(self) -> bool:
        return callable(
            getattr(self.visual, "register_post_visual_merger_controller", None)
        ) and callable(
            getattr(
                self.language_model,
                "register_decoder_layer_loop_controller",
                None,
            )
        )

    def _register_semantic_hooks(self) -> Sequence[Any]:
        register_visual = getattr(
            self.visual,
            "register_post_visual_merger_controller",
            None,
        )
        register_decoder = getattr(
            self.language_model,
            "register_decoder_layer_loop_controller",
            None,
        )
        if not callable(register_visual) or not callable(register_decoder):
            raise BackendIntegrationError(
                "Qwen3-VL RESTORE requires visual and decoder semantic endpoints."
            )
        visual_handle = register_visual(
            self,
            target_modality=Modality.IMAGE.value,
            synchronize_deepstack=True,
            requires_visual_metrics=True,
            requires_semantic_hidden_range=False,
        )
        try:
            decoder_handle = register_decoder(
                self,
                schedule=(),
                target_modality=Modality.IMAGE.value,
                requires_mask_position_repair=False,
                requires_kv_cache_repair=False,
                cache_policy="stable",
                cache_transform="none",
                attention_calibration="token_mass_plus_mrope_distance",
            )
        except BaseException:
            remove = getattr(visual_handle, "remove", None)
            if callable(remove):
                remove()
            raise
        return (visual_handle, decoder_handle)

    def compress_after_visual_mergers(
        self,
        main_image_features: Any,
        deepstack_image_features: Sequence[Any],
        *,
        attention: Any,
        vision_reference: Any,
        image_index: int,
        target_modality: Modality | str = Modality.IMAGE,
    ) -> Qwen3VLRestoreResult:
        _require_image_modality(target_modality)
        _require_image_index(image_index)
        branches = tuple(deepstack_image_features)
        if len(branches) != len(self.deepstack_mergers):
            raise InvariantError(
                "Qwen3-VL RESTORE DeepStack feature count differs from native mergers."
            )
        kernel = restore_image_tokens(
            main_image_features,
            vision_reference,
            attention,
            rules=self._rules, spec=self.spec,
            media_unit="image_feature",
            source_layout={
                "backbone": "qwen3_vl",
                "branch": "main_image_features",
                "image_index": image_index,
                "target_modality": Modality.IMAGE.value,
            },
            result_layout={
                "backbone": "qwen3_vl",
                "image_index": image_index,
                "target_site": self.projector_target_sites[0],
                "position_policy": "native_sparse_mrope_representatives",
            },
        )
        weights = kernel.membership.to(dtype=kernel.token_sizes.dtype)
        weights = weights / kernel.token_sizes[:, None]
        compressed_deepstack = tuple(
            replay_restore_values(branch, kernel, rules=self._rules) if "aggregator" in self._rules
            else weights.to(device=branch.device, dtype=branch.dtype) @ branch
            for branch in branches
        )
        shared_delta = kernel.layout_delta
        self._record_layout_delta(
            shared_delta,
            site=self.projector_target_sites[0],
            image_index=image_index,
            deepstack_branch_count=len(branches),
            representative_indices=[
                int(value)
                for value in kernel.representative_indices.detach().cpu().tolist()
            ],
            token_sizes=[
                float(value) for value in kernel.token_sizes.detach().cpu().tolist()
            ],
            shared_main_deepstack_delta=True,
        )
        branch_deltas = (shared_delta,) * (1 + len(compressed_deepstack))
        return Qwen3VLRestoreResult(
            main_image_features=kernel.tokens,
            deepstack_image_features=compressed_deepstack,
            layout_delta=shared_delta,
            branch_deltas=branch_deltas,
            kernel_result=kernel,
            metadata={
                "backbone": "qwen3_vl",
                "algorithm": "restore",
                "target_modality": Modality.IMAGE.value,
                "image_index": image_index,
                "source_tokens": shared_delta.source_count,
                "result_tokens": shared_delta.result_count,
                "shared_main_deepstack_delta": True,
                "attention_calibration": "token_mass_plus_mrope_distance",
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "composite_binding": True,
            "target_count": self.spec.target_count,
            "cache_policy": "stable",
            "cache_transform": "none",
            "position_policy": "native_sparse_mrope_representatives",
            "deepstack_branch_count": len(self.deepstack_mergers),
            "attention_calibration": "token_mass_plus_mrope_distance",
        }


class Qwen3VLPyramidDropActivation(_Qwen3VLImageActivation):
    """PyramidDrop controller registered inside the Qwen3-VL decoder loop.

    Required hook-host method on ``model.model.language_model``::

        register_decoder_layer_loop_controller(
            controller,
            *,
            schedule,
            target_modality="image",
            last_deepstack_injection_layer=N,
            requires_mask_position_repair=True,
            requires_kv_cache_repair=True_or_false,
        ) -> removable_handle

    A plain decoder-layer output hook is intentionally not accepted because it
    cannot atomically repair masks, positions, or generation caches.
    """

    algorithm = "pyramid_drop"
    variant = _PYRAMID_DROP_VARIANT
    binding_name = QWEN3_VL_PYRAMID_DROP_BINDING
    target_site = "llm_decoder_layer_output_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL PyramidDrop requires language_model.layers."
            )
        self.num_layers = len(layers)
        deepstack_count = len(
            tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())
        )
        # Upstream Qwen3-VL injects deepstack_visual_embeds[i] after decoder
        # layer i.  No DeepStack branch is represented by -1.
        self.last_deepstack_injection_layer = deepstack_count - 1
        self.decoder_selection = PyramidDecoderMethod(stage=self.stage, config=self.config, rules=self._rules, backbone="qwen3_vl")
        self.decoder_selection.resolve_depth(self.num_layers)
        self.keep_ratio = self.decoder_selection.spec.keep_ratio
        self.minimum_keep_tokens = self.decoder_selection.spec.minimum_keep_tokens
        self.schedule = self.decoder_selection.schedule
        _validate_pyramid_schedule(self.schedule)
        self.application_positions = self.decoder_selection.positions
        self.target_site = self.decoder_selection.target_site

    @property
    def first_drop_layer(self) -> int:
        return self.schedule.layer_indices[0]

    def _register_semantic_hooks(self) -> Sequence[Any]:
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "requires_decoder_loop_control": True,
            "schedule": list(self.schedule.layer_indices),
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.schedule.layer_indices),
            "first_drop_layer": self.first_drop_layer,
            "last_deepstack_injection_layer": (self.last_deepstack_injection_layer),
            "first_drop_after_deepstack": (
                self.application_positions[0].index > self.last_deepstack_injection_layer
            ),
            "keep_ratio": self.keep_ratio,
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLPACTActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "pact"
    variant = _PACT_VARIANT
    binding_name = QWEN3_VL_PACT_BINDING
    method_factory = PACTDecoderMethod


class Qwen3VLFastVActivation(_Qwen3VLImageActivation):
    """FastV controller at one pre-layer Qwen3-VL decoder boundary."""

    algorithm = "fastv"
    variant = _FASTV_VARIANT
    binding_name = QWEN3_VL_FASTV_BINDING
    target_site = "llm_decoder_pre_prune_layer_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL FastV requires language_model.layers."
            )
        self.num_layers = len(layers)
        self.decoder_selection = FastVDecoderMethod(stage=self.stage, config=self.config, rules=self._rules, backbone="qwen3_vl")
        self.decoder_selection.resolve_depth(self.num_layers)
        self.spec = self.decoder_selection.spec
        self._fastv_rules = self._rules
        self.application_position = self.decoder_selection.positions[0]
        self.target_site = self.decoder_selection.target_site
        self.last_deepstack_injection_layer = (
            len(tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1
        )

    def _register_semantic_hooks(self) -> Sequence[Any]:
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "requires_decoder_loop_control": True,
            "prune_layer": self.spec.prune_layer,
            "score_layer": self.spec.prune_layer - 1 if self.spec.prune_layer else None,
            "application_position": self.application_position.as_dict(),
            "prune_ratio": self.spec.prune_ratio,
            "score_source": "custom" if "scorer" in self._fastv_rules else self.spec.score_source,
            **({"rules": self._rule_specs, "rules_changed": True,
                "canonical_evidence_applicable": False, "development_runtime": True} if self._fastv_rules else {}),
            "cache_policy": "compact_prefix",
            "remaining_deepstack_synchronized": True,
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLSGLActivation(_Qwen3VLImageActivation):
    """Guide-scored SGL/SGP controller at one target decoder boundary."""

    algorithm = "sgl"
    variant = _SGL_VARIANT
    binding_name = QWEN3_VL_SGL_BINDING
    target_site = "llm_decoder_pre_prune_layer_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.execution_phase is not ExecutionPhase.GENERATE:
            raise BackendIntegrationError(
                "Qwen3-VL SGL supports image generation only."
            )
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL SGL requires language_model.layers."
            )
        self.num_layers = len(layers)
        self.spec = _sgl_spec(self.config)
        self.last_deepstack_injection_layer = (
            len(tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1
        )
        self.guide = SGLGuideController(
            rules=self._rules, backbone_name="qwen3_vl",
            target_runtime=self.runtime,
            spec=self.spec,
        )
        self.decoder_selection = SGLDecoderMethod(
            stage=self.stage,
            spec=self.spec,
            rules=self._rules,
            consume_scores=self.guide.consume_scores,
            num_layers=self.num_layers,
        )
        self.application_position = self.decoder_selection.positions[0]
        self.target_site = self.decoder_selection.target_site

    def install(self) -> None:
        if self.installed:
            return
        super().install()
        try:
            self.guide.attach()
        except BaseException:
            super().uninstall()
            raise

    def uninstall(self) -> None:
        self.guide.detach()
        super().uninstall()

    def _register_semantic_hooks(self):
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            **self.guide.runtime_metadata(),
            "requires_decoder_loop_control": True,
            "prune_layer": self.application_position.index,
            "application_position": self.application_position.as_dict(),
            "prune_layer_fraction": self.spec.prune_layer_fraction,
            "keep_ratio": self.spec.keep_ratio,
            "mode": self.spec.mode,
            "cache_policy": "compact_prefix",
            "remaining_deepstack_synchronized": True,
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLFEATHERActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "feather"
    variant = _FEATHER_VARIANT
    binding_name = QWEN3_VL_FEATHER_BINDING
    method_factory = FEATHERDecoderMethod


class Qwen3VLIVCPruneActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "ivc_prune"
    variant = _IVC_PRUNE_VARIANT
    binding_name = QWEN3_VL_IVC_PRUNE_BINDING
    method_factory = IVCPruneDecoderMethod


class Qwen3VLDARTActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "dart"
    variant = _DART_VARIANT
    binding_name = QWEN3_VL_DART_BINDING
    method_factory = DARTDecoderMethod


class Qwen3VLD2PrunerActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "d2pruner"
    variant = _D2PRUNER_VARIANT
    binding_name = QWEN3_VL_D2PRUNER_BINDING
    method_factory = D2PrunerDecoderMethod


class Qwen3VLSparseVLMActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "sparsevlm"
    variant = _SPARSEVLM_VARIANT
    binding_name = QWEN3_VL_SPARSEVLM_BINDING
    method_factory = SparseVLMDecoderMethod


class Qwen3VLFinePrunerActivation(Qwen3VLRegisteredDecoderSelectionActivation):
    algorithm = "finepruner"
    variant = _FINEPRUNER_VARIANT
    binding_name = QWEN3_VL_FINEPRUNER_BINDING
    method_factory = FinePrunerDecoderMethod


class Qwen3VLBalancedTokenPruningActivation(_Qwen3VLImageActivation):
    """Released BTP schedule on the shared post-layer decoder lifecycle."""

    algorithm = "btp"
    variant = _BTP_VARIANT
    binding_name = QWEN3_VL_BTP_BINDING
    target_site = "llm_decoder_layer_output_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL BTP requires language_model.layers."
            )
        try:
            self.decoder_selection = BTPDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="qwen3_vl")
            resolve_decoder_selection_depth(self.decoder_selection, len(layers))
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid BTP config: {exc}") from exc
        self.last_deepstack_injection_layer = (
            len(tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1
        )

    @property
    def schedule(self) -> tuple[int, ...]:
        return self.spec.boundary_after_layer_indices

    def _register_semantic_hooks(self) -> Sequence[Any]:
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "requires_decoder_loop_control": True,
            "schedule": list(self.schedule),
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.schedule[:-1]),
            "cache_policy": "compact_prefix",
            "position_policy": "preserve_original",
            "single_image_only": True,
            "final_stage": "remove_remaining_visual_span",
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLHalfVActivation(_Qwen3VLImageActivation):
    """HalfV IVR/SSR schedule on the shared compact-prefix lifecycle."""

    algorithm = "halfv"
    variant = _HALFV_VARIANT
    binding_name = QWEN3_VL_HALFV_BINDING
    target_site = "llm_decoder_layer_output_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL HalfV requires language_model.layers."
            )
        try:
            self.decoder_selection = HalfVDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="qwen3_vl")
            resolve_decoder_selection_depth(self.decoder_selection, len(layers))
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid HalfV config: {exc}") from exc
        self.last_deepstack_injection_layer = (
            len(tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1
        )

    @property
    def schedule(self) -> tuple[int, ...]:
        return self.spec.boundary_after_layer_indices

    def _register_semantic_hooks(self) -> Sequence[Any]:
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "requires_decoder_loop_control": True,
            "schedule": list(self.schedule),
            "score_layers": list(self.schedule),
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_rope_by_stage": {"ivr": False, "ssr": True},
            "cache_policy": "compact_prefix",
            "position_policy": "preserve_original",
            "single_image_only": True,
            "generate_requires_kv_cache_repair": True,
        }


class Qwen3VLTransPruneActivation(_Qwen3VLImageActivation):
    """TransPrune progressive TTV/IGA selection on layer-local caches."""

    algorithm = "transprune"
    variant = _TRANSPRUNE_VARIANT
    binding_name = QWEN3_VL_TRANSPRUNE_BINDING
    target_site = "llm_decoder_layer_output_image_tokens"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.visual, self.language_model = _qwen3_components(self.model)
        layers = tuple(getattr(self.language_model, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError(
                "Qwen3-VL TransPrune requires language_model.layers."
            )
        try:
            self.decoder_selection = TransPruneDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="qwen3_vl")
            resolve_decoder_selection_depth(self.decoder_selection, len(layers))
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid TransPrune config: {exc}") from exc
        self.last_deepstack_injection_layer = (
            len(tuple(getattr(self.visual, "deepstack_merger_list", ()) or ())) - 1
        )

    @property
    def schedule(self) -> tuple[int, ...]:
        return self.spec.boundary_after_layer_indices

    @property
    def accumulating_layers(self) -> tuple[int, ...]:
        return self.spec.accumulating_layer_indices

    def _register_semantic_hooks(self) -> Sequence[Any]:
        return self._register_decoder_selection()

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "requires_decoder_loop_control": True,
            "schedule": list(self.schedule),
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.decoder_selection.score_layers),
            "cache_policy": "layer_local",
            "position_policy": "contiguous",
            "single_image_only": True,
            "iga_source": "next_layer_post_image_text",
            "generate_requires_kv_cache_repair": True,
        }


def create_qwen3_vl_tome_executor(**kwargs: Any) -> Qwen3VLToMeActivation:
    """Create the Qwen3-VL encoder-dense-boundary ToMe contract."""

    return Qwen3VLToMeActivation(**kwargs)


def create_qwen3_vl_visionzip_executor(
    **kwargs: Any,
) -> Qwen3VLVisionZipActivation:
    """Create the synchronized main/DeepStack VisionZip contract."""

    return Qwen3VLVisionZipActivation(**kwargs)


def create_qwen3_vl_prumerge_executor(
    **kwargs: Any,
) -> Qwen3VLPruMergeActivation:
    return Qwen3VLPruMergeActivation(**kwargs)


def create_qwen3_vl_gap_executor(**kwargs: Any) -> Qwen3VLGAPActivation:
    return Qwen3VLGAPActivation(**kwargs)


def create_qwen3_vl_divprune_executor(
    **kwargs: Any,
) -> Qwen3VLDivPruneActivation:
    """Create the synchronized layer-0 DivPrune contract."""

    return Qwen3VLDivPruneActivation(**kwargs)


def create_qwen3_vl_saint_executor(**kwargs: Any) -> Qwen3VLSAINTActivation:
    """Create synchronized post-merger SAINT selection."""

    return Qwen3VLSAINTActivation(**kwargs)


def create_qwen3_vl_aim_executor(**kwargs: Any) -> Qwen3VLAIMActivation:
    """Create synchronized post-merger AIM merging."""

    return Qwen3VLAIMActivation(**kwargs)


def create_qwen3_vl_mmtok_executor(
    **kwargs: Any,
) -> Qwen3VLMMTokActivation:
    """Create synchronized pre-LLM MMTok coverage selection."""

    return Qwen3VLMMTokActivation(**kwargs)


def create_qwen3_vl_vispruner_executor(
    **kwargs: Any,
) -> Qwen3VLVisPrunerActivation:
    """Create synchronized pre-LLM VisPruner visual-cue selection."""

    return Qwen3VLVisPrunerActivation(**kwargs)


def create_qwen3_vl_zoo_prune_executor(
    **kwargs: Any,
) -> Qwen3VLZOOPruneActivation:
    """Create synchronized pre-LLM ZOO-Prune projector selection."""

    return Qwen3VLZOOPruneActivation(**kwargs)


def create_qwen3_vl_nuwa_executor(
    **kwargs: Any,
) -> Qwen3VLNuwaActivation:
    """Create the composite visual/decoder Nüwa contract."""

    return Qwen3VLNuwaActivation(**kwargs)


def create_qwen3_vl_restore_executor(
    **kwargs: Any,
) -> Qwen3VLRestoreActivation:
    """Create RESTORE's composite visual/decoder calibration contract."""

    return Qwen3VLRestoreActivation(**kwargs)


def create_qwen3_vl_pyramid_drop_executor(
    **kwargs: Any,
) -> Qwen3VLPyramidDropActivation:
    """Create the decoder-loop PyramidDrop contract."""

    return Qwen3VLPyramidDropActivation(**kwargs)


def create_qwen3_vl_pact_executor(**kwargs: Any) -> Qwen3VLPACTActivation:
    """Create the decoder-loop PACT contract."""

    return Qwen3VLPACTActivation(**kwargs)


def create_qwen3_vl_fastv_executor(
    **kwargs: Any,
) -> Qwen3VLFastVActivation:
    """Create the single-boundary decoder FastV contract."""

    return Qwen3VLFastVActivation(**kwargs)


def create_qwen3_vl_feather_executor(
    **kwargs: Any,
) -> Qwen3VLFEATHERActivation:
    return Qwen3VLFEATHERActivation(**kwargs)


def create_qwen3_vl_ivc_prune_executor(
    **kwargs: Any,
) -> Qwen3VLIVCPruneActivation:
    return Qwen3VLIVCPruneActivation(**kwargs)


def create_qwen3_vl_sgl_executor(
    **kwargs: Any,
) -> Qwen3VLSGLActivation:
    """Create the dual-model SGL/SGP decoder contract."""

    return Qwen3VLSGLActivation(**kwargs)


def create_qwen3_vl_dart_executor(
    **kwargs: Any,
) -> Qwen3VLDARTActivation:
    """Create the single-boundary decoder DART contract."""

    return Qwen3VLDARTActivation(**kwargs)


def create_qwen3_vl_d2pruner_executor(
    **kwargs: Any,
) -> Qwen3VLD2PrunerActivation:
    """Create the single-boundary decoder D²Pruner contract."""

    return Qwen3VLD2PrunerActivation(**kwargs)


def create_qwen3_vl_sparsevlm_executor(
    **kwargs: Any,
) -> Qwen3VLSparseVLMActivation:
    """Create the multistage post-layer SparseVLM contract."""

    return Qwen3VLSparseVLMActivation(**kwargs)


def create_qwen3_vl_finepruner_executor(
    **kwargs: Any,
) -> Qwen3VLFinePrunerActivation:
    """Create the multistage layer-local FinePruner contract."""

    return Qwen3VLFinePrunerActivation(**kwargs)


def create_qwen3_vl_btp_executor(
    **kwargs: Any,
) -> Qwen3VLBalancedTokenPruningActivation:
    """Create the released multistage BTP decoder contract."""

    return Qwen3VLBalancedTokenPruningActivation(**kwargs)


def create_qwen3_vl_halfv_executor(
    **kwargs: Any,
) -> Qwen3VLHalfVActivation:
    """Create the released two-stage HalfV decoder contract."""

    return Qwen3VLHalfVActivation(**kwargs)


def create_qwen3_vl_transprune_executor(
    **kwargs: Any,
) -> Qwen3VLTransPruneActivation:
    """Create the progressive TransPrune decoder contract."""

    return Qwen3VLTransPruneActivation(**kwargs)


class Qwen3VLImageCompressionAdapter:
    """Resolve one exact Qwen3-VL image binding without installing it."""

    def resolve(
        self,
        *,
        model: Any,
        runtime: Any = None,
        facade: Any = None,
        stage: Any = None,
        binding: Any,
        binding_config: Any = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
        contract_projection: Any = None,
    ) -> _Qwen3VLImageActivation:
        if contract_projection is not None and binding.strategy_factory is not None:
            return Qwen3VLRegisteredDecoderSelectionActivation(model=model, runtime=runtime,
                facade=facade, stage=stage, binding=binding, binding_config=binding_config,
                execution_phase=execution_phase, contract_projection=contract_projection)
        binding_name = getattr(binding, "name", None)
        try:
            contract = QWEN3_VL_IMAGE_BINDING_CONTRACTS[binding_name]
        except (KeyError, TypeError) as exc:
            raise BackendIntegrationError(
                f"Qwen3-VL has no ImageLLM executor for binding {binding_name!r}."
            ) from exc
        factories: dict[str, Callable[..., _Qwen3VLImageActivation]] = {
            _TOME_VARIANT: create_qwen3_vl_tome_executor,
            _VISIONZIP_VARIANT: create_qwen3_vl_visionzip_executor,
            _PRUMERGE_VARIANT: create_qwen3_vl_prumerge_executor,
            _GAP_VARIANT: create_qwen3_vl_gap_executor,
            _DIVPRUNE_VARIANT: create_qwen3_vl_divprune_executor,
            _SAINT_VARIANT: create_qwen3_vl_saint_executor,
            _AIM_VARIANT: create_qwen3_vl_aim_executor,
            _MMTOK_VARIANT: create_qwen3_vl_mmtok_executor,
            _VISPRUNER_VARIANT: create_qwen3_vl_vispruner_executor,
            _ZOO_PRUNE_VARIANT: create_qwen3_vl_zoo_prune_executor,
            _NUWA_VARIANT: create_qwen3_vl_nuwa_executor,
            _RESTORE_VARIANT: create_qwen3_vl_restore_executor,
            _PYRAMID_DROP_VARIANT: create_qwen3_vl_pyramid_drop_executor,
            _PACT_VARIANT: create_qwen3_vl_pact_executor,
            _FASTV_VARIANT: create_qwen3_vl_fastv_executor,
            _FEATHER_VARIANT: create_qwen3_vl_feather_executor,
            _IVC_PRUNE_VARIANT: create_qwen3_vl_ivc_prune_executor,
            _SGL_VARIANT: create_qwen3_vl_sgl_executor,
            _DART_VARIANT: create_qwen3_vl_dart_executor,
            _D2PRUNER_VARIANT: create_qwen3_vl_d2pruner_executor,
            _SPARSEVLM_VARIANT: create_qwen3_vl_sparsevlm_executor,
            _FINEPRUNER_VARIANT: create_qwen3_vl_finepruner_executor,
            _BTP_VARIANT: create_qwen3_vl_btp_executor,
            _HALFV_VARIANT: create_qwen3_vl_halfv_executor,
            _TRANSPRUNE_VARIANT: create_qwen3_vl_transprune_executor,
        }
        variant = getattr(binding, "variant", None)
        if variant != contract.variant:
            raise BackendIntegrationError(
                f"Qwen3-VL binding {binding_name!r} has variant {variant!r}, "
                f"expected {contract.variant!r}."
            )
        factory = factories[contract.variant]
        return factory(
            model=model,
            runtime=runtime,
            facade=facade,
            stage=stage,
            binding=binding,
            binding_config=binding_config,
            execution_phase=execution_phase,
        )


def create_qwen3_vl_image_compression_executor(
    *,
    model: Any,
    runtime: Any = None,
    facade: Any = None,
    stage: Any = None,
    binding: Any,
    binding_config: Any = None,
    execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
    contract_projection: Any = None,
) -> _Qwen3VLImageActivation:
    """Dispatch a Qwen3-VL ImageLLM binding to its fail-closed executor."""

    return Qwen3VLImageCompressionAdapter().resolve(
        model=model,
        runtime=runtime,
        facade=facade,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
        contract_projection=contract_projection,
    )


def _qwen3_components(model: Any) -> tuple[Any, Any]:
    core = getattr(model, "model", model)
    visual = getattr(core, "visual", None)
    language_model = getattr(core, "language_model", None)
    if visual is None or language_model is None:
        raise BackendIntegrationError(
            "Qwen3-VL compression requires model.model.visual and "
            "model.model.language_model."
        )
    return visual, language_model


def _validate_image_binding(
    binding: Any,
    *,
    expected_variant: str,
    expected_binding: str,
    execution_phase: ExecutionPhase,
) -> None:
    if getattr(binding, "backbone", None) != "qwen3_vl":
        raise BackendIntegrationError(
            "Qwen3-VL compression adapter received a binding for another backbone."
        )
    if getattr(binding, "name", None) != expected_binding:
        raise BackendIntegrationError(
            f"Expected Qwen3-VL binding {expected_binding!r}, got "
            f"{getattr(binding, 'name', None)!r}."
        )
    if getattr(binding, "variant", None) != expected_variant:
        raise BackendIntegrationError(
            f"Binding {expected_binding!r} does not implement variant "
            f"{expected_variant!r}."
        )
    target_modalities = frozenset(
        Modality(value) for value in getattr(binding, "target_modalities", ())
    )
    if target_modalities != frozenset({Modality.IMAGE}):
        raise BackendIntegrationError(
            "Qwen3-VL ImageLLM bindings must target image only; video requires "
            "a separate exact support unit."
        )
    phases = frozenset(
        ExecutionPhase(value) for value in getattr(binding, "supported_phases", ())
    )
    if phases and execution_phase not in phases:
        raise BackendIntegrationError(
            f"Binding {expected_binding!r} does not support phase "
            f"{execution_phase.value!r}."
        )


def _require_image_modality(value: Modality | str) -> None:
    modality = Modality(value)
    if modality is not Modality.IMAGE:
        raise BackendIntegrationError(
            "This Qwen3-VL executor is image-only; video compression requires "
            "a separate variant, binding, and support unit."
        )


def _require_image_index(value: Any) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendIntegrationError("image_index must be a non-negative integer.")


def _validate_image_stage(
    stage: Any,
    *,
    expected_algorithm: str,
    expected_variant: str,
) -> None:
    algorithm = getattr(stage, "algorithm", expected_algorithm)
    variant = getattr(stage, "variant", expected_variant)
    if algorithm != expected_algorithm or variant != expected_variant:
        raise BackendIntegrationError(
            "Qwen3-VL stage algorithm/variant does not match its image binding."
        )
    modalities = frozenset(
        Modality(value) for value in getattr(stage, "target_modalities", ()) or ()
    )
    if modalities and modalities != frozenset({Modality.IMAGE}):
        raise BackendIntegrationError(
            "Qwen3-VL ImageLLM stages must target image only; video requires a "
            "separate exact support unit."
        )


def _merged_executor_config(stage: Any, binding_config: Any) -> dict[str, Any]:
    """Flatten stable stage dictionaries with binding-local precedence."""

    result: dict[str, Any] = {}
    if stage is not None:
        for name in ("operation", "criterion", "schedule", "config"):
            value = getattr(stage, name, None)
            if isinstance(value, Mapping):
                result.update(dict(value))
    value = getattr(binding_config, "config", None)
    if isinstance(value, Mapping):
        result.update(dict(value))
    return result


def _divprune_spec(config: Mapping[str, Any]) -> DivPruneSpec:
    keys = {
        "retain_ratio",
        "target_count",
        "prune_layer",
        "distance",
        "selection",
        "min_visual_tokens",
        "distance_chunk_size",
    }
    values = {key: value for key, value in config.items() if key in keys}
    try:
        return DivPruneSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid DivPrune config: {exc}") from exc


def _saint_spec(config: Mapping[str, Any]) -> SAINTSpec:
    fields = {
        "keep_ratio",
        "target_count",
        "prune_level",
        "group_size",
        "similarity_threshold",
        "neighbour_count",
        "gamma",
        "max_iterations",
    }
    values = {name: config[name] for name in fields if name in config}
    if "keep_ratio" not in values:
        ratios = config.get("keep_ratios")
        if isinstance(ratios, Sequence) and not isinstance(ratios, (str, bytes)) and ratios:
            values["keep_ratio"] = ratios[-1]
    values.setdefault("group_size", 1)
    try:
        return SAINTSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid SAINT config: {exc}") from exc


def _aim_spec(config: Mapping[str, Any]) -> AIMSpec:
    fields = {"retain_ratio", "target_count", "min_visual_tokens"}
    values = {name: config[name] for name in fields if name in config}
    try:
        return AIMSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid AIM config: {exc}") from exc


def _mmtok_spec(config: Mapping[str, Any]) -> MMTokSpec:
    try:
        return MMTokSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid MMTok config: {exc}") from exc


def _vispruner_spec(config: Mapping[str, Any]) -> VisPrunerSpec:
    try:
        return VisPrunerSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid VisPruner config: {exc}") from exc


def _zoo_prune_spec(config: Mapping[str, Any]) -> ZOOPruneSpec:
    try:
        return ZOOPruneSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid ZOO-Prune config: {exc}") from exc


def _sgl_spec(config: Mapping[str, Any]) -> SGLSpec:
    try:
        return SGLSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid SGL config: {exc}") from exc


def _integer_config(
    config: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int,
) -> int:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise BackendIntegrationError(f"{name} must be an integer.")
    if value < minimum:
        raise BackendIntegrationError(f"{name} must be at least {minimum}.")
    return value


def _float_config(
    config: Mapping[str, Any],
    name: str,
    *,
    default: float,
    minimum_exclusive: float,
    maximum: float,
) -> float:
    value = config.get(name, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise BackendIntegrationError(f"{name} must be numeric.")
    result = float(value)
    if not minimum_exclusive < result <= maximum:
        raise BackendIntegrationError(
            f"{name} must be in ({minimum_exclusive}, {maximum}]."
        )
    return result


def _validate_pyramid_schedule(
    schedule: PyramidDropSchedule,
) -> None:
    indices = schedule.layer_indices
    if len(indices) != schedule.num_stages - 1:
        raise BackendIntegrationError(
            "PyramidDrop requires num_stages - 1 decoder boundaries."
        )
    if tuple(sorted(set(indices))) != indices:
        raise BackendIntegrationError(
            "PyramidDrop layer_indices must be unique and strictly increasing."
        )
    if any(index < 0 or index >= schedule.num_layers - 1 for index in indices):
        raise BackendIntegrationError(
            "PyramidDrop boundaries must leave at least one later decoder layer."
        )


def _dense_restore_indices(delta: LayoutDelta, *, device: Any) -> Any:
    torch = _torch()
    mapping = tuple(delta.old_to_new[index] for index in range(delta.source_count))
    if any(index is None for index in mapping):
        raise InvariantError("Dense restoration mapping contains dropped tokens.")
    return torch.tensor(mapping, dtype=torch.long, device=device)


def _visionzip_by_layout_delta(
    features: Any,
    delta: LayoutDelta,
    *,
    representatives: tuple[int, ...],
    contextual_indices: tuple[int, ...],
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay VisionZip's released merge values on one DeepStack branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            "Qwen3-VL synchronized DeepStack features must be rank-2 tensors."
        )
    if int(features.shape[0]) != delta.source_count:
        raise InvariantError(
            "Qwen3-VL DeepStack branch cardinality does not match the shared "
            "VisionZip LayoutDelta."
        )
    if len(representatives) != delta.result_count:
        raise InvariantError(
            "Qwen3-VL VisionZip representative count does not match LayoutDelta."
        )
    groups: list[list[int]] = [[] for _ in range(delta.result_count)]
    for source_index, result_index in delta.old_to_new.items():
        if result_index is not None:
            groups[result_index].append(source_index)
    return visionzip_merge_source_groups(
        features,
        tuple(tuple(group) for group in groups),
        representatives=representatives,
        contextual_indices=contextual_indices,
        rules=rules,
    )


def _select_by_layout_delta(features: Any, delta: LayoutDelta) -> Any:
    """Replay a source-ordered drop mapping on one synchronized feature branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            "Qwen3-VL synchronized DeepStack features must be rank-2 tensors."
        )
    if int(features.shape[0]) != delta.source_count:
        raise InvariantError(
            "Qwen3-VL DeepStack branch cardinality does not match the shared "
            "DivPrune LayoutDelta."
        )
    indices = torch.tensor(
        delta.kept_indices,
        dtype=torch.long,
        device=features.device,
    )
    return features.index_select(0, indices)


def _mean_by_layout_delta(features: Any, delta: LayoutDelta) -> Any:
    """Replay a merge mapping on one synchronized DeepStack branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError(
            "Qwen3-VL synchronized DeepStack features must be rank-2 tensors."
        )
    if int(features.shape[0]) != delta.source_count:
        raise InvariantError(
            "Qwen3-VL DeepStack branch cardinality does not match the shared "
            "AIM LayoutDelta."
        )
    groups: list[list[int]] = [[] for _ in range(delta.result_count)]
    for source, result in delta.old_to_new.items():
        if result is not None:
            groups[result].append(source)
    if any(not group for group in groups):
        raise InvariantError("AIM LayoutDelta contains an empty result group.")
    return torch.stack(
        [
            features.index_select(
                0,
                torch.tensor(group, dtype=torch.long, device=features.device),
            ).mean(dim=0)
            for group in groups
        ],
        dim=0,
    )


def _require_removable_handle(handle: Any, *, algorithm: str) -> None:
    if not callable(getattr(handle, "remove", None)):
        raise BackendIntegrationError(
            f"Qwen3-VL {algorithm} semantic hook registration did not return "
            "a removable handle."
        )


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - minimal control-plane env
        raise RuntimeError(
            "Qwen3-VL image compression execution requires PyTorch."
        ) from exc
    return torch


__all__ = [
    "QWEN3_VL_IMAGE_BINDING_CONTRACTS",
    "QWEN3_VL_BTP_BINDING",
    "QWEN3_VL_HALFV_BINDING",
    "QWEN3_VL_TRANSPRUNE_BINDING",
    "QWEN3_VL_PRUMERGE_BINDING",
    "QWEN3_VL_GAP_BINDING",
    "QWEN3_VL_FEATHER_BINDING",
    "QWEN3_VL_IVC_PRUNE_BINDING",
    "QWEN3_VL_D2PRUNER_BINDING",
    "QWEN3_VL_DIVPRUNE_BINDING",
    "QWEN3_VL_SAINT_BINDING",
    "QWEN3_VL_AIM_BINDING",
    "QWEN3_VL_PACT_BINDING",
    "QWEN3_VL_MMTOK_BINDING",
    "QWEN3_VL_VISPRUNER_BINDING",
    "QWEN3_VL_ZOO_PRUNE_BINDING",
    "QWEN3_VL_NUWA_BINDING",
    "QWEN3_VL_RESTORE_BINDING",
    "QWEN3_VL_DART_BINDING",
    "QWEN3_VL_FASTV_BINDING",
    "QWEN3_VL_SGL_BINDING",
    "QWEN3_VL_FINEPRUNER_BINDING",
    "QWEN3_VL_PYRAMID_DROP_BINDING",
    "QWEN3_VL_SPARSEVLM_BINDING",
    "QWEN3_VL_TOME_BINDING",
    "QWEN3_VL_VISIONZIP_BINDING",
    "Qwen3VLPyramidDropActivation",
    "Qwen3VLBalancedTokenPruningActivation",
    "Qwen3VLDivPruneActivation",
    "Qwen3VLDivPruneResult",
    "Qwen3VLSAINTActivation",
    "Qwen3VLSAINTResult",
    "Qwen3VLAIMActivation",
    "Qwen3VLAIMResult",
    "Qwen3VLPACTActivation",
    "Qwen3VLMMTokActivation",
    "Qwen3VLMMTokResult",
    "Qwen3VLVisPrunerActivation",
    "Qwen3VLVisPrunerResult",
    "Qwen3VLZOOPruneActivation",
    "Qwen3VLZOOPruneResult",
    "Qwen3VLNuwaActivation",
    "Qwen3VLNuwaStage1Result",
    "Qwen3VLRestoreActivation",
    "Qwen3VLRestoreResult",
    "Qwen3VLDARTActivation",
    "Qwen3VLD2PrunerActivation",
    "Qwen3VLFastVActivation",
    "Qwen3VLPruMergeActivation",
    "Qwen3VLGAPActivation",
    "Qwen3VLFEATHERActivation",
    "Qwen3VLIVCPruneActivation",
    "Qwen3VLSGLActivation",
    "Qwen3VLFinePrunerActivation",
    "Qwen3VLHalfVActivation",
    "Qwen3VLTransPruneActivation",
    "Qwen3VLSparseVLMActivation",
    "Qwen3VLToMeActivation",
    "Qwen3VLImageBindingContract",
    "Qwen3VLImageCompressionAdapter",
    "Qwen3VLVisionZipActivation",
    "Qwen3VLVisionZipResult",
    "create_qwen3_vl_image_compression_executor",
    "create_qwen3_vl_btp_executor",
    "create_qwen3_vl_halfv_executor",
    "create_qwen3_vl_transprune_executor",
    "create_qwen3_vl_divprune_executor",
    "create_qwen3_vl_saint_executor",
    "create_qwen3_vl_aim_executor",
    "create_qwen3_vl_pact_executor",
    "create_qwen3_vl_mmtok_executor",
    "create_qwen3_vl_vispruner_executor",
    "create_qwen3_vl_zoo_prune_executor",
    "create_qwen3_vl_nuwa_executor",
    "create_qwen3_vl_restore_executor",
    "create_qwen3_vl_dart_executor",
    "create_qwen3_vl_d2pruner_executor",
    "create_qwen3_vl_fastv_executor",
    "create_qwen3_vl_prumerge_executor",
    "create_qwen3_vl_gap_executor",
    "create_qwen3_vl_feather_executor",
    "create_qwen3_vl_ivc_prune_executor",
    "create_qwen3_vl_sgl_executor",
    "create_qwen3_vl_finepruner_executor",
    "create_qwen3_vl_pyramid_drop_executor",
    "create_qwen3_vl_sparsevlm_executor",
    "create_qwen3_vl_tome_executor",
    "create_qwen3_vl_visionzip_executor",
]
