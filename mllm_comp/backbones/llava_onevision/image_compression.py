"""ImageLLM callback contracts for the pinned vendored LLaVA-OneVision route.

The shared kernels remain independent of provider source.  The exact pinned
provider bridge emits these callbacks from guarded encoder, packed-AnyRes, and
decoder-loop boundaries and owns all sequence-layout repair.  A missing or
source-drifted provider still fails closed.

These activations do not themselves claim real-checkpoint or support-lifecycle
evidence; those remain external exact-support records.
"""

from __future__ import annotations

from mllm_comp.compressors.image.positions import (
    tome_legacy_budget_layers,
    tome_encoder_score_layer,
    tome_encoder_operation,
    tome_encoder_positions,
    tome_position_metadata,
    tome_output_site,
)
from mllm_comp.compressors.layer_positions import LayerPosition

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace, field, is_dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from mllm_comp.backbones._decoder_attention import (
    DecoderAttentionImplementationOverride,
)
from mllm_comp.compressors.image.pyramid_drop import (
    PyramidDropSchedule,
)
from mllm_comp.compressors.image.saint import SAINTSpec, saint_image_tokens
from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens
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
    DecoderSelectionRegistration,
    execute_decoder_selection,
)
from mllm_comp.compressors.image.decoder_registration import (
    create_registered_decoder_selection,
    prepare_decoder_selection,
)
from mllm_comp.compressors.rules import (
    load_rule_specs,
    builtin_rule_specs,
    resolve_rule_binding,
)
from mllm_comp.compressors.image.sgl import (
    SGLGuideController,
    SGLSpec,
)
from mllm_comp.compressors.image.btp import (
    BalancedTokenPruningSpec,
)
from mllm_comp.compressors.image.halfv import HalfVSpec
from mllm_comp.compressors.image.decoder_config import (
    merge_decoder_executor_config,
)
from mllm_comp.compressors.image.divprune import DivPruneSpec, divprune_image
from mllm_comp.compressors.image.mmtok import (
    MMTokSpec,
    merge_mmtok_executor_config,
    mmtok_image_tokens,
)
from mllm_comp.compressors.image.vispruner import (
    VisPrunerSpec,
    merge_vispruner_executor_config,
    vispruner_image_tokens,
)
from mllm_comp.compressors.image.zoo_prune import (
    ZOOProjectorProxyCache,
    ZOOPruneSpec,
    merge_zoo_prune_executor_config,
    zoo_prune_image_tokens,
)
from mllm_comp.compressors.image.nuwa import (
    NuwaSpec,
    merge_nuwa_executor_config,
    nuwa_stage1_image_tokens,
)
from mllm_comp.compressors.image.restore import (
    RestoreSpec,
    merge_restore_executor_config,
    restore_image_tokens,
)
from mllm_comp.compressors.image.tome import (
    tome_merge_image,
    tome_preserve_source_order,
)
from mllm_comp.compressors.image.visionzip import visionzip_image
from mllm_comp.compressors.image.prumerge import (
    PruMergeSpec,
    prumerge_image_tokens,
)
from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens
from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.layout import (
    LayoutDelta,
    OrderingPolicy,
    layout_delta_record,
)
from mllm_comp.core.vocabulary import ExecutionPhase, Modality


from mllm_comp.compressors.image.contracts.llava_onevision import (
    LLAVA_ONEVISION_BRIDGE_API_VERSION,
    LLAVA_ONEVISION_SOURCE_COMMIT,
    LLAVA_ONEVISION_SOURCE_ROUTE,
    TOME_MERGE_SITE,
    TOME_RESTORE_SITE,
    TOME_PACK_SITE,
    VISIONZIP_SITE,
    PYRAMID_DROP_SITE,
    RESTORE_CALIBRATION_SITE,
    FASTV_SITE,
    LlavaCallbackContract,
    LlavaImageBindingContract,
    LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS,
)


@runtime_checkable
class LlavaOneVisionCallbackHandle(Protocol):
    def remove(self) -> None:
        """Unregister one callback."""


@runtime_checkable
class LlavaOneVisionCompressionBridge(Protocol):
    """Protocol that the source-tree provider must expose on the root model."""

    api_version: str
    source_route: str
    capabilities: Mapping[str, bool]

    def available_sites(self) -> Sequence[str]:
        """Return callback sites emitted by the provider."""

    def register_callback(
        self,
        site: str,
        callback: Callable[[Any], Any],
        *,
        requires_packed_metrics: bool = False,
        decoder_selection: DecoderSelectionRegistration | None = None,
    ) -> LlavaOneVisionCallbackHandle:
        """Register one callback and return a removable handle."""


@dataclass(frozen=True)
class LlavaEncoderMergeRequest:
    invocation_id: str
    hidden_states: Any
    keys: Any
    layer_index: int
    stream_ids: tuple[str, ...]
    token_sizes: Any | None = None
    protected_indices: tuple[int, ...] = ()
    coordinates: tuple[Any, ...] = ()
    modality: Modality | str = Modality.IMAGE

    position: LayerPosition | None = None


@dataclass(frozen=True)
class LlavaEncoderMergeResponse:
    hidden_states: Any
    token_sizes: Any
    proportional_attention_bias: Any
    layout_deltas: tuple[LayoutDelta, ...]
    source_groups: tuple[tuple[tuple[int, ...], ...], ...]
    stream_ids: tuple[str, ...]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LlavaEncoderDenseRestoreRequest:
    invocation_id: str
    hidden_states: Any
    stream_ids: tuple[str, ...]
    modality: Modality | str = Modality.IMAGE


@dataclass(frozen=True)
class LlavaEncoderDenseRestoreResponse:
    hidden_states: Any
    dense_to_compact: tuple[tuple[int, ...], ...]
    stream_ids: tuple[str, ...]
    source_token_count: int
    compact_token_count: int
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class LlavaPackedImageRequest:
    """One already-packed image sequence immediately before LLM splice."""

    invocation_id: str
    image_index: int
    tokens: Any
    attention: Any
    keys: Any
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    newline_indices: tuple[int, ...]
    anyres_layout: Mapping[str, Any]
    coordinates: tuple[Any, ...] = ()
    cls_index: int | None = None
    projected_crops: Any | None = None
    vision_crops: Any | None = None
    vision_attentions: Any | None = None
    semantic_crops: Any | None = None
    vision_keys: Any | None = None
    text_embeddings: Any | None = None
    patches_per_side: int | None = None
    grid_width: int | None = None
    grid_height: int | None = None
    patch_merge_type: str | None = None
    projector_layers: tuple[Any, ...] | None = None
    vision_attention_layers: Mapping[int, Any] | None = None


@dataclass(frozen=True)
class LlavaPackedImageResponse:
    tokens: Any
    layout_delta: LayoutDelta
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    newline_indices: tuple[int, ...]
    anyres_layout: Mapping[str, Any]
    image_index: int
    metadata: Mapping[str, Any]
    visual_sizes: Any | None = None
    representative_indices: tuple[int, ...] = ()
    original_visual_length: int | None = None


@dataclass(frozen=True)
class LlavaDecoderLayoutRepair:
    """Provider-owned repair result after one decoder token drop."""

    attention_mask: Any
    position_ids: Any
    past_key_values: Any
    image_indices: Any
    newline_indices: tuple[int, ...]
    sequence_length: int
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LlavaDecoderLayerRequest:
    invocation_id: str
    layer_index: int
    num_layers: int
    hidden_states: Any
    image_indices: Any
    instruction_query: Any
    image_keys: Any
    attention_mask: Any
    position_ids: Any
    past_key_values: Any
    use_cache: bool
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    newline_indices: tuple[int, ...]
    repair_layout: Callable[..., LlavaDecoderLayoutRepair]
    attention_keys: Any | None = None
    attention_key_mask: Any | None = None
    image_attention_scores: Any | None = None


@dataclass(frozen=True)
class LlavaDecoderLayerResponse:
    hidden_states: Any
    attention_mask: Any
    position_ids: Any
    past_key_values: Any
    image_indices: Any
    newline_indices: tuple[int, ...]
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    layout_delta: LayoutDelta | None
    layer_index: int
    dropped: bool
    metadata: Mapping[str, Any]
    token_sizes: Any | None = None


@dataclass(frozen=True)
class LlavaPACTRequest:
    invocation_id: str
    layer_index: int
    num_layers: int
    hidden_states: Any
    image_indices: Any
    query_states: Any
    key_states: Any
    clustering_keys: Any
    attention_mask: Any
    position_ids: Any
    past_key_values: Any
    use_cache: bool
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    newline_indices: tuple[int, ...]
    repair_layout: Callable[..., LlavaDecoderLayoutRepair]
    image_attention_scores: Any = None


@dataclass(frozen=True)
class LlavaFastVRequest:
    invocation_id: str
    prune_layer: int
    num_layers: int
    hidden_states: Any
    image_indices: Any
    scorer_query: Any | None
    attention_keys: Any | None
    attention_key_mask: Any | None
    attention_mask: Any
    position_ids: Any
    past_key_values: Any
    use_cache: bool
    image_sizes: tuple[tuple[int, int], ...]
    modalities: tuple[str, ...]
    newline_indices: tuple[int, ...]
    repair_layout: Callable[..., LlavaDecoderLayoutRepair]


@dataclass
class _EncoderInvocationState:
    stream_ids: tuple[str, ...]
    deltas: tuple[LayoutDelta, ...]


class _LlavaBridgeActivation:
    """Common transactional lifecycle for one exact source-tree binding."""

    executor_name = "llava_onevision_image"

    def apply_decoder_selection(self, selection, point, hidden_states, *, sequence,
            attention_mask, position_ids, past_key_values, use_cache, repair_layout):
        method = self.decoder_selection
        result = execute_decoder_selection(selection, point, hidden_states, image_indices=sequence.image_indices.to(hidden_states.device),
            protected_indices=sequence.newline_indices, media_unit="llm_token",
            source_layout={"backbone": "llava_onevision", "application_position": point.as_dict(),
                "invocation_id": sequence.invocation_id},
            result_layout={"backbone": "llava_onevision", "application_position": point.as_dict(),
                "target_site": method.target_site, "position_policy": self.decoder_position_policy,
                "invocation_id": sequence.invocation_id})
        self._callback_counts[self.contract.callbacks[0].site] += 1
        return self._finish_decoder_selection(result, point, sequence=sequence,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, repair_layout=repair_layout)

    def _repair_decoder_selection(self, result, *, image_indices, newline_indices, attention_mask,
            position_ids, past_key_values, use_cache, layer_index, repair_layout):
        """Apply and validate the native repair while retaining only surviving newlines."""
        method = self.decoder_selection
        from mllm_comp.compressors.image.decoder_lifecycle import layout_result_representatives
        import torch
        representatives = torch.tensor(layout_result_representatives(result.layout_delta),
            dtype=torch.long, device=result.hidden_states.device)
        image_representatives = representatives[torch.isin(representatives, image_indices.to(representatives.device))]
        # BTP/HalfV/TransPrune can prune the serialized image span's newlines.
        surviving_newlines = tuple(index for index in newline_indices
            if result.layout_delta.old_to_new[index] is not None)
        repair = repair_layout(kept_indices=representatives, dropped_indices=result.layout_delta.dropped_indices,
            layout_delta=result.layout_delta, hidden_states=result.hidden_states,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, layer_index=layer_index, image_indices=image_representatives,
            newline_indices=surviving_newlines, position_policy=self.decoder_position_policy,
            cache_policy=method.cache_policy.value)
        if not isinstance(repair, LlavaDecoderLayoutRepair):
            raise BackendIntegrationError("Decoder selection repair must return LlavaDecoderLayoutRepair.")
        _validate_decoder_repair(repair, result.layout_delta,
            source_newline_indices=surviving_newlines, kept_source_image_indices=image_representatives)
        if use_cache and self.execution_phase is ExecutionPhase.GENERATE:
            if (repair.past_key_values is not past_key_values or
                    repair.metadata.get("cache_policy") != method.cache_policy.value or
                    repair.metadata.get("cache_validation_owner") != "shared_decoder_lifecycle"):
                raise BackendIntegrationError("Decoder selection must preserve the provider-owned cache and shared lifecycle repair.")
        return repair

    def _finish_decoder_selection(self, result, point, *, sequence, attention_mask,
            position_ids, past_key_values, use_cache, repair_layout):
        repair = self._repair_decoder_selection(result, image_indices=sequence.image_indices, newline_indices=sequence.newline_indices,
            attention_mask=attention_mask, position_ids=position_ids, past_key_values=past_key_values,
            use_cache=use_cache, layer_index=point.index, repair_layout=repair_layout)
        self._last_layout = {"invocation_id": sequence.invocation_id, "layer_index": point.index,
            "source_count": result.layout_delta.source_count, "result_count": result.layout_delta.result_count,
            "newline_indices_before": list(sequence.newline_indices), "newline_indices_after": list(repair.newline_indices),
            "mask_position_cache_repaired": True, "use_cache": use_cache}
        self._latest_layout_deltas = [_serialize_layout_delta(result.layout_delta)]
        self._record_layout_delta(result.layout_delta, site=self.decoder_selection.target_site,
            layer_index=point.index, application_position=point.as_dict())
        return LlavaDecoderLayerResponse(hidden_states=result.hidden_states,
            attention_mask=repair.attention_mask, position_ids=repair.position_ids, past_key_values=repair.past_key_values,
            image_indices=repair.image_indices, newline_indices=repair.newline_indices,
            image_sizes=sequence.image_sizes, modalities=sequence.modalities, layout_delta=result.layout_delta,
            layer_index=point.index, dropped=True,
            metadata={**dict(result.metadata), **dict(repair.metadata), "decoder_loop_control": True,
                "mask_position_cache_repaired": True, "image_sizes_preserved": True, "modalities_preserved": True})

    def __init__(
        self,
        *,
        model: Any,
        runtime: Any | None = None,
        stage: Any,
        binding: Any,
        binding_config: Any | None,
        execution_phase: ExecutionPhase | str,
        contract_projection: Any = None,
    ) -> None:
        registered = contract_projection is not None and binding.strategy_factory is not None
        if not registered and getattr(binding, "position_resolver", None) is not None:
            try:
                binding = binding.position_resolver(stage=stage, binding=binding, binding_config=binding_config)
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid image compression config: {exc}") from exc
        if registered:
            (self.decoder_selection, registered_config, self._rules, self._rule_specs) = create_registered_decoder_selection(
                stage=stage, binding=binding, contract_projection=contract_projection,
                backbone="llava_onevision", execution_phase=execution_phase, binding_config=binding_config)
            self.decoder_position_policy = self.decoder_selection.position_policy
        else:
            try:
                self._rule_specs = builtin_rule_specs(stage)
                self._rules = load_rule_specs(self._rule_specs)
            except ConfigError as exc:
                raise BackendIntegrationError(f"Invalid image compression rules: {exc}") from exc
        if is_dataclass(binding):
            binding = resolve_rule_binding(binding=binding, rules=self._rule_specs)
        self.model = model
        self.runtime = runtime
        self.stage = stage
        self.binding = binding
        self.binding_config = binding_config
        self.execution_phase = ExecutionPhase(execution_phase)
        self.config = registered_config if registered else _merged_config(stage, binding_config)
        if registered:
            site = FASTV_SITE if self.decoder_selection.target_site == FASTV_SITE else PYRAMID_DROP_SITE
            self.contract = LlavaImageBindingContract(binding=binding.name, variant=contract_projection.variant.name,
                callbacks=(LlavaCallbackContract(site=site,
                    request_type="DecoderSelectionRegistration", response_type="LlavaDecoderLayerResponse",
                    source_symbol="native decoder layer loop", timing="registered layer positions",
                    required_capabilities=frozenset({"decoder_loop_control", "sequence_layout_repair"}),
                    required_repairs=("protect image_newline", "repair native mask, position and cache")),))
        else:
            self.contract = _validate_exact_binding(binding, stage, self.execution_phase)
        if self.contract.variant in {"visionzip.image", "prumerge.image", "gap.image", "divprune.image", "aim.image", "vispruner.image", "saint.image.packed_feature_prune", "mmtok.image", "zoo_prune.image", "nuwa.image", "restore.image"}:
            from mllm_comp.compressors.image.positions import image_projector_sites
            self.projector_target_sites = image_projector_sites(stage, self.contract.variant.split(".")[0], "llava_onevision")
        self._bridge: Any | None = None
        self._handles: list[Any] = []
        self._owns_provider_bridge = False
        self._installed = False
        self._callback_counts = {site: 0 for site in self.contract.sites}
        self._last_layout: dict[str, Any] = {}
        self._latest_layout_deltas: list[dict[str, Any]] = []
        self._layout_delta_records: list[dict[str, Any]] = []
        self._layout_record_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        prepare_decoder_selection(getattr(self, "decoder_selection", None), self.runtime)
        bridge, owns_provider_bridge = _resolve_or_install_bridge(self.model)
        _validate_bridge(bridge, self.contract, self.execution_phase)
        handles: list[Any] = []
        try:
            for callback in self.contract.callbacks:
                handler = self._handler_for_site(callback.site)
                method = (
                    getattr(self, "decoder_selection", None)
                    if callback.site in {FASTV_SITE, PYRAMID_DROP_SITE}
                    else None
                )
                selection_registration = ({} if method is None else {"decoder_selection":
                    DecoderSelectionRegistration(method, self.apply_decoder_selection, self.execution_phase, id(self))})
                handle = bridge.register_callback(
                    callback.site,
                    self._counting_handler(callback.site, handler),
                    requires_packed_metrics=callback.requires_packed_metrics,
                    **selection_registration,
                )
                if not callable(getattr(handle, "remove", None)):
                    raise BackendIntegrationError(
                        f"LLaVA bridge callback {callback.site!r} returned a "
                        "non-removable handle."
                    )
                handles.append(handle)
        except BaseException:
            for handle in reversed(handles):
                try:
                    handle.remove()
                except Exception:
                    pass
            if owns_provider_bridge:
                try:
                    bridge.detach()
                except Exception:
                    pass
            raise
        self._bridge = bridge
        self._handles = handles
        self._owns_provider_bridge = owns_provider_bridge
        self._installed = True

    def uninstall(self) -> None:
        errors: list[str] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        bridge = self._bridge
        owns_provider_bridge = self._owns_provider_bridge
        self._handles.clear()
        self._bridge = None
        self._owns_provider_bridge = False
        self._installed = False
        if owns_provider_bridge and bridge is not None:
            try:
                bridge.detach()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        if errors:
            raise BackendIntegrationError(
                f"LLaVA compression callback removal failed: {errors}."
            )

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            "executor": self.executor_name,
            "backbone": "llava_onevision",
            **({"rules": self._rule_specs, "rules_changed": True, "canonical_evidence_applicable": False, "development_runtime": True} if self._rules else {}),
            "binding": self.contract.binding,
            "variant": self.contract.variant,
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "source_route": LLAVA_ONEVISION_SOURCE_ROUTE,
            "bridge_api_version": LLAVA_ONEVISION_BRIDGE_API_VERSION,
            "callback_sites": list(self.contract.sites),
            **({"projector_target_sites": list(self.projector_target_sites),
                "target_sites": list(self.projector_target_sites),
                "target_site": self.projector_target_sites[0],
                "dense_projector_metric_source": self.projector_target_sites[0] == "pre_vision_projector" and self.contract.variant in {"visionzip.image", "prumerge.image", "gap.image", "saint.image.packed_feature_prune", "mmtok.image", "zoo_prune.image"},
                "dense_projector_packing_reference": self.projector_target_sites[0] == "pre_vision_projector"}
                if hasattr(self, "projector_target_sites") else {}),
            "callback_counts": dict(self._callback_counts),
            "layout_record_count": self._layout_record_count,
            "last_layout": dict(self._last_layout),
            "latest_layout_deltas": list(self._latest_layout_deltas),
            "layout_delta_records": list(self._layout_delta_records),
            "image_only": True,
            "evidence_scope": "source_bridge_cpu_contract",
            "real_checkpoint_verified": False,
            "support_activation_claimed": False,
        }

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
                "event_index": self._layout_record_count,
                "site": site,
                **metadata,
            }
        )
        self._layout_delta_records.append(record)
        self._layout_record_count += 1

    def _counting_handler(
        self,
        site: str,
        handler: Callable[[Any], Any],
    ) -> Callable[[Any], Any]:
        def counted(request: Any) -> Any:
            self._callback_counts[site] += 1
            return handler(request)

        counted._mllm_rules = self._rules
        if (
            site in {FASTV_SITE, PYRAMID_DROP_SITE}
            and getattr(self, "decoder_selection", None) is not None
        ):
            return counted
        if hasattr(self, "projector_target_sites"):
            counted._mllm_projector_target_sites = self.projector_target_sites
        counted._mllm_position_stage = self.stage  # type: ignore[attr-defined]
        counted._mllm_position_config = self.config  # type: ignore[attr-defined]
        counted._mllm_execution_phase = self.execution_phase  # type: ignore[attr-defined]
        counted._mllm_algorithm = self.contract.binding.partition(".")[0]  # type: ignore[attr-defined]
        counted._mllm_activation_id = id(self)  # type: ignore[attr-defined]
        spec = getattr(self, "spec", None)
        if isinstance(spec, SGLSpec):
            counted._mllm_application_position = self.application_position
            counted._mllm_prune_layer = self.application_position.index  # type: ignore[attr-defined]
        elif isinstance(spec, NuwaSpec):
            counted._mllm_application_position = self.application_position
            counted._mllm_score_layers = (spec.stage2_score_layer,) if spec.stage2_score_layer is not None else ()
            counted._mllm_prune_layer = self.application_position.index  # type: ignore[attr-defined]
            counted._mllm_nuwa_spec = spec  # type: ignore[attr-defined]
        elif isinstance(spec, RestoreSpec):
            counted._mllm_restore_spec = spec  # type: ignore[attr-defined]
        elif isinstance(spec, BalancedTokenPruningSpec):
            counted._mllm_application_positions = self.application_positions
            counted._mllm_prune_layers = (  # type: ignore[attr-defined]
                spec.boundary_after_layer_indices
            )
            counted._mllm_score_layers = (  # type: ignore[attr-defined]
                spec.boundary_after_layer_indices[:-1]
            )
        elif isinstance(spec, HalfVSpec):
            counted._mllm_application_positions = self.application_positions
            counted._mllm_prune_layers = (  # type: ignore[attr-defined]
                spec.boundary_after_layer_indices
            )
            counted._mllm_score_layers = (  # type: ignore[attr-defined]
                spec.boundary_after_layer_indices
            )
        return counted

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        raise NotImplementedError


class LlavaRegisteredDecoderSelectionActivation(_LlavaBridgeActivation):
    """Use an explicit method while the native bridge owns AnyRes and cache repair."""

    executor_name = "llava_onevision_registered_decoder_selection"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if getattr(self, "decoder_selection", None) is None:
            algorithm = self.contract.variant.split(".")[0]
            try:
                self.config = merge_decoder_executor_config(algorithm,
                    stage=self.stage, binding_config=self.binding_config)
                self.decoder_selection = self.method_factory(stage=self.stage, config=self.config,
                    rules=self._rules, backbone="llava_onevision")
            except (TypeError, ValueError) as exc:
                raise BackendIntegrationError(f"Invalid {algorithm} config: {exc}") from exc
        self.decoder_position_policy = self.decoder_selection.position_policy

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

    def _handler_for_site(self, site):
        if site not in self.contract.sites:
            raise BackendIntegrationError(f"Unexpected registered image decoder site {site!r}.")
        return self.apply_decoder_selection

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "algorithm": self.decoder_selection.algorithm,
            "target_site": self.decoder_selection.target_site,
            "application_positions": [point.as_dict() for point in self.decoder_selection.positions],
            "score_layers": list(self.decoder_selection.score_layers),
            "cache_policy": self.decoder_selection.cache_policy.value,
            "position_policy": self.decoder_selection.position_policy,
            **({"decoder_attention_override": self._decoder_attention_override.metadata()}
               if self._decoder_attention_override is not None else {})}


class LlavaOneVisionToMeActivation(_LlavaBridgeActivation):
    """Encoder ToMe with a mandatory dense restore before AnyRes packing."""

    executor_name = "llava_onevision_tome_encoder_dense_boundary"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projector_target_sites = (tome_output_site(self.stage, "llava_onevision"),)
        operation = tome_encoder_operation(self.stage, "llava_onevision")
        self._rule_layer_by_application = {}
        if "layers" in operation:
            legacy_layers = tome_legacy_budget_layers(self.config)
            if legacy_layers is not None:
                positions = tome_encoder_positions(self.stage, "llava_onevision", None)
                if len(positions) != len(legacy_layers):
                    raise ValueError("ToMe application layers must match the legacy per-layer budget count.")
                self._rule_layer_by_application = {
                    p: int(source)
                    for p, source in zip(positions, legacy_layers)
                }
        self._invocations: dict[str, _EncoderInvocationState] = {}

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            **super().runtime_metadata(), **tome_position_metadata(self.stage, "llava_onevision"),
            **({"canonical_evidence_applicable": False} if self._rules else {}),
            "target_sites": [tome_encoder_operation(self.stage, "llava_onevision")["target_site"], TOME_RESTORE_SITE, self.projector_target_sites[0]],
            "dense_projector_metric_source": self.projector_target_sites[0] == "pre_vision_projector",
            "permanent_output_metric_source": "dense_projected_anyres_decoder_layer_0_qk",
            "permanent_output_value_space": "vision" if self.projector_target_sites[0] == "pre_vision_projector" else "projected",
        }

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site == TOME_MERGE_SITE:
            return self.merge_encoder_tokens
        if site == TOME_RESTORE_SITE:
            return self.restore_dense_encoder_tokens
        if site == TOME_PACK_SITE:
            return self.compress_packed_image
        raise BackendIntegrationError(f"Unexpected ToMe callback site {site!r}.")

    def merge_encoder_tokens(
        self,
        request: LlavaEncoderMergeRequest,
    ) -> LlavaEncoderMergeResponse:
        _require_request(request, LlavaEncoderMergeRequest, TOME_MERGE_SITE)
        _require_image(request.modality)
        _require_invocation_id(request.invocation_id)
        if isinstance(request.layer_index, bool) or not isinstance(
            request.layer_index, int
        ):
            raise BackendIntegrationError("ToMe layer_index must be an integer.")
        stream_count = _sequence_batch_size(request.hidden_states)
        if (
            len(request.stream_ids) != stream_count
            or len(set(request.stream_ids)) != stream_count
        ):
            raise BackendIntegrationError(
                "ToMe stream_ids must uniquely identify every crop sequence."
            )
        position = request.position or LayerPosition(
            "inter_layer" if tome_encoder_operation(self.stage, "llava_onevision")["target_site"].endswith("_output") else "post_attention_pre_ffn",
            request.layer_index + tome_encoder_operation(self.stage, "llava_onevision")["target_site"].endswith("_output"),
        )
        r = _tome_r_for_layer(self.config, self._rule_layer_by_application.get(position, request.layer_index))
        source = tome_encoder_score_layer(self.stage, "llava_onevision")
        if r and source is not None and source > position.index:
            raise ValueError("ToMe score_layer is a future layer at the non-identity application boundary.")
        result = tome_merge_image(
            request.hidden_states,
            request.keys,
            rules={name.removeprefix("encoder_"): rule for name, rule in self._rules.items() if name.startswith("encoder_")}, r=r,
            token_sizes=request.token_sizes,
            protected_indices=request.protected_indices,
            coordinates=request.coordinates,
            clamp_r=bool(self.config.get("clamp_r", True)),
        )
        prior = self._invocations.get(request.invocation_id)
        if prior is None:
            composed = tuple(result.layout_deltas)
        else:
            if prior.stream_ids != request.stream_ids:
                raise BackendIntegrationError(
                    "ToMe crop stream order changed inside one encoder invocation."
                )
            if len(prior.deltas) != len(result.layout_deltas):
                raise BackendIntegrationError(
                    "ToMe crop batch size changed inside one encoder invocation."
                )
            composed = tuple(
                before.compose(after)
                for before, after in zip(prior.deltas, result.layout_deltas)
            )
        self._invocations[request.invocation_id] = _EncoderInvocationState(
            stream_ids=request.stream_ids,
            deltas=composed,
        )
        self._latest_layout_deltas = [
            _serialize_layout_delta(delta) for delta in composed
        ]
        for stream_index, delta in enumerate(result.layout_deltas):
            self._record_layout_delta(
                delta,
                site=tome_encoder_operation(self.stage, "llava_onevision")["target_site"],
                stream_index=stream_index,
                layer_index=request.layer_index,
                application_position=position.as_dict(),
                score_layer=tome_encoder_score_layer(self.stage, "llava_onevision") if "score_layer" in (getattr(self.stage, "criterion", {}) or {}) else request.layer_index,
            )
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "layer_index": request.layer_index,
            "source_counts": [delta.source_count for delta in composed],
            "compact_counts": [delta.result_count for delta in composed],
            "crop_isolated": True,
            "dense_restored": False,
        }
        return LlavaEncoderMergeResponse(
            hidden_states=result.hidden_states,
            token_sizes=result.token_sizes,
            proportional_attention_bias=result.proportional_attention_bias,
            layout_deltas=tuple(result.layout_deltas),
            source_groups=tuple(result.source_groups),
            stream_ids=request.stream_ids,
            metadata={
                **dict(result.metadata),
                "invocation_id": request.invocation_id,
                "layer_index": request.layer_index,
                "crop_isolated": True,
                "dense_restore_pending": True,
            },
        )

    def restore_dense_encoder_tokens(
        self,
        request: LlavaEncoderDenseRestoreRequest,
    ) -> LlavaEncoderDenseRestoreResponse:
        _require_request(request, LlavaEncoderDenseRestoreRequest, TOME_RESTORE_SITE)
        _require_image(request.modality)
        try:
            state = self._invocations.pop(request.invocation_id)
        except KeyError as exc:
            raise BackendIntegrationError(
                "Dense restore has no matching ToMe encoder invocation."
            ) from exc
        if state.stream_ids != request.stream_ids:
            raise BackendIntegrationError(
                "Dense restore crop order does not match the encoder invocation."
            )
        compact_count = _sequence_token_count(request.hidden_states)
        if any(delta.result_count != compact_count for delta in state.deltas):
            raise BackendIntegrationError(
                "Dense restore input count does not match the tracked compact layout."
            )
        gather_rows = tuple(
            tuple(
                _non_null(delta.old_to_new[index])
                for index in range(delta.source_count)
            )
            for delta in state.deltas
        )
        restored = _batch_gather_sequence(request.hidden_states, gather_rows)
        source_counts = {delta.source_count for delta in state.deltas}
        if len(source_counts) != 1:
            raise BackendIntegrationError(
                "LLaVA crop batches require a common dense patch count."
            )
        source_count = next(iter(source_counts))
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "source_counts": [source_count] * len(state.deltas),
            "compact_counts": [delta.result_count for delta in state.deltas],
            "crop_isolated": True,
            "dense_restored": True,
            "restored_count": source_count,
        }
        return LlavaEncoderDenseRestoreResponse(
            hidden_states=restored,
            dense_to_compact=gather_rows,
            stream_ids=request.stream_ids,
            source_token_count=source_count,
            compact_token_count=compact_count,
            metadata={
                "algorithm": "tome",
                "dense_restore": "replicate_merged_representations_by_old_to_new",
                "pre_anyres_shape_repaired": True,
                "crop_order_preserved": True,
            },
        )

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        """Merge packed-layout values with the original projected QK metric."""

        _require_request(request, LlavaPackedImageRequest, TOME_PACK_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        keys = request.keys
        if getattr(keys, "ndim", None) == 3:
            keys = keys.permute(1, 0, 2).contiguous()
        result = tome_preserve_source_order(
            tome_merge_image(
                request.tokens,
                keys.to(request.tokens.device),
                rules={name.removeprefix("output_"): rule for name, rule in self._rules.items() if name.startswith("output_")}, r=_strict_non_negative_int(
                    self.config.get("output_r", self.config.get("r", 1)),
                    "output_r",
                ),
                protected_indices=request.newline_indices,
                coordinates=request.coordinates,
                media_unit="image_feature",
                clamp_r=bool(self.config.get("clamp_r", True)),
            )
        )
        delta = result.layout_delta
        repaired_newlines = tuple(
            _non_null(delta.old_to_new[index]) for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": delta.operation,
        }
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "image_index": request.image_index,
            "compression_site": self.projector_target_sites[0],
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "dense_grid_legalized_before_output_merge": True,
            "permanent_llm_token_reduction": delta.result_count < delta.source_count,
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
        )
        return LlavaPackedImageResponse(
            tokens=result.hidden_states,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                **dict(result.metadata),
                "target_site": self.projector_target_sites[0],
                "callback_site": TOME_PACK_SITE,
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "permanent_llm_token_reduction": True,
            },
        )

    def uninstall(self) -> None:
        try:
            super().uninstall()
        finally:
            self._invocations.clear()


class LlavaOneVisionVisionZipActivation(_LlavaBridgeActivation):
    """VisionZip at the packed-image boundary with explicit AnyRes repair."""

    executor_name = "llava_onevision_visionzip_post_anyres_pack"

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected VisionZip site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        budget = _strict_positive_int(self.config.get("budget", 32), "budget")
        result = visionzip_image(
            request.tokens,
            request.attention,
            request.keys,
            rules=self._rules, budget=budget,
            attention_mode=str(self.config.get("attention_mode", "auto")),
            cls_index=request.cls_index,
            protected_indices=request.newline_indices,
            dominant_weight=_strict_positive_int(
                self.config.get("dominant_weight", 27), "dominant_weight"
            ),
            contextual_weight=_strict_positive_int(
                self.config.get("contextual_weight", 5), "contextual_weight"
            ),
            coordinates=request.coordinates,
            source_layout=dict(request.anyres_layout),
            result_layout={
                **dict(request.anyres_layout),
                "compression_site": self.projector_target_sites[0],
            },
        )
        repaired_newlines = tuple(
            _non_null(result.layout_delta.old_to_new[index])
            for index in request.newline_indices
        )
        if len(set(repaired_newlines)) != len(repaired_newlines):
            raise BackendIntegrationError(
                "VisionZip merged distinct image_newline tokens, violating AnyRes."
            )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": result.layout_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, result.layout_delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": result.layout_delta.operation,
        }
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "source_count": result.layout_delta.source_count,
            "result_count": result.layout_delta.result_count,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
            "image_sizes_preserved": True,
            "modalities_preserved": True,
        }
        self._latest_layout_deltas = [_serialize_layout_delta(result.layout_delta)]
        self._record_layout_delta(
            result.layout_delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
        )
        return LlavaPackedImageResponse(
            tokens=result.tokens,
            layout_delta=result.layout_delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                **dict(result.metadata),
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "image_sizes_preserved": True,
                "modalities_preserved": True,
            },
        )


class _LlavaOneVisionAttentionVisualActivation(_LlavaBridgeActivation):
    """Packed AnyRes attention adapter for PruMerge and GAP."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.contract.variant == "prumerge.image":
            values = {
                name: self.config[name]
                for name in PruMergeSpec.__dataclass_fields__
                if name in self.config
            }
            self.spec = PruMergeSpec(**values)
        else:
            values = {
                name: self.config[name]
                for name in GAPSpec.__dataclass_fields__
                if name in self.config
            }
            self.spec = GAPSpec(**values)

    @property
    def algorithm(self) -> str:
        return "prumerge" if self.contract.variant == "prumerge.image" else "gap"

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(
                f"Unexpected {self.algorithm} site {site!r}."
            )
        return self.compress_packed_image

    def compress_packed_image(
        self, request: LlavaPackedImageRequest
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index, request.image_sizes, request.modalities
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if request.attention is None or request.keys is None:
            raise BackendIntegrationError(
                f"LLaVA-OneVision {self.algorithm} requires packed attention and keys."
            )
        common = {
            "protected_indices": request.newline_indices,
            "coordinates": request.coordinates,
            "media_unit": "image_feature",
            "source_layout": dict(request.anyres_layout),
            "result_layout": {
                **dict(request.anyres_layout),
                "compression_site": self.projector_target_sites[0],
                "position_policy": "preserve_original",
            },
        }
        if self.algorithm == "prumerge":
            result = prumerge_image_tokens(
                request.tokens,
                request.attention,
                request.keys,
                rules=self._rules, spec=self.spec,
                **common,
            )
        else:
            result = gap_image_tokens(
                request.tokens,
                request.attention,
                rules=self._rules, spec=self.spec,
                **common,
            )
        repaired_newlines = tuple(
            _non_null(result.layout_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": result.layout_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, result.layout_delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": result.layout_delta.operation,
            "algorithm": self.algorithm,
        }
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "source_count": result.layout_delta.source_count,
            "result_count": result.layout_delta.result_count,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
        }
        self._latest_layout_deltas = [_serialize_layout_delta(result.layout_delta)]
        self._record_layout_delta(
            result.layout_delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
        )
        return LlavaPackedImageResponse(
            tokens=result.tokens,
            layout_delta=result.layout_delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                **dict(result.metadata),
                "post_anyres_pack": True,
                "image_newline_protected": True,
                "position_policy": "preserve_original",
            },
        )


class LlavaOneVisionPruMergeActivation(_LlavaOneVisionAttentionVisualActivation):
    executor_name = "llava_onevision_prumerge_post_anyres_pack"


class LlavaOneVisionGAPActivation(_LlavaOneVisionAttentionVisualActivation):
    executor_name = "llava_onevision_gap_post_anyres_pack"


class LlavaOneVisionDivPruneActivation(_LlavaBridgeActivation):
    """Layer-0 DivPrune at the packed-image pre-LLM boundary."""

    executor_name = "llava_onevision_divprune_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.spec = _divprune_spec(self.config)

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected DivPrune site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        result = divprune_image(
            request.tokens,
            rules=self._rules, spec=self.spec,
            protected_indices=request.newline_indices,
            coordinates=request.coordinates,
            media_unit="image_feature",
            source_layout=dict(request.anyres_layout),
        )
        source_delta = result.layout_delta
        repaired_newlines = tuple(
            _non_null(source_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_crop_boundaries = _compact_crop_boundaries(
            source_delta,
            request.anyres_layout["crop_boundaries"],
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": source_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": repaired_crop_boundaries,
            "compression_site": self.projector_target_sites[0],
            "layout_operation": source_delta.operation,
            "algorithm": "divprune",
        }
        delta = LayoutDelta.drop(
            source_delta.source_count,
            source_delta.dropped_indices,
            media_unit=source_delta.media_unit,
            coordinates=source_delta.coordinates,
            protected_indices=source_delta.protected_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "planned_count": self.spec.resolve_target_count(token_count),
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "selected_indices": list(delta.kept_indices),
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
            "image_sizes_preserved": True,
            "modalities_preserved": True,
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
            planned_count=self.spec.resolve_target_count(token_count),
            selected_indices=list(delta.kept_indices),
        )
        return LlavaPackedImageResponse(
            tokens=result.tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                **dict(result.metadata),
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "image_sizes_preserved": True,
                "modalities_preserved": True,
            },
        )


class LlavaOneVisionSAINTActivation(_LlavaBridgeActivation):
    """SAINT redundancy pruning on the structurally legal packed feature span."""

    executor_name = "llava_onevision_saint_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        values = dict(self.config)
        values.pop("layer_indices", None)
        values.pop("keep_ratios", None)
        try:
            self.spec = SAINTSpec(
                keep_ratio=float(values.pop("keep_ratio", 0.2)),
                group_size=int(values.pop("group_size", 1)),
                **values,
            )
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid SAINT config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected SAINT site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self, request: LlavaPackedImageRequest
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        if request.keys is None:
            raise BackendIntegrationError("LLaVA-OneVision SAINT requires packed keys.")
        keys = request.keys.permute(1, 0, 2).contiguous()
        result = saint_image_tokens(
            request.tokens,
            keys,
            rules=self._rules, spec=self.spec,
            protected_indices=request.newline_indices,
            media_unit="image_feature",
            source_layout=dict(request.anyres_layout),
        )
        return _packed_reduction_response(
            self,
            request,
            result,
            algorithm="saint",
        )


class LlavaOneVisionAIMActivation(_LlavaBridgeActivation):
    """AIM differentiable merging at the packed-image boundary."""

    executor_name = "llava_onevision_aim_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.spec = AIMSpec(**dict(self.config))
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid AIM config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected AIM site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self, request: LlavaPackedImageRequest
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        result = aim_image_tokens(
            request.tokens,
            rules=self._rules, spec=self.spec,
            protected_indices=request.newline_indices,
            media_unit="image_feature",
            source_layout=dict(request.anyres_layout),
        )
        return _packed_reduction_response(
            self,
            request,
            result,
            algorithm="aim",
        )


class LlavaOneVisionMMTokActivation(_LlavaBridgeActivation):
    """Per-crop MMTok selection legalized through native AnyRes packing."""

    executor_name = "llava_onevision_mmtok_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_mmtok_executor_config(
                stage=self.stage,
                binding_config=self.binding_config,
                extra_keys=frozenset({"tokens_per_crop"}),
            )
        except ValueError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        budget_keys = {
            key
            for key in ("tokens_per_crop", "retain_ratio", "target_count")
            if key in self.config
        }
        if len(budget_keys) > 1:
            raise BackendIntegrationError(
                "LLaVA-OneVision MMTok accepts one budget: tokens_per_crop, "
                "retain_ratio, or target_count."
            )
        self.tokens_per_crop = (
            _strict_positive_int(
                self.config["tokens_per_crop"], "tokens_per_crop"
            )
            if "tokens_per_crop" in self.config
            else (None if budget_keys else 32)
        )
        self.spec = _mmtok_spec(self.config)
        if (
            self.tokens_per_crop is not None
            and self.spec.min_visual_tokens > self.tokens_per_crop
        ):
            raise BackendIntegrationError(
                "MMTok min_visual_tokens cannot exceed tokens_per_crop."
            )

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected MMTok site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if (
            request.projected_crops is None
            or request.vision_crops is None
            or request.text_embeddings is None
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision MMTok requires projected crops, pre-projector "
                "vision crops, and prompt input embeddings."
            )
        projected = request.projected_crops
        vision = request.vision_crops
        if getattr(projected, "ndim", None) != 3 or getattr(
            vision, "ndim", None
        ) != 3:
            raise BackendIntegrationError(
                "LLaVA-OneVision MMTok crop payloads must be rank three."
            )
        if tuple(projected.shape[:2]) != tuple(vision.shape[:2]):
            raise BackendIntegrationError(
                "LLaVA-OneVision MMTok pre/post-projector crop layouts differ."
            )
        crop_outputs = tuple(
            mmtok_image_tokens(
                projected[crop_index],
                vision[crop_index].to(projected.device),
                request.text_embeddings.to(projected.device),
                rules=self._rules, spec=self.spec,
                target_count=self.tokens_per_crop,
                media_unit="image_patch",
                source_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "vision_source": "siglip_pre_projector",
                    "text_source": "prompt_input_embeddings",
                },
            )
            for crop_index in range(int(projected.shape[0]))
        )
        keep = _crop_selection_packed_keep_indices(
            tuple(output.indices for output in crop_outputs),
            algorithm="MMTok",
            patch_count=int(projected.shape[1]),
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=token_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        selected_set = set(int(value) for value in keep.detach().cpu().tolist())
        dropped = tuple(
            index for index in range(token_count) if index not in selected_set
        )
        source_delta = LayoutDelta.drop(
            token_count,
            dropped,
            media_unit="image_feature",
            protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout),
        )
        repaired_newlines = tuple(
            _non_null(source_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": source_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, source_delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": source_delta.operation,
            "algorithm": "mmtok",
            "crop_count": len(crop_outputs),
        }
        delta = LayoutDelta.drop(
            source_delta.source_count,
            source_delta.dropped_indices,
            media_unit=source_delta.media_unit,
            protected_indices=source_delta.protected_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        tokens = request.tokens.index_select(0, keep)
        selected_per_crop = [
            list(output.layout_delta.kept_indices) for output in crop_outputs
        ]
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "crop_count": len(crop_outputs),
            "tokens_per_crop": self.tokens_per_crop,
            "budget_policy": (
                "tokens_per_crop"
                if self.tokens_per_crop is not None
                else (
                    "target_count"
                    if self.spec.target_count is not None
                    else "retain_ratio"
                )
            ),
            "selected_indices_per_crop": selected_per_crop,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
            crop_count=len(crop_outputs),
            selected_indices_per_crop=selected_per_crop,
        )
        return LlavaPackedImageResponse(
            tokens=tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                "algorithm": "mmtok",
                "variant": "mmtok.image",
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "crop_isolated_selection": True,
                "selected_indices_per_crop": selected_per_crop,
                "coverage_compute_dtype": "float32",
                "budget_policy": self._last_layout["budget_policy"],
            },
        )


class LlavaOneVisionVisPrunerActivation(_LlavaBridgeActivation):
    """Per-crop visual-cue selection legalized through native AnyRes packing."""

    executor_name = "llava_onevision_vispruner_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_vispruner_executor_config(
                stage=self.stage,
                binding_config=self.binding_config,
                extra_keys=frozenset({"tokens_per_crop"}),
            )
        except ValueError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        budget_keys = {
            key for key in ("tokens_per_crop", "target_count") if key in self.config
        }
        if len(budget_keys) > 1:
            raise BackendIntegrationError(
                "LLaVA-OneVision VisPruner accepts one budget: tokens_per_crop "
                "or target_count."
            )
        self.tokens_per_crop = (
            _strict_positive_int(
                self.config["tokens_per_crop"], "tokens_per_crop"
            )
            if "tokens_per_crop" in self.config
            else (None if "target_count" in self.config else 32)
        )
        self.spec = _vispruner_spec(self.config)
        if (
            self.tokens_per_crop is not None
            and self.spec.min_visual_tokens > self.tokens_per_crop
        ):
            raise BackendIntegrationError(
                "VisPruner min_visual_tokens cannot exceed tokens_per_crop."
            )

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected VisPruner site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if (
            request.projected_crops is None
            or request.vision_crops is None
            or request.vision_attentions is None
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision VisPruner requires projected crops, "
                "pre-projector vision crops, and final visual attention."
            )
        projected = request.projected_crops
        vision = request.vision_crops
        attention = request.vision_attentions
        if getattr(projected, "ndim", None) != 3 or getattr(
            vision, "ndim", None
        ) != 3:
            raise BackendIntegrationError(
                "LLaVA-OneVision VisPruner crop features must be rank three."
            )
        if tuple(projected.shape[:2]) != tuple(vision.shape[:2]):
            raise BackendIntegrationError(
                "LLaVA-OneVision VisPruner pre/post-projector crop layouts differ."
            )
        crop_count, patch_count = (int(value) for value in projected.shape[:2])
        if getattr(attention, "ndim", None) not in (3, 4) or int(
            attention.shape[0]
        ) != crop_count or tuple(attention.shape[-2:]) != (
            patch_count,
            patch_count,
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision VisPruner attention does not match crop patches. "
                "Encoder-compressed SigLIP attention cannot be used as dense crop scores."
            )
        values = vision if self.projector_target_sites[0] == "pre_vision_projector" else projected
        crop_outputs = tuple(
            vispruner_image_tokens(
                values[crop_index],
                vision[crop_index].to(values.device),
                attention[crop_index].to(values.device),
                rules=self._rules, spec=self.spec,
                target_count=self.tokens_per_crop,
                media_unit="image_patch",
                source_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "attention_source": "siglip_final_mean_received",
                    "selection_feature_source": "siglip_pre_projector",
                },
            )
            for crop_index in range(crop_count)
        )
        keep = _crop_selection_packed_keep_indices(
            tuple(output.indices for output in crop_outputs),
            algorithm="VisPruner",
            patch_count=patch_count,
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=token_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        selected_set = set(int(value) for value in keep.detach().cpu().tolist())
        dropped = tuple(
            index for index in range(token_count) if index not in selected_set
        )
        source_delta = LayoutDelta.drop(
            token_count,
            dropped,
            media_unit="image_feature",
            protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout),
        )
        repaired_newlines = tuple(
            _non_null(source_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": source_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, source_delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": source_delta.operation,
            "algorithm": "vispruner",
            "crop_count": len(crop_outputs),
        }
        delta = LayoutDelta.drop(
            source_delta.source_count,
            source_delta.dropped_indices,
            media_unit=source_delta.media_unit,
            protected_indices=source_delta.protected_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        tokens = request.tokens.index_select(0, keep)
        selected_per_crop = [
            list(output.layout_delta.kept_indices) for output in crop_outputs
        ]
        budget_policy = (
            "tokens_per_crop" if self.tokens_per_crop is not None else "target_count"
        )
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "crop_count": len(crop_outputs),
            "tokens_per_crop": self.tokens_per_crop,
            "budget_policy": budget_policy,
            "selected_indices_per_crop": selected_per_crop,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
            crop_count=len(crop_outputs),
            selected_indices_per_crop=selected_per_crop,
        )
        return LlavaPackedImageResponse(
            tokens=tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                "algorithm": "vispruner",
                "variant": "vispruner.image",
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "crop_isolated_selection": True,
                "selected_indices_per_crop": selected_per_crop,
                "attention_source": "siglip_final_mean_received",
                "budget_policy": budget_policy,
            },
        )


class LlavaOneVisionZOOPruneActivation(_LlavaBridgeActivation):
    """Per-crop zero-order projector selection with native AnyRes repair."""

    executor_name = "llava_onevision_zoo_prune_post_anyres_pack"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_zoo_prune_executor_config(
                stage=self.stage,
                binding_config=self.binding_config,
            )
        except ValueError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        self.spec = _zoo_prune_spec(self.config)
        self._proxy_cache = ZOOProjectorProxyCache()

    def uninstall(self) -> None:
        try:
            super().uninstall()
        finally:
            self._proxy_cache.clear()

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != VISIONZIP_SITE:
            raise BackendIntegrationError(f"Unexpected ZOO-Prune site {site!r}.")
        return self.compress_packed_image

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if (
            request.projected_crops is None
            or request.vision_crops is None
            or not request.projector_layers
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision ZOO-Prune requires projected crops, native "
                "projector inputs, and the exact MM-projector layers."
            )
        projector_device, stored_device = next(
            (getattr(getattr(layer, "_hf_hook", None), "execution_device", None), parameter.device)
            for layer in request.projector_layers for parameter in layer.parameters())
        if projector_device is None:
            projector_device = stored_device
        projected = request.projected_crops.to(projector_device)
        projector_inputs = request.vision_crops.to(projector_device)
        if getattr(projected, "ndim", None) != 3 or getattr(
            projector_inputs, "ndim", None
        ) != 3:
            raise BackendIntegrationError(
                "LLaVA-OneVision ZOO-Prune crop payloads must be rank three."
            )
        if tuple(projected.shape[:2]) != tuple(projector_inputs.shape[:2]):
            raise BackendIntegrationError(
                "LLaVA-OneVision ZOO-Prune pre/post-projector crop layouts differ."
            )
        crop_count, patch_count = (int(value) for value in projected.shape[:2])
        crop_outputs = tuple(
            zoo_prune_image_tokens(
                projected[crop_index],
                projector_inputs[crop_index],
                request.projector_layers,
                rules=self._rules, spec=self.spec,
                proxy_cache=self._proxy_cache,
                media_unit="image_patch",
                source_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "projector_input": "siglip_pre_projector",
                    "selection_scope": "per_crop",
                },
            )
            for crop_index in range(crop_count)
        )
        keep = _crop_selection_packed_keep_indices(
            tuple(output.indices for output in crop_outputs),
            algorithm="ZOO-Prune",
            patch_count=patch_count,
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=token_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        selected_set = set(int(value) for value in keep.detach().cpu().tolist())
        dropped = tuple(
            index for index in range(token_count) if index not in selected_set
        )
        source_delta = LayoutDelta.drop(
            token_count,
            dropped,
            media_unit="image_feature",
            protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout),
        )
        repaired_newlines = tuple(
            _non_null(source_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": source_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, source_delta.result_count),),
            "compression_site": self.projector_target_sites[0],
            "layout_operation": source_delta.operation,
            "algorithm": "zoo_prune",
            "crop_count": crop_count,
        }
        delta = LayoutDelta.drop(
            source_delta.source_count,
            source_delta.dropped_indices,
            media_unit=source_delta.media_unit,
            protected_indices=source_delta.protected_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        tokens = request.tokens.index_select(0, keep)
        selected_per_crop = [
            list(output.layout_delta.kept_indices) for output in crop_outputs
        ]
        solver_per_crop = [
            output.metadata["low_rank_solver_used"] for output in crop_outputs
        ]
        budget_policy = (
            "target_count" if self.spec.target_count is not None else "retention_ratio"
        )
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": self.projector_target_sites[0],
            "image_index": request.image_index,
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "crop_count": crop_count,
            "budget_policy": budget_policy,
            "selected_indices_per_crop": selected_per_crop,
            "low_rank_solver_used_per_crop": solver_per_crop,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=self.projector_target_sites[0],
            image_index=request.image_index,
            crop_count=crop_count,
            selected_indices_per_crop=selected_per_crop,
            low_rank_solver_used_per_crop=solver_per_crop,
        )
        return LlavaPackedImageResponse(
            tokens=tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                "algorithm": "zoo_prune",
                "variant": "zoo_prune.image",
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "crop_isolated_selection": True,
                "selected_indices_per_crop": selected_per_crop,
                "low_rank_solver_used_per_crop": solver_per_crop,
                "budget_policy": budget_policy,
                "training_free": True,
            },
        )


class LlavaOneVisionRestoreActivation(_LlavaBridgeActivation):
    """RESTORE crop-local anchor merge plus decoder attention calibration."""

    executor_name = "llava_onevision_restore_anchor_calibration"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            values = merge_restore_executor_config(
                stage=self.stage,
                binding_config=self.binding_config,
            )
            tokens_per_crop = values.pop("tokens_per_crop", None)
            if tokens_per_crop is not None:
                if "target_count" in values:
                    raise ValueError(
                        "RESTORE accepts either target_count or tokens_per_crop, not both."
                    )
                values["target_count"] = tokens_per_crop
            self.spec = RestoreSpec(**values)
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid RESTORE config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site == VISIONZIP_SITE:
            return self.compress_packed_image
        if site == RESTORE_CALIBRATION_SITE:
            return self.decoder_calibration_marker
        raise BackendIntegrationError(f"Unexpected RESTORE site {site!r}.")

    def decoder_calibration_marker(self, request: Any) -> Any:
        raise BackendIntegrationError(
            "RESTORE decoder calibration is provider-owned and is not a pruning callback."
        )

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        source_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=source_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if request.vision_crops is None or request.vision_attentions is None:
            raise BackendIntegrationError(
                "LLaVA-OneVision RESTORE requires configured SigLIP features and attention."
            )
        value_site = self.projector_target_sites[0]
        vision = request.vision_crops
        attention = request.vision_attentions
        if getattr(vision, "ndim", None) != 3 or getattr(
            attention, "ndim", None
        ) not in {3, 4}:
            raise BackendIntegrationError(
                "LLaVA-OneVision RESTORE crop payload ranks do not match SigLIP."
            )
        crop_count, patch_count = (int(value) for value in vision.shape[:2])
        if int(attention.shape[0]) != crop_count or tuple(
            int(value) for value in attention.shape[-2:]
        ) != (patch_count, patch_count):
            raise BackendIntegrationError(
                "LLaVA-OneVision RESTORE features and attention disagree on crop layout."
            )
        if request.patches_per_side is None:
            raise BackendIntegrationError(
                "LLaVA-OneVision RESTORE requires the native crop patch grid."
            )
        packed_sources = _crop_source_ids_in_packed_layout(
            crop_count=crop_count,
            patch_count=patch_count,
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=source_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        core = _safe_get_model(self.model)
        projector = getattr(core, "mm_projector", None)
        if not callable(projector):
            raise BackendIntegrationError(
                "LLaVA-OneVision RESTORE could not resolve the native MM projector."
            )
        import torch
        entries: list[tuple[int, Any, Any, tuple[int, ...]]] = []
        crop_metadata: list[dict[str, Any]] = []
        for crop_index in range(crop_count):
            crop_mask = (packed_sources >= crop_index * patch_count) & (
                packed_sources < (crop_index + 1) * patch_count
            )
            packed_positions = crop_mask.nonzero(as_tuple=False).flatten()
            if int(packed_positions.numel()) < 1:
                continue
            patch_indices = (
                packed_sources.index_select(0, packed_positions)
                - crop_index * patch_count
            )
            selection = vision[crop_index].index_select(0, patch_indices.to(vision.device))
            values = request.tokens.index_select(0, packed_positions) if value_site != VISIONZIP_SITE else selection
            crop_attention = attention[crop_index]
            crop_attention = crop_attention.index_select(
                -2, patch_indices.to(crop_attention.device)
            ).index_select(-1, patch_indices.to(crop_attention.device))
            kernel = restore_image_tokens(
                values,
                selection,
                crop_attention,
                rules=self._rules, spec=self.spec,
                target_count=min(self.spec.target_count, int(selection.shape[0])),
                media_unit="image_patch",
                source_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "visible_anyres_sources": True,
                },
                result_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "position_policy": "real_packed_source_representative",
                },
            )
            projected = (projector(kernel.tokens) if value_site == VISIONZIP_SITE else kernel.tokens).to(request.tokens.device, request.tokens.dtype)
            for result_index in range(kernel.layout_delta.result_count):
                local_representative = int(kernel.representative_indices[result_index])
                representative = int(packed_positions[local_representative])
                members = tuple(
                    int(packed_positions[source])
                    for source, target in kernel.layout_delta.old_to_new.items()
                    if target == result_index
                )
                entries.append(
                    (
                        representative,
                        projected[result_index],
                        kernel.token_sizes[result_index],
                        members,
                    )
                )
            crop_metadata.append(
                {
                    "crop_index": crop_index,
                    "source_count": kernel.layout_delta.source_count,
                    "result_count": kernel.layout_delta.result_count,
                    "representative_patch_indices": [
                        int(patch_indices[int(value)])
                        for value in kernel.representative_indices.detach().cpu().tolist()
                    ],
                }
            )
        for newline in request.newline_indices:
            entries.append(
                (
                    int(newline),
                    request.tokens[int(newline)],
                    torch.ones((), dtype=torch.float32, device=request.tokens.device),
                    (int(newline),),
                )
            )
        entries.sort(key=lambda item: item[0])
        if not entries:
            raise BackendIntegrationError("LLaVA-OneVision RESTORE produced no tokens.")
        mapping: dict[int, int] = {}
        for result_index, (_position, _token, _size, members) in enumerate(entries):
            for source in members:
                if source in mapping:
                    raise InvariantError(
                        "LLaVA-OneVision RESTORE mapped one source more than once."
                    )
                mapping[source] = result_index
        if set(mapping) != set(range(source_count)):
            raise InvariantError(
                "LLaVA-OneVision RESTORE did not partition the packed AnyRes sequence."
            )
        representatives = tuple(item[0] for item in entries)
        visual_sizes = torch.stack(tuple(item[2] for item in entries)).float()
        tokens = torch.stack(tuple(item[1] for item in entries))
        repaired_newlines = tuple(mapping[index] for index in request.newline_indices)
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": len(entries),
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, len(entries)),),
            "compression_site": value_site,
            "callback_site": VISIONZIP_SITE,
            "value_space": "projected" if value_site == "post_vision_projector" else "vision",
            "layout_operation": "merge",
            "algorithm": "restore",
            "position_policy": "real_packed_source_representative",
            "representative_indices": representatives,
            "token_sizes": tuple(
                float(value) for value in visual_sizes.detach().cpu().tolist()
            ),
        }
        delta = LayoutDelta.from_mapping(
            source_count=source_count,
            old_to_new=mapping,
            result_count=len(entries),
            media_unit="image_feature",
            protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
            ordering_policy=OrderingPolicy.EXPLICIT,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": value_site,
            "image_index": request.image_index,
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "representative_indices": list(representatives),
            "token_sizes": list(repaired_layout["token_sizes"]),
            "crop_results": crop_metadata,
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=value_site,
            callback_site=VISIONZIP_SITE,
            representative_indices=list(representatives),
            token_sizes=list(repaired_layout["token_sizes"]),
            mass_conserved=True,
        )
        return LlavaPackedImageResponse(
            tokens=tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                "algorithm": "restore",
                "variant": "restore.image",
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "crop_isolated_merge": True,
                "attention_calibration": "token_mass_plus_rope_distance",
            },
            visual_sizes=visual_sizes,
            representative_indices=representatives,
            original_visual_length=source_count,
        )

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            **super().runtime_metadata(),
            "target_sites": [*self.projector_target_sites, RESTORE_CALIBRATION_SITE],
            "target_count_per_visible_crop": self.spec.target_count,
            "cache_policy": "stable",
            "cache_transform": "none",
            "position_policy": "real_packed_source_representative",
            "attention_calibration": "token_mass_plus_rope_distance",
        }


class LlavaOneVisionNuwaActivation(_LlavaBridgeActivation):
    """Nüwa Stage 1 AnyRes aggregation plus Stage 2 decoder pruning."""

    executor_name = "llava_onevision_nuwa_two_stage"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_nuwa_executor_config(
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.spec = NuwaSpec(**self.config)
            self.decoder_selection = NuwaDecoderMethod(
                stage=self.stage,
                spec=self.spec,
                rules=self._rules,
                backbone="llava_onevision",
            )
            self.application_position = self.decoder_selection.positions[0]
            self.decoder_target_site = self.decoder_selection.target_site
            self.decoder_position_policy = self.decoder_selection.position_policy
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid Nüwa config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site == VISIONZIP_SITE:
            return self.compress_packed_image
        if site == FASTV_SITE:
            return self.apply_decoder_selection
        raise BackendIntegrationError(f"Unexpected Nüwa site {site!r}.")

    def compress_packed_image(
        self,
        request: LlavaPackedImageRequest,
    ) -> LlavaPackedImageResponse:
        _require_request(request, LlavaPackedImageRequest, VISIONZIP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(
            request.image_index,
            request.image_sizes,
            request.modalities,
        )
        token_count = _single_sequence_token_count(request.tokens)
        _validate_anyres_layout(
            request.anyres_layout,
            token_count=token_count,
            newline_indices=request.newline_indices,
            image_index=request.image_index,
        )
        if any(
            value is None
            for value in (
                request.projected_crops,
                request.vision_crops,
                request.semantic_crops,
                request.vision_attentions,
                request.vision_keys,
            )
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision Nüwa requires aggregation/semantic crops, final "
                "visual attention, keys, and projected crops."
            )
        value_site = self.projector_target_sites[0]
        projected = request.projected_crops
        aggregation = request.vision_crops
        semantic = request.semantic_crops
        attention = request.vision_attentions
        keys = request.vision_keys
        if any(
            getattr(value, "ndim", None) != 3
            for value in (projected, aggregation, semantic, keys)
        ) or getattr(attention, "ndim", None) != 4:
            raise BackendIntegrationError(
                "LLaVA-OneVision Nüwa crop payload ranks do not match SigLip."
            )
        crop_count, patch_count = (int(value) for value in aggregation.shape[:2])
        if any(
            tuple(value.shape[:2]) != (crop_count, patch_count)
            for value in (projected, semantic, keys)
        ) or int(attention.shape[0]) != crop_count or tuple(
            int(value) for value in attention.shape[-2:]
        ) != (patch_count, patch_count):
            raise BackendIntegrationError(
                "LLaVA-OneVision Nüwa crop payloads disagree on crop/patch layout."
            )
        if (
            request.patches_per_side is None
            or request.patches_per_side**2 != patch_count
        ):
            raise BackendIntegrationError(
                "LLaVA-OneVision Nüwa requires the exact square SigLip crop grid."
            )
        crop_outputs = tuple(
            nuwa_stage1_image_tokens(
                (projected if value_site == "post_vision_projector" else aggregation)[crop_index],
                semantic[crop_index],
                attention[crop_index],
                keys[crop_index],
                rules=self._rules, grid_height=request.patches_per_side,
                grid_width=request.patches_per_side,
                spec=self.spec,
                crop_count=crop_count,
                media_unit="image_patch",
                source_layout={
                    "backbone": "llava_onevision",
                    "image_index": request.image_index,
                    "crop_index": crop_index,
                    "stage": "stage1",
                },
            )
            for crop_index in range(crop_count)
        )
        keep = _crop_selection_packed_keep_indices(
            tuple(output.selected_indices for output in crop_outputs),
            algorithm="Nüwa",
            patch_count=patch_count,
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=token_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        packed_sources = _crop_source_ids_in_packed_layout(
            crop_count=crop_count,
            patch_count=patch_count,
            patches_per_side=request.patches_per_side,
            grid_width=request.grid_width,
            grid_height=request.grid_height,
            patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0],
            source_count=token_count,
            newline_indices=request.newline_indices,
            device=request.tokens.device,
        )
        core = _safe_get_model(self.model)
        projector = getattr(core, "mm_projector", None)
        if not callable(projector):
            raise BackendIntegrationError(
                "LLaVA-OneVision Nüwa could not resolve the native MM projector."
            )
        dense_values = (aggregation if value_site == "pre_vision_projector" else projected).reshape(crop_count * patch_count, -1).clone()
        selected_per_crop: list[list[int]] = []
        for crop_index, output in enumerate(crop_outputs):
            selected = output.selected_indices.to(device=dense_values.device)
            aggregates = (projector(output.tokens) if value_site == VISIONZIP_SITE else output.tokens).to(dense_values.device, dense_values.dtype)
            dense_indices = selected + crop_index * patch_count
            dense_values.index_copy_(0, dense_indices, aggregates)
            selected_per_crop.append(
                [int(value) for value in output.selected_indices.detach().cpu().tolist()]
            )
        replaced = request.tokens.clone()
        valid = packed_sources >= 0
        replaced[valid] = dense_values.index_select(
            0, packed_sources[valid].to(dense_values.device)
        ).to(replaced.device, replaced.dtype)
        tokens = replaced.index_select(0, keep)
        selected_set = set(int(value) for value in keep.detach().cpu().tolist())
        dropped = tuple(
            index for index in range(token_count) if index not in selected_set
        )
        source_delta = LayoutDelta.drop(
            token_count,
            dropped,
            media_unit="image_feature",
            protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout),
        )
        repaired_newlines = tuple(
            _non_null(source_delta.old_to_new[index])
            for index in request.newline_indices
        )
        repaired_layout = {
            **dict(request.anyres_layout),
            "packed_token_count": source_delta.result_count,
            "newline_indices": repaired_newlines,
            "crop_boundaries": ((0, source_delta.result_count),),
            "compression_site": value_site,
            "callback_site": VISIONZIP_SITE,
            "value_space": "projected" if value_site == "post_vision_projector" else "vision",
            "layout_operation": source_delta.operation,
            "algorithm": "nuwa",
            "stage": "stage1",
            "crop_count": crop_count,
            "position_policy": "rpme",
        }
        delta = LayoutDelta.drop(
            source_delta.source_count,
            source_delta.dropped_indices,
            media_unit=source_delta.media_unit,
            protected_indices=source_delta.protected_indices,
            source_layout=dict(request.anyres_layout),
            result_layout=repaired_layout,
        )
        _validate_anyres_layout(
            repaired_layout,
            token_count=delta.result_count,
            newline_indices=repaired_newlines,
            image_index=request.image_index,
        )
        representatives = list(delta.kept_indices)
        self._last_layout = {
            "invocation_id": request.invocation_id,
            "target_site": value_site,
            "image_index": request.image_index,
            "stage": "stage1",
            "source_count": delta.source_count,
            "result_count": delta.result_count,
            "crop_count": crop_count,
            "selected_indices_per_crop": selected_per_crop,
            "rpme_source_indices": representatives,
            "newline_indices_before": list(request.newline_indices),
            "newline_indices_after": list(repaired_newlines),
        }
        self._latest_layout_deltas = [_serialize_layout_delta(delta)]
        self._record_layout_delta(
            delta,
            site=value_site,
            callback_site=VISIONZIP_SITE,
            image_index=request.image_index,
            stage="stage1",
            selected_indices_per_crop=selected_per_crop,
            rpme_source_indices=representatives,
        )
        return LlavaPackedImageResponse(
            tokens=tokens,
            layout_delta=delta,
            image_sizes=request.image_sizes,
            modalities=request.modalities,
            newline_indices=repaired_newlines,
            anyres_layout=MappingProxyType(repaired_layout),
            image_index=request.image_index,
            metadata={
                "algorithm": "nuwa",
                "variant": "nuwa.image",
                "stage": "stage1",
                "post_anyres_pack": True,
                "pre_llm_splice": True,
                "image_newline_protected": True,
                "crop_isolated_selection": True,
                "weighted_visual_aggregation": True,
                "rpme_source_indices": representatives,
                "rpme_original_visual_tokens": token_count,
            },
        )

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "application_position": self.application_position.as_dict(),
            "target_sites": [*self.projector_target_sites, self.decoder_target_site],
            "decoder_target_site": self.decoder_target_site, "score_layer": self.spec.stage2_score_layer}


class LlavaOneVisionPyramidDropActivation(_LlavaBridgeActivation):
    """PyramidDrop executor that requires provider-owned decoder-loop repair."""

    executor_name = "llava_onevision_pyramid_drop_decoder_loop"
    decoder_position_policy = "contiguous"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.decoder_selection = PyramidDecoderMethod(stage=self.stage, config=self.config, rules=self._rules, backbone="llava_onevision")
        self.target_site = self.decoder_selection.target_site

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != PYRAMID_DROP_SITE:
            raise BackendIntegrationError(f"Unexpected PyramidDrop site {site!r}.")
        return self.process_decoder_layer

    def process_decoder_layer(
        self,
        request: LlavaDecoderLayerRequest,
    ) -> LlavaDecoderLayerResponse:
        _require_request(request, LlavaDecoderLayerRequest, PYRAMID_DROP_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(0, request.image_sizes, request.modalities)
        if isinstance(request.layer_index, bool) or not isinstance(
            request.layer_index, int
        ):
            raise BackendIntegrationError("PyramidDrop layer_index must be an integer.")
        if request.layer_index < 0 or request.layer_index >= request.num_layers:
            raise BackendIntegrationError(
                "PyramidDrop layer_index is outside decoder layers."
            )
        schedule = self._schedule(request.num_layers)
        if request.layer_index not in schedule.layer_indices:
            return LlavaDecoderLayerResponse(
                hidden_states=request.hidden_states,
                attention_mask=request.attention_mask,
                position_ids=request.position_ids,
                past_key_values=request.past_key_values,
                image_indices=request.image_indices,
                newline_indices=request.newline_indices,
                image_sizes=request.image_sizes,
                modalities=request.modalities,
                layout_delta=None,
                layer_index=request.layer_index,
                dropped=False,
                metadata={
                    "algorithm": "pyramid_drop",
                    "schedule_boundary": False,
                    "decoder_loop_control": True,
                },
            )
        point = self.application_positions[schedule.layer_indices.index(request.layer_index)]
        result = self.decoder_selection.select(
            request.hidden_states,
            image_indices=request.image_indices,
            image_attention_scores=request.image_attention_scores,
            instruction_query=request.instruction_query,
            image_keys=request.image_keys,
            attention_keys=request.attention_keys,
            attention_key_mask=request.attention_key_mask,
            protected_indices=request.newline_indices,
            media_unit="llm_token",
            source_layout={
                "backbone": "llava_onevision",
                "application_position": point.as_dict(),
                "invocation_id": request.invocation_id,
            },
            result_layout={
                "backbone": "llava_onevision",
                "application_position": point.as_dict(), "target_site": self.target_site,
                "position_policy": self.decoder_position_policy,
                "invocation_id": request.invocation_id,
            },
        )
        return self._finish_decoder_selection(result, point, sequence=request,
            attention_mask=request.attention_mask, position_ids=request.position_ids,
            past_key_values=request.past_key_values, use_cache=request.use_cache,
            repair_layout=request.repair_layout)

    def _schedule(self, num_layers: int) -> PyramidDropSchedule:
        self.decoder_selection.resolve_depth(num_layers)
        return self.decoder_selection.schedule

    @property
    def application_positions(self):
        return self.decoder_selection.positions

    @property
    def pyramid_rule_layers(self):
        schedule = self.decoder_selection.schedule
        return () if schedule is None else schedule.layer_indices


    def runtime_metadata(self):
        return {**super().runtime_metadata(), "target_site": self.target_site,
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.pyramid_rule_layers)}


class LlavaOneVisionPACTActivation(LlavaRegisteredDecoderSelectionActivation):
    """Registered PACT execution and the existing direct callback."""

    executor_name = "llava_onevision_pact_decoder_loop"
    method_factory = PACTDecoderMethod

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.spec = self.decoder_selection.spec
        self.application_position = self.decoder_selection.positions[0]
        self.target_site = self.decoder_selection.target_site

    def _handler_for_site(self, site):
        if site != PYRAMID_DROP_SITE:
            raise BackendIntegrationError(f"Unexpected PACT site {site!r}.")
        return self.process_decoder_layer

    def process_decoder_layer(
        self, request: LlavaPACTRequest
    ) -> LlavaDecoderLayerResponse:
        _require_request(request, LlavaPACTRequest, PYRAMID_DROP_SITE)
        _require_invocation_id(request.invocation_id)
        if request.layer_index != self.spec.boundary_layer:
            return LlavaDecoderLayerResponse(
                hidden_states=request.hidden_states,
                attention_mask=request.attention_mask,
                position_ids=request.position_ids,
                past_key_values=request.past_key_values,
                image_indices=request.image_indices,
                newline_indices=request.newline_indices,
                image_sizes=request.image_sizes,
                modalities=request.modalities,
                layout_delta=None,
                layer_index=request.layer_index,
                dropped=False,
                metadata={"algorithm": "pact", "schedule_boundary": False},
            )
        result = self.decoder_selection.select(request.hidden_states,
            image_indices=request.image_indices, protected_indices=request.newline_indices,
            query_states=request.query_states, key_states=request.key_states,
            clustering_keys=request.clustering_keys,
            image_attention_scores=request.image_attention_scores,
            source_layout={"backbone": "llava_onevision", "application_position": self.application_position.as_dict()},
            result_layout={"backbone": "llava_onevision", "application_position": self.application_position.as_dict(),
                "target_site": self.target_site, "position_policy": self.decoder_position_policy})
        response = self._finish_decoder_selection(result, self.application_position,
            sequence=request, attention_mask=request.attention_mask,
            position_ids=request.position_ids, past_key_values=request.past_key_values,
            use_cache=request.use_cache, repair_layout=request.repair_layout)
        return replace(response, layer_index=request.layer_index,
            dropped=not result.layout_delta.is_identity,
            token_sizes=result.token_sizes if self.spec.proportional_attention else None,
            metadata={**response.metadata, "image_newline_protected": True})


class LlavaOneVisionFastVActivation(_LlavaBridgeActivation):
    """FastV executor at one provider-owned pre-layer decoder boundary."""

    executor_name = "llava_onevision_fastv_decoder_loop"
    decoder_position_policy = "sparse_source"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_decoder_executor_config(
                "fastv",
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.decoder_selection = FastVDecoderMethod(stage=self.stage, config=self.config, rules=self._rules, backbone="llava_onevision")
            self.spec = self.decoder_selection.spec
            self._fastv_rules = self._rules
            self.application_position = self.decoder_selection.positions[0]
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid FastV config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != FASTV_SITE:
            raise BackendIntegrationError(f"Unexpected FastV site {site!r}.")
        return self.process_decoder_boundary

    def process_decoder_boundary(
        self,
        request: LlavaFastVRequest,
    ) -> LlavaDecoderLayerResponse:
        _require_request(request, LlavaFastVRequest, FASTV_SITE)
        _require_invocation_id(request.invocation_id)
        _validate_image_batch_identity(0, request.image_sizes, request.modalities)
        from mllm_comp.compressors.image.positions import decoder_token_position
        point = decoder_token_position(self.stage, self.spec.prune_layer, request.num_layers)
        if request.prune_layer != point.index:
            raise BackendIntegrationError("FastV callback ran at the wrong application boundary.")
        result = self.decoder_selection.select(
            request.hidden_states,
            image_indices=request.image_indices,
            scorer_query=request.scorer_query,
            attention_keys=request.attention_keys,
            attention_key_mask=request.attention_key_mask,
            protected_indices=request.newline_indices,
            media_unit="llm_token",
            source_layout={
                "backbone": "llava_onevision",
                "application_position": self.application_position.as_dict(),
                "invocation_id": request.invocation_id,
            },
            result_layout={
                "backbone": "llava_onevision",
                "position_policy": "sparse_source",
                "target_site": self.target_site,
                "application_position": self.application_position.as_dict(),
                "invocation_id": request.invocation_id,
            },
        )
        return self._finish_decoder_selection(result, point, sequence=request,
            attention_mask=request.attention_mask, position_ids=request.position_ids,
            past_key_values=request.past_key_values, use_cache=request.use_cache,
            repair_layout=request.repair_layout)

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            **super().runtime_metadata(), "target_sites": [self.target_site],
            "application_position": self.application_position.as_dict(),
            "prune_layer": self.spec.prune_layer,
            "score_layer": self.spec.prune_layer - 1 if self.spec.prune_layer else None,
            **({"rules": self._rule_specs, "rules_changed": True,
                "canonical_evidence_applicable": False, "development_runtime": True} if self._fastv_rules else {}),
        }


class LlavaOneVisionSGLActivation(_LlavaBridgeActivation):
    """SGL/SGP target controller using one lazily loaded compact guide."""

    executor_name = "llava_onevision_sgl_decoder_loop"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.execution_phase is not ExecutionPhase.GENERATE:
            raise BackendIntegrationError("LLaVA SGL supports image generation only.")
        if self.runtime is None:
            raise BackendIntegrationError("LLaVA SGL requires a target runtime.")
        try:
            self.config = merge_decoder_executor_config(
                "sgl",
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.spec = SGLSpec(**self.config)
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid SGL config: {exc}") from exc
        core = _safe_get_model(self.model)
        layers = tuple(getattr(core, "layers", ()) or ())
        if not layers:
            raise BackendIntegrationError("LLaVA SGL requires decoder layers.")
        self.guide = SGLGuideController(
            rules=self._rules, backbone_name="llava_onevision",
            target_runtime=self.runtime,
            spec=self.spec,
        )
        self.decoder_selection = SGLDecoderMethod(
            stage=self.stage,
            spec=self.spec,
            rules=self._rules,
            consume_scores=self.guide.consume_scores,
            num_layers=len(layers),
        )
        self.application_position = self.decoder_selection.positions[0]
        self.target_site = self.decoder_selection.target_site
        self.decoder_position_policy = self.decoder_selection.position_policy

    def install(self) -> None:
        if self._installed:
            return
        super().install()
        try:
            self.guide.attach()
        except BaseException:
            super().uninstall()
            raise

    def uninstall(self) -> None:
        try:
            super().uninstall()
        finally:
            self.guide.detach()

    def _handler_for_site(self, site):
        if site != FASTV_SITE:
            raise BackendIntegrationError(f"Unexpected SGL site {site!r}.")
        return self.apply_decoder_selection

    def runtime_metadata(self) -> Mapping[str, Any]:
        return {
            **super().runtime_metadata(),
            "prune_layer": self.application_position.index,
            "target_sites": [self.target_site],
            "application_position": self.application_position.as_dict(),
            "prune_layer_fraction": self.spec.prune_layer_fraction,
            "keep_ratio": self.spec.keep_ratio,
            "mode": self.spec.mode,
            "cache_policy": "compact_prefix",
            **self.guide.runtime_metadata(),
        }


class LlavaOneVisionFEATHERActivation(LlavaRegisteredDecoderSelectionActivation):
    executor_name = "llava_onevision_feather_decoder_loop"
    method_factory = FEATHERDecoderMethod


class LlavaOneVisionIVCPruneActivation(LlavaRegisteredDecoderSelectionActivation):
    executor_name = "llava_onevision_ivc_prune_decoder_loop"
    method_factory = IVCPruneDecoderMethod


class LlavaOneVisionDARTActivation(LlavaRegisteredDecoderSelectionActivation):
    method_factory = DARTDecoderMethod


class LlavaOneVisionD2PrunerActivation(LlavaRegisteredDecoderSelectionActivation):
    method_factory = D2PrunerDecoderMethod


class LlavaOneVisionSparseVLMActivation(LlavaRegisteredDecoderSelectionActivation):
    executor_name = "llava_onevision_sparsevlm_decoder_loop"
    method_factory = SparseVLMDecoderMethod


class LlavaOneVisionFinePrunerActivation(LlavaRegisteredDecoderSelectionActivation):
    executor_name = "llava_onevision_finepruner_shared_decoder_loop"
    method_factory = FinePrunerDecoderMethod


class LlavaOneVisionBTPActivation(_LlavaBridgeActivation):
    """Balanced Token Pruning on the shared compact-prefix decoder loop."""

    executor_name = "llava_onevision_btp_shared_decoder_loop"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_decoder_executor_config(
                "btp",
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.decoder_selection = BTPDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="llava_onevision")
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid BTP config: {exc}") from exc
        self.decoder_position_policy = self.decoder_selection.position_policy

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != PYRAMID_DROP_SITE:
            raise BackendIntegrationError(f"Unexpected BTP site {site!r}.")
        return self.apply_decoder_selection

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "application_positions": [point.as_dict() for point in self.application_positions],
            "target_site": self.target_site, "score_layers": list(self.spec.boundary_after_layer_indices[:-1])}


class LlavaOneVisionHalfVActivation(_LlavaBridgeActivation):
    """HalfV IVR/SSR on the shared compact-prefix decoder lifecycle."""

    executor_name = "llava_onevision_halfv_shared_decoder_loop"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_decoder_executor_config(
                "halfv",
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.decoder_selection = HalfVDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="llava_onevision")
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid HalfV config: {exc}") from exc
        self.decoder_position_policy = self.decoder_selection.position_policy

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != PYRAMID_DROP_SITE:
            raise BackendIntegrationError(f"Unexpected HalfV site {site!r}.")
        return self.apply_decoder_selection

    def runtime_metadata(self):
        return {**super().runtime_metadata(), "target_site": self.target_site,
            "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.spec.boundary_after_layer_indices)}


class LlavaOneVisionTransPruneActivation(_LlavaBridgeActivation):
    """TransPrune TTV/IGA stages on the shared layer-local decoder loop."""

    executor_name = "llava_onevision_transprune_shared_decoder_loop"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        try:
            self.config = merge_decoder_executor_config(
                "transprune",
                stage=self.stage,
                binding_config=self.binding_config,
            )
            self.decoder_selection = TransPruneDecoderMethod(
                stage=self.stage, config=self.config, rules=self._rules, backbone="llava_onevision")
            self.spec = self.decoder_selection.spec
            self.application_positions = self.decoder_selection.positions
            self.target_site = self.decoder_selection.target_site
            self.decoder_position_policy = self.decoder_selection.position_policy
        except (TypeError, ValueError) as exc:
            raise BackendIntegrationError(f"Invalid TransPrune config: {exc}") from exc

    def _handler_for_site(self, site: str) -> Callable[[Any], Any]:
        if site != PYRAMID_DROP_SITE:
            raise BackendIntegrationError(f"Unexpected TransPrune site {site!r}.")
        return self.apply_decoder_selection

    def runtime_metadata(self) -> dict[str, Any]:
        return {**super().runtime_metadata(), "application_positions": [point.as_dict() for point in self.application_positions],
            "score_layers": list(self.decoder_selection.score_layers),
            "target_site": self.target_site,
            "cache_policy": self.decoder_selection.cache_policy.value,
            "position_policy": self.decoder_selection.position_policy}


@dataclass(frozen=True)
class LlavaOneVisionImageCompressionAdapter:
    """Resolve exact LLaVA ImageLLM bindings without changing support state."""

    def resolve(
        self,
        *,
        model: Any,
        runtime: Any,
        stage: Any,
        binding: Any,
        binding_config: Any | None = None,
        execution_phase: ExecutionPhase | str,
        facade: Any | None = None,
        contract_projection: Any = None,
    ) -> _LlavaBridgeActivation:
        del facade
        if contract_projection is not None and binding.strategy_factory is not None:
            return LlavaRegisteredDecoderSelectionActivation(model=model, runtime=runtime,
                stage=stage, binding=binding, binding_config=binding_config,
                execution_phase=execution_phase, contract_projection=contract_projection)
        factory = _FACTORIES.get(getattr(binding, "name", ""))
        if factory is None:
            raise BackendIntegrationError(
                f"Unknown LLaVA-OneVision image binding {getattr(binding, 'name', None)!r}."
            )
        return factory(
            model=model,
            runtime=runtime,
            facade=None,
            stage=stage,
            binding=binding,
            binding_config=binding_config,
            execution_phase=execution_phase,
        )


def create_llava_onevision_tome_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionToMeActivation:
    del runtime, facade
    return LlavaOneVisionToMeActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_visionzip_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionVisionZipActivation:
    del runtime, facade
    return LlavaOneVisionVisionZipActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_prumerge_executor(**kwargs: Any) -> LlavaOneVisionPruMergeActivation:
    kwargs.pop("facade", None)
    return LlavaOneVisionPruMergeActivation(**kwargs)


def create_llava_onevision_gap_executor(**kwargs: Any) -> LlavaOneVisionGAPActivation:
    kwargs.pop("facade", None)
    return LlavaOneVisionGAPActivation(**kwargs)


def create_llava_onevision_divprune_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionDivPruneActivation:
    del runtime, facade
    return LlavaOneVisionDivPruneActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_saint_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionSAINTActivation:
    del runtime, facade
    return LlavaOneVisionSAINTActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_aim_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionAIMActivation:
    del runtime, facade
    return LlavaOneVisionAIMActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_mmtok_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionMMTokActivation:
    del runtime, facade
    return LlavaOneVisionMMTokActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_vispruner_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionVisPrunerActivation:
    del runtime, facade
    return LlavaOneVisionVisPrunerActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_zoo_prune_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionZOOPruneActivation:
    del runtime, facade
    return LlavaOneVisionZOOPruneActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_nuwa_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionNuwaActivation:
    del runtime, facade
    return LlavaOneVisionNuwaActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_restore_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionRestoreActivation:
    """Create RESTORE's AnyRes merge and decoder-calibration activation."""

    del facade
    return LlavaOneVisionRestoreActivation(
        model=model,
        runtime=runtime,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_pyramid_drop_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionPyramidDropActivation:
    del runtime, facade
    return LlavaOneVisionPyramidDropActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_pact_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionPACTActivation:
    del runtime, facade
    return LlavaOneVisionPACTActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_fastv_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionFastVActivation:
    del runtime, facade
    return LlavaOneVisionFastVActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_feather_executor(**kwargs: Any) -> LlavaOneVisionFEATHERActivation:
    kwargs.pop("facade", None)
    return LlavaOneVisionFEATHERActivation(**kwargs)


def create_llava_onevision_ivc_prune_executor(**kwargs: Any) -> LlavaOneVisionIVCPruneActivation:
    kwargs.pop("facade", None)
    return LlavaOneVisionIVCPruneActivation(**kwargs)


def create_llava_onevision_sgl_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionSGLActivation:
    del facade
    return LlavaOneVisionSGLActivation(
        model=model,
        runtime=runtime,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_dart_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionDARTActivation:
    del runtime, facade
    return LlavaOneVisionDARTActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_d2pruner_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionD2PrunerActivation:
    del facade
    return LlavaOneVisionD2PrunerActivation(
        model=model,
        runtime=runtime,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_sparsevlm_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionSparseVLMActivation:
    del runtime, facade
    return LlavaOneVisionSparseVLMActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_finepruner_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionFinePrunerActivation:
    del runtime, facade
    return LlavaOneVisionFinePrunerActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_btp_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionBTPActivation:
    del runtime, facade
    return LlavaOneVisionBTPActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_halfv_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionHalfVActivation:
    del runtime, facade
    return LlavaOneVisionHalfVActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


def create_llava_onevision_transprune_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
) -> LlavaOneVisionTransPruneActivation:
    del runtime, facade
    return LlavaOneVisionTransPruneActivation(
        model=model,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
    )


_FACTORIES: Mapping[str, Callable[..., _LlavaBridgeActivation]] = MappingProxyType(
    {
        "tome.image.llava_onevision": create_llava_onevision_tome_executor,
        "visionzip.image.llava_onevision": create_llava_onevision_visionzip_executor,
        "prumerge.image.llava_onevision": create_llava_onevision_prumerge_executor,
        "gap.image.llava_onevision": create_llava_onevision_gap_executor,
        "divprune.image.llava_onevision": create_llava_onevision_divprune_executor,
        "saint.image.llava_onevision": create_llava_onevision_saint_executor,
        "aim.image.llava_onevision": create_llava_onevision_aim_executor,
        "mmtok.image.llava_onevision": create_llava_onevision_mmtok_executor,
        "vispruner.image.llava_onevision": create_llava_onevision_vispruner_executor,
        "zoo_prune.image.llava_onevision": create_llava_onevision_zoo_prune_executor,
        "nuwa.image.llava_onevision": create_llava_onevision_nuwa_executor,
        "restore.image.llava_onevision": create_llava_onevision_restore_executor,
        "pyramid_drop.image.llava_onevision": (
            create_llava_onevision_pyramid_drop_executor
        ),
        "pact.image.llava_onevision": create_llava_onevision_pact_executor,
        "fastv.image.llava_onevision": create_llava_onevision_fastv_executor,
        "feather.image.llava_onevision": create_llava_onevision_feather_executor,
        "ivc_prune.image.llava_onevision": create_llava_onevision_ivc_prune_executor,
        "sgl.image.llava_onevision": create_llava_onevision_sgl_executor,
        "dart.image.llava_onevision": create_llava_onevision_dart_executor,
        "d2pruner.image.llava_onevision": create_llava_onevision_d2pruner_executor,
        "sparsevlm.image.llava_onevision": create_llava_onevision_sparsevlm_executor,
        "finepruner.image.llava_onevision": create_llava_onevision_finepruner_executor,
        "btp.image.llava_onevision": create_llava_onevision_btp_executor,
        "halfv.image.llava_onevision": create_llava_onevision_halfv_executor,
        "transprune.image.llava_onevision": (
            create_llava_onevision_transprune_executor
        ),
    }
)


def create_llava_onevision_image_compression_executor(
    *,
    model: Any,
    runtime: Any,
    facade: Any,
    stage: Any,
    binding: Any,
    binding_config: Any | None,
    execution_phase: ExecutionPhase | str,
    contract_projection: Any = None,
) -> _LlavaBridgeActivation:
    """Dispatcher-compatible factory for the exact LLaVA image bindings."""

    return LlavaOneVisionImageCompressionAdapter().resolve(
        model=model,
        runtime=runtime,
        facade=facade,
        stage=stage,
        binding=binding,
        binding_config=binding_config,
        execution_phase=execution_phase,
        contract_projection=contract_projection,
    )


def _merged_config(stage: Any, binding_config: Any | None) -> dict[str, Any]:
    config: dict[str, Any] = {}
    stage_config = getattr(stage, "config", None)
    if isinstance(stage_config, Mapping):
        config.update(stage_config)
    selection_config = getattr(binding_config, "config", None)
    if isinstance(selection_config, Mapping):
        config.update(selection_config)
    return config


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
    unknown = sorted(set(config) - keys)
    if unknown:
        raise BackendIntegrationError(
            f"Unknown DivPrune executor config keys: {unknown}."
        )
    try:
        return DivPruneSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid DivPrune config: {exc}") from exc


def _mmtok_spec(config: Mapping[str, Any]) -> MMTokSpec:
    values = {key: value for key, value in config.items() if key != "tokens_per_crop"}
    try:
        return MMTokSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid MMTok config: {exc}") from exc


def _vispruner_spec(config: Mapping[str, Any]) -> VisPrunerSpec:
    values = {key: value for key, value in config.items() if key != "tokens_per_crop"}
    try:
        return VisPrunerSpec(**values)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid VisPruner config: {exc}") from exc


def _zoo_prune_spec(config: Mapping[str, Any]) -> ZOOPruneSpec:
    try:
        return ZOOPruneSpec(**dict(config))
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Invalid ZOO-Prune config: {exc}") from exc


def _validate_exact_binding(
    binding: Any,
    stage: Any,
    execution_phase: ExecutionPhase,
) -> LlavaImageBindingContract:
    name = getattr(binding, "name", None)
    try:
        contract = LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS[name]
    except (KeyError, TypeError) as exc:
        raise BackendIntegrationError(
            f"Unknown LLaVA-OneVision image binding {name!r}."
        ) from exc
    if getattr(binding, "backbone", "llava_onevision") != "llava_onevision":
        raise BackendIntegrationError(
            "LLaVA executor received another backbone binding."
        )
    if getattr(binding, "variant", contract.variant) != contract.variant:
        raise BackendIntegrationError(
            "LLaVA binding variant does not match its contract."
        )
    stage_variant = getattr(stage, "variant", contract.variant)
    if stage_variant != contract.variant:
        raise BackendIntegrationError("LLaVA stage variant does not match its binding.")
    supported = getattr(binding, "supported_phases", None)
    if supported is not None and execution_phase not in {
        ExecutionPhase(item) for item in supported
    }:
        raise BackendIntegrationError(
            f"LLaVA binding does not support phase {execution_phase.value!r}."
        )
    target_modalities = getattr(binding, "target_modalities", None)
    if target_modalities is not None and {
        Modality(item) for item in target_modalities
    } != {Modality.IMAGE}:
        raise BackendIntegrationError("LLaVA ImageLLM bindings must target image only.")
    targets = getattr(binding, "targets", None)
    if targets is not None:
        physical_sites = tuple(
            getattr(target, "target_site", "")
            for target in sorted(targets, key=lambda item: getattr(item, "order", 0))
        )
        expected_sites = contract.sites
        if contract.binding.startswith("tome."):
            expected_sites = (tome_encoder_operation(stage, "llava_onevision")["target_site"], TOME_RESTORE_SITE, tome_output_site(stage, "llava_onevision"))
        if contract.binding.startswith(("fastv.", "dart.", "d2pruner.", "sgl.", "ivc_prune.", "feather.", "btp.", "halfv.", "pyramid_drop.", "finepruner.", "transprune.", "pact.", "sparsevlm.")):
            from mllm_comp.compressors.image.positions import decoder_token_operation
            expected_sites = (decoder_token_operation(stage, contract.variant.split(".")[0], backbone="llava_onevision")["target_site"],)
        if contract.variant in {"visionzip.image", "prumerge.image", "gap.image", "divprune.image", "aim.image", "vispruner.image", "saint.image.packed_feature_prune", "mmtok.image", "zoo_prune.image", "nuwa.image", "restore.image"}:
            from mllm_comp.compressors.image.positions import image_projector_sites
            projector_sites = image_projector_sites(stage, contract.variant.split(".")[0], "llava_onevision")
            expected_sites = (*projector_sites, *contract.sites[1:]) if contract.variant in {"nuwa.image", "restore.image"} else projector_sites
        if contract.binding.startswith("nuwa."):
            from mllm_comp.compressors.image.positions import decoder_token_operation
            expected_sites = (*expected_sites[:-1], decoder_token_operation(stage, "nuwa", backbone="llava_onevision")["target_site"])
        if physical_sites != expected_sites:
            raise BackendIntegrationError(
                "LLaVA binding physical sites drifted from the source bridge contract: "
                f"expected={contract.sites}, actual={physical_sites}."
            )
    return contract


def _resolve_bridge(model: Any) -> Any:
    for owner in (model, _safe_get_model(model)):
        if owner is None:
            continue
        for name in ("mllm_compression_bridge", "_mllm_compression_bridge"):
            bridge = getattr(owner, name, None)
            if bridge is not None:
                return bridge
    raise BackendIntegrationError(
        "LLaVA-OneVision has no mllm_compression_bridge. Install the exact "
        "pinned provider bridge before creating an image-compression activation."
    )


def _resolve_or_install_bridge(model: Any) -> tuple[Any, bool]:
    """Resolve an injected source bridge or install the pinned production one."""

    try:
        return _resolve_bridge(model), False
    except BackendIntegrationError:
        try:
            from .provider_bridge import (
                install_llava_onevision_ee14dbd_provider_bridge,
            )

            return install_llava_onevision_ee14dbd_provider_bridge(model), True
        except BaseException as exc:
            raise BackendIntegrationError(
                "LLaVA-OneVision could not resolve an injected bridge or install "
                "the pinned ee14dbd provider bridge."
            ) from exc


def _safe_get_model(model: Any) -> Any | None:
    get_model = getattr(model, "get_model", None)
    if not callable(get_model):
        return None
    try:
        return get_model()
    except Exception:
        return None


def _validate_bridge(
    bridge: Any,
    contract: LlavaImageBindingContract,
    execution_phase: ExecutionPhase,
) -> None:
    if getattr(bridge, "api_version", None) != LLAVA_ONEVISION_BRIDGE_API_VERSION:
        raise BackendIntegrationError("LLaVA compression bridge API version mismatch.")
    if getattr(bridge, "source_route", None) != LLAVA_ONEVISION_SOURCE_ROUTE:
        raise BackendIntegrationError(
            "LLaVA compression bridge did not attest the selected source-tree route."
        )
    available_sites = getattr(bridge, "available_sites", None)
    register = getattr(bridge, "register_callback", None)
    if not callable(available_sites) or not callable(register):
        raise BackendIntegrationError(
            "LLaVA compression bridge must implement available_sites/register_callback."
        )
    missing_sites = set(contract.sites) - set(available_sites())
    if missing_sites:
        raise BackendIntegrationError(
            f"LLaVA compression bridge is missing callback sites {sorted(missing_sites)}."
        )
    capabilities = getattr(bridge, "capabilities", None)
    if not isinstance(capabilities, Mapping):
        raise BackendIntegrationError("LLaVA compression bridge has no capability map.")
    required = set(contract.required_capabilities)
    if (
        contract.binding
        in {
            "pyramid_drop.image.llava_onevision",
            "fastv.image.llava_onevision",
            "sgl.image.llava_onevision",
            "dart.image.llava_onevision",
            "d2pruner.image.llava_onevision",
            "feather.image.llava_onevision",
            "ivc_prune.image.llava_onevision",
        }
        and execution_phase is ExecutionPhase.GENERATE
    ):
        required.add("kv_cache_repair")
    missing_capabilities = sorted(
        capability
        for capability in required
        if capabilities.get(capability) is not True
    )
    if missing_capabilities:
        raise BackendIntegrationError(
            "LLaVA compression bridge lacks required capabilities "
            f"{missing_capabilities}."
        )


def _require_request(request: Any, expected: type[Any], site: str) -> None:
    if not isinstance(request, expected):
        raise BackendIntegrationError(
            f"LLaVA callback {site!r} requires {expected.__name__}, got "
            f"{type(request).__name__}."
        )


def _require_image(value: Modality | str) -> None:
    try:
        modality = Modality(value)
    except (TypeError, ValueError) as exc:
        raise BackendIntegrationError(f"Unknown callback modality {value!r}.") from exc
    if modality is not Modality.IMAGE:
        raise BackendIntegrationError(
            "This LLaVA compression slice is image-only; video is a separate support unit."
        )


def _require_invocation_id(value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise BackendIntegrationError("LLaVA callback invocation_id must be non-empty.")


def _sequence_batch_size(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        ndim = len(shape)
        if ndim == 2:
            return 1
        if ndim == 3:
            return int(shape[0])
        raise BackendIntegrationError("Encoder hidden_states must be rank 2 or 3.")
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or not value:
        raise BackendIntegrationError("Encoder hidden_states has no sequence layout.")
    first = value[0]
    if isinstance(first, Sequence) and first and isinstance(first[0], Sequence):
        return len(value)
    return 1


def _sequence_token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) == 2:
            return int(shape[0])
        if len(shape) == 3:
            return int(shape[1])
        raise BackendIntegrationError("hidden_states must be rank 2 or 3.")
    batch_size = _sequence_batch_size(value)
    return len(value) if batch_size == 1 else len(value[0])


def _single_sequence_token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None:
        if len(shape) != 2:
            raise BackendIntegrationError("Packed image tokens must be rank 2.")
        return int(shape[0])
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise BackendIntegrationError("Packed image tokens must be a sequence.")
    return len(value)


def _batch_gather_sequence(
    hidden_states: Any,
    gather_rows: tuple[tuple[int, ...], ...],
) -> Any:
    shape = getattr(hidden_states, "shape", None)
    if shape is not None:
        try:
            import torch
        except ImportError as exc:
            raise BackendIntegrationError(
                "Tensor dense restore requires PyTorch."
            ) from exc
        if len(shape) == 2:
            indices = torch.tensor(
                gather_rows[0], dtype=torch.long, device=hidden_states.device
            )
            return hidden_states.index_select(0, indices)
        if len(shape) != 3 or len(gather_rows) != int(shape[0]):
            raise BackendIntegrationError("Batched dense restore shape mismatch.")
        rows = []
        for batch_index, indices_row in enumerate(gather_rows):
            indices = torch.tensor(
                indices_row, dtype=torch.long, device=hidden_states.device
            )
            rows.append(hidden_states[batch_index].index_select(0, indices))
        return torch.stack(rows, dim=0)
    if len(gather_rows) == 1:
        return [hidden_states[index] for index in gather_rows[0]]
    if len(hidden_states) != len(gather_rows):
        raise BackendIntegrationError("Batched dense restore list shape mismatch.")
    return [
        [row[index] for index in indices]
        for row, indices in zip(hidden_states, gather_rows)
    ]


def _tome_r_for_layer(config: Mapping[str, Any], layer_index: int) -> int:
    selected_layers = config.get("layers")
    if selected_layers is not None:
        if isinstance(selected_layers, (str, bytes)) or layer_index not in {
            int(item) for item in selected_layers
        }:
            return 0
    by_layer = config.get("r_by_layer")
    if isinstance(by_layer, Mapping):
        value = by_layer.get(layer_index, by_layer.get(str(layer_index), 0))
    else:
        value = config.get("r", 0)
    return _strict_non_negative_int(value, "r")


def _validate_image_batch_identity(
    image_index: int,
    image_sizes: tuple[tuple[int, int], ...],
    modalities: tuple[str, ...],
) -> None:
    if isinstance(image_index, bool) or not isinstance(image_index, int):
        raise BackendIntegrationError("image_index must be an integer.")
    if len(image_sizes) != len(modalities) or not image_sizes:
        raise BackendIntegrationError(
            "image_sizes and modalities must be non-empty and batch-aligned."
        )
    if image_index < 0 or image_index >= len(modalities):
        raise BackendIntegrationError("image_index is outside the media batch.")
    _require_image(modalities[image_index])
    for size in image_sizes:
        if (
            not isinstance(size, Sequence)
            or len(size) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in size
            )
        ):
            raise BackendIntegrationError(
                "Every image_sizes entry must be a positive (width, height) pair."
            )


def _validate_anyres_layout(
    layout: Mapping[str, Any],
    *,
    token_count: int,
    newline_indices: tuple[int, ...],
    image_index: int,
) -> None:
    if not isinstance(layout, Mapping):
        raise BackendIntegrationError("AnyRes layout metadata must be a mapping.")
    expected = {
        "packed_after_anyres": True,
        "pre_llm_splice": True,
        "image_index": image_index,
        "packed_token_count": token_count,
    }
    mismatches = {
        key: (value, layout.get(key))
        for key, value in expected.items()
        if layout.get(key) != value
    }
    if mismatches:
        raise BackendIntegrationError(
            f"AnyRes boundary metadata mismatch: {mismatches}."
        )
    if tuple(layout.get("newline_indices", ())) != newline_indices:
        raise BackendIntegrationError(
            "AnyRes layout newline_indices do not match the callback payload."
        )
    if len(set(newline_indices)) != len(newline_indices) or any(
        isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or index >= token_count
        for index in newline_indices
    ):
        raise BackendIntegrationError("image_newline indices are invalid.")
    crop_boundaries = layout.get("crop_boundaries")
    if not isinstance(crop_boundaries, Sequence) or not crop_boundaries:
        raise BackendIntegrationError(
            "AnyRes layout must declare ordered crop_boundaries."
        )
    previous_end = 0
    for boundary in crop_boundaries:
        if (
            not isinstance(boundary, Sequence)
            or len(boundary) != 2
            or any(
                isinstance(item, bool) or not isinstance(item, int) for item in boundary
            )
        ):
            raise BackendIntegrationError(
                "crop_boundaries entries must be (start, end)."
            )
        start, end = boundary
        if start < previous_end or start < 0 or end <= start or end > token_count:
            raise BackendIntegrationError(
                "crop_boundaries are not ordered valid slices."
            )
        previous_end = end


def _compact_crop_boundaries(
    delta: LayoutDelta,
    boundaries: Sequence[Any],
) -> tuple[tuple[int, int], ...]:
    """Map ordered non-empty AnyRes crop slices through one drop delta."""

    result: list[tuple[int, int]] = []
    for boundary in boundaries:
        start, end = boundary
        kept = tuple(
            _non_null(delta.old_to_new[index])
            for index in range(start, end)
            if delta.old_to_new[index] is not None
        )
        if not kept:
            raise BackendIntegrationError(
                "DivPrune cannot remove every token from an AnyRes crop boundary."
            )
        result.append((kept[0], kept[-1] + 1))
    return tuple(result)


def _crop_selection_packed_keep_indices(
    selected_per_crop: tuple[Any, ...],
    *,
    algorithm: str,
    patch_count: int,
    patches_per_side: int | None,
    grid_width: int | None,
    grid_height: int | None,
    patch_merge_type: str | None,
    image_size: tuple[int, int],
    source_count: int,
    newline_indices: tuple[int, ...],
    device: Any,
) -> Any:
    """Map crop-local discrete selections through pinned native AnyRes layout."""

    import torch

    if not selected_per_crop:
        raise BackendIntegrationError(f"{algorithm} requires at least one image crop.")
    merge_type = str(patch_merge_type or "flat")
    crop_count = len(selected_per_crop)
    crop_masks = torch.zeros(
        crop_count,
        patch_count,
        dtype=torch.bool,
        device=device,
    )
    for crop_index, selected in enumerate(selected_per_crop):
        if getattr(selected, "ndim", None) != 1:
            raise BackendIntegrationError(
                f"{algorithm} selected crop indices must be rank one."
            )
        crop_masks[crop_index, selected.to(device=device)] = True

    if merge_type == "flat":
        packed_mask = crop_masks.reshape(-1)
    elif merge_type.startswith("spatial"):
        if "maxpool2x2" in merge_type:
            raise BackendIntegrationError(
                f"{algorithm} cannot map discrete patch selections through native "
                "AnyRes max-pooling."
            )
        if patches_per_side is None or patches_per_side**2 != patch_count:
            raise BackendIntegrationError(
                f"{algorithm} AnyRes mapping requires the exact square crop patch grid."
            )
        if crop_count == 1:
            packed_mask = crop_masks[0]
            if "unpad" in merge_type:
                packed_mask = torch.cat(
                    (
                        packed_mask,
                        torch.ones(1, dtype=torch.bool, device=device),
                    )
                )
        else:
            if (
                grid_width is None
                or grid_height is None
                or grid_width * grid_height != crop_count - 1
            ):
                raise BackendIntegrationError(
                    f"{algorithm} AnyRes tile count does not match the resolved image grid."
                )
            side = patches_per_side
            tile_mask = crop_masks[1:].view(
                grid_height,
                grid_width,
                side,
                side,
            )
            tile_mask = (
                tile_mask.permute(0, 2, 1, 3)
                .contiguous()
                .view(grid_height * side, grid_width * side)
            )
            if "unpad" in merge_type:
                tile_mask = _crop_selection_unpad_mask(
                    tile_mask, image_size=image_size
                )
                tile_mask = torch.cat(
                    (
                        tile_mask,
                        torch.ones(
                            int(tile_mask.shape[0]),
                            1,
                            dtype=torch.bool,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
            spatial_mask = tile_mask.reshape(-1)
            packed_mask = (
                spatial_mask
                if "nobase" in merge_type
                else torch.cat((crop_masks[0], spatial_mask))
            )
    else:
        raise BackendIntegrationError(
            f"{algorithm} does not support LLaVA patch merge type {merge_type!r}."
        )

    if int(packed_mask.numel()) != source_count:
        raise BackendIntegrationError(
            f"{algorithm} discrete AnyRes mapping does not match the native packed "
            "sequence. Continuous anyres_max resampling is not a legal drop map."
        )
    source_newline_mask = torch.zeros(
        source_count, dtype=torch.bool, device=device
    )
    if newline_indices:
        source_newline_mask[
            torch.tensor(newline_indices, dtype=torch.long, device=device)
        ] = True
    computed_newlines = tuple(
        int(value)
        for value in (packed_mask & source_newline_mask)
        .nonzero(as_tuple=False)
        .flatten()
        .tolist()
    )
    if computed_newlines != newline_indices:
        raise InvariantError(
            f"{algorithm} AnyRes mapping did not retain every image newline."
        )
    return packed_mask.nonzero(as_tuple=False).flatten()


def _crop_selection_unpad_mask(mask: Any, *, image_size: tuple[int, int]) -> Any:
    original_width, original_height = image_size
    current_height, current_width = (int(value) for value in mask.shape)
    original_ratio = original_width / original_height
    current_ratio = current_width / current_height
    if original_ratio > current_ratio:
        new_height = int(original_height * (current_width / original_width))
        padding = (current_height - new_height) // 2
        return mask[padding : current_height - padding, :]
    new_width = int(original_width * (current_height / original_height))
    padding = (current_width - new_width) // 2
    return mask[:, padding : current_width - padding]


def _crop_source_ids_in_packed_layout(
    *,
    crop_count: int,
    patch_count: int,
    patches_per_side: int,
    grid_width: int | None,
    grid_height: int | None,
    patch_merge_type: str | None,
    image_size: tuple[int, int],
    source_count: int,
    newline_indices: tuple[int, ...],
    device: Any,
) -> Any:
    """Map each packed AnyRes token to its dense crop-local source ID."""

    import torch

    source_ids = torch.arange(
        crop_count * patch_count,
        dtype=torch.long,
        device=device,
    ).view(crop_count, patch_count)
    merge_type = str(patch_merge_type or "flat")
    if merge_type == "flat":
        packed = source_ids.reshape(-1)
    elif merge_type.startswith("spatial"):
        if "maxpool2x2" in merge_type:
            raise BackendIntegrationError(
                "Packed visual compression cannot map representatives through AnyRes "
                "max-pooling."
            )
        if patches_per_side**2 != patch_count:
            raise BackendIntegrationError(
                "Nüwa AnyRes mapping requires the exact square crop patch grid."
            )
        if crop_count == 1:
            packed = source_ids[0]
            if "unpad" in merge_type:
                packed = torch.cat((packed, packed.new_tensor([-1])))
        else:
            if (
                grid_width is None
                or grid_height is None
                or grid_width * grid_height != crop_count - 1
            ):
                raise BackendIntegrationError(
                    "Nüwa AnyRes tile count does not match the resolved image grid."
                )
            side = patches_per_side
            tiled = source_ids[1:].view(grid_height, grid_width, side, side)
            tiled = (
                tiled.permute(0, 2, 1, 3)
                .contiguous()
                .view(grid_height * side, grid_width * side)
            )
            if "unpad" in merge_type:
                tiled = _crop_selection_unpad_mask(tiled, image_size=image_size)
                tiled = torch.cat(
                    (
                        tiled,
                        torch.full(
                            (int(tiled.shape[0]), 1),
                            -1,
                            dtype=torch.long,
                            device=device,
                        ),
                    ),
                    dim=1,
                )
            spatial = tiled.reshape(-1)
            packed = (
                spatial
                if "nobase" in merge_type
                else torch.cat((source_ids[0], spatial))
            )
    else:
        raise BackendIntegrationError(
            f"Nüwa does not support LLaVA patch merge type {merge_type!r}."
        )
    if int(packed.numel()) != source_count:
        raise BackendIntegrationError(
            "Nüwa AnyRes source mapping does not match the packed image sequence."
        )
    mapped_newlines = tuple(
        int(value)
        for value in (packed < 0).nonzero(as_tuple=False).flatten().tolist()
    )
    if mapped_newlines != newline_indices:
        raise InvariantError(
            "Nüwa AnyRes source mapping disagrees with image_newline positions."
        )
    return packed


def _validate_decoder_repair(
    repair: LlavaDecoderLayoutRepair,
    delta: LayoutDelta,
    *,
    source_newline_indices: tuple[int, ...],
    kept_source_image_indices: Any,
) -> None:
    if repair.sequence_length != delta.result_count:
        raise BackendIntegrationError(
            "Decoder layout repair sequence_length does not match LayoutDelta."
        )
    if len(set(repair.newline_indices)) != len(repair.newline_indices) or any(
        index < 0 or index >= repair.sequence_length for index in repair.newline_indices
    ):
        raise BackendIntegrationError(
            "Decoder repair returned invalid newline indices."
        )
    expected_newlines = tuple(
        _non_null(delta.old_to_new[index]) for index in source_newline_indices
    )
    if repair.newline_indices != expected_newlines:
        raise BackendIntegrationError(
            "Decoder repair newline indices do not match LayoutDelta."
        )
    repaired_images = _index_values(repair.image_indices)
    kept_images = _index_values(kept_source_image_indices)
    if repaired_images is None or kept_images is None:
        raise BackendIntegrationError(
            "Decoder repair returned unreadable image_indices."
        )
    expected_images = tuple(_non_null(delta.old_to_new[index]) for index in kept_images)
    if repaired_images != expected_images:
        raise BackendIntegrationError(
            "Decoder repair image indices do not match kept image LayoutDelta entries."
        )
    for field_name in ("attention_mask", "position_ids"):
        value = getattr(repair, field_name)
        if value is None:
            raise BackendIntegrationError(f"Decoder repair must return {field_name}.")
        shape = getattr(value, "shape", None)
        if shape is not None and int(shape[-1]) != repair.sequence_length:
            raise BackendIntegrationError(
                f"Decoder repair {field_name} sequence axis does not match LayoutDelta."
            )
        if (
            shape is None
            and isinstance(value, Sequence)
            and not isinstance(value, (str, bytes))
            and len(value) != repair.sequence_length
        ):
            raise BackendIntegrationError(
                f"Decoder repair {field_name} length does not match LayoutDelta."
            )


def _index_values(value: Any) -> tuple[int, ...] | None:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return None
    return tuple(int(item) for item in raw)


def _non_null(value: int | None) -> int:
    if value is None:
        raise BackendIntegrationError("A protected/layout-preserved token was dropped.")
    return int(value)


def _packed_reduction_response(
    activation: _LlavaBridgeActivation,
    request: LlavaPackedImageRequest,
    result: Any,
    *,
    algorithm: str,
) -> LlavaPackedImageResponse:
    _require_invocation_id(request.invocation_id)
    _validate_image_batch_identity(
        request.image_index,
        request.image_sizes,
        request.modalities,
    )
    token_count = _single_sequence_token_count(request.tokens)
    _validate_anyres_layout(
        request.anyres_layout,
        token_count=token_count,
        newline_indices=request.newline_indices,
        image_index=request.image_index,
    )
    delta = result.layout_delta
    target_site = activation.projector_target_sites[0] if algorithm in {"aim", "saint"} else VISIONZIP_SITE
    repaired_newlines = tuple(
        _non_null(delta.old_to_new[index]) for index in request.newline_indices
    )
    repaired_layout = {
        **dict(request.anyres_layout),
        "packed_token_count": delta.result_count,
        "newline_indices": repaired_newlines,
        "crop_boundaries": ((0, delta.result_count),) if algorithm == "aim" else _compact_crop_boundaries(
            delta, request.anyres_layout["crop_boundaries"]
        ),
        "compression_site": target_site,
        "layout_operation": delta.operation,
        "algorithm": algorithm,
    }
    if algorithm in {"aim", "saint"}:
        from dataclasses import replace
        repaired_layout = {**dict(delta.result_layout), **repaired_layout}
        delta = replace(delta, result_layout=repaired_layout)
    _validate_anyres_layout(
        repaired_layout,
        token_count=delta.result_count,
        newline_indices=repaired_newlines,
        image_index=request.image_index,
    )
    activation._last_layout = {
        "invocation_id": request.invocation_id,
        "target_site": target_site,
        "image_index": request.image_index,
        "source_count": delta.source_count,
        "result_count": delta.result_count,
        "newline_indices_before": list(request.newline_indices),
        "newline_indices_after": list(repaired_newlines),
        "image_sizes_preserved": True,
        "modalities_preserved": True,
    }
    activation._latest_layout_deltas = [_serialize_layout_delta(delta)]
    activation._record_layout_delta(
        delta,
        site=target_site,
        image_index=request.image_index,
    )
    return LlavaPackedImageResponse(
        tokens=result.tokens,
        layout_delta=delta,
        image_sizes=request.image_sizes,
        modalities=request.modalities,
        newline_indices=repaired_newlines,
        anyres_layout=MappingProxyType(repaired_layout),
        image_index=request.image_index,
        **({"visual_sizes": result.token_sizes,
            "representative_indices": tuple(int(i) for i in result.representative_indices.tolist())} if algorithm == "aim" else {}),
        metadata={
            **dict(result.metadata),
            **({"token_sizes": result.token_sizes.tolist(),
                "representative_indices": result.representative_indices.tolist()} if algorithm == "aim" else {}),
            "post_anyres_pack": True,
            "pre_llm_splice": True,
            "image_newline_protected": True,
            "image_sizes_preserved": True,
            "modalities_preserved": True,
        },
    )


def _strict_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise BackendIntegrationError(f"{name} must be a positive integer.")
    return value


def _strict_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise BackendIntegrationError(f"{name} must be a non-negative integer.")
    return value


def _serialize_layout_delta(delta: LayoutDelta) -> dict[str, Any]:
    """Return a JSON-safe, lossless-enough runtime audit record."""

    return {
        "source_count": delta.source_count,
        "result_count": delta.result_count,
        "operation": delta.operation,
        "media_unit": delta.media_unit,
        "ordering_policy": delta.ordering_policy.value,
        "kept_indices": list(delta.kept_indices),
        "dropped_indices": list(delta.dropped_indices),
        "merge_groups": [list(group) for group in delta.merge_groups],
        "old_to_new": [delta.old_to_new[index] for index in range(delta.source_count)],
        "protected_indices": list(delta.protected_indices),
        "coordinates": _json_safe(delta.coordinates),
        "source_layout": _json_safe(dict(delta.source_layout)),
        "result_layout": _json_safe(dict(delta.result_layout)),
    }


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_json_safe(item) for item in value]
    return repr(value)


__all__ = [
    "LLAVA_ONEVISION_BRIDGE_API_VERSION",
    "LLAVA_ONEVISION_IMAGE_BINDING_CONTRACTS",
    "LLAVA_ONEVISION_SOURCE_COMMIT",
    "LLAVA_ONEVISION_SOURCE_ROUTE",
    "FASTV_SITE",
    "PYRAMID_DROP_SITE",
    "RESTORE_CALIBRATION_SITE",
    "TOME_MERGE_SITE",
    "TOME_PACK_SITE",
    "TOME_RESTORE_SITE",
    "VISIONZIP_SITE",
    "LlavaCallbackContract",
    "LlavaDecoderLayerRequest",
    "LlavaDecoderLayerResponse",
    "LlavaDecoderLayoutRepair",
    "LlavaPACTRequest",
    "LlavaFastVRequest",
    "LlavaEncoderDenseRestoreRequest",
    "LlavaEncoderDenseRestoreResponse",
    "LlavaEncoderMergeRequest",
    "LlavaEncoderMergeResponse",
    "LlavaImageBindingContract",
    "LlavaOneVisionCompressionBridge",
    "LlavaOneVisionImageCompressionAdapter",
    "LlavaOneVisionFastVActivation",
    "LlavaOneVisionFEATHERActivation",
    "LlavaOneVisionIVCPruneActivation",
    "LlavaOneVisionSGLActivation",
    "LlavaOneVisionDARTActivation",
    "LlavaOneVisionD2PrunerActivation",
    "LlavaOneVisionSparseVLMActivation",
    "LlavaOneVisionFinePrunerActivation",
    "LlavaOneVisionBTPActivation",
    "LlavaOneVisionHalfVActivation",
    "LlavaOneVisionPyramidDropActivation",
    "LlavaOneVisionPACTActivation",
    "LlavaOneVisionSAINTActivation",
    "LlavaOneVisionAIMActivation",
    "LlavaOneVisionDivPruneActivation",
    "LlavaOneVisionMMTokActivation",
    "LlavaOneVisionVisPrunerActivation",
    "LlavaOneVisionZOOPruneActivation",
    "LlavaOneVisionNuwaActivation",
    "LlavaOneVisionRestoreActivation",
    "LlavaOneVisionToMeActivation",
    "LlavaOneVisionVisionZipActivation",
    "LlavaOneVisionPruMergeActivation",
    "LlavaOneVisionGAPActivation",
    "LlavaPackedImageRequest",
    "LlavaPackedImageResponse",
    "create_llava_onevision_image_compression_executor",
    "create_llava_onevision_divprune_executor",
    "create_llava_onevision_saint_executor",
    "create_llava_onevision_aim_executor",
    "create_llava_onevision_mmtok_executor",
    "create_llava_onevision_vispruner_executor",
    "create_llava_onevision_zoo_prune_executor",
    "create_llava_onevision_nuwa_executor",
    "create_llava_onevision_restore_executor",
    "create_llava_onevision_fastv_executor",
    "create_llava_onevision_feather_executor",
    "create_llava_onevision_ivc_prune_executor",
    "create_llava_onevision_sgl_executor",
    "create_llava_onevision_dart_executor",
    "create_llava_onevision_d2pruner_executor",
    "create_llava_onevision_sparsevlm_executor",
    "create_llava_onevision_finepruner_executor",
    "create_llava_onevision_btp_executor",
    "create_llava_onevision_halfv_executor",
    "create_llava_onevision_transprune_executor",
    "create_llava_onevision_pyramid_drop_executor",
    "create_llava_onevision_pact_executor",
    "create_llava_onevision_tome_executor",
    "create_llava_onevision_visionzip_executor",
    "create_llava_onevision_prumerge_executor",
    "create_llava_onevision_gap_executor",
]
