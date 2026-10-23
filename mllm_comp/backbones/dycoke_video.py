"""Video Projector selection plus sparse decode KV views on three backbones.

The selected method supplies Projector values, branch replay, and a per-cache
decode policy. DyCoKe is the default method for direct construction.
Generate activations additionally patch one request-local ``DynamicCache``: its
physical layers stay complete, the layer immediately before the configured
first sparse layer supplies the native query and full keys to the selected
policy, and subsequent layers receive its sparse KV view (default source 2,
sparse layers 3+). Forward activations never install the decode controller;
training uses differentiable Projector compression and the native loss path.
"""

from __future__ import annotations

import inspect
import math
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Sequence
from weakref import WeakSet

from mllm_comp.backbones._video_bridge import (
    install_video_bridge,
    register_video_callback,
    release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.video import (
    DyCoKeKVViewPlan,
    DyCoKeVideoKernel,
    VideoCompressionResult,
)
from mllm_comp.compressors.video.dycoke import DyCoKeDecodePolicy
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.compressors.video.positions import (
    video_decode_prune_layer, video_projector_position, video_projector_site,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import CompressionMethod, ExecutionPhase, Modality, Placement


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_CACHE_CONTROLLER_ATTRIBUTE = "_mllm_comp_dycoke_cache_controller"


@dataclass(frozen=True)
class _PendingRequest:
    request: Any
    result: VideoCompressionResult
    response: Any | None
    visual_positions: tuple[int, ...]


class DyCoKeDecodeCacheController:
    """Return sparse decode views without compacting the physical cache."""

    def __init__(
        self,
        *,
        owner: "DyCoKeVideoExecutor",
        cache: Any,
        visual_positions: Sequence[int],
        layer_count: int,
        prune_layer: int,
        drop_ratio: float = 0.8,
        similarity_threshold: float = 0.9,
        attention_scaling: float,
        decode_policy: Any | None = None,
    ) -> None:
        self.owner = owner
        self.cache = cache
        self.visual_positions = tuple(visual_positions)
        self.layer_count = layer_count
        self.prune_layer = prune_layer
        self.trigger_layer = prune_layer - 1
        self.attention_scaling = attention_scaling
        self._decode_policy = decode_policy if decode_policy is not None else DyCoKeDecodePolicy(
            drop_ratio=drop_ratio, similarity_threshold=similarity_threshold,
            rules=getattr(owner, "_rules", {}),
        )
        self._patch: MethodPatch | None = None
        self._prompt_length: int | None = None
        self._prefill_layers: set[int] = set(range(self.trigger_layer))
        self._prefill_complete = False
        self._prepared_query: Any | None = None
        self._current_plan: DyCoKeKVViewPlan | None = None

        _validate_dynamic_cache(cache, layer_count=layer_count)
        if not 0 <= self.trigger_layer < self.prune_layer < layer_count:
            raise BackendIntegrationError(
                "DyCoKe decode layers are outside the native decoder"
            )
        if not self.visual_positions:
            raise BackendIntegrationError("DyCoKe decode requires visual positions")

    @classmethod
    def attach(
        cls,
        *,
        owner: "DyCoKeVideoExecutor",
        cache: Any,
        visual_positions: Sequence[int],
        layer_count: int,
        prune_layer: int,
        drop_ratio: float = 0.8,
        similarity_threshold: float = 0.9,
        attention_scaling: float,
        decode_policy: Any | None = None,
    ) -> "DyCoKeDecodeCacheController":
        existing = getattr(cache, _CACHE_CONTROLLER_ATTRIBUTE, None)
        if existing is not None:
            if not isinstance(existing, cls) or existing.owner is not owner:
                raise BackendIntegrationError(
                    "DynamicCache already belongs to another DyCoKe activation"
                )
            if existing.visual_positions != tuple(visual_positions):
                raise BackendIntegrationError(
                    "DyCoKe cache visual positions changed after prefill"
                )
            return existing
        controller = cls(
            owner=owner,
            cache=cache,
            visual_positions=visual_positions,
            layer_count=layer_count,
            prune_layer=prune_layer,
            drop_ratio=drop_ratio,
            similarity_threshold=similarity_threshold,
            attention_scaling=attention_scaling,
            decode_policy=decode_policy,
        )
        original = getattr(cache, "update", None)
        if not callable(original):
            raise BackendIntegrationError("DyCoKe DynamicCache lacks update()")
        try:
            controller._patch = MethodPatch.apply(
                cache,
                "update",
                controller._update_replacement(original),
                original,
                label="DyCoKe",
            )
            setattr(cache, _CACHE_CONTROLLER_ATTRIBUTE, controller)
        except BaseException:
            if controller._patch is not None:
                controller._patch.restore()
                controller._patch = None
            raise
        return controller

    @property
    def current_plan(self) -> DyCoKeKVViewPlan | None:
        return self._current_plan

    def prepare_decode_query(self, query: Any) -> None:
        torch = _torch()
        if (
            not isinstance(query, torch.Tensor)
            or query.ndim != 4
            or int(query.shape[0]) != 1
            or int(query.shape[2]) != 1
            or not query.is_floating_point()
            or not bool(torch.isfinite(query).all().item())
        ):
            raise BackendIntegrationError(
                "DyCoKe decode query must be finite [1, heads, 1, head_dim]"
            )
        if self._prepared_query is not None:
            raise InvariantError(f"DyCoKe received two layer-{self.trigger_layer} queries for one step")
        self._prepared_query = query.detach()

    def require_sparse_view_ready(self, layer_index: int) -> None:
        if layer_index < self.prune_layer:
            return
        plan = self._current_plan
        if plan is None:
            raise InvariantError(f"DyCoKe sparse layer ran before layer-{self.trigger_layer} attention")
        prior_length = _physical_layer_length(self.cache, layer_index)
        if prior_length is None or prior_length + 1 != plan.sequence_length:
            raise InvariantError(
                "DyCoKe sparse layer physical cache is not aligned to this decode step"
            )

    def restore(self) -> None:
        errors: list[str] = []
        if getattr(self.cache, _CACHE_CONTROLLER_ATTRIBUTE, None) is not self:
            errors.append("cache-controller ownership changed while active")
        else:
            delattr(self.cache, _CACHE_CONTROLLER_ATTRIBUTE)
        if self._patch is not None:
            try:
                self._patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._patch = None
        self._prepared_query = None
        if errors:
            raise BackendIntegrationError(
                f"DyCoKe cache-controller uninstall failed: {errors}"
            )

    def _update_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def replacement(
            _cache: Any,
            key_states: Any,
            value_states: Any,
            layer_idx: int,
            cache_kwargs: Mapping[str, Any] | None = None,
        ) -> tuple[Any, Any]:
            return self._update(
                original,
                key_states,
                value_states,
                layer_idx,
                cache_kwargs,
            )

        return replacement

    def _update(
        self,
        original: Callable[..., Any],
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: Mapping[str, Any] | None,
    ) -> tuple[Any, Any]:
        torch = _torch()
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise BackendIntegrationError("DyCoKe cache layer index must be integer")
        if not 0 <= layer_idx < self.layer_count:
            raise BackendIntegrationError("DyCoKe cache layer index is out of range")
        if (
            not isinstance(key_states, torch.Tensor)
            or not isinstance(value_states, torch.Tensor)
            or key_states.ndim != 4
            or tuple(key_states.shape) != tuple(value_states.shape)
            or int(key_states.shape[0]) != 1
        ):
            raise BackendIntegrationError(
                "DyCoKe requires batch-one rank-four native KV tensors"
            )
        query_length = int(key_states.shape[-2])
        if query_length < 1:
            raise BackendIntegrationError("DyCoKe cache update cannot be empty")
        if self._prefill_complete and query_length != 1:
            raise BackendIntegrationError(
                "DyCoKe supports single-token decode after one prefill only"
            )

        full_keys, full_values = original(
            key_states,
            value_states,
            layer_idx,
            cache_kwargs,
        )
        if (
            not isinstance(full_keys, torch.Tensor)
            or not isinstance(full_values, torch.Tensor)
            or full_keys.ndim != 4
            or tuple(full_keys.shape) != tuple(full_values.shape)
            or int(full_keys.shape[0]) != 1
        ):
            raise BackendIntegrationError(
                "DyCoKe native DynamicCache returned incompatible KV tensors"
            )
        sequence_length = int(full_keys.shape[-2])
        _require_physical_cache_storage(
            self.cache,
            layer_idx,
            returned_keys=full_keys,
            returned_values=full_values,
        )

        if query_length > 1:
            self._record_prefill_layer(layer_idx, sequence_length)
            return full_keys, full_values
        if not self._prefill_complete or self._prompt_length is None:
            raise BackendIntegrationError(
                "DyCoKe decode began before every cache layer completed prefill"
            )

        if layer_idx == self.trigger_layer:
            query = self._prepared_query
            self._prepared_query = None
            if query is None:
                raise InvariantError(f"DyCoKe layer-{self.trigger_layer} cache update missed its query")
            self._current_plan = self._plan_view(
                query,
                full_keys,
                sequence_length=sequence_length,
            )
            self.owner._record_decode_view(self._current_plan)
            return full_keys, full_values
        if layer_idx >= self.prune_layer:
            plan = self._current_plan
            if plan is None or plan.sequence_length != sequence_length:
                raise InvariantError(
                    "DyCoKe sparse cache view is stale or missing for this layer"
                )
            indices = torch.tensor(
                plan.view_indices,
                dtype=torch.long,
                device=full_keys.device,
            )
            return (
                full_keys.index_select(-2, indices),
                full_values.index_select(-2, indices.to(full_values.device)),
            )
        return full_keys, full_values

    def _record_prefill_layer(self, layer_idx: int, sequence_length: int) -> None:
        if layer_idx in self._prefill_layers:
            raise BackendIntegrationError(
                "DyCoKe observed a repeated multi-token cache update"
            )
        if self._prompt_length is None:
            if layer_idx != self.trigger_layer:
                raise InvariantError(f"DyCoKe cache controller did not attach at trigger layer {self.trigger_layer}")
            self._prompt_length = sequence_length
            if self.visual_positions[-1] >= sequence_length:
                raise InvariantError(
                    "DyCoKe visual positions exceed the compressed prompt cache"
                )
            for earlier in range(self.trigger_layer):
                if _physical_layer_length(self.cache, earlier) != sequence_length:
                    raise InvariantError(
                        "DyCoKe earlier cache layers differ from the prompt length"
                    )
        elif sequence_length != self._prompt_length:
            raise InvariantError("DyCoKe prefill cache layers changed sequence length")
        self._prefill_layers.add(layer_idx)
        if layer_idx == self.layer_count - 1:
            expected = set(range(self.layer_count))
            if self._prefill_layers != expected:
                raise InvariantError("DyCoKe prefill did not visit every cache layer")
            self._prefill_complete = True

    def _plan_view(
        self, query: Any, full_keys: Any, *, sequence_length: int,
    ) -> DyCoKeKVViewPlan:
        plan = self._decode_policy.plan_view(
            query, full_keys, sequence_length=sequence_length,
            visual_positions=self.visual_positions, attention_scaling=self.attention_scaling,
        )
        if (not isinstance(plan, DyCoKeKVViewPlan)
                or plan.sequence_length != sequence_length
                or plan.visual_positions != self.visual_positions):
            raise InvariantError("Decode policy returned a view for another cache layout")
        return plan


class DyCoKeVideoExecutor:
    """Run one method through Projector selection and full-storage KV views.

    A zero-argument binding strategy factory returns a callable accepting
    ``(payload, config=..., rules=...)`` plus ``replay`` with the same keywords,
    ``new_decode_policy(config=..., rules=...)`` and ``runtime_metadata()``.
    Each policy implements ``plan_view(query, full_keys, sequence_length=...,
    visual_positions=..., attention_scaling=...) -> DyCoKeKVViewPlan``.
    The policy owns algorithm history; this executor owns native cache lifetime.
    ``config.decoder_prune_layer`` supplies the default first sparse layer;
    scoped ``operation.targets`` may select another supported native position.
    """

    algorithm = "dycoke"
    variant = "dycoke.video"

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        runtime: Any | None = None,
        facade: Any | None = None,
        stage: Any | None = None,
        binding: Any | None = None,
        binding_config: Any | None = None,
        contract_projection: Any | None = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown DyCoKe video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.runtime = runtime
        self.facade = facade
        self.stage = stage
        self.binding = binding
        self.binding_config = binding_config
        self.execution_phase = ExecutionPhase(execution_phase)
        _validate_binding(binding, stage, backbone, self.execution_phase, contract_projection)
        if contract_projection is not None:
            self.algorithm = contract_projection.algorithm.name
            self.variant = contract_projection.variant.name
        self.config = (
            contract_projection.algorithm_config
            if contract_projection is not None
            else _validated_config(stage, binding_config)
        )
        self.projector_position = video_projector_position(stage, backbone, target_name="ttm_visual_merge")
        self.prune_layer = video_decode_prune_layer(stage, int(self.config["decoder_prune_layer"]), backbone=backbone)
        factory = binding.strategy_factory if binding is not None else DyCoKeVideoKernel
        if not callable(factory):
            raise BackendIntegrationError("Video Projector/KV binding requires a strategy factory")
        self.kernel = factory()
        if not callable(self.kernel) or any(
            not callable(getattr(self.kernel, name, None))
            for name in ("replay", "new_decode_policy", "runtime_metadata")
        ):
            raise BackendIntegrationError(
                "Video Projector/KV method requires callable compression, replay, "
                "new_decode_policy and runtime_metadata"
            )
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=contract_projection)

        self._pending: ContextVar[_PendingRequest | None] = ContextVar(
            f"dycoke_video_pending_{backbone}_{id(self)}",
            default=None,
        )
        self._bridge: Any | None = None
        self._bridge_handle: Any | None = None
        self._owns_bridge = False
        self._patches: list[MethodPatch] = []
        # A returned cache owns its controller. The activation only tracks live
        # requests for uninstall, without retaining completed requests' KV.
        self._cache_controllers: WeakSet[DyCoKeDecodeCacheController] = WeakSet()
        self._installed = False
        self._projector_count = 0
        self._decode_count = 0
        self._layout_records: list[dict[str, Any]] = []
        self._algorithm_records: list[dict[str, Any]] = []
        self._decode_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        patches: list[MethodPatch] = []
        bridge = None
        handle = None
        owns_bridge = False
        try:
            decoder = _decoder_owner(self.model, self.backbone)
            layers = tuple(getattr(decoder, "layers", ()))
            self.prune_layer = video_decode_prune_layer(
                self.stage, int(self.config["decoder_prune_layer"]), num_layers=len(layers), backbone=self.backbone,
            )
            native_model_forward = getattr(self.model, "forward", None)
            if not callable(native_model_forward):
                raise BackendIntegrationError("DyCoKe model lacks forward()")
            patches.append(
                MethodPatch.apply(
                    self.model,
                    "forward",
                    self._model_forward_replacement(native_model_forward),
                    native_model_forward,
                    label="DyCoKe",
                )
            )
            for layer_index in range(self.prune_layer - 1, len(layers)):
                layer = layers[layer_index]
                native_forward = getattr(layer, "forward", None)
                if not callable(native_forward):
                    raise BackendIntegrationError(
                        f"DyCoKe decoder layer {layer_index} lacks forward()"
                    )
                patches.append(
                    MethodPatch.apply(
                        layer,
                        "forward",
                        self._layer_forward_replacement(
                            layer_index,
                            native_forward,
                            layer_count=len(layers),
                        ),
                        native_forward,
                        label="DyCoKe",
                    )
                )
            bridge, owns_bridge, projector_site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(
                bridge,
                projector_site,
                self._projector_callback,
                label="DyCoKe",
                **(
                    {"projector_position": "before"}
                    if self.projector_position == "before"
                    else {}
                ),
            )
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            for patch in reversed(patches):
                try:
                    patch.restore()
                except Exception:
                    pass
            raise
        self._bridge = bridge
        self._bridge_handle = handle
        self._owns_bridge = owns_bridge
        self._patches = patches
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = release_video_callback(
            self._bridge, self._bridge_handle, self._owns_bridge
        )
        for controller in tuple(self._cache_controllers):
            try:
                controller.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        for patch in reversed(self._patches):
            try:
                patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = None
        self._bridge = None
        self._owns_bridge = False
        self._cache_controllers.clear()
        self._patches.clear()
        self._pending.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(f"DyCoKe video uninstall failed: {errors}")

    def runtime_metadata(self) -> dict[str, Any]:
        binding_metadata = getattr(self.binding, "metadata", {})
        if not isinstance(binding_metadata, Mapping):
            binding_metadata = {}
        verified_phases = tuple(binding_metadata.get("verified_phases", ()))
        verified = (self.execution_phase.value in verified_phases
            and self.projector_position == "after"
            and self.prune_layer == int(self.config["decoder_prune_layer"]))
        bridge_metadata = (
            self._bridge.runtime_metadata()
            if self._bridge is not None
            and callable(getattr(self._bridge, "runtime_metadata", None))
            else None
        )
        return {
            "executor": f"{self.backbone}_{self.algorithm}_video",
            "algorithm": self.algorithm,
            "variant": self.variant,
            "backbone": self.backbone,
            "binding": getattr(self.binding, "name", None),
            "execution_phase": self.execution_phase.value,
            "installed": self._installed,
            "canonical_config": dict(self.config),
            "projector_position": self.projector_position,
            "target_sites": [video_projector_site(self.backbone, self.projector_position), "decode_kv_before_attention"],
            "position_changed": self.projector_position != "after" or self.prune_layer != int(self.config["decoder_prune_layer"]),
            "canonical_evidence_applicable": (self.binding is None or bool(binding_metadata.get("canonical_config"))) and self.projector_position == "after" and self.prune_layer == int(self.config["decoder_prune_layer"]),
            "application_positions": [{"boundary": "decode_kv_before_attention", "index": self.prune_layer}],
            "projector_invocations": self._projector_count,
            "decode_view_invocations": self._decode_count,
            "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._layout_records),
            "algorithm_records": list(self._algorithm_records),
            "decode_kv_views": list(self._decode_records),
            "live_cache_controllers": len(self._cache_controllers),
            "bridge": bridge_metadata,
            "physical_cache_retained": True,
            "decode_trigger_layer": self.prune_layer - 1,
            "sparse_view_first_layer": self.prune_layer,
            "single_token_decode_only": True,
            "training_gradients_preserved": True,
            "real_checkpoint_verified": verified,
            "support_activation_claimed": verified,
            "evidence_artifact": (
                binding_metadata.get("generate_evidence_artifact") if verified else None
            ),
            **self.kernel.runtime_metadata(),
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard only diagnostic history; keep counters and replay state."""

        self._layout_records.clear()
        self._algorithm_records.clear()
        self._decode_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        if self._pending.get() is not None:
            raise BackendIntegrationError(
                "Nested DyCoKe video requests are unsupported"
            )
        _require_full_visibility(request)
        result = self.kernel(request.payload, config=self.config, rules=self._rules)
        if not isinstance(result, VideoCompressionResult):
            raise InvariantError("Video Projector/KV method must return VideoCompressionResult")
        result = replace(result, metadata={
            **result.metadata,
            "projector_position": self.projector_position,
            "target_site": video_projector_site(self.backbone, self.projector_position),
            "decode_stage": f"full_cache_sparse_view_from_layer_{self.prune_layer}",
            "attention_source_layer": self.prune_layer - 1,
            "first_sparse_layer": self.prune_layer,
        })
        response = None
        deepstack_record: dict[str, Any] | None = None
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import (
                Qwen3VLVideoCompressionResponse,
            )

            branches = tuple(
                self.kernel.replay(branch, result, config=self.config, rules=self._rules)
                for branch in request.deepstack_features
            )
            response = Qwen3VLVideoCompressionResponse.from_result(
                request,
                result,
                deepstack_features=branches,
            )
            if branches and result.features.shape[1] != request.payload.features.shape[1]:
                raise InvariantError("Qwen3-VL DyCoKe replay changed the hidden width")
            source_shapes = tuple(
                tuple(int(item) for item in branch.shape)
                for branch in request.deepstack_features
            )
            result_shapes = tuple(
                tuple(int(item) for item in branch.shape) for branch in branches
            )
            deepstack_record = {
                "deepstack_branch_count": len(branches),
                "deepstack_source_shapes": source_shapes,
                "deepstack_result_shapes": result_shapes,
            }
            returned = response
        else:
            returned = result
        visual_positions = _result_visual_positions(
            self.backbone,
            request,
            result,
            response,
        )
        if self.execution_phase is ExecutionPhase.GENERATE:
            self._pending.set(
                _PendingRequest(request, result, response, visual_positions)
            )
        self._projector_count += 1
        record = layout_delta_record(result.layout_deltas[0])
        record.update(
            {
                "stage": "ttm_visual_merge",
                "site": video_projector_site(self.backbone, self.projector_position),
                "projector_position": self.projector_position,
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "result_visual_positions": visual_positions,
            }
        )
        self._layout_records.append(record)
        self._algorithm_records.append({
            **result.metadata,
            "source_count": result.source_count,
            "result_count": result.result_count,
            "result_visual_positions": visual_positions,
            **(deepstack_record or {}),
        })
        return returned

    def _model_forward_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        def replacement(_model: Any, *args: Any, **kwargs: Any) -> Any:
            try:
                return original(*args, **kwargs)
            finally:
                self._pending.set(None)

        return replacement

    def _layer_forward_replacement(
        self,
        layer_index: int,
        original: Callable[..., Any],
        *,
        layer_count: int,
    ) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def replacement(_layer: Any, *args: Any, **kwargs: Any) -> Any:
            values = _bound_values(signature, args, kwargs)
            hidden_states = values.get("hidden_states")
            cache = values.get("past_key_values", values.get("past_key_value"))
            pending = self._pending.get()
            controller = self._cache_controller(cache)
            if pending is None and controller is None:
                return original(*args, **kwargs)
            sequence_length = _hidden_sequence_length(hidden_states)

            if layer_index == self.prune_layer - 1:
                if pending is not None:
                    if sequence_length <= 1:
                        raise BackendIntegrationError(
                            "DyCoKe projector state reached a non-prefill decoder call"
                        )
                    if cache is None:
                        raise BackendIntegrationError(
                            "DyCoKe generate prefill requires native DynamicCache"
                        )
                    attention = getattr(_layer, "self_attn", None)
                    head_dim = int(getattr(attention, "head_dim", 0))
                    scaling = float(
                        getattr(
                            attention,
                            "scaling",
                            head_dim**-0.5 if head_dim > 0 else 0.0,
                        )
                    )
                    if not math.isfinite(scaling) or scaling <= 0:
                        raise BackendIntegrationError(
                            "DyCoKe could not resolve native attention scaling"
                        )
                    policy = self.kernel.new_decode_policy(config=self.config, rules=self._rules)
                    if not callable(getattr(policy, "plan_view", None)):
                        raise BackendIntegrationError("Decode policy must implement plan_view()")
                    controller = DyCoKeDecodeCacheController.attach(
                        owner=self,
                        cache=cache,
                        visual_positions=pending.visual_positions,
                        layer_count=layer_count,
                        prune_layer=self.prune_layer,
                        attention_scaling=scaling,
                        decode_policy=policy,
                    )
                    self._cache_controllers.add(controller)
                    self._pending.set(None)
                if controller is not None and sequence_length == 1:
                    controller.prepare_decode_query(
                        _decoder_query(
                            self.backbone,
                            _layer,
                            hidden_states,
                            values.get("position_embeddings"),
                        )
                    )

            if (
                controller is not None
                and layer_index >= self.prune_layer
                and sequence_length == 1
            ):
                controller.require_sparse_view_ready(layer_index)
                args, kwargs = _replace_bound_argument(
                    signature,
                    args,
                    kwargs,
                    "attention_mask",
                    None,
                )
            return original(*args, **kwargs)

        return replacement

    def _cache_controller(self, cache: Any) -> DyCoKeDecodeCacheController | None:
        if cache is None:
            return None
        controller = getattr(cache, _CACHE_CONTROLLER_ATTRIBUTE, None)
        if controller is None:
            return None
        if not isinstance(controller, DyCoKeDecodeCacheController):
            raise BackendIntegrationError(
                "DynamicCache has an unknown DyCoKe controller"
            )
        if controller.owner is not self:
            raise BackendIntegrationError(
                "DynamicCache belongs to another DyCoKe executor"
            )
        return controller

    def _record_decode_view(self, plan: DyCoKeKVViewPlan) -> None:
        self._decode_count += 1
        record = plan.as_dict()
        record.update(
            {
                "stage": "decode_kv_view_prune",
                "site": "decode_kv_before_attention",
                "attention_source_layer": self.prune_layer - 1,
                "first_sparse_layer": self.prune_layer,
            }
        )
        self._decode_records.append(record)


def create_dycoke_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> DyCoKeVideoExecutor:
    return DyCoKeVideoExecutor(backbone=backbone, **kwargs)


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("DyCoKe stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "DyCoKe video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("dycoke", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
    projection: Any | None,
) -> None:
    algorithm, variant = "dycoke", "dycoke.video"
    if projection is not None:
        algorithm, variant = projection.algorithm.name, projection.variant.name
        if (binding is None or binding.name not in projection.binding_names
                or projection.variant.algorithm != algorithm
                or projection.algorithm.method is not CompressionMethod.TOKEN_COMPRESSION
                or projection.variant.target_modalities != frozenset({Modality.VIDEO})):
            raise BackendIntegrationError("Video Projector/KV execution requires the selected video Token contract")
    if binding is not None:
        if binding.variant != variant or binding.backbone != backbone:
            raise BackendIntegrationError(
                "Video Projector/KV binding differs from the selected method and backbone"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"Video Projector/KV binding {binding.name!r} does not support {phase.value}"
            )
        targets = binding.ordered_targets()
        if (tuple(item.name for item in targets) != ("ttm_visual_merge", "decode_kv_view_prune")
                or tuple(item.placement for item in targets) != (Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL)
                or any(item.target_modalities != frozenset({Modality.VIDEO}) for item in targets)):
            raise BackendIntegrationError("Video Projector/KV execution requires ordered video Projector and decode-view targets")
    if stage is not None:
        if stage.algorithm != algorithm or stage.variant != variant:
            raise BackendIntegrationError("Video Projector/KV stage differs from the selected method")
    if phase not in {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}:
        raise BackendIntegrationError("Video Projector/KV execution supports forward or generate only")


def _decoder_owner(model: Any, backbone: str) -> Any:
    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _resolve_components,
        )

        return _resolve_components(model).core
    core = getattr(model, "model", None)
    owner = getattr(core, "language_model", None)
    if owner is None:
        raise BackendIntegrationError(
            f"{backbone} DyCoKe could not resolve language_model"
        )
    return owner


def _result_visual_positions(
    backbone: str,
    request: Any,
    result: VideoCompressionResult,
    response: Any | None,
) -> tuple[int, ...]:
    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.video_compression import (
            plan_llava_onevision_video_splice,
        )

        plan = plan_llava_onevision_video_splice(
            request,
            result,
            sequence_length=request.source_visual_span[1],
        )
        positions = tuple(
            plan.result_visual_span[0] + index
            for index, (_packed, kind, _value) in enumerate(plan.packed_items)
            if kind == "patch"
        )
    elif backbone == "qwen25_vl":
        from mllm_comp.backbones.qwen25_vl.video_compression import (
            plan_qwen25_vl_video_splice,
        )

        plan = plan_qwen25_vl_video_splice(
            request,
            result,
            sequence_length=request.source_placeholder_positions[-1] + 1,
        )
        positions = plan.result_placeholder_positions
    else:
        from mllm_comp.backbones.qwen3_vl.video_compression import (
            plan_qwen3_vl_video_splice,
        )

        if response is None:
            raise InvariantError("Qwen3-VL DyCoKe response is missing")
        plan = plan_qwen3_vl_video_splice(
            request,
            response,
            sequence_length=request.source_placeholder_positions[-1] + 1,
        )
        positions = plan.result_placeholder_positions
    if len(positions) != result.result_count:
        raise InvariantError("DyCoKe result visual positions changed token count")
    return tuple(positions)


def _require_full_visibility(request: Any) -> None:
    torch = _torch()
    mask = getattr(
        request,
        "prepared_attention_mask",
        getattr(request, "source_attention_mask", None),
    )
    if mask is None:
        return
    if (
        not isinstance(mask, torch.Tensor)
        or mask.ndim != 2
        or int(mask.shape[0]) != 1
        or not bool(mask.bool().all().item())
    ):
        raise BackendIntegrationError(
            "DyCoKe supports one unpadded fully visible video prompt only"
        )


def _decoder_query(
    backbone: str,
    decoder_layer: Any,
    hidden_states: Any,
    position_embeddings: Any,
) -> Any:
    torch = _torch()
    if _hidden_sequence_length(hidden_states) != 1:
        raise BackendIntegrationError("DyCoKe decoder query requires one token")
    if not isinstance(position_embeddings, tuple) or len(position_embeddings) != 2:
        raise BackendIntegrationError(
            "DyCoKe decoder query requires native rotary embeddings"
        )
    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _decoder_query_keys,
        )
    elif backbone == "qwen25_vl":
        from mllm_comp.backbones.qwen25_vl.provider_bridge import (
            _decoder_query_keys,
        )
    else:
        from mllm_comp.backbones.qwen3_vl.provider_bridge import (
            _decoder_query_keys,
        )
    local = torch.zeros(1, dtype=torch.long, device=hidden_states.device)
    with torch.no_grad():
        query, _visual_keys, _keys = _decoder_query_keys(
            decoder_layer,
            hidden_states,
            instruction_indices=local,
            image_indices=local,
            position_embeddings=position_embeddings,
        )
    if query.ndim != 3 or int(query.shape[0]) != 1:
        raise InvariantError("DyCoKe native decoder query changed layout")
    return query.permute(1, 0, 2).unsqueeze(0).contiguous()


def _validate_dynamic_cache(cache: Any, *, layer_count: int) -> None:
    if type(cache).__name__ != "DynamicCache":
        raise BackendIntegrationError(
            "DyCoKe requires transformers 4.57.3 DynamicCache exactly"
        )
    if bool(getattr(cache, "offloading", False)):
        raise BackendIntegrationError("DyCoKe does not support offloaded caches")
    layers = getattr(cache, "layers", None)
    if not isinstance(layers, Sequence) or len(layers) != layer_count:
        raise BackendIntegrationError(
            "DyCoKe DynamicCache layer count differs from the decoder"
        )
    if any(type(layer).__name__ != "DynamicLayer" for layer in layers):
        raise BackendIntegrationError(
            "DyCoKe supports full DynamicLayer caches, not sliding/compiled caches"
        )


def _physical_layer_length(cache: Any, layer_index: int) -> int | None:
    layers = getattr(cache, "layers", ())
    if not 0 <= layer_index < len(layers):
        return None
    keys = getattr(layers[layer_index], "keys", None)
    if keys is None:
        return None
    if getattr(keys, "ndim", None) != 4:
        raise InvariantError("DyCoKe physical cache keys are not rank four")
    return int(keys.shape[-2])


def _require_physical_cache_storage(
    cache: Any,
    layer_index: int,
    *,
    returned_keys: Any,
    returned_values: Any,
) -> None:
    layer = cache.layers[layer_index]
    keys = getattr(layer, "keys", None)
    values = getattr(layer, "values", None)
    if keys is not returned_keys or values is not returned_values:
        raise InvariantError(
            "DyCoKe native DynamicLayer did not return its physical storage"
        )


def _hidden_sequence_length(hidden_states: Any) -> int:
    if getattr(hidden_states, "ndim", None) != 3 or int(hidden_states.shape[0]) != 1:
        raise BackendIntegrationError(
            "DyCoKe decoder requires batch-one rank-three hidden states"
        )
    return int(hidden_states.shape[1])


def _bound_values(
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    bound = signature.bind_partial(*args, **kwargs)
    values = dict(bound.arguments)
    extras = values.pop("kwargs", None)
    if isinstance(extras, Mapping):
        values.update(extras)
    return values


def _replace_bound_argument(
    signature: inspect.Signature,
    args: Sequence[Any],
    kwargs: Mapping[str, Any],
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    bound = signature.bind_partial(*args, **kwargs)
    if name in signature.parameters:
        bound.arguments[name] = value
    else:
        extras = bound.arguments.get("kwargs")
        if not isinstance(extras, dict):
            raise BackendIntegrationError(f"DyCoKe native layer lacks {name!r}")
        extras[name] = value
    return tuple(bound.args), dict(bound.kwargs)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "DyCoKe video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "DyCoKeDecodeCacheController",
    "DyCoKeVideoExecutor",
    "create_dycoke_video_executor",
]
