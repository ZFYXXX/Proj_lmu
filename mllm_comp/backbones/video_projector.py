"""Execute one video Projector method through the existing native bridges.

The algorithm owns compression and value replay. The executor owns callback
lifetime, diagnostics, and applying that replay to Qwen3 DeepStack branches.
"""

from __future__ import annotations

from typing import Any, Callable

from mllm_comp.backbones._video_bridge import (
    install_video_bridge, register_video_callback, release_video_callback,
)
from mllm_comp.compressors.video._rules import load_video_rules, video_rule_metadata
from mllm_comp.compressors.video.positions import video_projector_site
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.vocabulary import CompressionMethod, ExecutionPhase, Modality, Placement


class VideoProjectorExecutor:
    """Install one registered video method and replay its decision.

    ``binding.strategy_factory()`` creates a callable method with
    ``replay(features, result)``. Both calls receive keyword ``config`` and
    ``rules``. The method returns VideoCompressionResult; replay rebuilds
    differentiable branch values using that exact selection or grouping.
    An explicit kernel/replay pair remains available to custom executors.
    """

    def __init__(self, *, backbone: str, model: Any, contract_projection: Any,
                 binding: Any, replay: Callable[..., Any] | None = None, stage: Any = None,
                 runtime: Any = None, facade: Any = None, binding_config: Any = None,
                 signal_provider: Any = None,
                 execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE) -> None:
        if backbone not in {"llava_onevision", "qwen25_vl", "qwen3_vl"}:
            raise BackendIntegrationError(f"Unknown video Projector backbone {backbone!r}.")
        projection = contract_projection
        algorithm, variant = projection.algorithm, projection.variant
        if (binding.name not in projection.binding_names or binding.variant != variant.name
                or variant.algorithm != algorithm.name or binding.backbone != backbone):
            raise BackendIntegrationError("Video Projector execution requires one exact selected contract.")
        if stage is not None and (stage.algorithm != algorithm.name or stage.variant != variant.name):
            raise BackendIntegrationError("Video Projector stage differs from its selected method.")
        if (algorithm.method is not CompressionMethod.TOKEN_COMPRESSION
                or variant.target_modalities != frozenset({Modality.VIDEO})
                or len(binding.targets) != 1
                or binding.targets[0].placement is not Placement.PROJECTOR_LEVEL):
            raise BackendIntegrationError("Video Projector execution requires one video-only Projector target.")
        target = binding.targets[0]
        positions = {video_projector_site(backbone, edge): edge for edge in ("before", "after")}
        if target.target_site not in positions:
            raise BackendIntegrationError(f"Unsupported native video Projector target {target.target_site!r}.")
        phase = ExecutionPhase(execution_phase)
        if phase not in {ExecutionPhase.FORWARD, ExecutionPhase.GENERATE} or phase not in binding.supported_phases:
            raise BackendIntegrationError(f"Video Projector binding {binding.name!r} does not support {phase.value}.")
        kernel = binding.strategy_factory() if callable(binding.strategy_factory) else algorithm.kernel
        replay = replay if replay is not None else getattr(kernel, "replay", None)
        if not callable(kernel) or not callable(replay):
            raise BackendIntegrationError("Video Projector methods require a callable kernel and replay.")
        self.backbone, self.model, self.runtime, self.facade = backbone, model, runtime, facade
        self.stage, self.binding, self.binding_config = stage, binding, binding_config
        self.algorithm, self.variant = algorithm.name, variant.name
        self.execution_phase = phase
        self.config = projection.algorithm_config
        self.kernel, self.replay = kernel, replay
        self.signal_provider = signal_provider
        self.target = target
        self.projector_position = positions[target.target_site]
        self._rules, self._rule_specs = load_video_rules(stage, contract_projection=projection)
        self._bridge = None
        self._bridge_handle = None
        self._owns_bridge = False
        self._installed = False
        self._invocations = 0
        self._layout_records: list[dict[str, Any]] = []

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        bridge, handle, owns_bridge = None, None, False
        try:
            if self.signal_provider is not None:
                self.signal_provider.install()
            bridge, owns_bridge, site = install_video_bridge(self.model, self.backbone)
            handle = register_video_callback(bridge, site, self._projector_callback,
                label=self.algorithm,
                **({"signal_provider": self.signal_provider} if self.signal_provider is not None else {}),
                **({"projector_position": "before"} if self.projector_position == "before" else {}))
        except BaseException:
            release_video_callback(bridge, handle, owns_bridge)
            if self.signal_provider is not None:
                try:
                    self.signal_provider.uninstall()
                except Exception:
                    pass  # Preserve the installation error after attempting both cleanups.
            raise
        self._bridge, self._bridge_handle, self._owns_bridge = bridge, handle, owns_bridge
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors = release_video_callback(self._bridge, self._bridge_handle, self._owns_bridge)
        if self.signal_provider is not None:
            try:
                self.signal_provider.uninstall()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._bridge_handle = self._bridge = None
        self._owns_bridge = self._installed = False
        if errors:
            raise BackendIntegrationError(f"{self.algorithm} video uninstall failed: {errors}")

    def runtime_metadata(self) -> dict[str, Any]:
        bridge_metadata = (self._bridge.runtime_metadata()
            if self._bridge is not None and callable(getattr(self._bridge, "runtime_metadata", None)) else None)
        return {
            "executor": f"{self.backbone}_{self.algorithm}_video",
            "algorithm": self.algorithm, "variant": self.variant, "backbone": self.backbone,
            "binding": self.binding.name, "execution_phase": self.execution_phase.value,
            "installed": self._installed, "config": dict(self.config),
            "projector_invocations": self._invocations, "projector_position": self.projector_position,
            "target_site": self.target.target_site, "records_scope": "latest_video_request",
            "projector_layout_deltas": list(self._layout_records), "bridge": bridge_metadata,
            **({"signal_provider": self.signal_provider.runtime_metadata()} if self.signal_provider is not None else {}),
            **video_rule_metadata(self._rule_specs),
        }

    def clear_diagnostic_records(self) -> None:
        """Discard request diagnostics while retaining invocation counters."""
        self._layout_records.clear()

    def _projector_callback(self, request: Any) -> Any:
        self.clear_diagnostic_records()
        result = self.kernel(request.payload, config=self.config, rules=self._rules)
        deepstack = ()
        if self.backbone == "qwen3_vl":
            from mllm_comp.backbones.qwen3_vl.video_compression import Qwen3VLVideoCompressionResponse

            deepstack = tuple(self.replay(branch, result, config=self.config, rules=self._rules)
                              for branch in request.deepstack_features)
            returned = Qwen3VLVideoCompressionResponse.from_result(request, result, deepstack_features=deepstack)
        else:
            returned = result
        self._invocations += 1
        for delta in result.layout_deltas:
            record = layout_delta_record(delta)
            record.update({"stage": self.target.name,
                "invocation_id": getattr(request, "invocation_id", None),
                "media_id": getattr(request, "media_id", None),
                "target_site": self.target.target_site, "projector_position": self.projector_position})
            self._layout_records.append(record)
        self._record_result(request, result, deepstack)
        return returned

    def _record_result(self, request: Any, result: Any, deepstack: tuple[Any, ...]) -> None:
        """Built-in methods may append their algorithm-specific evidence."""


def builtin_video_projector_projection(
    *, algorithm_name: str, backbone: str, stage: Any, binding: Any, config: Any,
) -> Any:
    """Resolve a standalone built-in constructor through its registered contract."""
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.v2_validation import V2BindingSelection, V2ContractProjection

    registry = builtin_multimodal_compression_contracts()
    algorithm = registry.algorithm(algorithm_name)
    variant = registry.variant(f"{algorithm_name}.video")
    binding = binding if binding is not None else registry.binding(f"{algorithm_name}.video.{backbone}")
    binding = binding.position_resolver(stage=stage, binding=binding, algorithm=algorithm, variant=variant)
    return V2ContractProjection(algorithm=algorithm, variant=variant,
        selections=(V2BindingSelection(binding=binding.name),), bindings=(binding,),
        required_handles=binding.required_handles, algorithm_config=config)
