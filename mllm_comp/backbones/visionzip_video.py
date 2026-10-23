"""Canonical VisionZip video signal providers and three-backbone executor.

The video variant consumes penultimate vision-encoder attention and hidden
states at the exact projected-video boundary.  Providers in this module only
capture video calls entered by the independent video bridges; ordinary text
and image execution therefore retain their native paths.  The fixed 126+21
kernel and each backbone's prompt/grid/MRoPE/DeepStack legalization remain
separate responsibilities.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterator

from mllm_comp.backbones._vision_attention import (
    bind_qwen_vision_block_call,
    qwen_vision_mean_received,
)
from mllm_comp.backbones.video_projector import VideoProjectorExecutor, builtin_video_projector_projection
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_visionzip_video_signal_provider"


@dataclass
class _SignalCapture:
    frame_count: int | None = None
    grid_thw: Any | None = None
    attention: Any | None = None
    hidden_states: Any | None = None
    calls: int = 0


class VisionZipVideoSignalProvider:
    """Reversible penultimate-layer capture aligned to projected video rows."""

    attention_layer = -2
    hidden_state_layer = -2

    def __init__(self, *, backbone: str, model: Any) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown VisionZip video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self._active: ContextVar[_SignalCapture | None] = ContextVar(
            f"visionzip_video_signals_{backbone}_{id(self)}",
            default=None,
        )
        self._patch: MethodPatch | None = None
        self._installed = False
        self._capture_count = 0

    @property
    def installed(self) -> bool:
        return self._installed

    def install(self) -> None:
        if self._installed:
            return
        existing = getattr(self.model, _PROVIDER_ATTRIBUTE, None)
        if existing is not None and existing is not self:
            raise BackendIntegrationError(
                "A different VisionZip video signal provider is already installed"
            )
        owner = self._capture_owner()
        original = getattr(owner, "forward", None)
        if not callable(original):
            raise BackendIntegrationError(
                "VisionZip could not resolve the vision capture forward method"
            )
        patch = MethodPatch.apply(
            owner,
            "forward",
            (
                self._llava_forward_replacement(original)
                if self.backbone == "llava_onevision"
                else self._qwen_block_replacement(original, owner)
            ),
            original,
            label="VisionZip video",
        )
        try:
            setattr(self.model, _PROVIDER_ATTRIBUTE, self)
        except BaseException:
            patch.restore()
            raise
        self._patch = patch
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        errors: list[str] = []
        if getattr(self.model, _PROVIDER_ATTRIBUTE, None) is not self:
            errors.append("provider ownership changed while active")
        elif _PROVIDER_ATTRIBUTE in vars(self.model):
            delattr(self.model, _PROVIDER_ATTRIBUTE)
        if self._patch is not None:
            try:
                self._patch.restore()
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
        self._patch = None
        self._active.set(None)
        self._installed = False
        if errors:
            raise BackendIntegrationError(
                f"VisionZip video signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_SignalCapture]:
        if not self._installed:
            raise BackendIntegrationError(
                "VisionZip video signal provider is not installed"
            )
        if self._active.get() is not None:
            raise BackendIntegrationError(
                "Nested VisionZip video signal capture is unsupported"
            )
        if self.backbone == "llava_onevision":
            if getattr(frames, "ndim", None) != 4 or grid_thw is not None:
                raise BackendIntegrationError(
                    "LLaVA VisionZip capture requires one rank-four frame tensor"
                )
            capture = _SignalCapture(frame_count=int(frames.shape[0]))
        else:
            if frames is not None or getattr(grid_thw, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "Qwen VisionZip capture requires rank-two video_grid_thw"
                )
            capture = _SignalCapture(grid_thw=grid_thw)
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            self._active.reset(token)

    def payload_signals(
        self,
        capture: _SignalCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _SignalCapture) or capture.calls != 1:
            raise BackendIntegrationError(
                "VisionZip video capture did not observe exactly one encoder pass"
            )
        if capture.attention is None or capture.hidden_states is None:
            raise BackendIntegrationError(
                "VisionZip video capture is missing penultimate encoder signals"
            )
        if isinstance(token_count, bool) or not isinstance(token_count, int):
            raise TypeError("token_count must be an integer")
        if token_count < 1 or len(tuple(coordinates)) != token_count:
            raise BackendIntegrationError(
                "VisionZip projected coordinates do not match the payload"
            )
        if self.backbone == "llava_onevision":
            attention, hidden = self._align_llava(capture)
        else:
            attention, hidden = self._align_qwen(capture)
        if (
            int(attention.shape[0]) != token_count
            or int(hidden.shape[0]) != token_count
        ):
            raise BackendIntegrationError(
                "VisionZip aligned signals do not match projected video tokens"
            )
        self._capture_count += 1
        return {
            "vision_encoder_attention": {self.attention_layer: attention},
            "vision_encoder_hidden_states": {self.hidden_state_layer: hidden},
        }

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": f"{self.backbone}_visionzip_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "attention_layer": self.attention_layer,
            "hidden_state_layer": self.hidden_state_layer,
            "signal_boundary": "projected_video_tokens",
            "video_only": True,
            "real_checkpoint_verified": False,
        }

    def _capture_owner(self) -> Any:
        if self.backbone == "llava_onevision":
            get_vision = getattr(self.model, "get_vision_tower", None)
            outer = get_vision() if callable(get_vision) else None
            inner = getattr(outer, "vision_tower", None)
            if inner is None:
                raise BackendIntegrationError(
                    "LLaVA VisionZip requires the native inner vision tower"
                )
            return inner
        core = getattr(self.model, "model", None)
        visual = getattr(core, "visual", None)
        blocks = tuple(getattr(visual, "blocks", ()))
        if len(blocks) < 2:
            raise BackendIntegrationError(
                "VisionZip requires at least two native visual blocks"
            )
        return blocks[-2]

    def _llava_forward_replacement(
        self,
        original: Callable[..., Any],
    ) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def replacement(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            capture = self._active.get()
            if capture is None:
                return original(*args, **kwargs)
            if capture.calls:
                raise BackendIntegrationError(
                    "VisionZip observed multiple LLaVA vision encoder calls"
                )
            bound = signature.bind_partial(*args, **kwargs)
            for name in ("output_hidden_states", "output_attentions", "return_dict"):
                if name not in signature.parameters:
                    raise BackendIntegrationError(
                        f"LLaVA inner vision forward is missing {name!r}"
                    )
                bound.arguments[name] = True
            output = original(*bound.args, **bound.kwargs)
            hidden_states = getattr(output, "hidden_states", None)
            attentions = getattr(output, "attentions", None)
            if not hidden_states or len(hidden_states) < 2:
                raise BackendIntegrationError(
                    "LLaVA VisionZip did not receive penultimate hidden states"
                )
            if not attentions or len(attentions) < 2:
                raise BackendIntegrationError(
                    "LLaVA VisionZip did not receive penultimate attention"
                )
            attention = attentions[self.attention_layer]
            hidden = hidden_states[self.hidden_state_layer]
            if (
                getattr(attention, "ndim", None) != 4
                or getattr(hidden, "ndim", None) != 3
            ):
                raise BackendIntegrationError(
                    "LLaVA VisionZip received unknown SigLIP signal shapes"
                )
            capture.attention = attention.float().mean(dim=1).mean(dim=1)
            capture.hidden_states = hidden
            capture.calls = 1
            return output

        return replacement

    def _qwen_block_replacement(
        self,
        original: Callable[..., Any],
        block: Any,
    ) -> Callable[..., Any]:
        signature = inspect.signature(original)

        def replacement(_owner: Any, *args: Any, **kwargs: Any) -> Any:
            capture = self._active.get()
            if capture is None:
                return original(*args, **kwargs)
            if capture.calls:
                raise BackendIntegrationError(
                    "VisionZip observed multiple penultimate visual-block calls"
                )
            hidden, cu_seqlens, position_embeddings = bind_qwen_vision_block_call(
                signature,
                args,
                kwargs,
            )
            normalised = block.norm1(hidden)
            attention = qwen_vision_mean_received(
                block.attn,
                normalised,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            output = original(*args, **kwargs)
            if getattr(output, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "VisionZip penultimate visual block returned an unknown shape"
                )
            capture.attention = attention
            capture.hidden_states = output
            capture.calls = 1
            return output

        return replacement

    def _align_llava(self, capture: _SignalCapture) -> tuple[Any, Any]:
        get_vision = getattr(self.model, "get_vision_tower", None)
        outer = get_vision() if callable(get_vision) else None
        side = getattr(outer, "num_patches_per_side", None)
        frame_count = capture.frame_count
        attention = capture.attention
        hidden = capture.hidden_states
        if (
            isinstance(side, bool)
            or not isinstance(side, int)
            or side < 1
            or frame_count is None
            or int(attention.shape[0]) != frame_count
            or int(hidden.shape[0]) != frame_count
            or int(attention.shape[1]) != side * side
            or int(hidden.shape[1]) != side * side
        ):
            raise BackendIntegrationError(
                "LLaVA VisionZip raw SigLIP signals do not match the native raster"
            )
        stride = getattr(
            getattr(self.model, "config", None), "mm_spatial_pool_stride", 2
        )
        if stride != 2:
            raise BackendIntegrationError(
                "Canonical LLaVA VisionZip requires spatial-pool stride two"
            )
        pool = getattr(self.model, "get_2dPool", None)
        if not callable(pool):
            raise BackendIntegrationError(
                "LLaVA VisionZip requires native get_2dPool signal alignment"
            )
        pooled_attention = pool(attention.unsqueeze(-1), stride=stride).squeeze(-1)
        pooled_hidden = pool(hidden, stride=stride)
        return (
            pooled_attention.reshape(-1),
            pooled_hidden.reshape(-1, int(pooled_hidden.shape[-1])),
        )

    def _align_qwen(self, capture: _SignalCapture) -> tuple[Any, Any]:
        torch = _torch()
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        unit = getattr(visual, "spatial_merge_unit", None)
        grid = capture.grid_thw
        if isinstance(unit, bool) or not isinstance(unit, int) or unit < 1:
            raise BackendIntegrationError(
                "Qwen VisionZip requires a positive spatial_merge_unit"
            )
        patch_counts = tuple(int(value) for value in grid.prod(-1).tolist())
        total = sum(patch_counts)
        if any(count < 1 or count % unit for count in patch_counts):
            raise BackendIntegrationError(
                "Qwen VisionZip video grids must align to spatial merge groups"
            )
        attention = capture.attention
        hidden = capture.hidden_states
        if int(attention.shape[0]) != total or int(hidden.shape[0]) != total:
            raise BackendIntegrationError(
                "Qwen VisionZip raw signals do not match video_grid_thw"
            )
        feature_count = total // unit
        merged_attention = attention.reshape(feature_count, unit).mean(dim=1)
        merged_hidden = hidden.reshape(feature_count, unit, -1).reshape(
            feature_count, -1
        )
        if self.backbone == "qwen25_vl":
            get_window = getattr(visual, "get_window_index", None)
            if not callable(get_window):
                raise BackendIntegrationError(
                    "Qwen2.5-VL VisionZip requires native window indices"
                )
            window_index, _ = get_window(grid)
            if int(window_index.numel()) != feature_count:
                raise BackendIntegrationError(
                    "Qwen2.5-VL window indices do not match merged signals"
                )
            reverse = torch.argsort(window_index).to(merged_attention.device)
            merged_attention = merged_attention.index_select(0, reverse)
            merged_hidden = merged_hidden.index_select(
                0, reverse.to(merged_hidden.device)
            )
        return merged_attention, merged_hidden


class VisionZipVideoExecutor(VideoProjectorExecutor):
    """Penultimate visual signals on the shared video Projector lifecycle."""

    algorithm = "visionzip"
    variant = "visionzip.video"

    def __init__(
        self, *, backbone: str, model: Any, runtime: Any = None, facade: Any = None,
        stage: Any = None, binding: Any = None, binding_config: Any = None,
        contract_projection: Any = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
    ) -> None:
        phase = ExecutionPhase(execution_phase)
        if contract_projection is None or contract_projection.algorithm.name == "visionzip":
            _validate_binding(binding, stage, backbone, phase)
        if contract_projection is None:
            contract_projection = builtin_video_projector_projection(
                algorithm_name="visionzip", backbone=backbone, stage=stage, binding=binding,
                config=_validated_config(stage, binding_config),
            )
            binding = contract_projection.bindings[0]
        self._algorithm_records: list[dict[str, Any]] = []
        super().__init__(backbone=backbone, model=model, runtime=runtime, facade=facade,
            stage=stage, binding=binding, binding_config=binding_config,
            contract_projection=contract_projection, execution_phase=phase,
            signal_provider=VisionZipVideoSignalProvider(backbone=backbone, model=model))

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        if self.algorithm != "visionzip":
            return metadata
        return {
            **metadata, "canonical_config": dict(self.config),
            "canonical_evidence_applicable": self.projector_position == "after" and not self._rules,
            "algorithm_records": list(self._algorithm_records),
            "fixed_budget_per_temporal_slice": 147, "image_variant_shared": False,
            "training_gradients_preserved": True,
            "real_checkpoint_verified": False, "support_activation_claimed": False,
        }

    def clear_diagnostic_records(self) -> None:
        super().clear_diagnostic_records()
        self._algorithm_records.clear()

    def _record_result(self, request: Any, result: Any, deepstack: tuple[Any, ...]) -> None:
        if self.algorithm != "visionzip":
            return
        for layout in self._layout_records:
            layout["budget_per_temporal_slice"] = 147
        record = _visionzip_invocation_record(request, result)
        if self.backbone == "qwen3_vl":
            record.update({
                "deepstack_branch_count": len(deepstack),
                "deepstack_result_shapes": tuple(tuple(int(item) for item in branch.shape) for branch in deepstack),
            })
        self._algorithm_records.append(record)


def create_visionzip_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> VisionZipVideoExecutor:
    return VisionZipVideoExecutor(backbone=backbone, **kwargs)


def _visionzip_invocation_record(request: Any, result: Any) -> dict[str, Any]:
    payload = getattr(request, "payload", None)
    signals = getattr(payload, "signals", {})
    attention_by_layer = signals.get("vision_encoder_attention", {})
    hidden_by_layer = signals.get("vision_encoder_hidden_states", {})
    attention = (
        attention_by_layer.get(-2) if isinstance(attention_by_layer, Mapping) else None
    )
    hidden = hidden_by_layer.get(-2) if isinstance(hidden_by_layer, Mapping) else None
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(attention, "ndim", None) != 1
        or getattr(hidden, "ndim", None) != 2
        or int(attention.shape[0]) != source_count
        or int(hidden.shape[0]) != source_count
    ):
        raise BackendIntegrationError(
            "VisionZip invocation evidence lost aligned penultimate signals"
        )
    torch = _torch()
    attention_finite = bool(torch.isfinite(attention).all().item())
    hidden_finite = bool(torch.isfinite(hidden).all().item())
    if not attention_finite or not hidden_finite:
        raise BackendIntegrationError(
            "VisionZip invocation evidence contains non-finite signals"
        )

    metadata = getattr(result, "metadata", {})
    records_by_video = metadata.get("frame_records_by_video")
    targets_by_video = metadata.get("merge_target_indices_by_video")
    groups_by_video = getattr(result, "source_groups", ())
    representatives_by_video = getattr(result, "representative_indices", ())
    if (
        not isinstance(records_by_video, tuple)
        or len(records_by_video) != 1
        or not isinstance(targets_by_video, tuple)
        or len(targets_by_video) != 1
        or len(groups_by_video) != 1
        or len(representatives_by_video) != 1
    ):
        raise BackendIntegrationError(
            "VisionZip invocation evidence requires exactly one video"
        )
    frame_records = tuple(records_by_video[0])
    groups = tuple(tuple(int(item) for item in group) for group in groups_by_video[0])
    targets = tuple(int(item) for item in targets_by_video[0])
    representatives = tuple(int(item) for item in representatives_by_video[0])
    if len(groups) != result_count or len(targets) != result_count:
        raise BackendIntegrationError("VisionZip invocation group evidence drifted")

    frame_summaries = []
    group_offset = 0
    assigned_contextual_sources = 0
    non_singleton_contextual_groups = 0
    for frame in frame_records:
        frame_result_count = int(frame["result_count"])
        stop = group_offset + frame_result_count
        frame_groups = groups[group_offset:stop]
        frame_targets = targets[group_offset:stop]
        contextual = {int(item) for item in frame["contextual_target_indices"]}
        dominant = {int(item) for item in frame["dominant_indices"]}
        contextual_pairs = tuple(
            (group, target)
            for group, target in zip(frame_groups, frame_targets)
            if target in contextual
        )
        dominant_pairs = tuple(
            (group, target)
            for group, target in zip(frame_groups, frame_targets)
            if target in dominant
        )
        if (
            len(dominant_pairs) != 126
            or len(contextual_pairs) != 21
            or any(group != (target,) for group, target in dominant_pairs)
            or any(target not in group for group, target in contextual_pairs)
        ):
            raise BackendIntegrationError(
                "VisionZip invocation does not contain canonical 126+21 groups"
            )
        assigned = sum(len(group) - 1 for group, _target in contextual_pairs)
        non_singleton = sum(len(group) > 1 for group, _target in contextual_pairs)
        expected_assigned = int(frame["source_count"]) - frame_result_count
        if assigned != expected_assigned:
            raise BackendIntegrationError(
                "VisionZip contextual assignments do not cover dropped sources"
            )
        assigned_contextual_sources += assigned
        non_singleton_contextual_groups += non_singleton
        frame_summaries.append(
            {
                "temporal_index": int(frame["temporal_index"]),
                "source_count": int(frame["source_count"]),
                "result_count": frame_result_count,
                "dominant_count": len(dominant_pairs),
                "contextual_target_count": len(contextual_pairs),
                "assigned_contextual_source_count": assigned,
                "non_singleton_contextual_group_count": non_singleton,
                "anchor_legalizations": int(frame["anchor_legalizations"]),
            }
        )
        group_offset = stop
    if group_offset != result_count:
        raise BackendIntegrationError("VisionZip frame evidence changed output count")

    partition = tuple(sorted(item for group in groups for item in group))
    return {
        "invocation_id": getattr(request, "invocation_id", None),
        "media_id": getattr(request, "media_id", None),
        "source_count": source_count,
        "result_count": result_count,
        "canonical_config": dict(metadata.get("canonical_config", {})),
        "attention_layer": -2,
        "hidden_state_layer": -2,
        "attention_shape": tuple(int(item) for item in attention.shape),
        "hidden_state_shape": tuple(int(item) for item in hidden.shape),
        "attention_finite": attention_finite,
        "hidden_state_finite": hidden_finite,
        "temporal_slice_count": len(frame_records),
        "per_temporal_slice_budget": int(metadata.get("per_temporal_slice_budget", 0)),
        "dominant_count": sum(item["dominant_count"] for item in frame_summaries),
        "contextual_target_count": sum(
            item["contextual_target_count"] for item in frame_summaries
        ),
        "assigned_contextual_source_count": assigned_contextual_sources,
        "non_singleton_contextual_group_count": non_singleton_contextual_groups,
        "max_source_group_size": max(len(group) for group in groups),
        "source_partition_exact": partition == tuple(range(source_count)),
        "merge_targets_inside_groups": all(
            target in group for group, target in zip(groups, targets)
        ),
        "representatives_source_ordered": representatives
        == tuple(sorted(representatives)),
        "selection_scope": metadata.get("selection_scope"),
        "attention_statistic": metadata.get("attention_statistic"),
        "similarity": metadata.get("similarity"),
        "contextual_target_sampling": metadata.get("contextual_target_sampling"),
        "merge_value": metadata.get("merge_value"),
        "representative_policy": metadata.get("representative_policy"),
        "image_variant_shared": metadata.get("image_variant_shared"),
        "frame_summaries": frame_summaries,
    }


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("VisionZip stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "VisionZip video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("visionzip", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"visionzip.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"VisionZip executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"VisionZip binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("dominant_contextual_reduce",):
            raise BackendIntegrationError("VisionZip binding target changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "visionzip":
            raise BackendIntegrationError(
                "VisionZip executor received another algorithm"
            )
        if getattr(stage, "variant", None) != "visionzip.video":
            raise BackendIntegrationError("VisionZip executor received another variant")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "VisionZip video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "VisionZipVideoExecutor",
    "VisionZipVideoSignalProvider",
    "create_visionzip_video_executor",
]
