"""FastVID video capability providers and three-backbone executor.

LLaVA consumes the frozen SigLIP pooling head from an already-local checkpoint
and aligns its probe attention to the native pooled projector raster.  Qwen
backbones consume exact mean-received attention from their final visual block
and projected per-frame means.  Every provider also attaches an independent
raw/native stream to the shared payload; only video calls enter the capture
context, so text and image behavior remain native.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from mllm_comp.backbones._vision_attention import (
    bind_qwen_vision_block_call,
    qwen_vision_mean_received,
)
from mllm_comp.backbones.video_projector import VideoProjectorExecutor, builtin_video_projector_projection
from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.compressors.video import (
    VideoTokenCoordinate,
    VideoTokenPayload,
)
from mllm_comp.compressors.video.config import resolve_canonical_video_config
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase


_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
_PROVIDER_ATTRIBUTE = "_mllm_comp_fastvid_video_signal_provider"


@dataclass
class _FastVIDCapture:
    frame_count: int | None = None
    grid_thw: Any | None = None
    raw_features: Any | None = None
    raw_attention: Any | None = None
    calls: int = 0


class FastVIDVideoSignalProvider:
    """Capture and attach canonical FastVID provider signals for one model."""

    requires_projected_reference = True

    def __init__(
        self,
        *,
        backbone: str,
        model: Any,
        siglip_path: str | Path | None = None,
        pooling_head: Any | None = None,
    ) -> None:
        if backbone not in _BACKBONES:
            raise ValueError(f"Unknown FastVID video backbone {backbone!r}")
        self.backbone = backbone
        self.model = model
        self.siglip_path = None if siglip_path is None else Path(siglip_path)
        self._pooling_head = pooling_head
        self._active: ContextVar[_FastVIDCapture | None] = ContextVar(
            f"fastvid_video_signals_{backbone}_{id(self)}",
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
                "A different FastVID video signal provider is already installed"
            )
        owner = self._capture_owner()
        original = getattr(owner, "forward", None)
        if not callable(original):
            raise BackendIntegrationError(
                "FastVID could not resolve the native visual forward"
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
            label="FastVID",
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
                f"FastVID video signal-provider uninstall failed: {errors}"
            )

    @contextmanager
    def capture_context(
        self,
        *,
        frames: Any | None = None,
        grid_thw: Any | None = None,
    ) -> Iterator[_FastVIDCapture]:
        if not self._installed:
            raise BackendIntegrationError("FastVID signal provider is not installed")
        if self._active.get() is not None:
            raise BackendIntegrationError("Nested FastVID capture is unsupported")
        if self.backbone == "llava_onevision":
            if getattr(frames, "ndim", None) != 4 or grid_thw is not None:
                raise BackendIntegrationError(
                    "LLaVA FastVID capture requires one rank-four frame tensor"
                )
            capture = _FastVIDCapture(frame_count=int(frames.shape[0]))
        else:
            if frames is not None or getattr(grid_thw, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "Qwen FastVID capture requires rank-two video_grid_thw"
                )
            if int(grid_thw.shape[0]) != 1:
                raise BackendIntegrationError(
                    "FastVID supports exactly one video grid row per sample"
                )
            capture = _FastVIDCapture(grid_thw=grid_thw)
        token = self._active.set(capture)
        try:
            yield capture
        finally:
            self._active.reset(token)

    def payload_signals(
        self,
        capture: _FastVIDCapture,
        *,
        token_count: int,
        coordinates: Sequence[Any],
    ) -> Mapping[str, Any]:
        if not isinstance(capture, _FastVIDCapture) or capture.calls != 1:
            raise BackendIntegrationError(
                "FastVID capture did not observe exactly one visual encoder pass"
            )
        if capture.raw_features is None:
            raise BackendIntegrationError("FastVID capture is missing raw features")
        if token_count < 1 or len(tuple(coordinates)) != token_count:
            raise BackendIntegrationError(
                "FastVID projected coordinates do not match the payload"
            )
        return {}

    def enrich_payload(
        self,
        payload: VideoTokenPayload,
        capture: _FastVIDCapture,
    ) -> VideoTokenPayload:
        torch = _torch()
        if not isinstance(payload, VideoTokenPayload):
            raise TypeError("payload must be a VideoTokenPayload")
        if capture.calls != 1 or capture.raw_features is None:
            raise BackendIntegrationError("FastVID capture is incomplete")
        projected_frames = _dense_frame_spans(payload.coordinates)
        if self.backbone == "llava_onevision":
            raw, raw_coordinates, frame_globals, attention = self._llava_signals(
                payload,
                capture,
                projected_frames,
            )
            signal_source = "frozen_siglip_pooling_head"
        else:
            raw, raw_coordinates, attention = self._qwen_signals(
                payload,
                capture,
            )
            frame_globals = torch.stack(
                tuple(
                    payload.features[start:stop].mean(dim=0)
                    for _temporal, start, stop in projected_frames
                ),
                dim=0,
            )
            signal_source = "native_final_visual_attention_and_projected_mean"
        if int(attention.shape[0]) != payload.token_count:
            raise BackendIntegrationError(
                "FastVID saliency attention does not align with projected tokens"
            )
        if int(frame_globals.shape[0]) != len(projected_frames):
            raise BackendIntegrationError(
                "FastVID global features do not align with temporal slices"
            )
        self._capture_count += 1
        return VideoTokenPayload(
            features=payload.features,
            media_ids=payload.media_ids,
            video_offsets=payload.video_offsets,
            coordinates=payload.coordinates,
            raw_features=raw,
            raw_video_offsets=(0, int(raw.shape[0])),
            raw_coordinates=raw_coordinates,
            token_mask=payload.token_mask,
            signals={
                **dict(payload.signals),
                "frame_global_features": frame_globals,
                "token_saliency_attention": attention,
            },
            metadata={
                **dict(payload.metadata),
                "fastvid_signal_source": signal_source,
                "fastvid_raw_stream": "native_visual_encoder_output",
            },
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": f"{self.backbone}_fastvid_video_signals",
            "installed": self._installed,
            "capture_count": self._capture_count,
            "video_only": True,
            "raw_stream_attached": True,
            "siglip_path": (
                None if self.siglip_path is None else str(self.siglip_path)
            ),
            "siglip_pooling_head_loaded": self._pooling_head is not None,
            "real_checkpoint_verified": False,
        }

    def _capture_owner(self) -> Any:
        if self.backbone == "llava_onevision":
            get_vision = getattr(self.model, "get_vision_tower", None)
            outer = get_vision() if callable(get_vision) else None
            inner = getattr(outer, "vision_tower", None)
            if inner is None:
                raise BackendIntegrationError(
                    "LLaVA FastVID requires the native inner SigLIP tower"
                )
            return inner
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        blocks = tuple(getattr(visual, "blocks", ()))
        if not blocks:
            raise BackendIntegrationError("FastVID requires native visual blocks")
        return blocks[-1]

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
                    "FastVID observed multiple LLaVA visual encoder calls"
                )
            bound = signature.bind_partial(*args, **kwargs)
            for name in ("output_hidden_states", "return_dict"):
                if name not in signature.parameters:
                    raise BackendIntegrationError(
                        f"LLaVA inner vision forward is missing {name!r}"
                    )
                bound.arguments[name] = True
            output = original(*bound.args, **bound.kwargs)
            hidden_states = getattr(output, "hidden_states", None)
            if not hidden_states:
                raise BackendIntegrationError(
                    "LLaVA FastVID did not receive encoder hidden states"
                )
            raw = hidden_states[-1]
            if getattr(raw, "ndim", None) != 3:
                raise BackendIntegrationError(
                    "LLaVA FastVID received an unknown raw-feature shape"
                )
            capture.raw_features = raw
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
                    "FastVID observed multiple final visual-block calls"
                )
            hidden, cu_seqlens, position_embeddings = bind_qwen_vision_block_call(
                signature,
                args,
                kwargs,
            )
            attention = qwen_vision_mean_received(
                block.attn,
                block.norm1(hidden),
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            output = original(*args, **kwargs)
            if getattr(output, "ndim", None) != 2:
                raise BackendIntegrationError(
                    "FastVID final visual block returned an unknown shape"
                )
            capture.raw_features = output
            capture.raw_attention = attention
            capture.calls = 1
            return output

        return replacement

    def _llava_signals(
        self,
        payload: VideoTokenPayload,
        capture: _FastVIDCapture,
        projected_frames: tuple[tuple[int, int, int], ...],
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...], Any, Any]:
        torch = _torch()
        raw = capture.raw_features
        frame_count = capture.frame_count
        get_vision = getattr(self.model, "get_vision_tower", None)
        outer = get_vision() if callable(get_vision) else None
        side = getattr(outer, "num_patches_per_side", None)
        if (
            isinstance(side, bool)
            or not isinstance(side, int)
            or side < 1
            or frame_count is None
            or frame_count != len(projected_frames)
            or int(raw.shape[0]) != frame_count
            or int(raw.shape[1]) != side * side
        ):
            raise BackendIntegrationError(
                "LLaVA FastVID raw features do not match the native SigLIP raster"
            )
        pooling_head = self._resolve_pooling_head(outer)
        pooling_head = pooling_head.to(device=raw.device, dtype=payload.features.dtype)
        pooling_head.eval()
        with torch.no_grad():
            frame_globals, raw_attention = pooling_head(
                raw.to(dtype=payload.features.dtype)
            )
        if getattr(raw_attention, "ndim", None) == 3:
            raw_attention = raw_attention.squeeze(1)
        if tuple(raw_attention.shape) != (frame_count, side * side):
            raise BackendIntegrationError(
                "LLaVA FastVID pooling-head attention shape changed"
            )
        projected_count = projected_frames[0][2] - projected_frames[0][1]
        projected_side = int(projected_count**0.5)
        if projected_side * projected_side != projected_count:
            raise BackendIntegrationError(
                "LLaVA FastVID projected frame is not a square raster"
            )
        image = raw_attention.reshape(frame_count, 1, side, side)
        attention = torch.nn.functional.interpolate(
            image,
            size=(projected_side, projected_side),
            mode="bilinear",
        ).reshape(-1)
        raw_coordinates = tuple(
            VideoTokenCoordinate(t=frame, h=row, w=column, frame_index=frame)
            for frame in range(frame_count)
            for row in range(side)
            for column in range(side)
        )
        return (
            raw.reshape(-1, int(raw.shape[-1])),
            raw_coordinates,
            frame_globals,
            attention,
        )

    def _qwen_signals(
        self,
        payload: VideoTokenPayload,
        capture: _FastVIDCapture,
    ) -> tuple[Any, tuple[VideoTokenCoordinate, ...], Any]:
        torch = _torch()
        visual = getattr(getattr(self.model, "model", None), "visual", None)
        grid = capture.grid_thw
        raw = capture.raw_features
        raw_attention = capture.raw_attention
        if raw_attention is None:
            raise BackendIntegrationError("Qwen FastVID attention capture is missing")
        temporal, height, width = tuple(int(value) for value in grid[0].tolist())
        unit = int(getattr(visual, "spatial_merge_unit", 0))
        merge_size = int(getattr(visual, "spatial_merge_size", 0))
        raw_count = temporal * height * width
        if (
            unit < 1
            or merge_size < 1
            or unit != merge_size * merge_size
            or int(raw.shape[0]) != raw_count
            or int(raw_attention.shape[0]) != raw_count
        ):
            raise BackendIntegrationError(
                "Qwen FastVID raw signals do not match video_grid_thw"
            )
        feature_count = raw_count // unit
        grouped_raw = raw.reshape(feature_count, unit, -1)
        grouped_attention = raw_attention.reshape(feature_count, unit).mean(dim=1)
        if self.backbone == "qwen25_vl":
            get_window = getattr(visual, "get_window_index", None)
            if not callable(get_window):
                raise BackendIntegrationError(
                    "Qwen2.5-VL FastVID requires native window indices"
                )
            window_index, _ = get_window(grid)
            if int(window_index.numel()) != feature_count:
                raise BackendIntegrationError(
                    "Qwen2.5-VL FastVID window indices changed"
                )
            reverse = torch.argsort(window_index)
            grouped_raw = grouped_raw.index_select(0, reverse.to(grouped_raw.device))
            grouped_attention = grouped_attention.index_select(
                0, reverse.to(grouped_attention.device)
            )
        raw_raster = (
            grouped_raw.reshape(
                temporal,
                height // merge_size,
                width // merge_size,
                merge_size,
                merge_size,
                -1,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(raw_count, int(raw.shape[-1]))
        )
        raw_coordinates = tuple(
            VideoTokenCoordinate(t=t, h=h, w=w, tubelet_index=t)
            for t in range(temporal)
            for h in range(height)
            for w in range(width)
        )
        if int(grouped_attention.shape[0]) != payload.token_count:
            raise BackendIntegrationError(
                "Qwen FastVID merged attention does not match projected payload"
            )
        return raw_raster, raw_coordinates, grouped_attention

    def _resolve_pooling_head(self, outer: Any) -> Any:
        if self._pooling_head is not None:
            return self._pooling_head
        candidates = (
            self.siglip_path,
            _local_path(getattr(outer, "vision_tower_name", None)),
            _local_path(
                getattr(getattr(self.model, "config", None), "mm_vision_tower", None)
            ),
        )
        source = next(
            (path for path in candidates if path is not None and path.is_dir()),
            None,
        )
        if source is None:
            raise BackendIntegrationError(
                "FastVID requires a readable existing SigLIP checkpoint path; "
                "downloads are forbidden"
            )
        self.siglip_path = source
        self._pooling_head = _load_frozen_siglip_pooling_head(source)
        return self._pooling_head


class FastVIDVideoExecutor(VideoProjectorExecutor):
    """FastVID signals and evidence on the shared video Projector lifecycle."""

    algorithm = "fastvid"
    variant = "fastvid.video"

    def __init__(
        self, *, backbone: str, model: Any, runtime: Any = None, facade: Any = None,
        stage: Any = None, binding: Any = None, binding_config: Any = None,
        contract_projection: Any = None,
        execution_phase: ExecutionPhase | str = ExecutionPhase.GENERATE,
        signal_provider: Any = None,
    ) -> None:
        phase = ExecutionPhase(execution_phase)
        if contract_projection is None or contract_projection.algorithm.name == "fastvid":
            _validate_binding(binding, stage, backbone, phase)
        if contract_projection is None:
            contract_projection = builtin_video_projector_projection(
                algorithm_name="fastvid", backbone=backbone, stage=stage, binding=binding,
                config=_validated_config(stage, binding_config),
            )
            binding = contract_projection.bindings[0]
        runtime_metadata = getattr(runtime, "metadata", {}) if runtime is not None else {}
        siglip_path = runtime_metadata.get("fastvid_siglip_path") if isinstance(runtime_metadata, Mapping) else None
        signal_provider = signal_provider or FastVIDVideoSignalProvider(
            backbone=backbone, model=model, siglip_path=siglip_path,
        )
        self._algorithm_records: list[dict[str, Any]] = []
        super().__init__(backbone=backbone, model=model, runtime=runtime, facade=facade,
            stage=stage, binding=binding, binding_config=binding_config,
            contract_projection=contract_projection, execution_phase=phase,
            signal_provider=signal_provider)

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        if self.algorithm != "fastvid":
            return metadata
        return {
            **metadata, "canonical_config": dict(self.config),
            "canonical_evidence_applicable": self.projector_position == "after" and not self._rules,
            "algorithm_records": list(self._algorithm_records),
            "stages": ("dyseg", "stprune", "dtm"),
            "training_free_compressor": True, "training_gradients_preserved": True,
            "real_checkpoint_verified": False, "support_activation_claimed": False,
        }

    def clear_diagnostic_records(self) -> None:
        super().clear_diagnostic_records()
        self._algorithm_records.clear()

    def _record_result(self, request: Any, result: Any, deepstack: tuple[Any, ...]) -> None:
        if self.algorithm != "fastvid":
            return
        record = _fastvid_invocation_record(request, result)
        if self.backbone == "qwen3_vl":
            record.update({
                "deepstack_branch_count": len(deepstack),
                "deepstack_result_shapes": tuple(tuple(int(item) for item in branch.shape) for branch in deepstack),
            })
        self._algorithm_records.append(record)


def create_fastvid_video_executor(
    *,
    backbone: str,
    **kwargs: Any,
) -> FastVIDVideoExecutor:
    return FastVIDVideoExecutor(backbone=backbone, **kwargs)


def _fastvid_invocation_record(request: Any, result: Any) -> dict[str, Any]:
    """Freeze the defining FastVID signals and per-stage decisions for one video."""

    payload = getattr(request, "payload", None)
    features = getattr(payload, "features", None)
    raw_features = getattr(payload, "raw_features", None)
    signals = getattr(payload, "signals", {})
    frame_globals = (
        signals.get("frame_global_features") if isinstance(signals, Mapping) else None
    )
    saliency = (
        signals.get("token_saliency_attention")
        if isinstance(signals, Mapping)
        else None
    )
    source_count = int(getattr(result, "source_count", 0))
    result_count = int(getattr(result, "result_count", 0))
    if (
        getattr(features, "ndim", None) != 2
        or getattr(raw_features, "ndim", None) != 2
        or getattr(frame_globals, "ndim", None) != 2
        or getattr(saliency, "ndim", None) != 1
        or int(features.shape[0]) != source_count
        or int(saliency.shape[0]) != source_count
    ):
        raise BackendIntegrationError(
            "FastVID invocation evidence lost one of its aligned runtime signals"
        )
    torch = _torch()
    finite = {
        "projected_features_finite": bool(torch.isfinite(features).all().item()),
        "raw_features_finite": bool(torch.isfinite(raw_features).all().item()),
        "frame_global_features_finite": bool(
            torch.isfinite(frame_globals).all().item()
        ),
        "token_saliency_attention_finite": bool(torch.isfinite(saliency).all().item()),
    }
    if not all(finite.values()):
        raise BackendIntegrationError(
            "FastVID invocation evidence contains non-finite runtime signals"
        )

    coordinates = tuple(getattr(payload, "coordinates", ()))
    raw_coordinates = tuple(getattr(payload, "raw_coordinates", ()))
    projected_spans = _dense_frame_spans(coordinates)
    raw_spans = _dense_frame_spans(raw_coordinates)
    projected_counts = tuple(stop - start for _t, start, stop in projected_spans)
    raw_counts = tuple(stop - start for _t, start, stop in raw_spans)
    frame_count = len(projected_spans)
    if (
        frame_count < 1
        or len(raw_spans) != frame_count
        or len(set(projected_counts)) != 1
        or len(set(raw_counts)) != 1
        or int(frame_globals.shape[0]) != frame_count
    ):
        raise BackendIntegrationError(
            "FastVID invocation evidence lost uniform projected/raw frame layouts"
        )

    metadata = getattr(result, "metadata", {})
    records_by_video = metadata.get("value_records_by_video")
    segments_by_video = metadata.get("segment_sizes_by_video")
    allocations_by_video = metadata.get("context_allocations_by_video")
    frame_counts_by_video = metadata.get("frame_output_counts_by_video")
    groups_by_video = getattr(result, "source_groups", ())
    representatives_by_video = getattr(result, "representative_indices", ())
    collections = (
        records_by_video,
        segments_by_video,
        allocations_by_video,
        frame_counts_by_video,
        groups_by_video,
        representatives_by_video,
    )
    if any(not isinstance(value, tuple) or len(value) != 1 for value in collections):
        raise BackendIntegrationError(
            "FastVID invocation evidence requires exactly one video"
        )

    segments = tuple(int(item) for item in segments_by_video[0])
    allocations = tuple(int(item) for item in allocations_by_video[0])
    frame_output_counts = tuple(int(item) for item in frame_counts_by_video[0])
    groups = tuple(tuple(int(item) for item in group) for group in groups_by_video[0])
    representatives = tuple(int(item) for item in representatives_by_video[0])
    value_records = tuple(records_by_video[0])
    if (
        sum(segments) != frame_count
        or len(allocations) != frame_count
        or len(frame_output_counts) != frame_count
        or sum(frame_output_counts) != result_count
        or len(groups) != result_count
        or len(representatives) != result_count
        or len(value_records) != result_count
    ):
        raise BackendIntegrationError("FastVID per-stage decision counts drifted")

    frame_segment_sizes: list[int] = []
    for size in segments:
        if size < 1:
            raise BackendIntegrationError("FastVID emitted an empty DySeg segment")
        frame_segment_sizes.extend((size,) * size)
    assignment_occurrences: dict[int, int] = {}
    kind_counts = {
        "salient": 0,
        "single_frame_dtm": 0,
        "multi_frame_dtm": 0,
    }
    decision_records: list[dict[str, Any]] = []
    per_frame_kind_counts = [
        {"salient": 0, "single_frame_dtm": 0, "multi_frame_dtm": 0}
        for _ in range(frame_count)
    ]
    for record in value_records:
        if not isinstance(record, Mapping):
            raise BackendIntegrationError("FastVID value-lineage record changed")
        kind = str(record.get("kind"))
        if kind not in kind_counts:
            raise BackendIntegrationError("FastVID value-lineage kind changed")
        target_frame = int(record.get("target_frame", -1))
        center = int(record.get("center_index", -1))
        assigned = tuple(int(item) for item in record.get("assigned_indices", ()))
        center_weight = float(record.get("center_weight", -1.0))
        if (
            target_frame < 0
            or target_frame >= frame_count
            or center < 0
            or center >= source_count
            or any(item < 0 or item >= source_count for item in assigned)
            or not 0.0 < center_weight <= 1.0
        ):
            raise BackendIntegrationError("FastVID value-lineage record is invalid")
        expected_kind = (
            "salient"
            if kind == "salient"
            else (
                "single_frame_dtm"
                if frame_segment_sizes[target_frame] == 1
                else "multi_frame_dtm"
            )
        )
        if kind != expected_kind:
            raise BackendIntegrationError(
                "FastVID DTM path does not agree with its DySeg decision"
            )
        kind_counts[kind] += 1
        per_frame_kind_counts[target_frame][kind] += 1
        for source in assigned:
            assignment_occurrences[source] = assignment_occurrences.get(source, 0) + 1
        decision_records.append(
            {
                "kind": kind,
                "segment_index": (
                    None if kind == "salient" else int(record.get("segment_index", -1))
                ),
                "target_frame": target_frame,
                "center_index": center,
                "assigned_indices": assigned,
                "center_weight": center_weight,
            }
        )

    expected_frame_counts = tuple(
        sum(per_frame_kind_counts[frame].values()) for frame in range(frame_count)
    )
    dtm_per_frame = tuple(
        per_frame_kind_counts[frame]["single_frame_dtm"]
        + per_frame_kind_counts[frame]["multi_frame_dtm"]
        for frame in range(frame_count)
    )
    if expected_frame_counts != frame_output_counts or dtm_per_frame != allocations:
        raise BackendIntegrationError(
            "FastVID STPrune/DTM record counts do not match their decisions"
        )
    if tuple(record["center_index"] for record in decision_records) != representatives:
        raise BackendIntegrationError(
            "FastVID sampled centers no longer match layout representatives"
        )

    return {
        "invocation_id": getattr(request, "invocation_id", None),
        "media_id": getattr(request, "media_id", None),
        "source_count": source_count,
        "result_count": result_count,
        "canonical_config": dict(metadata.get("canonical_config", {})),
        "stages": tuple(metadata.get("stages", ())),
        "projected_features_shape": tuple(int(item) for item in features.shape),
        "raw_features_shape": tuple(int(item) for item in raw_features.shape),
        "frame_global_features_shape": tuple(int(item) for item in frame_globals.shape),
        "token_saliency_attention_shape": tuple(int(item) for item in saliency.shape),
        **finite,
        "signal_source": getattr(payload, "metadata", {}).get("fastvid_signal_source"),
        "raw_stream": getattr(payload, "metadata", {}).get("fastvid_raw_stream"),
        "frame_count": frame_count,
        "projected_tokens_per_frame": projected_counts[0],
        "raw_tokens_per_frame": raw_counts[0],
        "segment_sizes": segments,
        "segment_count": len(segments),
        "single_frame_segment_count": sum(size == 1 for size in segments),
        "multi_frame_segment_count": sum(size > 1 for size in segments),
        "context_allocations": allocations,
        "frame_output_counts": frame_output_counts,
        "salient_count": kind_counts["salient"],
        "single_frame_dtm_count": kind_counts["single_frame_dtm"],
        "multi_frame_dtm_count": kind_counts["multi_frame_dtm"],
        "dtm_count": (kind_counts["single_frame_dtm"] + kind_counts["multi_frame_dtm"]),
        "representative_indices": representatives,
        "representatives_source_ordered": representatives
        == tuple(sorted(representatives)),
        "sampled_center_singleton_layout": all(
            group == (representative,)
            for group, representative in zip(groups, representatives)
        ),
        "value_lineage_record_count": len(decision_records),
        "value_lineage_records": tuple(decision_records),
        "assigned_source_count": len(assignment_occurrences),
        "overlapping_assigned_source_count": sum(
            count > 1 for count in assignment_occurrences.values()
        ),
        "max_assignment_multiplicity": max(assignment_occurrences.values(), default=0),
        "dyseg_boundary_rule": metadata.get("dyseg_boundary_rule"),
        "stprune_rule": metadata.get("stprune_rule"),
        "density_tie_policy": metadata.get("density_tie_policy"),
        "dtm_merge_rule": metadata.get("dtm_merge_rule"),
        "layout_source_groups": metadata.get("layout_source_groups"),
        "layout_anchor_policy": metadata.get("layout_anchor_policy"),
        "value_lineage_policy": metadata.get("value_lineage_policy"),
    }


def _load_frozen_siglip_pooling_head(source: Path) -> Any:
    torch = _torch()
    config_path = source / "config.json"
    if not config_path.is_file():
        raise BackendIntegrationError(
            f"FastVID SigLIP config is missing: {config_path}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if isinstance(config.get("vision_config"), Mapping):
        config = dict(config["vision_config"])
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    heads = int(config["num_attention_heads"])
    eps = float(config.get("layer_norm_eps", 1e-6))
    hidden_act = str(config.get("hidden_act", "gelu_pytorch_tanh"))

    from transformers.activations import ACT2FN

    class _PoolingHead(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.post_layernorm = torch.nn.LayerNorm(hidden_size, eps=eps)
            self.probe = torch.nn.Parameter(torch.empty(1, 1, hidden_size))
            self.attention = torch.nn.MultiheadAttention(
                hidden_size,
                heads,
                batch_first=True,
            )
            self.head_layernorm = torch.nn.LayerNorm(hidden_size, eps=eps)
            self.fc1 = torch.nn.Linear(hidden_size, intermediate_size)
            self.fc2 = torch.nn.Linear(intermediate_size, hidden_size)
            self.activation = ACT2FN[hidden_act]

        def forward(self, raw_features: Any) -> tuple[Any, Any]:
            hidden = self.post_layernorm(raw_features)
            probe = self.probe.repeat(int(hidden.shape[0]), 1, 1)
            attended, weights = self.attention(
                probe,
                hidden,
                hidden,
                need_weights=True,
                average_attn_weights=True,
            )
            residual = attended
            attended = self.head_layernorm(attended)
            attended = residual + self.fc2(self.activation(self.fc1(attended)))
            return attended[:, 0], weights

    module = _PoolingHead()
    suffixes = {
        "post_layernorm.weight": "vision_model.post_layernorm.weight",
        "post_layernorm.bias": "vision_model.post_layernorm.bias",
        "probe": "vision_model.head.probe",
        "attention.in_proj_weight": "vision_model.head.attention.in_proj_weight",
        "attention.in_proj_bias": "vision_model.head.attention.in_proj_bias",
        "attention.out_proj.weight": "vision_model.head.attention.out_proj.weight",
        "attention.out_proj.bias": "vision_model.head.attention.out_proj.bias",
        "head_layernorm.weight": "vision_model.head.layernorm.weight",
        "head_layernorm.bias": "vision_model.head.layernorm.bias",
        "fc1.weight": "vision_model.head.mlp.fc1.weight",
        "fc1.bias": "vision_model.head.mlp.fc1.bias",
        "fc2.weight": "vision_model.head.mlp.fc2.weight",
        "fc2.bias": "vision_model.head.mlp.fc2.bias",
    }
    tensors = _read_safetensor_suffixes(source, tuple(suffixes.values()))
    state = {name: tensors[suffix] for name, suffix in suffixes.items()}
    missing, unexpected = module.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise BackendIntegrationError(
            f"FastVID SigLIP pooling-head state mismatch: missing={missing}, "
            f"unexpected={unexpected}"
        )
    module.requires_grad_(False)
    module.eval()
    return module


def _read_safetensor_suffixes(
    source: Path,
    suffixes: tuple[str, ...],
) -> dict[str, Any]:
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - required existing environment
        raise BackendIntegrationError(
            "FastVID requires the existing safetensors dependency"
        ) from exc
    files = tuple(sorted(source.glob("*.safetensors")))
    if not files:
        raise BackendIntegrationError(
            f"FastVID found no local SigLIP safetensors under {source}"
        )
    found: dict[str, Any] = {}
    for file in files:
        with safe_open(str(file), framework="pt", device="cpu") as stream:
            for key in stream.keys():
                matches = tuple(
                    suffix for suffix in suffixes if str(key).endswith(suffix)
                )
                if not matches:
                    continue
                if len(matches) != 1 or matches[0] in found:
                    raise BackendIntegrationError(
                        "FastVID SigLIP pooling-head tensor names are ambiguous"
                    )
                found[matches[0]] = stream.get_tensor(key)
    missing = tuple(suffix for suffix in suffixes if suffix not in found)
    if missing:
        raise BackendIntegrationError(
            f"FastVID SigLIP checkpoint is missing pooling-head tensors: {missing}"
        )
    return found


def _validated_config(stage: Any, binding_config: Any) -> Mapping[str, Any]:
    stage_config = getattr(stage, "config", {}) if stage is not None else {}
    if stage_config is None:
        stage_config = {}
    if not isinstance(stage_config, Mapping):
        raise BackendIntegrationError("FastVID stage config must be a mapping")
    binding_values = (
        getattr(binding_config, "config", {}) if binding_config is not None else {}
    )
    if binding_values not in ({}, None):
        raise BackendIntegrationError(
            "FastVID video binding does not accept binding-local config"
        )
    return resolve_canonical_video_config("fastvid", stage_config)


def _validate_binding(
    binding: Any,
    stage: Any,
    backbone: str,
    phase: ExecutionPhase,
) -> None:
    expected = f"fastvid.video.{backbone}"
    if binding is not None:
        if getattr(binding, "name", None) != expected:
            raise BackendIntegrationError(
                f"FastVID executor requires binding {expected!r}"
            )
        if phase not in getattr(binding, "supported_phases", ()):
            raise BackendIntegrationError(
                f"FastVID binding {expected!r} does not support {phase.value}"
            )
        targets = tuple(
            getattr(item, "name", None) for item in binding.ordered_targets()
        )
        if targets != ("dyseg_stprune_dtm",):
            raise BackendIntegrationError("FastVID binding target changed")
    if stage is not None:
        if getattr(stage, "algorithm", None) != "fastvid":
            raise BackendIntegrationError("FastVID executor received another algorithm")
        if getattr(stage, "variant", None) != "fastvid.video":
            raise BackendIntegrationError("FastVID executor received another variant")


def _dense_frame_spans(
    coordinates: tuple[Any, ...],
) -> tuple[tuple[int, int, int], ...]:
    if not coordinates:
        raise BackendIntegrationError("FastVID payload has no coordinates")
    spans: list[tuple[int, int, int]] = []
    start = 0
    temporal = int(coordinates[0].t)
    for index, coordinate in enumerate(coordinates[1:], start=1):
        current = int(coordinate.t)
        if current == temporal:
            continue
        spans.append((temporal, start, index))
        start = index
        temporal = current
    spans.append((temporal, start, len(coordinates)))
    return tuple(spans)


def _local_path(value: Any) -> Path | None:
    if not isinstance(value, (str, Path)):
        return None
    path = Path(value).expanduser()
    return path if path.is_dir() else None


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "FastVID video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "FastVIDVideoExecutor",
    "FastVIDVideoSignalProvider",
    "create_fastvid_video_executor",
]
