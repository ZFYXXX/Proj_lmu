"""Canonical METok video projector and progressive decoder-pruning plans.

The projector follows the pinned VideoLLM-Comp event segmentation, source-
stable keyframe selection, and event-aware bilinear pooling recipe.  Bilinear
values overlap numerically, so ``LayoutDelta`` provenance uses a disjoint
structural raster partition while metadata records the exact interpolation
operation separately.

Decoder pruning is represented by one explicit plan before each canonical
layer.  The backbone executor owns native Q/K reconstruction, masks, RoPE or
MRoPE, labels, DeepStack state, and cache repair; this module owns the frozen
event keep equation and cumulative layout bookkeeping.

PyTorch is imported lazily so registry and control-plane imports remain
dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit

from .config import (
    VIDEO_REFERENCE_REVISION,
    resolve_canonical_video_config,
    validate_video_payload_signals,
)
from .contracts import VideoCompressionResult, VideoTokenCoordinate, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "metok"
_VARIANT = "metok.video"


@dataclass(frozen=True)
class METokPruningResult:
    """One cumulative METok decoder-pruning decision before a native layer."""

    boundary_layer: int
    stage_index: int
    ratio: float
    kept_visual_indices: tuple[int, ...]
    dropped_visual_indices: tuple[int, ...]
    kept_sequence_indices: tuple[int, ...]
    retained_projector_indices: tuple[int, ...]
    dropped_projector_indices: tuple[int, ...]
    cumulative_prompt_keep_indices: tuple[int, ...]
    event_lengths_before: tuple[int, ...]
    event_lengths_after: tuple[int, ...]
    important_event_indices: tuple[int, ...]
    visual_layout_delta: LayoutDelta
    sequence_layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        for name in ("boundary_layer", "stage_index"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.ratio, bool) or not isinstance(self.ratio, (int, float)):
            raise TypeError("ratio must be numeric")
        if not 0 < float(self.ratio) < 1:
            raise ValueError("ratio must be between zero and one")

        kept_visual = _normalise_indices(
            self.kept_visual_indices,
            name="kept_visual_indices",
        )
        dropped_visual = _normalise_indices(
            self.dropped_visual_indices,
            name="dropped_visual_indices",
        )
        kept_sequence = _normalise_indices(
            self.kept_sequence_indices,
            name="kept_sequence_indices",
        )
        retained_projector = _normalise_indices(
            self.retained_projector_indices,
            name="retained_projector_indices",
        )
        dropped_projector = _normalise_indices(
            self.dropped_projector_indices,
            name="dropped_projector_indices",
        )
        cumulative_prompt = _normalise_indices(
            self.cumulative_prompt_keep_indices,
            name="cumulative_prompt_keep_indices",
        )
        event_before = _normalise_lengths(
            self.event_lengths_before,
            name="event_lengths_before",
        )
        event_after = _normalise_lengths(
            self.event_lengths_after,
            name="event_lengths_after",
        )
        important = _normalise_indices(
            self.important_event_indices,
            name="important_event_indices",
        )
        if len(event_before) != len(event_after):
            raise ValueError("METok event count must remain fixed across pruning")
        if any(index >= len(event_before) for index in important):
            raise ValueError("important_event_indices exceed the event layout")
        if set(kept_visual).intersection(dropped_visual):
            raise ValueError("kept and dropped visual indices must be disjoint")
        if not isinstance(self.visual_layout_delta, LayoutDelta):
            raise TypeError("visual_layout_delta must be a LayoutDelta")
        if not isinstance(self.sequence_layout_delta, LayoutDelta):
            raise TypeError("sequence_layout_delta must be a LayoutDelta")
        if self.visual_layout_delta.media_unit != LayoutTokenUnit.VIDEO_PATCH.value:
            raise ValueError("METok visual delta must use video_patch units")
        if self.sequence_layout_delta.media_unit != "llm_token":
            raise ValueError("METok sequence delta must use llm_token units")
        if self.visual_layout_delta.source_count != sum(event_before):
            raise ValueError("METok source event lengths do not match visual layout")
        if self.visual_layout_delta.result_count != sum(event_after):
            raise ValueError("METok result event lengths do not match visual layout")
        if self.visual_layout_delta.result_count != len(kept_visual):
            raise ValueError("METok visual keep list does not match its delta")
        if self.sequence_layout_delta.result_count != len(kept_sequence):
            raise ValueError("METok sequence keep list does not match its delta")
        if len(cumulative_prompt) != len(kept_sequence):
            raise ValueError("cumulative prompt map must match the result sequence")

        object.__setattr__(self, "ratio", float(self.ratio))
        object.__setattr__(self, "kept_visual_indices", kept_visual)
        object.__setattr__(self, "dropped_visual_indices", dropped_visual)
        object.__setattr__(self, "kept_sequence_indices", kept_sequence)
        object.__setattr__(self, "retained_projector_indices", retained_projector)
        object.__setattr__(self, "dropped_projector_indices", dropped_projector)
        object.__setattr__(self, "cumulative_prompt_keep_indices", cumulative_prompt)
        object.__setattr__(self, "event_lengths_before", event_before)
        object.__setattr__(self, "event_lengths_after", event_after)
        object.__setattr__(self, "important_event_indices", important)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class METokDecoderCall:
    """Request-local event budgets, fixed-source scores and cumulative selections."""

    llava_position_policy = "sparse_source"

    def __init__(
        self,
        projector_result,
        *,
        sequence_length,
        positions,
        pruning_layers,
        ratio,
        rules,
    ):
        self.projector_result = projector_result
        self.positions = tuple(positions)
        self.pruning_layers = tuple(pruning_layers)
        self.score_layers = (*self.pruning_layers[:2], None)
        self.ratio = ratio
        self.rules = rules
        self.event_lengths, self.important = metok_projector_events(projector_result)
        self.retained_projector = tuple(range(projector_result.result_count))
        self.cumulative_prompt = tuple(range(sequence_length))
        self.source_scores = {}
        self.completed = 0

    def score(self, values, query, keys, *, visual_positions, attention_key_mask):
        if "prune_scorer" in self.rules:
            return rule_contract.scores(
                self.rules["prune_scorer"],
                {
                    "values": values,
                    "instruction_queries": query,
                    "attention_keys": keys,
                    "visual_positions": visual_positions,
                    "attention_key_mask": attention_key_mask,
                },
                count=len(visual_positions),
                reference=values,
                name="prune_scorer",
            )
        return metok_attention_scores_from_qk(
            query,
            keys,
            visual_sequence_indices=visual_positions,
            attention_key_mask=attention_key_mask,
        )

    def select(self, stage_index, hidden_states, visual_positions):
        scores = (
            hidden_states.new_zeros(len(visual_positions))
            if stage_index == 2
            else self.source_scores.pop(stage_index).to(hidden_states.device)
        )
        result = plan_metok_pruning_stage(
            sequence_length=int(hidden_states.shape[1]),
            visual_sequence_indices=visual_positions,
            visual_scores=scores,
            projector_result=self.projector_result,
            event_lengths=self.event_lengths,
            important_event_indices=self.important,
            retained_projector_indices=self.retained_projector,
            cumulative_prompt_indices=self.cumulative_prompt,
            boundary_layer=self.pruning_layers[stage_index],
            stage_index=stage_index,
            ratio=self.ratio,
            pruning_layers=self.pruning_layers,
            application_position=self.positions[stage_index].as_dict(),
            **({"rules": self.rules} if self.rules else {}),
        )
        torch = _torch()
        for pending_stage, saved in tuple(self.source_scores.items()):
            keep = torch.tensor(
                result.kept_visual_indices, dtype=torch.long, device=saved.device
            )
            self.source_scores[pending_stage] = saved.index_select(0, keep)
        self.event_lengths = result.event_lengths_after
        self.retained_projector = result.retained_projector_indices
        self.cumulative_prompt = result.cumulative_prompt_keep_indices
        self.completed += 1
        return result


class METokVideoKernel:
    """Registry kernel for the single frozen ``metok.video`` projector."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-metok-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'metok'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'metok.video'")
        resolve_canonical_video_config(
            self.name,
            plugin_config.get("config", {}),
        )

    def __call__(
        self,
        payload: VideoTokenPayload,
        *,
        config: Mapping[str, Any] | None = None,
        rules: Mapping[str, Any] | None = None,
    ) -> VideoCompressionResult:
        return metok_video(payload, config=config, rules=rules)


METokKernel = METokVideoKernel


def metok_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run canonical event segmentation and event-aware bilinear pooling."""

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("METok projected features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError("METok features must be floating [tokens, hidden]")
    validate_video_payload_signals(_ALGORITHM, payload)
    resolved = resolve_canonical_video_config(_ALGORITHM, config)

    similarities = _frame_text_similarities(payload)
    all_features: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    segments_by_video: list[tuple[tuple[int, int], ...]] = []
    important_by_video: list[tuple[int, ...]] = []
    keyframes_by_video: list[tuple[tuple[int, ...], ...]] = []
    event_lengths_by_video: list[tuple[int, ...]] = []
    frames_by_video: list[tuple[Mapping[str, Any], ...]] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        features = payload.features[span]
        coordinates = payload.coordinates[span]
        frame_count, height, width = _rectangular_grid(coordinates)
        frame_scores = similarities[video_index]
        if (
            getattr(frame_scores, "ndim", None) != 1
            or int(frame_scores.shape[0]) != frame_count
        ):
            raise BackendIntegrationError(
                "METok frame-text similarities do not match the projected frames"
            )
        if not bool(torch.isfinite(frame_scores).all().item()):
            raise ValueError("METok frame-text similarities must be finite")

        raster = features.reshape(frame_count, height * width, -1)
        if rules and "frame_scorer" in rules:
            frame_scores = rule_contract.scores(rules["frame_scorer"],
                {"values": raster, "default_scores": frame_scores, "signals": payload.signals},
                count=frame_count, reference=features, name="frame_scorer")
        segments = _similarity_segments(
            raster.mean(dim=1),
            segment_count=int(resolved["num_segments"]),
            torch=torch,
        )
        event_scores = torch.stack(
            # Custom events keep the same ordered frame partition contract.
            [frame_scores[start:end].max() for start, end in segments]
        )
        if rules and "frame_grouper" in rules:
            grouped = rule_contract.groups(rules["frame_grouper"],
                {"values": raster, "scores": frame_scores, "group_count": len(segments)},
                count=None, candidates=range(frame_count), name="frame_grouper")
            if tuple(i for group in grouped for i in group) != tuple(range(frame_count)):
                raise BackendIntegrationError("METok frame_grouper must return contiguous ordered events")
            segments = tuple((group[0], group[-1] + 1) for group in grouped)
            event_scores = torch.stack([frame_scores[start:end].max() for start, end in segments])
        important = _stable_top_indices(
            event_scores,
            min(int(resolved["top_k_events"]), len(segments)),
            torch=torch,
        )
        important_set = set(important)
        if rules and "event_selector" in rules:
            important = rule_contract.select(rules["event_selector"],
                {"values": raster, "scores": event_scores, "events": segments, "keep_count": len(important)},
                count=len(important), candidates=range(len(segments)), name="event_selector")
            important = tuple(sorted(important))
            important_set = set(important)

        selected_keyframes: list[tuple[int, ...]] = []
        frame_key_set: set[int] = set()
        for start, end in segments:
            count = max(1, floor((end - start) * float(resolved["keyframe_ratio"])))
            local = _stable_top_indices(
                frame_scores[start:end],
                min(count, end - start),
                torch=torch,
            )
            source_sorted = tuple(start + index for index in sorted(local))
            if rules and "keyframe_selector" in rules:
                local = rule_contract.select(rules["keyframe_selector"],
                    {"values": raster[start:end], "scores": frame_scores[start:end], "keep_count": min(count, end-start)},
                    count=min(count, end-start), candidates=range(end-start), name="keyframe_selector")
                source_sorted = tuple(start + index for index in sorted(local))
            selected_keyframes.append(source_sorted)
            frame_key_set.update(source_sorted)

        compressed_rows: list[Any] = []
        groups: list[tuple[int, ...]] = []
        representatives: list[int] = []
        event_lengths: list[int] = []
        frame_records: list[Mapping[str, Any]] = []
        for event_index, (start, end) in enumerate(segments):
            event_start = len(compressed_rows)
            is_important = event_index in important_set
            for frame_index in range(start, end):
                is_keyframe = frame_index in frame_key_set
                stride = _projector_stride(
                    important=is_important,
                    keyframe=is_keyframe,
                    config=resolved,
                )
                output_height = ceil(height / stride)
                output_width = ceil(width / stride)
                source = raster[frame_index].reshape(height, width, -1)
                pooled = (
                    torch.nn.functional.interpolate(
                        source.permute(2, 0, 1).unsqueeze(0),
                        size=(output_height, output_width),
                        mode="bilinear",
                        align_corners=False,
                    )[0]
                    .permute(1, 2, 0)
                    .reshape(-1, int(source.shape[-1]))
                )
                structural_groups = _structural_raster_partition(
                    frame_index=frame_index,
                    source_height=height,
                    source_width=width,
                    output_height=output_height,
                    output_width=output_width,
                )
                if int(pooled.shape[0]) != len(structural_groups):
                    raise InvariantError("METok bilinear layout cardinality changed")
                if rules and "pool_transform" in rules:
                    pooled = _custom_pool(source, output_height, output_width, rules["pool_transform"], torch=torch)
                result_start = len(compressed_rows)
                compressed_rows.extend(pooled.unbind(0))
                groups.extend(structural_groups)
                representatives.extend(group[0] for group in structural_groups)
                frame_records.append(
                    MappingProxyType(
                        {
                            "frame_index": frame_index,
                            "event_index": event_index,
                            "important": is_important,
                            "keyframe": is_keyframe,
                            "stride": stride,
                            "source_shape": (height, width),
                            "result_shape": (output_height, output_width),
                            "result_span": (
                                result_start,
                                result_start + len(structural_groups),
                            ),
                            "interpolation": "custom" if rules and "pool_transform" in rules else "bilinear",
                            "align_corners": False,
                        }
                    )
                )
            event_lengths.append(len(compressed_rows) - event_start)

        if not compressed_rows:
            raise InvariantError("METok produced an empty projector layout")
        compressed = torch.stack(compressed_rows, dim=0)
        all_features.append(compressed)
        groups_by_video.append(tuple(groups))
        representatives_by_video.append(tuple(representatives))
        segments_by_video.append(segments)
        important_by_video.append(important)
        keyframes_by_video.append(tuple(selected_keyframes))
        event_lengths_by_video.append(tuple(event_lengths))
        frames_by_video.append(tuple(frame_records))

    compressed_features = torch.cat(all_features, dim=0)
    return VideoCompressionResult.from_source_groups(
        payload,
        compressed_features,
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=tuple(len(groups) for groups in groups_by_video),
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": _plain_config(resolved),
            "projector_stage": "event_segment_keyframe_adaptive_pool",
            "segmentation_tie_policy": "custom" if rules and ('frame_grouper' in rules) else "stable_source_index",
            "importance_reduction": "custom" if rules and ('frame_scorer' in rules) else "max_frame_text_similarity_per_event",
            "keyframe_count": "max_one_floor_event_frames_times_ratio",
            "output_order": "event_frame_raster_source_order",
            "value_operation": "custom" if rules and ('pool_transform' in rules) else "bilinear_interpolate_align_corners_false",
            "layout_source_groups": "disjoint_structural_raster_partition",
            "metok_segments": tuple(segments_by_video),
            "metok_important_events": tuple(important_by_video),
            "metok_keyframes": tuple(keyframes_by_video),
            "metok_event_lengths": tuple(event_lengths_by_video),
            "metok_frames": tuple(frames_by_video),
            "decoder_pruning_required": True,
        },
    )


def replay_metok_projector_layout(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Replay METok's exact per-frame bilinear operations on a synced branch."""

    torch = _torch()
    if not isinstance(features, torch.Tensor):
        raise TypeError("synchronized METok features must be a torch.Tensor")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError(
            "synchronized METok features must be floating [tokens, hidden]"
        )
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    if int(features.shape[0]) != source_span.stop - source_span.start:
        raise InvariantError("synchronized METok branch differs from source layout")
    records = _frame_records(result, video_index=video_index)
    rows: list[Any] = []
    for record in records:
        frame = int(record["frame_index"])
        height, width = tuple(int(value) for value in record["source_shape"])
        output_height, output_width = tuple(
            int(value) for value in record["result_shape"]
        )
        start = frame * height * width
        source = features[start : start + height * width].reshape(height, width, -1)
        pooled = (
            torch.nn.functional.interpolate(
                source.permute(2, 0, 1).unsqueeze(0),
                size=(output_height, output_width),
                mode="bilinear",
                align_corners=False,
            )[0]
            .permute(1, 2, 0)
            .reshape(-1, int(source.shape[-1]))
        )
        rows.append(pooled)
        if rules and "pool_transform" in rules:
            rows[-1] = _custom_pool(source, output_height, output_width, rules["pool_transform"], torch=torch)
    if not rows:
        raise InvariantError("METok result is missing frame replay records")
    replayed = torch.cat(rows, dim=0)
    span = result.video_span(video_index)
    if int(replayed.shape[0]) != span.stop - span.start:
        raise InvariantError("METok replay did not reproduce projector cardinality")
    return replayed


def _custom_pool(source: Any, height: int, width: int, rule: Any, *, torch: Any) -> Any:
    def default_transform():
        return torch.nn.functional.interpolate(source.permute(2, 0, 1).unsqueeze(0),
            size=(height, width), mode="bilinear", align_corners=False)[0].permute(1, 2, 0).reshape(-1, int(source.shape[-1]))
    return rule_contract.value(rule, {"values": source, "output_shape": (height, width),
        "default_transform": default_transform}, reference=source,
        shape=(height * width, int(source.shape[-1])), name="pool_transform")


def metok_projector_events(
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return validated projector event lengths and important event indices."""

    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    if isinstance(video_index, bool) or not isinstance(video_index, int):
        raise TypeError("video_index must be an integer")
    if not 0 <= video_index < result.video_count:
        raise IndexError("video_index is outside the METok result")
    raw_lengths = result.metadata.get("metok_event_lengths")
    raw_important = result.metadata.get("metok_important_events")
    if not isinstance(raw_lengths, Sequence) or len(raw_lengths) != result.video_count:
        raise InvariantError("METok result is missing event-length metadata")
    if (
        not isinstance(raw_important, Sequence)
        or len(raw_important) != result.video_count
    ):
        raise InvariantError("METok result is missing important-event metadata")
    lengths = _normalise_lengths(raw_lengths[video_index], name="event_lengths")
    important = _normalise_indices(
        raw_important[video_index],
        name="important_event_indices",
    )
    span = result.video_span(video_index)
    if sum(lengths) != span.stop - span.start:
        raise InvariantError("METok event lengths do not partition its result")
    if any(index >= len(lengths) for index in important):
        raise InvariantError("METok important-event metadata is out of range")
    return lengths, important


def metok_attention_scores_from_qk(
    instruction_queries: Any,
    attention_keys: Any,
    *,
    visual_sequence_indices: Iterable[int],
    attention_key_mask: Any | None = None,
) -> Any:
    """Compute full-key probabilities averaged over text queries and heads."""

    torch = _torch()
    if not isinstance(instruction_queries, torch.Tensor) or not isinstance(
        attention_keys, torch.Tensor
    ):
        raise TypeError("METok Q/K inputs must be torch tensors")
    if instruction_queries.ndim != 3 or attention_keys.ndim != 3:
        raise ValueError("METok Q/K inputs must have rank three")
    if tuple(instruction_queries.shape[1:]) != tuple(attention_keys.shape[1:]):
        raise ValueError("METok query and key head layouts differ")
    if int(instruction_queries.shape[0]) < 1:
        raise BackendIntegrationError("METok requires post-visual text queries")
    visual = _normalise_indices(
        visual_sequence_indices,
        name="visual_sequence_indices",
    )
    if not visual or visual[-1] >= int(attention_keys.shape[0]):
        raise ValueError("METok visual indices exceed the key layout")
    logits = torch.einsum(
        "qhd,thd->tqh",
        instruction_queries,
        attention_keys,
    ) * (float(instruction_queries.shape[-1]) ** -0.5)
    if attention_key_mask is not None:
        visibility = torch.as_tensor(attention_key_mask, device=logits.device).bool()
        expected = (
            int(instruction_queries.shape[0]),
            int(attention_keys.shape[0]),
        )
        if tuple(int(value) for value in visibility.shape) != expected:
            raise ValueError("METok attention_key_mask must be [queries, sequence]")
        if bool((~visibility.any(dim=-1)).any().item()):
            raise ValueError("Every METok query must see at least one key")
        logits = logits.masked_fill(
            ~visibility.transpose(0, 1).unsqueeze(-1),
            float("-inf"),
        )
    probabilities = torch.softmax(logits, dim=0, dtype=torch.float32).to(logits.dtype)
    index = torch.tensor(visual, dtype=torch.long, device=logits.device)
    return probabilities.index_select(0, index).mean(dim=1).mean(dim=1)


def plan_metok_pruning_stage(
    *,
    sequence_length: int,
    visual_sequence_indices: Iterable[int],
    visual_scores: Any,
    projector_result: VideoCompressionResult,
    event_lengths: Iterable[int],
    important_event_indices: Iterable[int],
    retained_projector_indices: Iterable[int] | None = None,
    cumulative_prompt_indices: Iterable[int] | None = None,
    boundary_layer: int,
    stage_index: int,
    ratio: float = 0.76,
    pruning_layers: Iterable[int] = (3, 10, 19),
    application_position: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> METokPruningResult:
    """Build one progressive full-sequence and video-only METok layout."""

    torch = _torch()
    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    layers = _normalise_indices(pruning_layers, name="pruning_layers")
    if not layers or tuple(sorted(layers)) != layers:
        raise ValueError("pruning_layers must be non-empty and increasing")
    if isinstance(stage_index, bool) or not isinstance(stage_index, int):
        raise TypeError("stage_index must be an integer")
    if not 0 <= stage_index < len(layers):
        raise ValueError("stage_index is outside pruning_layers")
    if boundary_layer != layers[stage_index]:
        raise ValueError("boundary_layer does not match the METok stage")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise TypeError("ratio must be numeric")
    if not 0 < float(ratio) < 1:
        raise ValueError("ratio must be between zero and one")
    if not isinstance(projector_result, VideoCompressionResult):
        raise TypeError("projector_result must be a VideoCompressionResult")
    if projector_result.video_count != 1:
        raise BackendIntegrationError("METok decoder pruning requires one video")

    visual_positions = _normalise_indices(
        visual_sequence_indices,
        name="visual_sequence_indices",
    )
    if visual_positions and visual_positions[-1] >= sequence_length:
        raise ValueError("visual_sequence_indices exceed sequence_length")
    lengths = _normalise_lengths(event_lengths, name="event_lengths")
    if sum(lengths) != len(visual_positions):
        raise InvariantError("METok event lengths do not match active visual tokens")
    important = _normalise_indices(
        important_event_indices,
        name="important_event_indices",
    )
    if any(index >= len(lengths) for index in important):
        raise ValueError("important_event_indices exceed the event layout")

    if not isinstance(visual_scores, torch.Tensor):
        raise TypeError("visual_scores must be a torch.Tensor")
    if visual_scores.ndim != 1 or int(visual_scores.shape[0]) != len(visual_positions):
        raise ValueError("visual_scores must match the active visual layout")
    if not visual_scores.is_floating_point() or not bool(
        torch.isfinite(visual_scores).all().item()
    ):
        raise ValueError("visual_scores must be finite floating values")

    if retained_projector_indices is None:
        retained_projector = tuple(range(projector_result.result_count))
    else:
        retained_projector = _normalise_indices(
            retained_projector_indices,
            name="retained_projector_indices",
        )
    if len(retained_projector) != len(visual_positions):
        raise InvariantError("METok cumulative projector map changed")
    if retained_projector and retained_projector[-1] >= projector_result.result_count:
        raise ValueError("retained_projector_indices exceed the projector result")
    if cumulative_prompt_indices is None:
        cumulative_prompt = tuple(range(sequence_length))
    else:
        cumulative_prompt = _normalise_indices(
            cumulative_prompt_indices,
            name="cumulative_prompt_indices",
        )
    if len(cumulative_prompt) != sequence_length:
        raise InvariantError("METok cumulative prompt map changed")

    important_set = set(important)
    kept_visual_parts: list[int] = []
    result_lengths: list[int] = []
    cursor = 0
    final_stage = stage_index == len(layers) - 1
    for event_index, length in enumerate(lengths):
        if final_stage or length == 0:
            keep_count = 0
        elif event_index in important_set:
            keep_count = min(length, max(1, ceil(float(ratio) * length)))
        elif stage_index == 0:
            keep_count = min(length, max(1, ceil(float(ratio) * length / 2)))
        else:
            keep_count = 0
        if rules and "prune_selector" in rules:
            local = rule_contract.select(rules["prune_selector"],
                {"scores": visual_scores[cursor:cursor + length], "keep_count": keep_count,
                 "stage_index": stage_index, "event_index": event_index,
                 "application_position": application_position},
                count=keep_count, candidates=range(length), name="prune_selector")
            kept_visual_parts.extend(cursor + index for index in sorted(local))
        elif keep_count:
            local = _stable_top_indices(
                visual_scores[cursor : cursor + length],
                keep_count,
                torch=torch,
            )
            kept_visual_parts.extend(cursor + index for index in sorted(local))
        result_lengths.append(keep_count)
        cursor += length

    kept_visual = tuple(kept_visual_parts)
    kept_visual_set = set(kept_visual)
    dropped_visual = tuple(
        index for index in range(len(visual_positions)) if index not in kept_visual_set
    )
    retained_visual_positions = {visual_positions[index] for index in kept_visual}
    visual_position_set = set(visual_positions)
    kept_sequence = tuple(
        index
        for index in range(sequence_length)
        if index not in visual_position_set or index in retained_visual_positions
    )
    next_projector = tuple(retained_projector[index] for index in kept_visual)
    next_projector_set = set(next_projector)
    dropped_projector = tuple(
        index
        for index in range(projector_result.result_count)
        if index not in next_projector_set
    )
    next_prompt = tuple(cumulative_prompt[index] for index in kept_sequence)

    visual_targets = {source: target for target, source in enumerate(kept_visual)}
    visual_mapping = {
        source: visual_targets.get(source) for source in range(len(visual_positions))
    }
    sequence_targets = {source: target for target, source in enumerate(kept_sequence)}
    sequence_mapping = {
        source: sequence_targets.get(source) for source in range(sequence_length)
    }
    current_coordinates = tuple(
        projector_result.coordinates[index] for index in retained_projector
    )
    media_id = projector_result.media_ids[0]
    stage_name = f"before_layer_{boundary_layer}"
    visual_delta = LayoutDelta.from_mapping(
        source_count=len(visual_positions),
        result_count=len(kept_visual),
        old_to_new=visual_mapping,
        media_unit=LayoutTokenUnit.VIDEO_PATCH,
        coordinates=current_coordinates,
        source_layout={
            "layout_id": f"video:{media_id}:metok:{stage_name}:source",
            "stream_id": f"video:{media_id}:metok:{stage_name}:source",
            "media_id": media_id,
            "stage": stage_name,
            **({"application_position": dict(application_position)} if application_position is not None else {}),
        },
        result_layout={
            "layout_id": f"video:{media_id}:metok:{stage_name}:result",
            "stream_id": f"video:{media_id}:metok:{stage_name}:result",
            "media_id": media_id,
            "stage": stage_name,
            **({"application_position": dict(application_position)} if application_position is not None else {}),
            "retained_projector_indices": next_projector,
            "coordinates": tuple(
                projector_result.coordinates[index].as_tuple()
                for index in next_projector
            ),
        },
    )
    sequence_delta = LayoutDelta.from_mapping(
        source_count=sequence_length,
        result_count=len(kept_sequence),
        old_to_new=sequence_mapping,
        media_unit="llm_token",
        source_layout={
            "layout_id": f"decoder:metok:{stage_name}:source",
            "stream_id": f"decoder:metok:{stage_name}:source",
            "stage": stage_name,
            **({"application_position": dict(application_position)} if application_position is not None else {}),
        },
        result_layout={
            "layout_id": f"decoder:metok:{stage_name}:result",
            "stream_id": f"decoder:metok:{stage_name}:result",
            "stage": stage_name,
            **({"application_position": dict(application_position)} if application_position is not None else {}),
            "kept_sequence_indices": kept_sequence,
            "cumulative_prompt_keep_indices": next_prompt,
        },
    )
    return METokPruningResult(
        boundary_layer=boundary_layer,
        stage_index=stage_index,
        ratio=float(ratio),
        kept_visual_indices=kept_visual,
        dropped_visual_indices=dropped_visual,
        kept_sequence_indices=kept_sequence,
        retained_projector_indices=next_projector,
        dropped_projector_indices=dropped_projector,
        cumulative_prompt_keep_indices=next_prompt,
        event_lengths_before=lengths,
        event_lengths_after=tuple(result_lengths),
        important_event_indices=important,
        visual_layout_delta=visual_delta,
        sequence_layout_delta=sequence_delta,
        metadata={
            "algorithm": _VARIANT,
            "stage": stage_name,
            **({"application_position": dict(application_position)} if application_position is not None else {}),
            "boundary": "before_layer",
            "attention_reduction": "custom" if rules and ('prune_scorer' in rules) else "mean_post_visual_queries_then_heads",
            "selection": "custom" if rules and ('prune_selector' in rules) else "stable_event_top_attention_restore_source_order",
            "cache_policy": "compact_completed_layers_before_boundary",
            "position_policy": "gather_native_positions",
            "source_visual_count": len(visual_positions),
            "result_visual_count": len(kept_visual),
            "final_visual_removal": final_stage,
            "score_layer": layers[stage_index] if not final_stage else None,
        },
    )


def _frame_records(
    result: VideoCompressionResult,
    *,
    video_index: int,
) -> tuple[Mapping[str, Any], ...]:
    raw = result.metadata.get("metok_frames")
    if not isinstance(raw, Sequence) or len(raw) != result.video_count:
        raise InvariantError("METok result is missing frame-layout metadata")
    records = tuple(raw[video_index])
    if not all(isinstance(record, Mapping) for record in records):
        raise InvariantError("METok frame-layout metadata is malformed")
    return records


def _frame_text_similarities(payload: VideoTokenPayload) -> tuple[Any, ...]:
    raw = payload.signals.get("metok.frame_text_similarity")
    if raw is None:
        raise BackendIntegrationError(
            "METok provider did not attach frame-text similarities"
        )
    if payload.video_count == 1 and getattr(raw, "ndim", None) == 1:
        values = (raw,)
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(raw)
    else:
        raise BackendIntegrationError("METok frame-text similarity signal is malformed")
    if len(values) != payload.video_count:
        raise BackendIntegrationError(
            "METok frame-text similarity count differs from video count"
        )
    return values


def _similarity_segments(
    frame_means: Any,
    *,
    segment_count: int,
    torch: Any,
) -> tuple[tuple[int, int], ...]:
    frame_count = int(frame_means.shape[0])
    if frame_count < 1:
        raise ValueError("METok requires at least one frame")
    count = min(segment_count, frame_count)
    if count == 1:
        return ((0, frame_count),)
    normalized = torch.nn.functional.normalize(frame_means.float(), dim=-1)
    adjacent = (normalized[:-1] * normalized[1:]).sum(dim=-1)
    cut_count = count - 1
    order = torch.argsort(adjacent, descending=False, stable=True)
    cuts = tuple(sorted(int(value) + 1 for value in order[:cut_count].tolist()))
    boundaries = (0, *cuts, frame_count)
    return tuple(
        (boundaries[index], boundaries[index + 1])
        for index in range(len(boundaries) - 1)
    )


def _projector_stride(
    *,
    important: bool,
    keyframe: bool,
    config: Mapping[str, Any],
) -> int:
    if important:
        key = "s1_important" if keyframe else "s2_important"
    else:
        key = "s1_normal" if keyframe else "s2_normal"
    return int(config[key])


def _structural_raster_partition(
    *,
    frame_index: int,
    source_height: int,
    source_width: int,
    output_height: int,
    output_width: int,
) -> tuple[tuple[int, ...], ...]:
    groups: list[list[int]] = [[] for _ in range(output_height * output_width)]
    frame_offset = frame_index * source_height * source_width
    for row in range(source_height):
        target_row = min((row * output_height) // source_height, output_height - 1)
        for column in range(source_width):
            target_column = min(
                (column * output_width) // source_width,
                output_width - 1,
            )
            target = target_row * output_width + target_column
            groups[target].append(frame_offset + row * source_width + column)
    if any(not group for group in groups):
        raise InvariantError("METok structural raster partition produced an empty cell")
    return tuple(tuple(group) for group in groups)


def _rectangular_grid(
    coordinates: Sequence[VideoTokenCoordinate],
) -> tuple[int, int, int]:
    if not coordinates:
        raise ValueError("METok requires non-empty video coordinates")
    temporal = tuple(sorted({coordinate.t for coordinate in coordinates}))
    if temporal != tuple(range(len(temporal))):
        raise BackendIntegrationError("METok temporal coordinates must be dense")
    height = max(coordinate.h for coordinate in coordinates) + 1
    width = max(coordinate.w for coordinate in coordinates) + 1
    expected = tuple(
        (t, h, w) for t in temporal for h in range(height) for w in range(width)
    )
    actual = tuple(
        (coordinate.t, coordinate.h, coordinate.w) for coordinate in coordinates
    )
    if actual != expected:
        raise BackendIntegrationError(
            "METok requires a source-ordered rectangular projected raster"
        )
    return len(temporal), height, width


def _stable_top_indices(values: Any, count: int, *, torch: Any) -> tuple[int, ...]:
    if count <= 0:
        return ()
    if count > int(values.shape[0]):
        raise ValueError("stable top-k count exceeds the source values")
    order = torch.argsort(values, descending=True, stable=True)
    return tuple(sorted(int(value) for value in order[:count].tolist()))


def _normalise_indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be integer indices")
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{name} entries must be integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} entries must be non-negative")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} must not contain duplicates")
    if tuple(sorted(result)) != result:
        raise ValueError(f"{name} must preserve source order")
    return result


def _normalise_lengths(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be integer lengths")
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{name} entries must be integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} entries must be non-negative")
    return result


def _plain_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            list(item)
            if isinstance(item, tuple)
            else dict(item) if isinstance(item, Mapping) else item
        )
        for key, item in value.items()
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "METok execution requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "METokKernel",
    "METokPruningResult",
    "METokVideoKernel",
    "metok_attention_scores_from_qk",
    "metok_projector_events",
    "metok_video",
    "plan_metok_pruning_stage",
    "replay_metok_projector_layout",
]
