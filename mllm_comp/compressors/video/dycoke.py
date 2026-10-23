"""Canonical DyCoKe temporal selection and decode-time KV-view policy.

The pinned recipe is training-free and has two inseparable stages.  TTM keeps
the first frame in every group of four intact and retains the spatially
corresponding least-similar tokens from the other three frames.  During
autoregressive decode, layer 2 supplies full-cache attention and layers 3+
consume a sparse KV *view*.  The physical DynamicCache remains complete.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenPayload
from . import _rules as rule_contract
from .visionzip import _temporal_spans, _validate_uniform_raster


_ALGORITHM = "dycoke"
_VARIANT = "dycoke.video"


@dataclass(frozen=True)
class DyCoKeKVViewPlan:
    """One decode step's non-destructive sparse cache view."""

    sequence_length: int
    visual_positions: tuple[int, ...]
    kept_visual_positions: tuple[int, ...]
    view_indices: tuple[int, ...]
    attention_similarity: float | None
    reused_visual_selection: bool

    def __post_init__(self) -> None:
        if (
            isinstance(self.sequence_length, bool)
            or not isinstance(self.sequence_length, int)
            or self.sequence_length < 1
        ):
            raise ValueError("DyCoKe sequence_length must be positive")
        visual = _ordered_indices(
            "visual_positions",
            self.visual_positions,
            upper_bound=self.sequence_length,
        )
        kept = _ordered_indices(
            "kept_visual_positions",
            self.kept_visual_positions,
            upper_bound=self.sequence_length,
        )
        view = _ordered_indices(
            "view_indices",
            self.view_indices,
            upper_bound=self.sequence_length,
        )
        if not visual or not kept or not set(kept).issubset(visual):
            raise ValueError("DyCoKe kept visual positions must be a non-empty subset")
        expected = tuple(
            index
            for index in range(self.sequence_length)
            if index not in set(visual) or index in set(kept)
        )
        if view != expected:
            raise InvariantError("DyCoKe KV view did not preserve every nonvisual key")
        if self.attention_similarity is not None and not math.isfinite(self.attention_similarity):
            raise ValueError("DyCoKe attention similarity must be finite")
        if not isinstance(self.reused_visual_selection, bool):
            raise TypeError("reused_visual_selection must be boolean")

    @property
    def dropped_visual_positions(self) -> tuple[int, ...]:
        kept = set(self.kept_visual_positions)
        return tuple(index for index in self.visual_positions if index not in kept)

    def as_dict(self) -> dict[str, Any]:
        return {
            "sequence_length": self.sequence_length,
            "visual_count": len(self.visual_positions),
            "kept_visual_count": len(self.kept_visual_positions),
            "dropped_visual_count": len(self.dropped_visual_positions),
            "view_length": len(self.view_indices),
            "visual_positions": self.visual_positions,
            "kept_visual_positions": self.kept_visual_positions,
            "view_indices": self.view_indices,
            "attention_similarity": self.attention_similarity,
            "reused_visual_selection": self.reused_visual_selection,
            "physical_cache_retained": True,
        }


class DyCoKeVideoKernel:
    """Registry kernel for the single frozen DyCoKe video profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-dycoke-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'dycoke'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'dycoke.video'")
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
        result = dycoke_video(payload, config=config, rules=rules)
        return replace(result, metadata={**result.metadata, **_dycoke_projector_metadata(result)})

    def replay(self, values: Any, result: VideoCompressionResult, *,
               config: Mapping[str, Any], rules: Mapping[str, Any]) -> Any:
        """Replay the exact selected values on another native feature branch."""
        return replay_dycoke_source_groups(values, result)

    def new_decode_policy(self, *, config: Mapping[str, Any],
                          rules: Mapping[str, Any]) -> "DyCoKeDecodePolicy":
        """Each returned cache owns a fresh policy, including its selection history."""
        return DyCoKeDecodePolicy(
            drop_ratio=float(config["kv_drop_ratio"]),
            similarity_threshold=float(config["attention_similarity_threshold"]),
            rules=rules,
        )

    def runtime_metadata(self) -> dict[str, Any]:
        return {"training_free_compressor": True}


DyCoKeKernel = DyCoKeVideoKernel


def dycoke_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Apply the released two-pass TTM selection independently per video."""

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("DyCoKe payload features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError("DyCoKe payload features must be floating [tokens, hidden]")
    if not bool(torch.isfinite(payload.features).all().item()):
        raise ValueError("DyCoKe payload features must be finite")
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("DyCoKe token_mask must match projected features")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "DyCoKe requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    merging_ratio = float(resolved["merging_ratio"])
    outputs: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    exact_budgets: list[int] = []
    frame_records_by_video: list[tuple[dict[str, Any], ...]] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        source = payload.features[span]
        coordinates = payload.coordinates[span]
        temporal_spans = _temporal_spans(coordinates)
        _validate_uniform_raster(coordinates, temporal_spans)
        tokens_per_frame = temporal_spans[0][2] - temporal_spans[0][1]
        keep_count = max(1, int((1.0 - merging_ratio) * tokens_per_frame))
        selected_by_frame: list[tuple[int, ...]] = [
            tuple(range(tokens_per_frame)) for _ in temporal_spans
        ]

        for frame_index in range(0, len(temporal_spans) - 1, 2):
            first = _frame_values(source, temporal_spans[frame_index])
            second = _frame_values(source, temporal_spans[frame_index + 1])
            selected_by_frame[frame_index + 1] = _ttm_indices(first, second,
                keep_count, rules=rules, frame_pair=(frame_index, frame_index + 1), torch=torch)

        for frame_index in range(0, len(temporal_spans) - 3, 4):
            first = _frame_values(source, temporal_spans[frame_index])
            third = _frame_values(source, temporal_spans[frame_index + 2])
            selected_by_frame[frame_index + 2] = _ttm_indices(first, third,
                keep_count, rules=rules, frame_pair=(frame_index, frame_index + 2), torch=torch)

        selected: list[int] = []
        frame_records: list[dict[str, Any]] = []
        for frame_index, (temporal, start, stop) in enumerate(temporal_spans):
            local = selected_by_frame[frame_index]
            global_local = tuple(start + index for index in local)
            selected.extend(global_local)
            frame_records.append(
                {
                    "temporal_index": temporal,
                    "source_count": stop - start,
                    "result_count": len(local),
                    "selected_source_indices": global_local,
                    "selection_stage": _frame_selection_stage(
                        frame_index,
                        len(temporal_spans),
                    ),
                }
            )
        selected_tuple = tuple(selected)
        if selected_tuple != tuple(sorted(selected_tuple)):
            raise InvariantError("DyCoKe TTM must restore source token order")
        if not selected_tuple:
            raise InvariantError("DyCoKe TTM cannot produce an empty video")
        index = torch.tensor(
            selected_tuple,
            dtype=torch.long,
            device=source.device,
        )
        outputs.append(source.index_select(0, index))
        groups = tuple((source_index,) for source_index in selected_tuple)
        groups_by_video.append(groups)
        representatives_by_video.append(selected_tuple)
        exact_budgets.append(len(groups))
        frame_records_by_video.append(tuple(frame_records))

    return VideoCompressionResult.from_source_groups(
        payload,
        torch.cat(outputs, dim=0),
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=exact_budgets,
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": dict(resolved),
            "stage": "ttm_visual_merge",
            "ttm_pairing": "adjacent_then_zero_two_per_group_of_four",
            "selection": "custom" if rules and ('ttm_scorer' in rules or 'ttm_selector' in rules) else "lowest_corresponding_spatial_cosine",
            "selection_ties": "custom" if rules and ('ttm_selector' in rules) else "source_index_stable",
            "selection_output_order": "source_order_restored",
            "merge_value": "selected_source_value_no_averaging",
            "frame_records_by_video": tuple(frame_records_by_video),
            "decode_stage": "full_cache_sparse_view_from_layer_3",
            "physical_cache_retained": True,
            "training_free_compressor": True,
            "image_variant_shared": False,
        },
    )


def plan_dycoke_kv_view(
    visual_attention_scores: Any,
    *,
    visual_positions: Sequence[int],
    sequence_length: int,
    drop_ratio: float = 0.8,
    similarity_threshold: float = 0.9,
    previous_attention_scores: Any | None = None,
    previous_kept_visual_positions: Sequence[int] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> DyCoKeKVViewPlan:
    """Select the visual KV view while retaining every nonvisual cache row."""

    torch = _torch()
    if not isinstance(visual_attention_scores, torch.Tensor):
        raise TypeError("DyCoKe visual attention scores must be a tensor")
    scores = visual_attention_scores.detach().float()
    if scores.ndim != 1 or not bool(torch.isfinite(scores).all().item()):
        raise ValueError("DyCoKe visual attention scores must be finite rank one")
    if (
        isinstance(sequence_length, bool)
        or not isinstance(sequence_length, int)
        or sequence_length < 1
    ):
        raise ValueError("DyCoKe sequence_length must be positive")
    visual = _ordered_indices(
        "visual_positions",
        visual_positions,
        upper_bound=sequence_length,
    )
    if not visual or len(visual) != int(scores.numel()):
        raise ValueError("DyCoKe visual scores must match visual positions")
    if (
        isinstance(drop_ratio, bool)
        or not isinstance(drop_ratio, (int, float))
        or not 0.0 <= float(drop_ratio) < 1.0
        or isinstance(similarity_threshold, bool)
        or not isinstance(similarity_threshold, (int, float))
        or not -1.0 <= float(similarity_threshold) <= 1.0
    ):
        raise ValueError("DyCoKe drop ratio or similarity threshold is invalid")
    has_previous_scores = previous_attention_scores is not None
    has_previous_keep = previous_kept_visual_positions is not None
    if has_previous_scores != has_previous_keep:
        raise ValueError("DyCoKe previous attention and selection must be paired")

    keep_count = max(1, int(len(visual) * (1.0 - float(drop_ratio))))
    similarity = 0.0
    reused = False
    if has_previous_scores:
        if not isinstance(previous_attention_scores, torch.Tensor):
            raise TypeError("DyCoKe previous attention scores must be a tensor")
        previous = previous_attention_scores.detach().float()
        if tuple(previous.shape) != tuple(scores.shape) or not bool(
            torch.isfinite(previous).all().item()
        ):
            raise ValueError("DyCoKe previous attention scores changed shape")
        similarity_tensor = torch.nn.functional.cosine_similarity(
            scores,
            previous,
            dim=0,
        )
        similarity = float(similarity_tensor.item())
        reused = similarity >= float(similarity_threshold)

    if rules and "decode_decision" in rules:
        with torch.no_grad():
            decision = rules["decode_decision"]({"scores": scores,
                "previous_scores": previous_attention_scores,
                "previous_kept_visual_positions": previous_kept_visual_positions,
                "similarity": similarity, "default_reuse": reused,
                "keep_count": keep_count})
        if not isinstance(decision, bool) or (decision and not has_previous_scores):
            raise BackendIntegrationError("DyCoKe decode_decision must return bool and cannot reuse absent history")
        reused = decision

    if reused:
        kept = _ordered_indices(
            "previous_kept_visual_positions",
            previous_kept_visual_positions or (),
            upper_bound=sequence_length,
        )
        if len(kept) != keep_count or not set(kept).issubset(visual):
            raise ValueError("DyCoKe previous visual selection is incompatible")
    else:
        local = _stable_extreme_indices(
            scores,
            count=keep_count,
            largest=True,
            torch=torch,
        )
        kept = tuple(visual[index] for index in local)
        if rules and "decode_selector" in rules:
            kept = tuple(sorted(rule_contract.select(rules["decode_selector"],
                {"scores": scores, "visual_positions": visual, "keep_count": keep_count,
                 "sequence_length": sequence_length}, count=keep_count,
                candidates=visual, name="decode_selector")))
    visual_set = set(visual)
    kept_set = set(kept)
    view = tuple(
        index
        for index in range(sequence_length)
        if index not in visual_set or index in kept_set
    )
    return DyCoKeKVViewPlan(
        sequence_length=sequence_length,
        visual_positions=visual,
        kept_visual_positions=kept,
        view_indices=view,
        attention_similarity=similarity,
        reused_visual_selection=reused,
    )


class DyCoKeDecodePolicy:
    """DyCoKe scoring and selection history for one independent decode cache."""

    def __init__(self, *, drop_ratio: float = 0.8, similarity_threshold: float = 0.9,
                 rules: Mapping[str, Any] | None = None) -> None:
        self.drop_ratio = drop_ratio
        self.similarity_threshold = similarity_threshold
        self.rules = rules or {}
        self._previous_attention: Any | None = None
        self._previous_kept: tuple[int, ...] | None = None

    def plan_view(
        self,
        query: Any,
        full_keys: Any,
        *,
        sequence_length: int,
        visual_positions: Sequence[int],
        attention_scaling: float,
    ) -> DyCoKeKVViewPlan:
        torch = _torch()
        groups = int(query.shape[1]) // int(full_keys.shape[1])
        if groups < 1 or int(full_keys.shape[1]) * groups != int(query.shape[1]):
            raise InvariantError("DyCoKe query/KV head grouping changed")
        expanded_keys = (
            torch.repeat_interleave(full_keys, groups, dim=1)
            if groups > 1
            else full_keys
        )
        with torch.no_grad():
            logits = (
                torch.matmul(
                    query.float(),
                    expanded_keys.float().transpose(-2, -1),
                )
                * attention_scaling
            )
            attention = torch.softmax(logits, dim=-1).mean(dim=(0, 1, 2))
            visual = torch.tensor(
                visual_positions,
                dtype=torch.long,
                device=attention.device,
            )
            visual_attention = attention.index_select(0, visual)
            rules = self.rules
            if "decode_scorer" in rules:
                visual_attention = rule_contract.scores(rules["decode_scorer"],
                    {"query": query, "full_keys": full_keys, "visual_positions": visual_positions,
                     "default_scores": visual_attention, "sequence_length": sequence_length},
                    count=len(visual_positions), reference=query, name="decode_scorer")
            plan = plan_dycoke_kv_view(
                visual_attention,
                visual_positions=visual_positions,
                sequence_length=sequence_length,
                drop_ratio=self.drop_ratio,
                similarity_threshold=self.similarity_threshold,
                previous_attention_scores=self._previous_attention,
                previous_kept_visual_positions=self._previous_kept,
                rules=rules,
            )
            self._previous_attention = visual_attention.detach().clone()
            self._previous_kept = plan.kept_visual_positions
        return plan


def _dycoke_projector_metadata(result: VideoCompressionResult) -> dict[str, Any]:
    frame_records_by_video = tuple(
        tuple(dict(frame) for frame in records)
        for records in result.metadata["frame_records_by_video"]
    )
    flattened_frames = tuple(
        frame for records in frame_records_by_video for frame in records
    )
    selection_stages = tuple(frame["selection_stage"] for frame in flattened_frames)
    source_counts = tuple(delta.source_count for delta in result.layout_deltas)
    result_counts = tuple(delta.result_count for delta in result.layout_deltas)
    return {
        "video_count": result.video_count,
        "per_video_source_counts": source_counts,
        "per_video_result_counts": result_counts,
        "frame_count": len(flattened_frames),
        "frame_records_by_video": frame_records_by_video,
        "frame_source_counts": tuple(
            int(frame["source_count"]) for frame in flattened_frames
        ),
        "frame_result_counts": tuple(
            int(frame["result_count"]) for frame in flattened_frames
        ),
        "selection_stages": selection_stages,
        "group_anchor_count": selection_stages.count("group_anchor_full"),
        "adjacent_selection_count": selection_stages.count("adjacent_unique_selection"),
        "stride_zero_two_selection_count": selection_stages.count(
            "stride_zero_two_unique_selection"
        ),
        "trailing_full_count": selection_stages.count("unpaired_trailing_full"),
        "singleton_source_groups_exact": all(
            group == (representative,)
            for groups, representatives in zip(
                result.source_groups,
                result.representative_indices,
            )
            for group, representative in zip(groups, representatives)
        ),
        "representatives_source_ordered": all(
            representatives == tuple(sorted(representatives))
            and len(set(representatives)) == len(representatives)
            for representatives in result.representative_indices
        ),
    }


def replay_dycoke_source_groups(values: Any, result: VideoCompressionResult) -> Any:
    """Gather the exact TTM source selection on a Qwen3 DeepStack branch."""

    torch = _torch()
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    if not isinstance(values, torch.Tensor) or values.ndim != 2:
        raise TypeError("DyCoKe replay values must be a rank-two tensor")
    if int(values.shape[0]) != result.source_count:
        raise ValueError("DyCoKe replay values differ from the source token count")
    global_indices: list[int] = []
    for video_index, (groups, representatives) in enumerate(
        zip(result.source_groups, result.representative_indices)
    ):
        if any(
            group != (representative,)
            for group, representative in zip(groups, representatives)
        ):
            raise InvariantError("DyCoKe TTM replay requires singleton source groups")
        offset = result.source_video_offsets[video_index]
        global_indices.extend(offset + index for index in representatives)
    index = torch.tensor(global_indices, dtype=torch.long, device=values.device)
    return values.index_select(0, index)


def _ttm_indices(first: Any, second: Any, count: int, *, rules: Any, frame_pair: Any, torch: Any) -> tuple[int, ...]:
    scores = _corresponding_cosine(first, second, torch=torch)
    request = {"first_values": first, "values": second, "frame_pair": frame_pair,
               "default_scores": scores, "keep_count": count}
    if rules and "ttm_scorer" in rules:
        scores = rule_contract.scores(rules["ttm_scorer"], request,
            count=int(first.shape[0]), reference=first, name="ttm_scorer")
    if rules and "ttm_selector" in rules:
        return tuple(sorted(rule_contract.select(rules["ttm_selector"],
            {**request, "scores": scores}, count=count,
            candidates=range(int(first.shape[0])), name="ttm_selector")))
    return _stable_extreme_indices(scores, count=count, largest=False, torch=torch)


def _frame_values(source: Any, temporal_span: tuple[int, int, int]) -> Any:
    _temporal, start, stop = temporal_span
    return source[start:stop]


def _corresponding_cosine(first: Any, second: Any, *, torch: Any) -> Any:
    if tuple(first.shape) != tuple(second.shape):
        raise InvariantError("DyCoKe corresponding frame rasters changed shape")
    return torch.nn.functional.cosine_similarity(
        first.detach().float(),
        second.detach().float(),
        dim=1,
    )


def _stable_extreme_indices(
    values: Any,
    *,
    count: int,
    largest: bool,
    torch: Any,
) -> tuple[int, ...]:
    if values.ndim != 1 or not bool(torch.isfinite(values).all().item()):
        raise ValueError("DyCoKe selection scores must be finite rank one")
    if not 0 < count <= int(values.numel()):
        raise ValueError("DyCoKe selection count is outside the score vector")
    order = torch.argsort(values, descending=largest, stable=True)[:count]
    return tuple(sorted(int(index) for index in order.tolist()))


def _frame_selection_stage(frame_index: int, frame_count: int) -> str:
    if frame_index % 4 == 0:
        return "group_anchor_full"
    if frame_index % 2 == 1:
        return "adjacent_unique_selection"
    if frame_index + 1 >= frame_count:
        return "unpaired_trailing_full"
    return "stride_zero_two_unique_selection"


def _ordered_indices(
    name: str,
    values: Sequence[int],
    *,
    upper_bound: int,
) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an integer sequence")
    result = tuple(values)
    if any(
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value < upper_bound
        for value in result
    ):
        raise ValueError(f"{name} contains an out-of-range index")
    if result != tuple(sorted(set(result))):
        raise ValueError(f"{name} must be unique and source ordered")
    return result


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "DyCoKe video runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "DyCoKeDecodePolicy",
    "DyCoKeKVViewPlan",
    "DyCoKeKernel",
    "DyCoKeVideoKernel",
    "dycoke_video",
    "plan_dycoke_kv_view",
    "replay_dycoke_source_groups",
]
