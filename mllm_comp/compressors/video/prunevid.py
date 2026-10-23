"""Canonical PruneVid video reduction and VTP layout planning.

The projector kernel follows the pinned VideoLLM-Comp recipe: DPC-KNN first
forms contiguous temporal windows, each window is split into static spatial
locations and per-frame dynamic locations, and both branches use the frozen
spatial cluster ratio when their candidate count exceeds the canonical
threshold.  The reference adds random noise to DPC density ties.  This port
uses stable source-index ordering instead so repeated validation runs produce
the same layout.

VTP is represented as a second, explicit decoder-prefill layout change.  The
backbone provider owns native attention reconstruction, masks, positions, and
cache repair; this module owns the paper selection equation and publishes the
two LayoutDelta records needed by those providers.

PyTorch is imported lazily.  Importing the registry and control-plane
contracts therefore remains dependency-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from mllm_comp.core.errors import BackendIntegrationError, ConfigError, InvariantError
from mllm_comp.core.layout import LayoutDelta, LayoutTokenUnit

from .config import VIDEO_REFERENCE_REVISION, resolve_canonical_video_config
from .contracts import VideoCompressionResult, VideoTokenCoordinate, VideoTokenPayload
from . import _rules as rule_contract


_ALGORITHM = "prunevid"
_VARIANT = "prunevid.video"


@dataclass(frozen=True)
class PruneVidWindowLayout:
    """Result-token ownership for one contiguous temporal window.

    ``static_result_indices`` identifies outputs whose groups span every frame
    in the window.  ``dynamic_result_indices_by_frame`` keeps the reference
    VTP rule explicit after causal source-order legalization has interleaved
    static and dynamic outputs.
    """

    window_index: int
    source_frame_start: int
    frame_count: int
    static_result_indices: tuple[int, ...]
    dynamic_result_indices_by_frame: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        for name in ("window_index", "source_frame_start"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 0:
                raise ValueError(f"{name} must be non-negative")
        if isinstance(self.frame_count, bool) or not isinstance(self.frame_count, int):
            raise TypeError("frame_count must be an integer")
        if self.frame_count < 1:
            raise ValueError("frame_count must be positive")
        if len(self.dynamic_result_indices_by_frame) != self.frame_count:
            raise ValueError(
                "dynamic_result_indices_by_frame must contain one entry per frame"
            )
        static = _normalise_indices(
            self.static_result_indices,
            name="static_result_indices",
        )
        dynamic = tuple(
            _normalise_indices(values, name="dynamic_result_indices_by_frame")
            for values in self.dynamic_result_indices_by_frame
        )
        flattened = (*static, *(value for row in dynamic for value in row))
        if len(set(flattened)) != len(flattened):
            raise ValueError("PruneVid window result indices must be disjoint")
        object.__setattr__(self, "static_result_indices", static)
        object.__setattr__(self, "dynamic_result_indices_by_frame", dynamic)

    def record(self) -> dict[str, Any]:
        return {
            "window_index": self.window_index,
            "source_frame_start": self.source_frame_start,
            "frame_count": self.frame_count,
            "static_result_indices": self.static_result_indices,
            "dynamic_result_indices_by_frame": self.dynamic_result_indices_by_frame,
        }

    @classmethod
    def from_record(cls, value: Mapping[str, Any]) -> "PruneVidWindowLayout":
        if not isinstance(value, Mapping):
            raise TypeError("PruneVid window metadata must be a mapping")
        try:
            return cls(
                window_index=value["window_index"],
                source_frame_start=value["source_frame_start"],
                frame_count=value["frame_count"],
                static_result_indices=tuple(value["static_result_indices"]),
                dynamic_result_indices_by_frame=tuple(
                    tuple(row) for row in value["dynamic_result_indices_by_frame"]
                ),
            )
        except KeyError as exc:
            raise ValueError(
                f"PruneVid window metadata is missing {exc.args[0]!r}"
            ) from exc


@dataclass(frozen=True)
class PruneVidVTPResult:
    """One layer-10 VTP decision before backbone-specific legalization."""

    selected_layer: int
    alpha: float
    kept_visual_indices: tuple[int, ...]
    dropped_visual_indices: tuple[int, ...]
    kept_sequence_indices: tuple[int, ...]
    visual_layout_delta: LayoutDelta
    sequence_layout_delta: LayoutDelta
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        if isinstance(self.selected_layer, bool) or not isinstance(
            self.selected_layer, int
        ):
            raise TypeError("selected_layer must be an integer")
        if self.selected_layer < 0:
            raise ValueError("selected_layer must be non-negative")
        _require_alpha(self.alpha)
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
        if set(kept_visual).intersection(dropped_visual):
            raise ValueError("kept and dropped visual indices must be disjoint")
        if not isinstance(self.visual_layout_delta, LayoutDelta):
            raise TypeError("visual_layout_delta must be a LayoutDelta")
        if not isinstance(self.sequence_layout_delta, LayoutDelta):
            raise TypeError("sequence_layout_delta must be a LayoutDelta")
        if self.visual_layout_delta.media_unit != LayoutTokenUnit.VIDEO_PATCH.value:
            raise ValueError("VTP visual delta must use video_patch units")
        if self.sequence_layout_delta.media_unit != "llm_token":
            raise ValueError("VTP sequence delta must use llm_token units")
        if self.visual_layout_delta.result_count != len(kept_visual):
            raise ValueError("VTP visual delta does not match kept visual indices")
        if self.sequence_layout_delta.result_count != len(kept_sequence):
            raise ValueError("VTP sequence delta does not match kept sequence indices")
        object.__setattr__(self, "kept_visual_indices", kept_visual)
        object.__setattr__(self, "dropped_visual_indices", dropped_visual)
        object.__setattr__(self, "kept_sequence_indices", kept_sequence)
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))


class PruneVidDecoderCall:
    """One fixed-source VTP decision, independent of the native decoder loop."""

    llava_position_policy = "contiguous"

    def __init__(self, projector_result, *, positions, selected_layer, alpha, rules):
        self.projector_result = projector_result
        self.positions = tuple(positions)
        self.score_layers = (selected_layer,)
        self.selected_layer = selected_layer
        self.alpha = alpha
        self.rules = rules
        self.source_scores = {}
        self.completed = 0

    def score(self, values, query, keys, *, visual_positions, attention_key_mask):
        if "vtp_scorer" in self.rules:
            return rule_contract.scores(
                self.rules["vtp_scorer"],
                {
                    "values": values,
                    "instruction_queries": query,
                    "attention_keys": keys,
                    "visual_positions": visual_positions,
                    "attention_key_mask": attention_key_mask,
                },
                count=len(visual_positions),
                reference=values,
                name="vtp_scorer",
            )
        return prunevid_vtp_scores_from_qk(
            query,
            keys,
            visual_sequence_indices=visual_positions,
            attention_key_mask=attention_key_mask,
        )

    def select(self, stage_index, hidden_states, visual_positions):
        if stage_index not in self.source_scores:
            raise InvariantError(
                "PruneVid application preceded its fixed scoring source."
            )
        result = plan_prunevid_vtp(
            sequence_length=int(hidden_states.shape[1]),
            visual_sequence_indices=visual_positions,
            visual_scores=self.source_scores.pop(stage_index).to(hidden_states.device),
            projector_result=self.projector_result,
            selected_layer=self.selected_layer,
            alpha=self.alpha,
            application_position=self.positions[stage_index].as_dict(),
            **({"rules": self.rules} if self.rules else {}),
        )
        self.completed += 1
        return result


class PruneVidVideoKernel:
    """Registry kernel for the single frozen ``prunevid.video`` profile."""

    name = _ALGORITHM
    version = f"videollm-comp-{VIDEO_REFERENCE_REVISION[:8]}-prunevid-v1"

    def validate_config(self, plugin_config: Mapping[str, Any]) -> None:
        if not isinstance(plugin_config, Mapping):
            raise ConfigError("video plugin config must be a mapping")
        contract = plugin_config.get("modality_contract", {})
        if contract:
            if not isinstance(contract, Mapping):
                raise ConfigError("modality_contract must be a mapping")
            if contract.get("algorithm") != self.name:
                raise ConfigError("video plugin algorithm must be 'prunevid'")
            if contract.get("variant") != _VARIANT:
                raise ConfigError("video plugin variant must be 'prunevid.video'")
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
        return prunevid_video(payload, config=config, rules=rules)


# Short public alias matching the other kernel modules.
PruneVidKernel = PruneVidVideoKernel


def prunevid_video(
    payload: VideoTokenPayload,
    *,
    config: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> VideoCompressionResult:
    """Run canonical temporal/static-dynamic PruneVid reduction.

    Selection and cluster assignment are non-differentiable, matching the
    reference.  Every emitted token is a mean over its explicit source group,
    so gradients still reach the provider/backbone features during SFT.
    """

    torch = _torch()
    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    if not isinstance(payload.features, torch.Tensor):
        raise TypeError("PruneVid payload features must be a torch.Tensor")
    if payload.features.ndim != 2 or not payload.features.is_floating_point():
        raise ValueError(
            "PruneVid payload features must have floating shape [tokens, hidden]"
        )
    if payload.token_mask is not None:
        mask = torch.as_tensor(payload.token_mask, device=payload.features.device)
        if mask.ndim != 1 or int(mask.numel()) != payload.token_count:
            raise ValueError("PruneVid token_mask must match the projected layout")
        if not bool(mask.bool().all().item()):
            raise BackendIntegrationError(
                "PruneVid requires provider-trimmed videos without padded tokens"
            )

    resolved = resolve_canonical_video_config(_ALGORITHM, config or {})
    records_by_video: list[list[_OutputRecord]] = []
    windows_by_video: list[tuple[PruneVidWindowLayout, ...]] = []
    all_features: list[Any] = []
    groups_by_video: list[tuple[tuple[int, ...], ...]] = []
    representatives_by_video: list[tuple[int, ...]] = []
    anchor_legalizations_by_video: list[int] = []

    for video_index in range(payload.video_count):
        span = payload.video_span(video_index)
        video_features = payload.features[span]
        coordinates = payload.coordinates[span]
        frame_count, height, width = _rectangular_grid(coordinates)
        records, temporal_windows = _compress_one_video(
            video_features,
            frame_count=frame_count,
            height=height,
            width=width,
            config=resolved,
            torch=torch,
            rules=rules,
        )
        if not records:
            raise InvariantError("PruneVid produced an empty video layout")
        records.sort(key=lambda record: record.source_group[0])
        groups = tuple(record.source_group for record in records)
        representatives, legalization_count = _causal_representatives(records)
        result_positions = {id(record): index for index, record in enumerate(records)}
        windows = _result_window_layouts(
            temporal_windows,
            result_positions=result_positions,
        )
        _validate_window_partition(windows, result_count=len(records))

        gather_groups = [
            torch.tensor(
                record.source_group,
                dtype=torch.long,
                device=video_features.device,
            )
            for record in records
        ]
        compressed = torch.stack(
            [
                video_features.index_select(0, indices).mean(dim=0)
                for indices in gather_groups
            ],
            dim=0,
        )
        all_features.append(compressed)
        if rules and "aggregator" in rules:
            all_features[-1] = _aggregate_values(video_features, groups, rules["aggregator"], torch=torch)
        records_by_video.append(records)
        windows_by_video.append(windows)
        groups_by_video.append(groups)
        representatives_by_video.append(representatives)
        anchor_legalizations_by_video.append(legalization_count)

    compressed_features = torch.cat(all_features, dim=0)
    result = VideoCompressionResult.from_source_groups(
        payload,
        compressed_features,
        source_groups=groups_by_video,
        representative_indices=representatives_by_video,
        exact_budgets=tuple(len(records) for records in records_by_video),
        metadata={
            "algorithm": _VARIANT,
            "reference_revision": VIDEO_REFERENCE_REVISION,
            "canonical_config": _plain_config(resolved),
            "temporal_tie_policy": "custom" if rules and ('frame_grouper' in rules) else "stable_source_index",
            "output_order": "source_group_first_index",
            "representative_policy": "reference_anchor_with_causal_source_fallback",
            "anchor_legalizations_by_video": tuple(anchor_legalizations_by_video),
            "prunevid_windows": tuple(
                tuple(window.record() for window in windows)
                for windows in windows_by_video
            ),
            "projector_stage": "temporal_spatial_reduce",
            "vtp_required": bool(resolved["enable_vtp"]),
        },
    )
    return result


def prunevid_windows(
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
) -> tuple[PruneVidWindowLayout, ...]:
    """Recover and validate one video's VTP window partition."""

    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    if isinstance(video_index, bool) or not isinstance(video_index, int):
        raise TypeError("video_index must be an integer")
    if not 0 <= video_index < result.video_count:
        raise IndexError("video_index is outside the PruneVid result")
    raw_by_video = result.metadata.get("prunevid_windows")
    if not isinstance(raw_by_video, Sequence) or isinstance(raw_by_video, (str, bytes)):
        raise InvariantError("PruneVid result is missing window metadata")
    if len(raw_by_video) != result.video_count:
        raise InvariantError("PruneVid window metadata does not match video count")
    windows = tuple(
        PruneVidWindowLayout.from_record(record) for record in raw_by_video[video_index]
    )
    span = result.video_span(video_index)
    _validate_window_partition(windows, result_count=span.stop - span.start)
    return windows


def merge_prunevid_source_groups(
    features: Any,
    result: VideoCompressionResult,
    *,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> Any:
    """Apply an existing PruneVid group layout to a synchronized feature branch.

    Qwen3-VL uses this for every DeepStack branch so the main and injected
    streams retain identical cardinality, group membership, and source order.
    """

    torch = _torch()
    if not isinstance(features, torch.Tensor):
        raise TypeError("synchronized PruneVid features must be a torch.Tensor")
    if features.ndim != 2 or not features.is_floating_point():
        raise ValueError(
            "synchronized PruneVid features must be floating [tokens, hidden]"
        )
    if not isinstance(result, VideoCompressionResult):
        raise TypeError("result must be a VideoCompressionResult")
    source_span = result.source_video_span(video_index)
    source_count = source_span.stop - source_span.start
    if int(features.shape[0]) != source_count:
        raise InvariantError(
            "synchronized PruneVid branch does not match the source video layout"
        )
    merged = []
    if rules and "aggregator" in rules:
        return _aggregate_values(features, result.source_groups[video_index], rules["aggregator"], torch=torch)
    for group in result.source_groups[video_index]:
        indices = torch.tensor(group, dtype=torch.long, device=features.device)
        merged.append(features.index_select(0, indices).mean(dim=0))
    return torch.stack(merged, dim=0)


def _aggregate_values(features: Any, groups: Any, rule: Any, *, torch: Any) -> Any:
    return torch.stack(tuple(rule_contract.value(rule,
        {"values": features, "group": group}, reference=features,
        shape=(int(features.shape[-1]),), name="aggregator") for group in groups), dim=0)


def prunevid_vtp_visual_scores(
    attention_weights: Any,
    *,
    visual_sequence_indices: Iterable[int],
    query_sequence_indices: Iterable[int],
) -> Any:
    """Reduce native layer-10 attention to the reference VTP visual score.

    The score is the maximum over attention heads and post-visual text query
    tokens.  Providers must supply the explicit query positions; inferring the
    visual span from ignored training labels is intentionally not accepted.
    """

    torch = _torch()
    if not isinstance(attention_weights, torch.Tensor):
        raise TypeError("attention_weights must be a torch.Tensor")
    weights = attention_weights
    if weights.ndim == 4:
        if int(weights.shape[0]) != 1:
            raise BackendIntegrationError("PruneVid VTP supports batch size one")
        weights = weights[0]
    if weights.ndim != 3 or int(weights.shape[-2]) != int(weights.shape[-1]):
        raise ValueError("VTP attention must have shape [heads, sequence, sequence]")
    visual = _normalise_indices(
        visual_sequence_indices,
        name="visual_sequence_indices",
    )
    queries = _normalise_indices(
        query_sequence_indices,
        name="query_sequence_indices",
    )
    sequence_length = int(weights.shape[-1])
    if not visual:
        raise BackendIntegrationError("PruneVid VTP requires visual tokens")
    if not queries:
        raise BackendIntegrationError(
            "PruneVid VTP requires at least one post-visual text query token"
        )
    if visual[-1] >= sequence_length or queries[-1] >= sequence_length:
        raise ValueError("VTP sequence indices exceed the attention layout")
    if any(index <= visual[-1] for index in queries):
        raise BackendIntegrationError(
            "PruneVid VTP query positions must follow the complete visual span"
        )
    visual_index = torch.tensor(visual, dtype=torch.long, device=weights.device)
    query_index = torch.tensor(queries, dtype=torch.long, device=weights.device)
    selected = weights.index_select(1, query_index).index_select(2, visual_index)
    return selected.amax(dim=0).amax(dim=0)


def prunevid_vtp_scores_from_qk(
    instruction_queries: Any,
    attention_keys: Any,
    *,
    visual_sequence_indices: Iterable[int],
    attention_key_mask: Any | None = None,
) -> Any:
    """Compute exact full-key VTP probabilities from provider-native Q/K.

    ``instruction_queries`` is ``[queries, heads, head_dim]`` and
    ``attention_keys`` is ``[sequence, heads, head_dim]`` after native RoPE
    and GQA expansion.  Softmax is taken over the complete visible key axis
    before visual probabilities are selected, matching the pinned decoder
    attention equation.  VTP then takes the maximum over query tokens and
    heads rather than PyramidDrop's mean reduction.
    """

    torch = _torch()
    if not isinstance(instruction_queries, torch.Tensor) or not isinstance(
        attention_keys, torch.Tensor
    ):
        raise TypeError("PruneVid VTP Q/K inputs must be torch tensors")
    if instruction_queries.ndim != 3 or attention_keys.ndim != 3:
        raise ValueError("PruneVid VTP Q/K inputs must have rank three")
    if tuple(instruction_queries.shape[1:]) != tuple(attention_keys.shape[1:]):
        raise ValueError("PruneVid VTP query and key head layouts differ")
    if int(instruction_queries.shape[0]) < 1:
        raise BackendIntegrationError("PruneVid VTP requires text query tokens")
    visual = _normalise_indices(
        visual_sequence_indices,
        name="visual_sequence_indices",
    )
    if not visual or visual[-1] >= int(attention_keys.shape[0]):
        raise ValueError("PruneVid VTP visual indices exceed the key layout")
    scale = float(instruction_queries.shape[-1]) ** -0.5
    logits = (
        torch.einsum(
            "qhd,thd->tqh",
            instruction_queries,
            attention_keys,
        )
        * scale
    )
    if attention_key_mask is not None:
        visibility = torch.as_tensor(
            attention_key_mask,
            device=logits.device,
        ).bool()
        expected = (
            int(instruction_queries.shape[0]),
            int(attention_keys.shape[0]),
        )
        if tuple(int(value) for value in visibility.shape) != expected:
            raise ValueError(
                "PruneVid VTP attention_key_mask must be [queries, sequence]"
            )
        if bool((~visibility.any(dim=-1)).any().item()):
            raise ValueError("Every PruneVid VTP query must see at least one key")
        logits = logits.masked_fill(
            ~visibility.transpose(0, 1).unsqueeze(-1),
            float("-inf"),
        )
    probabilities = torch.softmax(logits, dim=0, dtype=torch.float32).to(logits.dtype)
    index = torch.tensor(visual, dtype=torch.long, device=logits.device)
    return probabilities.index_select(0, index).amax(dim=1).amax(dim=1)


def prunevid_vtp_select(
    visual_scores: Any,
    projector_result: VideoCompressionResult,
    *,
    alpha: float = 0.4,
    video_index: int = 0,
    rules: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    """Apply canonical per-window static and per-frame dynamic top-k VTP."""

    torch = _torch()
    _require_alpha(alpha)
    if not isinstance(visual_scores, torch.Tensor):
        raise TypeError("visual_scores must be a torch.Tensor")
    if visual_scores.ndim != 1 or not visual_scores.is_floating_point():
        raise ValueError("visual_scores must be a floating one-dimensional tensor")
    span = projector_result.video_span(video_index)
    visual_count = span.stop - span.start
    if int(visual_scores.shape[0]) != visual_count:
        raise ValueError("visual_scores do not match the PruneVid projector layout")
    if not bool(torch.isfinite(visual_scores).all().item()):
        raise ValueError("visual_scores must be finite")

    selected: list[int] = []
    for window in prunevid_windows(projector_result, video_index=video_index):
        selected.extend(
            _stable_top_fraction(
                visual_scores,
                window.static_result_indices,
                alpha=alpha,
                torch=torch,
                rules=rules,
            )
        )
        for frame_indices in window.dynamic_result_indices_by_frame:
            selected.extend(
                _stable_top_fraction(
                    visual_scores,
                    frame_indices,
                    alpha=alpha,
                    torch=torch,
                    rules=rules,
                )
            )
    kept = tuple(sorted(selected))
    if not kept:
        raise BackendIntegrationError(
            "Canonical PruneVid VTP would remove every projected video token"
        )
    if len(set(kept)) != len(kept):
        raise InvariantError("PruneVid VTP selected a visual token more than once")
    return kept


def plan_prunevid_vtp(
    *,
    sequence_length: int,
    visual_sequence_indices: Iterable[int],
    visual_scores: Any,
    projector_result: VideoCompressionResult,
    selected_layer: int = 10,
    alpha: float = 0.4,
    application_position: Mapping[str, Any] | None = None,
    rules: Mapping[str, Any] | None = None,
) -> PruneVidVTPResult:
    """Build the exact full-sequence and video-only VTP layout deltas."""

    if isinstance(sequence_length, bool) or not isinstance(sequence_length, int):
        raise TypeError("sequence_length must be an integer")
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    if isinstance(selected_layer, bool) or not isinstance(selected_layer, int):
        raise TypeError("selected_layer must be an integer")
    if selected_layer < 0:
        raise ValueError("selected_layer must be non-negative")
    original_position = {"boundary": "inter_layer", "index": selected_layer + 1}
    actual_position = dict(application_position or original_position)
    layout_stage = "after_selected_layer" if actual_position == original_position else "vtp_prefill_prune"
    visual_positions = _normalise_indices(
        visual_sequence_indices,
        name="visual_sequence_indices",
    )
    if visual_positions and visual_positions[-1] >= sequence_length:
        raise ValueError("visual_sequence_indices exceed sequence_length")
    if projector_result.video_count != 1:
        raise BackendIntegrationError(
            "PruneVid VTP currently requires exactly one video per sample"
        )
    if len(visual_positions) != projector_result.result_count:
        raise InvariantError(
            "VTP visual span does not match the PruneVid projector result"
        )
    kept_visual = prunevid_vtp_select(
        visual_scores,
        projector_result,
        alpha=alpha,
        rules=rules,
    )
    kept_visual_set = set(kept_visual)
    dropped_visual = tuple(
        index
        for index in range(projector_result.result_count)
        if index not in kept_visual_set
    )
    retained_visual_positions = {visual_positions[index] for index in kept_visual}
    visual_position_set = set(visual_positions)
    kept_sequence = tuple(
        index
        for index in range(sequence_length)
        if index not in visual_position_set or index in retained_visual_positions
    )

    visual_mapping = {
        source_index: (
            kept_visual.index(source_index) if source_index in kept_visual_set else None
        )
        for source_index in range(projector_result.result_count)
    }
    sequence_targets = {value: index for index, value in enumerate(kept_sequence)}
    sequence_mapping = {
        source_index: sequence_targets.get(source_index)
        for source_index in range(sequence_length)
    }
    media_id = projector_result.media_ids[0]
    visual_delta = LayoutDelta.from_mapping(
        source_count=projector_result.result_count,
        result_count=len(kept_visual),
        old_to_new=visual_mapping,
        media_unit=LayoutTokenUnit.VIDEO_PATCH,
        coordinates=projector_result.coordinates,
        source_layout={
            "layout_id": f"video:{media_id}:prunevid_projector",
            "stream_id": f"video:{media_id}:prunevid_projector",
            "media_id": media_id,
            "stage": "vtp_prefill_prune",
        },
        result_layout={
            "layout_id": f"video:{media_id}:prunevid_vtp",
            "stream_id": f"video:{media_id}:prunevid_vtp",
            "media_id": media_id,
            "stage": "vtp_prefill_prune",
            "representative_indices": kept_visual,
            "coordinates": tuple(
                projector_result.coordinates[index].as_tuple() for index in kept_visual
            ),
        },
    )
    sequence_delta = LayoutDelta.from_mapping(
        source_count=sequence_length,
        result_count=len(kept_sequence),
        old_to_new=sequence_mapping,
        media_unit="llm_token",
        source_layout={
            "layout_id": "decoder:prunevid_vtp:source",
            "stream_id": "decoder:prunevid_vtp:source",
            "stage": layout_stage,
            "application_position": actual_position,
        },
        result_layout={
            "layout_id": "decoder:prunevid_vtp:result",
            "stream_id": "decoder:prunevid_vtp:result",
            "stage": layout_stage,
            "application_position": actual_position,
            "kept_sequence_indices": kept_sequence,
            "visual_sequence_indices": visual_positions,
        },
    )
    return PruneVidVTPResult(
        selected_layer=selected_layer,
        alpha=float(alpha),
        kept_visual_indices=kept_visual,
        dropped_visual_indices=dropped_visual,
        kept_sequence_indices=kept_sequence,
        visual_layout_delta=visual_delta,
        sequence_layout_delta=sequence_delta,
        metadata={
            "algorithm": _VARIANT,
            "stage": "vtp_prefill_prune",
            "attention_reduction": "custom" if rules and ('vtp_scorer' in rules) else "max_heads_then_post_visual_queries",
            "selection": "custom" if rules and ('vtp_selector' in rules) else "floor_alpha_static_and_each_dynamic_frame",
            "cache_policy": "compact_layers_zero_through_selected_inclusive" if actual_position == original_position else "compact_executed_layer_prefix",
            "application_position": actual_position,
            "score_layer": selected_layer,
            "position_policy": "gather_native_positions",
            "source_visual_count": projector_result.result_count,
            "result_visual_count": len(kept_visual),
        },
    )


@dataclass
class _OutputRecord:
    source_group: tuple[int, ...]
    representative: int
    kind: str
    window_index: int
    frame_offset: int | None


@dataclass
class _TemporalWindow:
    window_index: int
    source_frame_start: int
    frame_count: int
    static_records: list[_OutputRecord]
    dynamic_records_by_frame: list[list[_OutputRecord]]


def _compress_one_video(
    features: Any,
    *,
    frame_count: int,
    height: int,
    width: int,
    config: Mapping[str, Any],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> tuple[list[_OutputRecord], list[_TemporalWindow]]:
    token_count, hidden = (int(value) for value in features.shape)
    tokens_per_frame = height * width
    if token_count != frame_count * tokens_per_frame:
        raise InvariantError("PruneVid grid does not match the projected token count")
    frames = features.reshape(frame_count, tokens_per_frame, hidden)
    frame_means = frames.mean(dim=1)
    temporal_clusters = max(1, int(frame_count * float(config["temporal_ratio"])))
    temporal_labels = _dpc_knn(
        frame_means,
        cluster_count=temporal_clusters,
        k=min(int(config["k"]), max(frame_count - 1, 1)),
        torch=torch,
    )
    refined = _refine_cluster_segments(tuple(int(value) for value in temporal_labels))
    lengths = _run_lengths(refined)
    if rules and "frame_grouper" in rules:
        grouped = rule_contract.groups(rules["frame_grouper"],
            {"values": frames, "frame_means": frame_means, "default_lengths": lengths},
            count=None, candidates=range(frame_count), name="frame_grouper")
        if tuple(i for group in grouped for i in group) != tuple(range(frame_count)):
            raise BackendIntegrationError("PruneVid frame_grouper must return contiguous ordered windows")
        lengths = tuple(len(group) for group in grouped)

    all_records: list[_OutputRecord] = []
    windows: list[_TemporalWindow] = []
    start_frame = 0
    for window_index, window_size in enumerate(lengths):
        window_frames = frames[start_frame : start_frame + window_size]
        static_mask = _window_static_mask(
            window_frames,
            tau=float(config["tau"]),
            torch=torch,
        )
        static_positions = tuple(
            int(value)
            for value in static_mask.nonzero(as_tuple=False).flatten().tolist()
        )
        if rules and "static_selector" in rules:
            static_positions = rule_contract.select(rules["static_selector"],
                {"values": window_frames, "default_static_mask": static_mask,
                 "window_index": window_index}, count=None,
                candidates=range(tokens_per_frame), name="static_selector")
            static_mask = torch.zeros(tokens_per_frame, dtype=torch.bool, device=features.device)
            static_mask[list(static_positions)] = True
        dynamic_positions = tuple(
            int(value)
            for value in (~static_mask).nonzero(as_tuple=False).flatten().tolist()
        )
        static_records = _static_records(
            frames=frames,
            spatial_positions=static_positions,
            start_frame=start_frame,
            window_size=window_size,
            tokens_per_frame=tokens_per_frame,
            window_index=window_index,
            config=config,
            torch=torch,
            rules=rules,
        )
        dynamic_records = _dynamic_records(
            frames=frames,
            spatial_positions=dynamic_positions,
            start_frame=start_frame,
            window_size=window_size,
            tokens_per_frame=tokens_per_frame,
            window_index=window_index,
            config=config,
            torch=torch,
            rules=rules,
        )
        all_records.extend(static_records)
        all_records.extend(record for row in dynamic_records for record in row)
        windows.append(
            _TemporalWindow(
                window_index=window_index,
                source_frame_start=start_frame,
                frame_count=window_size,
                static_records=static_records,
                dynamic_records_by_frame=dynamic_records,
            )
        )
        start_frame += window_size
    if start_frame != frame_count:
        raise InvariantError("PruneVid temporal windows do not cover every frame")
    return all_records, windows


def _static_records(
    *,
    frames: Any,
    spatial_positions: tuple[int, ...],
    start_frame: int,
    window_size: int,
    tokens_per_frame: int,
    window_index: int,
    config: Mapping[str, Any],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> list[_OutputRecord]:
    if not spatial_positions:
        return []
    index = torch.tensor(spatial_positions, dtype=torch.long, device=frames.device)
    candidates = (
        frames[start_frame : start_frame + window_size]
        .index_select(1, index)
        .mean(dim=0)
    )
    candidate_groups = tuple(
        tuple(
            frame * tokens_per_frame + spatial
            for frame in range(start_frame, start_frame + window_size)
        )
        for spatial in spatial_positions
    )
    center_frame = start_frame + window_size // 2
    candidate_representatives = tuple(
        center_frame * tokens_per_frame + spatial for spatial in spatial_positions
    )
    cluster_groups = _spatial_cluster_groups(
        candidates,
        config=config,
        torch=torch,
        rules=rules, kind="static",
    )
    records = []
    for cluster in cluster_groups:
        group = tuple(
            sorted(
                source
                for candidate_index in cluster
                for source in candidate_groups[candidate_index]
            )
        )
        representative_candidate = _representative_candidate(
            candidates,
            cluster,
            torch=torch,
        )
        records.append(
            _OutputRecord(
                source_group=group,
                representative=candidate_representatives[representative_candidate],
                kind="static",
                window_index=window_index,
                frame_offset=None,
            )
        )
    return records


def _dynamic_records(
    *,
    frames: Any,
    spatial_positions: tuple[int, ...],
    start_frame: int,
    window_size: int,
    tokens_per_frame: int,
    window_index: int,
    config: Mapping[str, Any],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> list[list[_OutputRecord]]:
    by_frame: list[list[_OutputRecord]] = []
    if not spatial_positions:
        return [[] for _ in range(window_size)]
    index = torch.tensor(spatial_positions, dtype=torch.long, device=frames.device)
    for frame_offset in range(window_size):
        frame_index = start_frame + frame_offset
        candidates = frames[frame_index].index_select(0, index)
        cluster_groups = _spatial_cluster_groups(
            candidates,
            config=config,
            torch=torch,
            rules=rules, kind="dynamic",
        )
        records = []
        for cluster in cluster_groups:
            group = tuple(
                sorted(
                    frame_index * tokens_per_frame + spatial_positions[value]
                    for value in cluster
                )
            )
            representative_candidate = _representative_candidate(
                candidates,
                cluster,
                torch=torch,
            )
            records.append(
                _OutputRecord(
                    source_group=group,
                    representative=(
                        frame_index * tokens_per_frame
                        + spatial_positions[representative_candidate]
                    ),
                    kind="dynamic",
                    window_index=window_index,
                    frame_offset=frame_offset,
                )
            )
        by_frame.append(records)
    return by_frame


def _spatial_cluster_groups(
    candidates: Any,
    *,
    config: Mapping[str, Any],
    torch: Any,
    rules: Mapping[str, Any] | None = None,
    kind: str = "spatial",
) -> tuple[tuple[int, ...], ...]:
    count = int(candidates.shape[0])
    if count <= int(config["min_tokens_for_cluster"]):
        return tuple((index,) for index in range(count))
    cluster_count = max(1, int(count * float(config["cluster_ratio"])))
    if rules and "token_grouper" in rules:
        return rule_contract.groups(rules["token_grouper"],
            {"values": candidates, "kind": kind, "config": config,
             "group_count": cluster_count},
            count=cluster_count, candidates=range(count), name="token_grouper")
    labels = _dpc_knn(
        candidates,
        cluster_count=cluster_count,
        k=int(config["k"]),
        torch=torch,
    )
    return tuple(
        tuple(index for index, label in enumerate(labels) if int(label) == cluster)
        for cluster in range(cluster_count)
    )


def _dpc_knn(
    features: Any,
    *,
    cluster_count: int,
    k: int,
    torch: Any,
) -> Any:
    """Deterministic single-sequence DPC-KNN used by both PruneVid stages."""

    count, hidden = (int(value) for value in features.shape)
    if count < 1:
        raise ValueError("DPC-KNN requires at least one candidate")
    if not 1 <= cluster_count <= count:
        raise ValueError("DPC-KNN cluster_count must be within candidate count")
    if count == 1:
        return torch.zeros(1, dtype=torch.long, device=features.device)
    neighbours = min(max(1, int(k)), count)
    with torch.no_grad():
        distances = torch.cdist(features.float(), features.float()) / (hidden**0.5)
        nearest = torch.topk(
            distances,
            k=neighbours,
            dim=-1,
            largest=False,
            sorted=True,
        ).values
        density = (-(nearest.square().mean(dim=-1))).exp()
        higher = density.unsqueeze(0) > density.unsqueeze(1)
        maximum = distances.max()
        parent_distance = torch.where(higher, distances, maximum).min(dim=-1).values
        score = parent_distance * density
        centers = torch.argsort(
            score,
            dim=-1,
            descending=True,
            stable=True,
        )[:cluster_count]
        center_distances = distances.index_select(0, centers)
        labels = center_distances.argmin(dim=0)
        labels[centers] = torch.arange(
            cluster_count,
            dtype=torch.long,
            device=features.device,
        )
    return labels


def _representative_candidate(
    candidates: Any,
    group: tuple[int, ...],
    *,
    torch: Any,
) -> int:
    if len(group) == 1:
        return group[0]
    index = torch.tensor(group, dtype=torch.long, device=candidates.device)
    members = candidates.index_select(0, index).float()
    center = members.mean(dim=0, keepdim=True)
    scores = torch.nn.functional.cosine_similarity(members, center, dim=-1)
    # ``argmax`` returns the first member, giving source-index tie stability.
    return group[int(scores.argmax().item())]


def _window_static_mask(window_frames: Any, *, tau: float, torch: Any) -> Any:
    window_size = int(window_frames.shape[0])
    token_count = int(window_frames.shape[1])
    if window_size == 1:
        similarity = torch.ones(
            token_count,
            dtype=torch.float32,
            device=window_frames.device,
        )
    else:
        normalised = torch.nn.functional.normalize(window_frames, p=2, dim=-1)
        pairwise = torch.einsum("wlc,tlc->wtl", normalised, normalised)
        similarity = (pairwise.sum(dim=1) - 1).sum(dim=0) / (
            window_size * (window_size - 1)
        )
    return similarity > tau


def _refine_cluster_segments(values: tuple[int, ...]) -> tuple[int, ...]:
    """Match the pinned temporal run refinement without tensor-side mutation."""

    if not values:
        raise ValueError("temporal cluster labels must not be empty")
    segments = _segments(values)
    maximum_by_label: dict[int, int] = {}
    for start, end, label in segments:
        maximum_by_label[label] = max(maximum_by_label.get(label, 0), end - start)
    refined: list[int | None] = [None] * len(values)
    for start, end, label in segments:
        length = end - start
        if maximum_by_label[label] > 1 and length == maximum_by_label[label]:
            refined[start:end] = [label] * length
    if all(value is None for value in refined):
        return (0,) * len(values)

    valid_segments = [
        (start, end, label)
        for start, end, label in _segments(tuple(refined))
        if label is not None
    ]
    for index, value in enumerate(refined):
        if value is not None:
            continue
        left = next(
            (segment for segment in reversed(valid_segments) if segment[1] <= index),
            None,
        )
        right = next(
            (segment for segment in valid_segments if segment[0] > index),
            None,
        )
        left_length = 0 if left is None else left[1] - left[0]
        right_length = 0 if right is None else right[1] - right[0]
        if left is not None and left_length >= right_length:
            refined[index] = left[2]
        elif right is not None:
            refined[index] = right[2]
        else:
            refined[index] = 0
    return tuple(int(value) for value in refined)


def _segments(values: Sequence[Any]) -> list[tuple[int, int, Any]]:
    segments: list[tuple[int, int, Any]] = []
    start = 0
    for index in range(1, len(values)):
        if values[index] != values[index - 1]:
            segments.append((start, index, values[start]))
            start = index
    segments.append((start, len(values), values[start]))
    return segments


def _run_lengths(values: Sequence[Any]) -> tuple[int, ...]:
    return tuple(end - start for start, end, _ in _segments(values))


def _result_window_layouts(
    windows: Sequence[_TemporalWindow],
    *,
    result_positions: Mapping[int, int],
) -> tuple[PruneVidWindowLayout, ...]:
    return tuple(
        PruneVidWindowLayout(
            window_index=window.window_index,
            source_frame_start=window.source_frame_start,
            frame_count=window.frame_count,
            static_result_indices=tuple(
                sorted(result_positions[id(record)] for record in window.static_records)
            ),
            dynamic_result_indices_by_frame=tuple(
                tuple(sorted(result_positions[id(record)] for record in records))
                for records in window.dynamic_records_by_frame
            ),
        )
        for window in windows
    )


def _causal_representatives(
    records: Sequence[_OutputRecord],
) -> tuple[tuple[int, ...], int]:
    representatives = []
    legalized = 0
    for index, record in enumerate(records):
        upper = records[index + 1].source_group[0] if index + 1 < len(records) else None
        representative = record.representative
        if upper is not None and representative >= upper:
            representative = record.source_group[0]
            legalized += 1
        representatives.append(representative)
    result = tuple(representatives)
    if result != tuple(sorted(result)) or len(set(result)) != len(result):
        raise InvariantError("PruneVid representatives do not preserve causal order")
    return result, legalized


def _validate_window_partition(
    windows: Sequence[PruneVidWindowLayout],
    *,
    result_count: int,
) -> None:
    expected_window = 0
    expected_frame = 0
    represented: list[int] = []
    for window in windows:
        if window.window_index != expected_window:
            raise InvariantError("PruneVid window indices are not contiguous")
        if window.source_frame_start != expected_frame:
            raise InvariantError("PruneVid windows do not preserve frame order")
        represented.extend(window.static_result_indices)
        represented.extend(
            value for row in window.dynamic_result_indices_by_frame for value in row
        )
        expected_window += 1
        expected_frame += window.frame_count
    if tuple(sorted(represented)) != tuple(range(result_count)):
        raise InvariantError(
            "PruneVid static/dynamic metadata does not partition projector outputs"
        )


def _rectangular_grid(
    coordinates: Sequence[VideoTokenCoordinate],
) -> tuple[int, int, int]:
    if not coordinates:
        raise BackendIntegrationError("PruneVid requires a non-empty video grid")
    frame_count = max(value.t for value in coordinates) + 1
    height = max(value.h for value in coordinates) + 1
    width = max(value.w for value in coordinates) + 1
    expected = tuple(
        (temporal, row, column)
        for temporal in range(frame_count)
        for row in range(height)
        for column in range(width)
    )
    actual = tuple((value.t, value.h, value.w) for value in coordinates)
    if actual != expected:
        raise BackendIntegrationError(
            "PruneVid requires one source-ordered rectangular frame grid"
        )
    return frame_count, height, width


def _stable_top_fraction(
    scores: Any,
    candidates: tuple[int, ...],
    *,
    alpha: float,
    torch: Any,
    rules: Mapping[str, Any] | None = None,
) -> tuple[int, ...]:
    keep_count = int(len(candidates) * alpha)
    if rules and "vtp_selector" in rules:
        return rule_contract.select(rules["vtp_selector"],
            {"scores": scores, "candidates": candidates, "keep_count": keep_count},
            count=keep_count, candidates=candidates, name="vtp_selector")
    if keep_count < 1:
        return ()
    index = torch.tensor(candidates, dtype=torch.long, device=scores.device)
    candidate_scores = scores.index_select(0, index)
    order = torch.argsort(
        candidate_scores,
        descending=True,
        stable=True,
    )[:keep_count]
    return tuple(int(value) for value in index.index_select(0, order).tolist())


def _normalise_indices(values: Iterable[int], *, name: str) -> tuple[int, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be an iterable of integers")
    result = tuple(values)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in result):
        raise TypeError(f"{name} entries must be integers")
    if any(value < 0 for value in result):
        raise ValueError(f"{name} entries must be non-negative")
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise ValueError(f"{name} must contain unique indices in source order")
    return result


def _require_alpha(value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("alpha must be a finite number")
    if not isfinite(float(value)) or not 0.0 < float(value) <= 1.0:
        raise ValueError("alpha must be finite and within (0, 1]")


def _plain_config(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: (
            _plain_config(item)
            if isinstance(item, Mapping)
            else list(item) if isinstance(item, tuple) else item
        )
        for key, item in value.items()
    }


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dependency-light control plane
        raise BackendIntegrationError(
            "PruneVid runtime requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = [
    "PruneVidKernel",
    "PruneVidVideoKernel",
    "PruneVidVTPResult",
    "PruneVidWindowLayout",
    "merge_prunevid_source_groups",
    "plan_prunevid_vtp",
    "prunevid_video",
    "prunevid_vtp_select",
    "prunevid_vtp_scores_from_qk",
    "prunevid_vtp_visual_scores",
    "prunevid_windows",
]
