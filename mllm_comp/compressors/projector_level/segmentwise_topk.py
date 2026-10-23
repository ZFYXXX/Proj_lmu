"""Training-free segmentwise audio-token selection.

The paper-facing algorithm is intentionally separated from the Qwen2.5-Omni
runtime adapter. This module owns deterministic budgets and indices; the
backbone adapter owns attention capture, feature mutation and sequence repair.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any, Literal

from mllm_comp.compressors.audio_rule_utils import (
    AudioTokenStrategyBase,
    _torch,
    token_indices,
    token_scores,
)
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)


AllocationPolicy = Literal["balanced_exact", "paper_floor"]
ScoreSource = Literal[
    "encoder_last_layer_attention",
    "uniform",
    "random",
]


@dataclass(frozen=True)
class SegmentwiseTopKSpec:
    """Normalized configuration shared by the plugin and runtime controller."""

    keep_ratio: float = 0.25
    num_segments: int = 10
    allocation: AllocationPolicy = "balanced_exact"
    score_source: ScoreSource = "encoder_last_layer_attention"
    target_site: str = "audio_encoder_output_pre_projector"
    minimum_keep_tokens: int = 1
    random_seed: int = 42

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SegmentwiseTopKSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        legacy = _mapping(config.get("config"), "config")

        target = _first_schedule_target(schedule)
        keep_ratio = _first_present(
            target.get("keep_ratio"),
            schedule.get("keep_ratio"),
            legacy.get("keep_ratio"),
            cls.keep_ratio,
        )
        num_segments = _first_present(
            target.get("num_segments"),
            schedule.get("num_segments"),
            criterion.get("num_segments"),
            legacy.get("num_segments"),
            cls.num_segments,
        )
        allocation = _first_present(
            schedule.get("allocation"),
            legacy.get("allocation"),
            cls.allocation,
        )
        score_source = _first_present(
            criterion.get("score_source"),
            legacy.get("score_source"),
            cls.score_source,
        )
        target_site = _first_present(
            operation.get("target_site"),
            legacy.get("target_site"),
            cls.target_site,
        )
        minimum_keep_tokens = _first_present(
            schedule.get("minimum_keep_tokens"),
            legacy.get("minimum_keep_tokens"),
            cls.minimum_keep_tokens,
        )
        random_seed = _first_present(
            criterion.get("random_seed"),
            legacy.get("random_seed"),
            cls.random_seed,
        )

        spec = cls(
            keep_ratio=float(keep_ratio),
            num_segments=_strict_int(num_segments, "num_segments"),
            allocation=str(allocation),  # type: ignore[arg-type]
            score_source=str(score_source),  # type: ignore[arg-type]
            target_site=str(target_site),
            minimum_keep_tokens=_strict_int(
                minimum_keep_tokens,
                "minimum_keep_tokens",
            ),
            random_seed=_strict_int(random_seed, "random_seed"),
        )
        spec.validate()
        operation_type = operation.get("type")
        if operation_type not in {None, "keep"}:
            raise ValueError(
                "segmentwise_topk_audio requires operation.type=keep"
            )
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError(
                f"keep_ratio must be in (0, 1], got {self.keep_ratio}"
            )
        if self.num_segments < 1:
            raise ValueError(
                f"num_segments must be at least 1, got {self.num_segments}"
            )
        if self.minimum_keep_tokens < 1:
            raise ValueError(
                "minimum_keep_tokens must be at least 1, got "
                f"{self.minimum_keep_tokens}"
            )
        if self.allocation not in {"balanced_exact", "paper_floor"}:
            raise ValueError(
                "allocation must be 'balanced_exact' or 'paper_floor', got "
                f"{self.allocation!r}"
            )
        if self.score_source not in {
            "encoder_last_layer_attention",
            "uniform",
            "random",
        }:
            raise ValueError(
                "score_source must be encoder_last_layer_attention, uniform, or "
                f"random, got {self.score_source!r}"
            )


def target_keep_count(
    length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the deterministic total budget for one audio item."""

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if not 0 < keep_ratio <= 1:
        raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")
    if minimum_keep_tokens < 0:
        raise ValueError("minimum_keep_tokens must be non-negative")
    if length == 0:
        return 0
    return min(length, max(minimum_keep_tokens, int(math.floor(length * keep_ratio))))


def segment_boundaries(length: int, num_segments: int) -> list[tuple[int, int]]:
    """Partition ``range(length)`` into balanced, non-empty temporal segments."""

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if num_segments < 1:
        raise ValueError(f"num_segments must be at least 1, got {num_segments}")
    if length == 0:
        return []
    active_segments = min(length, num_segments)
    base, remainder = divmod(length, active_segments)
    boundaries: list[tuple[int, int]] = []
    start = 0
    for index in range(active_segments):
        width = base + (1 if index < remainder else 0)
        boundaries.append((start, start + width))
        start += width
    return boundaries


def segment_budgets(
    *,
    length: int,
    keep_count: int,
    num_segments: int,
    allocation: AllocationPolicy = "balanced_exact",
) -> tuple[list[tuple[int, int]], list[int]]:
    """Return temporal segments and their keep budgets.

    ``paper_floor`` mirrors the paper's floor(K/S) description and can retain
    fewer than K tokens. ``balanced_exact`` distributes the remainder and is
    the platform default because it preserves an exact global budget.
    """

    if not 0 <= keep_count <= length:
        raise ValueError(
            f"keep_count must satisfy 0 <= keep_count <= {length}, got {keep_count}"
        )
    if allocation not in {"balanced_exact", "paper_floor"}:
        raise ValueError(f"unknown allocation policy {allocation!r}")
    if length == 0:
        return [], []

    active_segments = min(length, num_segments)
    if allocation == "balanced_exact" and keep_count > 0:
        # When K < S, use K temporal partitions so every active partition can
        # contribute one token instead of biasing the earliest K of S bins.
        active_segments = min(active_segments, keep_count)
    boundaries = segment_boundaries(length, active_segments)
    if allocation == "paper_floor":
        per_segment = keep_count // len(boundaries)
        budgets = [min(end - start, per_segment) for start, end in boundaries]
        return boundaries, budgets

    base, remainder = divmod(keep_count, len(boundaries))
    budgets = [
        base + (1 if index < remainder else 0)
        for index in range(len(boundaries))
    ]
    if any(
        budget > end - start
        for budget, (start, end) in zip(budgets, boundaries)
    ):
        raise AssertionError("balanced segment budget exceeded segment capacity")
    return boundaries, budgets


def realized_keep_count(length: int, spec: SegmentwiseTopKSpec) -> int:
    """Return the count that the configured allocation will actually emit."""

    requested = target_keep_count(
        length,
        spec.keep_ratio,
        minimum_keep_tokens=spec.minimum_keep_tokens,
    )
    _, budgets = segment_budgets(
        length=length,
        keep_count=requested,
        num_segments=spec.num_segments,
        allocation=spec.allocation,
    )
    realized = sum(budgets)
    if length > 0 and realized < 1:
        raise ValueError(
            "The configured paper_floor budget keeps zero tokens for an audio "
            "item. Increase keep_ratio, reduce num_segments, or use "
            "allocation='balanced_exact'."
        )
    return realized


def segmentwise_topk_indices(
    scores: Sequence[float],
    *,
    keep_ratio: float,
    num_segments: int,
    allocation: AllocationPolicy = "balanced_exact",
    minimum_keep_tokens: int = 1,
) -> list[int]:
    """Select top-scoring tokens per temporal segment and restore time order."""

    length = len(scores)
    keep_count = target_keep_count(
        length,
        keep_ratio,
        minimum_keep_tokens=minimum_keep_tokens,
    )
    boundaries, budgets = segment_budgets(
        length=length,
        keep_count=keep_count,
        num_segments=num_segments,
        allocation=allocation,
    )
    selected: list[int] = []
    for (start, end), budget in zip(boundaries, budgets):
        ranked = sorted(
            range(start, end),
            key=lambda index: (-float(scores[index]), index),
        )
        selected.extend(ranked[:budget])
    return sorted(selected)


def segmentwise_uniform_indices(
    length: int,
    *,
    keep_ratio: float,
    num_segments: int,
    minimum_keep_tokens: int = 1,
) -> list[int]:
    """Deterministic coverage baseline with the same exact segment budgets."""

    keep_count = target_keep_count(
        length,
        keep_ratio,
        minimum_keep_tokens=minimum_keep_tokens,
    )
    boundaries, budgets = segment_budgets(
        length=length,
        keep_count=keep_count,
        num_segments=num_segments,
        allocation="balanced_exact",
    )
    selected: list[int] = []
    for (start, end), budget in zip(boundaries, budgets):
        width = end - start
        if budget == 0:
            continue
        if budget >= width:
            selected.extend(range(start, end))
            continue
        selected.extend(
            start + min(width - 1, int((index + 0.5) * width / budget))
            for index in range(budget)
        )
    return sorted(set(selected))


def segmentwise_random_indices(
    length: int,
    *,
    keep_ratio: float,
    num_segments: int,
    seed: int,
    minimum_keep_tokens: int = 1,
) -> list[int]:
    """Seeded random baseline with the same exact segment budgets."""

    keep_count = target_keep_count(
        length,
        keep_ratio,
        minimum_keep_tokens=minimum_keep_tokens,
    )
    boundaries, budgets = segment_budgets(
        length=length,
        keep_count=keep_count,
        num_segments=num_segments,
        allocation="balanced_exact",
    )
    generator = random.Random(seed)
    selected: list[int] = []
    for (start, end), budget in zip(boundaries, budgets):
        selected.extend(generator.sample(range(start, end), budget))
    return sorted(selected)


class SegmentwiseTopKProjectorStrategy(AudioTokenStrategyBase):
    token_operation = "keep"
    layer_families = frozenset({"encoder_token", "llm_token"})
    metadata_fields = (
        "keep_ratio",
        "num_segments",
        "allocation",
        "random_seed",
        "minimum_keep_tokens",
    )

    def output_length(self, length: int) -> int:
        return realized_keep_count(length, self.spec)

    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
        return indices.detach()

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        if int(plan.numel()) == int(hidden_states.shape[0]):
            return hidden_states
        return hidden_states.index_select(0, plan.to(hidden_states.device))

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, Any, dict[str, Any]]:
        torch = _torch()
        length = int(hidden_states.shape[0])
        if keep_count == length:
            indices = torch.arange(length, device=hidden_states.device)
            return hidden_states, indices, {}
        if "scorer" in self.rules:
            scores = token_scores(self.rules["scorer"]({"values": hidden_states.detach(), "scores": scores,
                "keep_count": keep_count, "item_index": item_index}), hidden_states, length=length)
        if "selector" in self.rules:
            boundaries, budgets = segment_budgets(length=length, keep_count=keep_count,
                num_segments=self.spec.num_segments, allocation=self.spec.allocation)
            indices = token_indices(self.rules["selector"]({"values": hidden_states.detach(), "scores": scores,
                "keep_count": keep_count, "segment_boundaries": boundaries, "segment_budgets": budgets,
                "item_index": item_index}), length=length, count=keep_count, device=hidden_states.device)
            if any(int(((indices >= start) & (indices < end)).sum()) != budget
                   for (start, end), budget in zip(boundaries, budgets)):
                raise InvariantError("Segmentwise selector must retain each original segment budget.")
            return hidden_states.index_select(0, indices), indices, {}
        if self.score_source == "uniform" and "scorer" not in self.rules:
            values = segmentwise_uniform_indices(
                length,
                keep_ratio=keep_count / length,
                num_segments=self.spec.num_segments,
                minimum_keep_tokens=keep_count,
            )
            indices = torch.tensor(values, dtype=torch.long, device=hidden_states.device)
        elif self.score_source == "random" and "scorer" not in self.rules:
            values = segmentwise_random_indices(
                length,
                keep_ratio=keep_count / length,
                num_segments=self.spec.num_segments,
                seed=self.spec.random_seed + item_index,
                minimum_keep_tokens=keep_count,
            )
            indices = torch.tensor(values, dtype=torch.long, device=hidden_states.device)
        else:
            if scores is None:
                raise InvariantError("Attention Top-K selection requires token scores.")
            boundaries, budgets = segment_budgets(
                length=length,
                keep_count=keep_count,
                num_segments=self.spec.num_segments,
                allocation=self.spec.allocation,
            )
            selected: list[Any] = []
            for (start, end), budget in zip(boundaries, budgets):
                if budget == 0:
                    continue
                try:
                    order = torch.argsort(
                        scores[start:end], descending=True, stable=True
                    )
                except TypeError:  # pragma: no cover
                    order = torch.argsort(scores[start:end], descending=True)
                selected.append(order[:budget] + start)
            if not selected:
                raise InvariantError("Segmentwise Top-K selected no audio tokens.")
            indices = torch.sort(torch.cat(selected))[0].to(
                device=hidden_states.device,
                dtype=torch.long,
            )
        return hidden_states.index_select(0, indices), indices, {}



class SegmentwiseTopKAudio:
    """Projector-level plan provider for dynamic audio-token pruning."""

    name = "segmentwise_topk_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        SegmentwiseTopKSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        # An encoder layer strategy budgets native chunks before pooling. The
        # processor's projected count is only an input span in that case.
        defer_budget = bool(context.extra.get("defer_token_budget", False))
        spec = SegmentwiseTopKSpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        requested_total = 0
        planned_total = 0
        audio_unit_index = 0
        for span in state.modality_token_spans:
            if span.modality != "audio":
                continue
            original_count = span.end - span.start
            requested_keep_count = target_keep_count(
                original_count,
                spec.keep_ratio,
                minimum_keep_tokens=spec.minimum_keep_tokens,
            )
            keep_count = original_count if defer_budget else realized_keep_count(original_count, spec)
            original_total += original_count
            requested_total += requested_keep_count
            planned_total += keep_count
            actions.append(
                TokenAction(
                    op="keep",
                    target_indices=None,
                    metadata={
                        "selection_deferred": True,
                        "selector": self.name,
                        "target_site": spec.target_site,
                        "sample_index": span.sample_index,
                        "audio_unit_index": audio_unit_index,
                        "span_start": span.start,
                        "span_end": span.end,
                        "original_count": original_count,
                        **({} if defer_budget else {"requested_keep_count": requested_keep_count}),
                        "keep_count": keep_count,
                        "keep_ratio": spec.keep_ratio,
                        "num_segments": spec.num_segments,
                        "allocation": spec.allocation,
                        "score_source": spec.score_source,
                        "token_order": "chronological",
                    },
                )
            )
            audio_unit_index += 1

        stage_name = str(config.get("_stage_name") or self.name)
        metadata = StageMetadata(
            stage_name=stage_name,
            changed_spans=[
                span
                for span in state.modality_token_spans
                if span.modality == "audio"
            ],
            stats={
                "target_site": spec.target_site,
                "score_source": spec.score_source,
                "num_audio_units": len(actions),
                "original_audio_tokens": original_total,
                **({} if defer_budget else {"requested_audio_tokens": requested_total}),
                "planned_audio_tokens": planned_total,
                "requested_keep_ratio": spec.keep_ratio,
                "selection_deferred_to_runtime": True,
            },
        )
        execution_phase = ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        )
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.PROJECTOR_LEVEL,
            execution_phase=execution_phase,
            modalities={Modality.AUDIO},
            token_roles={TokenRole.OBSERVATION},
            method=CompressionMethod.TOKEN_COMPRESSION,
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype=spec.target_site,
            token_actions=actions,
            legalization_policy=LegalizationPolicy(
                position_policy="compact",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
                coordinate_policy="drop_removed",
            ),
            this_stage_metadata=metadata,
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("SegmentwiseTopKAudio has no trainable state")


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_schedule_target(schedule: Mapping[str, Any]) -> Mapping[str, Any]:
    targets = schedule.get("targets")
    if targets is None:
        return {}
    if not isinstance(targets, list):
        raise ValueError("schedule.targets must be a list")
    if not targets:
        return {}
    first = targets[0]
    if not isinstance(first, Mapping):
        raise ValueError("schedule.targets[0] must be a mapping")
    return first


def _first_present(*values: Any) -> Any:
    return next(value for value in values if value is not None)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value
