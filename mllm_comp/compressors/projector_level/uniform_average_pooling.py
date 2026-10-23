"""Uniform local average pooling from arXiv:2511.20973.

The paper applies a one-dimensional average-pooling kernel of size ``K`` and
stride ``K`` to audio-encoder tokens.  This module keeps that operation
separate from the Qwen2.5-Omni hook and placeholder legalization machinery.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from mllm_comp.compressors.audio_rule_utils import (
    AudioTokenStrategyBase,
    _torch,
    aggregate_groups,
)
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import InvariantError
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


@dataclass(frozen=True)
class UniformAveragePoolingSpec:
    """Normalized settings for the paper's UniAvg method."""

    compression_factor: int = 2
    minimum_keep_tokens: int = 1
    score_source: str = "uniform_local_average"
    merge_value: str = "mean"
    target_site: str = "audio_encoder_output_pre_projector"

    @property
    def keep_ratio(self) -> float:
        return 1.0 / self.compression_factor

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "UniformAveragePoolingSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        legacy = _mapping(config.get("config"), "config")
        compression_factor = _resolve_compression_factor(
            schedule=schedule,
            legacy=legacy,
            default=cls.compression_factor,
        )
        spec = cls(
            compression_factor=compression_factor,
            minimum_keep_tokens=_strict_int(
                _first_present(
                    schedule.get("minimum_keep_tokens"),
                    legacy.get("minimum_keep_tokens"),
                    cls.minimum_keep_tokens,
                ),
                "minimum_keep_tokens",
            ),
            score_source=str(
                _first_present(
                    criterion.get("score_source"),
                    legacy.get("score_source"),
                    cls.score_source,
                )
            ),
            merge_value=str(
                _first_present(
                    operation.get("merge_value"),
                    legacy.get("merge_value"),
                    cls.merge_value,
                )
            ),
            target_site=str(
                _first_present(
                    operation.get("target_site"),
                    legacy.get("target_site"),
                    cls.target_site,
                )
            ),
        )
        spec.validate()
        if operation.get("type") not in {None, "merge"}:
            raise ValueError(
                "uniform_average_pooling_audio requires operation.type=merge"
            )
        return spec

    def validate(self) -> None:
        if self.compression_factor < 1:
            raise ValueError("compression_factor must be at least 1")
        if self.minimum_keep_tokens != 1:
            raise ValueError(
                "paper-faithful uniform average pooling requires "
                "minimum_keep_tokens=1"
            )
        if self.score_source != "uniform_local_average":
            raise ValueError(
                "uniform average pooling requires "
                "score_source='uniform_local_average'"
            )
        if self.merge_value != "mean":
            raise ValueError("uniform average pooling requires merge_value='mean'")


def uniform_average_keep_count(
    length: int,
    compression_factor: int,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the output length of kernel-size/stride-``K`` average pooling."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if compression_factor < 1:
        raise ValueError("compression_factor must be at least 1")
    if minimum_keep_tokens != 1:
        raise ValueError("uniform average pooling requires minimum_keep_tokens=1")
    if length == 0:
        return 0
    return max(1, length // compression_factor)


def uniform_average_pool(
    hidden_states: Any,
    *,
    keep_count: int,
    compression_factor: int,
    grouper: Any = None, aggregator: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Average non-overlapping chronological windows of ``K`` audio tokens."""

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError(
            "Uniform average pooling expects rank-2 token hidden states."
        )
    length = int(hidden_states.shape[0])
    expected_keep_count = uniform_average_keep_count(length, compression_factor)
    if keep_count != expected_keep_count:
        raise InvariantError(
            "Uniform average pooling keep_count must match kernel-size/stride-K "
            f"output length {expected_keep_count}, got {keep_count}."
        )
    if compression_factor == 1:
        return hidden_states, {
            "compression_factor": 1,
            "pool_windows": [(index, index + 1) for index in range(length)],
            "dropped_tail_tokens": 0,
            "merge_value": "mean",
            "token_order": "chronological",
        }

    if length < compression_factor:
        output = hidden_states.mean(dim=0, keepdim=True)
        windows = [(0, length)]
        dropped_tail_tokens = 0
    else:
        used_length = keep_count * compression_factor
        output = hidden_states[:used_length].reshape(
            keep_count,
            compression_factor,
            int(hidden_states.shape[1]),
        ).mean(dim=1)
        windows = [
            (index * compression_factor, (index + 1) * compression_factor)
            for index in range(keep_count)
        ]
        dropped_tail_tokens = length - used_length
    if grouper is not None or aggregator is not None:
        from mllm_comp.compressors.audio_rule_utils import contiguous_groups, aggregate_groups
        groups = [tuple(range(start, end)) for start, end in windows]
        if grouper is not None:
            groups = contiguous_groups(grouper({"values": hidden_states.detach(), "keep_count": keep_count,
                "groups": tuple(groups), "compression_factor": compression_factor}),
                length=length, count=keep_count, allow_tail=True)
        windows = [(group[0], group[-1] + 1) for group in groups]
        dropped_tail_tokens = length - sum(map(len, groups))
        output = aggregate_groups(hidden_states, groups, aggregator=aggregator)
    return output, {
        "compression_factor": compression_factor,
        "pool_windows": windows,
        "dropped_tail_tokens": dropped_tail_tokens,
        "merge_value": "mean" if aggregator is None else "custom",
        "token_order": "chronological",
    }


class UniformAveragePoolingProjectorStrategy(AudioTokenStrategyBase):
    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "pool"
    metadata_fields = (
        "keep_ratio",
        "compression_factor",
        "minimum_keep_tokens",
        "merge_value",
    )

    def output_length(self, length: int) -> int:
        return uniform_average_keep_count(length, self.spec.compression_factor)

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        return self.replay(reference, tuple(details["pool_windows"]))

    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
        return tuple(details["pool_windows"])

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        if len(plan) == int(hidden_states.shape[0]):
            return hidden_states
        if not self.rules:
            # Checkpoint replay must retain the forward reshape/mean graph.
            return uniform_average_pool(hidden_states, keep_count=len(plan),
                compression_factor=self.spec.compression_factor)[0]
        return aggregate_groups(hidden_states, [tuple(range(start, end)) for start, end in plan],
            aggregator=self.rules.get("aggregator"))

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        return _torch().tensor([start for start, _ in plan], device=device, dtype=_torch().long)

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, None, dict[str, Any]]:
        del scores, item_index
        output, details = uniform_average_pool(
            hidden_states,
            keep_count=keep_count,
            compression_factor=self.spec.compression_factor,
            grouper=self.rules.get("grouper"), aggregator=self.rules.get("aggregator"),
        )
        return output, None, details



class UniformAveragePoolingAudio:
    """Projector-level plan provider for UniAvg."""

    name = "uniform_average_pooling_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        UniformAveragePoolingSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = UniformAveragePoolingSpec.from_config(config)
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        for audio_unit_index, span in enumerate(audio_spans):
            original_count = span.end - span.start
            keep_count = uniform_average_keep_count(
                original_count,
                spec.compression_factor,
                minimum_keep_tokens=spec.minimum_keep_tokens,
            )
            original_total += original_count
            planned_total += keep_count
            actions.append(
                TokenAction(
                    op="merge",
                    target_indices=None,
                    merge_groups=None,
                    metadata={
                        "selection_deferred": True,
                        "selector": self.name,
                        "target_site": spec.target_site,
                        "sample_index": span.sample_index,
                        "audio_unit_index": audio_unit_index,
                        "span_start": span.start,
                        "span_end": span.end,
                        "original_count": original_count,
                        "keep_count": keep_count,
                        "keep_ratio": spec.keep_ratio,
                        "compression_factor": spec.compression_factor,
                        "token_order": "chronological",
                    },
                )
            )
        stage_name = str(config.get("_stage_name") or self.name)
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
            legalization_policy=_legalization_policy(),
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                changed_spans=audio_spans,
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "compression_factor": spec.compression_factor,
                    "num_audio_units": len(actions),
                    "original_audio_tokens": original_total,
                    "planned_audio_tokens": planned_total,
                    "requested_keep_ratio": spec.keep_ratio,
                    "selection_deferred_to_runtime": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("UniformAveragePoolingAudio has no trainable state")


def _resolve_compression_factor(
    *,
    schedule: Mapping[str, Any],
    legacy: Mapping[str, Any],
    default: int,
) -> int:
    raw_factor = schedule.get("compression_factor")
    if raw_factor is None:
        raw_factor = legacy.get("compression_factor")
    raw_ratio = schedule.get("keep_ratio")
    if raw_ratio is None:
        raw_ratio = legacy.get("keep_ratio")
    factor = default if raw_factor is None else _strict_int(
        raw_factor, "compression_factor"
    )
    if raw_ratio is not None:
        ratio = float(raw_ratio)
        if not 0 < ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        reciprocal = 1.0 / ratio
        ratio_factor = int(round(reciprocal))
        if ratio_factor < 1 or not math.isclose(
            reciprocal, ratio_factor, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError(
                "paper-faithful local pooling requires keep_ratio=1/K for an "
                "integer compression_factor K"
            )
        if raw_factor is not None and factor != ratio_factor:
            raise ValueError("keep_ratio and compression_factor disagree")
        factor = ratio_factor
    return factor


def _legalization_policy() -> LegalizationPolicy:
    return LegalizationPolicy(
        position_policy="compact",
        mask_policy="rebuild_from_tokens",
        label_policy="rebuild_ignore_index",
        coordinate_policy="drop_removed",
    )


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_present(*values: Any) -> Any:
    return next(value for value in values if value is not None)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer, got {value!r}")
    return value


__all__ = [
    "UniformAveragePoolingProjectorStrategy",
    "UniformAveragePoolingAudio",
    "UniformAveragePoolingSpec",
    "uniform_average_keep_count",
    "uniform_average_pool",
]
