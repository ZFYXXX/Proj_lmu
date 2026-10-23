"""Uniform audio-token sampling from arXiv:2511.20973."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from typing import Any

from mllm_comp.compressors.audio_rule_utils import (
    AudioTokenStrategyBase,
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
class UniformSamplingSpec:
    """Normalized settings for the paper's UniSamp method."""

    compression_factor: int = 2
    minimum_keep_tokens: int = 1
    score_source: str = "uniform_temporal_stride"
    sampling_offset: int = 0
    target_site: str = "audio_encoder_output_pre_projector"

    @property
    def keep_ratio(self) -> float:
        return 1.0 / self.compression_factor

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "UniformSamplingSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        legacy = _mapping(config.get("config"), "config")
        spec = cls(
            compression_factor=_resolve_compression_factor(
                schedule=schedule,
                legacy=legacy,
                default=cls.compression_factor,
            ),
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
            sampling_offset=_strict_int(
                _first_present(
                    criterion.get("sampling_offset"),
                    legacy.get("sampling_offset"),
                    cls.sampling_offset,
                ),
                "sampling_offset",
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
        if operation.get("type") not in {None, "keep"}:
            raise ValueError("uniform_sampling_audio requires operation.type=keep")
        return spec

    def validate(self) -> None:
        if self.compression_factor < 1:
            raise ValueError("compression_factor must be at least 1")
        if self.minimum_keep_tokens != 1:
            raise ValueError(
                "paper-faithful uniform sampling requires minimum_keep_tokens=1"
            )
        if self.score_source != "uniform_temporal_stride":
            raise ValueError(
                "uniform sampling requires score_source='uniform_temporal_stride'"
            )
        if self.sampling_offset != 0:
            raise ValueError(
                "paper-faithful uniform sampling starts with the first token "
                "(sampling_offset=0)"
            )


def uniform_sampling_keep_count(
    length: int,
    compression_factor: int,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the count produced by selecting indices ``0, K, 2K, ...``."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if compression_factor < 1:
        raise ValueError("compression_factor must be at least 1")
    if minimum_keep_tokens != 1:
        raise ValueError("uniform sampling requires minimum_keep_tokens=1")
    if length == 0:
        return 0
    return (length - 1) // compression_factor + 1


def uniform_sample(
    hidden_states: Any,
    *,
    keep_count: int,
    compression_factor: int,
    selector: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Keep every ``K``-th feature frame, beginning with the first frame."""

    import torch

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError("Uniform sampling expects rank-2 token hidden states.")
    length = int(hidden_states.shape[0])
    expected_keep_count = uniform_sampling_keep_count(length, compression_factor)
    if keep_count != expected_keep_count:
        raise InvariantError(
            "Uniform sampling keep_count must match every-Kth output length "
            f"{expected_keep_count}, got {keep_count}."
        )
    if compression_factor == 1:
        indices = torch.arange(length, dtype=torch.long, device=hidden_states.device)
        output = hidden_states
    else:
        indices = torch.arange(
            0,
            length,
            compression_factor,
            dtype=torch.long,
            device=hidden_states.device,
        )
        if selector is not None:
            from mllm_comp.compressors.audio_rule_utils import token_indices
            indices = token_indices(selector({"values": hidden_states.detach(), "keep_count": keep_count,
                "default_indices": indices.detach(), "compression_factor": compression_factor}),
                length=length, count=keep_count, device=hidden_states.device)
        output = hidden_states.index_select(0, indices)
    return output, {
        "compression_factor": compression_factor,
        "selected_indices": indices.detach(),
        "sampling_offset": 0 if selector is None else "custom",
        "token_order": "chronological",
    }


class UniformSamplingProjectorStrategy(AudioTokenStrategyBase):
    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "keep"
    metadata_fields = (
        "keep_ratio",
        "compression_factor",
        "minimum_keep_tokens",
        "sampling_offset",
    )

    def output_length(self, length: int) -> int:
        return uniform_sampling_keep_count(length, self.spec.compression_factor)

    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
        return indices.detach()

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        if self.spec.compression_factor == 1:
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
        del scores, item_index
        output, details = uniform_sample(
            hidden_states,
            keep_count=keep_count,
            compression_factor=self.spec.compression_factor,
            selector=self.rules.get("selector"),
        )
        return output, details["selected_indices"], details



class UniformSamplingAudio:
    """Projector-level plan provider for UniSamp."""

    name = "uniform_sampling_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        UniformSamplingSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = UniformSamplingSpec.from_config(config)
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        for audio_unit_index, span in enumerate(audio_spans):
            original_count = span.end - span.start
            keep_count = uniform_sampling_keep_count(
                original_count,
                spec.compression_factor,
                minimum_keep_tokens=spec.minimum_keep_tokens,
            )
            original_total += original_count
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
                        "keep_count": keep_count,
                        "keep_ratio": spec.keep_ratio,
                        "compression_factor": spec.compression_factor,
                        "sampling_offset": spec.sampling_offset,
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
                    "sampling_offset": spec.sampling_offset,
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
            raise ValueError("UniformSamplingAudio has no trainable state")


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
                "paper-faithful uniform sampling requires keep_ratio=1/K for "
                "an integer compression_factor K"
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
    "UniformSamplingProjectorStrategy",
    "UniformSamplingAudio",
    "UniformSamplingSpec",
    "uniform_sample",
    "uniform_sampling_keep_count",
]
