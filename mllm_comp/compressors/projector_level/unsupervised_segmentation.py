"""Paper-faithful UnSeg audio-token compression from arXiv:2511.20973.

UnSeg computes cosine dissimilarity between adjacent audio-encoder tokens,
uses every strict local maximum as a segment boundary, and mean-pools each
resulting variable-length segment. The number of output tokens is therefore
content-dependent; the 8.23 tokens/s reported by the paper is an observed
average, not a fixed keep ratio.

Qwen2.5-Omni needs the compressed audio-placeholder count before its normal
audio forward begins. The Qwen controller consequently performs a no-grad
preflight pass through the same audio encoder and supplies those projector
inputs to :meth:`UnsupervisedSegmentationAudio.plan`. The kernel recomputes
the natural boundaries during the real forward and rejects a stale preflight
count instead of pruning peaks or inventing supplemental boundaries.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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
class UnsupervisedSegmentationSpec:
    """Normalized settings for the paper's data-dependent UnSeg method."""

    minimum_keep_tokens: int = 1
    score_source: str = "adjacent_token_cosine_dissimilarity"
    boundary_detector: str = "strict_local_peak"
    budget_policy: str = "natural_peaks"
    merge_value: str = "mean"
    target_site: str = "audio_encoder_output_pre_projector"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "UnsupervisedSegmentationSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        legacy = _mapping(config.get("config"), "config")
        if schedule.get("keep_ratio") is not None or legacy.get("keep_ratio") is not None:
            raise ValueError(
                "paper-faithful unsupervised segmentation has no keep_ratio; "
                "its output length is determined by natural local peaks"
            )
        spec = cls(
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
            boundary_detector=str(
                _first_present(
                    criterion.get("boundary_detector"),
                    legacy.get("boundary_detector"),
                    cls.boundary_detector,
                )
            ),
            budget_policy=str(
                _first_present(
                    schedule.get("budget_policy"),
                    legacy.get("budget_policy"),
                    cls.budget_policy,
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
                "unsupervised_segmentation_audio requires operation.type=merge"
            )
        return spec

    def validate(self) -> None:
        if self.minimum_keep_tokens != 1:
            raise ValueError(
                "unsupervised segmentation requires minimum_keep_tokens=1"
            )
        if self.score_source != "adjacent_token_cosine_dissimilarity":
            raise ValueError(
                "unsupervised segmentation requires score_source="
                "'adjacent_token_cosine_dissimilarity'"
            )
        if self.boundary_detector != "strict_local_peak":
            raise ValueError(
                "unsupervised segmentation requires "
                "boundary_detector='strict_local_peak'"
            )
        if self.budget_policy != "natural_peaks":
            raise ValueError(
                "paper-faithful unsupervised segmentation requires "
                "budget_policy='natural_peaks'"
            )
        if self.merge_value != "mean":
            raise ValueError(
                "unsupervised segmentation requires merge_value='mean'"
            )


def unsupervised_segmentation_boundaries(hidden_states: Any, *, scorer: Any = None, grouper: Any = None) -> list[int]:
    """Return chronological split positions for all strict local peaks.

    If ``d[i]`` is the dissimilarity between token ``i`` and ``i + 1``, a
    peak at ``i`` splits the token sequence at ``i + 1``. The first and last
    dissimilarities cannot be strict peaks because they do not have two
    neighbours, exactly matching the paper's definition.
    """

    import torch

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError(
            "Unsupervised segmentation expects rank-2 token hidden states."
        )
    length = int(hidden_states.shape[0])
    if length < 1:
        raise InvariantError(
            "Unsupervised segmentation requires at least one audio token."
        )

    with torch.no_grad():
        normalized = torch.nn.functional.normalize(
            hidden_states.detach().float(),
            dim=-1,
            eps=1e-8,
        )
        dissimilarity = 1.0 - (normalized[:-1] * normalized[1:]).sum(dim=-1)
        from mllm_comp.compressors.audio_rule_utils import token_scores, contiguous_groups
        if scorer is not None and length > 1:
            dissimilarity = token_scores(scorer({"values": hidden_states.detach(), "scores": dissimilarity}),
                hidden_states, length=length - 1)
        if grouper is not None and length > 1:
            groups = contiguous_groups(grouper({"values": hidden_states.detach(), "scores": dissimilarity}), length=length)
            return [group[0] for group in groups[1:]]
        if int(dissimilarity.numel()) < 3:
            return []
        peak_slots = (
            (dissimilarity[1:-1] > dissimilarity[:-2])
            & (dissimilarity[1:-1] > dissimilarity[2:])
        ).nonzero(as_tuple=False).flatten()
        return [int(slot.item()) + 2 for slot in peak_slots]


def unsupervised_segment_merge(
    hidden_states: Any,
    *,
    keep_count: int | None = None,
    scorer: Any = None, grouper: Any = None, aggregator: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Mean-pool the natural segments defined by adjacent-token peaks.

    ``keep_count`` is only the Qwen preflight expectation. It never changes
    boundary selection; a mismatch is rejected so the implementation cannot
    silently fall back to fixed-rate compression. Boundary decisions are
    detached, while the segment means remain differentiable.
    """

    import torch

    boundaries = unsupervised_segmentation_boundaries(hidden_states, scorer=scorer, grouper=grouper)
    length = int(hidden_states.shape[0])
    natural_keep_count = len(boundaries) + 1
    if keep_count is not None and keep_count != natural_keep_count:
        raise InvariantError(
            "UnSeg preflight count no longer matches the natural local peaks: "
            f"planned {keep_count}, observed {natural_keep_count}."
        )

    endpoints = [0, *boundaries, length]
    segments = list(zip(endpoints, endpoints[1:]))
    output = (
        hidden_states
        if natural_keep_count == length
        else torch.stack(
            [hidden_states[start:end].mean(dim=0) for start, end in segments],
            dim=0,
        )
    )
    if aggregator is not None and natural_keep_count != length:
        from mllm_comp.compressors.audio_rule_utils import aggregate_groups
        output = aggregate_groups(hidden_states, [tuple(range(start, end)) for start, end in segments], aggregator=aggregator)
    if int(output.shape[0]) != natural_keep_count:
        raise InvariantError(
            "Unsupervised segmentation failed to produce one token per segment."
        )
    return output, {
        "natural_peak_count": len(boundaries),
        "natural_peak_boundaries": boundaries,
        "natural_keep_count": natural_keep_count,
        "segments": segments,
        "boundary_detector": "strict_local_peak" if grouper is None else "custom",
        "budget_policy": "natural_peaks",
        "data_dependent_output_length": True,
        "preflight_count_verified": keep_count is not None,
        "merge_value": "mean" if aggregator is None else "custom",
        "token_order": "chronological",
    }


class UnsupervisedSegmentationProjectorStrategy(AudioTokenStrategyBase):
    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "merge"
    metadata_fields = (
        "minimum_keep_tokens",
        "boundary_detector",
        "budget_policy",
        "merge_value",
    )

    def output_length(self, length: int) -> None:
        # Natural peaks depend on the actual application-boundary features.
        return None

    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
        return tuple(details["segments"])

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        if len(plan) == int(hidden_states.shape[0]):
            return hidden_states
        if "aggregator" not in self.rules:
            # Match the original slice/mean graph without recomputing boundaries.
            return _torch().stack([hidden_states[start:end].mean(0) for start, end in plan])
        return aggregate_groups(hidden_states, [tuple(range(start, end)) for start, end in plan],
            aggregator=self.rules.get("aggregator"))

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        return _torch().tensor([start for start, _ in plan], dtype=_torch().long, device=device)

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        return self.replay(reference, details["segments"])

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int | None,
        item_index: int,
    ) -> tuple[Any, None, dict[str, Any]]:
        del scores, item_index
        output, details = unsupervised_segment_merge(
            hidden_states,
            keep_count=keep_count, scorer=self.rules.get("scorer"),
            grouper=self.rules.get("grouper"), aggregator=self.rules.get("aggregator"),
        )
        return output, None, details



class UnsupervisedSegmentationAudio:
    """Projector-level plan provider for paper-faithful UnSeg."""

    name = "unsupervised_segmentation_audio"
    version = "2"
    requires_projector_hidden_state_plan = True

    def validate_config(self, config: dict[str, Any]) -> None:
        UnsupervisedSegmentationSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        spec = UnsupervisedSegmentationSpec.from_config(config)
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        projector_inputs = context.extra.get(
            "projector_hidden_states_by_audio_unit"
        )
        deferred = bool(context.extra.get("defer_token_budget", False))
        if deferred:
            projector_inputs = [None] * len(audio_spans)
        if not isinstance(projector_inputs, Sequence):
            raise InvariantError(
                "UnSeg requires a preflight audio-encoder pass before planning."
            )
        if len(projector_inputs) != len(audio_spans):
            raise InvariantError(
                "UnSeg preflight projector inputs do not match the audio spans: "
                f"expected {len(audio_spans)}, got {len(projector_inputs)}."
            )

        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        peak_total = 0
        for audio_unit_index, (span, hidden_states) in enumerate(
            zip(audio_spans, projector_inputs)
        ):
            original_count = span.end - span.start
            if deferred:
                boundaries = None
                keep_count = None
            else:
                hidden_count = int(hidden_states.shape[0])
                if hidden_count != original_count:
                    raise InvariantError(
                        "UnSeg preflight projector length does not match the audio "
                        f"placeholder span: expected {original_count}, got {hidden_count}."
                    )
                rules = context.extra.get("audio_rules", {})
                boundaries = unsupervised_segmentation_boundaries(hidden_states,
                    scorer=rules.get("scorer"), grouper=rules.get("grouper"))
                keep_count = len(boundaries) + 1
                planned_total += keep_count
                peak_total += len(boundaries)
            original_total += original_count
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
                        "natural_peak_count": None if boundaries is None else len(boundaries),
                        "boundary_detector": spec.boundary_detector,
                        "budget_policy": spec.budget_policy,
                        "data_dependent_output_length": True,
                        **({"_audio_preflight_details": {
                            "segments": list(zip([0, *boundaries], [*boundaries, original_count])),
                            "natural_peak_boundaries": boundaries,
                            "natural_keep_count": keep_count,
                            "preflight_count_verified": True,
                            "custom_preflight_plan_replayed": True,
                            "token_order": "chronological",
                        }} if not deferred and ("scorer" in rules or "grouper" in rules) else {}),
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
            legalization_policy=LegalizationPolicy(
                position_policy="compact",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
                coordinate_policy="drop_removed",
            ),
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                changed_spans=audio_spans,
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "boundary_detector": spec.boundary_detector,
                    "budget_policy": spec.budget_policy,
                    "num_audio_units": len(actions),
                    "natural_peak_count": None if deferred and audio_spans else peak_total,
                    "original_audio_tokens": original_total,
                    "planned_audio_tokens": None if deferred and audio_spans else planned_total,
                    "planned_keep_ratio": (
                        None if deferred and audio_spans else planned_total / original_total if original_total else 1.0
                    ),
                    "paper_natural_output_is_data_dependent": True,
                    "preflight_audio_encoder_pass": not deferred,
                    "selection_deferred_to_runtime": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError(
                "UnsupervisedSegmentationAudio has no trainable state"
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
    "UnsupervisedSegmentationProjectorStrategy",
    "UnsupervisedSegmentationAudio",
    "UnsupervisedSegmentationSpec",
    "unsupervised_segment_merge",
    "unsupervised_segmentation_boundaries",
]
