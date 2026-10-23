"""SA-MAP similarity/attention merging followed by diversity pruning."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from mllm_comp.compressors.audio_rule_utils import AudioTokenStrategyBase, _torch
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
class SimilarityAttentionMergePruneSpec:
    """Normalized settings for the composite SA-MAP stage."""

    keep_ratio: float = 0.50
    similarity_threshold: float = 0.80
    minimum_keep_tokens: int = 1
    attention_power: float = 1.0
    dpp_jitter: float = 1e-6
    score_source: str = "encoder_penultimate_layer_attention"
    merge_value: str = "attention_weighted_mean"
    pruning: str = "greedy_attention_dpp"
    target_site: str = "audio_projector_output_post_projector"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "SimilarityAttentionMergePruneSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        legacy = _mapping(config.get("config"), "config")
        spec = cls(
            keep_ratio=float(
                _first_present(
                    schedule.get("keep_ratio"),
                    legacy.get("keep_ratio"),
                    cls.keep_ratio,
                )
            ),
            similarity_threshold=float(
                _first_present(
                    criterion.get("similarity_threshold"),
                    legacy.get("similarity_threshold"),
                    cls.similarity_threshold,
                )
            ),
            minimum_keep_tokens=_strict_int(
                _first_present(
                    schedule.get("minimum_keep_tokens"),
                    legacy.get("minimum_keep_tokens"),
                    cls.minimum_keep_tokens,
                ),
                "minimum_keep_tokens",
            ),
            attention_power=float(
                _first_present(
                    criterion.get("attention_power"),
                    legacy.get("attention_power"),
                    cls.attention_power,
                )
            ),
            dpp_jitter=float(
                _first_present(
                    criterion.get("dpp_jitter"),
                    legacy.get("dpp_jitter"),
                    cls.dpp_jitter,
                )
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
            pruning=str(
                _first_present(
                    criterion.get("pruning"),
                    legacy.get("pruning"),
                    cls.pruning,
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
        if operation.get("type") not in {None, "merge_prune"}:
            raise ValueError(
                "similarity_attention_merge_prune_audio requires "
                "operation.type=merge_prune"
            )
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if not -1 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be in [-1, 1]")
        if self.minimum_keep_tokens < 1:
            raise ValueError("minimum_keep_tokens must be at least 1")
        if self.attention_power < 0:
            raise ValueError("attention_power must be non-negative")
        if self.dpp_jitter <= 0:
            raise ValueError("dpp_jitter must be positive")
        if self.score_source != "encoder_penultimate_layer_attention":
            raise ValueError(
                "SA-MAP requires score_source=" "'encoder_penultimate_layer_attention'"
            )
        if self.merge_value != "attention_weighted_mean":
            raise ValueError("SA-MAP requires merge_value='attention_weighted_mean'")
        if self.pruning != "greedy_attention_dpp":
            raise ValueError("SA-MAP requires pruning='greedy_attention_dpp'")


def sa_map_keep_count(
    length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if length == 0:
        return 0
    return min(
        length,
        max(minimum_keep_tokens, int(math.floor(length * keep_ratio))),
    )


def similarity_attention_merge_prune(
    hidden_states: Any,
    merge_importance: Any,
    prune_importance: Any,
    *,
    keep_count: int,
    similarity_threshold: float,
    attention_power: float = 1.0,
    dpp_jitter: float = 1e-6,
    scorer: Any = None, grouper: Any = None, aggregator: Any = None, selector: Any = None,
) -> tuple[Any, dict[str, Any]]:
    """Run the paper's G-SAM grouping followed by ADPruner MAP inference.

    ``merge_importance`` is Eq. (4): max over heads followed by aggregation
    over queries. ``prune_importance`` is the diagonal of the head-mean
    attention matrix used by Eq. (8). They are deliberately separate because
    the paper defines different attention reductions for the two stages.
    """

    import torch

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError("SA-MAP expects rank-2 token hidden states.")
    length = int(hidden_states.shape[0])
    for name, scores in (
        ("merge importance", merge_importance),
        ("prune importance", prune_importance),
    ):
        if getattr(scores, "ndim", None) != 1 or int(scores.shape[0]) != length:
            raise InvariantError(
                f"SA-MAP {name} must be rank-1 and match token length."
            )
        if not bool(torch.isfinite(scores).all().item()):
            raise InvariantError(f"SA-MAP {name} contains NaN or Inf.")
        if bool((scores < 0).any().item()):
            raise InvariantError(f"SA-MAP {name} must be non-negative.")
    if not 1 <= keep_count <= length:
        raise InvariantError(
            f"SA-MAP keep_count must be in [1, {length}], got {keep_count}."
        )
    if keep_count == length:
        indices = torch.arange(length, dtype=torch.long, device=hidden_states.device)
        return hidden_states, {
            "groups": [[index] for index in range(length)],
            "post_merge_tokens": length,
            "selected_group_indices": indices,
            "token_order": "chronological",
        }

    from mllm_comp.compressors.audio_rule_utils import token_scores, token_indices, contiguous_groups
    if scorer is not None:
        custom = scorer({"values": hidden_states.detach(), "merge": merge_importance,
            "prune": prune_importance, "keep_count": keep_count})
        if not isinstance(custom, dict) or set(custom) != {"merge", "prune"}:
            raise InvariantError("SA-MAP scorer must return distinct merge/prune importance vectors.")
        merge_importance = token_scores(custom["merge"], hidden_states, length=length, nonnegative=True)
        prune_importance = token_scores(custom["prune"], hidden_states, length=length, nonnegative=True)
    detached = torch.nn.functional.normalize(
        hidden_states.detach().float(),
        dim=-1,
        eps=1e-8,
    )
    groups: list[list[int]] = []
    current: list[int] = []
    for token_index in range(length):
        if not current:
            current = [token_index]
            continue
        # Eq. (3): average the pairwise cosine similarities between the next
        # token and every token already in the contiguous group. Cosine with
        # the group's mean feature is not equivalent because it renormalizes
        # the group center.
        similarity = (detached[current] @ detached[token_index]).mean()
        if float(similarity.item()) >= similarity_threshold:
            current.append(token_index)
        else:
            groups.append(current)
            current = [token_index]
    groups.append(current)

    if grouper is not None:
        groups = contiguous_groups(grouper({"values": hidden_states.detach(), "merge": merge_importance,
            "prune": prune_importance, "similarity_threshold": similarity_threshold,
            "keep_count": keep_count}), length=length)
    positive_prune_importance = prune_importance.to(hidden_states.device).float()
    merged = merge_sa_map_groups(hidden_states, merge_importance, groups, aggregator=aggregator)
    merged_attention: list[Any] = []
    group_centers: list[float] = []
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long, device=hidden_states.device)
        # The official ADPruner implementation averages the per-token
        # diagonal importance within a merged group before Eq. (8).
        merged_attention.append(
            positive_prune_importance.index_select(0, indices).mean()
        )
        group_centers.append(sum(group) / len(group))
    group_importance = torch.stack(merged_attention).to(device=merged.device)
    post_merge_count = int(merged.shape[0])
    if post_merge_count < keep_count:
        raise InvariantError(
            "SA-MAP threshold grouping produced fewer tokens than the requested "
            "final budget. The paper requires selecting a higher similarity "
            "threshold; grouping is not truncated by N-keep_count."
        )

    if post_merge_count == keep_count:
        selected = torch.arange(keep_count, dtype=torch.long, device=merged.device)
    elif selector is not None:
        selected = token_indices(selector({"values": merged.detach(), "scores": group_importance,
            "groups": groups, "keep_count": keep_count}), length=post_merge_count,
            count=keep_count, device=merged.device)
    else:
        selected = _greedy_attention_dpp_indices(merged, group_importance,
            keep_count=keep_count, attention_power=attention_power, jitter=dpp_jitter)
    selected = torch.sort(selected)[0]
    output = merged.index_select(0, selected)
    return output, {
        "groups": groups,
        **({"_merge_importance": merge_importance.detach()} if scorer is not None else {}),
        "post_merge_tokens": post_merge_count,
        "selected_group_indices": selected.detach(),
        "selected_group_centers": [
            group_centers[int(index)] for index in selected.detach().cpu().tolist()
        ],
        "token_order": "chronological",
    }


def merge_sa_map_groups(hidden_states: Any, merge_importance: Any, groups: Any, *, aggregator: Any = None) -> Any:
    """Original Eq. (5) value path, shared by compression and checkpoint replay."""
    import torch

    importance = merge_importance.to(hidden_states.device).float()
    merged_values = []
    for group in groups:
        indices = torch.tensor(group, dtype=torch.long, device=hidden_states.device)
        group_scores = importance.index_select(0, indices)
        denominator = group_scores.sum()
        if float(denominator.item()) <= 0:
            raise InvariantError("SA-MAP Eq. (5) received a group with zero merge importance.")
        weights = group_scores / denominator
        values = hidden_states.index_select(0, indices)
        if aggregator is None:
            output = (values * weights.to(hidden_states.dtype).unsqueeze(-1)).sum(dim=0)
        else:
            from mllm_comp.compressors.audio_rule_utils import validate_values
            output = aggregator({"values": values, "indices": tuple(group),
                "group_index": len(merged_values), "weights": weights})
            validate_values(output, hidden_states[group[0]])
        merged_values.append(output)
    return torch.stack(merged_values, dim=0)


def _greedy_attention_dpp_indices(
    hidden_states: Any,
    importance: Any,
    *,
    keep_count: int,
    attention_power: float,
    jitter: float,
) -> Any:
    """Greedy MAP inference for a PSD attention-weighted DPP kernel."""

    import torch

    with torch.no_grad(), torch.autocast(
        device_type=hidden_states.device.type, enabled=False
    ):
        kernel = _attention_weighted_dpp_kernel(
            hidden_states,
            importance,
            attention_power=attention_power,
        )
        count = int(kernel.shape[0])
        if not bool(torch.isfinite(kernel).all().item()):
            raise InvariantError(
                "SA-MAP DPP kernel contains NaN or Inf before selection: "
                f"candidates={count}, keep_count={keep_count}, "
                f"dtype={kernel.dtype}, device={kernel.device}."
            )
        residual = kernel.diagonal().clone()
        coefficients = torch.zeros(
            keep_count,
            count,
            dtype=kernel.dtype,
            device=kernel.device,
        )
        selected_mask = torch.zeros(count, dtype=torch.bool, device=kernel.device)
        selected: list[int] = []
        for step in range(keep_count):
            # Keep exclusion separate from numeric residuals: clamp_min(0)
            # would otherwise turn a previously selected -inf back into zero.
            candidate_scores = residual.masked_fill(selected_mask, float("-inf"))
            index = int(torch.argmax(candidate_scores).item())
            if not bool(torch.isfinite(candidate_scores[index]).item()):
                raise InvariantError(
                    "SA-MAP DPP encountered a non-finite candidate score: "
                    f"step={step + 1}, candidates={count}, keep_count={keep_count}, "
                    f"selected_count={len(selected)}, "
                    f"score={candidate_scores[index].item()}, "
                    f"dtype={kernel.dtype}, device={kernel.device}, jitter={jitter}."
                )
            selected.append(index)
            selected_mask[index] = True
            if step + 1 < keep_count:
                denominator = residual[index].clamp_min(jitter).sqrt()
                correction = (
                    (
                        coefficients[:step, index].unsqueeze(0) @ coefficients[:step]
                    ).squeeze(0)
                    if step
                    else 0.0
                )
                row = (kernel[index] - correction) / denominator
                coefficients[step] = row
                residual = (residual - row.square()).clamp_min(0)
        return torch.tensor(selected, dtype=torch.long, device=kernel.device)


def _attention_weighted_dpp_kernel(
    hidden_states: Any,
    importance: Any,
    *,
    attention_power: float,
) -> Any:
    """Construct the Eq. (8) ADPruner kernel for testing and MAP inference."""

    import torch

    # Repeated features make the DPP residual updates cancellation-sensitive.
    # Build the kernel in FP64 from the start; promoting an FP32 Gram matrix
    # afterwards cannot recover its lost precision. Only selection uses this
    # dtype, while the merged output retains its original dtype and gradient.
    features = torch.nn.functional.normalize(
        hidden_states.detach().to(dtype=torch.float64),
        dim=-1,
        eps=1e-8,
    )
    quality = importance.detach().to(dtype=torch.float64).clamp_min(0).pow(attention_power)
    basis = features * quality.unsqueeze(-1)
    # Eq. (8): diag(A) @ L @ diag(A). At attention_power=1 this is
    # exactly the paper kernel, rather than diag(sqrt(A)) @ L @
    # diag(sqrt(A)).
    return basis @ basis.transpose(0, 1)


class SimilarityAttentionMergePruneProjectorStrategy(AudioTokenStrategyBase):
    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "merge_prune"
    uses_distinct_merge_prune_importance = True
    metadata_fields = (
        "keep_ratio",
        "similarity_threshold",
        "pruning",
        "attention_power",
        "dpp_jitter",
    )

    def output_length(self, length: int) -> int:
        return sa_map_keep_count(length, self.spec.keep_ratio, minimum_keep_tokens=self.spec.minimum_keep_tokens)


    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
        importance = details.get("_merge_importance")
        if importance is None:
            importance = scores["merge"] if isinstance(scores, Mapping) else scores[:, 0]
        return (tuple(tuple(group) for group in details["groups"]), importance.detach(), details["selected_group_indices"].detach())

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        groups, importance, selected = plan
        if selected.numel() == int(hidden_states.shape[0]):
            return hidden_states
        return merge_sa_map_groups(hidden_states, importance, groups, aggregator=self.rules.get("aggregator")).index_select(0, selected.to(hidden_states.device))

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        groups, _, selected = plan
        anchors = _torch().tensor([group[0] for group in groups], dtype=_torch().long, device=device)
        return anchors.index_select(0, selected.to(device))

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        if "aggregator" in self.rules:
            return self.replay(reference, self.replay_plan(keep_count, details, indices, scores=scores))
        # Preserve the original reference arithmetic (including its FP32 weights).
        importance = details.get("_merge_importance", scores["merge"]).to(reference.device).float()
        groups = []
        for group_index in details["selected_group_indices"].tolist():
            group = _torch().tensor(details["groups"][group_index], device=reference.device)
            weights = importance.index_select(0, group)
            weights = weights / weights.sum()
            groups.append((reference.index_select(0, group) * weights.unsqueeze(-1)).sum(0))
        return _torch().stack(groups)

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, None, dict[str, Any]]:
        del item_index
        if _torch().is_tensor(scores) and scores.ndim == 2 and scores.shape[-1] == 2:
            scores = {"merge": scores[:, 0], "prune": scores[:, 1]}
        if not isinstance(scores, Mapping):
            raise InvariantError(
                "SA-MAP requires separate merge and prune attention importance."
            )
        merge_importance = scores.get("merge")
        prune_importance = scores.get("prune")
        if merge_importance is None or prune_importance is None:
            raise InvariantError(
                "SA-MAP attention importance must contain merge and prune values."
            )
        output, details = similarity_attention_merge_prune(
            hidden_states,
            merge_importance,
            prune_importance,
            keep_count=keep_count,
            similarity_threshold=self.spec.similarity_threshold,
            attention_power=self.spec.attention_power,
            dpp_jitter=self.spec.dpp_jitter, **self.rules,
        )
        return output, None, details



class SimilarityAttentionMergePruneAudio:
    """Post-projector composite SA-MAP plan provider."""

    name = "similarity_attention_merge_prune_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        SimilarityAttentionMergePruneSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = SimilarityAttentionMergePruneSpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        for audio_unit_index, span in enumerate(audio_spans):
            original_count = span.end - span.start
            keep_count = sa_map_keep_count(
                original_count,
                spec.keep_ratio,
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
                        "similarity_threshold": spec.similarity_threshold,
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
                    "similarity_threshold": spec.similarity_threshold,
                    "pruning": spec.pruning,
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
            raise ValueError(
                "SimilarityAttentionMergePruneAudio has no trainable state"
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
