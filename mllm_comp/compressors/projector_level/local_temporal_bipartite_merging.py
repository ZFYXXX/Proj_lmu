"""Paper-style local temporal bipartite merging for audio tokens.

Each iteration fixes the even/odd source and destination partitions, lets every
source choose its best destination inside the paper's partition-order window,
and merges the highest-ranked sources as one batch. Pair selection is
non-differentiable, while the equal-mean merge remains on the autograd path so
the implementation can also be exercised by the platform's training runtime.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Any

from mllm_comp.compressors.audio_rule_utils import (
    AudioTokenStrategyBase,
    _torch,
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
class LocalTemporalBipartiteMergingSpec:
    """Normalized LTBM settings."""

    keep_ratio: float = 0.50
    window_size: int = 8
    minimum_keep_tokens: int = 1
    score_source: str = "projector_token_cosine"
    pairing: str = "alternating_bipartite_local"
    merge_value: str = "mean"
    target_site: str = "audio_encoder_output_pre_projector"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "LocalTemporalBipartiteMergingSpec":
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
            window_size=_strict_int(
                _first_present(
                    criterion.get("window_size"),
                    legacy.get("window_size"),
                    cls.window_size,
                ),
                "window_size",
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
            pairing=str(
                _first_present(
                    criterion.get("pairing"),
                    legacy.get("pairing"),
                    cls.pairing,
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
                "local_temporal_bipartite_merging_audio requires operation.type=merge"
            )
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if self.window_size < 1:
            raise ValueError("window_size must be at least 1")
        if self.minimum_keep_tokens != 1:
            raise ValueError("paper-faithful LTBM requires minimum_keep_tokens=1")
        if self.score_source != "projector_token_cosine":
            raise ValueError("LTBM requires score_source='projector_token_cosine'")
        if self.pairing != "alternating_bipartite_local":
            raise ValueError("LTBM requires pairing='alternating_bipartite_local'")
        if self.merge_value != "mean":
            raise ValueError("LTBM requires merge_value='mean'")


def ltbm_keep_count(
    length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the exact LTBM output budget for one audio unit."""

    if length < 0:
        raise ValueError("length must be non-negative")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if minimum_keep_tokens != 1:
        raise ValueError("LTBM requires minimum_keep_tokens=1")
    if length == 0:
        return 0
    # Platform budget adaptation: deliberately keep no more than rho * L
    # tokens. The paper uses round(rho * L), so half-token cases retain one
    # fewer token here (for example, 185 * 0.5 becomes 92 instead of 93).
    target_length = int(math.floor(length * keep_ratio))
    return min(length, max(minimum_keep_tokens, target_length))


def local_temporal_bipartite_merge(
    hidden_states: Any,
    *,
    keep_count: int,
    window_size: int,
    grouper: Callable[[Mapping[str, Any]], Any] | None = None,
    aggregator: Callable[[Mapping[str, Any]], Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Apply the paper's iterative local temporal bipartite merge.

    At the beginning of every round, even positions form the fixed source set
    ``S`` and odd positions form the fixed destination set ``D``.  Every source
    chooses its best destination satisfying ``abs(i - j) <= window_size``,
    where ``i`` and ``j`` are indices *within S and D*.  Sources are ranked by
    their best similarity and the required highest-ranked sources are merged
    simultaneously.  A destination and all sources assigned to it in that
    round receive equal weight, as in Equation (5).  Only the next round
    repartitions the already compressed chronological sequence.

    Optional stateless rules replace only the decisions or values of this loop.
    The grouper receives detached ``values``, S/D index tuples, ``window_size``,
    ``merge_count`` and ``round_index``; it returns current-round index pairs.
    The aggregator receives differentiable group ``values`` (destination first),
    ``indices``, ``destination`` and ``round_index`` and returns one value vector.
    The same aggregator must be used when replaying saved pairs.
    """

    import torch

    if getattr(hidden_states, "ndim", None) != 2:
        raise InvariantError("LTBM expects rank-2 token hidden states.")
    length = int(hidden_states.shape[0])
    if not 1 <= keep_count <= length:
        raise InvariantError(
            f"LTBM keep_count must be in [1, {length}], got {keep_count}."
        )
    if window_size < 1:
        raise InvariantError("LTBM window_size must be positive.")
    if keep_count == length:
        return hidden_states, {
            "merge_count": 0,
            "merge_pairs": [],
            "merge_rounds": [],
            "window_coordinate": "source_destination_partition_order",
            "merge_value": "round_equal_mean",
            "token_order": "chronological",
        }

    values = hidden_states
    merge_pairs: list[tuple[int, int]] = []
    merge_rounds: list[dict[str, Any]] = []
    while int(values.shape[0]) > keep_count:
        current_length = int(values.shape[0])
        source_indices = torch.arange(
            0,
            current_length,
            2,
            dtype=torch.long,
            device=values.device,
        )
        destination_indices = torch.arange(
            1,
            current_length,
            2,
            dtype=torch.long,
            device=values.device,
        )
        if int(destination_indices.numel()) == 0:
            raise InvariantError("LTBM could not form a destination partition.")
        if grouper is not None:
            merges_this_round = min(current_length - keep_count, int(source_indices.numel()))
            with torch.no_grad():
                proposed = grouper({
                    "values": values.detach(),
                    "source_indices": tuple(range(0, current_length, 2)),
                    "destination_indices": tuple(range(1, current_length, 2)),
                    "window_size": window_size,
                    "merge_count": merges_this_round,
                    "round_index": len(merge_rounds),
                })
            round_pairs = _validate_ltbm_pairs(
                proposed, current_length=current_length,
                merge_count=merges_this_round, window_size=window_size,
            )
        else:
            with torch.no_grad():
                source = torch.nn.functional.normalize(
                    values.index_select(0, source_indices).detach().float(),
                    dim=-1,
                    eps=1e-8,
                )
                destination = torch.nn.functional.normalize(
                    values.index_select(0, destination_indices).detach().float(),
                    dim=-1,
                    eps=1e-8,
                )
                similarity = source @ destination.transpose(0, 1)
                # Paper Equation (4): i and j are order indices inside S and D,
                # rather than positions or accumulated temporal centers in X.
                source_order = torch.arange(
                    int(source_indices.numel()),
                    dtype=torch.long,
                    device=values.device,
                )
                destination_order = torch.arange(
                    int(destination_indices.numel()),
                    dtype=torch.long,
                    device=values.device,
                )
                local = (
                    source_order[:, None] - destination_order[None, :]
                ).abs() <= window_size
                similarity = similarity.masked_fill(~local, float("-inf"))
                best_similarity, best_destination_slots = similarity.max(dim=1)
                if not bool(torch.isfinite(best_similarity).all().item()):
                    raise InvariantError(
                        "LTBM locality window left a source without a destination."
                    )
                source_ranking = torch.argsort(
                    best_similarity,
                    descending=True,
                    stable=True,
                )

                merges_this_round = min(
                    current_length - keep_count,
                    int(source_indices.numel()),
                )
                selected_source_slots = source_ranking[:merges_this_round].tolist()
                selected_destination_slots = [
                    int(best_destination_slots[source_slot].item())
                    for source_slot in selected_source_slots
                ]

            round_pairs: list[tuple[int, int]] = []
            for source_slot, destination_slot in zip(
                selected_source_slots,
                selected_destination_slots,
            ):
                pair = (
                    int(source_indices[source_slot].item()),
                    int(destination_indices[destination_slot].item()),
                )
                round_pairs.append(pair)

        merge_pairs.extend(round_pairs)
        values = replay_ltbm_round(values, round_pairs, aggregator=aggregator, round_index=len(merge_rounds))
        merge_rounds.append(
            {
                "round_index": len(merge_rounds),
                "input_tokens": current_length,
                "output_tokens": int(values.shape[0]),
                "merge_count": merges_this_round,
                "merge_pairs": round_pairs,
            }
        )

    return values, {
        "merge_count": length - keep_count,
        "merge_pairs": merge_pairs,
        "merge_rounds": merge_rounds,
        "window_coordinate": "source_destination_partition_order",
        "merge_value": "custom" if aggregator is not None else "round_equal_mean",
        **({"grouping_rule": "custom"} if grouper is not None else {}),
        "token_order": "chronological",
    }


def _validate_ltbm_pairs(pairs: Any, *, current_length: int, merge_count: int, window_size: int) -> list[tuple[int, int]]:
    """Validate the existing local S/D domain before any values are changed."""
    import torch

    if isinstance(pairs, torch.Tensor):
        if pairs.ndim != 2 or pairs.shape[1] != 2 or pairs.dtype not in {
            torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8,
        }:
            raise InvariantError("LTBM grouper must return integer pairs with shape [merge_count, 2].")
        pairs = pairs.tolist()
    if not isinstance(pairs, (list, tuple)) or len(pairs) != merge_count:
        raise InvariantError("LTBM grouper must return the exact per-round merge count.")
    result = []
    seen = set()
    for pair in pairs:
        if (not isinstance(pair, (list, tuple)) or len(pair) != 2
                or any(isinstance(i, bool) or not isinstance(i, int) for i in pair)):
            raise InvariantError("LTBM grouper pairs must contain two integer token indices.")
        source, destination = pair
        if not (0 <= source < current_length and source % 2 == 0
                and 0 <= destination < current_length and destination % 2 == 1):
            raise InvariantError("LTBM grouper must map even sources to retained odd destinations.")
        if source in seen:
            raise InvariantError("LTBM grouper sources must be unique within a round.")
        if abs(source // 2 - destination // 2) > window_size:
            raise InvariantError("LTBM grouper pair exceeds the partition-order locality window.")
        seen.add(source)
        result.append((source, destination))
    return result


def replay_ltbm_round(
    values: Any, pairs: Any, *,
    aggregator: Callable[[Mapping[str, Any]], Any] | None = None,
    round_index: int = 0,
) -> Any:
    """Replay saved decisions using the same differentiable aggregation rule."""
    import torch

    if not pairs:
        return values
    sources = {source for source, _ in pairs}
    groups: dict[int, list[int]] = {}
    for source, destination in pairs:
        groups.setdefault(destination, []).append(source)
    # Updated destinations keep their positions; removed sources disappear.
    outputs = []
    for index in range(int(values.shape[0])):
        if index in sources:
            continue
        assigned = groups.get(index)
        if not assigned:
            outputs.append(values[index])
            continue
        indices = (index, *assigned)
        group_values = torch.stack([values[i] for i in indices])
        if aggregator is None:
            output = group_values.mean(0)
        else:
            output = aggregator({"values": group_values, "indices": indices,
                                 "destination": index, "round_index": round_index})
            if (not isinstance(output, torch.Tensor) or output.shape != values[index].shape
                    or output.device != values.device or output.dtype != values.dtype):
                raise InvariantError("LTBM aggregator must return one vector with the input width, device and dtype.")
            if not bool(torch.isfinite(output).all().item()):
                raise InvariantError("LTBM aggregator returned non-finite values.")
            if torch.is_grad_enabled() and group_values.requires_grad and not output.requires_grad:
                raise InvariantError("LTBM aggregator detached values from the gradient graph.")
        outputs.append(output)
    return torch.stack(outputs)


class LocalTemporalBipartiteProjectorStrategy(AudioTokenStrategyBase):
    layer_families = frozenset({"encoder_token", "llm_token"})
    token_operation = "merge"
    merge_position_policy = "preserve_absolute_destination"
    metadata_fields = (
        "keep_ratio",
        "window_size",
        "pairing",
        "merge_value",
    )

    def __init__(self, spec: Any, *, rules: Mapping[str, Any] | None = None,
                 rule_specs: Mapping[str, Any] | None = None) -> None:
        super().__init__(spec)
        self.rules = dict(rules or {})
        self.rule_specs = dict(rule_specs or {})

    @property
    def layer_score_source(self) -> str:
        return (
            "custom_grouping_rule"
            if "grouper" in self.rules
            else "application_boundary_token_cosine"
        )

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        if self.rule_specs:
            metadata.update(rules=self.rule_specs, rules_changed=True, canonical_evidence_applicable=False, development_runtime=True)
        if "grouper" in self.rules:
            metadata.update(score_source="custom_grouping_rule", pairing="custom_local_bipartite")
        if "aggregator" in self.rules:
            metadata["merge_value"] = "custom"
        return metadata

    def output_length(self, length: int) -> int:
        return ltbm_keep_count(length, self.spec.keep_ratio)

    def replay_plan(self, keep_count: int, details: dict[str, Any], indices: Any = None, *, scores: Any = None) -> Any:
        return tuple(tuple(row["merge_pairs"]) for row in details["merge_rounds"])

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        for round_index, pairs in enumerate(plan):
            hidden_states = replay_ltbm_round(hidden_states, pairs,
                aggregator=self.rules.get("aggregator"), round_index=round_index)
        return hidden_states

    def map_reference(self, reference: Any, *, indices: Any, details: dict[str, Any], scores: Any, keep_count: int) -> Any:
        return self.replay(reference, self.replay_plan(keep_count, details))

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        indices = list(range(length))
        for pairs in plan:
            removed = {source for source, _ in pairs}
            indices = [index for slot, index in enumerate(indices) if slot not in removed]
        return _torch().tensor(indices, device=device, dtype=_torch().long)

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, None, dict[str, Any]]:
        del scores, item_index
        output, details = local_temporal_bipartite_merge(
            hidden_states,
            keep_count=keep_count,
            window_size=self.spec.window_size,
            grouper=self.rules.get("grouper"), aggregator=self.rules.get("aggregator"),
        )
        return output, None, details



class LocalTemporalBipartiteMergingAudio:
    """Projector-level plan provider for LTBM."""

    name = "local_temporal_bipartite_merging_audio"
    version = "3"

    def validate_config(self, config: dict[str, Any]) -> None:
        LocalTemporalBipartiteMergingSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = LocalTemporalBipartiteMergingSpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        for audio_unit_index, span in enumerate(audio_spans):
            original_count = span.end - span.start
            keep_count = ltbm_keep_count(
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
                        "window_size": spec.window_size,
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
                    "window_size": spec.window_size,
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
                "LocalTemporalBipartiteMergingAudio has no trainable state"
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
