"""Paper-default Adjacent Token Merging for audio encoders.

The algorithm contract is kept separate from the Qwen2.5-Omni runtime
adapter. This module owns the paper defaults, deterministic length planning
and deferred merge actions. The backbone adapter owns key capture, layer
execution and the repair of provider-specific sequence metadata.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, Literal

from mllm_comp.compressors.audio_rule_utils import AudioLayerTokenStrategyBase, _torch
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


MergePolicy = Literal["fixed_ratio"]


@dataclass(frozen=True)
class AdjacentTokenMergingSpec:
    """Normalized A-ToMe settings.

    The defaults follow the paper's utterance-based setup: A-ToMe starts at
    layer 2, repeats every three encoder layers and merges 10% of the current
    tokens at every active layer.
    """

    merge_ratio: float = 0.10
    first_layer: int = 2
    layer_interval: int = 3
    layer_indices: tuple[int, ...] = ()
    policy: MergePolicy = "fixed_ratio"
    score_source: str = "encoder_key_cosine"
    pairing: str = "disjoint_adjacent"
    merge_value: str = "mean"
    target_site: str = "audio_encoder_layer_post_mhsa_pre_ffn"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AdjacentTokenMergingSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        activation_scope = _mapping(
            config.get("activation_scope"),
            "activation_scope",
        )
        legacy = _mapping(config.get("config"), "config")

        raw_layers = _first_present(
            activation_scope.get("layers"),
            legacy.get("layers"),
            (),
        )
        if not isinstance(raw_layers, (list, tuple)):
            raise ValueError("activation_scope.layers must be a list of layer indices")
        layer_indices = tuple(
            _strict_int(value, "activation_scope.layers") for value in raw_layers
        )
        spec = cls(
            merge_ratio=float(
                _first_present(
                    schedule.get("merge_ratio"),
                    legacy.get("merge_ratio"),
                    cls.merge_ratio,
                )
            ),
            first_layer=_strict_int(
                _first_present(
                    activation_scope.get("first_layer"),
                    schedule.get("first_layer"),
                    legacy.get("first_layer"),
                    cls.first_layer,
                ),
                "first_layer",
            ),
            layer_interval=_strict_int(
                _first_present(
                    activation_scope.get("layer_interval"),
                    schedule.get("layer_interval"),
                    legacy.get("layer_interval"),
                    cls.layer_interval,
                ),
                "layer_interval",
            ),
            layer_indices=layer_indices,
            policy=str(
                _first_present(
                    criterion.get("policy"),
                    legacy.get("policy"),
                    cls.policy,
                )
            ),  # type: ignore[arg-type]
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
        operation_type = operation.get("type")
        if operation_type not in {None, "merge"}:
            raise ValueError(
                "adjacent_token_merging_audio requires operation.type=merge"
            )
        return spec

    def validate(self) -> None:
        if not 0 <= self.merge_ratio <= 0.5:
            raise ValueError(
                f"merge_ratio must be in [0, 0.5], got {self.merge_ratio}"
            )
        if self.first_layer < 0:
            raise ValueError("first_layer must be non-negative")
        if self.layer_interval < 1:
            raise ValueError("layer_interval must be at least 1")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("activation_scope.layers contains duplicate indices")
        if any(index < 0 for index in self.layer_indices):
            raise ValueError("activation_scope.layers must be non-negative")
        if self.policy != "fixed_ratio":
            raise ValueError(
                "the first A-ToMe vertical supports only policy='fixed_ratio'"
            )
        if self.score_source != "encoder_key_cosine":
            raise ValueError(
                "adjacent_token_merging_audio requires score_source='encoder_key_cosine'"
            )
        if self.pairing != "disjoint_adjacent":
            raise ValueError(
                "adjacent_token_merging_audio requires pairing='disjoint_adjacent'"
            )
        if self.merge_value != "mean":
            raise ValueError(
                "adjacent_token_merging_audio requires merge_value='mean'"
            )

    def resolve_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        """Resolve explicit layers or the paper's every-three-layers rule."""

        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        indices = self.layer_indices or tuple(
            range(self.first_layer, num_layers, self.layer_interval)
        )
        if not indices:
            raise ValueError(
                "A-ToMe activation scope selects no encoder layers; adjust "
                "first_layer or activation_scope.layers"
            )
        outside = [index for index in indices if index >= num_layers]
        if outside:
            raise ValueError(
                f"A-ToMe layer indices {outside} are outside {num_layers} encoder layers"
            )
        return indices


def adjacent_merge_count(length: int, merge_ratio: float) -> int:
    """Number of disjoint adjacent pairs merged at one layer."""

    if length < 0:
        raise ValueError(f"length must be non-negative, got {length}")
    if not 0 <= merge_ratio <= 0.5:
        raise ValueError(f"merge_ratio must be in [0, 0.5], got {merge_ratio}")
    return min(length // 2, int(math.floor(length * merge_ratio)))


def adjacent_merge_output_length(length: int, merge_ratio: float) -> int:
    """Length after replacing selected adjacent pairs with their means."""

    return length - adjacent_merge_count(length, merge_ratio)


def qwen_audio_chunk_lengths_after_cnn(
    feature_length: int,
    n_window: int,
) -> list[int]:
    """Mirror Qwen2.5-Omni's audio chunking and stride-2 convolution."""

    if feature_length < 1:
        raise ValueError("audio feature length must be positive")
    if n_window < 1:
        raise ValueError("n_window must be positive")
    raw_chunk_size = n_window * 2
    chunks: list[int] = []
    remaining = feature_length
    while remaining > 0:
        raw_length = min(raw_chunk_size, remaining)
        chunks.append((raw_length - 1) // 2 + 1)
        remaining -= raw_length
    return chunks


def qwen_planned_a_tome_lengths(
    feature_length: int,
    *,
    n_window: int,
    merge_ratio: float,
    active_layer_count: int,
) -> tuple[list[int], list[int]]:
    """Return initial and final per-chunk encoder lengths for fixed-ratio A-ToMe."""

    if active_layer_count < 0:
        raise ValueError("active_layer_count must be non-negative")
    initial = qwen_audio_chunk_lengths_after_cnn(feature_length, n_window)
    final = list(initial)
    for _ in range(active_layer_count):
        final = [
            adjacent_merge_output_length(length, merge_ratio)
            for length in final
        ]
    return initial, final


def merge_adjacent_hidden_states(
    hidden_states: Any,
    key_states: Any,
    *,
    merge_ratio: float,
    scorer: Any = None, grouper: Any = None, aggregator: Any = None,
) -> tuple[Any, Any, Any]:
    """Merge the most similar disjoint pairs while preserving time order.

    Returns ``(merged_hidden_states, selected_pair_starts, similarities)``.
    Pair starts use coordinates in the input chunk and are always even.
    """

    torch = _torch()
    if hidden_states.ndim != 2 or key_states.ndim < 2:
        raise InvariantError(
            "A-ToMe expects rank-2 hidden states and token-major key states."
        )
    length = int(hidden_states.shape[0])
    if int(key_states.shape[0]) != length:
        raise InvariantError(
            "A-ToMe hidden-state and key-state lengths do not match."
        )
    pair_count = length // 2
    merge_count = min(pair_count, int(length * merge_ratio))
    empty_indices = torch.empty(
        0,
        dtype=torch.long,
        device=hidden_states.device,
    )
    empty_scores = torch.empty(
        0,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    if merge_count == 0:
        return hidden_states, empty_indices, empty_scores

    paired_keys = key_states[: pair_count * 2].reshape(pair_count, 2, -1)
    similarities = torch.nn.functional.cosine_similarity(
        paired_keys[:, 0].float(),
        paired_keys[:, 1].float(),
        dim=-1,
        eps=1e-8,
    )
    from mllm_comp.compressors.audio_rule_utils import token_scores, token_indices
    if scorer is not None:
        similarities = token_scores(scorer({"values": hidden_states.detach(), "keys": key_states.detach(),
            "scores": similarities, "merge_count": merge_count}), hidden_states, length=pair_count)
    try:
        order = torch.argsort(similarities, descending=True, stable=True)
    except TypeError:  # pragma: no cover - verified PyTorch supports stable
        order = torch.argsort(similarities, descending=True)
    selected_pairs = order[:merge_count]
    pair_starts = torch.sort(selected_pairs * 2)[0].to(dtype=torch.long)
    if grouper is not None:
        pair_starts = token_indices(grouper({"values": hidden_states.detach(), "keys": key_states.detach(),
            "scores": similarities, "pair_starts": tuple(range(0, pair_count * 2, 2)),
            "merge_count": merge_count}), length=pair_count * 2, count=merge_count, device=hidden_states.device)
        if bool((pair_starts % 2 != 0).any()):
            raise InvariantError("Adjacent grouper must choose original disjoint even pair starts.")
    return replay_adjacent_pairs(hidden_states, pair_starts, aggregator=aggregator), pair_starts, similarities



def replay_adjacent_pairs(hidden_states: Any, pair_starts: Any, *, aggregator: Any = None) -> Any:
    """Apply saved disjoint pairs with the same value graph as the first pass."""
    torch = _torch()
    if not int(pair_starts.numel()):
        return hidden_states
    length = int(hidden_states.shape[0])
    if aggregator is not None:
        from mllm_comp.compressors.audio_rule_utils import aggregate_groups
        starts = set(pair_starts.tolist())
        groups = [(i, i + 1) if i in starts else (i,) for i in range(length) if i - 1 not in starts]
        # Only merged groups invoke the aggregator; retained singletons are identities.
        return torch.cat([aggregate_groups(hidden_states, [group], aggregator=aggregator)
            if len(group) > 1 else hidden_states[group[0]:group[0] + 1] for group in groups])
    pair_count = length // 2
    selected_pairs = pair_starts.to(hidden_states.device) // 2
    selected_mask = torch.zeros(
        pair_count,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    selected_mask[selected_pairs] = True

    paired_hidden = hidden_states[: pair_count * 2].reshape(
        pair_count,
        2,
        *hidden_states.shape[1:],
    )
    pair_means = paired_hidden.mean(dim=1)
    mask_shape = (pair_count,) + (1,) * (pair_means.ndim - 1)
    first_outputs = torch.where(
        selected_mask.reshape(mask_shape),
        pair_means,
        paired_hidden[:, 0],
    )
    rewritten = torch.stack((first_outputs, paired_hidden[:, 1]), dim=1).reshape(
        pair_count * 2,
        *hidden_states.shape[1:],
    )
    keep_mask = torch.ones(
        pair_count * 2,
        dtype=torch.bool,
        device=hidden_states.device,
    )
    keep_mask[1::2] = ~selected_mask
    merged = rewritten[keep_mask]
    if length % 2:
        merged = torch.cat((merged, hidden_states[-1:]), dim=0)
    return merged



def merge_ragged_adjacent_hidden_states(
    hidden_states: Any,
    key_states: Any,
    chunk_lengths: list[int],
    *,
    merge_ratio: float,
    scorer: Any = None, grouper: Any = None, aggregator: Any = None,
    collect_details: bool = False,
    replay_decisions: list[Any] | None = None,
) -> tuple[Any, list[int], list[dict[str, Any]]]:
    """Apply adjacent merging independently inside each ragged audio chunk."""

    if sum(chunk_lengths) != int(hidden_states.shape[0]):
        raise InvariantError(
            "A-ToMe chunk lengths do not sum to the hidden-state length."
        )
    if int(key_states.shape[0]) != int(hidden_states.shape[0]):
        raise InvariantError(
            "A-ToMe key-state length does not match the hidden-state length."
        )
    hidden_chunks = hidden_states.split(chunk_lengths, dim=0)
    key_chunks = key_states.split(chunk_lengths, dim=0)
    merged_chunks: list[Any] = []
    output_lengths: list[int] = []
    reports: list[dict[str, Any]] = []
    for chunk_index, (hidden_chunk, key_chunk) in enumerate(
        zip(hidden_chunks, key_chunks)
    ):
        merged, pair_starts, similarities = merge_adjacent_hidden_states(
            hidden_chunk,
            key_chunk,
            merge_ratio=merge_ratio, scorer=scorer, grouper=grouper, aggregator=aggregator,
        )
        if replay_decisions is not None:
            replay_decisions.append(pair_starts.detach())
        merged_chunks.append(merged)
        output_length = int(merged.shape[0])
        output_lengths.append(output_length)
        if collect_details:
            reports.append(
                {
                    "chunk_index": chunk_index,
                    "input_tokens": int(hidden_chunk.shape[0]),
                    "output_tokens": output_length,
                    "merged_pairs": int(pair_starts.numel()),
                    "selected_pair_starts": pair_starts.detach(),
                    "similarity_min": (
                        similarities.min().detach()
                        if int(similarities.numel())
                        else None
                    ),
                    "similarity_max": (
                        similarities.max().detach()
                        if int(similarities.numel())
                        else None
                    ),
                }
            )
    return _torch().cat(merged_chunks, dim=0), output_lengths, reports



class AdjacentMergingEncoderStrategy(AudioLayerTokenStrategyBase):
    token_operation = "merge"
    encoder_reference_kind = "key_vectors"
    metadata_fields = (
        "merge_ratio",
        "pairing",
        "merge_value",
        "layer_interval",
    )

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]:
        result = super().runtime_metadata(active_layers)
        result["merge_ratio_per_layer"] = self.spec.merge_ratio
        return result

    def enabled(self, layer_index: int, active_layers: tuple[int, ...]) -> bool:
        return layer_index in active_layers and self.spec.merge_ratio > 0

    def output_lengths(
        self,
        layer_index: int,
        chunk_lengths: list[int],
        active_layers: tuple[int, ...],
    ) -> list[int]:
        del layer_index, active_layers
        return [
            adjacent_merge_output_length(length, self.spec.merge_ratio)
            for length in chunk_lengths
        ]

    def compress(
        self,
        hidden_states: Any,
        probe: Any,
        chunk_lengths: list[int],
        *,
        layer_index: int,
        active_layers: tuple[int, ...],
        collect_details: bool,
    ) -> tuple[Any, list[int], list[dict[str, Any]], Any]:
        del layer_index, active_layers
        decisions: list[Any] = []
        compressed, lengths, reports = merge_ragged_adjacent_hidden_states(
            hidden_states,
            probe,
            chunk_lengths,
            merge_ratio=self.spec.merge_ratio,
            collect_details=collect_details,
            replay_decisions=decisions, scorer=self.rules.get("scorer"),
            grouper=self.rules.get("grouper"), aggregator=self.rules.get("aggregator"),
        )
        return compressed, lengths, reports, (tuple(chunk_lengths), tuple(decisions))

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        lengths, decisions = plan
        return _torch().cat([
            replay_adjacent_pairs(chunk, pairs, aggregator=self.rules.get("aggregator"))
            for chunk, pairs in zip(hidden_states.split(lengths), decisions)
        ], dim=0)

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        indices = _torch().arange(length, device=device)
        keep = _torch().ones(length, dtype=_torch().bool, device=device)
        offset = 0
        for count, starts in zip(*plan):
            keep[starts.to(device) + offset + 1] = False
            offset += count
        return indices[keep]



class AdjacentTokenMergingAudio:
    """Encoder-level plan provider for paper-default audio A-ToMe."""

    name = "adjacent_token_merging_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        AdjacentTokenMergingSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        spec = AdjacentTokenMergingSpec.from_config(config)
        defer_token_budget = bool(context.extra.get("defer_token_budget", False))
        audio_encoder = context.structural_handles.get("audio_encoder")
        layers = list(getattr(audio_encoder, "layers", ()))
        n_window = getattr(audio_encoder, "n_window", None)
        if not layers or n_window is None:
            raise InvariantError(
                "A-ToMe planning requires audio_encoder.layers and audio_encoder.n_window"
            )
        active_layers = spec.resolve_layer_indices(len(layers))
        feature_lengths = _integer_sequence(
            state.audio_feature_lengths,
            "state.audio_feature_lengths",
        )
        audio_spans = [
            span for span in state.modality_token_spans if span.modality == "audio"
        ]
        if len(feature_lengths) != len(audio_spans):
            raise InvariantError(
                "A-ToMe audio feature lengths do not match serialized audio spans: "
                f"features={len(feature_lengths)}, spans={len(audio_spans)}"
            )

        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        initial_encoder_total = 0
        final_encoder_total = 0
        for audio_unit_index, (span, feature_length) in enumerate(
            zip(audio_spans, feature_lengths)
        ):
            initial_chunks, final_chunks = qwen_planned_a_tome_lengths(
                feature_length,
                n_window=int(n_window),
                merge_ratio=spec.merge_ratio,
                active_layer_count=len(active_layers),
            )
            initial_encoder_count = sum(initial_chunks)
            final_encoder_count = sum(final_chunks)
            original_count = span.end - span.start
            expected_original_count = initial_encoder_count // 2
            if not defer_token_budget and original_count != expected_original_count:
                raise InvariantError(
                    "A-ToMe raw audio length does not reproduce the processor placeholder "
                    f"count for audio unit {audio_unit_index}: span={original_count}, "
                    f"planned={expected_original_count}"
                )
            keep_count = original_count if defer_token_budget else final_encoder_count // 2
            if keep_count < 1:
                raise InvariantError(
                    "A-ToMe would leave no post-pool audio token for audio unit "
                    f"{audio_unit_index}"
                )
            original_total += original_count
            planned_total += keep_count
            initial_encoder_total += initial_encoder_count
            final_encoder_total += final_encoder_count
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
                        "feature_length": feature_length,
                        "initial_chunk_lengths": initial_chunks,
                        "final_chunk_lengths": final_chunks,
                        "initial_encoder_count": initial_encoder_count,
                        "final_encoder_count": final_encoder_count,
                        "original_count": original_count,
                        "keep_count": keep_count,
                        "merge_ratio": spec.merge_ratio,
                        "active_layers": list(active_layers),
                        "token_order": "chronological",
                    },
                )
            )

        stage_name = str(config.get("_stage_name") or self.name)
        metadata = StageMetadata(
            stage_name=stage_name,
            changed_spans=audio_spans,
            stats={
                "target_site": spec.target_site,
                "score_source": spec.score_source,
                "merge_ratio_per_layer": spec.merge_ratio,
                "active_layers": list(active_layers),
                "num_audio_units": len(actions),
                "original_audio_tokens": original_total,
                "planned_audio_tokens": planned_total,
                "initial_encoder_tokens": initial_encoder_total,
                "planned_encoder_tokens": final_encoder_total,
                "selection_deferred_to_runtime": True,
            },
        )
        execution_phase = ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        )
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.ENCODER_LEVEL,
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
            raise ValueError("AdjacentTokenMergingAudio has no trainable state")


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


def _integer_sequence(value: Any, name: str) -> list[int]:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise InvariantError(f"{name} must be a sequence")
    return [int(item) for item in raw]
