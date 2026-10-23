"""FastAdaSP weighted audio-token merging for decoder prefill.

The paper method has two decisions: adjacent audio keys choose merge
boundaries, while cumulative full-prompt attention supplies the weighted mean
inside every resulting temporal cluster.  Qwen-specific decoder execution and
layout bookkeeping remain in the backbone controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.llm_level._rotary import apply_rotary_position_embeddings
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.compressors.audio_rule_utils import audio_rule_metadata
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


_TARGET_SITE = "llm_decoder_layer_post_attention_audio_tokens"
_SCORE_SOURCE = "full_prompt_received_attention"
_SIMILARITY_SOURCE = "rotated_decoder_key_cosine"
_MERGE_VALUE = "attention_weighted_mean"


@dataclass(frozen=True)
class FastAdaSPScores:
    """Per-token weights and per-adjacent-pair similarity for one layer."""

    merge_weights: Any
    adjacent_similarity: Any


@dataclass(frozen=True)
class FastAdaSPMergeReplayPlan:
    """Detached decisions needed to replay one weighted audio merge."""

    clusters_relative: tuple[tuple[int, ...], ...]
    normalized_weights: Any


@dataclass(frozen=True)
class FastAdaSPSpec:
    """Normalized FastAdaSP inference configuration."""

    merge_ratio: float = 0.5
    task_mode: str = "sparse"
    operation_layer: int = 3
    minimum_keep_tokens: int = 1
    score_source: str = _SCORE_SOURCE
    similarity_source: str = _SIMILARITY_SOURCE
    merge_value: str = _MERGE_VALUE
    target_site: str = _TARGET_SITE
    lifecycle: str = "prefill_only"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "FastAdaSPSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        activation_scope = _mapping(
            config.get("activation_scope"), "activation_scope"
        )
        legacy = _mapping(config.get("config"), "config")

        raw_layers = activation_scope.get("layers")
        if raw_layers is None:
            operation_layer = _strict_int(
                _first_present(legacy.get("perform_layer"), cls.operation_layer),
                "operation_layer",
            )
        else:
            if (
                not isinstance(raw_layers, (list, tuple))
                or len(raw_layers) != 1
            ):
                raise ValueError(
                    "FastAdaSP activation_scope.layers must contain exactly one "
                    "zero-based operation/start layer."
                )
            operation_layer = _strict_int(raw_layers[0], "operation_layer")

        task_mode = str(
            _first_present(
                schedule.get("task_mode"),
                legacy.get("task_mode"),
                legacy.get("schedule"),
                cls.task_mode,
            )
        ).lower()
        # The released code calls the one-layer sparse schedule "none".
        if task_mode == "none":
            task_mode = "sparse"

        spec = cls(
            merge_ratio=float(
                _first_present(
                    schedule.get("merge_ratio"),
                    legacy.get("ratio"),
                    cls.merge_ratio,
                )
            ),
            task_mode=task_mode,
            operation_layer=operation_layer,
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
            similarity_source=str(
                _first_present(
                    criterion.get("similarity_source"),
                    legacy.get("similarity_source"),
                    cls.similarity_source,
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
            lifecycle=str(
                _first_present(
                    activation_scope.get("lifecycle"),
                    legacy.get("lifecycle"),
                    cls.lifecycle,
                )
            ),
        )
        spec.validate()
        if operation.get("type") not in {None, "merge"}:
            raise ValueError("FastAdaSP requires operation.type=merge.")
        return spec

    def validate(self) -> None:
        if not math.isfinite(self.merge_ratio) or not 0 <= self.merge_ratio < 1:
            raise ValueError("FastAdaSP merge_ratio must be in [0, 1).")
        if self.task_mode not in {"sparse", "constant", "decay"}:
            raise ValueError(
                "FastAdaSP task_mode must be sparse, constant, or decay."
            )
        if self.operation_layer < 0:
            raise ValueError("FastAdaSP operation layer must be non-negative.")
        if self.minimum_keep_tokens < 1:
            raise ValueError("FastAdaSP minimum_keep_tokens must be positive.")
        if self.score_source != _SCORE_SOURCE:
            raise ValueError(
                f"FastAdaSP requires score_source={_SCORE_SOURCE!r}."
            )
        if self.similarity_source != _SIMILARITY_SOURCE:
            raise ValueError(
                "FastAdaSP requires similarity_source="
                f"{_SIMILARITY_SOURCE!r}."
            )
        if self.merge_value != _MERGE_VALUE:
            raise ValueError(
                f"FastAdaSP requires merge_value={_MERGE_VALUE!r}."
            )
        if self.lifecycle != "prefill_only":
            raise ValueError("FastAdaSP lifecycle must be 'prefill_only'.")

    def layer_indices(self, num_layers: int) -> tuple[int, ...]:
        if self.operation_layer >= num_layers:
            raise InvariantError(
                "FastAdaSP operation layer is outside the decoder: "
                f"layer={self.operation_layer}, num_layers={num_layers}."
            )
        if self.task_mode == "sparse" or self.operation_layer == num_layers - 1:
            return (self.operation_layer,)
        # The decay ratio is exactly zero at the final layer, so no operation is
        # registered there.  Constant scheduling continues through the last layer.
        stop = num_layers if self.task_mode == "constant" else num_layers - 1
        return tuple(range(self.operation_layer, stop))

    def ratio_at(self, layer_index: int, num_layers: int) -> float:
        if self.task_mode != "decay" or layer_index == self.operation_layer:
            return self.merge_ratio
        denominator = num_layers - 1 - self.operation_layer
        if denominator <= 0:
            return self.merge_ratio
        return self.merge_ratio * (
            1.0 - (layer_index - self.operation_layer) / denominator
        )


def fastadasp_output_length(
    current_length: int,
    merge_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the official floor-based adjacent-merge budget."""

    if current_length < 0:
        raise ValueError("current_length must be non-negative")
    if current_length == 0:
        return 0
    if not math.isfinite(merge_ratio) or not 0 <= merge_ratio < 1:
        raise ValueError("merge_ratio must be in [0, 1)")
    if minimum_keep_tokens < 1:
        raise ValueError("minimum_keep_tokens must be positive")
    merge_count = int(current_length * merge_ratio)
    return min(
        current_length,
        max(minimum_keep_tokens, current_length - merge_count),
    )


def weighted_adjacent_token_merge(
    hidden_states: Any,
    adjacent_similarity: Any,
    merge_weights: Any,
    *,
    output_length: int,
    scorer: Any = None, grouper: Any = None, aggregator: Any = None,
) -> tuple[Any, Any, dict[str, Any]]:
    """Implement Algorithm 1 over one chronological audio sequence."""

    torch = _torch()
    if hidden_states.ndim != 2:
        raise ValueError("hidden_states must have shape [tokens, hidden_size]")
    token_count = int(hidden_states.shape[0])
    if adjacent_similarity.ndim != 1 or int(adjacent_similarity.numel()) != max(
        token_count - 1, 0
    ):
        raise ValueError("adjacent_similarity must contain one value per token pair")
    if merge_weights.ndim != 1 or int(merge_weights.numel()) != token_count:
        raise ValueError("merge_weights must contain one value per token")
    if not 1 <= output_length <= token_count:
        raise ValueError("output_length must be in [1, token_count]")
    if not bool(torch.isfinite(adjacent_similarity).all().item()):
        raise InvariantError("FastAdaSP adjacent similarity contains NaN or Inf.")
    if not bool(torch.isfinite(merge_weights).all().item()):
        raise InvariantError("FastAdaSP merge weights contain NaN or Inf.")

    from mllm_comp.compressors.audio_rule_utils import token_scores, contiguous_groups
    merge_count = token_count - output_length
    if scorer is not None and merge_count:
        custom = scorer({"values": hidden_states.detach(), "adjacent_similarity": adjacent_similarity,
            "merge_weights": merge_weights, "keep_count": output_length})
        if not isinstance(custom, dict) or set(custom) != {"adjacent_similarity", "merge_weights"}:
            raise InvariantError("FastAdaSP scorer must return adjacent_similarity and merge_weights.")
        adjacent_similarity = token_scores(custom["adjacent_similarity"], hidden_states, length=token_count - 1)
        merge_weights = token_scores(custom["merge_weights"], hidden_states, length=token_count, nonnegative=True)
    if merge_count:
        try:
            ranking = torch.argsort(
                adjacent_similarity, descending=True, stable=True
            )
        except TypeError:  # pragma: no cover - older supported torch builds
            ranking = torch.argsort(adjacent_similarity, descending=True)
        boundaries = torch.sort(ranking[:merge_count])[0].to(dtype=torch.long)
        boundary_set = {int(item) for item in boundaries.detach().cpu().tolist()}
    else:
        boundaries = torch.empty(
            (0,), dtype=torch.long, device=hidden_states.device
        )
        boundary_set = set()

    clusters: list[tuple[int, ...]] = []
    index = 0
    while index < token_count:
        start = index
        while index < token_count - 1 and index in boundary_set:
            index += 1
        clusters.append(tuple(range(start, index + 1)))
        index += 1

    if grouper is not None and merge_count:
        clusters = contiguous_groups(grouper({"values": hidden_states.detach(),
            "adjacent_similarity": adjacent_similarity, "merge_weights": merge_weights,
            "keep_count": output_length}), length=token_count, count=output_length)
        boundaries = torch.tensor([i for cluster in clusters for i in cluster[:-1]],
            device=hidden_states.device, dtype=torch.long)
    normalized_parts: list[Any] = []
    for cluster in clusters:
        cluster_weights = merge_weights[list(cluster)]
        tiny = torch.finfo(cluster_weights.dtype).tiny
        normalized_parts.append(
            cluster_weights / cluster_weights.sum().clamp_min(tiny)
        )
    normalized_weights = torch.cat(normalized_parts, dim=0)
    live_plan = FastAdaSPMergeReplayPlan(
        clusters_relative=tuple(clusters),
        normalized_weights=normalized_weights,
    )
    result = _apply_weighted_merge_plan(hidden_states, live_plan, aggregator=aggregator if merge_count else None)
    representative_indices = [cluster[0] for cluster in clusters]
    representatives = torch.tensor(
        representative_indices,
        dtype=torch.long,
        device=hidden_states.device,
    )
    if int(result.shape[0]) != output_length:
        raise InvariantError(
            f"FastAdaSP produced {int(result.shape[0])} tokens, expected "
            f"{output_length}."
        )
    return result, representatives, {
        "merge_count": merge_count,
        "merge_boundaries_relative": boundaries,
        "clusters_relative": [list(cluster) for cluster in clusters],
        "merge_value": _MERGE_VALUE if aggregator is None else "custom",
        "_checkpoint_replay_plan": FastAdaSPMergeReplayPlan(
            clusters_relative=tuple(clusters),
            normalized_weights=normalized_weights.detach().cpu().clone(),
        ),
    }


def _apply_weighted_merge_plan(
    hidden_states: Any,
    replay_plan: FastAdaSPMergeReplayPlan,
    *, aggregator: Any = None,
) -> Any:
    torch = _torch()
    token_count = int(hidden_states.shape[0])
    flattened = tuple(
        item for cluster in replay_plan.clusters_relative for item in cluster
    )
    if flattened != tuple(range(token_count)):
        raise InvariantError(
            "FastAdaSP replay clusters must partition the chronological audio unit."
        )
    weights = replay_plan.normalized_weights.to(device=hidden_states.device)
    if weights.ndim != 1 or int(weights.numel()) != token_count:
        raise InvariantError(
            "FastAdaSP normalized replay weights must contain one value per "
            "audio token."
        )
    if not bool(torch.isfinite(weights).all().item()):
        raise InvariantError("FastAdaSP replay weights contain NaN or Inf.")

    merged: list[Any] = []
    for cluster in replay_plan.clusters_relative:
        cluster_indices = list(cluster)
        cluster_weights = weights[cluster_indices]
        if not bool(
            torch.isclose(
                cluster_weights.sum(),
                cluster_weights.new_tensor(1.0),
                rtol=1e-4,
                atol=1e-6,
            ).item()
        ):
            raise InvariantError(
                "FastAdaSP replay weights must be normalized within each cluster."
            )
        if aggregator is not None and len(replay_plan.clusters_relative) != token_count:
            from mllm_comp.compressors.audio_rule_utils import validate_values
            output = aggregator({"values": hidden_states[cluster_indices], "indices": tuple(cluster),
                "group_index": len(merged), "weights": cluster_weights})
            validate_values(output, hidden_states[cluster_indices[0]])
            merged.append(output)
        else:
            merged.append((hidden_states[cluster_indices] * cluster_weights.unsqueeze(-1)).sum(dim=0))
    return torch.stack(merged, dim=0).to(dtype=hidden_states.dtype)


class FastAdaSPLLMStrategy:
    """Paper-exact scoring, scheduling, and weighted temporal merge policy."""

    compression_boundary = "post_attention_pre_ffn"
    requires_instruction_queries = False
    score_with_grad = False

    def __init__(self, spec: FastAdaSPSpec) -> None:
        self.spec = spec
        self._num_layers: int | None = None

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        layers = self.spec.layer_indices(num_layers)
        self._num_layers = num_layers
        return layers

    def output_length(
        self, current_length: int, *, layer_index: int | None = None
    ) -> int:
        if self._num_layers is None:
            raise InvariantError(
                "FastAdaSP stage layers must be resolved before output_length()."
            )
        active_layer = self.spec.operation_layer if layer_index is None else layer_index
        return fastadasp_output_length(
            current_length,
            self.spec.ratio_at(active_layer, self._num_layers),
            minimum_keep_tokens=self.spec.minimum_keep_tokens,
        )

    def runtime_metadata(self, stage_layers: tuple[int, ...]) -> dict[str, Any]:
        if self._num_layers is None:
            raise InvariantError("FastAdaSP decoder layer count was not resolved.")
        return {
            "target_site": self.spec.target_site,
            "score_source": self.spec.score_source,
            "similarity_source": self.spec.similarity_source,
            "merge_value": self.spec.merge_value,
            "task_mode": self.spec.task_mode,
            "base_merge_ratio": self.spec.merge_ratio,
            "operation_layer_zero_based": self.spec.operation_layer,
            "stage_layers_zero_based": list(stage_layers),
            "stage_merge_ratios": [
                self.spec.ratio_at(item, self._num_layers) for item in stage_layers
            ],
            "minimum_keep_tokens": self.spec.minimum_keep_tokens,
            "compression_boundary": self.compression_boundary,
            "lifecycle": self.spec.lifecycle,
            "prefill_compression": True,
            "decode_compression": False,
            "position_policy": "preserve_absolute_cluster_start",
            "training_support": "fixed_merge_plan_checkpoint_replay",
            **audio_rule_metadata(self),
        }

    def score_audio_tokens(
        self,
        *,
        attention_module: Any,
        normalized_hidden_states: Any,
        position_embeddings: Any | None,
        absolute_positions: Any,
        instruction_mask: Any,
        valid_mask: Any,
        audio_mask: Any,
        layer_index: int | None = None,
    ) -> FastAdaSPScores:
        """Recompute official full-prompt attention and rotated decoder keys."""

        del instruction_mask, audio_mask, layer_index
        torch = _torch()
        hidden = normalized_hidden_states
        batch_size, sequence_length, _ = hidden.shape
        config = attention_module.config
        num_heads = int(config.num_attention_heads)
        num_key_value_heads = int(config.num_key_value_heads)
        head_dim = int(getattr(attention_module, "head_dim", 0))
        if head_dim <= 0:
            head_dim = int(attention_module.q_proj.out_features // num_heads)

        query = attention_module.q_proj(hidden).view(
            batch_size, sequence_length, num_heads, head_dim
        ).transpose(1, 2)
        key = attention_module.k_proj(hidden).view(
            batch_size, sequence_length, num_key_value_heads, head_dim
        ).transpose(1, 2)
        if position_embeddings is None:
            raise InvariantError(
                "FastAdaSP requires the pinned Qwen decoder's native M-RoPE "
                "embeddings so its Key similarity matches the attention path."
            )
        query, key = apply_rotary_position_embeddings(
            query,
            key,
            position_embeddings=position_embeddings,
            attention_module=attention_module,
            batch_size=batch_size,
            sequence_length=sequence_length,
        )
        repeated_key = key
        groups = num_heads // num_key_value_heads
        if groups > 1:
            repeated_key = key.repeat_interleave(groups, dim=1)
        scale = float(getattr(attention_module, "scaling", head_dim**-0.5))

        merge_weights = hidden.new_zeros(
            (batch_size, sequence_length), dtype=torch.float32
        )
        adjacent_similarity = hidden.new_full(
            (batch_size, max(sequence_length - 1, 0)),
            float("-inf"),
            dtype=torch.float32,
        )
        for batch_index in range(batch_size):
            slots = torch.nonzero(valid_mask[batch_index], as_tuple=False).flatten()
            if int(slots.numel()) == 0:
                continue
            q = query[batch_index, :, slots, :]
            k = repeated_key[batch_index]
            query_absolute = absolute_positions[batch_index, slots]
            key_absolute = absolute_positions[batch_index]
            causal = key_absolute.unsqueeze(0) <= query_absolute.unsqueeze(1)
            allowed = causal & valid_mask[batch_index].unsqueeze(0)
            with torch.autocast(device_type=q.device.type, enabled=False):
                logits = torch.matmul(
                    q.float(), k.float().transpose(-2, -1)
                ) * scale
                logits = logits.masked_fill(
                    ~allowed.unsqueeze(0), torch.finfo(logits.dtype).min
                )
                attention = torch.softmax(logits, dim=-1, dtype=torch.float32)
            merge_weights[batch_index] = attention.sum(dim=(0, 1))

            if sequence_length > 1:
                flattened_key = key[batch_index].transpose(0, 1).reshape(
                    sequence_length, -1
                )
                pair_valid = valid_mask[batch_index, :-1] & valid_mask[
                    batch_index, 1:
                ]
                similarity = torch.nn.functional.cosine_similarity(
                    flattened_key[:-1].float(),
                    flattened_key[1:].float(),
                    dim=-1,
                )
                adjacent_similarity[batch_index, pair_valid] = similarity[pair_valid]

        if not bool(torch.isfinite(merge_weights[valid_mask]).all().item()):
            raise InvariantError("FastAdaSP merge weights contain NaN or Inf.")
        return FastAdaSPScores(
            merge_weights=merge_weights,
            adjacent_similarity=adjacent_similarity,
        )

    def merge_audio_tokens(
        self,
        *,
        hidden_states: Any,
        scores: FastAdaSPScores,
        sample_index: int,
        slots: Any,
        output_length: int,
        audio_unit_index: int = 0,
    ) -> tuple[Any, Any, dict[str, Any]]:
        del audio_unit_index
        if int(slots.numel()) > 1 and not bool(
            (slots[1:] == slots[:-1] + 1).all().item()
        ):
            raise InvariantError(
                "FastAdaSP requires each audio unit to remain physically contiguous."
            )
        adjacent = scores.adjacent_similarity[sample_index, slots[:-1]]
        weights = scores.merge_weights[sample_index, slots]
        return weighted_adjacent_token_merge(
            hidden_states,
            adjacent,
            weights,
            output_length=output_length, **getattr(self, "rules", {}),
        )

    def replay_merge_audio_tokens(
        self,
        *,
        hidden_states: Any,
        replay_plan: Any,
    ) -> Any:
        if not isinstance(replay_plan, FastAdaSPMergeReplayPlan):
            raise InvariantError("FastAdaSP received an invalid merge replay plan.")
        return _apply_weighted_merge_plan(hidden_states, replay_plan, aggregator=getattr(self, "rules", {}).get("aggregator"))

class FastAdaSP:
    name = "fastadasp_weighted_token_merging"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        FastAdaSPSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        spec = FastAdaSPSpec.from_config(config)
        llm = context.structural_handles.get("llm")
        num_layers = len(tuple(getattr(llm, "layers", ()) or ()))
        if num_layers < 1:
            raise InvariantError("FastAdaSP requires resolved LLM decoder layers.")
        stage_layers = spec.layer_indices(num_layers)
        actions: list[TokenAction] = []
        original_total = 0
        final_total = 0
        unit_index = 0
        for span in state.modality_token_spans:
            if span.modality != "audio":
                continue
            length = span.end - span.start
            stage_lengths = [length]
            for layer_index in stage_layers:
                stage_lengths.append(
                    fastadasp_output_length(
                        stage_lengths[-1],
                        spec.ratio_at(layer_index, num_layers),
                        minimum_keep_tokens=spec.minimum_keep_tokens,
                    )
                )
            original_total += length
            final_total += stage_lengths[-1]
            actions.append(
                TokenAction(
                    op="merge",
                    target_indices=None,
                    metadata={
                        "selection_deferred": True,
                        "selector": self.name,
                        "target_site": spec.target_site,
                        "sample_index": span.sample_index,
                        "audio_unit_index": unit_index,
                        "span_start": span.start,
                        "span_end": span.end,
                        "original_count": length,
                        "keep_count": stage_lengths[-1],
                        "stage_lengths": stage_lengths,
                        "stage_layers": list(stage_layers),
                        "task_mode": spec.task_mode,
                        "lifecycle": spec.lifecycle,
                        "token_order": "chronological",
                    },
                )
            )
            unit_index += 1

        stage_name = str(config.get("_stage_name") or self.name)
        execution_phase = ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        )
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.LLM_LEVEL,
            execution_phase=execution_phase,
            modalities={Modality.AUDIO},
            token_roles={TokenRole.OBSERVATION},
            method=CompressionMethod.TOKEN_COMPRESSION,
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype=spec.target_site,
            token_actions=actions,
            legalization_policy=LegalizationPolicy(
                position_policy="preserve_absolute",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
                coordinate_policy="drop_removed",
            ),
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                changed_spans=[
                    span
                    for span in state.modality_token_spans
                    if span.modality == "audio"
                ],
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "similarity_source": spec.similarity_source,
                    "merge_value": spec.merge_value,
                    "task_mode": spec.task_mode,
                    "lifecycle": spec.lifecycle,
                    "stage_layers": list(stage_layers),
                    "original_audio_tokens": original_total,
                    "planned_final_audio_tokens": final_total,
                    "selection_deferred_to_prefill": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("FastAdaSP has no trainable state")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InvariantError("FastAdaSP requires PyTorch.") from exc
    return torch


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_present(*values: Any) -> Any:
    return next((item for item in values if item is not None), None)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


__all__ = [
    "FastAdaSP",
    "FastAdaSPLLMStrategy",
    "FastAdaSPMergeReplayPlan",
    "FastAdaSPScores",
    "FastAdaSPSpec",
    "fastadasp_output_length",
    "weighted_adjacent_token_merge",
]
