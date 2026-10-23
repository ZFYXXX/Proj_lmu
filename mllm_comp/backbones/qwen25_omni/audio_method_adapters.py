"""Qwen2.5-Omni adapters for audio token methods.

Model geometry, native attention probes and placement adapters remain here.
Model-independent compression and replay live beside each method definition.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mllm_comp.compressors.encoder_level.adjacent_token_merging import (
    _integer_sequence,
    AdjacentMergingEncoderStrategy,
    qwen_audio_chunk_lengths_after_cnn,
)
from mllm_comp.compressors.encoder_level.attention_topk import (
    AttentionTopKEncoderStrategy,
)
from mllm_comp.compressors.encoder_level.progressive_attention_pruning import (
    ProgressivePruningEncoderStrategy,
)
from mllm_comp.compressors.projector_level.fastslm_hta import FastSLMHTAProjectorStrategy
from mllm_comp.compressors.audio_rule_utils import _torch
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.vocabulary import Placement


class QwenAdjacentMergingEncoderStrategy(AdjacentMergingEncoderStrategy):
    """Supply the native Qwen audio probe for the shared rule."""

    def probe(self, layer: Any, normalized: Any, chunk_lengths: list[int]) -> Any:
        del chunk_lengths
        with _torch().no_grad():
            return layer.self_attn.k_proj(normalized)


class QwenAttentionTopKEncoderStrategy(AttentionTopKEncoderStrategy):
    """Supply the native Qwen audio probe for the shared rule."""

    def probe(self, layer: Any, normalized: Any, chunk_lengths: list[int]) -> Any:
        from .encoder_compression import mean_received_attention_scores

        return mean_received_attention_scores(layer.self_attn, normalized, chunk_lengths)


class QwenProgressivePruningEncoderStrategy(ProgressivePruningEncoderStrategy):
    """Supply the native Qwen audio probe for the shared rule."""

    def probe(self, layer: Any, normalized: Any, chunk_lengths: list[int]) -> Any:
        from .encoder_compression import mean_received_attention_scores

        return mean_received_attention_scores(layer.self_attn, normalized, chunk_lengths)


class QwenFastSLMHTAProjectorStrategy(FastSLMHTAProjectorStrategy):
    """Resolve Qwen geometry and execution placement for the shared HTA rule."""

    def build_runtime_module(self, context: Mapping[str, Any]) -> Any:
        if self._runtime_module is not None:
            return self._runtime_module
        audio_encoder = context.get("audio_encoder")
        audio_projector = context.get("audio_projector")
        hidden_size = context.get("compression_hidden_size", getattr(audio_projector, "in_features", None))
        encoder_config = getattr(audio_encoder, "config", None)
        if hidden_size is None:
            hidden_size = getattr(encoder_config, "d_model", None)
        if hidden_size is None:
            raise InvariantError(
                "FastSLM HTA could not resolve the audio projector input width."
            )
        configured_heads = int(self.spec.num_attention_heads)
        num_attention_heads = configured_heads or int(
            getattr(encoder_config, "encoder_attention_heads", 0)
        )
        encoder_width = getattr(encoder_config, "d_model", None)
        if not configured_heads and num_attention_heads > 0 and encoder_width:
            # Automatic HTA geometry follows the selected feature width while
            # retaining the encoder's per-head width where it is representable.
            head_width = int(encoder_width) // num_attention_heads
            if int(hidden_size) != int(encoder_width) and head_width > 0:
                num_attention_heads = (
                    int(hidden_size) // head_width if int(hidden_size) % head_width == 0 else 1
                )
        if num_attention_heads < 1:
            if int(hidden_size) % 64 == 0:
                num_attention_heads = int(hidden_size) // 64
            else:
                num_attention_heads = 1
        return super().build_runtime_module(
            hidden_size=int(hidden_size),
            num_attention_heads=num_attention_heads,
            module_name=context.get("runtime_module_name", "_mllm_comp_fastslm_hta_audio"),
        )

    @property
    def runtime_device(self) -> Any:
        from .audio_compression import _module_execution_device

        return _module_execution_device(self._runtime_module)

    def runtime_metadata(self) -> dict[str, Any]:
        result = super().runtime_metadata()
        if self.spec.target_site in {"audio_encoder_output_pre_projector", "audio_projector_output_post_projector"}:
            result["qwen_rate_adaptation"] = (
                "25hz_postprojector_to_1.67hz"
                if self.spec.target_site == "audio_projector_output_post_projector"
                else "25hz_preprojector_to_1.67hz"
            )
        else:
            result["window_units"] = "tokens_at_application_boundary"
        return result


def _temporal_replay_plan(rule: Any, output_length: int, details: dict[str, Any], indices: Any, *, scores: Any = None) -> Any:
    factory = getattr(rule, "replay_plan", None)
    return factory(output_length, details, indices, scores=scores) if callable(factory) else output_length


def _temporal_replay(rule: Any, hidden_states: Any, plan: Any) -> Any:
    replay = getattr(rule, "replay", None)
    if callable(replay):
        return replay(hidden_states, plan)
    return rule.compress(hidden_states, scores=None, keep_count=plan, item_index=0)[0]


def _adapt_audio_stage_plan(plan: Any, layers: tuple[int, ...], output_length: Any, *, placement: Placement = Placement.LLM_LEVEL) -> Any:
    is_llm = placement is Placement.LLM_LEVEL
    lifecycle = "prefill_only" if is_llm else "native"
    actions = []
    for action in plan.token_actions:
        lengths = [int(action.metadata["original_count"])]
        for layer in layers:
            lengths.append(None if lengths[-1] is None else output_length(lengths[-1], layer))
        actions.append(replace(action, metadata={
            **action.metadata, "keep_count": lengths[-1], "stage_lengths": lengths,
            "stage_layers": list(layers), "lifecycle": lifecycle,
        }))
    counts = [action.metadata["keep_count"] for action in actions]
    final_count = None if None in counts else sum(counts)
    return replace(
        plan, placement=placement, token_actions=actions,
        legalization_policy=replace(plan.legalization_policy, position_policy="preserve_absolute" if is_llm else "compact"),
        this_stage_metadata=replace(plan.this_stage_metadata, stats={
            **plan.this_stage_metadata.stats, "planned_audio_tokens": final_count,
            "planned_final_audio_tokens": final_count, "stage_layers": list(layers),
            "lifecycle": lifecycle, **({"selection_deferred_to_prefill": True} if is_llm else {}),
        }),
    )


class TemporalResamplingLLMStrategy:
    """Use the existing deterministic temporal rule on each LLM audio unit."""

    requires_layer_input = False
    requires_instruction_queries = False
    query_source = "none"
    answer_query_leakage = False

    def __init__(self, rule: Any, layers: tuple[int, ...]) -> None:
        self.rule = rule
        self.spec = rule.spec
        self.layers = layers
        self.token_operation = rule.token_operation
        self.encoder_attention_layer_offset = rule.attention_layer_offset
        self.uses_distinct_merge_prune_importance = rule.uses_distinct_merge_prune_importance
        self.merge_position_policy = getattr(rule, "merge_position_policy", "preserve_absolute_cluster_start")

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        # Actual edge and model-depth validation are shared by the controller.
        return self.layers

    def output_length(self, current_length: int) -> int | None:
        return self.rule.output_length(current_length)

    def runtime_metadata(self, stage_layers: tuple[int, ...]) -> dict[str, Any]:
        return {**self.rule.runtime_metadata(), "token_operation": self.token_operation,
                "score_source": getattr(self.rule, "layer_score_source", self.spec.score_source)}

    def adapt_plan(self, plan: Any) -> Any:
        return _adapt_audio_stage_plan(plan, self.layers, lambda length, layer: self.output_length(length))

    def merge_audio_tokens(
        self, *, hidden_states: Any, scores: Any, sample_index: int,
        slots: Any, output_length: int | None, audio_unit_index: int = 0,
    ) -> tuple[Any, Any, dict[str, Any]]:
        unit_scores = None if scores is None else scores[sample_index, slots]
        output, indices, details = self.rule.compress(
            hidden_states, scores=unit_scores,
            keep_count=output_length, item_index=audio_unit_index,
        )
        plan = _temporal_replay_plan(self.rule, int(output.shape[0]), details, indices, scores=unit_scores)
        if indices is None:
            representative_indices = getattr(self.rule, "representative_indices", None)
            if callable(representative_indices):
                indices = representative_indices(int(hidden_states.shape[0]), plan, hidden_states.device)
            else:
                # Uniform pooling uses each window's first token for absolute RoPE.
                indices = _torch().arange(output_length, device=hidden_states.device) * self.spec.compression_factor
        return output, indices, {**details, "_checkpoint_replay_plan": plan}

    def replay_merge_audio_tokens(self, *, hidden_states: Any, replay_plan: Any) -> Any:
        return _temporal_replay(self.rule, hidden_states, replay_plan)


class TemporalResamplingEncoderStrategy:
    """Apply the same temporal rule within each native encoder chunk."""

    def __init__(self, rule: Any, layers: tuple[int, ...]) -> None:
        self.rule = rule
        self.spec = rule.spec
        self.layers = layers
        self.requires_layer_input = rule.attention_layer_offset is not None

    def score_layer_index(self, rule_layer: int, num_layers: int) -> int:
        index = num_layers + self.rule.attention_layer_offset
        if index < 0:
            raise InvariantError("The audio encoder has no layer for the rule's fixed attention source.")
        return index

    def probe(self, layer: Any, normalized: Any, chunk_lengths: list[int]) -> Any:
        from .encoder_compression import mean_received_attention_scores

        if self.rule.uses_distinct_merge_prune_importance:
            from .audio_compression import sa_map_audio_importance

            scores = sa_map_audio_importance(layer.self_attn, normalized, chunk_lengths, chunk_lengths, pooled=False)
            return _torch().stack((scores["merge"], scores["prune"]), dim=-1)
        return mean_received_attention_scores(layer.self_attn, normalized, chunk_lengths)

    def active_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        return self.layers

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]:
        return {**self.rule.runtime_metadata(), "active_layers": list(active_layers),
                "score_source": getattr(self.rule, "layer_score_source", self.spec.score_source)}

    def enabled(self, layer_index: int, active_layers: tuple[int, ...]) -> bool:
        # Allocation/rounding can reduce a chunk even when keep_ratio == 1.
        return layer_index in active_layers

    def output_lengths(self, layer_index: int, chunk_lengths: list[int], active_layers: tuple[int, ...]) -> list[int | None]:
        return [self.rule.output_length(length) for length in chunk_lengths]

    def adapt_plan(self, plan: Any, *, state: Any, audio_tower: Any) -> Any:
        from mllm_comp.core.vocabulary import Placement

        feature_lengths = _integer_sequence(state.audio_feature_lengths, "state.audio_feature_lengths")
        if len(feature_lengths) != len(plan.token_actions):
            raise InvariantError("Encoder feature lengths do not match audio spans.")
        actions = []
        for action, feature_length in zip(plan.token_actions, feature_lengths):
            initial = qwen_audio_chunk_lengths_after_cnn(feature_length, audio_tower.n_window)
            final = list(initial)
            for layer in self.layers:
                final = None if final is None else self.output_lengths(layer, final, self.layers)
                if final is not None and None in final:
                    final = None
            if int(action.metadata["original_count"]) != sum(initial) // 2:
                raise InvariantError("Encoder raw audio length does not reproduce the processor placeholder count.")
            keep_count = None if final is None else sum(final) // 2
            if keep_count is not None and keep_count < 1:
                raise InvariantError("Encoder resampling would leave no pooled audio token.")
            actions.append(replace(action, metadata={
                **action.metadata, "feature_length": feature_length,
                "initial_chunk_lengths": initial, "final_chunk_lengths": final,
                "initial_encoder_count": sum(initial), "final_encoder_count": None if final is None else sum(final),
                "keep_count": keep_count, "active_layers": list(self.layers),
            }))
        unresolved = any(action.metadata["keep_count"] is None for action in actions)
        return replace(
            plan, placement=Placement.ENCODER_LEVEL, token_actions=actions,
            this_stage_metadata=replace(plan.this_stage_metadata, stats={
                **plan.this_stage_metadata.stats, "active_layers": list(self.layers),
                "initial_encoder_tokens": sum(a.metadata["initial_encoder_count"] for a in actions),
                "planned_encoder_tokens": None if unresolved else sum(a.metadata["final_encoder_count"] for a in actions),
                "planned_audio_tokens": None if unresolved else sum(a.metadata["keep_count"] for a in actions),
            }),
        )

    def compress(
        self, hidden_states: Any, probe: Any, chunk_lengths: list[int], *,
        layer_index: int, active_layers: tuple[int, ...], collect_details: bool,
    ) -> tuple[Any, list[int], list[dict[str, Any]], Any]:
        lengths = self.output_lengths(layer_index, chunk_lengths, active_layers)
        chunks = []
        actual_lengths = []
        reports = []
        plans = []
        score_chunks = probe.split(chunk_lengths) if probe is not None else [None] * len(chunk_lengths)
        for index, (chunk, length) in enumerate(zip(hidden_states.split(chunk_lengths), lengths)):
            output, indices, details = self.rule.compress(chunk, scores=score_chunks[index], keep_count=length, item_index=index)
            chunks.append(output)
            actual_lengths.append(int(output.shape[0]))
            plans.append(_temporal_replay_plan(self.rule, actual_lengths[-1], details, indices, scores=score_chunks[index]))
            if collect_details:
                reports.append({"chunk_index": index, **details})
        return _torch().cat(chunks), actual_lengths, reports, (tuple(chunk_lengths), tuple(plans))

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        input_lengths, plans = plan
        return _torch().cat([
            _temporal_replay(self.rule, chunk, decision)
            for chunk, decision in zip(hidden_states.split(input_lengths), plans)
        ])

    def map_reference(self, hidden_states: Any, plan: Any) -> Any:
        transfer = getattr(self.rule, "map_layer_reference", None)
        if not callable(transfer):
            return self.replay(hidden_states, plan)
        input_lengths, decisions = plan
        return _torch().cat([
            transfer(chunk, decision)
            for chunk, decision in zip(hidden_states.split(input_lengths), decisions)
        ])


class EncoderAttentionTokenStrategy:
    """Apply original encoder selection or merging rules to pooled audio spans."""

    requires_layer_input = False
    requires_instruction_queries = False
    answer_query_leakage = False
    attention_layer_offset = None

    @staticmethod
    def validate_rule(rule: Any) -> None:
        """Require the rule operations consumed by pooled encoder references."""
        required = ("active_layer_indices", "output_lengths", "runtime_metadata",
                    "compress", "replay", "representative_indices")
        missing = [name for name in required if not callable(getattr(rule, name, None))]
        if not hasattr(getattr(rule, "spec", None), "score_source"):
            missing.append("spec.score_source")
        if missing:
            raise ValueError(f"Pooled encoder rules require: {', '.join(missing)}.")
        if getattr(rule, "token_operation", None) not in {"keep", "merge"}:
            raise ValueError("Pooled encoder rules require token_operation='keep' or 'merge'.")
        if getattr(rule, "encoder_reference_kind", None) not in {"key_vectors", "received_attention"}:
            raise ValueError("Pooled encoder rules require encoder_reference_kind='key_vectors' or 'received_attention'.")

    def __init__(self, rule: Any, placement: Placement) -> None:
        self.validate_rule(rule)
        self.rule = rule
        self.spec = rule.spec
        self.placement = placement
        self.token_operation = rule.token_operation
        self.encoder_reference_kind = rule.encoder_reference_kind
        self.query_source = "encoder_keys" if self.encoder_reference_kind == "key_vectors" else "encoder_self_attention"
        self.layers = ()

    @property
    def score_source(self) -> str:
        return self.spec.score_source

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        self.active_layers = tuple(self.rule.active_layer_indices(num_layers))
        self.layers = tuple(sorted(self.active_layers))
        return self.layers

    def output_length(self, current_length: int, layer_index: int) -> int:
        return self.rule.output_lengths(layer_index, [current_length], self.active_layers)[0]

    def adapt_plan(self, plan: Any) -> Any:
        plan = _adapt_audio_stage_plan(plan, self.layers, self.output_length, placement=self.placement)
        # These historical encoder counts do not describe pooled audio tokens.
        unused = {"initial_chunk_lengths", "final_chunk_lengths", "initial_encoder_count", "final_encoder_count", "active_layers"}
        actions = [replace(a, metadata={k: v for k, v in a.metadata.items() if k not in unused}) for a in plan.token_actions]
        stats = {k: v for k, v in plan.this_stage_metadata.stats.items() if k not in {"initial_encoder_tokens", "planned_encoder_tokens", "active_layers"}}
        for metadata in [stats, *(action.metadata for action in actions)]:
            if "pruning_rates" in metadata:
                rates = dict(zip(self.active_layers, metadata["pruning_rates"]))
                metadata["pruning_rates"] = [rates[layer] for layer in self.layers]
        return replace(plan, token_actions=actions, this_stage_metadata=replace(plan.this_stage_metadata, stats=stats))

    def runtime_metadata(self, stage_layers: tuple[int, ...] = ()) -> dict[str, Any]:
        return {**self.rule.runtime_metadata(self.active_layers), "token_operation": self.token_operation,
                "encoder_score_layers_zero_based": list(self.layers),
                "lifecycle": "prefill_only" if self.placement is Placement.LLM_LEVEL else "native"}

    def compress(self, hidden_states: Any, *, scores: Any, keep_count: int, item_index: int) -> tuple[Any, Any, dict[str, Any]]:
        indices = _torch().arange(len(hidden_states), device=hidden_states.device)
        reports = []
        replay_plans = []
        references = {layer: values.to(hidden_states.device) for layer, values in scores.items()}
        for event_index, layer in enumerate(self.layers):
            values = references[layer]
            length = len(hidden_states)
            hidden_states, lengths, details, plan = self.rule.compress(
                hidden_states, values, [len(hidden_states)], layer_index=layer,
                active_layers=self.active_layers, collect_details=True,
            )
            representatives = self.rule.representative_indices(length, plan, indices.device)
            indices = indices.index_select(0, representatives)
            with _torch().no_grad():
                for pending in self.layers[event_index + 1:]:
                    references[pending] = self.rule.replay(references[pending], plan)
            reports.append({**details[0], "encoder_score_layer_zero_based": layer})
            replay_plans.append(plan)
        if len(hidden_states) != keep_count:
            raise InvariantError("Encoder rule and projector per-stage budgets differ.")
        return hidden_states, indices, {"stages": reports, "_reference_replay_plans": replay_plans}

    def map_reference(self, reference: Any, *, details: Mapping[str, Any], **kwargs: Any) -> Any:
        # Plans belong to the rule; diagnostics need not encode its replay format.
        for plan in details["_reference_replay_plans"]:
            reference = self.rule.replay(reference, plan)
        return reference

    def merge_audio_tokens(self, *, hidden_states: Any, scores: Any, sample_index: int,
                           slots: Any, output_length: int, layer_index: int, audio_unit_index: int = 0) -> tuple[Any, Any, dict[str, Any]]:
        output, lengths, reports, plan = self.rule.compress(
            hidden_states, scores[sample_index, slots], [int(hidden_states.shape[0])],
            layer_index=layer_index, active_layers=self.active_layers, collect_details=True,
        )
        if lengths != [output_length]:
            raise InvariantError("Encoder rule and LLM per-stage budgets differ.")
        representatives = self.rule.representative_indices(len(hidden_states), plan, hidden_states.device)
        return output, representatives, {**reports[0], "_checkpoint_replay_plan": plan}

    def replay_merge_audio_tokens(self, *, hidden_states: Any, replay_plan: Any) -> Any:
        return self.rule.replay(hidden_states, replay_plan)


__all__ = [
    "EncoderAttentionTokenStrategy",
    "QwenAdjacentMergingEncoderStrategy",
    "QwenAttentionTopKEncoderStrategy",
    "QwenFastSLMHTAProjectorStrategy",
    "QwenProgressivePruningEncoderStrategy",
    "TemporalResamplingEncoderStrategy",
    "TemporalResamplingLLMStrategy",
]
