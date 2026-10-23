"""Encoder attention Top-K pruning for audio transformer blocks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

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

from .adjacent_token_merging import qwen_audio_chunk_lengths_after_cnn


@dataclass(frozen=True)
class EncoderAttentionTopKSpec:
    """Normalized ECAI Top-K settings.

    ``layer_indices`` are zero-based.  The defaults ``(3, 6, 9)`` therefore
    correspond to the paper's 4th, 7th and 10th transformer blocks.
    """

    keep_ratio: float = 0.90
    layer_indices: tuple[int, ...] = (3, 6, 9)
    # Qwen's stride-2 audio pooler needs at least two encoder tokens.
    minimum_keep_tokens: int = 2
    score_source: str = "mean_received_self_attention"
    target_site: str = "audio_encoder_layer_post_mhsa_pre_ffn"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "EncoderAttentionTopKSpec":
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
            cls.layer_indices,
        )
        if not isinstance(raw_layers, (list, tuple)):
            raise ValueError("activation_scope.layers must be a list")
        spec = cls(
            keep_ratio=float(
                _first_present(
                    schedule.get("keep_ratio"),
                    legacy.get("keep_ratio"),
                    cls.keep_ratio,
                )
            ),
            layer_indices=tuple(
                _strict_int(value, "activation_scope.layers") for value in raw_layers
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
            raise ValueError(
                "encoder_attention_topk_audio requires operation.type=keep"
            )
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("keep_ratio must be in (0, 1]")
        if not self.layer_indices:
            raise ValueError("activation_scope.layers must not be empty")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("activation_scope.layers contains duplicates")
        if any(index < 0 for index in self.layer_indices):
            raise ValueError("activation_scope.layers must be non-negative")
        if self.minimum_keep_tokens < 1:
            raise ValueError("minimum_keep_tokens must be at least 1")
        if self.score_source != "mean_received_self_attention":
            raise ValueError(
                "encoder_attention_topk_audio requires score_source="
                "'mean_received_self_attention'"
            )

    def resolve_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        outside = [index for index in self.layer_indices if index >= num_layers]
        if outside:
            raise ValueError(
                f"Top-K layer indices {outside} are outside {num_layers} encoder layers"
            )
        return self.layer_indices


def attention_topk_output_length(
    length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return 0
    return min(
        length,
        max(minimum_keep_tokens, int(math.floor(length * keep_ratio))),
    )


def qwen_planned_attention_topk_lengths(
    feature_length: int,
    *,
    n_window: int,
    keep_ratio: float,
    active_layer_count: int,
    minimum_keep_tokens: int = 1,
) -> tuple[list[int], list[int]]:
    initial = qwen_audio_chunk_lengths_after_cnn(feature_length, n_window)
    final = list(initial)
    for _ in range(active_layer_count):
        final = [
            attention_topk_output_length(
                length,
                keep_ratio,
                minimum_keep_tokens=minimum_keep_tokens,
            )
            for length in final
        ]
    return initial, final


def select_ragged_attention_topk(
    hidden_states: Any,
    scores: Any,
    chunk_lengths: list[int],
    output_lengths: list[int],
    *,
    collect_details: bool = False,
    replay_decisions: list[Any] | None = None,
    scorer: Any = None, selector: Any = None,
) -> tuple[Any, list[dict[str, Any]]]:
    """Select exact per-chunk Top-K budgets and restore chronological order."""

    torch = _torch()
    if len(chunk_lengths) != len(output_lengths):
        raise InvariantError("Top-K input and output chunk counts do not match.")
    if sum(chunk_lengths) != int(hidden_states.shape[0]) or int(
        scores.shape[0]
    ) != int(hidden_states.shape[0]):
        raise InvariantError("Top-K tensors do not match packed chunk lengths.")
    hidden_chunks = hidden_states.split(chunk_lengths, dim=0)
    score_chunks = scores.split(chunk_lengths, dim=0)
    selected_chunks: list[Any] = []
    reports: list[dict[str, Any]] = []
    for chunk_index, (hidden_chunk, score_chunk, output_length) in enumerate(
        zip(hidden_chunks, score_chunks, output_lengths)
    ):
        input_length = int(hidden_chunk.shape[0])
        if not 1 <= output_length <= input_length:
            raise InvariantError(
                f"Top-K output length {output_length} is invalid for {input_length}."
            )
        if output_length == input_length:
            indices = torch.arange(
                input_length,
                dtype=torch.long,
                device=hidden_states.device,
            )
            selected = hidden_chunk
        else:
            from mllm_comp.compressors.audio_rule_utils import token_scores, token_indices
            if scorer is not None:
                score_chunk = token_scores(scorer({"values": hidden_chunk.detach(), "scores": score_chunk,
                    "keep_count": output_length, "chunk_index": chunk_index}), hidden_chunk, length=input_length)
            if selector is not None:
                indices = token_indices(selector({"values": hidden_chunk.detach(), "scores": score_chunk,
                    "keep_count": output_length, "chunk_index": chunk_index}),
                    length=input_length, count=output_length, device=hidden_chunk.device)
            else:
                indices = None
            try:
                ranking = torch.argsort(
                    score_chunk,
                    descending=True,
                    stable=True,
                )
            except TypeError:  # pragma: no cover
                ranking = torch.argsort(score_chunk, descending=True)
            if indices is None:
                indices = torch.sort(ranking[:output_length])[0].to(device=hidden_chunk.device, dtype=torch.long)
            selected = hidden_chunk.index_select(0, indices)
        selected_chunks.append(selected)
        if replay_decisions is not None:
            replay_decisions.append(indices.detach())
        if collect_details:
            reports.append(
                {
                    "chunk_index": chunk_index,
                    "input_tokens": input_length,
                    "output_tokens": output_length,
                    "kept_indices": indices.detach(),
                    "score_min": score_chunk.min().detach(),
                    "score_max": score_chunk.max().detach(),
                }
            )
    return torch.cat(selected_chunks, dim=0), reports



class AttentionTopKEncoderStrategy(AudioLayerTokenStrategyBase):
    token_operation = "keep"
    encoder_reference_kind = "received_attention"
    metadata_fields = ("keep_ratio", "policy", "layer_interval")

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]:
        result = super().runtime_metadata(active_layers)
        result["keep_ratio_per_layer"] = self.spec.keep_ratio
        return result

    def enabled(self, layer_index: int, active_layers: tuple[int, ...]) -> bool:
        return layer_index in active_layers and self.spec.keep_ratio < 1

    def output_lengths(
        self,
        layer_index: int,
        chunk_lengths: list[int],
        active_layers: tuple[int, ...],
    ) -> list[int]:
        del layer_index, active_layers
        return [
            attention_topk_output_length(
                length,
                self.spec.keep_ratio,
                minimum_keep_tokens=self.spec.minimum_keep_tokens,
            )
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
        lengths = self.output_lengths(layer_index, chunk_lengths, active_layers)
        decisions: list[Any] = []
        compressed, reports = select_ragged_attention_topk(
            hidden_states,
            probe,
            chunk_lengths,
            lengths,
            collect_details=collect_details,
            replay_decisions=decisions, scorer=self.rules.get("scorer"), selector=self.rules.get("selector"),
        )
        return compressed, lengths, reports, (tuple(chunk_lengths), tuple(decisions))

    def replay(self, hidden_states: Any, plan: Any) -> Any:
        lengths, decisions = plan
        return _torch().cat([
            chunk if int(indices.numel()) == length else chunk.index_select(0, indices.to(chunk.device))
            for chunk, length, indices in zip(hidden_states.split(lengths), lengths, decisions)
        ], dim=0)

    def representative_indices(self, length: int, plan: Any, device: Any) -> Any:
        indices = _torch().arange(length, device=device)
        return self.replay(indices, plan)



class EncoderAttentionTopKAudio:
    """Encoder-level plan provider for attention Top-K pruning."""

    name = "encoder_attention_topk_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        EncoderAttentionTopKSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        spec = EncoderAttentionTopKSpec.from_config(config)
        audio_encoder = context.structural_handles.get("audio_encoder")
        layers = list(getattr(audio_encoder, "layers", ()))
        n_window = getattr(audio_encoder, "n_window", None)
        if not layers or n_window is None:
            raise InvariantError(
                "Encoder Top-K planning requires audio_encoder.layers and n_window"
            )
        active_layers = spec.resolve_layer_indices(len(layers))
        return _build_plan(
            state=state,
            config=config,
            spec=spec,
            active_layers=active_layers,
            n_window=int(n_window),
            defer_token_budget=bool(context.extra.get("defer_token_budget", False)),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("EncoderAttentionTopKAudio has no trainable state")


def _build_plan(
    *,
    state: ForwardState,
    config: dict[str, Any],
    spec: EncoderAttentionTopKSpec,
    active_layers: tuple[int, ...],
    n_window: int,
    defer_token_budget: bool = False,
) -> CompressionPlan:
    feature_lengths = _integer_sequence(
        state.audio_feature_lengths,
        "state.audio_feature_lengths",
    )
    audio_spans = [
        span for span in state.modality_token_spans if span.modality == "audio"
    ]
    if len(feature_lengths) != len(audio_spans):
        raise InvariantError("Encoder Top-K feature lengths do not match audio spans.")
    actions: list[TokenAction] = []
    original_total = 0
    planned_total = 0
    initial_encoder_total = 0
    final_encoder_total = 0
    for audio_unit_index, (span, feature_length) in enumerate(
        zip(audio_spans, feature_lengths)
    ):
        initial_chunks, final_chunks = qwen_planned_attention_topk_lengths(
            feature_length,
            n_window=n_window,
            keep_ratio=spec.keep_ratio,
            active_layer_count=len(active_layers),
            minimum_keep_tokens=spec.minimum_keep_tokens,
        )
        initial_encoder_count = sum(initial_chunks)
        final_encoder_count = sum(final_chunks)
        original_count = span.end - span.start
        if not defer_token_budget and original_count != initial_encoder_count // 2:
            raise InvariantError(
                "Encoder Top-K raw audio length does not reproduce the processor "
                f"placeholder count for audio unit {audio_unit_index}."
            )
        keep_count = original_count if defer_token_budget else final_encoder_count // 2
        if keep_count < 1:
            raise InvariantError("Encoder Top-K would leave no pooled audio token.")
        original_total += original_count
        planned_total += keep_count
        initial_encoder_total += initial_encoder_count
        final_encoder_total += final_encoder_count
        actions.append(
            TokenAction(
                op="keep",
                target_indices=None,
                metadata={
                    "selection_deferred": True,
                    "selector": EncoderAttentionTopKAudio.name,
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
                    "keep_ratio": spec.keep_ratio,
                    "active_layers": list(active_layers),
                    "token_order": "chronological",
                },
            )
        )
    stage_name = str(config.get("_stage_name") or EncoderAttentionTopKAudio.name)
    return CompressionPlan(
        stage_name=stage_name,
        placement=Placement.ENCODER_LEVEL,
        execution_phase=ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        ),
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
                "keep_ratio_per_layer": spec.keep_ratio,
                "active_layers": list(active_layers),
                "original_audio_tokens": original_total,
                "planned_audio_tokens": planned_total,
                "initial_encoder_tokens": initial_encoder_total,
                "planned_encoder_tokens": final_encoder_total,
                "selection_deferred_to_runtime": True,
            },
        ),
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


def _integer_sequence(value: Any, name: str) -> list[int]:
    tolist = getattr(value, "tolist", None)
    raw = tolist() if callable(tolist) else value
    if raw is None:
        return []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise InvariantError(f"{name} must be a sequence")
    return [int(item) for item in raw]
