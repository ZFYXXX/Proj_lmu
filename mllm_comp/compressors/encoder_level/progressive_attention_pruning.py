"""Progressive attention pruning between audio encoder blocks."""

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

from .attention_topk import AttentionTopKEncoderStrategy
from .adjacent_token_merging import qwen_audio_chunk_lengths_after_cnn


PruningPolicy = Literal["fixed", "linear"]


@dataclass(frozen=True)
class ProgressiveAttentionPruningSpec:
    """Normalized ICME progressive-pruning settings (zero-based layers)."""

    pruning_rate: float = 0.10
    start_pruning_rate: float = 0.05
    end_pruning_rate: float = 0.20
    policy: PruningPolicy = "fixed"
    first_layer: int = 6
    layer_interval: int = 1
    layer_indices: tuple[int, ...] = ()
    # Qwen's stride-2 audio pooler needs at least two encoder tokens.
    minimum_keep_tokens: int = 2
    score_source: str = "mean_received_self_attention"
    target_site: str = "audio_encoder_layer_output_post_block"

    @classmethod
    def from_config(
        cls,
        config: Mapping[str, Any],
    ) -> "ProgressiveAttentionPruningSpec":
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
            raise ValueError("activation_scope.layers must be a list")
        spec = cls(
            pruning_rate=float(
                _first_present(
                    schedule.get("pruning_rate"),
                    legacy.get("pruning_rate"),
                    cls.pruning_rate,
                )
            ),
            start_pruning_rate=float(
                _first_present(
                    schedule.get("start_pruning_rate"),
                    legacy.get("start_pruning_rate"),
                    cls.start_pruning_rate,
                )
            ),
            end_pruning_rate=float(
                _first_present(
                    schedule.get("end_pruning_rate"),
                    legacy.get("end_pruning_rate"),
                    cls.end_pruning_rate,
                )
            ),
            policy=str(
                _first_present(
                    schedule.get("policy"),
                    criterion.get("policy"),
                    legacy.get("policy"),
                    cls.policy,
                )
            ),  # type: ignore[arg-type]
            first_layer=_strict_int(
                _first_present(
                    activation_scope.get("first_layer"),
                    legacy.get("first_layer"),
                    cls.first_layer,
                ),
                "first_layer",
            ),
            layer_interval=_strict_int(
                _first_present(
                    activation_scope.get("layer_interval"),
                    legacy.get("layer_interval"),
                    cls.layer_interval,
                ),
                "layer_interval",
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
                "progressive_attention_pruning_audio requires operation.type=keep"
            )
        return spec

    def validate(self) -> None:
        for name, value in (
            ("pruning_rate", self.pruning_rate),
            ("start_pruning_rate", self.start_pruning_rate),
            ("end_pruning_rate", self.end_pruning_rate),
        ):
            if not 0 <= value < 1:
                raise ValueError(f"{name} must be in [0, 1)")
        if self.policy not in {"fixed", "linear"}:
            raise ValueError("policy must be 'fixed' or 'linear'")
        if self.first_layer < 0:
            raise ValueError("first_layer must be non-negative")
        if self.layer_interval < 1:
            raise ValueError("layer_interval must be at least 1")
        if len(set(self.layer_indices)) != len(self.layer_indices):
            raise ValueError("activation_scope.layers contains duplicates")
        if any(index < 0 for index in self.layer_indices):
            raise ValueError("activation_scope.layers must be non-negative")
        if self.minimum_keep_tokens < 1:
            raise ValueError("minimum_keep_tokens must be at least 1")
        if self.score_source != "mean_received_self_attention":
            raise ValueError(
                "progressive_attention_pruning_audio requires score_source="
                "'mean_received_self_attention'"
            )

    def resolve_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        indices = self.layer_indices or tuple(
            range(self.first_layer, num_layers, self.layer_interval)
        )
        if not indices:
            raise ValueError("Progressive pruning selects no encoder layers")
        outside = [index for index in indices if index >= num_layers]
        if outside:
            raise ValueError(
                f"Progressive-pruning layers {outside} are outside {num_layers} layers"
            )
        return indices

    def pruning_rate_for_layer(
        self,
        layer_index: int,
        active_layers: tuple[int, ...],
    ) -> float:
        position = active_layers.index(layer_index)
        if self.policy == "fixed" or len(active_layers) == 1:
            return self.pruning_rate
        fraction = position / (len(active_layers) - 1)
        return self.start_pruning_rate + fraction * (
            self.end_pruning_rate - self.start_pruning_rate
        )


def progressive_pruning_output_length(
    length: int,
    pruning_rate: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    if length < 0:
        raise ValueError("length must be non-negative")
    if length == 0:
        return 0
    return min(
        length,
        max(
            minimum_keep_tokens,
            int(math.floor(length * (1.0 - pruning_rate))),
        ),
    )


class ProgressivePruningEncoderStrategy(AttentionTopKEncoderStrategy):
    metadata_fields = (
        "pruning_rate",
        "start_pruning_rate",
        "end_pruning_rate",
        "layer_interval",
    )

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]:
        result = AudioLayerTokenStrategyBase.runtime_metadata(self, active_layers)
        result["pruning_rates"] = [
            self.spec.pruning_rate_for_layer(layer, active_layers)
            for layer in active_layers
        ]
        return result

    def enabled(self, layer_index: int, active_layers: tuple[int, ...]) -> bool:
        return (
            layer_index in active_layers
            and self.spec.pruning_rate_for_layer(layer_index, active_layers) > 0
        )

    def output_lengths(
        self,
        layer_index: int,
        chunk_lengths: list[int],
        active_layers: tuple[int, ...],
    ) -> list[int]:
        rate = self.spec.pruning_rate_for_layer(layer_index, active_layers)
        return [
            progressive_pruning_output_length(
                length,
                rate,
                minimum_keep_tokens=self.spec.minimum_keep_tokens,
            )
            for length in chunk_lengths
        ]



class ProgressiveAttentionPruningAudio:
    """Encoder-level plan provider for fixed or linearly progressive pruning."""

    name = "progressive_attention_pruning_audio"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        ProgressiveAttentionPruningSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        spec = ProgressiveAttentionPruningSpec.from_config(config)
        defer_token_budget = bool(context.extra.get("defer_token_budget", False))
        audio_encoder = context.structural_handles.get("audio_encoder")
        layers = list(getattr(audio_encoder, "layers", ()))
        n_window = getattr(audio_encoder, "n_window", None)
        if not layers or n_window is None:
            raise InvariantError(
                "Progressive pruning requires audio_encoder.layers and n_window"
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
                "Progressive-pruning feature lengths do not match audio spans."
            )
        rates = [
            spec.pruning_rate_for_layer(layer, active_layers) for layer in active_layers
        ]
        actions: list[TokenAction] = []
        original_total = 0
        planned_total = 0
        initial_encoder_total = 0
        final_encoder_total = 0
        for audio_unit_index, (span, feature_length) in enumerate(
            zip(audio_spans, feature_lengths)
        ):
            initial_chunks = qwen_audio_chunk_lengths_after_cnn(
                feature_length,
                int(n_window),
            )
            final_chunks = list(initial_chunks)
            for rate in rates:
                final_chunks = [
                    progressive_pruning_output_length(
                        length,
                        rate,
                        minimum_keep_tokens=spec.minimum_keep_tokens,
                    )
                    for length in final_chunks
                ]
            initial_encoder_count = sum(initial_chunks)
            final_encoder_count = sum(final_chunks)
            original_count = span.end - span.start
            if not defer_token_budget and original_count != initial_encoder_count // 2:
                raise InvariantError(
                    "Progressive pruning raw audio length does not reproduce the "
                    f"processor placeholder count for audio unit {audio_unit_index}."
                )
            keep_count = original_count if defer_token_budget else final_encoder_count // 2
            if keep_count < 1:
                raise InvariantError(
                    "Progressive pruning would leave no pooled audio token."
                )
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
                        "active_layers": list(active_layers),
                        "pruning_rates": rates,
                        "token_order": "chronological",
                    },
                )
            )
        stage_name = str(config.get("_stage_name") or self.name)
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
                    "policy": spec.policy,
                    "active_layers": list(active_layers),
                    "pruning_rates": rates,
                    "original_audio_tokens": original_total,
                    "planned_audio_tokens": planned_total,
                    "initial_encoder_tokens": initial_encoder_total,
                    "planned_encoder_tokens": final_encoder_total,
                    "selection_deferred_to_runtime": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("ProgressiveAttentionPruningAudio has no trainable state")


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
