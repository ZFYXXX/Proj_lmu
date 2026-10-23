"""Audio-Pyramid: progressive attention-based audio-token pruning in the LLM.

The plugin describes intent and per-audio-unit budgets.  Qwen-specific layer
layout handling is owned by the LLM-token executor; the strategy below owns
only score computation and chronological Top-K selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Any, Mapping

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.llm_level._rotary import apply_rotary_position_embeddings
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.compressors.audio_rule_utils import audio_rule_metadata, token_scores, token_indices
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
class AudioPyramidSpec:
    """Normalized Audio-Pyramid configuration.

    ``stage_layers`` are zero-based decoder-layer indices.  Compression occurs
    after each listed layer and only during prefill.
    """

    keep_ratio: float = 0.5
    stage_layers: tuple[int, ...] = (8, 17, 26)
    minimum_keep_tokens: int = 1
    score_source: str = "instruction_to_audio_received_attention"
    target_site: str = "llm_decoder_layer_output_audio_tokens"
    lifecycle: str = "prefill_only"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AudioPyramidSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        activation_scope = _mapping(
            config.get("activation_scope"), "activation_scope"
        )
        legacy = _mapping(config.get("config"), "config")
        raw_layers = _first_present(
            activation_scope.get("layers"),
            schedule.get("stage_layers"),
            legacy.get("stage_layers"),
            cls.stage_layers,
        )
        if not isinstance(raw_layers, (list, tuple)) or any(
            isinstance(item, bool) or not isinstance(item, int) for item in raw_layers
        ):
            raise ValueError("Audio-Pyramid stage layers must be integer indices.")
        spec = cls(
            keep_ratio=float(
                _first_present(
                    schedule.get("keep_ratio"),
                    legacy.get("keep_ratio"),
                    cls.keep_ratio,
                )
            ),
            stage_layers=tuple(int(item) for item in raw_layers),
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
            lifecycle=str(
                _first_present(
                    activation_scope.get("lifecycle"),
                    legacy.get("lifecycle"),
                    cls.lifecycle,
                )
            ),
        )
        spec.validate()
        operation_type = operation.get("type")
        if operation_type not in {None, "keep"}:
            raise ValueError("Audio-Pyramid requires operation.type=keep.")
        return spec

    def validate(self) -> None:
        if not 0 < self.keep_ratio <= 1:
            raise ValueError("Audio-Pyramid keep_ratio must be in (0, 1].")
        if self.minimum_keep_tokens < 1:
            raise ValueError("Audio-Pyramid minimum_keep_tokens must be positive.")
        if not self.stage_layers:
            raise ValueError("Audio-Pyramid requires at least one stage layer.")
        if tuple(sorted(set(self.stage_layers))) != self.stage_layers:
            raise ValueError(
                "Audio-Pyramid stage layers must be unique and strictly increasing."
            )
        if self.stage_layers[0] < 0:
            raise ValueError("Audio-Pyramid stage layers must be non-negative.")
        if self.score_source != "instruction_to_audio_received_attention":
            raise ValueError(
                "Audio-Pyramid requires instruction-to-audio received attention."
            )
        if self.lifecycle != "prefill_only":
            raise ValueError("Audio-Pyramid lifecycle must be 'prefill_only'.")


def pyramid_output_length(
    current_length: int,
    keep_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    if current_length < 0:
        raise ValueError("current_length must be non-negative")
    if current_length == 0:
        return 0
    return min(
        current_length,
        max(minimum_keep_tokens, int(ceil(current_length * keep_ratio))),
    )


class AudioPyramidLLMStrategy:
    """Attention score and exact chronological selection strategy."""

    def __init__(self, spec: AudioPyramidSpec) -> None:
        self.spec = spec

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        if self.spec.stage_layers[-1] >= num_layers:
            raise InvariantError(
                "Audio-Pyramid stage layer is outside the decoder: "
                f"layers={self.spec.stage_layers}, num_layers={num_layers}."
            )
        return self.spec.stage_layers

    def output_length(self, current_length: int) -> int:
        return pyramid_output_length(
            current_length,
            self.spec.keep_ratio,
            minimum_keep_tokens=self.spec.minimum_keep_tokens,
        )

    def runtime_metadata(self, stage_layers: tuple[int, ...]) -> dict[str, Any]:
        return {
            "target_site": self.spec.target_site,
            "score_source": self.spec.score_source,
            "keep_ratio_per_stage": self.spec.keep_ratio,
            "stage_layers_zero_based": list(stage_layers),
            "stage_boundaries_one_based": [item + 1 for item in stage_layers],
            "minimum_keep_tokens": self.spec.minimum_keep_tokens,
            "lifecycle": self.spec.lifecycle,
            "prefill_compression": True,
            "decode_compression": False,
            "position_policy": "preserve_absolute",
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
    ) -> Any:
        """Recompute local Q/K attention without requesting model attentions."""

        torch = _torch()
        hidden = normalized_hidden_states
        batch_size, sequence_length, _ = hidden.shape
        num_heads = int(attention_module.config.num_attention_heads)
        num_key_value_heads = int(attention_module.config.num_key_value_heads)
        head_dim = int(getattr(attention_module, "head_dim", 0))
        if head_dim <= 0:
            head_dim = int(attention_module.q_proj.out_features // num_heads)
        with torch.no_grad():
            query = attention_module.q_proj(hidden).view(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2)
            key = attention_module.k_proj(hidden).view(
                batch_size, sequence_length, num_key_value_heads, head_dim
            ).transpose(1, 2)
            if position_embeddings is not None:
                query, key = apply_rotary_position_embeddings(
                    query,
                    key,
                    position_embeddings=position_embeddings,
                    attention_module=attention_module,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                )
            groups = num_heads // num_key_value_heads
            if groups > 1:
                key = key.repeat_interleave(groups, dim=1)
            scale = float(getattr(attention_module, "scaling", head_dim**-0.5))
            outputs = hidden.new_zeros(
                (batch_size, sequence_length), dtype=torch.float32
            )
            for batch_index in range(batch_size):
                query_indices = torch.nonzero(
                    instruction_mask[batch_index] & valid_mask[batch_index],
                    as_tuple=False,
                ).flatten()
                audio_indices = torch.nonzero(
                    audio_mask[batch_index] & valid_mask[batch_index],
                    as_tuple=False,
                ).flatten()
                if int(audio_indices.numel()) == 0:
                    continue
                if int(query_indices.numel()) == 0:
                    outputs[batch_index, audio_indices] = 0
                    continue
                q = query[batch_index, :, query_indices, :]
                k = key[batch_index]
                query_absolute = absolute_positions[batch_index, query_indices]
                key_absolute = absolute_positions[batch_index]
                causal = key_absolute.unsqueeze(0) <= query_absolute.unsqueeze(1)
                allowed = causal & valid_mask[batch_index].unsqueeze(0)
                # Swift/Accelerate wraps mixed-precision training in autocast.
                # Keep this auxiliary ranking calculation in FP32: otherwise
                # autocast can return BF16 logits and the FP32 mask sentinel
                # cannot be represented by their dtype.
                with torch.autocast(device_type=q.device.type, enabled=False):
                    logits = torch.matmul(
                        q.float(), k.float().transpose(-2, -1)
                    ) * scale
                    logits = logits.masked_fill(
                        ~allowed.unsqueeze(0), torch.finfo(logits.dtype).min
                    )
                    weights = torch.softmax(
                        logits, dim=-1, dtype=torch.float32
                    )
                outputs[batch_index, audio_indices] = weights[
                    :, :, audio_indices
                ].mean(dim=(0, 1))
            scorer = getattr(self, "rules", {}).get("scorer")
            if scorer is not None:
                for batch_index in range(batch_size):
                    slots = (audio_mask[batch_index] & valid_mask[batch_index]).nonzero(as_tuple=False).flatten()
                    if slots.numel():
                        outputs[batch_index, slots] = token_scores(scorer({
                            "values": hidden[batch_index, slots].detach(), "scores": outputs[batch_index, slots],
                            "query": query[batch_index].detach(), "key": key[batch_index].detach(),
                            "instruction_mask": instruction_mask[batch_index], "valid_mask": valid_mask[batch_index],
                            "audio_indices": slots, "absolute_positions": absolute_positions[batch_index]}),
                            hidden, length=int(slots.numel()))
            if not bool(torch.isfinite(outputs).all().item()):
                raise InvariantError("Audio-Pyramid importance contains NaN or Inf.")
            return outputs

    def select_indices(self, scores: Any, keep_count: int) -> Any:
        torch = _torch()
        if not 1 <= keep_count <= int(scores.numel()):
            raise InvariantError(
                f"Audio-Pyramid keep_count={keep_count} is invalid for "
                f"{int(scores.numel())} tokens."
            )
        selector = getattr(self, "rules", {}).get("selector")
        if selector is not None and keep_count != int(scores.numel()):
            return token_indices(selector({"scores": scores.detach(), "keep_count": keep_count}),
                length=int(scores.numel()), count=keep_count, device=scores.device)
        try:
            ranking = torch.argsort(scores, descending=True, stable=True)
        except TypeError:  # pragma: no cover
            ranking = torch.argsort(scores, descending=True)
        return torch.sort(ranking[:keep_count])[0].to(dtype=torch.long)


class AudioPyramid:
    name = "audio_pyramid_attention_pruning"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        AudioPyramidSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = AudioPyramidSpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        final_total = 0
        unit_index = 0
        for span in state.modality_token_spans:
            if span.modality != "audio":
                continue
            length = span.end - span.start
            stage_lengths = [length]
            for _ in spec.stage_layers:
                stage_lengths.append(
                    pyramid_output_length(
                        stage_lengths[-1],
                        spec.keep_ratio,
                        minimum_keep_tokens=spec.minimum_keep_tokens,
                    )
                )
            original_total += length
            final_total += stage_lengths[-1]
            actions.append(
                TokenAction(
                    op="keep",
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
                        "stage_layers": list(spec.stage_layers),
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
                    span for span in state.modality_token_spans if span.modality == "audio"
                ],
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "lifecycle": spec.lifecycle,
                    "stage_layers": list(spec.stage_layers),
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
            raise ValueError("AudioPyramid has no trainable state")


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InvariantError("Audio-Pyramid requires PyTorch.") from exc
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
    "AudioPyramid",
    "AudioPyramidLLMStrategy",
    "AudioPyramidSpec",
    "pyramid_output_length",
]
