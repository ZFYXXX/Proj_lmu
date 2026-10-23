"""Small public strategy contracts for the four implemented executor families."""

from __future__ import annotations

from typing import Any, Protocol


class ProjectorTokenStrategy(Protocol):
    """Algorithm surface used by a pre/post-projector token executor."""

    spec: Any

    @property
    def score_source(self) -> str: ...

    @property
    def attention_layer_offset(self) -> int | None: ...

    @property
    def uses_distinct_merge_prune_importance(self) -> bool: ...

    def runtime_metadata(self) -> dict[str, Any]: ...

    def compress(
        self,
        hidden_states: Any,
        *,
        scores: Any | None,
        keep_count: int,
        item_index: int,
    ) -> tuple[Any, Any | None, dict[str, Any]]: ...


class EncoderTokenStrategy(Protocol):
    """Common surface used by a layerwise audio-encoder executor.

    Scored rules additionally provide probe(layer, normalized, chunk_lengths).
    A rule without a layer-input score declares requires_layer_input=False.
    Content-dependent output lengths may be None until compress returns the
    realized layout; the executor resolves placeholder counts by preflight.
    """

    spec: Any

    def active_layer_indices(self, num_layers: int) -> tuple[int, ...]: ...

    def runtime_metadata(self, active_layers: tuple[int, ...]) -> dict[str, Any]: ...

    def enabled(self, layer_index: int, active_layers: tuple[int, ...]) -> bool: ...

    def output_lengths(
        self,
        layer_index: int,
        chunk_lengths: list[int],
        active_layers: tuple[int, ...],
    ) -> list[int | None]: ...

    def compress(
        self,
        hidden_states: Any,
        probe: Any,
        chunk_lengths: list[int],
        *,
        layer_index: int,
        active_layers: tuple[int, ...],
        collect_details: bool,
    ) -> tuple[Any, list[int], list[dict[str, Any]], Any]: ...

    def replay(self, hidden_states: Any, plan: Any) -> Any: ...


class LLMTokenStrategy(Protocol):
    """Common surface used at decoder-layer token boundaries.

    Scored rules additionally provide score_audio_tokens/select_indices; value
    transforms provide merge_audio_tokens/replay_merge_audio_tokens. A rule
    without a layer-input score declares requires_layer_input=False. Value
    transforms may return output_length=None when the actual boundary features
    determine the budget; merge_audio_tokens then supplies the realized length.
    """

    spec: Any

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]: ...

    def output_length(self, current_length: int) -> int | None: ...

    def runtime_metadata(self, stage_layers: tuple[int, ...]) -> dict[str, Any]: ...


class LLMOperationStrategy(Protocol):
    """Algorithm surface that emits semantic structural operation actions."""

    spec: Any

    def plan_operations(self, *, model: Any, structural_handles: Any) -> list[Any]: ...

    def runtime_metadata(self) -> dict[str, Any]: ...


__all__ = [
    "EncoderTokenStrategy",
    "LLMOperationStrategy",
    "LLMTokenStrategy",
    "ProjectorTokenStrategy",
]
