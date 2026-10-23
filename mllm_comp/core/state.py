"""Normalized model-side state consumed by compressors and legalizers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .layout import AlignmentDelta, LayoutDelta, ModalityLayout
    from .metadata import StageMetadata

Tensor = Any


@dataclass(frozen=True)
class TokenSpan:
    """Span in the serialized LLM token or embedding sequence."""

    sample_index: int
    start: int
    end: int
    modality: Literal["text", "image", "video", "audio", "scene_3d"] = "image"
    role: Literal["instruction", "observation", "answer"] = "observation"
    source: str | None = None
    media_unit: str | None = None
    time_span: tuple[float, float] | None = None
    encoder_origin: str | None = None
    alignment_group: str | None = None


@dataclass
class ForwardState:
    """Backbone-normalized forward state.

    The state is created by a backbone StateBuilder. Compressor plugins read it
    and produce actions. The executor that owns the active compression path
    writes stage_history; integrations must not update it independently.
    """

    input_ids: Tensor | None = None
    inputs_embeds: Tensor | None = None
    attention_mask: Tensor | None = None
    position_ids: Tensor | None = None
    labels: Tensor | None = None

    pixel_values: Tensor | None = None
    image_grid_thw: Tensor | None = None
    video_pixel_values: Tensor | None = None
    video_grid_thw: Tensor | None = None
    audio_values: Tensor | None = None
    audio_attention_mask: Tensor | None = None
    audio_feature_lengths: Tensor | None = None
    video_second_per_grid: Tensor | None = None

    # Exact provider-native kwargs are retained so runners do not have to
    # reconstruct version-specific names from normalized platform fields.
    native_model_inputs: dict[str, Tensor] = field(default_factory=dict)

    token_type_mask: Tensor | None = None
    modality_token_spans: list[TokenSpan] = field(default_factory=list)
    modality_metadata: dict[str, Any] = field(default_factory=dict)

    # Compression-v2 keeps each physical modality stream independent.  A
    # stream may be one image, one video clip, one audio track, or a named side
    # branch such as Qwen3-VL DeepStack.  Layout and alignment histories make
    # multi-stage legalization auditable instead of reconstructing it from
    # mutable provider kwargs.
    modality_layouts: dict[str, "ModalityLayout"] = field(default_factory=dict)
    layout_history: dict[str, list["LayoutDelta"]] = field(default_factory=dict)
    alignment_state: dict[str, "AlignmentDelta"] = field(default_factory=dict)

    hidden_states: Tensor | None = None
    sample_ids: list[str | int] | None = None
    stage_history: dict[str, "StageMetadata"] = field(default_factory=dict)
