"""Backbone capability profiles for supported MLLM families."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class BackboneCapabilities:
    """Public capability declaration for one backbone adapter."""

    name: Literal[
        "llava_onevision", "qwen25_vl", "qwen25_omni", "qwen3_vl", "qwen3_5"
    ]
    family: Literal["llava", "qwen_vl", "qwen_omni"]
    code_origin: Literal["external_source_tree", "transformers_native"]
    source_ref: str
    loader_strategy: Literal["source_tree_adapter", "transformers_auto", "transformers_custom"]
    supports_image: bool
    supports_video: bool
    supports_audio: bool
    supports_anyres_image: bool
    output_modalities: tuple[Literal["text", "speech"], ...]
    position_system: Literal["absolute", "rope", "m_rope", "backbone_default"]
    visual_grid_key: str | None
    audio_feature_key: str | None
    processor_output_keys: tuple[str, ...]
    required_state_fields: tuple[str, ...]
    modality_span_strategy: Literal[
        "image_token_positions",
        "image_grid_thw",
        "processor_metadata",
        "backbone_default",
    ]
    projector_boundary: Literal[
        "pre_projector_features",
        "post_projector_embeddings",
        "backbone_default",
    ]
    max_analysis_eval_batch_size: int | None
    batch_constraint_reason: str | None = None
    processor_returns_inputs_embeds: bool = False
    position_ids_owner: Literal["platform", "backbone", "backbone_default"] = "backbone_default"


LLAVA_ONEVISION_PROFILE = BackboneCapabilities(
    name="llava_onevision",
    family="llava",
    code_origin="external_source_tree",
    source_ref="LLaVA-OneVision source tree",
    loader_strategy="source_tree_adapter",
    supports_image=True,
    supports_video=True,
    supports_audio=False,
    supports_anyres_image=True,
    output_modalities=("text",),
    position_system="backbone_default",
    visual_grid_key=None,
    audio_feature_key=None,
    processor_output_keys=(
        "input_ids",
        "pixel_values",
        "attention_mask",
        "labels",
        "image_sizes",
        "modalities",
    ),
    required_state_fields=(
        "input_ids",
        "pixel_values",
        "attention_mask",
        "labels",
        "image_sizes",
        "modalities",
        "modality_token_spans",
    ),
    modality_span_strategy="processor_metadata",
    projector_boundary="post_projector_embeddings",
    max_analysis_eval_batch_size=1,
    batch_constraint_reason="LLaVA-family source-tree inference/eval/analysis paths are single-sample.",
    processor_returns_inputs_embeds=False,
    position_ids_owner="backbone_default",
)

QWEN25_VL_PROFILE = BackboneCapabilities(
    name="qwen25_vl",
    family="qwen_vl",
    code_origin="transformers_native",
    source_ref="transformers Qwen2.5-VL implementation",
    loader_strategy="transformers_auto",
    supports_image=True,
    supports_video=True,
    supports_audio=False,
    supports_anyres_image=True,
    output_modalities=("text",),
    position_system="m_rope",
    visual_grid_key="image_grid_thw",
    audio_feature_key=None,
    processor_output_keys=(
        "input_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "attention_mask",
        "labels",
    ),
    required_state_fields=(
        "input_ids",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "second_per_grid_ts",
        "attention_mask",
        "labels",
        "modality_token_spans",
    ),
    modality_span_strategy="image_grid_thw",
    projector_boundary="post_projector_embeddings",
    max_analysis_eval_batch_size=None,
    batch_constraint_reason=None,
    processor_returns_inputs_embeds=False,
    position_ids_owner="backbone",
)

QWEN25_OMNI_PROFILE = BackboneCapabilities(
    name="qwen25_omni",
    family="qwen_omni",
    code_origin="transformers_native",
    source_ref="transformers Qwen2.5-Omni implementation",
    loader_strategy="transformers_custom",
    supports_image=True,
    supports_video=True,
    supports_audio=True,
    supports_anyres_image=True,
    output_modalities=("text", "speech"),
    position_system="m_rope",
    visual_grid_key="image_grid_thw",
    audio_feature_key="input_features",
    processor_output_keys=(
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "video_second_per_grid",
        "input_features",
        "feature_attention_mask",
    ),
    required_state_fields=(
        "input_ids",
        "attention_mask",
        "modality_token_spans",
        "modality_metadata",
    ),
    modality_span_strategy="processor_metadata",
    projector_boundary="post_projector_embeddings",
    max_analysis_eval_batch_size=None,
    batch_constraint_reason=None,
    processor_returns_inputs_embeds=False,
    position_ids_owner="backbone",
)


QWEN3_VL_PROFILE = BackboneCapabilities(
    name="qwen3_vl",
    family="qwen_vl",
    code_origin="transformers_native",
    source_ref="transformers Qwen3-VL implementation",
    loader_strategy="transformers_auto",
    supports_image=True,
    supports_video=True,
    supports_audio=False,
    supports_anyres_image=True,
    output_modalities=("text",),
    position_system="m_rope",
    visual_grid_key="image_grid_thw",
    audio_feature_key=None,
    processor_output_keys=(
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "mm_token_type_ids",
    ),
    required_state_fields=(
        "input_ids",
        "attention_mask",
        "modality_token_spans",
        "modality_metadata",
    ),
    modality_span_strategy="processor_metadata",
    projector_boundary="post_projector_embeddings",
    max_analysis_eval_batch_size=None,
    batch_constraint_reason=None,
    processor_returns_inputs_embeds=False,
    position_ids_owner="backbone",
)

QWEN3_5_PROFILE = BackboneCapabilities(
    name="qwen3_5",
    family="qwen_vl",
    code_origin="transformers_native",
    source_ref="transformers Qwen3.5 multimodal implementation",
    loader_strategy="transformers_auto",
    supports_image=True,
    supports_video=True,
    supports_audio=False,
    supports_anyres_image=True,
    output_modalities=("text",),
    position_system="m_rope",
    visual_grid_key="image_grid_thw",
    audio_feature_key=None,
    processor_output_keys=(
        "input_ids",
        "attention_mask",
        "pixel_values",
        "image_grid_thw",
        "pixel_values_videos",
        "video_grid_thw",
        "mm_token_type_ids",
    ),
    required_state_fields=(
        "input_ids",
        "attention_mask",
        "modality_token_spans",
        "modality_metadata",
    ),
    modality_span_strategy="processor_metadata",
    projector_boundary="post_projector_embeddings",
    max_analysis_eval_batch_size=1,
    batch_constraint_reason="Qwen3.5 projector compression is verified for single-sample image/video requests.",
    processor_returns_inputs_embeds=False,
    position_ids_owner="backbone",
)
