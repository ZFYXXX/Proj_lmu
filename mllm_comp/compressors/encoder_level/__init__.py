"""Encoder-level compression namespace."""

from .adjacent_token_merging import (
    AdjacentTokenMergingAudio,
    AdjacentTokenMergingSpec,
    adjacent_merge_count,
    adjacent_merge_output_length,
    qwen_audio_chunk_lengths_after_cnn,
    qwen_planned_a_tome_lengths,
)
from .attention_topk import (
    EncoderAttentionTopKAudio,
    EncoderAttentionTopKSpec,
    attention_topk_output_length,
    qwen_planned_attention_topk_lengths,
)
from .progressive_attention_pruning import (
    ProgressiveAttentionPruningAudio,
    ProgressiveAttentionPruningSpec,
    progressive_pruning_output_length,
)

__all__ = [
    "AdjacentTokenMergingAudio",
    "AdjacentTokenMergingSpec",
    "EncoderAttentionTopKAudio",
    "EncoderAttentionTopKSpec",
    "ProgressiveAttentionPruningAudio",
    "ProgressiveAttentionPruningSpec",
    "adjacent_merge_count",
    "adjacent_merge_output_length",
    "attention_topk_output_length",
    "progressive_pruning_output_length",
    "qwen_audio_chunk_lengths_after_cnn",
    "qwen_planned_a_tome_lengths",
    "qwen_planned_attention_topk_lengths",
]
