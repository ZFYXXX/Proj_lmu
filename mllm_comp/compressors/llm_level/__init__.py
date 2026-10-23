"""LLM-level compression namespace."""

from .audio_pyramid import AudioPyramid, AudioPyramidLLMStrategy, AudioPyramidSpec
from .fastadasp import FastAdaSP, FastAdaSPLLMStrategy, FastAdaSPSpec
from .headrouter import HeadRouter, HeadRouterLLMStrategy, HeadRouterSpec
from .qwen3_vl_attention_replacement import Qwen3VLAttentionReplacement
from .qwen3_vl_ffn_replacement import Qwen3VLFFNReplacement
from .structured_layer_pruning import (
    StructuredLLMLayerPruning,
    StructuredLLMLayerPruningSpec,
    StructuredLLMLayerPruningStrategy,
)

__all__ = [
    "AudioPyramid",
    "AudioPyramidLLMStrategy",
    "AudioPyramidSpec",
    "FastAdaSP",
    "FastAdaSPLLMStrategy",
    "FastAdaSPSpec",
    "HeadRouter",
    "HeadRouterLLMStrategy",
    "HeadRouterSpec",
    "Qwen3VLAttentionReplacement",
    "Qwen3VLFFNReplacement",
    "StructuredLLMLayerPruning",
    "StructuredLLMLayerPruningSpec",
    "StructuredLLMLayerPruningStrategy",
]
