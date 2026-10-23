"""Qwen2.5-Omni composed facade components and factory."""

from .backbone import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    QWEN25_OMNI_OUTPUT_SAMPLE_RATE,
    Qwen25OmniLegalizer,
    Qwen25OmniModelAdapter,
    Qwen25OmniProcessorAdapter,
    Qwen25OmniRunner,
    Qwen25OmniSample,
    Qwen25OmniStateBuilder,
    create_qwen25_omni_backbone,
)
from .validation import Qwen25OmniNativeValidator
from .audio_compression import (
    Qwen25OmniAudioCompressionController,
    audio_placeholder_keep_positions,
    legalize_qwen25_omni_audio_tokens,
)
from .token_debug import Qwen25OmniTokenDebugCollector
from .encoder_compression import Qwen25OmniEncoderCompressionController
from .training_compression import (
    build_qwen25_omni_training_state,
    model_inputs_from_forward_call,
)


def export_qwen25_omni_operation_checkpoint(**kwargs):
    """Load the v2 export binding lazily to keep registry imports acyclic."""

    from .v2_compression import export_qwen25_omni_operation_checkpoint as export

    return export(**kwargs)

__all__ = [
    "DEFAULT_AUDIO_SYSTEM_PROMPT",
    "QWEN25_OMNI_OUTPUT_SAMPLE_RATE",
    "Qwen25OmniLegalizer",
    "Qwen25OmniModelAdapter",
    "Qwen25OmniNativeValidator",
    "Qwen25OmniAudioCompressionController",
    "Qwen25OmniEncoderCompressionController",
    "build_qwen25_omni_training_state",
    "model_inputs_from_forward_call",
    "Qwen25OmniTokenDebugCollector",
    "Qwen25OmniProcessorAdapter",
    "Qwen25OmniRunner",
    "Qwen25OmniSample",
    "Qwen25OmniStateBuilder",
    "create_qwen25_omni_backbone",
    "export_qwen25_omni_operation_checkpoint",
    "audio_placeholder_keep_positions",
    "legalize_qwen25_omni_audio_tokens",
]
