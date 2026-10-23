"""Native Qwen3.5 image/text input and output validation."""

from mllm_comp.backbones.qwen3_vl.validation import Qwen3VLNativeValidator


class Qwen3_5NativeValidator(Qwen3VLNativeValidator):
    """Reuse the shared M-RoPE/input-shape checks under a Qwen3.5 identity."""

