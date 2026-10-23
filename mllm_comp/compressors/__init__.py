"""Compression plugin namespaces and researcher-facing registration API."""

from .registry import (
    CompressionMethodDescriptor,
    CompressionMethodRegistry,
    ExecutorFamily,
    builtin_compression_methods,
    get_compression_method_descriptor,
    register_compression_method,
)
from .contracts import (
    AlgorithmDescriptor,
    BackbonePlacementBinding,
    BindingTarget,
    CompressionContractRegistry,
    CouplingPolicy,
    MediaAlignment,
    ModalityVariantDescriptor,
    TokenUnit,
)
from .strategies import (
    EncoderTokenStrategy,
    LLMOperationStrategy,
    LLMTokenStrategy,
    ProjectorTokenStrategy,
)
from .multimodal_registry import builtin_multimodal_compression_contracts
from .support_report import (
    SupportMatrixRow,
    builtin_imagellm_rows,
    render_support_matrix_markdown,
    support_matrix_rows,
)

__all__ = [
    "CompressionMethodDescriptor",
    "CompressionMethodRegistry",
    "AlgorithmDescriptor",
    "BackbonePlacementBinding",
    "BindingTarget",
    "CompressionContractRegistry",
    "CouplingPolicy",
    "EncoderTokenStrategy",
    "ExecutorFamily",
    "LLMOperationStrategy",
    "LLMTokenStrategy",
    "MediaAlignment",
    "ModalityVariantDescriptor",
    "ProjectorTokenStrategy",
    "SupportMatrixRow",
    "TokenUnit",
    "builtin_imagellm_rows",
    "builtin_multimodal_compression_contracts",
    "builtin_compression_methods",
    "get_compression_method_descriptor",
    "render_support_matrix_markdown",
    "register_compression_method",
    "support_matrix_rows",
]
