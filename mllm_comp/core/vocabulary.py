"""Canonical vocabulary shared by configs, plans, validators, and CLI choices.

Keeping these values in one module prevents a new placement or method from
requiring coordinated edits across dataclasses, validators, and parsers.
"""

from __future__ import annotations

from enum import Enum
from typing import TypeVar


class StringEnum(str, Enum):
    """Python 3.10-compatible string enum."""

    def __str__(self) -> str:
        return self.value


class Placement(StringEnum):
    INPUT_LEVEL = "input_level"
    ENCODER_LEVEL = "encoder_level"
    PROJECTOR_LEVEL = "projector_level"
    LLM_LEVEL = "llm_level"


class ExecutionPhase(StringEnum):
    PREPARE = "prepare"
    FORWARD = "forward"
    GENERATE = "generate"


class CompressionMode(StringEnum):
    NONE = "none"
    COMPRESSED = "compressed"


class RuntimeMode(StringEnum):
    DEFAULT = "default"
    PROFILE = "profile"
    DEBUG = "debug"


class AnalysisPhase(StringEnum):
    PRE_COMPRESSION = "pre_compression"
    POST_COMPRESSION = "post_compression"


class EvalTarget(StringEnum):
    BASELINE = "baseline"
    COMPRESSED = "compressed"


class DistributedLauncher(StringEnum):
    ACCELERATE = "accelerate"
    TORCHRUN = "torchrun"


class CompressionMethod(StringEnum):
    TOKEN_COMPRESSION = "token_compression"
    OPERATION_COMPRESSION = "operation_compression"
    QUANTIZATION = "quantization"


class ObjectFamily(StringEnum):
    TENSOR_SLOT = "tensor_slot"
    MODULE_UNIT = "module_unit"
    WEIGHT = "weight"
    WEIGHT_ACTIVATION = "weight_activation"


class Modality(StringEnum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    SCENE_3D = "scene_3d"


class ModalityCouplingPolicy(StringEnum):
    """How multiple target modality streams are transformed together."""

    INDEPENDENT = "independent"
    SYNCHRONIZED = "synchronized"
    JOINT = "joint"


class MediaAlignment(StringEnum):
    """Alignment information consumed or preserved by a compression stage."""

    NONE = "none"
    TOKEN = "token"
    SPATIAL = "spatial"
    TEMPORAL = "temporal"
    SPATIOTEMPORAL = "spatiotemporal"


class CompressionTokenUnit(StringEnum):
    """Semantic unit transformed by a token-compression algorithm."""

    ENCODER_TOKEN = "encoder_token"
    FEATURE_TOKEN = "feature_token"
    LLM_TOKEN = "llm_token"
    MEDIA_UNIT = "media_unit"


class AlignmentCouplingPolicy(StringEnum):
    """How an alignment change propagates between named media streams."""

    INDEPENDENT = "independent"
    PRESERVE = "preserve"
    CASCADE = "cascade"
    LOCKSTEP = "lockstep"


class TokenRole(StringEnum):
    INSTRUCTION = "instruction"
    OBSERVATION = "observation"
    ANSWER = "answer"


class TokenOperation(StringEnum):
    KEEP = "keep"
    DROP = "drop"
    MERGE = "merge"
    POOL = "pool"
    REORDER = "reorder"
    REWEIGHT = "reweight"


class OperationKind(StringEnum):
    SKIP = "skip"
    MASK = "mask"
    PRUNE = "prune"
    REPLACE = "replace"


class OperationApplyMode(StringEnum):
    RUNTIME_GATE = "runtime_gate"
    STATIC_MASK = "static_mask"
    STRUCTURAL_REWRITE = "structural_rewrite"


class ShapeEffect(StringEnum):
    PRESERVE_SHAPE = "preserve_shape"
    PRESERVE_IO_SHAPE = "preserve_io_shape"
    REMOVE_MODULE = "remove_module"
    CHANGE_NUM_HEADS = "change_num_heads"
    CHANGE_HIDDEN_CHANNELS = "change_hidden_channels"
    CHANGE_NUM_EXPERTS = "change_num_experts"


class QuantizationFormat(StringEnum):
    W4A16 = "W4A16"
    W8A16 = "W8A16"
    W8A8 = "W8A8"
    FP8 = "FP8"


_EnumT = TypeVar("_EnumT", bound=StringEnum)


def enum_values(enum_type: type[_EnumT]) -> tuple[str, ...]:
    """Return stable string values for argparse and validation messages."""

    return tuple(member.value for member in enum_type)
