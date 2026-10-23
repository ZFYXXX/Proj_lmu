"""An independent image algorithm installed through existing decoder bindings."""

from dataclasses import dataclass
from typing import Any, Mapping

from mllm_comp.compressors.contracts import AlgorithmDescriptor, CompressionContractRegistry
from mllm_comp.compressors.extensions import register_method
from mllm_comp.compressors.image.decoder_lifecycle import CacheRepairPolicy
from mllm_comp.compressors.image.decoder_selection import DecoderSelectionCall
from mllm_comp.compressors.image.token_selection import ImageTokenSelection
from mllm_comp.compressors.layer_positions import LayerPosition
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation


ALGORITHM = "researcher_alternating_image"
VARIANT = f"{ALGORITHM}.image"
BACKBONES = ("qwen25_vl", "qwen3_vl", "llava_onevision")
TARGET_SITE = "llm_decoder_pre_prune_layer_image_tokens"


@dataclass(frozen=True)
class AlternatingImageSpec:
    stride: int = 2
    offset: int = 1
    apply_before_layer: int = 2

    def __post_init__(self):
        for name in ("stride", "offset", "apply_before_layer"):
            if isinstance(getattr(self, name), bool) or not isinstance(getattr(self, name), int):
                raise ValueError(f"{name} must be an integer.")
        if self.stride < 1 or not 0 <= self.offset < self.stride or self.apply_before_layer < 0:
            raise ValueError("Require stride >= 1, 0 <= offset < stride and apply_before_layer >= 0.")


def resolve_config(config: Mapping[str, Any]) -> dict[str, Any]:
    spec = AlternatingImageSpec(**dict(config))
    return {"stride": spec.stride, "offset": spec.offset, "apply_before_layer": spec.apply_before_layer}


def alternating_image_tokens(
    hidden_states, *, image_indices, spec: AlternatingImageSpec,
    protected_indices=(), **_layout,
):
    """Sample visual content positions; structural and text tokens always survive.

    This method has no attention scorer. Protected image tokens, such as
    LLaVA row newlines, do not consume content ordinals in the stride rule.
    """
    protected = set(int(index) for index in protected_indices)
    content = [index for index in image_indices.tolist() if index not in protected]
    selected = content[spec.offset::spec.stride]
    if content and not selected:
        selected = content[:1]
    return ImageTokenSelection(
        kept_image_indices=selected,
        metadata={"algorithm": ALGORITHM, "variant": VARIANT,
            "selection": "content_ordinal_stride", "stride": spec.stride, "offset": spec.offset},
    )


class AlternatingImageMethod:
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    score_layers = ()
    position_policy = "sparse_source"
    single_image_only = True
    target_site = TARGET_SITE

    def __init__(self, *, stage, config, rules, backbone):
        if rules:
            raise ValueError("This example has no configurable scorer or selector hooks.")
        self.algorithm = stage.algorithm
        self.spec = AlternatingImageSpec(**dict(config))
        self.positions = (LayerPosition("inter_layer", self.spec.apply_before_layer),)

    def resolve_depth(self, num_layers):
        if self.spec.apply_before_layer > num_layers:
            raise ValueError("apply_before_layer is outside this decoder stack.")

    def new_call(self):
        return _AlternatingImageCall(self)


class _AlternatingImageCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        if position not in self.method.positions:
            raise ValueError("Selection ran at an unregistered application position.")
        return alternating_image_tokens(
            hidden_states, image_indices=image_indices, spec=self.method.spec, **kwargs,
        )


def register(registry: CompressionContractRegistry) -> None:
    register_method(
        registry,
        algorithm=AlgorithmDescriptor(
            name=ALGORITHM, version="1", method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=alternating_image_tokens, supported_placements=frozenset({Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.DROP}),
            metadata={"selection": "content_ordinal_stride"},
        ),
        variant=VARIANT,
        reuse_bindings=[f"fastv.image.{backbone}" for backbone in BACKBONES],
        strategy_factory=AlternatingImageMethod,
        config_defaults=resolve_config({}), config_validator=resolve_config,
    )


def experiment_config(backbone):
    return experiment_config_from_dict({
        "backbone": {"name": backbone, "model_id": "native-tiny-image"},
        "pipeline": {
            "extensions": ["examples.compression_extensions.image_selection:register"],
            "stages": [{"name": "alternate-content", "algorithm": ALGORITHM,
                "execution_phase": "forward", "config": {"apply_before_layer": 2}}],
        },
        "train": {"backend": "ms_swift", "compression_mode": "compressed"},
    })
