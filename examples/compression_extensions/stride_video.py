"""An independent video algorithm using the shared native Projector executor."""

from dataclasses import asdict, dataclass

from mllm_comp.compressors.contracts import AlgorithmDescriptor, CompressionContractRegistry
from mllm_comp.compressors.extensions import register_method
from mllm_comp.compressors.video.contracts import VideoCompressionResult
from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation


ALGORITHM = "researcher_stride_video"
VARIANT = f"{ALGORITHM}.video.ordinal"
TARGET = "ordinal_sample"
BACKBONES = ("qwen25_vl", "qwen3_vl", "llava_onevision")


@dataclass(frozen=True)
class StrideVideoSpec:
    stride: int = 3
    offset: int = 1

    def __post_init__(self):
        if any(isinstance(value, bool) or not isinstance(value, int) for value in (self.stride, self.offset)):
            raise ValueError("stride and offset must be integers.")
        if self.stride < 1 or not 0 <= self.offset < self.stride:
            raise ValueError("Require stride >= 1 and 0 <= offset < stride.")


def resolve_config(config):
    return asdict(StrideVideoSpec(**dict(config)))


def stride_video_tokens(payload, *, config, rules):
    """Keep fixed source ordinals independently in each video's feature stream."""
    import torch

    if rules:
        raise ValueError("This example has no configurable rule hooks.")
    spec = StrideVideoSpec(**dict(config))
    outputs, groups, representatives = [], [], []
    for index in range(payload.video_count):
        source = payload.features[payload.video_span(index)]
        selected = tuple(range(spec.offset, source.shape[0], spec.stride))
        if not selected:
            raise ValueError("The selected offset leaves this video without a token.")
        keep = torch.tensor(selected, dtype=torch.long, device=source.device)
        outputs.append(source.index_select(0, keep))
        groups.append(tuple((ordinal,) for ordinal in selected))
        representatives.append(selected)
    return VideoCompressionResult.from_source_groups(
        payload, torch.cat(outputs), source_groups=groups, representative_indices=representatives,
        metadata={"algorithm": ALGORITHM, "variant": VARIANT, "selection": "source_ordinal_stride",
            "stride": spec.stride, "offset": spec.offset},
    )


def replay_stride_video_features(features, result, *, config, rules):
    """Apply the original source indices to another feature stream, without scoring."""
    import torch

    selected = [start + ordinal
        for start, ordinals in zip(result.source_video_offsets, result.representative_indices)
        for ordinal in ordinals]
    indices = torch.tensor(selected, dtype=torch.long, device=features.device)
    return features.index_select(0, indices)


class StrideVideoMethod:
    """Algorithm and branch replay; the framework owns native installation."""

    def __call__(self, payload, *, config, rules):
        return stride_video_tokens(payload, config=config, rules=rules)

    def replay(self, features, result, *, config, rules):
        return replay_stride_video_features(features, result, config=config, rules=rules)


def register(registry: CompressionContractRegistry):
    register_method(
        registry,
        algorithm=AlgorithmDescriptor(
            name=ALGORITHM, version="1", method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=stride_video_tokens, supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
            operations=frozenset({TokenOperation.DROP}), metadata={"selection": "source_ordinal_stride"},
        ),
        variant=VARIANT,
        reuse_bindings=[f"vqtoken.video.{backbone}" for backbone in BACKBONES],
        strategy_factory=StrideVideoMethod, target_name=TARGET,
        config_defaults=resolve_config({}), config_validator=resolve_config,
    )
