"""Stride-selected video tokens with a recent-visual sparse decode KV view.

This changes both DyCoKe decisions while reusing its native Projector/cache
lifecycle. It does not prune physical KV storage or change text/decode keys.
"""

from dataclasses import asdict, dataclass, replace

from mllm_comp.compressors.contracts import AlgorithmDescriptor, CompressionContractRegistry
from mllm_comp.compressors.extensions import register_method
from mllm_comp.compressors.video.dycoke import DyCoKeKVViewPlan
from mllm_comp.compressors.video.positions import resolve_video_position
from mllm_comp.core.vocabulary import CompressionMethod, Placement, TokenOperation

from .stride_video import StrideVideoSpec, replay_stride_video_features, stride_video_tokens


ALGORITHM = "researcher_stride_recent_video"
VARIANT = f"{ALGORITHM}.video"


@dataclass(frozen=True)
class StrideRecentSpec:
    stride: int = 4
    offset: int = 1
    kv_keep_tokens: int = 2
    decoder_prune_layer: int = 3

    def __post_init__(self):
        StrideVideoSpec(self.stride, self.offset)
        for name in ("kv_keep_tokens", "decoder_prune_layer"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")


def resolve_config(config):
    return asdict(StrideRecentSpec(**dict(config)))


class RecentVisualKVPolicy:
    def __init__(self, keep_tokens):
        self.keep_tokens = keep_tokens

    def plan_view(self, query, full_keys, *, visual_positions, sequence_length, attention_scaling):
        # Native cache positions preserve source order, so the suffix contains
        # the most recent retained video observations. No attention history is used.
        kept = tuple(visual_positions[-self.keep_tokens:])
        dropped = set(visual_positions) - set(kept)
        return DyCoKeKVViewPlan(
            sequence_length=sequence_length, visual_positions=tuple(visual_positions),
            kept_visual_positions=kept,
            view_indices=tuple(index for index in range(sequence_length) if index not in dropped),
            attention_similarity=None, reused_visual_selection=False,
        )


class StrideRecentVideoMethod:
    def __call__(self, payload, *, config, rules):
        result = stride_video_tokens(payload,
            config={key: config[key] for key in ("stride", "offset")}, rules=rules)
        return replace(result, metadata={**result.metadata, "algorithm": ALGORITHM,
            "variant": VARIANT, "decode_selection": "most_recent_visual_positions"})

    def replay(self, features, result, *, config, rules):
        return replay_stride_video_features(features, result, config=config, rules=rules)

    def new_decode_policy(self, *, config, rules):
        return RecentVisualKVPolicy(config["kv_keep_tokens"])

    def runtime_metadata(self):
        return {"training_free_compressor": True,
                "decode_selection": "most_recent_visual_positions",
                "attention_similarity_computed": False}


def register(registry: CompressionContractRegistry):
    register_method(
        registry,
        algorithm=AlgorithmDescriptor(
            name=ALGORITHM, version="1", method=CompressionMethod.TOKEN_COMPRESSION,
            kernel=StrideRecentVideoMethod(),
            supported_placements=frozenset({Placement.PROJECTOR_LEVEL, Placement.LLM_LEVEL}),
            operations=frozenset({TokenOperation.DROP}),
        ),
        variant=VARIANT,
        reuse_bindings=[f"dycoke.video.{backbone}" for backbone in
                       ("llava_onevision", "qwen25_vl", "qwen3_vl")],
        strategy_factory=StrideRecentVideoMethod,
        config_defaults=resolve_config({}), config_validator=resolve_config,
        position_resolver=resolve_video_position,
    )
