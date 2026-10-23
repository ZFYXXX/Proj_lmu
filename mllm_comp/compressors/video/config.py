"""Canonical video rules and runtime signals with configurable decoder layers.

The primary algorithm rules remain fixed; decoder source layers may vary.
Validation-profile inputs such as frame caps, processor pixel limits,
generation length, and optimizer settings intentionally do not belong here.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from types import MappingProxyType
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, ConfigError

from .contracts import VideoTokenPayload


VIDEO_ALGORITHMS = (
    "metok",
    "prunevid",
    "visionzip",
    "fastvid",
    "vqtoken",
    "dytok",
    "longvu",
    "llamavid",
    "dycoke",
)
VIDEO_BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
VIDEO_REFERENCE_REVISION = "7e3ae876bc6306a4179f389e5f960225cd9e3a79"


_CANONICAL_CONFIGS_RAW: Mapping[str, Mapping[str, Any]] = {
    "metok": {
        "segmentation_method": "similarity",
        "num_segments": 4,
        "keyframe_method": "similarity",
        "keyframe_ratio": 0.4,
        "top_k_events": 3,
        "distinguish_important_events": True,
        "s1_important": 2,
        "s2_important": 4,
        "s1_normal": 4,
        "s2_normal": 8,
        "pool_mode": "bilinear",
        "non_keyframe_method": "adaptive_pool",
        "keyframe_pool_enable": False,
        "inside_compression": True,
        "pruning_layers": (3, 10, 19),
        "ratio": 0.76,
    },
    "prunevid": {
        "tau": 0.8,
        "cluster_ratio": 0.5,
        "temporal_ratio": 0.25,
        "k": 7,
        "min_tokens_for_cluster": 14,
        "enable_vtp": True,
        "vtp_selected_layer": 10,
        "vtp_alpha": 0.4,
    },
    "visionzip": {
        "dominant_num": 126,
        "contextual_num": 21,
        "attn_layer": -2,
        "hidden_state_layer": -2,
    },
    "fastvid": {
        "retention_ratio": 0.10,
        "dyseg_c": 8,
        "dyseg_tau": 0.9,
        "stprune_d": 0.4,
        "dtm_p": 4,
        "dtm_beta": 0.6,
    },
    "vqtoken": {
        "num_clusters": 32,
        "adaptive": False,
        "use_cross_attention": False,
    },
    "dytok": {
        "mode": "dynamic",
        "base_compressor": "visionzip",
        "base_config": {
            "dominant_num": 126,
            "contextual_num": 21,
            "attn_layer": -2,
            "hidden_state_layer": -2,
        },
        "llm_config": {"target_layers": tuple(range(16, 24))},
        "upper_limit": 196,
    },
    "longvu": {
        "window_size": 16,
        "threshold": 0.83,
        "max_frames": 400,
        "min_frames": 1,
        "dino_interp_size": 576,
        "vision_hidden_size": 1024,
        "query_num": 144,
        "connector_depth": 3,
    },
    "llamavid": {
        "num_query": 32,
        "bert_type": "qformer",
        "compress_type": "mean",
    },
    "dycoke": {
        "merging_ratio": 0.7,
        "decoder_prune_layer": 3,
        "kv_drop_ratio": 0.8,
        "attention_similarity_threshold": 0.9,
    },
}


_REQUIRED_RUNTIME_SIGNALS_RAW: Mapping[str, frozenset[str]] = {
    "metok": frozenset(
        {
            "projected_features",
            "raw_features",
            "video_boundaries",
            "coordinates",
            "text_context",
            "siglip_text_encoder",
            "prefill_decoder_attention",
            "prefill_layout_state",
        }
    ),
    "prunevid": frozenset(
        {
            "projected_features",
            "video_boundaries",
            "coordinates",
            "frame_grid_boundaries",
            "prefill_decoder_attention",
            "visual_span",
        }
    ),
    "visionzip": frozenset(
        {
            "projected_features",
            "video_boundaries",
            "coordinates",
            "vision_encoder_attention",
            "vision_encoder_hidden_states",
        }
    ),
    "fastvid": frozenset(
        {
            "projected_features",
            "raw_features",
            "video_boundaries",
            "coordinates",
            "frame_global_features",
            "token_saliency_attention",
        }
    ),
    "vqtoken": frozenset(
        {
            "projected_features",
            "video_boundaries",
            "coordinates",
        }
    ),
    "dytok": frozenset(
        {
            "projected_features",
            "video_boundaries",
            "coordinates",
            "vision_encoder_attention",
            "vision_encoder_hidden_states",
            "prompt_layout",
            "detection_attention",
        }
    ),
    "longvu": frozenset(
        {
            "projected_features",
            "raw_features",
            "video_boundaries",
            "coordinates",
            "pixels",
            "dinov2_features",
        }
    ),
    "llamavid": frozenset(
        {
            "projected_features",
            "raw_features",
            "video_boundaries",
            "coordinates",
            "text_context",
            "bert_qformer",
            "projector_dimensions",
        }
    ),
    "dycoke": frozenset(
        {
            "projected_features",
            "video_boundaries",
            "coordinates",
            "visual_span",
            "autoregressive_cache",
            "decode_attention",
        }
    ),
}


def resolve_canonical_video_config(
    algorithm_or_variant: str,
    overrides: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Return immutable rules, allowing the existing decoder layer selectors.

    ``overrides`` may omit canonical fields and may express tuple fields as
    YAML-style lists. Outside METok/PruneVid decoder layer selectors, supplied
    values must equal the canonical recipe; unknown fields are rejected.
    """

    algorithm = _algorithm_name(algorithm_or_variant)
    if overrides is None:
        overrides = {}
    if not isinstance(overrides, Mapping):
        raise ConfigError("video algorithm config must be a mapping")
    expected = CANONICAL_VIDEO_CONFIGS[algorithm]
    layer_key = {"metok": "pruning_layers", "prunevid": "vtp_selected_layer"}.get(algorithm)
    if layer_key in overrides:
        supplied = overrides[layer_key]
        layers = supplied if algorithm == "metok" else (supplied,)
        if (not isinstance(layers, (tuple, list)) or len(layers) != (3 if algorithm == "metok" else 1)
                or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
                or tuple(layers) != tuple(sorted(set(layers)))):
            raise ConfigError(f"{algorithm}.{layer_key} requires {'three increasing' if algorithm == 'metok' else 'one'} non-negative decoder layer indices")
        expected = {**expected, layer_key: tuple(layers) if algorithm == "metok" else layers[0]}
    return _resolve_canonical_value(
        expected,
        overrides,
        path=f"{algorithm}.video.config",
    )


def canonical_video_config_record(algorithm_or_variant: str) -> dict[str, Any]:
    """Return a plain serializable copy for metadata and evidence records."""

    return _thaw(resolve_canonical_video_config(algorithm_or_variant))


def video_required_runtime_signals(
    algorithm_or_variant: str,
) -> frozenset[str]:
    return VIDEO_REQUIRED_RUNTIME_SIGNALS[_algorithm_name(algorithm_or_variant)]


def validate_video_runtime_signals(
    algorithm_or_variant: str,
    available: Mapping[str, Any] | Iterable[str],
) -> frozenset[str]:
    """Require every semantic provider signal for one canonical method."""

    algorithm = _algorithm_name(algorithm_or_variant)
    if isinstance(available, Mapping):
        names = frozenset(
            name
            for name, value in available.items()
            if isinstance(name, str) and name and value is not None
        )
    else:
        if isinstance(available, (str, bytes)):
            raise TypeError("available runtime signals must be names, not a string")
        names = frozenset(available)
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("runtime signal names must be non-empty strings")
    missing = video_required_runtime_signals(algorithm) - names
    if missing:
        raise BackendIntegrationError(
            f"{algorithm}.video requires missing runtime signals "
            f"{sorted(missing)}"
        )
    return names


def validate_video_payload_signals(
    algorithm_or_variant: str,
    payload: VideoTokenPayload,
) -> frozenset[str]:
    """Validate one shared payload plus its provider-owned signal mapping."""

    if not isinstance(payload, VideoTokenPayload):
        raise TypeError("payload must be a VideoTokenPayload")
    available: dict[str, Any] = {
        "projected_features": payload.features,
        "video_boundaries": payload.video_offsets,
        "coordinates": payload.coordinates,
        **dict(payload.signals),
    }
    if payload.raw_features is not None:
        available["raw_features"] = payload.raw_features
    return validate_video_runtime_signals(algorithm_or_variant, available)


def _algorithm_name(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError("video algorithm or variant name must be non-empty")
    name = value.strip()
    if name.endswith(".video"):
        name = name[: -len(".video")]
    if name not in VIDEO_ALGORITHMS:
        raise ConfigError(f"unknown canonical video algorithm {value!r}")
    return name


def _resolve_canonical_value(expected: Any, supplied: Any, *, path: str) -> Any:
    if isinstance(expected, Mapping):
        if not isinstance(supplied, Mapping):
            raise ConfigError(f"{path} must be a mapping")
        unknown = sorted(set(supplied) - set(expected))
        if unknown:
            raise ConfigError(f"{path} contains unknown fields {unknown}")
        return MappingProxyType(
            {
                key: _resolve_canonical_value(
                    expected_value,
                    supplied[key] if key in supplied else expected_value,
                    path=f"{path}.{key}",
                )
                for key, expected_value in expected.items()
            }
        )
    if isinstance(expected, tuple):
        if not isinstance(supplied, Sequence) or isinstance(
            supplied, (str, bytes, bytearray)
        ):
            raise ConfigError(f"{path} must be a sequence")
        values = tuple(supplied)
        if len(values) != len(expected):
            raise ConfigError(f"{path} must contain {len(expected)} entries")
        return tuple(
            _resolve_canonical_value(item, value, path=f"{path}[{index}]")
            for index, (item, value) in enumerate(zip(expected, values))
        )
    if isinstance(expected, bool):
        valid_type = isinstance(supplied, bool)
    elif isinstance(expected, int):
        valid_type = isinstance(supplied, int) and not isinstance(supplied, bool)
    elif isinstance(expected, float):
        valid_type = isinstance(supplied, (int, float)) and not isinstance(
            supplied, bool
        )
    elif expected is None:
        valid_type = supplied is None
    else:
        valid_type = type(supplied) is type(expected)
    if not valid_type or supplied != expected:
        raise ConfigError(
            f"{path}={supplied!r} is non-canonical; expected {expected!r}"
        )
    return expected


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType(
        {key: _freeze_value(item) for key, item in value.items()}
    )


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _thaw(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


CANONICAL_VIDEO_CONFIGS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        algorithm: _freeze_mapping(config)
        for algorithm, config in _CANONICAL_CONFIGS_RAW.items()
    }
)
VIDEO_REQUIRED_RUNTIME_SIGNALS: Mapping[str, frozenset[str]] = MappingProxyType(
    dict(_REQUIRED_RUNTIME_SIGNALS_RAW)
)


__all__ = [
    "CANONICAL_VIDEO_CONFIGS",
    "VIDEO_ALGORITHMS",
    "VIDEO_BACKBONES",
    "VIDEO_REFERENCE_REVISION",
    "VIDEO_REQUIRED_RUNTIME_SIGNALS",
    "canonical_video_config_record",
    "resolve_canonical_video_config",
    "validate_video_payload_signals",
    "validate_video_runtime_signals",
    "video_required_runtime_signals",
]
