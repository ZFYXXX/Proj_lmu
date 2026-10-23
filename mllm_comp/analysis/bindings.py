"""Minimal semantic binding from BackboneFacade handles to analysis concepts."""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Mapping, Sequence

from mllm_comp.core.errors import BackendIntegrationError


_SUPPORTED_BACKBONES = {
    "llava_onevision",
    "qwen25_vl",
    "qwen25_omni",
    "qwen3_vl",
}
_LAYER_PATTERN = re.compile(r"^llm\.layer\[(\d+)\]$")


@dataclass(frozen=True)
class AttentionGeometry:
    num_query_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int

    @property
    def query_heads_per_key_value_head(self) -> int:
        return self.num_query_heads // self.num_key_value_heads


@dataclass(frozen=True)
class DecoderLayerBinding:
    index: int
    layer: Any
    attention: Any
    ffn: Any
    attention_value_projection: Any | None = None
    attention_output_projection: Any | None = None
    ffn_down_projection: Any | None = None


@dataclass(frozen=True)
class AnalysisBackboneBinding:
    backbone_name: str
    layers: tuple[DecoderLayerBinding, ...]
    final_norm: Any
    lm_head: Any
    attention_geometry: AttentionGeometry
    input_embeddings: Any | None = None
    deepstack_projectors: tuple[Any, ...] = ()
    evidence: Mapping[str, Any] = field(default_factory=dict)


def resolve_analysis_binding(backbone: Any, model: Any) -> AnalysisBackboneBinding:
    """Resolve common decoder semantics or fail before probe collection."""

    name = str(backbone.capabilities().name)
    if name not in _SUPPORTED_BACKBONES:
        raise BackendIntegrationError(
            f"Backbone {name!r} has no activated analysis semantic binding."
        )
    handles = backbone.model_adapter.structural_handles(model)
    if not isinstance(handles, Mapping):
        raise BackendIntegrationError(
            f"Backbone {name!r} structural_handles() must return a mapping."
        )
    if name == "qwen25_omni" and (
        handles.get("thinker") is None
        or handles.get("thinker.llm") is None
        or handles.get("thinker.llm") is not handles.get("llm")
    ):
        raise BackendIntegrationError(
            "Qwen2.5-Omni analysis must bind llm semantics to Thinker."
        )

    indexed_layers = sorted(
        (
            (int(match.group(1)), value)
            for path, value in handles.items()
            if (match := _LAYER_PATTERN.match(str(path))) is not None
        ),
        key=lambda item: item[0],
    )
    if not indexed_layers:
        raise BackendIntegrationError(
            f"Backbone {name!r} exposes no llm.layer[i] structural handles."
        )
    indices = [index for index, _ in indexed_layers]
    if indices != list(range(len(indices))):
        raise BackendIntegrationError(
            f"Backbone {name!r} decoder layer handles are not contiguous: {indices}."
        )

    layers: list[DecoderLayerBinding] = []
    for index, layer in indexed_layers:
        attention = handles.get(f"llm.layer[{index}].attn")
        ffn = handles.get(f"llm.layer[{index}].ffn")
        if attention is None or ffn is None:
            raise BackendIntegrationError(
                f"Backbone {name!r} layer {index} lacks Attention or FFN handles."
            )
        layers.append(
            DecoderLayerBinding(
                index=index,
                layer=layer,
                attention=attention,
                ffn=ffn,
                attention_value_projection=handles.get(
                    f"llm.layer[{index}].attn.v_proj"
                ),
                attention_output_projection=handles.get(
                    f"llm.layer[{index}].attn.o_proj"
                ),
                ffn_down_projection=handles.get(
                    f"llm.layer[{index}].ffn.down_proj"
                ),
            )
        )

    final_norm = handles.get("llm.final_norm")
    lm_head = handles.get("lm_head")
    if final_norm is None or lm_head is None:
        raise BackendIntegrationError(
            f"Backbone {name!r} analysis requires llm.final_norm and lm_head handles."
        )
    geometry = _attention_geometry(layers[0].attention, handles.get("llm"), name)
    deepstack = tuple(handles.get("vision_deepstack_projectors", ()) or ())
    evidence: dict[str, Any] = {
        "binding_version": "analysis_semantic_binding_v1",
        "decoder_layer_count": len(layers),
        "layer_paths": [f"llm.layer[{item.index}]" for item in layers],
        "final_norm_path": "llm.final_norm",
        "lm_head_path": "lm_head",
        "input_embeddings_path": (
            "llm.input_embeddings"
            if handles.get("llm.input_embeddings") is not None
            else None
        ),
        "taylor_projection_layer_count": sum(
            1
            for layer in layers
            if layer.attention_value_projection is not None
            and layer.attention_output_projection is not None
            and layer.ffn_down_projection is not None
        ),
        "deepstack_projector_count": len(deepstack),
        "semantic_scope": (
            "thinker.llm" if name == "qwen25_omni" else "language_model"
        ),
        "excluded_branches": (
            ["audio_encoder", "talker", "token2wav"]
            if name == "qwen25_omni"
            else []
        ),
    }
    model_source = getattr(model, "_mllm_comp_source_evidence", None)
    if isinstance(model_source, Mapping):
        evidence["model_source"] = dict(model_source)
    return AnalysisBackboneBinding(
        backbone_name=name,
        layers=tuple(layers),
        final_norm=final_norm,
        lm_head=lm_head,
        attention_geometry=geometry,
        input_embeddings=handles.get("llm.input_embeddings"),
        deepstack_projectors=deepstack,
        evidence=evidence,
    )


def require_taylor_semantics(
    binding: AnalysisBackboneBinding,
    layers: tuple[DecoderLayerBinding, ...],
) -> None:
    """Fail before collection unless every selected Taylor hook is explicit."""

    if binding.input_embeddings is None:
        raise BackendIntegrationError(
            f"Backbone {binding.backbone_name!r} does not expose "
            "llm.input_embeddings for frozen-model Taylor autograd."
        )
    missing: list[str] = []
    for layer in layers:
        if layer.attention_value_projection is None:
            missing.append(f"llm.layer[{layer.index}].attn.v_proj")
        if layer.attention_output_projection is None:
            missing.append(f"llm.layer[{layer.index}].attn.o_proj")
        if layer.ffn_down_projection is None:
            missing.append(f"llm.layer[{layer.index}].ffn.down_proj")
    if missing:
        raise BackendIntegrationError(
            f"Backbone {binding.backbone_name!r} lacks Taylor semantic handles: "
            + ", ".join(missing)
            + "."
        )


def select_decoder_layers(
    binding: AnalysisBackboneBinding,
    configured: Any,
    *,
    context: str,
) -> tuple[DecoderLayerBinding, ...]:
    """Resolve an ordered, de-duplicated decoder-layer selection."""

    if configured == "all":
        return binding.layers
    requested = list(dict.fromkeys(int(value) for value in configured))
    if not requested:
        raise BackendIntegrationError(f"{context} layer selection cannot be empty.")
    available = {layer.index: layer for layer in binding.layers}
    missing = sorted(set(requested) - set(available))
    if missing:
        raise BackendIntegrationError(
            f"{context} requested unavailable decoder layers: {missing}."
        )
    return tuple(available[index] for index in requested)


def require_eager_attention(
    layers: Sequence[DecoderLayerBinding],
    *,
    context: str,
) -> dict[str, str]:
    """Require exact attention weights from every selected decoder layer."""

    result: dict[str, str] = {}
    for layer in layers:
        backend = getattr(
            getattr(layer.attention, "config", None),
            "_attn_implementation",
            None,
        )
        normalized = str(backend or "unknown")
        result[str(layer.index)] = normalized
        if normalized != "eager":
            raise BackendIntegrationError(
                f"{context} requires exact eager attention weights; layer "
                f"{layer.index} reports {normalized!r}. Configure "
                "attn_implementation=eager."
            )
    return result


def layer_batches(
    layers: Sequence[Any],
    batch_size: int,
) -> tuple[tuple[Any, ...], ...]:
    """Partition selected layers into bounded capture batches."""

    if batch_size < 1:
        raise ValueError("Layer batch size must be positive.")
    return tuple(
        tuple(layers[start : start + batch_size])
        for start in range(0, len(layers), batch_size)
    )


def _attention_geometry(
    attention: Any,
    language_model: Any,
    backbone_name: str,
) -> AttentionGeometry:
    config = getattr(attention, "config", None)
    if config is None:
        config = getattr(language_model, "config", None)

    num_query_heads = _positive_int(
        getattr(attention, "num_heads", None),
        getattr(attention, "num_attention_heads", None),
        getattr(config, "num_attention_heads", None),
    )
    num_key_value_heads = _positive_int(
        getattr(attention, "num_key_value_heads", None),
        getattr(config, "num_key_value_heads", None),
    )
    hidden_size = _positive_int(
        getattr(config, "hidden_size", None),
        getattr(getattr(attention, "q_proj", None), "in_features", None),
    )
    head_dim = _positive_int(
        getattr(attention, "head_dim", None),
        getattr(config, "head_dim", None),
    )
    if head_dim is None and hidden_size is not None and num_query_heads is not None:
        if hidden_size % num_query_heads == 0:
            head_dim = hidden_size // num_query_heads
    if hidden_size is None and head_dim is not None and num_query_heads is not None:
        hidden_size = head_dim * num_query_heads
    if num_key_value_heads is None and num_query_heads is not None:
        # Standard MHA has one KV head per query head. GQA/MQA providers expose
        # num_key_value_heads explicitly and therefore never take this fallback.
        num_key_value_heads = num_query_heads

    values = (num_query_heads, num_key_value_heads, head_dim, hidden_size)
    if any(value is None for value in values):
        raise BackendIntegrationError(
            f"Backbone {backbone_name!r} could not resolve Attention head geometry."
        )
    assert num_query_heads is not None
    assert num_key_value_heads is not None
    assert head_dim is not None
    assert hidden_size is not None
    if num_query_heads % num_key_value_heads:
        raise BackendIntegrationError(
            f"Backbone {backbone_name!r} query heads ({num_query_heads}) are not "
            f"divisible by KV heads ({num_key_value_heads})."
        )
    if head_dim * num_query_heads != hidden_size:
        raise BackendIntegrationError(
            f"Backbone {backbone_name!r} Attention geometry is inconsistent: "
            f"heads={num_query_heads}, head_dim={head_dim}, hidden={hidden_size}."
        )
    return AttentionGeometry(
        num_query_heads=num_query_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        hidden_size=hidden_size,
    )


def _positive_int(*values: Any) -> int | None:
    for value in values:
        if value is None:
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


__all__ = [
    "AnalysisBackboneBinding",
    "AttentionGeometry",
    "DecoderLayerBinding",
    "layer_batches",
    "require_eager_attention",
    "require_taylor_semantics",
    "resolve_analysis_binding",
    "select_decoder_layers",
]
