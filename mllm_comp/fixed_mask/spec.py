"""Portable selections; indices always identify components to disable."""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping

KINDS = ("heads", "v2v", "t2v", "t2t", "ffn_all", "ffn_vision", "ffn_text")


def normalize_spec(value):
    if not isinstance(value, Mapping):
        raise ValueError("fixed_mask config must be a mapping")
    unknown = set(value) - {*KINDS, "schema_version", "mask_scope", "text_scope", "provenance"}
    if unknown:
        raise ValueError(f"Unknown fixed_mask fields: {sorted(unknown)}")
    if value.get("schema_version", 1) != 1:
        raise ValueError("fixed_mask requires schema_version=1")
    if value.get("mask_scope", "prefill_only") != "prefill_only":
        raise ValueError("fixed_mask currently supports mask_scope=prefill_only")
    # Path text excludes the prefix and inter-image text. FFN text includes all
    # non-image, non-padding tokens, matching the source experiments.
    if value.get("text_scope", "post_image") != "post_image":
        raise ValueError("fixed_mask paths require text_scope=post_image")
    result = {"schema_version": 1, "mask_scope": "prefill_only", "text_scope": "post_image"}
    for kind in KINDS:
        entries = value.get(kind, {})
        if not isinstance(entries, Mapping):
            raise ValueError(f"{kind} must map layer indices to disabled component indices")
        normalized = {}
        for layer, indices in entries.items():
            if isinstance(layer, bool) or not str(layer).isdigit():
                raise ValueError(f"Invalid {kind} layer {layer!r}")
            key = str(int(layer))
            if key in normalized or not isinstance(indices, (list, tuple)):
                raise ValueError(f"Invalid/duplicate selection for {kind} layer {layer}")
            if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in indices):
                raise ValueError(f"{kind}[{layer}] requires nonnegative integer indices")
            if len(set(indices)) != len(indices):
                raise ValueError(f"{kind}[{layer}] contains duplicate indices")
            if indices:
                normalized[key] = sorted(indices)
        result[kind] = dict(sorted(normalized.items(), key=lambda item: int(item[0])))
    if not isinstance(value.get("provenance", {}), Mapping):
        raise ValueError("fixed_mask provenance must be a mapping")
    result["provenance"] = deepcopy(dict(value.get("provenance", {})))
    return result


def validate_geometry(spec, layers):
    for kind in KINDS:
        for index, selected in spec[kind].items():
            index = int(index)
            if index >= len(layers):
                raise ValueError(f"{kind} layer {index} exceeds decoder depth {len(layers)}")
            layer = layers[index]
            width = (layer.mlp.down_proj.in_features if kind.startswith("ffn_")
                     else layer.self_attn.config.num_attention_heads)
            if selected and max(selected) >= width:
                raise ValueError(f"{kind}[{index}] exceeds component count {width}")
