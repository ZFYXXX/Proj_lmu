"""Method evidence checks shared by video diagnostics across backbones."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any


CANONICAL_PRUNEVID_CONFIG = {
    "tau": 0.8,
    "cluster_ratio": 0.5,
    "temporal_ratio": 0.25,
    "k": 7,
    "min_tokens_for_cluster": 14,
    "enable_vtp": True,
    "vtp_selected_layer": 10,
    "vtp_alpha": 0.4,
}


def prunevid_cache_layers_exact(
    cache: Mapping[str, Any],
    *,
    source_length: int,
    result_length: int,
) -> bool:
    touched = tuple(int(item) for item in cache["touched_layers"])
    before = {int(item["layer_index"]): item for item in cache.get("layers_before", ())}
    after = {int(item["layer_index"]): item for item in cache.get("layers_after", ())}
    expected = tuple(range(CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"] + 1))
    return (
        cache.get("policy") == "compact_prefix"
        and int(cache.get("boundary_layer", -1))
        == CANONICAL_PRUNEVID_CONFIG["vtp_selected_layer"]
        and cache.get("boundary") == "after_layer"
        and int(cache.get("source_length", -1)) == source_length
        and int(cache.get("result_length", -1)) == result_length
        and touched == expected
        and cache.get("cache_type") == "DynamicCache"
        and all(
            int(before[index]["logical_length"]) == source_length
            and int(before[index]["physical_length"]) == source_length
            and int(after[index]["logical_length"]) == result_length
            and int(after[index]["physical_length"]) == result_length
            for index in expected
        )
    )


def dycoke_forward_runtime_checks(metadata: Mapping[str, Any]) -> dict[str, bool]:
    return {
        "forward_phase_exact": metadata.get("execution_phase") == "forward",
        "decode_view_not_invoked": (
            int(metadata.get("decode_view_invocations", -1)) == 0
            and not tuple(metadata.get("decode_kv_views", ()))
        ),
        "decode_cache_controller_not_installed": int(
            metadata.get("live_cache_controllers", -1)
        )
        == 0,
        "generation_layer_contract_unchanged": (
            int(metadata.get("decode_trigger_layer", -1)) == 2
            and int(metadata.get("sparse_view_first_layer", -1)) == 3
            and metadata.get("single_token_decode_only") is True
            and metadata.get("physical_cache_retained") is True
        ),
        "ttm_gradients_declared": (
            metadata.get("training_free_compressor") is True
            and metadata.get("training_gradients_preserved") is True
        ),
    }


def strict_timing_matches(source: float | None, recorded: Any) -> bool:
    if source is None:
        return recorded is None
    try:
        source_value = float(source)
        recorded_value = float(recorded)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(source_value)
        and source_value > 0
        and math.isfinite(recorded_value)
        and math.isclose(
            recorded_value,
            source_value,
            rel_tol=0.0,
            abs_tol=1e-9,
        )
    )
