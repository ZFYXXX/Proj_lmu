"""Skip-Vision projector merge, adaptive summaries and FFN bypass.

The released LLaVA path orders retained tokens first, ToMe-merges redundant
tokens, places one adaptive summary on either side of the skipped group,
bypasses decoder FFNs for that group except for its last local token, and
removes only the FFN-skipped rows from KV after prefill.  SigLIP has no CLS
token, so OV uses the projected mean feature as the closest available CLS
proxy when a natural global/local split is unavailable; the base AnyRes crop
is kept as the natural global group whenever local crops are present.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .qwen2_flops import aggregate_samplewise_flops
from torch import nn


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "skip_vision",
    "attention_visual_retain_ratio": 0.90,
    "ffn_visual_retain_ratio": 0.60,
    "target_flops_retain_ratio": 0.67,
    "summary_token_count": 2,
    "keep_terminal_visual_token_ffn": True,
    "keep_last_contextual_visual_token_ffn": True,
    "terminal_visual_token_role": "latter_summary",
    "global_token_policy": "anyres_base_then_projected_distinctiveness",
    "no_cls_adaptation": "projected_mean_proxy_top_cosine",
    "contextual_merge": "tome_most_similar_distinctive_representative",
    "summary_position": "around_contextual_tokens",
    "decode_cache_policy": "remove_contextual_skip_ffn_rows",
    "similarity_chunk_size": 512,
    "image_only": True,
}


def normalize_skip_vision_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        cfg.update(raw)
    cfg["method"] = "skip_vision"
    for key in (
        "attention_visual_retain_ratio",
        "ffn_visual_retain_ratio",
        "target_flops_retain_ratio",
    ):
        cfg[key] = float(cfg[key])
    cfg["summary_token_count"] = int(cfg["summary_token_count"])
    cfg["similarity_chunk_size"] = int(cfg["similarity_chunk_size"])
    cfg["keep_terminal_visual_token_ffn"] = bool(
        cfg["keep_terminal_visual_token_ffn"]
    )
    cfg["keep_last_contextual_visual_token_ffn"] = bool(
        cfg["keep_last_contextual_visual_token_ffn"]
    )
    cfg["image_only"] = bool(cfg["image_only"])
    unknown = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError("Unknown Skip-Vision config keys: " + ", ".join(unknown))
    attention_ratio = cfg["attention_visual_retain_ratio"]
    ffn_ratio = cfg["ffn_visual_retain_ratio"]
    if not 0.0 < ffn_ratio <= attention_ratio <= 1.0:
        raise ValueError(
            "Skip-Vision requires 0 < FFN visual ratio <= attention visual ratio <= 1"
        )
    if not 0.0 < cfg["target_flops_retain_ratio"] <= 1.0:
        raise ValueError("Skip-Vision target FLOPs ratio must lie in (0,1]")
    if cfg["summary_token_count"] != 2:
        raise ValueError("The released Skip-Vision design uses exactly two summaries")
    if cfg["similarity_chunk_size"] < 1:
        raise ValueError("Skip-Vision similarity_chunk_size must be positive")
    if not cfg["keep_terminal_visual_token_ffn"]:
        raise ValueError(
            "The paper requires the latter summary token to pass through FFN"
        )
    if not cfg["keep_last_contextual_visual_token_ffn"]:
        raise ValueError(
            "The full Skip-Vision design requires the last local token to pass "
            "through FFN"
        )
    expected = {
        "global_token_policy": "anyres_base_then_projected_distinctiveness",
        "no_cls_adaptation": "projected_mean_proxy_top_cosine",
        "contextual_merge": "tome_most_similar_distinctive_representative",
        "summary_position": "around_contextual_tokens",
        "decode_cache_policy": "remove_contextual_skip_ffn_rows",
        "terminal_visual_token_role": "latter_summary",
    }
    for key, value in expected.items():
        if cfg[key] != value:
            raise ValueError(f"Unsupported Skip-Vision {key}: {cfg[key]!r}")
    return cfg


class SkipVisionAdaptiveSummary(nn.Module):
    """Official LayerNorm + linear softmax pooling shared by both groups."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.score = nn.Sequential(
            nn.LayerNorm(self.hidden_size), nn.Linear(self.hidden_size, 1)
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        if tokens.ndim != 2 or int(tokens.shape[0]) < 1:
            raise ValueError("Skip-Vision summary expects non-empty [N,D] tokens")
        weights = torch.softmax(self.score(tokens), dim=0).transpose(0, 1)
        return weights @ tokens


def ensure_skip_vision_summary(decoder: nn.Module) -> nn.Module:
    module = getattr(decoder, "skip_vision_summary", None)
    if module is None:
        module = SkipVisionAdaptiveSummary(int(decoder.config.hidden_size))
        init = getattr(decoder, "_init_weights", None)
        if callable(init):
            module.apply(init)
        decoder.add_module("skip_vision_summary", module)
    if int(getattr(module, "hidden_size", -1)) != int(decoder.config.hidden_size):
        raise ValueError("Skip-Vision summary hidden size mismatches the decoder")
    return module


def _mean_cosine_similarity(
    tokens: torch.Tensor, *, chunk_size: int
) -> torch.Tensor:
    count = int(tokens.shape[0])
    if count == 1:
        return tokens.new_zeros((1,), dtype=torch.float32)
    normalized = torch.nn.functional.normalize(tokens.float(), dim=-1)
    totals = normalized.new_empty((count,))
    transposed = normalized.transpose(0, 1)
    for start in range(0, count, int(chunk_size)):
        end = min(count, start + int(chunk_size))
        similarity = normalized[start:end] @ transposed
        rows = torch.arange(start, end, device=tokens.device)
        similarity[
            torch.arange(end - start, device=tokens.device), rows
        ] = 0.0
        totals[start:end] = similarity.sum(dim=-1) / float(count - 1)
    return totals


def _distinctive_indices(
    tokens: torch.Tensor, count: int, *, chunk_size: int
) -> torch.LongTensor:
    requested = int(count)
    if requested < 0 or requested > int(tokens.shape[0]):
        raise ValueError("Skip-Vision distinctive-token count is invalid")
    if requested == 0:
        return torch.empty((0,), device=tokens.device, dtype=torch.long)
    if requested == int(tokens.shape[0]):
        return torch.arange(requested, device=tokens.device, dtype=torch.long)
    with torch.no_grad():
        scores = _mean_cosine_similarity(tokens, chunk_size=chunk_size)
        return scores.topk(requested, largest=False, sorted=False).indices.sort().values


def _mean_proxy_relevance_indices(
    tokens: torch.Tensor, count: int
) -> torch.LongTensor:
    """Approximate the released CLIP-CLS ranking for CLS-free SigLIP.

    Skip-Vision keeps the LLaVA patch tokens with the largest cosine
    similarity to CLIP's CLS token.  OneVision's SigLIP tower exposes patch
    tokens but no CLS token to the projector, so the projected mean feature is
    used as an image-level proxy and the same *largest cosine* rule is kept.
    """

    requested = int(count)
    token_count = int(tokens.shape[0])
    if requested < 0 or requested > token_count:
        raise ValueError("Skip-Vision CLS-proxy token count is invalid")
    if requested == 0:
        return torch.empty((0,), device=tokens.device, dtype=torch.long)
    if requested == token_count:
        return torch.arange(token_count, device=tokens.device, dtype=torch.long)
    with torch.no_grad():
        normalized = torch.nn.functional.normalize(tokens.float(), dim=-1)
        proxy = torch.nn.functional.normalize(
            tokens.float().mean(dim=0, keepdim=True), dim=-1
        )
        relevance = (normalized @ proxy.transpose(0, 1)).squeeze(-1)
        return relevance.topk(requested, largest=True, sorted=False).indices.sort().values


def _merge_to_distinctive_representatives(
    tokens: torch.Tensor, target_count: int, *, chunk_size: int
) -> Tuple[torch.Tensor, torch.LongTensor]:
    source_count = int(tokens.shape[0])
    target = int(target_count)
    if not 1 <= target <= source_count:
        raise ValueError("Skip-Vision contextual merge target is invalid")
    representative_indices = _distinctive_indices(
        tokens, target, chunk_size=chunk_size
    )
    if target == source_count:
        return tokens, representative_indices, torch.eye(source_count, device=tokens.device, dtype=tokens.dtype)
    with torch.no_grad():
        source_mask = torch.ones(
            source_count, device=tokens.device, dtype=torch.bool
        )
        source_mask[representative_indices] = False
        merged_indices = source_mask.nonzero(as_tuple=True)[0]
        representatives = torch.nn.functional.normalize(
            tokens.index_select(0, representative_indices).float(), dim=-1
        )
        normalized_sources = torch.nn.functional.normalize(
            tokens.index_select(0, merged_indices).float(), dim=-1
        )
        assignments = []
        for start in range(0, int(merged_indices.numel()), int(chunk_size)):
            end = min(int(merged_indices.numel()), start + int(chunk_size))
            assignments.append(
                (normalized_sources[start:end] @ representatives.transpose(0, 1))
                .argmax(dim=-1)
                .to(dtype=torch.long)
            )
        assignment = torch.cat(assignments, dim=0)
    output = tokens.index_select(0, representative_indices).clone()
    counts = output.new_ones((target, 1))
    merged_values = tokens.index_select(0, merged_indices)
    output.index_add_(0, assignment, merged_values)
    counts.index_add_(0, assignment, counts.new_ones((assignment.numel(), 1)))
    output = output / counts
    mixing = tokens.new_zeros((target, source_count))
    mixing[torch.arange(target, device=tokens.device), representative_indices] = 1
    mixing[assignment, merged_indices] = 1
    return output, representative_indices, mixing / counts


def compress_skip_vision_features(
    features: torch.Tensor,
    *,
    cfg: Dict[str, Any],
    base_crop_tokens: int,
    summary_module: SkipVisionAdaptiveSummary,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Compress one fully assembled OV image block and insert two summaries."""

    if features.ndim != 2 or int(features.shape[0]) < 2:
        raise ValueError("Skip-Vision expects one [visual_tokens, hidden] block")
    original = int(features.shape[0])
    attention_count = min(
        original,
        max(2, int(round(original * cfg["attention_visual_retain_ratio"]))),
    )
    natural_global = int(base_crop_tokens) if original > int(base_crop_tokens) else 0
    # The OV adaptation promises to retain the complete AnyRes base crop as
    # the global group.  Leave at least one contextual representative as well;
    # otherwise a low ratio on a small two-view image would make the declared
    # attention budget smaller than the mandatory global group.
    if natural_global:
        attention_count = max(
            attention_count, min(original, natural_global + 1)
        )
    retained_count = min(
        attention_count - 1,
        max(1, int(round(original * cfg["ffn_visual_retain_ratio"]))),
    )
    if natural_global > retained_count:
        retained_count = natural_global
    chunk_size = int(cfg["similarity_chunk_size"])

    if natural_global:
        global_indices = torch.arange(
            natural_global, device=features.device, dtype=torch.long
        )
        promote_count = retained_count - natural_global
        local_features = features[natural_global:]
        promoted_local = _distinctive_indices(
            local_features, promote_count, chunk_size=chunk_size
        ) + natural_global
        dominant_indices = torch.cat((global_indices, promoted_local), dim=0)
    else:
        dominant_indices = _mean_proxy_relevance_indices(
            features, retained_count
        )
    dominant_mask = torch.zeros(
        original, device=features.device, dtype=torch.bool
    )
    dominant_mask[dominant_indices] = True
    contextual_indices = (~dominant_mask).nonzero(as_tuple=True)[0]
    contextual_target = attention_count - retained_count
    contextual, representative_local, contextual_mixing = _merge_to_distinctive_representatives(
        features.index_select(0, contextual_indices),
        contextual_target,
        chunk_size=chunk_size,
    )
    dominant = features.index_select(0, dominant_indices)
    retained_summary = summary_module(dominant)
    contextual_summary = summary_module(contextual)
    output = torch.cat(
        (dominant, retained_summary, contextual, contextual_summary), dim=0
    )
    dominant_mixing = features.new_zeros((retained_count, original))
    dominant_mixing[torch.arange(retained_count, device=features.device), dominant_indices] = 1
    local_mixing = features.new_zeros((contextual_target, original))
    local_mixing[:, contextual_indices] = contextual_mixing
    retained_weights = torch.softmax(summary_module.score(dominant), dim=0).T
    contextual_weights = torch.softmax(summary_module.score(contextual), dim=0).T
    mixing = torch.cat((dominant_mixing, retained_weights @ dominant_mixing,
                        local_mixing, contextual_weights @ local_mixing))
    layout = {
        "original_visual_tokens": original,
        "attention_visual_tokens": attention_count,
        # The paper calls this group Nr (retained).  Its full design also
        # routes the last token of the Ns/local group through every FFN.
        "retained_visual_tokens": retained_count,
        "ffn_visual_tokens": retained_count + 1,
        "dominant_tokens": retained_count,
        "contextual_tokens": contextual_target,
        "summary_tokens": 2,
        "output_visual_tokens": int(output.shape[0]),
        "natural_global_tokens": natural_global,
        "dominant_source_indices": dominant_indices.detach().cpu().tolist(),
        "contextual_representative_source_indices": contextual_indices
        .index_select(0, representative_local)
        .detach()
        .cpu()
        .tolist(),
        "relative_contextual_start": retained_count + 1,
        "relative_contextual_end": retained_count + 1 + contextual_target,
        "relative_last_contextual": retained_count + contextual_target,
        # The latter adaptive summary follows the local group and is the
        # terminal visual token.  The last local token immediately before it
        # is separately restored to FFN, matching Table 6's LV component.
        "relative_terminal_visual": int(output.shape[0]) - 1,
        "_mixing_weights": mixing,
    }
    return output, layout


def apply_skip_vision_to_feature_collection(
    image_features: Any,
    *,
    cfg: Dict[str, Any],
    base_crop_tokens: int,
    summary_module: SkipVisionAdaptiveSummary,
) -> Tuple[Any, list[Dict[str, Any]]]:
    tensor_input = isinstance(image_features, torch.Tensor)
    if tensor_input:
        if image_features.ndim != 3:
            raise ValueError("Skip-Vision tensor feature collection must be [B,N,D]")
        blocks: Sequence[torch.Tensor] = list(image_features.unbind(0))
    else:
        blocks = list(image_features)
    compressed = []
    layouts = []
    for block in blocks:
        output, layout = compress_skip_vision_features(
            block,
            cfg=cfg,
            base_crop_tokens=base_crop_tokens,
            summary_module=summary_module,
        )
        compressed.append(output)
        layouts.append(layout)
    if tensor_input and len({int(item.shape[0]) for item in compressed}) == 1:
        return torch.stack(compressed, dim=0), layouts
    return compressed, layouts


def skip_vision_operation_indices(
    runtime: Optional[dict], sequence_length: int, *, device: torch.device
) -> Tuple[Optional[torch.LongTensor], Optional[torch.LongTensor]]:
    """Return FFN rows and compact-cache rows for exact expanded spans."""

    if runtime is None:
        return None, None
    spans = runtime.get("expanded_visual_spans")
    layouts = runtime.get("skip_vision_layouts")
    if not isinstance(spans, list) or not isinstance(layouts, list):
        return None, None
    if len(spans) != len(layouts):
        raise RuntimeError("Skip-Vision visual spans/layouts disagree")
    if not spans:
        raise RuntimeError("Skip-Vision requires a non-empty visual span layout")
    skipped = []
    for span, layout in zip(spans, layouts):
        start, end = int(span[0]), int(span[1])
        if not 0 <= start < end <= sequence_length:
            raise RuntimeError("Skip-Vision expanded span is outside the sequence")
        if end - start != int(layout["output_visual_tokens"]):
            raise RuntimeError("Skip-Vision expanded span has an unexpected length")
        contextual_start = start + int(layout["relative_contextual_start"])
        contextual_end = start + int(layout["relative_contextual_end"])
        last_contextual = start + int(layout["relative_last_contextual"])
        if not start <= contextual_start < contextual_end < end:
            raise RuntimeError("Skip-Vision contextual layout is outside its visual span")
        if last_contextual != contextual_end - 1:
            raise RuntimeError("Skip-Vision last-local layout is inconsistent")
        # Full Skip-Vision (Table 6, LV) restores the last local token to FFN;
        # because it is no longer an FFN-skipped token, it also remains in KV.
        skipped.extend(range(contextual_start, last_contextual))
    skipped_tensor = torch.tensor(skipped, device=device, dtype=torch.long)
    # A small AnyRes block can retain the entire mandatory base crop and just
    # one contextual representative.  LV keeps that last contextual token, so
    # there are legitimately zero FFN/KV rows to skip.  Keep every row while
    # preserving projector merging, both summaries, and the configured budget.
    ffn_mask = torch.ones(sequence_length, device=device, dtype=torch.bool)
    ffn_mask[skipped_tensor] = False
    ffn_indices = ffn_mask.nonzero(as_tuple=True)[0]
    cache_mask = torch.ones(sequence_length, device=device, dtype=torch.bool)
    cache_mask[skipped_tensor] = False
    cache_indices = cache_mask.nonzero(as_tuple=True)[0]
    return ffn_indices, cache_indices


def qwen2_skip_vision_prefill_flops(
    *,
    visual_tokens: int,
    text_tokens: int,
    attention_visual_tokens: int,
    ffn_visual_tokens: int,
    hidden_size: int = 3584,
    intermediate_size: int = 18944,
    attention_heads: int = 28,
    key_value_heads: int = 4,
    layers: int = 28,
    summary_tokens: int = 2,
) -> Tuple[float, float]:
    """Return compressed/dense Qwen2 prefill FLOPs for one sample."""

    head_dim = hidden_size // attention_heads

    def layer_cost(sequence: int, mlp_rows: int) -> float:
        q = 2.0 * sequence * hidden_size * (attention_heads * head_dim)
        kv = 4.0 * sequence * hidden_size * (key_value_heads * head_dim)
        out = 2.0 * sequence * (attention_heads * head_dim) * hidden_size
        attn = 4.0 * sequence * sequence * (attention_heads * head_dim)
        mlp = 6.0 * mlp_rows * hidden_size * intermediate_size
        return q + kv + out + attn + mlp

    dense_sequence = int(visual_tokens) + int(text_tokens)
    compressed_sequence = (
        int(attention_visual_tokens) + int(summary_tokens) + int(text_tokens)
    )
    dense = float(layers) * layer_cost(dense_sequence, dense_sequence)
    compressed_mlp_rows = (
        int(ffn_visual_tokens) + int(summary_tokens) + int(text_tokens)
    )
    compressed = float(layers) * layer_cost(
        compressed_sequence, min(compressed_sequence, compressed_mlp_rows)
    )
    return compressed, dense


def select_skip_vision_flops_policy(
    *,
    sample_shapes: Sequence[Dict[str, Any]],
    target_flops_ratio: float = 0.67,
    base_crop_tokens: int = 729,
    minimum_attention_ratio: float = 0.50,
    maximum_attention_ratio: float = 0.99,
) -> Dict[str, Any]:
    """Search fixed OV projector/FFN ratios against samplewise Qwen2 FLOPs."""

    if not sample_shapes:
        raise ValueError("Skip-Vision FLOPs calibration needs sample shapes")
    target = float(target_flops_ratio)
    if not 0.0 < target <= 1.0:
        raise ValueError("Skip-Vision FLOPs target must lie in (0,1]")
    parsed = []
    for item in sample_shapes:
        visual = int(item["visual_tokens"])
        text = int(item["nonvisual_tokens"])
        if visual < 2 or text < 1:
            raise ValueError("Skip-Vision calibration shape is invalid")
        parsed.append((visual, text))

    best = None

    def evaluate(attention_ratio: float, ffn_ratio: float):
        method_costs = []
        baseline_costs = []
        counts = []
        for visual, text in parsed:
            attention_count = min(
                visual, max(2, int(round(visual * attention_ratio)))
            )
            natural_global = (
                int(base_crop_tokens) if visual > int(base_crop_tokens) else 0
            )
            if natural_global:
                attention_count = max(
                    attention_count, min(visual, natural_global + 1)
                )
            retained_count = min(
                attention_count - 1,
                max(1, int(round(visual * ffn_ratio))),
            )
            if natural_global and retained_count < natural_global:
                retained_count = natural_global
            # Both adaptive summaries and the final local token execute the
            # FFN in the paper's complete model.  ``summary_tokens`` accounts
            # for the former; this +1 accounts for the latter.
            effective_ffn_visual = retained_count + 1
            compressed, dense = qwen2_skip_vision_prefill_flops(
                visual_tokens=visual,
                text_tokens=text,
                attention_visual_tokens=attention_count,
                ffn_visual_tokens=effective_ffn_visual,
            )
            method_costs.append(float(compressed))
            baseline_costs.append(float(dense))
            counts.append(
                (attention_count, retained_count, effective_ffn_visual)
            )
        aggregate = aggregate_samplewise_flops(
            method_costs, baseline_costs
        )
        mean = float(aggregate["mean_flops_ratio"])
        # Prefer a real projector merge, then the smaller deviation from the
        # official Nr/Ns split when two discrete policies tie.
        key = (
            abs(mean - target),
            -float(attention_ratio < 1.0),
            abs(ffn_ratio / attention_ratio - (256.0 / 576.0)),
            -attention_ratio,
        )
        return key, aggregate, counts, mean

    coarse_step = 0.01
    attention = float(minimum_attention_ratio)
    while attention <= float(maximum_attention_ratio) + 1e-12:
        ffn = 0.10
        while ffn <= attention + 1e-12:
            candidate = evaluate(attention, ffn)
            if best is None or candidate[0] < best[0]:
                best = (*candidate, attention, ffn)
            ffn += coarse_step
        attention += coarse_step
    assert best is not None
    coarse_attention = float(best[4])
    coarse_ffn = float(best[5])
    for attention_index in range(-20, 21):
        attention = coarse_attention + attention_index * 0.0005
        if not minimum_attention_ratio <= attention <= maximum_attention_ratio:
            continue
        for ffn_index in range(-20, 21):
            ffn = coarse_ffn + ffn_index * 0.0005
            if not 0.01 < ffn <= attention:
                continue
            candidate = evaluate(attention, ffn)
            if candidate[0] < best[0]:
                best = (*candidate, attention, ffn)
    _, aggregate, counts, mean, attention, ffn = best
    return {
        "attention_visual_retain_ratio": float(attention),
        "ffn_visual_retain_ratio": float(ffn),
        **aggregate,
        "absolute_target_error": float(abs(mean - target)),
        "sample_attention_visual_tokens": [int(item[0]) for item in counts],
        "sample_retained_visual_tokens": [int(item[1]) for item in counts],
        "sample_ffn_visual_tokens": [int(item[2]) for item in counts],
    }
