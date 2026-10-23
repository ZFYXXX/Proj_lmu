"""Train-free initialization-state port of TwigVLM's TTP path.

TwigVLM itself is *not* train-free: its extra twig layers, norm and LM head are
initialized from the base VLM and then post-trained with an autoregressive
loss.  No compatible OneVision/Qwen2 twig checkpoint is available in the
provided release.  For the requested no-training setting, this module exposes
the exact pre-training initialization: base layers K..K+T are reused as a
weight-identical twig branch, whose final attention guides pruning at layer K.

This is a runnable architecture/algorithm port and a useful controlled
baseline, but it must not be reported as the trained paper model.  The config,
runtime metadata and logs deliberately retain the ``initialization_only`` tag.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple

import torch


TWIGVLM_DEFAULTS: Dict[str, Any] = {
    "method": "twigvlm",
    "adaptation": "initialization_only",
    "twig_start_layer": 2,
    "twig_layer_count": 3,
    "twig_weight_source": "shared_base_initialization",
    # Paper default: average 64 / 576 visual tokens (88.9% reduction).
    "average_retention_ratio": 1.0 / 9.0,
    # K_f=24 for the 32-layer LLaVA-1.5/NeXT backbones.
    "final_wipe_layer_fraction": 0.75,
    "enable_final_wipe": True,
    # The fixed release selects from the last prompt row, although Eq. (3)
    # describes a sum over textual queries.
    "score_source": "released_last_prompt_token",
    "min_visual_tokens": 1,
    "generation_strategy": "autoregressive",
    "trained_twig_checkpoint": None,
    "single_image_only": True,
    "log_first_n": 1,
}


def normalize_twigvlm_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(TWIGVLM_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(
                f"Unknown TwigVLM config keys: {', '.join(unknown)}"
            )
        cfg.update(raw)

    cfg["method"] = "twigvlm"
    cfg["adaptation"] = str(cfg["adaptation"]).strip().lower()
    cfg["twig_start_layer"] = int(cfg["twig_start_layer"])
    cfg["twig_layer_count"] = int(cfg["twig_layer_count"])
    cfg["twig_weight_source"] = str(cfg["twig_weight_source"]).strip().lower()
    cfg["average_retention_ratio"] = float(cfg["average_retention_ratio"])
    cfg["final_wipe_layer_fraction"] = float(
        cfg["final_wipe_layer_fraction"]
    )
    cfg["enable_final_wipe"] = bool(cfg["enable_final_wipe"])
    cfg["score_source"] = str(cfg["score_source"]).strip().lower()
    cfg["min_visual_tokens"] = int(cfg["min_visual_tokens"])
    cfg["generation_strategy"] = str(cfg["generation_strategy"]).strip().lower()
    checkpoint = cfg["trained_twig_checkpoint"]
    cfg["trained_twig_checkpoint"] = (
        None if checkpoint in {None, ""} else str(checkpoint)
    )
    cfg["single_image_only"] = bool(cfg["single_image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["adaptation"] != "initialization_only":
        raise ValueError(
            "The no-training OV port requires adaptation='initialization_only'"
        )
    if cfg["twig_start_layer"] < 1:
        raise ValueError("TwigVLM twig_start_layer must be >= 1")
    if cfg["twig_layer_count"] < 1:
        raise ValueError("TwigVLM twig_layer_count must be >= 1")
    if cfg["twig_weight_source"] != "shared_base_initialization":
        raise ValueError(
            "The no-training port requires "
            "twig_weight_source='shared_base_initialization'"
        )
    if not 0.0 < cfg["average_retention_ratio"] <= 1.0:
        raise ValueError("TwigVLM average_retention_ratio must lie in (0,1]")
    if not 0.0 < cfg["final_wipe_layer_fraction"] <= 1.0:
        raise ValueError(
            "TwigVLM final_wipe_layer_fraction must lie in (0,1]"
        )
    if cfg["score_source"] not in {
        "released_last_prompt_token",
        "paper_all_post_image_text",
    }:
        raise ValueError(
            "TwigVLM score_source must be released_last_prompt_token or "
            "paper_all_post_image_text"
        )
    if cfg["min_visual_tokens"] < 1:
        raise ValueError("TwigVLM min_visual_tokens must be >= 1")
    if cfg["generation_strategy"] != "autoregressive":
        raise ValueError(
            "Self-speculative decoding requires a trained Qwen-compatible twig; "
            "the initialization-only port uses generation_strategy='autoregressive'"
        )
    if cfg["trained_twig_checkpoint"] is not None:
        raise ValueError(
            "The released LLaMA twig checkpoint is shape-incompatible with "
            "OneVision/Qwen2; trained_twig_checkpoint must be null"
        )
    if not cfg["single_image_only"]:
        raise ValueError("TwigVLM TTP currently requires a single visual span")
    if cfg["log_first_n"] < 0:
        raise ValueError("TwigVLM log_first_n must be >= 0")
    return cfg


def resolve_twigvlm_schedule(
    layer_count: int, cfg: Dict[str, Any]
) -> Tuple[int, int, int]:
    """Return ``(K, T, K_f)`` using the paper's depth semantics."""

    layers = int(layer_count)
    if layers < 1:
        raise ValueError("TwigVLM requires a non-empty decoder")
    start = int(cfg["twig_start_layer"])
    twig_count = int(cfg["twig_layer_count"])
    final_wipe = int(layers * float(cfg["final_wipe_layer_fraction"]))
    final_wipe = min(layers, max(1, final_wipe))
    if start + twig_count > layers:
        raise ValueError(
            f"TwigVLM branch [{start},{start + twig_count}) exceeds {layers} layers"
        )
    if bool(cfg["enable_final_wipe"]) and not start < final_wipe < layers:
        raise ValueError(
            "TwigVLM FinalWipe must occur after TTP and before the final layer: "
            f"K={start}, K_f={final_wipe}, L={layers}"
        )
    return start, twig_count, final_wipe


def twigvlm_intermediate_budget(
    visual_tokens: int,
    layer_count: int,
    twig_start_layer: int,
    final_wipe_layer: int,
    average_retention_ratio: float,
    *,
    min_visual_tokens: int,
) -> Tuple[int, float]:
    """Solve paper Eq. (6) for the K..K_f intermediate token count."""

    visual = int(visual_tokens)
    layers = int(layer_count)
    start = int(twig_start_layer)
    wipe = int(final_wipe_layer)
    if visual < 1 or not 0 <= start < wipe <= layers:
        raise ValueError("Invalid TwigVLM budget dimensions")
    numerator = visual * (
        layers * float(average_retention_ratio) - start
    )
    raw = numerator / float(wipe - start)
    retained = int(math.ceil(raw))
    retained = min(visual, max(int(min_visual_tokens), retained))
    actual = (
        visual * start + retained * (wipe - start)
    ) / float(visual * layers)
    return retained, actual


@torch.no_grad()
def run_initialization_twig_branch(
    hidden_states: torch.Tensor,
    decoder_layers: Sequence[torch.nn.Module],
    *,
    twig_start_layer: int,
    twig_layer_count: int,
    attention_mask: Optional[torch.Tensor],
    position_ids: torch.LongTensor,
) -> torch.Tensor:
    """Run the exact base-weight initialization of the untrained twig branch."""

    start = int(twig_start_layer)
    end = start + int(twig_layer_count)
    if hidden_states.ndim != 3 or hidden_states.shape[0] != 1:
        raise ValueError("TwigVLM branch requires hidden states [1,L,D]")
    if not 0 <= start < end <= len(decoder_layers):
        raise ValueError("TwigVLM branch layer range is invalid")
    branch_states = hidden_states
    last_attention = None
    for branch_index, layer in enumerate(decoder_layers[start:end]):
        capture = branch_index == int(twig_layer_count) - 1
        outputs = layer(
            branch_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=None,
            output_attentions=capture,
            use_cache=False,
        )
        branch_states = outputs[0]
        if capture:
            last_attention = outputs[1]
    if last_attention is None:
        raise RuntimeError("TwigVLM branch did not return its final attention")
    return last_attention


@torch.no_grad()
def build_twigvlm_ttp_plan(
    scorer_attention: torch.Tensor,
    *,
    sequence_length: int,
    visual_span: Tuple[int, int],
    retained_visual_tokens: int,
    score_source: str,
    device: torch.device,
    prompt_query_index: Optional[int] = None,
) -> Dict[str, Any]:
    """Select TTP survivors using paper or fixed-release query aggregation."""

    if scorer_attention.ndim != 4 or scorer_attention.shape[0] != 1:
        raise ValueError("TwigVLM scorer attention must have shape [1,H,Q,K]")
    start, end = map(int, visual_span)
    if not 0 <= start < end <= int(sequence_length):
        raise ValueError("TwigVLM visual span is outside the sequence")
    if scorer_attention.shape[-1] < end:
        raise ValueError("TwigVLM scorer attention is shorter than visual span")
    visual_tokens = end - start
    retained = min(visual_tokens, max(1, int(retained_visual_tokens)))
    prompt_query = int(scorer_attention.shape[-2]) - 1 if prompt_query_index is None else int(prompt_query_index)
    if not end <= prompt_query < int(scorer_attention.shape[-2]):
        raise ValueError("TwigVLM prompt query must follow the visual span")
    visual_attention = scorer_attention[..., start:end].float()
    source = str(score_source)
    if source == "released_last_prompt_token":
        scores = visual_attention[:, :, prompt_query, :].mean(dim=1)[0]
    elif source == "paper_all_post_image_text":
        if end >= int(scorer_attention.shape[-2]):
            raise ValueError("TwigVLM paper scorer has no post-image text query")
        scores = visual_attention[:, :, end : prompt_query + 1, :].sum(dim=(1, 2))[0]
    else:
        raise ValueError(f"Unsupported TwigVLM score source: {source}")
    if not torch.isfinite(scores).all():
        raise RuntimeError("TwigVLM TTP scores contain non-finite values")
    selected = torch.argsort(scores, descending=True, stable=True)[:retained]
    selected = selected.to(device=device, dtype=torch.long).sort().values
    keep = torch.cat(
        (
            torch.arange(start, dtype=torch.long, device=device),
            selected + start,
            torch.arange(end, int(sequence_length), dtype=torch.long, device=device),
        )
    )
    return {
        "keep_indices": keep,
        "selected_visual_indices": selected,
        "visual_tokens_before": visual_tokens,
        "visual_tokens_after": retained,
        "sequence_tokens_after": int(keep.numel()),
    }


def build_twigvlm_finalwipe_indices(
    sequence_length: int,
    visual_span: Tuple[int, int],
    *,
    device: torch.device,
) -> torch.LongTensor:
    """Remove every survivor visual token at K_f, preserving all text."""

    start, end = map(int, visual_span)
    if not 0 <= start < end <= int(sequence_length):
        raise ValueError("TwigVLM FinalWipe visual span is invalid")
    return torch.cat(
        (
            torch.arange(start, dtype=torch.long, device=device),
            torch.arange(end, int(sequence_length), dtype=torch.long, device=device),
        )
    )
