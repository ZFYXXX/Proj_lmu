"""CDPruner and its official DPPruner ablation for LLaVA-OneVision.

The selection equations intentionally mirror the released NeurIPS 2025 code:
cosine visual kernel, released negative-mean instruction relevance, global
min-max normalization, ``diag(r) L diag(r)``, and Cholesky-style fast greedy
MAP DPP inference.  The OneVision adapter applies that selector independently
to every SigLIP crop and restores original patch order before AnyRes assembly.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

def normalize_projector_position_id_mode(value: object) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    mode = {
        "original": "keep_original",
        "original_gap": "keep_original",
        "gap": "keep_original",
        "contiguous": "reindex",
        "continuous": "reindex",
        "pyramiddrop": "reindex",
    }.get(mode, mode)
    if mode not in {"keep_original", "reindex"}:
        raise ValueError(
            "position_id_mode must be 'keep_original' or 'reindex'"
        )
    return mode


_COMMON_DEFAULTS: Dict[str, Any] = {
    # The official LLaVA-NeXT example retains 32 tokens in each of five crops.
    "tokens_per_crop": 32,
    "selection_scope": "per_crop",
    "similarity": "cosine",
    "dpp_solver": "official_fast_map",
    "relevance_source": "projector_llm_instruction_mean",
    "relevance_transform": "released_negative_mean_minmax",
    "position_id_mode": "reindex",
    # The released LLaVA-NeXT path removes the ``_unpad`` suffix before
    # sparse AnyRes assembly.  OP/Position-ID experiments use the native
    # OneVision unpad geometry instead and opt into it explicitly.
    "anyres_assembly": "official_drop_unpad",
    "spatial_restore": True,
    "image_only": True,
    "log_first_n": 1,
}


def _normalize(raw: Optional[Dict[str, Any]], *, method: str) -> Dict[str, Any]:
    cfg = {"method": method, **_COMMON_DEFAULTS}
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(
                f"Unknown {method} config keys: {', '.join(unknown)}"
            )
        cfg.update(raw)
    cfg["method"] = method
    cfg["tokens_per_crop"] = int(cfg["tokens_per_crop"])
    cfg["selection_scope"] = str(cfg["selection_scope"]).strip().lower()
    cfg["similarity"] = str(cfg["similarity"]).strip().lower()
    cfg["dpp_solver"] = str(cfg["dpp_solver"]).strip().lower()
    cfg["relevance_source"] = str(cfg["relevance_source"]).strip().lower()
    cfg["relevance_transform"] = str(
        cfg["relevance_transform"]
    ).strip().lower()
    cfg["position_id_mode"] = normalize_projector_position_id_mode(
        cfg["position_id_mode"]
    )
    cfg["anyres_assembly"] = str(cfg["anyres_assembly"]).strip().lower()
    cfg["spatial_restore"] = bool(cfg["spatial_restore"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["tokens_per_crop"] < 1:
        raise ValueError(f"{method} tokens_per_crop must be >= 1")
    if cfg["selection_scope"] != "per_crop":
        raise ValueError(f"{method} requires selection_scope='per_crop'")
    if cfg["similarity"] != "cosine":
        raise ValueError(f"{method} official reproduction requires cosine")
    if cfg["dpp_solver"] != "official_fast_map":
        raise ValueError(
            f"{method} official reproduction requires dpp_solver='official_fast_map'"
        )
    if cfg["relevance_source"] != "projector_llm_instruction_mean":
        raise ValueError(
            "The OV route uses the paper's projector + LLM instruction embedding"
        )
    if cfg["relevance_transform"] != "released_negative_mean_minmax":
        raise ValueError(
            "CDPruner must retain the released relevance transform"
        )
    if not cfg["spatial_restore"]:
        raise ValueError("OneVision AnyRes requires spatial_restore=true")
    if cfg["anyres_assembly"] not in {
        "official_drop_unpad",
        "onevision_unpad",
    }:
        raise ValueError(
            "CDPruner anyres_assembly must be 'official_drop_unpad' or "
            "'onevision_unpad'"
        )
    if cfg["log_first_n"] < 0:
        raise ValueError(f"{method} log_first_n must be >= 0")
    return cfg


def normalize_cdpruner_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _normalize(raw, method="cdpruner")


def normalize_dppruner_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _normalize(raw, method="dppruner")


def official_fast_map_dpp_indices(
    kernel: torch.Tensor,
    target_count: int,
    *,
    preserve_selection_order: bool = False,
) -> torch.LongTensor:
    """Released batched fast greedy MAP-DPP loop, including update order."""

    if kernel.ndim != 3 or kernel.shape[-1] != kernel.shape[-2]:
        raise ValueError("CDPruner kernel must have shape [B,N,N]")
    batch_size, token_count, _ = kernel.shape
    target = min(max(1, int(target_count)), int(token_count))
    device = kernel.device
    batch_indices = torch.arange(batch_size, device=device)
    cis = torch.zeros(
        (target, batch_size, token_count),
        dtype=kernel.dtype,
        device=device,
    )
    di2s = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    selected = torch.empty(
        (target, batch_size), dtype=torch.long, device=device
    )
    for step in range(target):
        chosen = torch.argmax(di2s, dim=-1)
        selected[step] = chosen
        correction = torch.einsum(
            "tb,tbn->bn",
            cis[:step, batch_indices, chosen],
            cis[:step],
        )
        eis = (kernel[batch_indices, chosen] - correction) / torch.sqrt(
            di2s[batch_indices, chosen]
        ).unsqueeze(-1)
        cis[step] = eis
        di2s -= torch.square(eis)
        di2s[batch_indices, chosen] = -float("inf")
    selected = selected.transpose(0, 1)
    return selected if preserve_selection_order else selected.sort(dim=1).values


def build_cdpruner_kernel(
    projected_features: torch.Tensor,
    instruction_embeddings: Optional[torch.Tensor],
    *,
    conditional: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Construct the released visual or conditional DPP kernel."""

    if projected_features.ndim != 3:
        raise ValueError("CDPruner features must have shape [crops,tokens,dim]")
    image_normalized = projected_features / projected_features.norm(
        dim=-1, keepdim=True
    )
    image_normalized = image_normalized.float()
    similarity = image_normalized @ image_normalized.transpose(1, 2)
    if not conditional:
        return similarity, None
    if instruction_embeddings is None or instruction_embeddings.ndim != 2:
        raise ValueError(
            "CDPruner requires instruction embeddings with shape [tokens,dim]"
        )
    if instruction_embeddings.shape[-1] != projected_features.shape[-1]:
        raise ValueError(
            "CDPruner projector and instruction embedding dimensions differ"
        )
    # Paper Sec. 3.3 specifies the *average instruction-token embedding* for
    # MLLMs with a dedicated vision encoder.  Average first, then normalize;
    # averaging token-wise cosine scores would be a different method.
    instruction_mean = instruction_embeddings.mean(dim=0, keepdim=True)
    text_normalized = instruction_mean / instruction_mean.norm(
        dim=-1, keepdim=True
    )
    relevance = -(
        image_normalized @ text_normalized.float().transpose(0, 1)
    ).squeeze(-1)
    # Preserve the released implementation exactly: +1e-6 is in the
    # numerator; no epsilon is added to the denominator.
    relevance = (relevance - relevance.min() + 1e-6) / (
        relevance.max() - relevance.min()
    )
    kernel = relevance.unsqueeze(2) * similarity * relevance.unsqueeze(1)
    return kernel, relevance


def select_cdpruner_tokens(
    projected_features: torch.Tensor,
    instruction_embeddings: Optional[torch.Tensor],
    *,
    cfg: Dict[str, Any],
    preserve_selection_order: bool = False,
) -> Tuple[torch.Tensor, torch.LongTensor, Dict[str, Any]]:
    """Select crop-local tokens while keeping gradients through survivors."""

    if projected_features.ndim != 3:
        raise ValueError("CDPruner expects [crops,patches,hidden]")
    crop_count, patch_count, hidden_size = projected_features.shape
    target = min(int(cfg["tokens_per_crop"]), int(patch_count))
    conditional = str(cfg["method"]) == "cdpruner"
    with torch.no_grad():
        kernel, relevance = build_cdpruner_kernel(
            projected_features.detach(),
            (
                instruction_embeddings.detach()
                if instruction_embeddings is not None
                else None
            ),
            conditional=conditional,
        )
        selected = official_fast_map_dpp_indices(
            kernel,
            target,
            preserve_selection_order=preserve_selection_order,
        )
    compressed = projected_features.gather(
        1, selected.unsqueeze(-1).expand(-1, -1, hidden_size)
    )
    stats: Dict[str, Any] = {
        "variant": "conditional_dpp" if conditional else "dpp_only",
        "crop_count": int(crop_count),
        "input_tokens_per_crop": int(patch_count),
        "output_tokens_per_crop": int(target),
        "greedy_selection_order_preserved": bool(preserve_selection_order),
        "position_id_mode": str(cfg["position_id_mode"]),
        "official_position_id_mode": "reindex",
        "anyres_assembly": str(cfg["anyres_assembly"]),
    }
    if relevance is not None:
        stats.update(
            {
                "relevance_min": float(relevance.min().item()),
                "relevance_max": float(relevance.max().item()),
            }
        )
    return compressed, selected, stats


def restore_cdpruner_spatial_grid(
    image_feature: torch.Tensor,
    selected_indices: torch.Tensor,
    *,
    grid_width: int,
    grid_height: int,
    patches_per_side: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Restore official selected patches before OneVision unpadding."""

    from .vispruner import restore_vispruner_spatial_grid

    return restore_vispruner_spatial_grid(
        image_feature,
        selected_indices,
        grid_width=grid_width,
        grid_height=grid_height,
        patches_per_side=patches_per_side,
    )
