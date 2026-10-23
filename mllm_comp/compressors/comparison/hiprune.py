"""Paper/code-faithful HiPrune selector for OneVision SigLIP crops.

The released LLaVA implementation selects shallow object anchors, expands
each anchor to a five-token spatial cluster, and fills the remaining budget
with tokens ranked by deep vision attention.  This isolated port keeps that
algorithm and adapts only the patch-grid width from CLIP's 24 to OV SigLIP's
27 (resolved at runtime rather than hard-coded).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


HIPRUNE_DEFAULTS: Dict[str, Any] = {
    "method": "hiprune",
    # The paper's LLaVA-NeXT checkpoints use total budgets 640/320/160 over
    # five 576-token crops.  128 is therefore the native middle point per crop.
    "tokens_per_crop": 128,
    "alpha": 0.1,
    # The released LLaVA-NeXT recipe uses one-based layer 9 in a 24-layer
    # CLIP tower.  OV retains 26 SigLIP blocks after dropping the original
    # final block, so normalized-depth mapping gives round(9*26/24)=10.
    "object_layer": 10,
    "vision_layer": -1,
    "attention_source": "siglip_global_received",
    "neighborhood": "released_linear_clamp",
    "spatial_restore": True,
    "image_only": True,
    "log_first_n": 1,
}


def normalize_hiprune_config(
    raw: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    cfg = dict(HIPRUNE_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown HiPrune config keys: {', '.join(unknown)}")
        cfg.update(raw)

    cfg["method"] = "hiprune"
    cfg["tokens_per_crop"] = int(cfg["tokens_per_crop"])
    cfg["alpha"] = float(cfg["alpha"])
    cfg["object_layer"] = int(cfg["object_layer"])
    cfg["vision_layer"] = int(cfg["vision_layer"])
    cfg["attention_source"] = str(cfg["attention_source"]).strip().lower()
    cfg["neighborhood"] = str(cfg["neighborhood"]).strip().lower()
    cfg["spatial_restore"] = bool(cfg["spatial_restore"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["tokens_per_crop"] < 1:
        raise ValueError("HiPrune tokens_per_crop must be >= 1")
    if not 0.0 <= cfg["alpha"] <= 1.0:
        raise ValueError("HiPrune alpha must satisfy 0 <= alpha <= 1")
    if cfg["object_layer"] < 1:
        raise ValueError("HiPrune object_layer is one-based and must be >= 1")
    if cfg["attention_source"] != "siglip_global_received":
        raise ValueError(
            "OV HiPrune requires attention_source='siglip_global_received'"
        )
    if cfg["neighborhood"] != "released_linear_clamp":
        raise ValueError(
            "HiPrune reproduction requires neighborhood='released_linear_clamp'"
        )
    if not cfg["spatial_restore"]:
        raise ValueError("OneVision AnyRes requires HiPrune spatial_restore=true")
    if cfg["log_first_n"] < 0:
        raise ValueError("HiPrune log_first_n must be >= 0")
    return cfg


def _global_received_attention(attention: torch.Tensor) -> torch.Tensor:
    if attention.ndim == 2:
        return attention.float()
    if attention.ndim != 4 or attention.shape[-1] != attention.shape[-2]:
        raise ValueError("HiPrune attention must have shape [B,H,N,N]")
    # SigLIP has no CLS token.  The released alternative is global received
    # attention: average over heads and over all query rows.
    return attention.float().mean(dim=1).mean(dim=1)


def select_hiprune_indices(
    shallow_attention: torch.Tensor,
    deep_attention: torch.Tensor,
    *,
    tokens_per_crop: int,
    alpha: float,
    patches_per_side: int,
    grid_height: Optional[int] = None,
) -> Tuple[torch.LongTensor, Dict[str, int]]:
    """Return HiPrune patch indices in original spatial order."""

    shallow = _global_received_attention(shallow_attention)
    deep = _global_received_attention(deep_attention)
    if shallow.shape != deep.shape:
        raise ValueError("HiPrune shallow/deep attention shapes do not match")
    batch_size, patch_count = shallow.shape
    side = int(patches_per_side)
    height = side if grid_height is None else int(grid_height)
    if side < 1 or height < 1 or height * side != patch_count:
        raise ValueError(
            f"HiPrune patch grid {side}x{side} does not match N={patch_count}"
        )
    target = min(int(tokens_per_crop), int(patch_count))
    if target < 1:
        raise ValueError("HiPrune must retain at least one patch")
    if target == patch_count:
        selected = torch.arange(
            patch_count, device=shallow.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        return selected, {
            "shallow_anchor_tokens_per_crop": 0,
            "shallow_buffer_tokens_max_per_crop": 0,
            "deep_tokens_per_crop": patch_count,
        }

    # This is exactly the released budget equation: alpha is the intended
    # fraction occupied by five-token anchor clusters.
    anchor_count = int(round(target * float(alpha) / 5.0))
    anchor_count = min(anchor_count, patch_count)
    rows = []
    shallow_counts = []
    deep_counts = []
    for batch_index in range(batch_size):
        if anchor_count:
            anchors = torch.topk(
                shallow[batch_index], k=anchor_count, largest=True
            ).indices
            # Preserve the public repository's linear +/-1,+/-width expansion
            # and clamp semantics.  Only the width/max index change 24->27 and
            # 575->728 for the OV SigLIP tower.
            shallow_indices = torch.cat(
                (
                    anchors,
                    anchors - 1,
                    anchors + 1,
                    anchors - side,
                    anchors + side,
                )
            ).clamp_(0, patch_count - 1)
            shallow_indices = torch.unique(shallow_indices, sorted=False)
        else:
            shallow_indices = torch.empty(
                0, dtype=torch.long, device=shallow.device
            )

        # alpha<=1 guarantees at most target positions before de-duplication,
        # but keep this explicit so a future config change fails loudly.
        if int(shallow_indices.numel()) > target:
            raise RuntimeError(
                "HiPrune shallow anchor expansion exceeds the token budget"
            )
        deep_count = target - int(shallow_indices.numel())
        deep_scores = deep[batch_index].clone()
        if shallow_indices.numel():
            deep_scores[shallow_indices] = -torch.inf
        if deep_count:
            deep_indices = torch.topk(
                deep_scores, k=deep_count, largest=True
            ).indices
            selected = torch.cat((shallow_indices, deep_indices))
        else:
            selected = shallow_indices
        if int(torch.unique(selected).numel()) != target:
            raise RuntimeError("HiPrune produced duplicate retained indices")
        rows.append(selected.sort().values)
        shallow_counts.append(int(shallow_indices.numel()))
        deep_counts.append(int(deep_count))

    return torch.stack(rows), {
        "shallow_anchor_tokens_per_crop": int(anchor_count),
        "shallow_buffer_tokens_max_per_crop": int(anchor_count * 5),
        "shallow_selected_tokens_min_per_crop": int(min(shallow_counts)),
        "shallow_selected_tokens_max_per_crop": int(max(shallow_counts)),
        "deep_tokens_min_per_crop": int(min(deep_counts)),
        "deep_tokens_max_per_crop": int(max(deep_counts)),
    }


@torch.no_grad()
def encode_siglip_hiprune(
    vision_tower: torch.nn.Module,
    images: torch.Tensor,
    config: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.LongTensor, Dict[str, Any]]:
    """Run frozen SigLIP once and apply the released hierarchical selector."""

    if not isinstance(images, torch.Tensor) or images.ndim != 4:
        raise ValueError("HiPrune OV expects a 4-D image crop tensor")
    inner = getattr(vision_tower, "vision_tower", None)
    vision_model = getattr(inner, "vision_model", None)
    encoder = getattr(vision_model, "encoder", None)
    layers = getattr(encoder, "layers", None)
    if inner is None or layers is None or len(layers) == 0:
        raise TypeError("HiPrune OV requires the repository SigLipVisionTower")

    object_index = int(config["object_layer"]) - 1
    requested_deep = int(config["vision_layer"])
    deep_index = (
        requested_deep if requested_deep >= 0 else len(layers) + requested_deep
    )
    if not 0 <= object_index < len(layers):
        raise ValueError(
            f"HiPrune object_layer={config['object_layer']} is outside the "
            f"{len(layers)} retained SigLIP layers"
        )
    if not 0 <= deep_index < len(layers):
        raise ValueError(
            f"HiPrune vision_layer={requested_deep} is outside the "
            f"{len(layers)} retained SigLIP layers"
        )

    captured: Dict[int, torch.Tensor] = {}

    def capture(layer_index: int):
        def _hook(_module, _args, output):
            if not isinstance(output, tuple) or len(output) < 2:
                raise RuntimeError("HiPrune could not capture SigLIP attention")
            captured[layer_index] = output[1]

        return _hook

    handles = []
    for layer_index in sorted({object_index, deep_index}):
        handles.append(
            layers[layer_index].self_attn.register_forward_hook(
                capture(layer_index)
            )
        )
    try:
        outputs = inner(
            images.to(device=vision_tower.device, dtype=vision_tower.dtype),
            output_hidden_states=True,
            output_attentions=False,
        )
    finally:
        for handle in handles:
            handle.remove()

    if object_index not in captured or deep_index not in captured:
        raise RuntimeError("HiPrune attention capture is incomplete")
    dense_features = outputs.hidden_states[-1].to(images.dtype)
    side = int(vision_tower.num_patches_per_side)
    patch_count = int(dense_features.shape[1])
    initial_budget = min(int(config["tokens_per_crop"]), patch_count)

    def selection_for_budget(budget: int) -> torch.LongTensor:
        selected, _ = select_hiprune_indices(
            captured[object_index],
            captured[deep_index],
            tokens_per_crop=int(budget),
            alpha=float(config["alpha"]),
            patches_per_side=side,
        )
        return selected

    budget_audit: Dict[str, Any] = {}
    geometry = config.get("_average_budget_geometry")
    if geometry is not None:
        from .mmtok import search_sparse_anyres_native_budget

        budget, selected, budget_audit = search_sparse_anyres_native_budget(
            selection_for_budget,
            initial_tokens_per_crop=initial_budget,
            minimum_tokens_per_crop=1,
            maximum_tokens_per_crop=patch_count,
            target_ratio=float(geometry["target_ratio"]),
            uncompressed_visual_tokens=int(
                geometry["uncompressed_visual_tokens"]
            ),
            grid_width=int(geometry["grid_width"]),
            grid_height=int(geometry["grid_height"]),
            patches_per_side=side,
            original_size=tuple(geometry["original_size"]),
            spatial_unpad=bool(geometry["spatial_unpad"]),
            include_base=True,
            max_evaluations=int(geometry.get("max_evaluations", 10)),
        )
    else:
        budget = initial_budget
        selected = selection_for_budget(budget)

    selected, selection_stats = select_hiprune_indices(
        captured[object_index],
        captured[deep_index],
        tokens_per_crop=int(budget),
        alpha=float(config["alpha"]),
        patches_per_side=side,
    )
    compressed = dense_features.gather(
        1,
        selected.unsqueeze(-1).expand(-1, -1, dense_features.shape[-1]),
    )
    stats: Dict[str, Any] = {
        **selection_stats,
        **budget_audit,
        "crop_count": int(dense_features.shape[0]),
        "input_tokens_per_crop": patch_count,
        "output_tokens_per_crop": int(selected.shape[1]),
        "resolved_object_layer": object_index,
        "resolved_vision_layer": deep_index,
        "patches_per_side": side,
    }
    return compressed, selected, stats
