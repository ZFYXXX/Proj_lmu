"""Entropy-Aware Dense Pruning (EADP) for LLaVA-OneVision/Qwen2.

This port follows the authors' public Qwen path because both the projected
visual tokens and the prompt tokens already live in Qwen2's LLM embedding
space.  It preserves the released negative-cosine dense guidance, entropy
filter, Gaussian smoothing, power transform, importance-proportional AnyRes
allocation, and greedy facility-location selector.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


EADP_DEFAULTS: Dict[str, Any] = {
    "method": "eadp",
    "tokens_per_crop": 128,
    "alpha": 0.5,
    "beta": 1.0,
    "entropy_temperature": 100.0,
    "entropy_keep_ratio": 0.2,
    "entropy_weight_temperature": 0.01,
    "spatial_kernel_size": 3,
    "spatial_sigma": 1.0,
    "allocation": "importance_based",
    "feature_space": "qwen2_llm_projected",
    "cross_similarity": "released_negative_cosine",
    "selection": "greedy_facility_location",
    "spatial_restore": True,
    "image_only": True,
    "log_first_n": 1,
}


# Qwen2's ChatML role markers are ordinary vocabulary ids around the two
# special message-boundary ids.  The released EADP Qwen path embeds only the
# raw text item(s) from the user message, not the system prompt, role labels,
# image placeholder, or assistant-generation prefix.  LLaVA reaches the
# compression hook after tokenisation, so recover that same content span from
# the qwen_1_5 input ids instead of embedding the whole serialised chat.
QWEN2_IM_START_ID = 151644
QWEN2_IM_END_ID = 151645
QWEN2_USER_ROLE_ID = 872
QWEN2_NEWLINE_ID = 198


def extract_qwen2_user_instruction_ids(
    input_ids: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    answer_start: Optional[int] = None,
) -> torch.Tensor:
    """Return the last Qwen2 ChatML user-content span, excluding image ids.

    The formal OneVision jobs use the qwen_1_5 template.  Taking the last user
    turn matches the public EADP evaluation interface, whose ``message`` is the
    current multimodal question rather than the fully rendered chat template.
    A malformed/non-Qwen prompt is rejected rather than silently changing the
    paper method's text guidance.
    """

    if input_ids.ndim != 1:
        raise ValueError("EADP Qwen2 prompt extraction requires 1-D input_ids")
    valid = torch.ones_like(input_ids, dtype=torch.bool)
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise ValueError("EADP attention_mask must match input_ids")
        valid &= attention_mask.bool()
    if answer_start is not None:
        boundary = int(answer_start)
        if boundary < 0 or boundary > input_ids.numel():
            raise ValueError("EADP assistant-content boundary is out of range")
        valid &= torch.arange(input_ids.numel(), device=input_ids.device) < boundary

    active = input_ids[valid]
    values = active.detach().cpu().tolist()
    user_spans: List[Tuple[int, int]] = []
    cursor = 0
    while cursor + 2 < len(values):
        if (
            values[cursor] == QWEN2_IM_START_ID
            and values[cursor + 1] == QWEN2_USER_ROLE_ID
            and values[cursor + 2] == QWEN2_NEWLINE_ID
        ):
            start = cursor + 3
            end = start
            while end < len(values) and values[end] != QWEN2_IM_END_ID:
                end += 1
            if end >= len(values):
                raise ValueError("EADP found an unterminated Qwen2 user message")
            user_spans.append((start, end))
            cursor = end + 1
        else:
            cursor += 1

    if not user_spans:
        raise ValueError(
            "EADP could not find a qwen_1_5 user-content span in input_ids"
        )
    start, end = user_spans[-1]
    content = active[start:end]
    # IMAGE_TOKEN_INDEX and any other multimodal sentinels are negative.  The
    # public EADP caller joins only text items, so remove those placeholders and
    # the newline(s) left immediately beside them.
    content = content[content >= 0]
    while content.numel() and int(content[0].item()) == QWEN2_NEWLINE_ID:
        content = content[1:]
    while content.numel() and int(content[-1].item()) == QWEN2_NEWLINE_ID:
        content = content[:-1]
    if content.numel() == 0:
        raise ValueError("EADP found no text tokens in the Qwen2 user message")
    return content


def normalize_eadp_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(EADP_DEFAULTS)
    if raw:
        unknown = sorted(set(raw) - set(cfg))
        if unknown:
            raise ValueError(f"Unknown EADP config keys: {', '.join(unknown)}")
        cfg.update(raw)
    cfg["method"] = "eadp"
    cfg["tokens_per_crop"] = int(cfg["tokens_per_crop"])
    for key in (
        "alpha",
        "beta",
        "entropy_temperature",
        "entropy_keep_ratio",
        "entropy_weight_temperature",
        "spatial_sigma",
    ):
        cfg[key] = float(cfg[key])
    cfg["spatial_kernel_size"] = int(cfg["spatial_kernel_size"])
    for key in (
        "allocation",
        "feature_space",
        "cross_similarity",
        "selection",
    ):
        cfg[key] = str(cfg[key]).strip().lower()
    cfg["spatial_restore"] = bool(cfg["spatial_restore"])
    cfg["image_only"] = bool(cfg["image_only"])
    cfg["log_first_n"] = int(cfg["log_first_n"])

    if cfg["tokens_per_crop"] < 1:
        raise ValueError("EADP tokens_per_crop must be >= 1")
    if not 0.0 <= cfg["alpha"] <= 1.0:
        raise ValueError("EADP alpha must satisfy 0 <= alpha <= 1")
    if cfg["beta"] <= 0:
        raise ValueError("EADP beta must be positive")
    if cfg["entropy_temperature"] <= 0:
        raise ValueError("EADP entropy_temperature must be positive")
    if not 0.0 < cfg["entropy_keep_ratio"] <= 1.0:
        raise ValueError("EADP entropy_keep_ratio must lie in (0, 1]")
    if cfg["entropy_weight_temperature"] <= 0:
        raise ValueError("EADP entropy_weight_temperature must be positive")
    kernel = cfg["spatial_kernel_size"]
    if kernel < 1 or kernel % 2 == 0:
        raise ValueError("EADP spatial_kernel_size must be positive and odd")
    if cfg["spatial_sigma"] <= 0:
        raise ValueError("EADP spatial_sigma must be positive")
    if cfg["allocation"] != "importance_based":
        raise ValueError("EADP reproduction requires importance_based allocation")
    if cfg["feature_space"] != "qwen2_llm_projected":
        raise ValueError("OV EADP requires Qwen2 LLM-space projected features")
    if cfg["cross_similarity"] != "released_negative_cosine":
        raise ValueError("EADP requires the released negative-cosine guidance")
    if cfg["selection"] != "greedy_facility_location":
        raise ValueError("EADP requires greedy facility-location selection")
    if not cfg["spatial_restore"]:
        raise ValueError("OneVision AnyRes requires EADP spatial_restore=true")
    if cfg["log_first_n"] < 0:
        raise ValueError("EADP log_first_n must be >= 0")
    return cfg


def visual_similarity(image_features: torch.Tensor) -> torch.Tensor:
    """Released cosine-to-[0,1] pairwise visual similarity."""

    if image_features.ndim != 3:
        raise ValueError("EADP image features must have shape [crops,N,D]")
    # The released Qwen kernel rounds normalization in the input dtype before
    # its FP32 coverage matmul. Moving .float() earlier changes greedy picks.
    normalized = (image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)).float()
    cosine = torch.bmm(normalized, normalized.transpose(1, 2)).clamp(-1, 1)
    return 0.5 * (cosine + 1.0)


def _spatial_smoothing(
    importance: torch.Tensor,
    *,
    patches_per_side: int,
    kernel_size: int,
    sigma: float,
    grid_height: Optional[int] = None,
) -> torch.Tensor:
    crops, patch_count = importance.shape
    side = int(patches_per_side)
    height = side if grid_height is None else int(grid_height)
    if height * side != patch_count:
        raise ValueError("EADP smoothing grid does not match native features")
    coords = torch.arange(
        kernel_size, device=importance.device, dtype=importance.dtype
    ) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx.square() + yy.square()) / (2 * sigma * sigma))
    kernel = (kernel / kernel.sum()).view(1, 1, kernel_size, kernel_size)
    grid = importance.view(crops, 1, height, side)
    pad = kernel_size // 2
    # Qwen can produce a one-row/one-column grid. Extend the same reflection
    # rule to singleton axes rather than treating them as square crops.
    def reflected_indices(length):
        values = torch.arange(-pad, length + pad, device=grid.device)
        if length == 1:
            return torch.zeros_like(values)
        values = values.remainder(2 * (length - 1))
        return torch.minimum(values, 2 * (length - 1) - values)
    grid = grid.index_select(-2, reflected_indices(height)).index_select(-1, reflected_indices(side))
    return F.conv2d(grid, kernel).view(crops, patch_count)


@torch.no_grad()
def compute_eadp_importance(
    image_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    patches_per_side: int,
    cfg: Dict[str, Any],
    grid_height: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Compute EADP importance and visual coverage matrices once per sample."""

    if image_features.ndim != 3:
        raise ValueError("EADP image_features must be [crops,N,D]")
    if text_embeddings.ndim != 2 or text_embeddings.shape[0] < 1:
        raise ValueError("EADP text_embeddings must be a non-empty [L,D] tensor")
    if image_features.shape[-1] != text_embeddings.shape[-1]:
        raise ValueError("EADP projected vision/text dimensions do not match")
    if not torch.isfinite(image_features).all() or not torch.isfinite(text_embeddings).all():
        raise ValueError("EADP visual/text features must be finite")

    image = F.normalize(image_features.float(), dim=-1, eps=1e-8)
    text_raw = text_embeddings.float()
    text = F.normalize(text_raw, dim=-1, eps=1e-8)
    # The Qwen release forms the global instruction vector by averaging raw
    # token embeddings first, then normalizing that mean. Averaging already
    # normalized token vectors is not equivalent when embedding norms differ.
    global_text = F.normalize(
        text_embeddings.mean(dim=0, keepdim=True).float(), dim=-1, eps=1e-8
    )

    # These leading minus signs look counter-intuitive but are present in both
    # public LLaVA and Qwen EADP implementations; preserve them exactly.
    global_sim = -torch.einsum("cnd,md->cnm", image, global_text)
    local_sim_all = -torch.einsum("cnd,ld->cnl", image, text)

    probabilities = torch.softmax(
        local_sim_all * float(cfg["entropy_temperature"]), dim=1
    )
    entropy = -(
        probabilities * torch.log(probabilities + 1e-12)
    ).sum(dim=1)
    text_keep = max(
        1, int(entropy.shape[-1] * float(cfg["entropy_keep_ratio"]))
    )
    kept_entropy, kept_text = torch.topk(
        entropy, k=text_keep, dim=-1, largest=False
    )
    gathered_local = local_sim_all.gather(
        2, kept_text.unsqueeze(1).expand(-1, image.shape[1], -1)
    )
    entropy_weights = torch.softmax(
        -kept_entropy / (float(cfg["entropy_weight_temperature"]) + 1e-6),
        dim=-1,
    )
    local_sim = (gathered_local * entropy_weights.unsqueeze(1)).sum(dim=-1)
    fused = (
        float(cfg["alpha"]) * global_sim.squeeze(-1)
        + (1.0 - float(cfg["alpha"])) * local_sim
    )

    # The public AnyRes code normalizes across all crops belonging to one
    # image, then reshapes back before per-crop smoothing.
    flat = fused.reshape(1, -1)
    low = flat.min(dim=-1, keepdim=True).values
    high = flat.max(dim=-1, keepdim=True).values
    importance = ((flat - low + 1e-6) / (high - low + 1e-6)).view_as(fused)
    importance = _spatial_smoothing(
        importance,
        patches_per_side=patches_per_side,
        kernel_size=int(cfg["spatial_kernel_size"]),
        sigma=float(cfg["spatial_sigma"]),
        grid_height=grid_height,
    )
    importance = importance.pow(float(cfg["beta"]))
    similarity = visual_similarity(image_features)
    return importance, similarity, {
        "prompt_text_tokens": int(text.shape[0]),
        "entropy_text_tokens_kept": int(text_keep),
        "entropy_text_keep_ratio_actual": float(text_keep / text.shape[0]),
    }


def _allocate_exact_quotas(
    scores: torch.Tensor,
    capacities: torch.LongTensor,
    total_budget: int,
) -> torch.LongTensor:
    """Importance-proportional, capacity-aware largest-remainder allocation."""

    if scores.ndim != 1 or capacities.ndim != 1 or scores.shape != capacities.shape:
        raise ValueError("EADP quota inputs must be equal-length vectors")
    caps = capacities.to(dtype=torch.long)
    total_capacity = int(caps.sum().item())
    budget = min(max(1, int(total_budget)), total_capacity)
    active = caps > 0
    quotas = torch.zeros_like(caps)
    active_count = int(active.sum().item())
    if active_count == 0:
        raise ValueError("EADP has no valid visual candidate")

    # Start from the public importance-proportional floor allocation and its
    # minimum-one rule.  The release can miss the requested total because of
    # flooring; the correction below changes only the integer remainder so the
    # experiment's expanded-token contract is exact.
    if budget >= active_count:
        weights = scores.clamp_min(0).masked_fill(~active, 0)
        if float(weights.sum().item()) <= 0:
            weights = caps.to(dtype=torch.float32)
        ideal = weights / weights.sum() * budget
        quotas = torch.minimum(torch.floor(ideal).long(), caps)
        quotas[active] = torch.maximum(
            quotas[active], torch.ones_like(quotas[active])
        )
    else:
        ranked = torch.argsort(scores.masked_fill(~active, -torch.inf), descending=True)
        quotas[ranked[:budget]] = 1
        return quotas

    excess = int(quotas.sum().item()) - budget
    while excess > 0:
        removable = quotas > 1
        if not bool(removable.any()):
            raise RuntimeError("EADP minimum crop quotas exceed the total budget")
        # Remove from the crop whose public floor overshoots its ideal share
        # most; ties are deterministic by crop index.
        overshoot = (quotas.float() - ideal).masked_fill(
            ~removable, -torch.inf
        )
        index = int(overshoot.argmax().item())
        quotas[index] -= 1
        excess -= 1

    remaining = budget - int(quotas.sum().item())
    while remaining > 0:
        room = caps - quotas
        eligible = room > 0
        if not bool(eligible.any()):
            break
        deficit = (ideal - quotas.float()).masked_fill(~eligible, -torch.inf)
        index = int(deficit.argmax().item())
        quotas[index] += 1
        remaining = budget - int(quotas.sum().item())
    if int(quotas.sum().item()) != budget:
        raise RuntimeError("EADP failed to allocate its exact visual-token budget")
    return quotas


@torch.no_grad()
def greedy_facility_location_indices(
    importance: torch.Tensor,
    similarity: torch.Tensor,
    token_num: int,
) -> torch.LongTensor:
    """The released greedy facility-location maximization for one crop."""

    if importance.ndim != 1 or similarity.ndim != 2:
        raise ValueError("EADP greedy selector expects [N] and [N,N]")
    if similarity.shape != (importance.numel(), importance.numel()):
        raise ValueError("EADP importance/similarity sizes do not match")
    count = int(importance.numel())
    target = min(int(token_num), count)
    if target < 1:
        return torch.empty(0, dtype=torch.long, device=importance.device)
    if target == count:
        return torch.arange(count, device=importance.device)

    current_coverage = torch.zeros(
        count, device=importance.device, dtype=similarity.dtype
    )
    selected_mask = torch.zeros(count, device=importance.device, dtype=torch.bool)
    selected = []
    for _ in range(target):
        new_coverage = torch.maximum(similarity, current_coverage.unsqueeze(0))
        gains = ((new_coverage - current_coverage.unsqueeze(0)) * importance).sum(
            dim=1
        )
        gains[selected_mask] = -torch.inf
        best = int(gains.argmax().item())
        selected.append(best)
        selected_mask[best] = True
        current_coverage = torch.maximum(current_coverage, similarity[best])
    return torch.tensor(selected, device=importance.device, dtype=torch.long)


@torch.no_grad()
def select_eadp_indices(
    image_features: torch.Tensor,
    text_embeddings: torch.Tensor,
    *,
    patches_per_side: int,
    cfg: Dict[str, Any],
    valid_masks: Optional[torch.Tensor] = None,
    total_patch_budget: Optional[int] = None,
    grid_height: Optional[int] = None,
) -> Tuple[List[torch.LongTensor], Dict[str, Any]]:
    """Select a variable, importance-allocated quota for every OV crop."""

    importance, similarity, stats = compute_eadp_importance(
        image_features,
        text_embeddings,
        patches_per_side=patches_per_side,
        cfg=cfg,
        grid_height=grid_height,
    )
    crops, patch_count = importance.shape
    if valid_masks is None:
        valid_masks = torch.ones(
            crops, patch_count, dtype=torch.bool, device=importance.device
        )
    if valid_masks.shape != (crops, patch_count):
        raise ValueError("EADP valid-mask shape does not match crop features")
    capacities = valid_masks.sum(dim=1).long()
    requested_total = (
        int(total_patch_budget)
        if total_patch_budget is not None
        else int(cfg["tokens_per_crop"]) * crops
    )
    allocation_scores = (importance * valid_masks).sum(dim=1)
    quotas = _allocate_exact_quotas(
        allocation_scores, capacities, requested_total
    )

    selections: List[torch.LongTensor] = []
    for crop_index in range(crops):
        candidates = valid_masks[crop_index].nonzero(as_tuple=True)[0]
        quota = int(quotas[crop_index].item())
        local = greedy_facility_location_indices(
            importance[crop_index].index_select(0, candidates),
            similarity[crop_index]
            .index_select(0, candidates)
            .index_select(1, candidates),
            quota,
        )
        selections.append(candidates.index_select(0, local).sort().values)

    selected_total = sum(int(item.numel()) for item in selections)
    stats.update(
        {
            "crop_count": int(crops),
            "input_tokens_per_crop": int(patch_count),
            "requested_patch_tokens_total": int(requested_total),
            "selected_patch_tokens_total": int(selected_total),
            "crop_token_quotas": [int(value) for value in quotas.tolist()],
            "crop_valid_capacities": [
                int(value) for value in capacities.tolist()
            ],
            "allocation": "importance_based_exact_integer",
        }
    )
    return selections, stats


def anyres_valid_patch_masks(
    *,
    crop_count: int,
    grid_width: int,
    grid_height: int,
    patches_per_side: int,
    original_size: Tuple[int, int],
    spatial_unpad: bool,
    device: torch.device,
) -> Tuple[torch.BoolTensor, Dict[str, int]]:
    """Map OV base/local crops to the real (unpadded) patch coordinates."""

    crops = int(crop_count)
    width_tiles = int(grid_width)
    height_tiles = int(grid_height)
    side = int(patches_per_side)
    if crops != 1 + width_tiles * height_tiles:
        raise ValueError(
            f"EADP crop count {crops} != base + {width_tiles}x{height_tiles}"
        )
    patch_count = side * side
    masks = torch.ones(crops, patch_count, dtype=torch.bool, device=device)
    full_height = height_tiles * side
    full_width = width_tiles * side
    row_low, row_high = 0, full_height
    col_low, col_high = 0, full_width
    if spatial_unpad:
        original_width, original_height = map(int, original_size)
        original_aspect = original_width / float(original_height)
        current_aspect = full_width / float(full_height)
        if original_aspect > current_aspect:
            new_height = int(original_height * (full_width / original_width))
            padding = (full_height - new_height) // 2
            row_low, row_high = padding, full_height - padding
        else:
            new_width = int(original_width * (full_height / original_height))
            padding = (full_width - new_width) // 2
            col_low, col_high = padding, full_width - padding

    patch_ids = torch.arange(patch_count, device=device)
    patch_rows = torch.div(patch_ids, side, rounding_mode="floor")
    patch_cols = patch_ids % side
    for local_index in range(width_tiles * height_tiles):
        tile_row = local_index // width_tiles
        tile_col = local_index % width_tiles
        global_rows = tile_row * side + patch_rows
        global_cols = tile_col * side + patch_cols
        masks[local_index + 1] = (
            (global_rows >= row_low)
            & (global_rows < row_high)
            & (global_cols >= col_low)
            & (global_cols < col_high)
        )
    return masks, {
        "unpadded_grid_height": int(row_high - row_low),
        "unpadded_grid_width": int(col_high - col_low),
        "newline_tokens": int(row_high - row_low) if spatial_unpad else 0,
        "valid_patch_capacity": int(masks.sum().item()),
    }


def restore_eadp_spatial_grid(
    local_features: torch.Tensor,
    local_selected_indices: Sequence[torch.LongTensor],
    *,
    grid_width: int,
    grid_height: int,
    patches_per_side: int,
) -> Tuple[torch.Tensor, torch.BoolTensor]:
    """Restore variable per-tile EADP selections to the OV stitched grid."""

    if local_features.ndim != 3:
        raise ValueError("EADP spatial restore expects local [tiles,N,D] features")
    tiles, patch_count, hidden_size = local_features.shape
    if tiles != int(grid_width) * int(grid_height):
        raise ValueError("EADP local tile count does not match the AnyRes grid")
    if len(local_selected_indices) != tiles:
        raise ValueError("EADP needs one selected-index vector per local tile")
    side = int(patches_per_side)
    if side * side != patch_count:
        raise ValueError("EADP local feature count does not match its crop grid")

    restored = local_features.new_zeros(tiles, patch_count, hidden_size)
    mask = torch.zeros(
        tiles, patch_count, dtype=torch.bool, device=local_features.device
    )
    for tile_index, selected in enumerate(local_selected_indices):
        selected = selected.to(device=local_features.device, dtype=torch.long)
        if selected.numel() and (
            int(selected.min()) < 0 or int(selected.max()) >= patch_count
        ):
            raise ValueError("EADP selected index lies outside its crop")
        restored[tile_index, selected] = local_features[tile_index, selected]
        mask[tile_index, selected] = True

    restored = restored.view(
        grid_height, grid_width, side, side, hidden_size
    ).permute(0, 2, 1, 3, 4).contiguous()
    mask = mask.view(grid_height, grid_width, side, side).permute(
        0, 2, 1, 3
    ).contiguous()
    restored = restored.view(grid_height * side, grid_width * side, hidden_size)
    mask = mask.view(grid_height * side, grid_width * side)
    return restored.permute(2, 0, 1).contiguous(), mask
