"""AdaLLaVA-L/H schedulers and latency-token helpers for OV-Qwen2.

AdaLLaVA is operation compression, not token compression: neither variant
deletes, merges, or renumbers visual/text tokens.  The L variant schedules
whole decoder blocks.  The H variant schedules GQA-aligned attention-head
groups together with equally sized MLP channel groups.  Both variants keep the
paper's latency token, hard Gumbel sampling during joint training, and a fixed
budget shared by training and evaluation.

The released LLaVA-1.5 model uses a 16/32 fixed prefix.  OV-Qwen2 has 28
decoder layers.  A 14/28 prefix is therefore used whenever the requested
budget permits it.  A retained-FLOPs budget below 50% cannot be represented by
that prefix, so formal low-budget configs must explicitly provide a smaller
prefix and record that adaptation in ``depth_adaptation``.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn


ADALLAVA_METHODS = frozenset({"adallava", "adallava_l", "adallava_h"})


DEFAULT_CONFIG: Dict[str, Any] = {
    "method": "adallava_l",
    "scheduler_variant": "L",
    "num_prefix_layers": 14,
    "active_layers": None,
    "active_units": None,
    "original_layer_count": 28,
    "target_flops_retain_ratio": None,
    # ``None`` means derive the exact discrete point from active_layers/units.
    # Formal generated policies record it explicitly for provenance.
    "calibrated_flops_retain_ratio": None,
    "target_flops_tolerance": 0.02,
    "scheduler_tau": 5.0,
    "fixed_latency_training": True,
    "latency_token_position": "after_last_ignored_prompt_token",
    "depth_adaptation": "16_of_32_to_14_of_28",
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "head_group_size": 7,
    "token_compression": False,
    "flops_scope": "llm_prefill_only",
    "image_only": False,
}


def normalize_adallava_config(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    if raw:
        cfg.update(raw)
    cfg["method"] = str(cfg["method"]).strip().lower()
    cfg["scheduler_variant"] = str(cfg["scheduler_variant"]).upper()
    cfg["num_prefix_layers"] = int(cfg["num_prefix_layers"])
    cfg["active_layers"] = (
        None if cfg["active_layers"] is None else int(cfg["active_layers"])
    )
    cfg["active_units"] = (
        None if cfg["active_units"] is None else int(cfg["active_units"])
    )
    cfg["original_layer_count"] = int(cfg["original_layer_count"])
    if cfg["target_flops_retain_ratio"] is None:
        raise ValueError(
            "AdaLLaVA requires an explicit target_flops_retain_ratio policy"
        )
    cfg["target_flops_retain_ratio"] = float(cfg["target_flops_retain_ratio"])
    cfg["calibrated_flops_retain_ratio"] = (
        None
        if cfg["calibrated_flops_retain_ratio"] is None
        else float(cfg["calibrated_flops_retain_ratio"])
    )
    cfg["target_flops_tolerance"] = float(cfg["target_flops_tolerance"])
    cfg["scheduler_tau"] = float(cfg["scheduler_tau"])
    cfg["fixed_latency_training"] = bool(cfg["fixed_latency_training"])
    cfg["num_attention_heads"] = int(cfg["num_attention_heads"])
    cfg["num_key_value_heads"] = int(cfg["num_key_value_heads"])
    cfg["head_group_size"] = int(cfg["head_group_size"])
    cfg["token_compression"] = bool(cfg["token_compression"])
    cfg["image_only"] = bool(cfg["image_only"])

    unknown = sorted(set(cfg) - set(DEFAULT_CONFIG))
    if unknown:
        raise ValueError("Unknown AdaLLaVA config keys: " + ", ".join(unknown))
    if cfg["method"] not in ADALLAVA_METHODS:
        raise ValueError("AdaLLaVA method must be adallava_l or adallava_h")
    if cfg["scheduler_variant"] not in {"L", "H"}:
        raise ValueError("AdaLLaVA scheduler_variant must be L or H")
    if cfg["method"] == "adallava_l" and cfg["scheduler_variant"] != "L":
        raise ValueError("adallava_l requires scheduler_variant=L")
    if cfg["method"] == "adallava_h" and cfg["scheduler_variant"] != "H":
        raise ValueError("adallava_h requires scheduler_variant=H")
    if cfg["method"] == "adallava" and cfg["scheduler_variant"] != "L":
        raise ValueError("legacy adallava is an alias for AdaLLaVA-L only")
    layers = cfg["original_layer_count"]
    prefix = cfg["num_prefix_layers"]
    if not 0 < prefix < layers:
        raise ValueError("AdaLLaVA requires 0 < prefix < decoder depth")
    if cfg["scheduler_tau"] <= 0:
        raise ValueError("AdaLLaVA scheduler_tau must be positive")
    if not 0.0 < cfg["target_flops_retain_ratio"] <= 1.0:
        raise ValueError("AdaLLaVA target FLOPs ratio must lie in (0, 1]")
    if not 0.0 <= cfg["target_flops_tolerance"] < 1.0:
        raise ValueError("AdaLLaVA target FLOPs tolerance is invalid")
    if cfg["token_compression"]:
        raise ValueError("AdaLLaVA must not delete or merge tokens")
    if cfg["flops_scope"] != "llm_prefill_only":
        raise ValueError("AdaLLaVA FLOPs scope must be llm_prefill_only")

    if cfg["scheduler_variant"] == "L":
        active = cfg["active_layers"]
        if active is None or not prefix <= active <= layers:
            raise ValueError("AdaLLaVA-L requires prefix <= active_layers <= depth")
        if cfg["active_units"] is not None:
            raise ValueError("AdaLLaVA-L must not define active_units")
        measured = active / float(layers)
    else:
        if cfg["active_layers"] is not None:
            raise ValueError("AdaLLaVA-H must not define active_layers")
        heads = cfg["num_attention_heads"]
        kv_heads = cfg["num_key_value_heads"]
        group = cfg["head_group_size"]
        if heads <= 0 or kv_heads <= 0 or heads % kv_heads or heads % group:
            raise ValueError("AdaLLaVA-H head geometry is not divisible")
        if group != heads // kv_heads:
            raise ValueError(
                "AdaLLaVA-H groups must align one Qwen2 GQA query group to one KV head"
            )
        groups_per_layer = heads // group
        active_units = cfg["active_units"]
        prefix_units = prefix * groups_per_layer
        total_units = layers * groups_per_layer
        if active_units is None or not prefix_units <= active_units <= total_units:
            raise ValueError(
                "AdaLLaVA-H active_units must include every fixed-prefix group"
            )
        measured = active_units / float(total_units)

    calibrated = cfg["calibrated_flops_retain_ratio"]
    if calibrated is None:
        # Backward compatibility for the original single AdaLLaVA-L config,
        # which stored its discrete point directly in the target field.
        calibrated = measured
        cfg["calibrated_flops_retain_ratio"] = measured
    if not math.isclose(measured, calibrated, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "AdaLLaVA calibrated FLOPs ratio does not match its execution units"
        )
    if abs(calibrated - cfg["target_flops_retain_ratio"]) > cfg[
        "target_flops_tolerance"
    ]:
        raise ValueError("AdaLLaVA discrete policy misses its target tolerance")
    if not cfg["fixed_latency_training"]:
        raise ValueError("AdaLLaVA training/evaluation budgets must be identical")
    if cfg["latency_token_position"] != "after_last_ignored_prompt_token":
        raise ValueError("Unsupported AdaLLaVA latency-token placement")
    return cfg


class _SchedulerFeedForward(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(256),
            nn.Linear(256, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class AdaLlavaLayerScheduler(nn.Module):
    """Paper-faithful whole-layer (L) scheduler without an einops dependency."""

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_count: int,
        prefix_layers: int,
        tau: float = 5.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_count = int(layer_count)
        self.prefix_layers = int(prefix_layers)
        self.tail_layers = self.layer_count - self.prefix_layers
        self.tau = float(tau)
        if self.tail_layers < 1:
            raise ValueError("AdaLLaVA scheduler needs a non-empty adaptive tail")
        self.mlp_head = nn.Linear(self.hidden_size, self.tail_layers, bias=True)
        self.scheduler_up_proj = _SchedulerFeedForward(self.hidden_size)

    def latency_encoding(
        self, active_layers: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        active = active_layers.to(dtype=torch.float32)
        tail = active - float(self.prefix_layers)
        if bool(((tail < 0) | (tail > self.tail_layers)).any()):
            raise ValueError("AdaLLaVA active-layer budget lies outside its tail")
        # This is the official latency_quantizing normalization.
        quantized = tail / float(self.tail_layers + 1)
        scaled = quantized * (2.0 * torch.pi)
        frequencies = 1.0 / (
            10000.0
            ** (
                torch.arange(128, device=scaled.device, dtype=torch.float32)
                / 128.0
            )
        )
        values = torch.cat(
            (
                torch.sin(scaled[:, None] * frequencies),
                torch.cos(scaled[:, None] * frequencies),
            ),
            dim=1,
        ).to(dtype=dtype)
        return self.scheduler_up_proj(values)

    def _sample_without_replacement(
        self, logits: torch.Tensor, count: int
    ) -> torch.Tensor:
        if count == 0:
            return logits.new_zeros(logits.shape)
        if count == logits.numel():
            return logits.new_ones(logits.shape)
        selected = torch.zeros_like(logits)
        unavailable = torch.zeros_like(logits, dtype=torch.bool)
        for _ in range(int(count)):
            candidate = logits.masked_fill(unavailable, float("-inf"))
            if self.training:
                uniform = torch.rand_like(candidate).clamp_(1e-10, 1.0 - 1e-10)
                candidate = candidate - torch.log(-torch.log(uniform))
            probabilities = torch.softmax(candidate / self.tau, dim=-1)
            index = int(probabilities.argmax(dim=-1).item())
            hard = F.one_hot(
                torch.tensor(index, device=logits.device),
                num_classes=int(logits.numel()),
            ).to(dtype=logits.dtype)
            straight_through = hard - probabilities.detach() + probabilities
            selected = selected + straight_through
            # Do not mutate a mask saved by ``masked_fill`` for backward.
            unavailable = unavailable | hard.detach().bool()
        return selected

    def forward(
        self, latency_hidden: torch.Tensor, active_layers: torch.Tensor
    ) -> torch.Tensor:
        if latency_hidden.ndim != 2:
            raise ValueError("AdaLLaVA scheduler expects [B,D] latency states")
        active = active_layers.to(device=latency_hidden.device, dtype=torch.long)
        if active.ndim == 0:
            active = active.expand(latency_hidden.shape[0])
        if active.shape != (latency_hidden.shape[0],):
            raise ValueError("AdaLLaVA budget count must match the batch")
        logits = self.mlp_head(latency_hidden)
        tail_plans = []
        for sample_logits, sample_active in zip(logits, active):
            tail_plans.append(
                self._sample_without_replacement(
                    sample_logits,
                    int(sample_active.item()) - self.prefix_layers,
                )
            )
        tail_plan = torch.stack(tail_plans, dim=0)
        prefix_plan = tail_plan.new_ones(
            (tail_plan.shape[0], self.prefix_layers)
        )
        return torch.cat((prefix_plan, tail_plan), dim=1)


class AdaLlavaHeadNeuronScheduler(nn.Module):
    """Paper H scheduler adapted to Qwen2 grouped-query attention.

    One schedulable unit activates one complete GQA query group (and its KV
    head) plus one equally sized group of MLP channels in the same decoder
    layer.  Qwen2-7B has 28 query heads and 4 KV heads, hence four groups of
    seven heads per layer.  This is the architecture-preserving counterpart of
    the released LLaVA-1.5 scheduler's eight-head groups.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        layer_count: int,
        prefix_layers: int,
        num_attention_heads: int,
        head_group_size: int,
        tau: float = 5.0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.layer_count = int(layer_count)
        self.prefix_layers = int(prefix_layers)
        self.tail_layers = self.layer_count - self.prefix_layers
        self.num_attention_heads = int(num_attention_heads)
        self.head_group_size = int(head_group_size)
        self.group_count = self.num_attention_heads // self.head_group_size
        self.tau = float(tau)
        if self.tail_layers < 1 or self.group_count < 1:
            raise ValueError("AdaLLaVA-H needs a non-empty adaptive tail")
        if self.num_attention_heads % self.head_group_size:
            raise ValueError("AdaLLaVA-H head_group_size must divide query heads")
        self.mlp_head = nn.Linear(
            self.hidden_size,
            self.tail_layers * 2 * self.group_count,
            bias=True,
        )
        self.scheduler_up_proj = _SchedulerFeedForward(self.hidden_size)

    @property
    def prefix_units(self) -> int:
        return self.prefix_layers * self.group_count

    @property
    def total_units(self) -> int:
        return self.layer_count * self.group_count

    @property
    def tail_capacity(self) -> int:
        return self.tail_layers * self.group_count

    def latency_encoding(
        self, active_units: torch.Tensor, *, dtype: torch.dtype
    ) -> torch.Tensor:
        active = active_units.to(dtype=torch.float32)
        tail = active - float(self.prefix_units)
        if bool(((tail < 0) | (tail > self.tail_capacity)).any()):
            raise ValueError("AdaLLaVA-H active-unit budget lies outside its tail")
        quantized = tail / float(self.tail_capacity + 1)
        scaled = quantized * (2.0 * torch.pi)
        frequencies = 1.0 / (
            10000.0
            ** (
                torch.arange(128, device=scaled.device, dtype=torch.float32)
                / 128.0
            )
        )
        values = torch.cat(
            (
                torch.sin(scaled[:, None] * frequencies),
                torch.cos(scaled[:, None] * frequencies),
            ),
            dim=1,
        ).to(dtype=dtype)
        return self.scheduler_up_proj(values)

    def _one_hot(self, logits: torch.Tensor) -> torch.Tensor:
        candidate = logits
        if self.training:
            uniform = torch.rand_like(candidate).clamp_(1e-10, 1.0 - 1e-10)
            candidate = candidate - torch.log(-torch.log(uniform))
        probabilities = torch.softmax(candidate / self.tau, dim=-1)
        index = probabilities.argmax(dim=-1)
        hard = F.one_hot(index, num_classes=int(logits.numel())).to(
            dtype=logits.dtype
        )
        return hard - probabilities.detach() + probabilities

    def _sample_one(
        self, logits: torch.Tensor, count: int
    ) -> torch.Tensor:
        if count == 0:
            return logits.new_zeros(
                self.tail_layers, 2, self.num_attention_heads
            )
        if count > self.tail_capacity:
            raise ValueError("AdaLLaVA-H tail budget exceeds available groups")

        selected = logits.new_zeros(self.tail_layers, 2, self.group_count)
        unavailable = torch.zeros_like(selected, dtype=torch.bool)
        for _ in range(int(count)):
            # Reproduce the released H scheduler's two-stage decision: choose
            # a layer, then choose one unused attention group and one unused
            # MLP group inside that layer.
            available_logits = logits.masked_fill(unavailable, float("-inf"))
            flat_layer_choice = self._one_hot(available_logits.reshape(-1))
            layer_choice = flat_layer_choice.view(
                self.tail_layers, 2, self.group_count
            ).sum(dim=(1, 2))
            layer_index = int(layer_choice.detach().argmax().item())

            operation_choices = []
            for operation in range(2):
                group_logits = logits[layer_index, operation].masked_fill(
                    unavailable[layer_index, operation], float("-inf")
                )
                operation_choices.append(self._one_hot(group_logits))
            operation_plan = torch.stack(operation_choices, dim=0)
            # Straight-through layer selection carries gradients into the
            # layer choice while retaining an exactly binary forward plan.
            selected = selected + (
                layer_choice[:, None, None] * operation_plan[None, :, :]
            )
            hard_groups = operation_plan.detach().argmax(dim=-1)
            newly_unavailable = torch.zeros_like(unavailable)
            for operation, group_index in enumerate(hard_groups.tolist()):
                newly_unavailable[
                    layer_index, operation, int(group_index)
                ] = True
            # ``masked_fill`` saves the boolean mask for backward.  Rebinding
            # avoids an in-place version bump across scheduler iterations.
            unavailable = unavailable | newly_unavailable

        head_plan = selected.repeat_interleave(self.head_group_size, dim=-1)
        if int(head_plan.shape[-1]) != self.num_attention_heads:
            raise RuntimeError("AdaLLaVA-H failed to expand its GQA groups")
        return head_plan

    def forward(
        self, latency_hidden: torch.Tensor, active_units: torch.Tensor
    ) -> torch.Tensor:
        if latency_hidden.ndim != 2:
            raise ValueError("AdaLLaVA-H scheduler expects [B,D] latency states")
        active = active_units.to(device=latency_hidden.device, dtype=torch.long)
        if active.ndim == 0:
            active = active.expand(latency_hidden.shape[0])
        if active.shape != (latency_hidden.shape[0],):
            raise ValueError("AdaLLaVA-H unit count must match the batch")
        tail_counts = active - int(self.prefix_units)
        if bool(((tail_counts < 0) | (tail_counts > self.tail_capacity)).any()):
            raise ValueError("AdaLLaVA-H active-unit budget is invalid")
        logits = self.mlp_head(latency_hidden).view(
            latency_hidden.shape[0], self.tail_layers, 2, self.group_count
        )
        tail_plans = [
            self._sample_one(sample_logits, int(sample_count.item()))
            for sample_logits, sample_count in zip(logits, tail_counts)
        ]
        tail_plan = torch.stack(tail_plans, dim=0)
        prefix_plan = tail_plan.new_ones(
            (
                tail_plan.shape[0],
                self.prefix_layers,
                2,
                self.num_attention_heads,
            )
        )
        return torch.cat((prefix_plan, tail_plan), dim=1)


def build_adallava_scheduler(config: Any, cfg: Dict[str, Any]) -> nn.Module:
    variant = str(cfg.get("scheduler_variant", "L")).upper()
    common = {
        "hidden_size": int(config.hidden_size),
        "layer_count": int(config.num_hidden_layers),
        "prefix_layers": int(cfg["num_prefix_layers"]),
        "tau": float(cfg["scheduler_tau"]),
    }
    if variant == "L":
        return AdaLlavaLayerScheduler(**common)
    if variant == "H":
        return AdaLlavaHeadNeuronScheduler(
            **common,
            num_attention_heads=int(config.num_attention_heads),
            head_group_size=int(cfg["head_group_size"]),
        )
    raise ValueError(f"Unsupported AdaLLaVA scheduler variant: {variant}")


def ensure_adallava_scheduler(decoder: nn.Module, cfg: Dict[str, Any]) -> nn.Module:
    scheduler = getattr(decoder, "adallava_scheduler", None)
    if scheduler is None:
        scheduler = build_adallava_scheduler(decoder.config, cfg)
        init = getattr(decoder, "_init_weights", None)
        if callable(init):
            scheduler.apply(init)
        decoder.add_module("adallava_scheduler", scheduler)
    expected = (
        int(decoder.config.hidden_size),
        int(decoder.config.num_hidden_layers),
        int(cfg["num_prefix_layers"]),
        str(cfg["scheduler_variant"]),
    )
    actual = (
        int(getattr(scheduler, "hidden_size", -1)),
        int(getattr(scheduler, "layer_count", -1)),
        int(getattr(scheduler, "prefix_layers", -1)),
        "H" if isinstance(scheduler, AdaLlavaHeadNeuronScheduler) else "L",
    )
    if actual != expected:
        raise ValueError(
            f"AdaLLaVA scheduler shape mismatch: expected={expected}, actual={actual}"
        )
    return scheduler


def _decoder(owner: nn.Module) -> Optional[nn.Module]:
    getter = getattr(owner, "get_model", None)
    if callable(getter):
        candidate = getter()
        if isinstance(candidate, nn.Module):
            return candidate
    candidate = getattr(owner, "model", None)
    return candidate if isinstance(candidate, nn.Module) else None


def maybe_insert_adallava_latency_token(
    owner: nn.Module,
    inputs_embeds: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    labels: Optional[torch.Tensor],
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    decoder = _decoder(owner)
    if decoder is None or inputs_embeds is None:
        return inputs_embeds, position_ids, attention_mask, labels
    cfg = getattr(decoder, "_ov_token_compression_config", None) or {}
    if cfg.get("method") not in ADALLAVA_METHODS:
        return inputs_embeds, position_ids, attention_mask, labels
    runtime = getattr(decoder, "_ov_token_compression_runtime", None)
    if runtime is None:
        raise RuntimeError("AdaLLaVA runtime metadata is unavailable")
    if runtime.get("adallava_latency_token_inserted"):
        return inputs_embeds, position_ids, attention_mask, labels

    scheduler = ensure_adallava_scheduler(decoder, cfg)
    batch, sequence, _ = inputs_embeds.shape
    budget_key = (
        "active_units" if cfg["scheduler_variant"] == "H" else "active_layers"
    )
    active = torch.full(
        (batch,),
        int(cfg[budget_key]),
        device=inputs_embeds.device,
        dtype=torch.long,
    )
    latency_embeddings = scheduler.latency_encoding(
        active, dtype=inputs_embeds.dtype
    )
    mask_was_none = attention_mask is None
    labels_were_none = labels is None
    positions_were_none = position_ids is None
    mask = (
        torch.ones(
            (batch, sequence),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        if attention_mask is None
        else attention_mask.to(device=inputs_embeds.device)
    )
    from llava.constants import IGNORE_INDEX

    label_rows = (
        torch.full(
            (batch, sequence),
            int(IGNORE_INDEX),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        if labels is None
        else labels.to(device=inputs_embeds.device)
    )
    insertion_positions = []
    new_embeds = []
    new_masks = []
    new_labels = []
    for row in range(batch):
        valid = mask[row].bool()
        ignored = ((label_rows[row] == int(IGNORE_INDEX)) & valid).nonzero(
            as_tuple=True
        )[0]
        if ignored.numel() == 0:
            raise ValueError("AdaLLaVA cannot locate the ignored prompt suffix")
        insertion = int(ignored[-1].item()) + 1
        insertion_positions.append(insertion)
        new_embeds.append(
            torch.cat(
                (
                    inputs_embeds[row, :insertion],
                    latency_embeddings[row : row + 1],
                    inputs_embeds[row, insertion:],
                ),
                dim=0,
            )
        )
        new_masks.append(
            torch.cat(
                (
                    mask[row, :insertion],
                    mask.new_ones((1,)),
                    mask[row, insertion:],
                ),
                dim=0,
            )
        )
        new_labels.append(
            torch.cat(
                (
                    label_rows[row, :insertion],
                    label_rows.new_full((1,), int(IGNORE_INDEX)),
                    label_rows[row, insertion:],
                ),
                dim=0,
            )
        )
    inputs_embeds = torch.stack(new_embeds, dim=0)
    mask = torch.stack(new_masks, dim=0)
    label_rows = torch.stack(new_labels, dim=0)
    if not positions_were_none:
        position_ids = mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(~mask.bool(), 0)
    if mask_was_none:
        attention_mask = None
    else:
        attention_mask = mask.to(dtype=attention_mask.dtype)
    labels = None if labels_were_none else label_rows.to(dtype=labels.dtype)

    runtime.update(
        {
            "adallava_latency_token_inserted": True,
            "adallava_latency_positions": torch.tensor(
                insertion_positions,
                device=inputs_embeds.device,
                dtype=torch.long,
            ),
            "adallava_scheduler_variant": str(cfg["scheduler_variant"]),
            "adallava_budget_key": budget_key,
            "adallava_active_budget": int(cfg[budget_key]),
            "adallava_prefix_layers": int(cfg["num_prefix_layers"]),
            "adallava_target_flops_retain_ratio": float(
                cfg["target_flops_retain_ratio"]
            ),
            "adallava_calibrated_flops_retain_ratio": float(
                cfg["calibrated_flops_retain_ratio"]
            ),
            "adallava_token_compression": False,
            "prefill_tokens_after": int(inputs_embeds.shape[1]),
        }
    )
    return inputs_embeds, position_ids, attention_mask, labels
