"""HeadRouter dynamic head-weight routing for LLM audio-token pruning.

This module ports the released HeadRouter equations into the platform's
descriptor-driven LLM-token executor.  The Qwen-specific executor owns token
layout, masks, absolute M-RoPE positions, cache lifecycle, and training loss;
this strategy owns the no-RoPE QK probe, Gaussian profile routing, and Top-K
selection.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping

from mllm_comp.compressors.base import CompressionContext
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.compressors.audio_rule_utils import audio_rule_metadata, token_scores, token_indices
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)


PROFILE_NAMES = ("semantic", "uniform", "acoustic")

# Released Qwen2.5-Omni-3B layer-1 calibration artifacts from HeadRouter.
OFFICIAL_QWEN25_OMNI_3B_PROFILES: dict[str, tuple[float, ...]] = {
    "semantic": (
        0.8986257513711806,
        0.000932261469425521,
        0.004217083859387259,
        0.0009898923578186168,
        0.003534318795342848,
        0.01497778528098744,
        0.000278350157139189,
        0.004011496112173126,
        0.000278350157139189,
        0.00029414360499537684,
        0.0012049019284003835,
        0.0041129488020734215,
        0.004143869092035752,
        0.003626696667623956,
        0.05425837226129246,
        0.004513778082984899,
    ),
    "uniform": (0.0625,) * 16,
    "acoustic": (
        0.004082073302105461,
        0.0246985778019487,
        0.0006746672855976127,
        0.004082073302105461,
        0.0246985778019487,
        0.0246985778019487,
        0.0006746672855976127,
        0.9041786351009512,
        0.0006746672855976127,
        0.0006746672855976127,
        0.0006746672855976127,
        0.004082073302105461,
        0.0006746672855976127,
        0.0006746672855976127,
        0.004082073302105461,
        0.0006746672855976127,
    ),
}
OFFICIAL_QWEN25_OMNI_3B_CENTERS = {
    "semantic": 0.030,
    "uniform": 0.036,
    "acoustic": 0.042,
}
OFFICIAL_ROUTING_SIGMA = 0.006


@dataclass(frozen=True)
class RoutingResult:
    """Gaussian soft-routing result for one input sample."""

    weights: Any
    coefficients: dict[str, float]
    selectivity: Any
    spread: float


@dataclass(frozen=True)
class HeadRouterSpec:
    """Normalized HeadRouter configuration.

    ``pruning_layer`` follows the released implementation: it is the zero-based
    decoder layer that consumes the pruned sequence.  Its preceding layer
    supplies the no-RoPE Q/K projections, so the platform compression boundary
    is ``pruning_layer - 1``.
    """

    pruning_ratio: float = 0.6
    pruning_layer: int = 2
    minimum_keep_tokens: int = 1
    profiles: tuple[tuple[str, tuple[float, ...]], ...] = tuple(
        (name, OFFICIAL_QWEN25_OMNI_3B_PROFILES[name]) for name in PROFILE_NAMES
    )
    profile_centers: tuple[tuple[str, float], ...] = tuple(
        (name, OFFICIAL_QWEN25_OMNI_3B_CENTERS[name]) for name in PROFILE_NAMES
    )
    routing_sigma: float = OFFICIAL_ROUTING_SIGMA
    profile_source: str = "official_qwen2_5_omni_3b_layer1"
    score_source: str = "position_agnostic_text_to_audio_qk"
    target_site: str = "llm_decoder_layer_output_audio_tokens"
    lifecycle: str = "prefill_only"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HeadRouterSpec":
        operation = _mapping(config.get("operation"), "operation")
        criterion = _mapping(config.get("criterion"), "criterion")
        schedule = _mapping(config.get("schedule"), "schedule")
        activation_scope = _mapping(
            config.get("activation_scope"), "activation_scope"
        )
        legacy = _mapping(config.get("config"), "config")

        raw_pruning_layer = _first_present(
            schedule.get("pruning_layer"),
            activation_scope.get("pruning_layer"),
            legacy.get("pruning_layer"),
        )
        raw_layers = activation_scope.get("layers")
        if raw_pruning_layer is None and raw_layers is not None:
            if (
                not isinstance(raw_layers, (list, tuple))
                or len(raw_layers) != 1
                or isinstance(raw_layers[0], bool)
                or not isinstance(raw_layers[0], int)
            ):
                raise ValueError(
                    "HeadRouter activation_scope.layers must contain exactly one "
                    "zero-based Q/K probe layer."
                )
            raw_pruning_layer = int(raw_layers[0]) + 1
        pruning_layer = _strict_int(
            cls.pruning_layer if raw_pruning_layer is None else raw_pruning_layer,
            "pruning_layer",
        )

        profile_dir = legacy.get("profile_dir")
        inline_profiles = legacy.get("profiles")
        if profile_dir is not None and inline_profiles is not None:
            raise ValueError(
                "HeadRouter accepts either config.profile_dir or config.profiles, "
                "not both."
            )
        if profile_dir is not None:
            profiles = _load_profiles(Path(str(profile_dir)))
            profile_source = str(Path(str(profile_dir)).expanduser())
        elif inline_profiles is not None:
            profiles = _normalize_profile_mapping(inline_profiles)
            profile_source = "inline"
        else:
            profiles = dict(OFFICIAL_QWEN25_OMNI_3B_PROFILES)
            profile_source = cls.profile_source

        centers = dict(OFFICIAL_QWEN25_OMNI_3B_CENTERS)
        sigma = cls.routing_sigma
        routing_config = legacy.get("routing_config")
        if routing_config is not None:
            centers, sigma = _load_routing_config(Path(str(routing_config)))
        if legacy.get("routing_centers") is not None:
            centers = _normalize_centers(legacy["routing_centers"])
        if legacy.get("routing_sigma") is not None:
            sigma = float(legacy["routing_sigma"])

        spec = cls(
            pruning_ratio=float(
                _first_present(
                    schedule.get("pruning_ratio"),
                    legacy.get("pruning_ratio"),
                    cls.pruning_ratio,
                )
            ),
            pruning_layer=pruning_layer,
            minimum_keep_tokens=_strict_int(
                _first_present(
                    schedule.get("minimum_keep_tokens"),
                    legacy.get("minimum_keep_tokens"),
                    cls.minimum_keep_tokens,
                ),
                "minimum_keep_tokens",
            ),
            profiles=tuple((name, tuple(profiles[name])) for name in PROFILE_NAMES),
            profile_centers=tuple(
                (name, float(centers[name])) for name in PROFILE_NAMES
            ),
            routing_sigma=float(sigma),
            profile_source=profile_source,
            score_source=str(
                _first_present(
                    criterion.get("score_source"),
                    legacy.get("score_source"),
                    cls.score_source,
                )
            ),
            target_site=str(
                _first_present(
                    operation.get("target_site"),
                    legacy.get("target_site"),
                    cls.target_site,
                )
            ),
            lifecycle=str(
                _first_present(
                    activation_scope.get("lifecycle"),
                    legacy.get("lifecycle"),
                    cls.lifecycle,
                )
            ),
        )
        spec.validate()
        if operation.get("type") not in {None, "keep"}:
            raise ValueError("HeadRouter requires operation.type=keep.")
        return spec

    @property
    def profile_mapping(self) -> dict[str, tuple[float, ...]]:
        return dict(self.profiles)

    @property
    def center_mapping(self) -> dict[str, float]:
        return dict(self.profile_centers)

    @property
    def probe_layer(self) -> int:
        return self.pruning_layer - 1

    def validate(self) -> None:
        if not math.isfinite(self.pruning_ratio) or not 0 <= self.pruning_ratio < 1:
            raise ValueError("HeadRouter pruning_ratio must be in [0, 1).")
        if self.pruning_layer < 1:
            raise ValueError("HeadRouter pruning_layer must be at least 1.")
        if self.minimum_keep_tokens < 1:
            raise ValueError("HeadRouter minimum_keep_tokens must be positive.")
        _normalize_profile_mapping(self.profile_mapping)
        _normalize_centers(self.center_mapping)
        if not math.isfinite(self.routing_sigma) or self.routing_sigma <= 0:
            raise ValueError("HeadRouter routing_sigma must be positive and finite.")
        if self.score_source != "position_agnostic_text_to_audio_qk":
            raise ValueError(
                "HeadRouter requires score_source="
                "'position_agnostic_text_to_audio_qk'."
            )
        if self.lifecycle != "prefill_only":
            raise ValueError("HeadRouter lifecycle must be 'prefill_only'.")


def headrouter_output_length(
    current_length: int,
    pruning_ratio: float,
    *,
    minimum_keep_tokens: int = 1,
) -> int:
    """Return the released implementation's floor-based Top-K budget."""

    if current_length < 0:
        raise ValueError("current_length must be non-negative")
    if current_length == 0:
        return 0
    if not math.isfinite(pruning_ratio) or not 0 <= pruning_ratio < 1:
        raise ValueError("pruning_ratio must be in [0, 1)")
    if minimum_keep_tokens < 1:
        raise ValueError("minimum_keep_tokens must be positive")
    return min(
        current_length,
        max(minimum_keep_tokens, int(current_length * (1.0 - pruning_ratio))),
    )


def compute_selectivity(per_head_attention: Any) -> Any:
    """Compute Eq. (10), entropy-normalized selectivity for every head."""

    probabilities = _normalize_attention(per_head_attention.float())
    num_audio_tokens = int(probabilities.shape[-1])
    if num_audio_tokens == 1:
        return probabilities.new_ones((int(probabilities.shape[0]),))
    entropy = -(
        probabilities * probabilities.clamp_min(1e-12).log()
    ).sum(dim=-1)
    return (1.0 - entropy / math.log(num_audio_tokens)).clamp(0.0, 1.0)


def gaussian_route(
    per_head_attention: Any,
    profiles: Mapping[str, Any],
    centers: Mapping[str, float],
    sigma: float,
) -> RoutingResult:
    """Compute Eqs. (11)-(14), Gaussian soft profile routing."""

    torch = _torch()
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be a positive finite value")
    selectivity = compute_selectivity(per_head_attention)
    spread_tensor = selectivity.std(unbiased=False)
    spread = float(spread_tensor.item())
    normalized_profiles = _tensor_profiles(
        profiles,
        centers,
        num_heads=int(per_head_attention.shape[0]),
        device=per_head_attention.device,
    )
    logits = torch.tensor(
        [
            -((spread - float(centers[name])) ** 2) / (2.0 * sigma**2)
            for name in PROFILE_NAMES
        ],
        dtype=torch.float32,
        device=per_head_attention.device,
    )
    coefficients = torch.softmax(logits, dim=0)
    stacked = torch.stack([normalized_profiles[name] for name in PROFILE_NAMES])
    weights = (coefficients[:, None] * stacked).sum(dim=0)
    weights = weights / weights.sum()
    return RoutingResult(
        weights=weights,
        coefficients={
            name: float(coefficients[index].item())
            for index, name in enumerate(PROFILE_NAMES)
        },
        selectivity=selectivity,
        spread=spread,
    )


def weighted_token_importance(
    per_head_attention: Any,
    head_weights: Any,
) -> Any:
    """Compute Eq. (8), routed head-weighted audio-token importance."""

    torch = _torch()
    probabilities = _normalize_attention(per_head_attention.float())
    weights = torch.as_tensor(
        head_weights,
        dtype=torch.float32,
        device=probabilities.device,
    )
    if weights.ndim != 1 or int(weights.numel()) != int(probabilities.shape[0]):
        raise ValueError("head_weights must contain one value per attention head")
    if (
        not bool(torch.isfinite(weights).all().item())
        or bool((weights < 0).any().item())
        or float(weights.sum().item()) <= 0
    ):
        raise ValueError("head_weights must contain finite non-negative values")
    weights = weights / weights.sum()
    return (probabilities * weights[:, None]).sum(dim=0)


class HeadRouterLLMStrategy:
    """No-RoPE QK probe, dynamic routing, and exact chronological selection."""

    def __init__(self, spec: HeadRouterSpec) -> None:
        self.spec = spec
        self._last_routes: list[dict[str, Any]] = []

    def stage_layer_indices(self, num_layers: int) -> tuple[int, ...]:
        if self.spec.probe_layer >= num_layers:
            raise InvariantError(
                "HeadRouter probe layer is outside the decoder: "
                f"probe_layer={self.spec.probe_layer}, num_layers={num_layers}."
            )
        return (self.spec.probe_layer,)

    def reset_runtime_state(self) -> None:
        self._last_routes = []

    def output_length(self, current_length: int) -> int:
        return headrouter_output_length(
            current_length,
            self.spec.pruning_ratio,
            minimum_keep_tokens=self.spec.minimum_keep_tokens,
        )

    def runtime_metadata(self, stage_layers: tuple[int, ...]) -> dict[str, Any]:
        return {
            "target_site": self.spec.target_site,
            "score_source": self.spec.score_source,
            "pruning_ratio": self.spec.pruning_ratio,
            "qk_probe_layer_zero_based": self.spec.probe_layer,
            "pruning_layer_zero_based": self.spec.pruning_layer,
            "compression_boundary_after_layer_zero_based": stage_layers[0],
            "minimum_keep_tokens": self.spec.minimum_keep_tokens,
            "lifecycle": self.spec.lifecycle,
            "prefill_compression": True,
            "decode_compression": False,
            "position_policy": "preserve_absolute",
            "position_encoding_in_probe": "none",
            "probe_hidden_source": "input_embeddings",
            "query_scope": "instruction_text_tokens",
            "profile_source": self.spec.profile_source,
            "profile_centers": self.spec.center_mapping,
            "routing_sigma": self.spec.routing_sigma,
            "last_routes": list(self._last_routes) or None,
            **audio_rule_metadata(self),
        }

    def score_audio_tokens(
        self,
        *,
        attention_module: Any,
        normalized_hidden_states: Any,
        position_embeddings: Any | None,
        absolute_positions: Any,
        instruction_mask: Any,
        valid_mask: Any,
        audio_mask: Any,
        initial_hidden_states: Any | None = None,
    ) -> Any:
        """Run the released position-agnostic per-head text-to-audio probe."""

        del position_embeddings, absolute_positions
        torch = _torch()
        if initial_hidden_states is None:
            raise InvariantError(
                "HeadRouter requires the original LLM input embeddings for its "
                "released no-RoPE QK probe."
            )
        hidden = initial_hidden_states
        if tuple(hidden.shape) != tuple(normalized_hidden_states.shape):
            raise InvariantError(
                "HeadRouter input-embedding layout does not match the probe "
                "boundary layout."
            )
        batch_size, sequence_length, _ = hidden.shape
        num_heads = int(attention_module.config.num_attention_heads)
        num_key_value_heads = int(attention_module.config.num_key_value_heads)
        head_dim = int(getattr(attention_module, "head_dim", 0))
        if head_dim <= 0:
            head_dim = int(attention_module.q_proj.out_features // num_heads)
        if num_heads % num_key_value_heads != 0:
            raise InvariantError(
                "HeadRouter requires num_attention_heads to be divisible by "
                "num_key_value_heads."
            )
        profile_heads = {len(weights) for weights in self.spec.profile_mapping.values()}
        if profile_heads != {num_heads}:
            raise InvariantError(
                "HeadRouter calibrated profiles do not match this decoder: "
                f"profile_heads={sorted(profile_heads)}, model_heads={num_heads}. "
                "Provide config.profile_dir or config.profiles calibrated for "
                "the selected checkpoint."
            )

        with torch.no_grad():
            query = attention_module.q_proj(hidden).view(
                batch_size, sequence_length, num_heads, head_dim
            ).transpose(1, 2)
            key = attention_module.k_proj(hidden).view(
                batch_size, sequence_length, num_key_value_heads, head_dim
            ).transpose(1, 2)
            groups = num_heads // num_key_value_heads
            if groups > 1:
                key = key.repeat_interleave(groups, dim=1)
            scale = float(getattr(attention_module, "scaling", head_dim**-0.5))
            scores = hidden.new_zeros(
                (batch_size, sequence_length), dtype=torch.float32
            )
            routes: list[dict[str, Any]] = []
            for batch_index in range(batch_size):
                text_indices = torch.nonzero(
                    instruction_mask[batch_index] & valid_mask[batch_index],
                    as_tuple=False,
                ).flatten()
                audio_indices = torch.nonzero(
                    audio_mask[batch_index] & valid_mask[batch_index],
                    as_tuple=False,
                ).flatten()
                if int(audio_indices.numel()) == 0:
                    continue
                if int(text_indices.numel()) == 0:
                    raise InvariantError(
                        "HeadRouter requires at least one instruction text token "
                        f"for sample {batch_index}."
                    )
                text_query = query[batch_index, :, text_indices, :]
                audio_key = key[batch_index, :, audio_indices, :]
                with torch.autocast(
                    device_type=text_query.device.type,
                    enabled=False,
                ):
                    affinity = torch.matmul(
                        text_query.float(),
                        audio_key.float().transpose(-2, -1),
                    ) * scale
                    per_head_attention = torch.softmax(
                        affinity,
                        dim=-1,
                        dtype=torch.float32,
                    ).mean(dim=1)
                route = gaussian_route(
                    per_head_attention,
                    self.spec.profile_mapping,
                    self.spec.center_mapping,
                    self.spec.routing_sigma,
                )
                importance = weighted_token_importance(
                    per_head_attention,
                    route.weights,
                )
                scorer = getattr(self, "rules", {}).get("scorer")
                if scorer is not None:
                    importance = token_scores(scorer({"values": hidden[batch_index, audio_indices].detach(),
                        "scores": importance, "per_head_attention": per_head_attention,
                        "profiles": self.spec.profile_mapping, "centers": self.spec.center_mapping,
                        "routing_sigma": self.spec.routing_sigma}), hidden, length=int(audio_indices.numel()))
                scores[batch_index, audio_indices] = importance
                routes.append(
                    {
                        "sample_index": batch_index,
                        "selectivity_spread": route.spread,
                        "coefficients": route.coefficients if scorer is None else None,
                        "custom_scorer": scorer is not None,
                    }
                )
            if not bool(torch.isfinite(scores).all().item()):
                raise InvariantError("HeadRouter importance contains NaN or Inf.")
            self._last_routes = routes
            return scores

    def select_indices(self, scores: Any, keep_count: int) -> Any:
        if not 1 <= keep_count <= int(scores.numel()):
            raise InvariantError(
                f"HeadRouter keep_count={keep_count} is invalid for "
                f"{int(scores.numel())} tokens."
            )
        selector = getattr(self, "rules", {}).get("selector")
        if selector is not None and keep_count != int(scores.numel()):
            return token_indices(selector({"scores": scores.detach(), "keep_count": keep_count}),
                length=int(scores.numel()), count=keep_count, device=scores.device)
        return scores.topk(keep_count).indices.sort().values.to(dtype=_torch().long)


class HeadRouter:
    name = "headrouter_dynamic_audio_pruning"
    version = "1"

    def validate_config(self, config: dict[str, Any]) -> None:
        HeadRouterSpec.from_config(config)

    def plan(
        self,
        *,
        state: ForwardState,
        context: CompressionContext,
        config: dict[str, Any],
    ) -> CompressionPlan:
        del context
        spec = HeadRouterSpec.from_config(config)
        actions: list[TokenAction] = []
        original_total = 0
        final_total = 0
        unit_index = 0
        for span in state.modality_token_spans:
            if span.modality != "audio":
                continue
            length = span.end - span.start
            keep_count = headrouter_output_length(
                length,
                spec.pruning_ratio,
                minimum_keep_tokens=spec.minimum_keep_tokens,
            )
            original_total += length
            final_total += keep_count
            actions.append(
                TokenAction(
                    op="keep",
                    target_indices=None,
                    metadata={
                        "selection_deferred": True,
                        "selector": self.name,
                        "target_site": spec.target_site,
                        "sample_index": span.sample_index,
                        "audio_unit_index": unit_index,
                        "span_start": span.start,
                        "span_end": span.end,
                        "original_count": length,
                        "keep_count": keep_count,
                        "stage_lengths": [length, keep_count],
                        "stage_layers": [spec.probe_layer],
                        "pruning_layer": spec.pruning_layer,
                        "lifecycle": spec.lifecycle,
                        "token_order": "chronological",
                    },
                )
            )
            unit_index += 1
        stage_name = str(config.get("_stage_name") or self.name)
        execution_phase = ExecutionPhase(
            config.get("_execution_phase", ExecutionPhase.GENERATE)
        )
        return CompressionPlan(
            stage_name=stage_name,
            placement=Placement.LLM_LEVEL,
            execution_phase=execution_phase,
            modalities={Modality.AUDIO},
            token_roles={TokenRole.OBSERVATION},
            method=CompressionMethod.TOKEN_COMPRESSION,
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype=spec.target_site,
            token_actions=actions,
            legalization_policy=LegalizationPolicy(
                position_policy="preserve_absolute",
                mask_policy="rebuild_from_tokens",
                label_policy="rebuild_ignore_index",
                coordinate_policy="drop_removed",
            ),
            this_stage_metadata=StageMetadata(
                stage_name=stage_name,
                changed_spans=[
                    span
                    for span in state.modality_token_spans
                    if span.modality == "audio"
                ],
                stats={
                    "target_site": spec.target_site,
                    "score_source": spec.score_source,
                    "lifecycle": spec.lifecycle,
                    "qk_probe_layer_zero_based": spec.probe_layer,
                    "pruning_layer_zero_based": spec.pruning_layer,
                    "original_audio_tokens": original_total,
                    "planned_final_audio_tokens": final_total,
                    "selection_deferred_to_prefill": True,
                },
            ),
        )

    def state_dict(self) -> dict[str, Any]:
        return {}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state:
            raise ValueError("HeadRouter has no trainable state")


def _normalize_attention(per_head_attention: Any) -> Any:
    torch = _torch()
    if per_head_attention.ndim != 2:
        raise ValueError("per_head_attention must have shape [heads, audio_tokens]")
    if int(per_head_attention.shape[0]) == 0 or int(per_head_attention.shape[1]) == 0:
        raise ValueError("per_head_attention cannot have an empty dimension")
    if not bool(torch.isfinite(per_head_attention).all().item()):
        raise ValueError("per_head_attention must contain only finite values")
    if bool((per_head_attention < 0).any().item()):
        raise ValueError("per_head_attention must be non-negative")
    row_sums = per_head_attention.sum(dim=-1, keepdim=True)
    if bool((row_sums <= 0).any().item()):
        raise ValueError("each attention head must have positive mass")
    return per_head_attention / row_sums


def _tensor_profiles(
    profiles: Mapping[str, Any],
    centers: Mapping[str, float],
    *,
    num_heads: int,
    device: Any,
) -> dict[str, Any]:
    torch = _torch()
    if not isinstance(profiles, Mapping) or set(profiles) != set(PROFILE_NAMES):
        raise ValueError(
            "HeadRouter profiles must contain semantic, uniform, and acoustic."
        )
    _normalize_centers(centers)
    result: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        tensor = torch.as_tensor(
            profiles[name],
            dtype=torch.float32,
            device=device,
        )
        if tensor.ndim != 1 or int(tensor.numel()) != num_heads:
            raise ValueError(f"profile {name!r} must contain {num_heads} weights")
        if (
            not bool(torch.isfinite(tensor).all().item())
            or bool((tensor < 0).any().item())
            or float(tensor.sum().item()) <= 0
        ):
            raise ValueError(
                f"profile {name!r} must contain finite non-negative weights"
            )
        result[name] = tensor / tensor.sum()
    return result


def _normalize_profile_mapping(value: Any) -> dict[str, tuple[float, ...]]:
    if not isinstance(value, Mapping) or set(value) != set(PROFILE_NAMES):
        raise ValueError(
            "HeadRouter profiles must contain semantic, uniform, and acoustic."
        )
    result: dict[str, tuple[float, ...]] = {}
    head_counts: set[int] = set()
    for name in PROFILE_NAMES:
        weights = value[name]
        if not isinstance(weights, (list, tuple)) or not weights:
            raise ValueError(f"HeadRouter profile {name!r} must be a non-empty list.")
        converted = tuple(float(item) for item in weights)
        if any(not math.isfinite(item) or item < 0 for item in converted) or sum(
            converted
        ) <= 0:
            raise ValueError(
                f"HeadRouter profile {name!r} must contain finite non-negative weights."
            )
        result[name] = converted
        head_counts.add(len(converted))
    if len(head_counts) != 1:
        raise ValueError("All HeadRouter profiles must have the same head count.")
    return result


def _normalize_centers(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping) or set(value) != set(PROFILE_NAMES):
        raise ValueError(
            "HeadRouter routing centers must contain semantic, uniform, and acoustic."
        )
    result = {name: float(value[name]) for name in PROFILE_NAMES}
    if any(not math.isfinite(item) for item in result.values()):
        raise ValueError("HeadRouter routing centers must be finite.")
    return result


def _load_profiles(profile_dir: Path) -> dict[str, tuple[float, ...]]:
    directory = profile_dir.expanduser()
    profiles: dict[str, Any] = {}
    for name in PROFILE_NAMES:
        path = directory / f"{name}.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot load HeadRouter profile {path}: {exc}") from exc
        weights = data.get("weights")
        if weights is None:
            layer_keys = [
                key
                for key in data
                if key.startswith("layer") and key.endswith("_weights")
            ]
            if len(layer_keys) != 1:
                raise ValueError(
                    f"{path} must contain 'weights' or one layer weights array."
                )
            weights = data[layer_keys[0]]
        profiles[name] = weights
    return _normalize_profile_mapping(profiles)


def _load_routing_config(path: Path) -> tuple[dict[str, float], float]:
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot load HeadRouter routing config {path}: {exc}") from exc
    centers = _normalize_centers(data.get("centers"))
    sigma = float(data.get("sigma"))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("HeadRouter routing sigma must be positive and finite.")
    return centers, sigma


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _first_present(*values: Any) -> Any:
    return next((item for item in values if item is not None), None)


def _strict_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return int(value)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise InvariantError("HeadRouter requires PyTorch.") from exc
    return torch


__all__ = [
    "HeadRouter",
    "HeadRouterLLMStrategy",
    "HeadRouterSpec",
    "OFFICIAL_QWEN25_OMNI_3B_CENTERS",
    "OFFICIAL_QWEN25_OMNI_3B_PROFILES",
    "OFFICIAL_ROUTING_SIGMA",
    "RoutingResult",
    "compute_selectivity",
    "gaussian_route",
    "headrouter_output_length",
    "weighted_token_importance",
]
