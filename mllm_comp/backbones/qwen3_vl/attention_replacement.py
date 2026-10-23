"""Qwen3-VL attention replacements used by operation-compression experiments.

The replacements preserve the public ``Qwen3VLTextAttention.forward`` contract
while changing the token mixer itself.  They deliberately reuse the checkpoint's
projection and normalization modules so a replacement can be installed without
copying the full attention block.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields
from typing import Any
from weakref import WeakMethod

from mllm_comp.core.errors import BackendIntegrationError

try:  # Keep CPU/schema-only imports usable without the optional torch extra.
    import torch
    import torch.nn.functional as F
    from torch import nn
except ImportError:  # pragma: no cover - exercised by no-torch contract jobs.
    torch = None
    F = None
    nn = None


def _missing_torch() -> BackendIntegrationError:
    return BackendIntegrationError(
        "Qwen3-VL attention replacement requires PyTorch in the runtime environment."
    )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(dim) for dim in shape)


if nn is not None:

    @dataclass
    class _RecurrentState:
        """Generation history; every tensor field uses the cache's batch axis."""

        seen_tokens: int

        def reorder(self, beam_idx: Any) -> None:
            for field in fields(self):
                value = getattr(self, field.name)
                if isinstance(value, torch.Tensor):
                    setattr(self, field.name, value.index_select(0, beam_idx.to(value.device)))

    @dataclass
    class _LoLCATsState(_RecurrentState):
        window_size: int
        k_state: Any
        kv_state: Any
        window_keys: Any
        window_values: Any
        window_features: Any
        window_keep: Any

    @dataclass
    class _NHAState(_RecurrentState):
        long_keys: Any
        long_values: Any
        window_keys: Any
        window_rope_keys: Any
        window_values: Any
        window_retentions: Any
        window_keep: Any

    class _RecurrentCacheStates:
        """Keep replacement history with its native cache, including beam order."""

        def __init__(self, cache: Any) -> None:
            self.layers: dict[int, _RecurrentState] = {}
            # A bound native method would retain the cache through this owner.
            self._native_reorder = WeakMethod(cache.reorder_cache)

        def reorder_cache(self, beam_idx: Any) -> None:
            self._native_reorder()(beam_idx)
            for state in self.layers.values():
                state.reorder(beam_idx)

    def _cache_state(cache: Any, layer_idx: int) -> _RecurrentState | None:
        states = getattr(cache, "_mllm_comp_attention_states", None)
        return None if states is None else states.layers.get(layer_idx)

    def _store_cache_state(cache: Any, layer_idx: int, state: _RecurrentState) -> None:
        states = getattr(cache, "_mllm_comp_attention_states", None)
        if states is None:
            states = _RecurrentCacheStates(cache)
            cache._mllm_comp_attention_states = states
            cache.reorder_cache = states.reorder_cache
        states.layers[layer_idx] = state

    def _repeat_kv(hidden_states: Any, repetitions: int) -> Any:
        if repetitions == 1:
            return hidden_states
        batch, key_value_heads, sequence_length, head_dim = hidden_states.shape
        expanded = hidden_states[:, :, None, :, :].expand(
            batch,
            key_value_heads,
            repetitions,
            sequence_length,
            head_dim,
        )
        return expanded.reshape(
            batch,
            key_value_heads * repetitions,
            sequence_length,
            head_dim,
        )

    def _apply_qwen3_vl_rope(
        query: Any, key: Any, position_embeddings: Any
    ) -> tuple[Any, Any]:
        if position_embeddings is None:
            return query, key
        try:
            from transformers.models.qwen3_vl.modeling_qwen3_vl import (
                apply_rotary_pos_emb,
            )
        except ImportError as exc:  # pragma: no cover - needs the real runtime.
            raise BackendIntegrationError(
                "Cannot import Qwen3-VL rotary-position helper from Transformers."
            ) from exc
        cos, sin = position_embeddings
        return apply_rotary_pos_emb(query, key, cos, sin)

    def _update_cache(
        past_key_values: Any,
        key_states: Any,
        value_states: Any,
        *,
        layer_idx: int,
        kwargs: dict[str, Any],
        position_embeddings: Any | None = None,
    ) -> tuple[Any, Any]:
        if past_key_values is None:
            return key_states, value_states
        cache_kwargs = {
            name: kwargs[name]
            for name in ("cache_position",)
            if kwargs.get(name) is not None
        }
        if position_embeddings is not None:
            cache_kwargs["cos"], cache_kwargs["sin"] = position_embeddings
        try:
            return past_key_values.update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs,
            )
        except TypeError:
            return past_key_values.update(key_states, value_states, layer_idx)

    def _cache_length(past_key_values: Any, layer_idx: int) -> int:
        if past_key_values is None:
            return 0
        getter = getattr(past_key_values, "get_seq_length", None)
        if not callable(getter):
            return 0
        try:
            return int(getter(layer_idx))
        except TypeError:
            return int(getter())

    def _key_keep_mask(attention_mask: Any, total_length: int) -> Any | None:
        if attention_mask is None or not hasattr(attention_mask, "ndim"):
            return None
        if attention_mask.ndim == 2:
            mask = attention_mask[:, -total_length:]
            return mask.to(dtype=torch.bool)
        if attention_mask.ndim == 4:
            mask = attention_mask[:, 0, -1, -total_length:]
            if mask.dtype == torch.bool:
                return mask
            return mask > -1.0e4
        return None

    class _Qwen3VLReplacementBase(nn.Module):
        strategy_name = "unknown"

        def __init__(self, original_attention: Any, *, debug: bool = False) -> None:
            super().__init__()
            required = ("config", "layer_idx", "v_proj", "o_proj")
            missing = [
                name for name in required if not hasattr(original_attention, name)
            ]
            if missing:
                raise BackendIntegrationError(
                    "Qwen3-VL attention replacement received an incompatible module; "
                    f"missing attributes: {missing}."
                )
            self.config = original_attention.config
            self.layer_idx = int(original_attention.layer_idx)
            self.head_dim = int(
                getattr(
                    original_attention,
                    "head_dim",
                    getattr(self.config, "head_dim", 0),
                )
            )
            self.num_heads = int(getattr(self.config, "num_attention_heads", 0))
            if self.head_dim <= 0 or self.num_heads <= 0:
                raise BackendIntegrationError(
                    "Qwen3-VL attention replacement could not resolve head_dim/num_heads."
                )
            self.num_key_value_heads = int(
                getattr(
                    self.config,
                    "num_key_value_heads",
                    original_attention.v_proj.out_features // self.head_dim,
                )
            )
            if self.num_heads % self.num_key_value_heads:
                raise BackendIntegrationError(
                    "Qwen3-VL attention heads must be divisible by key/value heads."
                )
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
            self.v_proj = original_attention.v_proj
            self.o_proj = original_attention.o_proj
            self.debug = bool(debug)
            self.forward_calls = 0
            self.last_debug: dict[str, Any] = {}
            self.train(bool(getattr(original_attention, "training", False)))

        def _record_debug(self, hidden_states: Any, output: Any) -> None:
            self.forward_calls += 1
            self.last_debug = {
                "strategy": self.strategy_name,
                "layer_idx": self.layer_idx,
                "forward_calls": self.forward_calls,
                "input_shape": _shape(hidden_states),
                "output_shape": _shape(output),
                "dtype": str(getattr(output, "dtype", None)),
                "device": str(getattr(output, "device", None)),
            }
            if self.debug:
                finite = torch.isfinite(output.detach()).all()
                self.last_debug["all_finite"] = bool(finite.item())

    class Qwen3VLLinearAttention(_Qwen3VLReplacementBase):
        """Causal ELU-feature linear attention with Qwen3-VL projections/cache."""

        strategy_name = "attn_linear_attention"

        def __init__(
            self,
            original_attention: Any,
            *,
            eps: float = 1.0e-6,
            debug: bool = False,
        ) -> None:
            super().__init__(original_attention, debug=debug)
            for name in ("q_proj", "k_proj"):
                if not hasattr(original_attention, name):
                    raise BackendIntegrationError(
                        f"Linear attention requires Qwen3-VL {name}."
                    )
            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            self.q_norm = getattr(original_attention, "q_norm", nn.Identity())
            self.k_norm = getattr(original_attention, "k_norm", nn.Identity())
            self.eps = float(eps)
            if self.eps <= 0:
                raise ValueError("linear-attention eps must be positive.")

        @staticmethod
        def _feature_map(value: Any) -> Any:
            return F.elu(value.float()) + 1.0

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = _apply_qwen3_vl_rope(
                query_states,
                key_states,
                position_embeddings,
            )
            key_states, value_states = _update_cache(
                past_key_values,
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
                position_embeddings=position_embeddings,
            )
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)

            query_features = self._feature_map(query_states)
            key_features = self._feature_map(key_states)
            values = value_states.float()
            total_length = int(key_features.shape[2])
            query_length = int(query_features.shape[2])
            keep_mask = _key_keep_mask(attention_mask, total_length)
            if keep_mask is not None:
                keep = keep_mask[:, None, :, None].to(dtype=key_features.dtype)
                key_features = key_features * keep
                values = values * keep

            past_length = max(total_length - query_length, 0)
            batch, heads, _, head_dim = key_features.shape
            running_key = key_features.new_zeros((batch, heads, head_dim))
            running_key_value = key_features.new_zeros(
                (batch, heads, head_dim, head_dim)
            )
            if past_length:
                past_keys = key_features[:, :, :past_length, :]
                past_values = values[:, :, :past_length, :]
                running_key = past_keys.sum(dim=2)
                running_key_value = torch.einsum(
                    "bhtd,bhte->bhde",
                    past_keys,
                    past_values,
                )

            outputs: list[Any] = []
            for query_index in range(query_length):
                key_index = past_length + query_index
                current_key = key_features[:, :, key_index, :]
                current_value = values[:, :, key_index, :]
                running_key = running_key + current_key
                running_key_value = running_key_value + torch.einsum(
                    "bhd,bhe->bhde",
                    current_key,
                    current_value,
                )
                current_query = query_features[:, :, query_index, :]
                denominator = torch.einsum(
                    "bhd,bhd->bh",
                    current_query,
                    running_key,
                ).unsqueeze(-1)
                numerator = torch.einsum(
                    "bhd,bhde->bhe",
                    current_query,
                    running_key_value,
                )
                outputs.append(numerator / denominator.clamp_min(self.eps))

            attention_output = torch.stack(outputs, dim=2).to(hidden_states.dtype)
            attention_output = attention_output.transpose(1, 2).reshape(
                *input_shape,
                self.num_heads * self.head_dim,
            )
            output = self.o_proj(attention_output)
            self._record_debug(hidden_states, output)
            return output, None

    class Qwen3VLCausalConvAttention(_Qwen3VLReplacementBase):
        """Depthwise causal CNN token mixer reusing Qwen3-VL value/output projections."""

        strategy_name = "attn_cnn"

        def __init__(
            self,
            original_attention: Any,
            *,
            kernel_size: int = 3,
            debug: bool = False,
        ) -> None:
            super().__init__(original_attention, debug=debug)
            self.kernel_size = int(kernel_size)
            if self.kernel_size < 1 or self.kernel_size % 2 == 0:
                raise ValueError(
                    "CNN replacement kernel_size must be a positive odd integer."
                )
            channels = int(original_attention.v_proj.out_features)
            self.depthwise_conv = nn.Conv1d(
                channels,
                channels,
                kernel_size=self.kernel_size,
                groups=channels,
                bias=False,
            )
            reference_weight = original_attention.v_proj.weight
            self.depthwise_conv.to(
                device=reference_weight.device,
                dtype=reference_weight.dtype,
            )
            with torch.no_grad():
                self.depthwise_conv.weight.fill_(1.0 / self.kernel_size)

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            del position_embeddings
            batch, query_length, _ = hidden_states.shape
            value_states = (
                self.v_proj(hidden_states)
                .view(
                    batch,
                    query_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )
            current_keep = _key_keep_mask(attention_mask, query_length)
            if current_keep is not None:
                value_states = value_states * current_keep[:, None, :, None].to(
                    dtype=value_states.dtype
                )
            placeholder_keys = torch.zeros_like(value_states)
            _, value_states = _update_cache(
                past_key_values,
                placeholder_keys,
                value_states,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
            )
            total_length = int(value_states.shape[2])
            flattened = (
                value_states.transpose(1, 2)
                .reshape(
                    batch,
                    total_length,
                    self.num_key_value_heads * self.head_dim,
                )
                .transpose(1, 2)
            )
            mixed = self.depthwise_conv(F.pad(flattened, (self.kernel_size - 1, 0)))
            mixed = (
                mixed[:, :, -query_length:]
                .transpose(1, 2)
                .reshape(
                    batch,
                    query_length,
                    self.num_key_value_heads,
                    self.head_dim,
                )
                .transpose(1, 2)
            )
            mixed = _repeat_kv(mixed, self.num_key_value_groups)
            attention_output = mixed.transpose(1, 2).reshape(
                batch,
                query_length,
                self.num_heads * self.head_dim,
            )
            output = self.o_proj(attention_output)
            self._record_debug(hidden_states, output)
            return output, None

    class Qwen3VLIdentityAttention(_Qwen3VLReplacementBase):
        """Return the attention input unchanged."""

        strategy_name = "attn_identity"

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            del attention_mask

            batch, query_length, _ = hidden_states.shape

            # generation 时维护 KV cache 长度，否则多轮生成可能出错
            placeholder = hidden_states.new_zeros(
                batch,
                self.num_key_value_heads,
                query_length,
                self.head_dim,
            )

            _update_cache(
                past_key_values,
                placeholder,
                placeholder,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
                position_embeddings=position_embeddings,
            )

            output = hidden_states
            self._record_debug(hidden_states, output)
            return output, None

    class Qwen3VLLoLCATsLinearWindowAttention(_Qwen3VLReplacementBase):
        """LoLCATs learned linear attention plus an exact local softmax window."""

        strategy_name = "attn_lolcats_linear_window"

        def __init__(
            self,
            original_attention: Any,
            *,
            window_size: int = 64,
            decode_window_size: int | None = None,
            feature_dim: int = 64,
            feature_map: str = "hedgehog_softmax_dim",
            tie_qk_feature_maps: bool = False,
            window_weight_init: float = 0.1,
            train_window_weight: bool = True,
            affine_attention_factors: bool = False,
            eps: float = 1.0e-12,
            debug: bool = False,
        ) -> None:
            super().__init__(original_attention, debug=debug)
            for name in ("q_proj", "k_proj"):
                if not hasattr(original_attention, name):
                    raise BackendIntegrationError(
                        f"LoLCATs attention requires Qwen3-VL {name}."
                    )
            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            self.q_norm = getattr(original_attention, "q_norm", nn.Identity())
            self.k_norm = getattr(original_attention, "k_norm", nn.Identity())
            self.scaling = float(
                getattr(original_attention, "scaling", self.head_dim**-0.5)
            )
            self.window_size = int(window_size)
            self.decode_window_size = int(
                self.window_size if decode_window_size is None else decode_window_size
            )
            self.feature_dim = int(feature_dim)
            self.feature_map = str(feature_map)
            self.tie_qk_feature_maps = bool(tie_qk_feature_maps)
            self.window_weight_init = float(window_weight_init)
            self.train_window_weight = bool(train_window_weight)
            self.affine_attention_factors = bool(affine_attention_factors)
            self.eps = float(eps)
            if self.window_size < 1 or self.decode_window_size < 1:
                raise ValueError("LoLCATs window sizes must be positive integers.")
            if self.feature_dim < 1:
                raise ValueError("LoLCATs feature_dim must be a positive integer.")
            if self.feature_map != "hedgehog_softmax_dim":
                raise ValueError("LoLCATs feature_map must be 'hedgehog_softmax_dim'.")
            if not 0.0 < self.window_weight_init < 1.0:
                raise ValueError("LoLCATs window_weight_init must be between 0 and 1.")
            if self.eps <= 0:
                raise ValueError("LoLCATs eps must be positive.")

            reference_weight = self.q_proj.weight
            parameter_kwargs = {
                "device": reference_weight.device,
                "dtype": reference_weight.dtype,
            }
            self.q_feature_weight = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    self.head_dim,
                    self.feature_dim,
                    **parameter_kwargs,
                )
            )
            nn.init.kaiming_uniform_(self.q_feature_weight, a=math.sqrt(5))
            if self.tie_qk_feature_maps:
                self.k_feature_weight = self.q_feature_weight
            else:
                self.k_feature_weight = nn.Parameter(
                    torch.empty(
                        self.num_heads,
                        self.head_dim,
                        self.feature_dim,
                        **parameter_kwargs,
                    )
                )
                nn.init.kaiming_uniform_(self.k_feature_weight, a=math.sqrt(5))
            initial_logit = math.log(
                self.window_weight_init / (1.0 - self.window_weight_init)
            )
            window_logits = torch.full(
                (1, self.num_heads, 1, 1),
                initial_logit,
                **parameter_kwargs,
            )
            if self.train_window_weight:
                self.window_factor_logits = nn.Parameter(window_logits)
            else:
                self.register_buffer("window_factor_logits", window_logits)

        @property
        def expanded_feature_dim(self) -> int:
            return 2 * self.feature_dim

        def _feature(self, value: Any, weight: Any) -> Any:
            projected = torch.einsum(
                "bhld,hdf->bhlf",
                value.float(),
                weight.float(),
            )
            return torch.cat(
                (
                    torch.softmax(projected, dim=-1),
                    torch.softmax(-projected, dim=-1),
                ),
                dim=-1,
            ).clamp_min(self.eps)

        def _factors(self) -> tuple[Any, Any]:
            window_factor = torch.sigmoid(self.window_factor_logits.float())
            linear_factor = (
                1.0 - window_factor
                if self.affine_attention_factors
                else torch.ones_like(window_factor)
            )
            return window_factor, linear_factor

        def _hybrid_from_full_history(
            self,
            query_states: Any,
            key_states: Any,
            value_states: Any,
            keep_mask: Any | None,
            *,
            past_length: int,
            window_size: int,
        ) -> tuple[Any, dict[str, Any]]:
            query_features = self._feature(query_states, self.q_feature_weight)
            key_features = self._feature(key_states, self.k_feature_weight)
            values = value_states.float()
            batch, heads, total_length, _ = key_states.shape
            query_length = int(query_states.shape[2])
            if keep_mask is None:
                keep = torch.ones(
                    batch,
                    1,
                    total_length,
                    1,
                    dtype=torch.bool,
                    device=key_states.device,
                )
            else:
                keep = keep_mask[:, None, :, None].to(dtype=torch.bool)
            masked_features = key_features * keep.to(dtype=key_features.dtype)
            masked_values = values * keep.to(dtype=values.dtype)
            running_key = key_features.new_zeros(
                (batch, heads, self.expanded_feature_dim)
            )
            running_key_value = key_features.new_zeros(
                (batch, heads, self.expanded_feature_dim, self.head_dim)
            )
            old_end = 0
            window_factor, linear_factor = self._factors()
            window_factor = window_factor[:, :, 0, :]
            linear_factor = linear_factor[:, :, 0, :]
            outputs: list[Any] = []
            local_denominators: list[Any] = []
            linear_denominators: list[Any] = []
            local_counts: list[int] = []
            for query_index in range(query_length):
                query_position = past_length + query_index
                next_old_end = max(0, query_position - window_size + 1)
                if next_old_end > old_end:
                    moved_features = masked_features[:, :, old_end:next_old_end, :]
                    moved_values = masked_values[:, :, old_end:next_old_end, :]
                    running_key = running_key + moved_features.sum(dim=2)
                    running_key_value = running_key_value + torch.einsum(
                        "bhtf,bhtd->bhfd",
                        moved_features,
                        moved_values,
                    )
                    old_end = next_old_end

                local_start = next_old_end
                local_end = query_position + 1
                local_keys = key_states[:, :, local_start:local_end, :]
                local_values = values[:, :, local_start:local_end, :]
                local_keep = keep[:, :, local_start:local_end, 0]
                current_query = query_states[:, :, query_index, :]
                local_scores = (
                    torch.einsum(
                        "bhd,bhkd->bhk",
                        current_query.float(),
                        local_keys.float(),
                    )
                    * self.scaling
                )
                masked_scores = local_scores.masked_fill(~local_keep, -torch.inf)
                score_max = masked_scores.amax(dim=-1, keepdim=True)
                score_max = torch.where(
                    torch.isfinite(score_max),
                    score_max,
                    torch.zeros_like(score_max),
                )
                local_weights = torch.where(
                    local_keep,
                    torch.exp(local_scores - score_max),
                    torch.zeros_like(local_scores),
                )
                local_weights = window_factor * local_weights
                local_denominator = local_weights.sum(dim=-1, keepdim=True)
                local_numerator = torch.einsum(
                    "bhk,bhkd->bhd",
                    local_weights,
                    local_values.float(),
                )

                current_feature = query_features[:, :, query_index, :]
                linear_denominator = linear_factor * torch.einsum(
                    "bhf,bhf->bh",
                    current_feature,
                    running_key,
                ).unsqueeze(-1)
                linear_numerator = linear_factor * torch.einsum(
                    "bhf,bhfd->bhd",
                    current_feature,
                    running_key_value,
                )
                denominator = (local_denominator + linear_denominator).clamp_min(
                    self.eps
                )
                outputs.append((local_numerator + linear_numerator) / denominator)
                local_denominators.append(local_denominator.detach())
                linear_denominators.append(linear_denominator.detach())
                local_counts.append(local_end - local_start)

            output = torch.stack(outputs, dim=2).to(dtype=query_states.dtype)
            stats = {
                "local_denominator": torch.stack(local_denominators, dim=2),
                "linear_denominator": torch.stack(linear_denominators, dim=2),
                "local_keys_min": min(local_counts),
                "local_keys_max": max(local_counts),
                "local_keys_last": local_counts[-1],
            }
            return output, stats

        def _make_runtime_state(
            self,
            key_states: Any,
            value_states: Any,
            keep_mask: Any | None,
            *,
            window_size: int,
        ) -> _LoLCATsState:
            total_length = int(key_states.shape[2])
            window_start = max(0, total_length - window_size)
            key_features = self._feature(key_states, self.k_feature_weight)
            if keep_mask is None:
                keep = torch.ones(
                    key_states.shape[0],
                    1,
                    total_length,
                    1,
                    dtype=torch.bool,
                    device=key_states.device,
                )
            else:
                keep = keep_mask[:, None, :, None].to(dtype=torch.bool)
            old_features = key_features[:, :, :window_start, :] * keep[
                :, :, :window_start, :
            ].to(dtype=key_features.dtype)
            old_values = value_states[:, :, :window_start, :].float() * keep[
                :, :, :window_start, :
            ].to(dtype=torch.float32)
            return _LoLCATsState(
                seen_tokens=total_length,
                window_size=window_size,
                k_state=old_features.sum(dim=2).detach(),
                kv_state=torch.einsum("bhtf,bhtd->bhfd", old_features, old_values).detach(),
                window_keys=key_states[:, :, window_start:, :].detach(),
                window_values=value_states[:, :, window_start:, :].detach(),
                window_features=key_features[:, :, window_start:, :].detach(),
                window_keep=keep[:, :, window_start:, :].detach(),
            )

        @staticmethod
        def _runtime_matches(state: Any, cache_length_before: int, window_size: int) -> bool:
            return bool(
                cache_length_before > 0
                and isinstance(state, _LoLCATsState)
                and state.seen_tokens == cache_length_before
                and state.window_size == window_size
            )

        def _decode_from_runtime(
            self,
            state: _LoLCATsState,
            query_states: Any,
            new_key_states: Any,
            new_value_states: Any,
            new_keep: Any,
            *,
            window_size: int,
        ) -> tuple[Any, dict[str, Any]]:
            key_features = self._feature(new_key_states, self.k_feature_weight)
            window_keys = torch.cat((state.window_keys, new_key_states), dim=2)
            window_values = torch.cat(
                (state.window_values, new_value_states), dim=2
            )
            window_features = torch.cat(
                (state.window_features, key_features), dim=2
            )
            window_keep = torch.cat((state.window_keep, new_keep), dim=2)
            if window_keys.shape[2] > window_size:
                moved_features = window_features[:, :, :1, :] * window_keep[
                    :, :, :1, :
                ].to(dtype=window_features.dtype)
                moved_values = window_values[:, :, :1, :].float() * window_keep[
                    :, :, :1, :
                ].to(dtype=torch.float32)
                state.k_state = (
                    state.k_state + moved_features[:, :, 0, :]
                )
                state.kv_state = state.kv_state + torch.einsum(
                    "bhf,bhd->bhfd",
                    moved_features[:, :, 0, :],
                    moved_values[:, :, 0, :],
                )
                window_keys = window_keys[:, :, 1:, :]
                window_values = window_values[:, :, 1:, :]
                window_features = window_features[:, :, 1:, :]
                window_keep = window_keep[:, :, 1:, :]

            window_factor, linear_factor = self._factors()
            window_factor = window_factor[:, :, 0, :]
            linear_factor = linear_factor[:, :, 0, :]
            current_query = query_states[:, :, 0, :]
            local_scores = (
                torch.einsum(
                    "bhd,bhkd->bhk",
                    current_query.float(),
                    window_keys.float(),
                )
                * self.scaling
            )
            local_keep = window_keep[:, :, :, 0]
            masked_scores = local_scores.masked_fill(~local_keep, -torch.inf)
            score_max = masked_scores.amax(dim=-1, keepdim=True)
            score_max = torch.where(
                torch.isfinite(score_max), score_max, torch.zeros_like(score_max)
            )
            local_weights = torch.where(
                local_keep,
                torch.exp(local_scores - score_max),
                torch.zeros_like(local_scores),
            )
            local_weights = window_factor * local_weights
            local_denominator = local_weights.sum(dim=-1, keepdim=True)
            local_numerator = torch.einsum(
                "bhk,bhkd->bhd",
                local_weights,
                window_values.float(),
            )
            current_feature = self._feature(query_states, self.q_feature_weight)[
                :, :, 0, :
            ]
            linear_denominator = linear_factor * torch.einsum(
                "bhf,bhf->bh",
                current_feature,
                state.k_state,
            ).unsqueeze(-1)
            linear_numerator = linear_factor * torch.einsum(
                "bhf,bhfd->bhd",
                current_feature,
                state.kv_state,
            )
            output = (
                (
                    (local_numerator + linear_numerator)
                    / (local_denominator + linear_denominator).clamp_min(self.eps)
                )
                .unsqueeze(2)
                .to(dtype=query_states.dtype)
            )
            state.window_keys = window_keys.detach()
            state.window_values = window_values.detach()
            state.window_features = window_features.detach()
            state.window_keep = window_keep.detach()
            state.seen_tokens += 1
            stats = {
                "local_denominator": local_denominator.detach().unsqueeze(2),
                "linear_denominator": linear_denominator.detach().unsqueeze(2),
                "local_keys_min": int(window_keys.shape[2]),
                "local_keys_max": int(window_keys.shape[2]),
                "local_keys_last": int(window_keys.shape[2]),
            }
            return output, stats

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            new_key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            new_value_states = (
                self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            )
            query_states, new_key_states = _apply_qwen3_vl_rope(
                query_states,
                new_key_states,
                position_embeddings,
            )
            cache_length_before = _cache_length(past_key_values, self.layer_idx)
            query_length = int(query_states.shape[2])
            full_key_states, full_value_states = _update_cache(
                past_key_values,
                new_key_states,
                new_value_states,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
                position_embeddings=position_embeddings,
            )
            full_key_states = _repeat_kv(full_key_states, self.num_key_value_groups)
            full_value_states = _repeat_kv(full_value_states, self.num_key_value_groups)
            new_key_states = _repeat_kv(new_key_states, self.num_key_value_groups)
            new_value_states = _repeat_kv(new_value_states, self.num_key_value_groups)
            total_length = int(full_key_states.shape[2])
            keep_mask = _key_keep_mask(attention_mask, total_length)
            window_size = (
                self.decode_window_size if cache_length_before else self.window_size
            )
            used_recurrent_decode = False
            state = None if self.training else _cache_state(past_key_values, self.layer_idx)
            if query_length == 1 and self._runtime_matches(
                state, cache_length_before, window_size
            ):
                if keep_mask is None:
                    new_keep = torch.ones(
                        new_key_states.shape[0],
                        1,
                        1,
                        1,
                        dtype=torch.bool,
                        device=new_key_states.device,
                    )
                else:
                    new_keep = keep_mask[:, None, -1:, None].to(dtype=torch.bool)
                attention_output, stats = self._decode_from_runtime(
                    state,
                    query_states,
                    new_key_states,
                    new_value_states,
                    new_keep,
                    window_size=window_size,
                )
                used_recurrent_decode = True
            else:
                attention_output, stats = self._hybrid_from_full_history(
                    query_states,
                    full_key_states,
                    full_value_states,
                    keep_mask,
                    past_length=total_length - query_length,
                    window_size=window_size,
                )
                if past_key_values is not None and not self.training:
                    state = self._make_runtime_state(
                        full_key_states,
                        full_value_states,
                        keep_mask,
                        window_size=window_size,
                    )
                    _store_cache_state(past_key_values, self.layer_idx, state)
            attention_output = attention_output.transpose(1, 2).reshape(
                *input_shape,
                self.num_heads * self.head_dim,
            )
            output = self.o_proj(attention_output)
            self._record_debug(hidden_states, output)
            if self.debug:
                window_factor, linear_factor = self._factors()
                self.last_debug.update(
                    {
                        "window_size": window_size,
                        "feature_dim": self.feature_dim,
                        "expanded_feature_dim": self.expanded_feature_dim,
                        "window_weight_min": float(window_factor.min().item()),
                        "window_weight_max": float(window_factor.max().item()),
                        "linear_weight_min": float(linear_factor.min().item()),
                        "linear_weight_max": float(linear_factor.max().item()),
                        "total_kv_length": total_length,
                        "used_recurrent_decode": used_recurrent_decode,
                        "local_keys_min": stats["local_keys_min"],
                        "local_keys_max": stats["local_keys_max"],
                        "local_keys_last": stats["local_keys_last"],
                        "local_denominator_min": float(
                            stats["local_denominator"].min().item()
                        ),
                        "linear_denominator_max": float(
                            stats["linear_denominator"].max().item()
                        ),
                    }
                )
            return output, None

    class Qwen3VLNativeHybridAttention(_Qwen3VLReplacementBase):
        """Native Hybrid Attention with gated long-term slots and exact local KV."""

        strategy_name = "attn_nha"

        def __init__(
            self,
            original_attention: Any,
            *,
            num_slots: int = 16,
            window_size: int = 16,
            gate_logit_normalizer: float = 8.0,
            eps: float = 1.0e-6,
            debug: bool = False,
        ) -> None:
            super().__init__(original_attention, debug=debug)
            for name in ("q_proj", "k_proj"):
                if not hasattr(original_attention, name):
                    raise BackendIntegrationError(f"NHA requires Qwen3-VL {name}.")
            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            self.q_norm = getattr(original_attention, "q_norm", nn.Identity())
            self.k_norm = getattr(original_attention, "k_norm", nn.Identity())
            self.scaling = float(
                getattr(original_attention, "scaling", self.head_dim**-0.5)
            )
            self.num_slots = int(num_slots)
            self.window_size = int(window_size)
            self.gate_logit_normalizer = float(gate_logit_normalizer)
            self.eps = float(eps)
            if self.num_slots < 1:
                raise ValueError("NHA num_slots must be a positive integer.")
            if self.window_size < 1:
                raise ValueError("NHA window_size must be a positive integer.")
            if self.head_dim % self.num_slots:
                raise ValueError(
                    "NHA num_slots must divide the Qwen3-VL attention head_dim "
                    "for pooled-K gate initialization."
                )
            if self.gate_logit_normalizer <= 0:
                raise ValueError("NHA gate_logit_normalizer must be positive.")
            if self.eps <= 0:
                raise ValueError("NHA eps must be positive.")

            hidden_size = int(self.k_proj.in_features)
            gate_features = self.num_key_value_heads * self.num_slots
            self.g_proj = nn.Linear(
                hidden_size,
                gate_features,
                bias=False,
                device=self.k_proj.weight.device,
                dtype=self.k_proj.weight.dtype,
            )
            pooling_factor = self.head_dim // self.num_slots
            with torch.no_grad():
                pooled = (
                    self.k_proj.weight.detach()
                    .view(
                        self.num_key_value_heads,
                        self.num_slots,
                        pooling_factor,
                        hidden_size,
                    )
                    .mean(dim=2)
                )
                self.g_proj.weight.copy_(pooled.reshape(gate_features, hidden_size))

        def _retentions(self, hidden_states: Any) -> Any:
            batch, sequence_length, _ = hidden_states.shape
            logits = (
                self.g_proj(hidden_states)
                .view(
                    batch,
                    sequence_length,
                    self.num_key_value_heads,
                    self.num_slots,
                )
                .transpose(1, 2)
            )
            logits = _repeat_kv(logits, self.num_key_value_groups)
            log_retention = F.logsigmoid(logits.float()) / self.gate_logit_normalizer
            return torch.exp(log_retention)

        def _empty_long_state(self, reference: Any) -> tuple[Any, Any]:
            shape = (
                reference.shape[0],
                self.num_heads,
                self.num_slots,
                self.head_dim,
            )
            return (
                torch.zeros(shape, device=reference.device, dtype=torch.float32),
                torch.zeros(shape, device=reference.device, dtype=torch.float32),
            )

        @staticmethod
        def _update_long_slots(
            long_keys: Any,
            long_values: Any,
            key: Any,
            value: Any,
            retention: Any,
            keep: Any,
        ) -> tuple[Any, Any]:
            keep = keep.to(dtype=torch.bool)
            retention = torch.where(
                keep[:, None, None],
                retention.float(),
                torch.ones_like(retention, dtype=torch.float32),
            )
            write = 1.0 - retention
            long_keys = retention.unsqueeze(-1) * long_keys + write.unsqueeze(
                -1
            ) * key.float().unsqueeze(-2)
            long_values = retention.unsqueeze(-1) * long_values + write.unsqueeze(
                -1
            ) * value.float().unsqueeze(-2)
            return long_keys, long_values

        def _attend_hybrid_memory(
            self,
            query: Any,
            rope_query: Any,
            long_keys: Any,
            long_values: Any,
            local_rope_keys: Any,
            local_values: Any,
            local_keep: Any,
        ) -> tuple[Any, Any, Any]:
            long_logits = (
                torch.einsum(
                    "bhd,bhmd->bhm",
                    query.float(),
                    long_keys,
                )
                * self.scaling
            )
            local_logits = (
                torch.einsum(
                    "bhd,bhwd->bhw",
                    rope_query.float(),
                    local_rope_keys.float(),
                )
                * self.scaling
            )
            local_logits = local_logits.masked_fill(~local_keep, -torch.inf)
            logits = torch.cat((long_logits, local_logits), dim=-1)
            probabilities = torch.softmax(logits, dim=-1)
            long_probability = probabilities[:, :, : self.num_slots]
            local_probability = probabilities[:, :, self.num_slots :]
            output = torch.einsum(
                "bhm,bhmd->bhd",
                long_probability,
                long_values,
            ) + torch.einsum(
                "bhw,bhwd->bhd",
                local_probability,
                local_values.float(),
            )
            return output, long_probability, local_probability

        def _prefill(
            self,
            query_states: Any,
            rope_query_states: Any,
            key_states: Any,
            rope_key_states: Any,
            value_states: Any,
            retentions: Any,
            keep_mask: Any,
        ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
            sequence_length = int(query_states.shape[2])
            long_keys, long_values = self._empty_long_state(query_states)
            outputs: list[Any] = []
            long_masses: list[Any] = []
            local_masses: list[Any] = []
            local_counts: list[int] = []
            old_end = 0
            for position in range(sequence_length):
                next_old_end = max(0, position - self.window_size + 1)
                for moved_index in range(old_end, next_old_end):
                    long_keys, long_values = self._update_long_slots(
                        long_keys,
                        long_values,
                        key_states[:, :, moved_index, :],
                        value_states[:, :, moved_index, :],
                        retentions[:, :, moved_index, :],
                        keep_mask[:, moved_index],
                    )
                old_end = next_old_end
                local_end = position + 1
                local_keep = keep_mask[:, None, old_end:local_end].to(torch.bool)
                output, long_probability, local_probability = (
                    self._attend_hybrid_memory(
                        query_states[:, :, position, :],
                        rope_query_states[:, :, position, :],
                        long_keys,
                        long_values,
                        rope_key_states[:, :, old_end:local_end, :],
                        value_states[:, :, old_end:local_end, :],
                        local_keep,
                    )
                )
                outputs.append(output)
                long_masses.append(long_probability.sum(dim=-1))
                local_masses.append(local_probability.sum(dim=-1))
                local_counts.append(local_end - old_end)

            window_start = max(0, sequence_length - self.window_size)
            state = {
                "long_keys": long_keys,
                "long_values": long_values,
                "window_keys": key_states[:, :, window_start:, :],
                "window_rope_keys": rope_key_states[:, :, window_start:, :],
                "window_values": value_states[:, :, window_start:, :],
                "window_retentions": retentions[:, :, window_start:, :],
                "window_keep": keep_mask[:, None, window_start:, None].to(torch.bool),
            }
            evaluated_pairs = self.num_slots * sequence_length + sum(local_counts)
            full_pairs = sequence_length * (sequence_length + 1) // 2
            stats = {
                "long_probability_mass": torch.stack(long_masses, dim=2),
                "local_probability_mass": torch.stack(local_masses, dim=2),
                "local_keys_min": min(local_counts),
                "local_keys_max": max(local_counts),
                "local_keys_last": local_counts[-1],
                "nha_score_pairs": evaluated_pairs,
                "full_attention_score_pairs": full_pairs,
            }
            output = torch.stack(outputs, dim=2).to(dtype=query_states.dtype)
            return output, stats, state

        @staticmethod
        def _make_runtime_state(state: dict[str, Any], seen_tokens: int) -> _NHAState:
            return _NHAState(seen_tokens=int(seen_tokens), **{
                name: value.detach() for name, value in state.items()
            })

        @staticmethod
        def _runtime_matches(state: Any, cache_length_before: int) -> bool:
            return bool(
                cache_length_before > 0
                and isinstance(state, _NHAState)
                and state.seen_tokens == cache_length_before
            )

        def _decode(
            self,
            state: _NHAState,
            query_states: Any,
            rope_query_states: Any,
            key_states: Any,
            rope_key_states: Any,
            value_states: Any,
            retentions: Any,
            keep_mask: Any,
        ) -> tuple[Any, dict[str, Any]]:
            outputs: list[Any] = []
            long_masses: list[Any] = []
            local_masses: list[Any] = []
            local_counts: list[int] = []
            for index in range(int(query_states.shape[2])):
                window_keys = torch.cat(
                    (state.window_keys, key_states[:, :, index : index + 1]),
                    dim=2,
                )
                window_rope_keys = torch.cat(
                    (
                        state.window_rope_keys,
                        rope_key_states[:, :, index : index + 1],
                    ),
                    dim=2,
                )
                window_values = torch.cat(
                    (
                        state.window_values,
                        value_states[:, :, index : index + 1],
                    ),
                    dim=2,
                )
                window_retentions = torch.cat(
                    (
                        state.window_retentions,
                        retentions[:, :, index : index + 1],
                    ),
                    dim=2,
                )
                window_keep = torch.cat(
                    (
                        state.window_keep,
                        keep_mask[:, None, index : index + 1, None].to(torch.bool),
                    ),
                    dim=2,
                )
                if window_keys.shape[2] > self.window_size:
                    state.long_keys, state.long_values = (
                        self._update_long_slots(
                            state.long_keys,
                            state.long_values,
                            window_keys[:, :, 0, :],
                            window_values[:, :, 0, :],
                            window_retentions[:, :, 0, :],
                            window_keep[:, 0, 0, 0],
                        )
                    )
                    window_keys = window_keys[:, :, 1:, :]
                    window_rope_keys = window_rope_keys[:, :, 1:, :]
                    window_values = window_values[:, :, 1:, :]
                    window_retentions = window_retentions[:, :, 1:, :]
                    window_keep = window_keep[:, :, 1:, :]

                output, long_probability, local_probability = (
                    self._attend_hybrid_memory(
                        query_states[:, :, index, :],
                        rope_query_states[:, :, index, :],
                        state.long_keys,
                        state.long_values,
                        window_rope_keys,
                        window_values,
                        window_keep[:, :, :, 0],
                    )
                )
                outputs.append(output)
                long_masses.append(long_probability.sum(dim=-1))
                local_masses.append(local_probability.sum(dim=-1))
                local_counts.append(int(window_keys.shape[2]))
                state.window_keys = window_keys.detach()
                state.window_rope_keys = window_rope_keys.detach()
                state.window_values = window_values.detach()
                state.window_retentions = window_retentions.detach()
                state.window_keep = window_keep.detach()
                state.seen_tokens += 1

            query_length = int(query_states.shape[2])
            current_start = state.seen_tokens - query_length
            full_pairs = sum(current_start + index + 1 for index in range(query_length))
            stats = {
                "long_probability_mass": torch.stack(long_masses, dim=2),
                "local_probability_mass": torch.stack(local_masses, dim=2),
                "local_keys_min": min(local_counts),
                "local_keys_max": max(local_counts),
                "local_keys_last": local_counts[-1],
                "nha_score_pairs": sum(
                    self.num_slots + count for count in local_counts
                ),
                "full_attention_score_pairs": full_pairs,
            }
            return torch.stack(outputs, dim=2).to(query_states.dtype), stats

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            rope_query_states, rope_key_states = _apply_qwen3_vl_rope(
                query_states,
                key_states,
                position_embeddings,
            )
            cache_key_states = rope_key_states
            cache_value_states = value_states
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            rope_key_states = _repeat_kv(rope_key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)
            retentions = self._retentions(hidden_states)
            cache_length_before = _cache_length(past_key_values, self.layer_idx)
            query_length = int(query_states.shape[2])
            total_length = cache_length_before + query_length
            keep = _key_keep_mask(attention_mask, total_length)
            if keep is None:
                new_keep = torch.ones(
                    hidden_states.shape[0],
                    query_length,
                    dtype=torch.bool,
                    device=hidden_states.device,
                )
            else:
                new_keep = keep[:, -query_length:].to(torch.bool)

            _update_cache(
                past_key_values,
                cache_key_states,
                cache_value_states,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
                position_embeddings=position_embeddings,
            )
            used_recurrent_decode = False
            if cache_length_before == 0:
                attention_output, stats, state = self._prefill(
                    query_states,
                    rope_query_states,
                    key_states,
                    rope_key_states,
                    value_states,
                    retentions,
                    new_keep,
                )
                if past_key_values is not None and not self.training:
                    _store_cache_state(past_key_values, self.layer_idx,
                                       self._make_runtime_state(state, query_length))
            else:
                state = None if self.training else _cache_state(past_key_values, self.layer_idx)
                if not self._runtime_matches(state, cache_length_before):
                    raise BackendIntegrationError(
                        "NHA recurrent state does not match the Qwen3-VL cache. "
                        "Start generation with a fresh cache."
                    )
                attention_output, stats = self._decode(
                    state,
                    query_states,
                    rope_query_states,
                    key_states,
                    rope_key_states,
                    value_states,
                    retentions,
                    new_keep,
                )
                used_recurrent_decode = True

            attention_output = attention_output.transpose(1, 2).reshape(
                *input_shape,
                self.num_heads * self.head_dim,
            )
            output = self.o_proj(attention_output)
            self._record_debug(hidden_states, output)
            if self.debug:
                nha_pairs = int(stats["nha_score_pairs"])
                full_pairs = int(stats["full_attention_score_pairs"])
                self.last_debug.update(
                    {
                        "num_slots": self.num_slots,
                        "window_size": self.window_size,
                        "gate_logit_normalizer": self.gate_logit_normalizer,
                        "total_kv_length": total_length,
                        "used_recurrent_decode": used_recurrent_decode,
                        "local_keys_min": stats["local_keys_min"],
                        "local_keys_max": stats["local_keys_max"],
                        "local_keys_last": stats["local_keys_last"],
                        "long_probability_mass_min": float(
                            stats["long_probability_mass"].min().item()
                        ),
                        "long_probability_mass_max": float(
                            stats["long_probability_mass"].max().item()
                        ),
                        "nha_score_pairs": nha_pairs,
                        "full_attention_score_pairs": full_pairs,
                        "token_mixing_flops_ratio": nha_pairs / max(full_pairs, 1),
                    }
                )
            return output, None

    class Qwen3VLSparseWindowAttention(_Qwen3VLReplacementBase):
        """Configurable sparse causal/non-causal attention over selected KV sites."""

        strategy_name = "attn_sparse_window"

        def __init__(
            self,
            original_attention: Any,
            *,
            causal: bool = True,
            include_self: bool = True,
            local_left: int = 127,
            local_right: int = 0,
            prefix_size: int = 0,
            prompt_suffix_size: int = 0,
            fixed_positions: list[int] | tuple[int, ...] = (),
            stride: int | None = None,
            debug: bool = False,
        ) -> None:
            super().__init__(original_attention, debug=debug)
            for name in ("q_proj", "k_proj"):
                if not hasattr(original_attention, name):
                    raise BackendIntegrationError(
                        f"Sparse-window attention requires Qwen3-VL {name}."
                    )
            self.q_proj = original_attention.q_proj
            self.k_proj = original_attention.k_proj
            self.q_norm = getattr(original_attention, "q_norm", nn.Identity())
            self.k_norm = getattr(original_attention, "k_norm", nn.Identity())
            self.scaling = float(
                getattr(original_attention, "scaling", self.head_dim**-0.5)
            )
            self.causal = bool(causal)
            self.include_self = bool(include_self)
            self.local_left = int(local_left)
            self.local_right = int(local_right)
            self.prefix_size = int(prefix_size)
            self.prompt_suffix_size = int(prompt_suffix_size)
            self.fixed_positions = tuple(int(item) for item in fixed_positions)
            self.stride = None if stride is None else int(stride)
            for name in (
                "local_left",
                "local_right",
                "prefix_size",
                "prompt_suffix_size",
            ):
                if getattr(self, name) < 0:
                    raise ValueError(f"sparse-window {name} must be non-negative.")
            if self.causal and self.local_right:
                raise ValueError(
                    "causal sparse-window attention requires local_right=0."
                )
            if any(item < 0 for item in self.fixed_positions):
                raise ValueError("sparse-window fixed_positions must be non-negative.")
            if self.stride is not None and self.stride < 1:
                raise ValueError(
                    "sparse-window stride must be null or a positive integer."
                )
            if not (
                self.include_self
                or self.local_left
                or self.local_right
                or self.prefix_size
                or self.prompt_suffix_size
                or self.fixed_positions
                or self.stride is not None
            ):
                raise ValueError("sparse-window pattern must select at least one key.")
            self.prompt_length: int | None = None

        def _selected_key_indices(
            self,
            *,
            query_position: int,
            total_length: int,
            prompt_length: int,
            device: Any,
        ) -> Any:
            selected: set[int] = set()
            if self.include_self:
                selected.add(query_position)
            if self.local_left or self.local_right:
                selected.update(
                    range(
                        max(0, query_position - self.local_left),
                        min(total_length, query_position + self.local_right + 1),
                    )
                )
            selected.update(range(min(self.prefix_size, total_length)))
            if self.prompt_suffix_size:
                suffix_start = max(0, prompt_length - self.prompt_suffix_size)
                selected.update(range(suffix_start, min(prompt_length, total_length)))
            selected.update(
                position for position in self.fixed_positions if position < total_length
            )
            if self.stride is not None:
                selected.update(range(0, total_length, self.stride))
            if self.causal:
                selected = {
                    position for position in selected if position <= query_position
                }
            if not selected:
                raise BackendIntegrationError(
                    f"Sparse-window pattern selected no keys for query {query_position}."
                )
            return torch.tensor(sorted(selected), dtype=torch.long, device=device)

        @staticmethod
        def _apply_selected_mask(
            scores: Any,
            attention_mask: Any,
            selected_indices: Any,
            *,
            query_index: int,
            query_length: int,
            total_length: int,
        ) -> Any:
            if attention_mask is None or not hasattr(attention_mask, "ndim"):
                return scores
            if attention_mask.ndim == 2:
                full_key_mask = attention_mask[:, -total_length:]
                selected_mask = full_key_mask.index_select(-1, selected_indices)
                return scores.masked_fill(
                    ~selected_mask[:, None, :].to(dtype=torch.bool),
                    torch.finfo(scores.dtype).min,
                )
            if attention_mask.ndim == 4:
                if attention_mask.shape[-2] == 1:
                    mask_query_index = 0
                else:
                    mask_query_index = (
                        attention_mask.shape[-2] - query_length + query_index
                    )
                selected_mask = attention_mask[:, :, mask_query_index, :].index_select(
                    -1, selected_indices
                )
                if selected_mask.dtype == torch.bool:
                    return scores.masked_fill(
                        ~selected_mask,
                        torch.finfo(scores.dtype).min,
                    )
                return scores + selected_mask
            return scores

        def forward(
            self,
            hidden_states: Any,
            position_embeddings: Any,
            attention_mask: Any | None,
            past_key_values: Any | None = None,
            **kwargs: Any,
        ) -> tuple[Any, None]:
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)
            query_states = self.q_norm(
                self.q_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            key_states = self.k_norm(
                self.k_proj(hidden_states).view(hidden_shape)
            ).transpose(1, 2)
            value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            query_states, key_states = _apply_qwen3_vl_rope(
                query_states,
                key_states,
                position_embeddings,
            )
            cache_length_before = _cache_length(past_key_values, self.layer_idx)
            query_length = int(query_states.shape[2])
            if cache_length_before == 0:
                self.prompt_length = query_length
            elif self.prompt_length is None:
                self.prompt_length = cache_length_before
            key_states, value_states = _update_cache(
                past_key_values,
                key_states,
                value_states,
                layer_idx=self.layer_idx,
                kwargs=kwargs,
                position_embeddings=position_embeddings,
            )
            key_states = _repeat_kv(key_states, self.num_key_value_groups)
            value_states = _repeat_kv(value_states, self.num_key_value_groups)
            total_length = int(key_states.shape[2])
            past_length = total_length - query_length
            prompt_length = int(self.prompt_length or query_length)
            outputs: list[Any] = []
            selected_counts: list[int] = []
            for query_index in range(query_length):
                query_position = past_length + query_index
                selected_indices = self._selected_key_indices(
                    query_position=query_position,
                    total_length=total_length,
                    prompt_length=prompt_length,
                    device=key_states.device,
                )
                selected_keys = key_states.index_select(2, selected_indices)
                selected_values = value_states.index_select(2, selected_indices)
                current_query = query_states[:, :, query_index, :]
                scores = (
                    torch.einsum(
                        "bhd,bhkd->bhk",
                        current_query,
                        selected_keys,
                    )
                    * self.scaling
                )
                scores = self._apply_selected_mask(
                    scores,
                    attention_mask,
                    selected_indices,
                    query_index=query_index,
                    query_length=query_length,
                    total_length=total_length,
                )
                probabilities = torch.softmax(scores.float(), dim=-1).to(
                    dtype=selected_values.dtype
                )
                outputs.append(
                    torch.einsum(
                        "bhk,bhkd->bhd",
                        probabilities,
                        selected_values,
                    )
                )
                selected_counts.append(int(selected_indices.numel()))
            attention_output = (
                torch.stack(outputs, dim=2)
                .transpose(1, 2)
                .reshape(
                    *input_shape,
                    self.num_heads * self.head_dim,
                )
            )
            output = self.o_proj(attention_output)
            self._record_debug(hidden_states, output)
            if self.debug:
                self.last_debug.update(
                    {
                        "causal": self.causal,
                        "prompt_length": prompt_length,
                        "total_kv_length": total_length,
                        "selected_keys_min": min(selected_counts),
                        "selected_keys_max": max(selected_counts),
                        "selected_keys_last": selected_counts[-1],
                    }
                )
            return output, None

else:

    class Qwen3VLLinearAttention:  # pragma: no cover - import-only fallback.
        strategy_name = "attn_linear_attention"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLCausalConvAttention:  # pragma: no cover - import-only fallback.
        strategy_name = "attn_cnn"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLIdentityAttention:  # pragma: no cover
        strategy_name = "attn_identity"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLSparseWindowAttention:  # pragma: no cover
        strategy_name = "attn_sparse_window"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLLoLCATsLinearWindowAttention:  # pragma: no cover
        strategy_name = "attn_lolcats_linear_window"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLNativeHybridAttention:  # pragma: no cover
        strategy_name = "attn_nha"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()


__all__ = [
    "Qwen3VLCausalConvAttention",
    "Qwen3VLIdentityAttention",
    "Qwen3VLLinearAttention",
    "Qwen3VLLoLCATsLinearWindowAttention",
    "Qwen3VLNativeHybridAttention",
    "Qwen3VLSparseWindowAttention",
]
