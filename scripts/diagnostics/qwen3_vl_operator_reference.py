"""Independent reference calculations for Qwen3-VL replacement traces."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _cpu_snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device="cpu", copy=True)
    if isinstance(value, tuple):
        return tuple(_cpu_snapshot(item) for item in value)
    if isinstance(value, list):
        return [_cpu_snapshot(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _cpu_snapshot(item) for key, item in value.items()}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return {"python_type": type(value).__qualname__}


def _argument(
    positional: tuple[Any, ...],
    keyword: dict[str, Any],
    name: str,
    index: int,
    default: Any = None,
) -> Any:
    if name in keyword:
        return keyword[name]
    return positional[index] if len(positional) > index else default


def _cache_length(cache: Any, layer_idx: int) -> int:
    if cache is None:
        return 0
    getter = getattr(cache, "get_seq_length", None)
    if not callable(getter):
        return 0
    try:
        return int(getter(layer_idx))
    except TypeError:
        return int(getter())


def _linear(value: Any, module: Any) -> Any:
    return F.linear(value, module.weight, module.bias)


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


def _rotate_half(value: Any) -> Any:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _apply_rope(query: Any, key: Any, position_embeddings: Any) -> tuple[Any, Any]:
    if position_embeddings is None:
        return query, key
    cos, sin = position_embeddings
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (
        query * cos + _rotate_half(query) * sin,
        key * cos + _rotate_half(key) * sin,
    )


def _key_keep_mask(attention_mask: Any, total_length: int) -> Any | None:
    if attention_mask is None or not hasattr(attention_mask, "ndim"):
        return None
    if attention_mask.ndim == 2:
        return attention_mask[:, -total_length:].to(dtype=torch.bool)
    if attention_mask.ndim == 4:
        mask = attention_mask[:, 0, -1, -total_length:]
        return mask if mask.dtype == torch.bool else mask > -1.0e4
    return None


def _reference_linear_attention(
    module: Any,
    hidden_states: Any,
    position_embeddings: Any,
    attention_mask: Any,
) -> Any:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(
        _linear(hidden_states, module.q_proj).view(hidden_shape)
    ).transpose(1, 2)
    key_states = module.k_norm(
        _linear(hidden_states, module.k_proj).view(hidden_shape)
    ).transpose(1, 2)
    value_states = (
        _linear(hidden_states, module.v_proj).view(hidden_shape).transpose(1, 2)
    )
    query_states, key_states = _apply_rope(
        query_states,
        key_states,
        position_embeddings,
    )
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    value_states = _repeat_kv(value_states, module.num_key_value_groups)
    query_features = F.elu(query_states.float()) + 1.0
    key_features = F.elu(key_states.float()) + 1.0
    values = value_states.float()
    total_length = int(key_features.shape[2])
    query_length = int(query_features.shape[2])
    keep_mask = _key_keep_mask(attention_mask, total_length)
    if keep_mask is not None:
        keep = keep_mask[:, None, :, None].to(dtype=key_features.dtype)
        key_features = key_features * keep
        values = values * keep

    batch, heads, _, head_dim = key_features.shape
    running_key = key_features.new_zeros((batch, heads, head_dim))
    running_key_value = key_features.new_zeros((batch, heads, head_dim, head_dim))
    outputs: list[Any] = []
    for query_index in range(query_length):
        current_key = key_features[:, :, query_index, :]
        current_value = values[:, :, query_index, :]
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
        outputs.append(numerator / denominator.clamp_min(module.eps))
    attention_output = torch.stack(outputs, dim=2).to(hidden_states.dtype)
    attention_output = attention_output.transpose(1, 2).reshape(
        *input_shape,
        module.num_heads * module.head_dim,
    )
    return _linear(attention_output, module.o_proj)


def _reference_cnn(module: Any, hidden_states: Any, attention_mask: Any) -> Any:
    batch, query_length, _ = hidden_states.shape
    value_states = (
        _linear(hidden_states, module.v_proj)
        .view(
            batch,
            query_length,
            module.num_key_value_heads,
            module.head_dim,
        )
        .transpose(1, 2)
    )
    current_keep = _key_keep_mask(attention_mask, query_length)
    if current_keep is not None:
        value_states = value_states * current_keep[:, None, :, None].to(
            dtype=value_states.dtype
        )
    flattened = (
        value_states.transpose(1, 2)
        .reshape(
            batch,
            query_length,
            module.num_key_value_heads * module.head_dim,
        )
        .transpose(1, 2)
    )
    mixed = F.conv1d(
        F.pad(flattened, (module.kernel_size - 1, 0)),
        module.depthwise_conv.weight,
        module.depthwise_conv.bias,
        stride=module.depthwise_conv.stride,
        padding=module.depthwise_conv.padding,
        dilation=module.depthwise_conv.dilation,
        groups=module.depthwise_conv.groups,
    )
    mixed = (
        mixed[:, :, -query_length:]
        .transpose(1, 2)
        .reshape(
            batch,
            query_length,
            module.num_key_value_heads,
            module.head_dim,
        )
        .transpose(1, 2)
    )
    mixed = _repeat_kv(mixed, module.num_key_value_groups)
    attention_output = mixed.transpose(1, 2).reshape(
        batch,
        query_length,
        module.num_heads * module.head_dim,
    )
    return _linear(attention_output, module.o_proj)


def _sparse_indices(
    module: Any,
    *,
    query_position: int,
    total_length: int,
    prompt_length: int,
    device: Any,
) -> Any:
    selected: set[int] = set()
    if module.include_self:
        selected.add(query_position)
    if module.local_left or module.local_right:
        selected.update(
            range(
                max(0, query_position - module.local_left),
                min(total_length, query_position + module.local_right + 1),
            )
        )
    selected.update(range(min(module.prefix_size, total_length)))
    if module.prompt_suffix_size:
        selected.update(
            range(
                max(0, prompt_length - module.prompt_suffix_size),
                min(prompt_length, total_length),
            )
        )
    selected.update(
        position for position in module.fixed_positions if position < total_length
    )
    if module.stride is not None:
        selected.update(range(0, total_length, module.stride))
    if module.causal:
        selected = {position for position in selected if position <= query_position}
    if not selected:
        raise RuntimeError(
            f"Reference sparse pattern selected no keys for query {query_position}."
        )
    return torch.tensor(sorted(selected), dtype=torch.long, device=device)


def _selected_mask(
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
        selected_mask = attention_mask[:, -total_length:].index_select(
            -1,
            selected_indices,
        )
        return scores.masked_fill(
            ~selected_mask[:, None, :].to(dtype=torch.bool),
            torch.finfo(scores.dtype).min,
        )
    if attention_mask.ndim == 4:
        mask_query_index = (
            0
            if attention_mask.shape[-2] == 1
            else attention_mask.shape[-2] - query_length + query_index
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


def _reference_sparse_window(
    module: Any,
    hidden_states: Any,
    position_embeddings: Any,
    attention_mask: Any,
) -> Any:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(
        _linear(hidden_states, module.q_proj).view(hidden_shape)
    ).transpose(1, 2)
    key_states = module.k_norm(
        _linear(hidden_states, module.k_proj).view(hidden_shape)
    ).transpose(1, 2)
    value_states = (
        _linear(hidden_states, module.v_proj).view(hidden_shape).transpose(1, 2)
    )
    query_states, key_states = _apply_rope(
        query_states,
        key_states,
        position_embeddings,
    )
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    value_states = _repeat_kv(value_states, module.num_key_value_groups)
    query_length = int(query_states.shape[2])
    total_length = int(key_states.shape[2])
    prompt_length = query_length
    outputs: list[Any] = []
    for query_index in range(query_length):
        selected_indices = _sparse_indices(
            module,
            query_position=query_index,
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
            * module.scaling
        )
        scores = _selected_mask(
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
    attention_output = (
        torch.stack(outputs, dim=2)
        .transpose(1, 2)
        .reshape(
            *input_shape,
            module.num_heads * module.head_dim,
        )
    )
    return _linear(attention_output, module.o_proj)


def _lolcats_feature(value: Any, weight: Any, eps: float) -> Any:
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
    ).clamp_min(eps)


def _reference_lolcats_linear_window(
    module: Any,
    hidden_states: Any,
    position_embeddings: Any,
    attention_mask: Any,
) -> tuple[Any, dict[str, Any]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(
        _linear(hidden_states, module.q_proj).view(hidden_shape)
    ).transpose(1, 2)
    key_states = module.k_norm(
        _linear(hidden_states, module.k_proj).view(hidden_shape)
    ).transpose(1, 2)
    value_states = (
        _linear(hidden_states, module.v_proj).view(hidden_shape).transpose(1, 2)
    )
    query_states, key_states = _apply_rope(
        query_states,
        key_states,
        position_embeddings,
    )
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    value_states = _repeat_kv(value_states, module.num_key_value_groups)
    query_features = _lolcats_feature(
        query_states,
        module.q_feature_weight,
        module.eps,
    )
    key_features = _lolcats_feature(
        key_states,
        module.k_feature_weight,
        module.eps,
    )
    sequence_length = int(query_states.shape[2])
    query_positions = torch.arange(sequence_length, device=query_states.device)[:, None]
    key_positions = torch.arange(sequence_length, device=query_states.device)[None, :]
    causal = key_positions <= query_positions
    local_mask = causal & (key_positions >= query_positions - module.window_size + 1)
    linear_mask = causal & ~local_mask
    keep_mask = _key_keep_mask(attention_mask, sequence_length)
    if keep_mask is not None:
        key_keep = keep_mask[:, None, None, :].to(dtype=torch.bool)
        local_mask = local_mask[None, None, :, :] & key_keep
        linear_mask = linear_mask[None, None, :, :] & key_keep
    else:
        local_mask = local_mask[None, None, :, :]
        linear_mask = linear_mask[None, None, :, :]

    local_scores = (
        torch.einsum(
            "bhmd,bhnd->bhmn",
            query_states.float(),
            key_states.float(),
        )
        * module.scaling
    )
    masked_local_scores = local_scores.masked_fill(~local_mask, -torch.inf)
    local_max = masked_local_scores.amax(dim=-1, keepdim=True)
    local_max = torch.where(
        torch.isfinite(local_max),
        local_max,
        torch.zeros_like(local_max),
    )
    window_factor = torch.sigmoid(module.window_factor_logits.float())
    linear_factor = (
        1.0 - window_factor
        if module.affine_attention_factors
        else torch.ones_like(window_factor)
    )
    local_weights = window_factor * torch.where(
        local_mask,
        torch.exp(local_scores - local_max),
        torch.zeros_like(local_scores),
    )
    linear_scores = torch.einsum(
        "bhmf,bhnf->bhmn",
        query_features,
        key_features,
    )
    linear_weights = linear_factor * linear_scores.masked_fill(~linear_mask, 0)
    local_numerator = torch.einsum(
        "bhmn,bhnd->bhmd",
        local_weights,
        value_states.float(),
    )
    linear_numerator = torch.einsum(
        "bhmn,bhnd->bhmd",
        linear_weights,
        value_states.float(),
    )
    local_denominator = local_weights.sum(dim=-1, keepdim=True)
    linear_denominator = linear_weights.sum(dim=-1, keepdim=True)
    denominator = (local_denominator + linear_denominator).clamp_min(module.eps)
    attention_output = ((local_numerator + linear_numerator) / denominator).to(
        dtype=hidden_states.dtype
    )
    attention_output = attention_output.transpose(1, 2).reshape(
        *input_shape,
        module.num_heads * module.head_dim,
    )
    output = _linear(attention_output, module.o_proj)
    components = {
        "query_features": query_features,
        "key_features": key_features,
        "local_mask": local_mask,
        "linear_mask": linear_mask,
        "local_weights": local_weights,
        "linear_weights": linear_weights,
        "local_numerator": local_numerator,
        "linear_numerator": linear_numerator,
        "local_denominator": local_denominator,
        "linear_denominator": linear_denominator,
        "attention_output_before_o_proj": attention_output,
    }
    return output, components


def _reference_nha(
    module: Any,
    hidden_states: Any,
    position_embeddings: Any,
    attention_mask: Any,
) -> tuple[Any, dict[str, Any]]:
    """Independent explicit Native Hybrid Attention prefill calculation."""
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, module.head_dim)
    query_states = module.q_norm(
        _linear(hidden_states, module.q_proj).view(hidden_shape)
    ).transpose(1, 2)
    key_states = module.k_norm(
        _linear(hidden_states, module.k_proj).view(hidden_shape)
    ).transpose(1, 2)
    value_states = (
        _linear(hidden_states, module.v_proj).view(hidden_shape).transpose(1, 2)
    )
    rope_query_states, rope_key_states = _apply_rope(
        query_states,
        key_states,
        position_embeddings,
    )
    key_states = _repeat_kv(key_states, module.num_key_value_groups)
    rope_key_states = _repeat_kv(
        rope_key_states,
        module.num_key_value_groups,
    )
    value_states = _repeat_kv(value_states, module.num_key_value_groups)

    batch, _, sequence_length, _ = query_states.shape
    gate_logits = (
        _linear(hidden_states, module.g_proj)
        .view(
            batch,
            sequence_length,
            module.num_key_value_heads,
            module.num_slots,
        )
        .transpose(1, 2)
    )
    gate_logits = _repeat_kv(gate_logits, module.num_key_value_groups)
    retentions = torch.exp(
        F.logsigmoid(gate_logits.float()) / module.gate_logit_normalizer
    )
    keep_mask = _key_keep_mask(attention_mask, sequence_length)
    if keep_mask is None:
        keep_mask = torch.ones(
            batch,
            sequence_length,
            dtype=torch.bool,
            device=hidden_states.device,
        )

    long_shape = (
        batch,
        module.num_heads,
        module.num_slots,
        module.head_dim,
    )
    long_keys = torch.zeros(
        long_shape, dtype=torch.float32, device=hidden_states.device
    )
    long_values = torch.zeros_like(long_keys)
    outputs: list[Any] = []
    logits_by_query: list[Any] = []
    probabilities_by_query: list[Any] = []
    valid_by_query: list[Any] = []
    long_keys_by_query: list[Any] = []
    long_values_by_query: list[Any] = []
    old_end = 0
    total_columns = module.num_slots + module.window_size
    for position in range(sequence_length):
        next_old_end = max(0, position - module.window_size + 1)
        for moved_index in range(old_end, next_old_end):
            keep = keep_mask[:, moved_index].to(torch.bool)
            retention = torch.where(
                keep[:, None, None],
                retentions[:, :, moved_index, :].float(),
                torch.ones_like(retentions[:, :, moved_index, :], dtype=torch.float32),
            )
            write = 1.0 - retention
            long_keys = retention.unsqueeze(-1) * long_keys + write.unsqueeze(-1) * (
                key_states[:, :, moved_index, :].float().unsqueeze(-2)
            )
            long_values = retention.unsqueeze(-1) * long_values + write.unsqueeze(
                -1
            ) * (value_states[:, :, moved_index, :].float().unsqueeze(-2))
        old_end = next_old_end
        local_end = position + 1
        local_keep = keep_mask[:, None, old_end:local_end].to(torch.bool)
        long_logits = (
            torch.einsum(
                "bhd,bhmd->bhm",
                query_states[:, :, position, :].float(),
                long_keys,
            )
            * module.scaling
        )
        local_logits = (
            torch.einsum(
                "bhd,bhwd->bhw",
                rope_query_states[:, :, position, :].float(),
                rope_key_states[:, :, old_end:local_end, :].float(),
            )
            * module.scaling
        )
        local_logits = local_logits.masked_fill(~local_keep, -torch.inf)
        logits = torch.cat((long_logits, local_logits), dim=-1)
        probabilities = torch.softmax(logits, dim=-1)
        output = torch.einsum(
            "bhm,bhmd->bhd",
            probabilities[:, :, : module.num_slots],
            long_values,
        ) + torch.einsum(
            "bhw,bhwd->bhd",
            probabilities[:, :, module.num_slots :],
            value_states[:, :, old_end:local_end, :].float(),
        )
        outputs.append(output)
        padding = total_columns - logits.shape[-1]
        logits_by_query.append(F.pad(logits, (0, padding), value=-torch.inf))
        probabilities_by_query.append(F.pad(probabilities, (0, padding), value=0.0))
        valid = torch.zeros(
            batch,
            module.num_heads,
            total_columns,
            dtype=torch.bool,
            device=hidden_states.device,
        )
        valid[:, :, : module.num_slots] = True
        valid[:, :, module.num_slots : module.num_slots + local_keep.shape[-1]] = (
            local_keep
        )
        valid_by_query.append(valid)
        long_keys_by_query.append(long_keys.clone())
        long_values_by_query.append(long_values.clone())

    attention_output_heads = torch.stack(outputs, dim=2).to(hidden_states.dtype)
    attention_output = attention_output_heads.transpose(1, 2).reshape(
        *input_shape,
        module.num_heads * module.head_dim,
    )
    output = _linear(attention_output, module.o_proj)
    components = {
        "query_states_without_rope": query_states,
        "query_states_with_rope": rope_query_states,
        "key_states_without_rope": key_states,
        "key_states_with_rope": rope_key_states,
        "value_states": value_states,
        "gate_logits": gate_logits,
        "retentions": retentions,
        "long_keys_by_query": torch.stack(long_keys_by_query, dim=2),
        "long_values_by_query": torch.stack(long_values_by_query, dim=2),
        "unified_logits": torch.stack(logits_by_query, dim=2),
        "unified_probabilities": torch.stack(probabilities_by_query, dim=2),
        "unified_valid_mask": torch.stack(valid_by_query, dim=2),
        "attention_output_heads": attention_output_heads,
        "attention_output_before_o_proj": attention_output,
    }
    return output, components


def _factorized_linear(value: Any, module: Any) -> Any:
    projected = F.linear(value, module.input_factor.weight, module.input_factor.bias)
    return F.linear(
        projected,
        module.output_factor.weight,
        module.output_factor.bias,
    )


def _reference_low_rank_ffn(module: Any, hidden_states: Any) -> Any:
    gate = _factorized_linear(hidden_states, module.gate_proj)
    up = _factorized_linear(hidden_states, module.up_proj)
    mixed = module.act_fn(gate) * up
    return _factorized_linear(mixed, module.down_proj)


def _reference_two_layer_mlp(module: Any, hidden_states: Any) -> Any:
    reduced = _linear(hidden_states, module.reduce_proj)
    activated = F.silu(reduced)
    return _linear(activated, module.expand_proj)


def capture_reference(
    module: Any,
    positional: tuple[Any, ...],
    keyword: dict[str, Any],
    *,
    strategy: str,
    target: str,
) -> dict[str, Any]:
    hidden_states = _argument(positional, keyword, "hidden_states", 0)
    if not torch.is_tensor(hidden_states):
        raise RuntimeError("Could not capture replacement hidden_states.")
    position_embeddings = _argument(positional, keyword, "position_embeddings", 1)
    attention_mask = _argument(positional, keyword, "attention_mask", 2)
    past_key_values = _argument(positional, keyword, "past_key_values", 3)
    cache_length_before = _cache_length(
        past_key_values,
        int(getattr(module, "layer_idx", 0)),
    )
    if strategy.startswith("attn_") and cache_length_before != 0:
        raise RuntimeError(
            "Reference capture must start at prefill with an empty KV cache; "
            f"observed cache length {cache_length_before}."
        )

    lolcats_components: dict[str, Any] | None = None
    nha_components: dict[str, Any] | None = None
    with torch.no_grad():
        if strategy == "attn_linear_attention":
            expected = _reference_linear_attention(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        elif strategy == "attn_cnn":
            expected = _reference_cnn(module, hidden_states, attention_mask)
        elif strategy == "attn_sparse_window":
            expected = _reference_sparse_window(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        elif strategy == "attn_lolcats_linear_window":
            expected, lolcats_components = _reference_lolcats_linear_window(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        elif strategy == "attn_nha":
            expected, nha_components = _reference_nha(
                module,
                hidden_states,
                position_embeddings,
                attention_mask,
            )
        elif strategy in {"attn_identity", "ffn_identity"}:
            expected = hidden_states.clone()
        elif strategy == "ffn_low_rank":
            expected = _reference_low_rank_ffn(module, hidden_states)
        elif strategy == "ffn_two_layer_mlp":
            expected = _reference_two_layer_mlp(module, hidden_states)
        else:
            raise RuntimeError(f"No reference implementation for {strategy!r}.")

    config = {
        name: _cpu_snapshot(getattr(module, name))
        for name in (
            "layer_idx",
            "head_dim",
            "num_heads",
            "num_key_value_heads",
            "num_key_value_groups",
            "eps",
            "kernel_size",
            "rank",
            "initialization",
            "bottleneck_dim",
            "bias",
            "causal",
            "include_self",
            "local_left",
            "local_right",
            "prefix_size",
            "prompt_suffix_size",
            "fixed_positions",
            "stride",
            "window_size",
            "decode_window_size",
            "feature_dim",
            "feature_map",
            "tie_qk_feature_maps",
            "window_weight_init",
            "train_window_weight",
            "affine_attention_factors",
            "num_slots",
            "gate_logit_normalizer",
        )
        if hasattr(module, name)
    }
    state_dict = {
        name: tensor.detach().to(device="cpu", copy=True)
        for name, tensor in module.state_dict().items()
    }
    extra_kwargs = {
        name: _cpu_snapshot(value)
        for name, value in keyword.items()
        if name
        not in {
            "hidden_states",
            "position_embeddings",
            "attention_mask",
            "past_key_values",
        }
    }
    trace = {
        "format_version": 2,
        "strategy": strategy,
        "target": target,
        "replacement_class": type(module).__qualname__,
        "call_index": 0,
        "capture_scope": "first_forward_prefill",
        "call_inputs": {
            "hidden_states": _cpu_snapshot(hidden_states),
            "position_embeddings": _cpu_snapshot(position_embeddings),
            "attention_mask": _cpu_snapshot(attention_mask),
            "past_key_values": {
                "python_type": type(past_key_values).__qualname__,
                "sequence_length_before": cache_length_before,
            },
            "kwargs": extra_kwargs,
        },
        "operator_config": config,
        "operator_state_dict": state_dict,
        "expected_output": _cpu_snapshot(expected),
    }
    if lolcats_components is not None:
        trace["lolcats_reference_components"] = _cpu_snapshot(lolcats_components)
    if nha_components is not None:
        trace["nha_reference_components"] = _cpu_snapshot(nha_components)
    return trace


def _validate_nha_components(trace: dict[str, Any]) -> dict[str, Any]:
    components = trace["nha_reference_components"]
    config = trace["operator_config"]
    num_slots = int(config["num_slots"])
    window_size = int(config["window_size"])
    normalizer = float(config["gate_logit_normalizer"])
    probabilities = components["unified_probabilities"].float()
    valid = components["unified_valid_mask"].to(torch.bool)
    probability_sums = probabilities.sum(dim=-1)
    invalid_probability = probabilities.masked_select(~valid)

    retentions = components["retentions"].float()
    expected_retentions = torch.exp(
        F.logsigmoid(components["gate_logits"].float()) / normalizer
    )
    key_states = components["key_states_without_rope"].float()
    value_states = components["value_states"].float()
    long_keys_history = components["long_keys_by_query"].float()
    long_values_history = components["long_values_by_query"].float()
    attention_heads = components["attention_output_heads"].float()
    sequence_length = int(key_states.shape[2])
    keep_mask = _key_keep_mask(
        trace["call_inputs"]["attention_mask"],
        sequence_length,
    )
    if keep_mask is None:
        keep_mask = torch.ones(key_states.shape[0], sequence_length, dtype=torch.bool)

    long_keys = torch.zeros_like(long_keys_history[:, :, 0])
    long_values = torch.zeros_like(long_values_history[:, :, 0])
    reconstructed_long_keys: list[Any] = []
    reconstructed_long_values: list[Any] = []
    reconstructed_heads: list[Any] = []
    old_end = 0
    for position in range(sequence_length):
        next_old_end = max(0, position - window_size + 1)
        for moved_index in range(old_end, next_old_end):
            keep = keep_mask[:, moved_index].to(torch.bool)
            retention = torch.where(
                keep[:, None, None],
                retentions[:, :, moved_index],
                torch.ones_like(retentions[:, :, moved_index]),
            )
            write = 1.0 - retention
            long_keys = retention.unsqueeze(-1) * long_keys + write.unsqueeze(-1) * (
                key_states[:, :, moved_index].unsqueeze(-2)
            )
            long_values = retention.unsqueeze(-1) * long_values + write.unsqueeze(
                -1
            ) * (value_states[:, :, moved_index].unsqueeze(-2))
        old_end = next_old_end
        reconstructed_long_keys.append(long_keys.clone())
        reconstructed_long_values.append(long_values.clone())
        local_count = position - old_end + 1
        current_probability = probabilities[:, :, position]
        reconstructed_heads.append(
            torch.einsum(
                "bhm,bhmd->bhd",
                current_probability[:, :, :num_slots],
                long_values,
            )
            + torch.einsum(
                "bhw,bhwd->bhd",
                current_probability[:, :, num_slots : num_slots + local_count],
                value_states[:, :, old_end : position + 1],
            )
        )
    reconstructed_long_keys_tensor = torch.stack(reconstructed_long_keys, dim=2)
    reconstructed_long_values_tensor = torch.stack(reconstructed_long_values, dim=2)
    reconstructed_heads_tensor = torch.stack(reconstructed_heads, dim=2)
    slot_key_difference = reconstructed_long_keys_tensor - long_keys_history
    slot_value_difference = reconstructed_long_values_tensor - long_values_history
    head_difference = reconstructed_heads_tensor - attention_heads
    expected_pairs = num_slots * sequence_length + sum(
        min(position + 1, window_size) for position in range(sequence_length)
    )
    full_pairs = sequence_length * (sequence_length + 1) // 2
    checks = {
        "retentions_match_gate_formula": bool(
            torch.allclose(
                retentions,
                expected_retentions,
                rtol=1.0e-5,
                atol=1.0e-6,
            )
        ),
        "retention_formula_max_abs_diff": float(
            (retentions - expected_retentions).abs().max().item()
        ),
        "retention_min": float(retentions.min().item()),
        "retention_max": float(retentions.max().item()),
        "unified_probability_sums_allclose_one": bool(
            torch.allclose(
                probability_sums,
                torch.ones_like(probability_sums),
                rtol=1.0e-6,
                atol=1.0e-6,
            )
        ),
        "unified_probability_sum_max_abs_error": float(
            (probability_sums - 1.0).abs().max().item()
        ),
        "invalid_probabilities_are_zero": bool(
            invalid_probability.numel() == 0
            or torch.count_nonzero(invalid_probability).item() == 0
        ),
        "long_key_recurrence_exact": bool(
            torch.equal(reconstructed_long_keys_tensor, long_keys_history)
        ),
        "long_key_recurrence_max_abs_diff": float(
            slot_key_difference.abs().max().item()
        ),
        "long_value_recurrence_exact": bool(
            torch.equal(reconstructed_long_values_tensor, long_values_history)
        ),
        "long_value_recurrence_max_abs_diff": float(
            slot_value_difference.abs().max().item()
        ),
        "probability_times_value_reconstructs_heads": bool(
            torch.allclose(
                reconstructed_heads_tensor,
                attention_heads,
                rtol=5.0e-3,
                atol=5.0e-3,
            )
        ),
        "head_reconstruction_max_abs_diff": float(head_difference.abs().max().item()),
        "nha_score_pairs": expected_pairs,
        "full_attention_score_pairs": full_pairs,
        "token_mixing_flops_ratio": expected_pairs / full_pairs,
    }
    checks["all_checks_passed"] = all(
        checks[name]
        for name in (
            "retentions_match_gate_formula",
            "unified_probability_sums_allclose_one",
            "invalid_probabilities_are_zero",
            "long_key_recurrence_exact",
            "long_value_recurrence_exact",
            "probability_times_value_reconstructs_heads",
        )
    )
    return checks


def finalize_reference(trace: dict[str, Any], actual_output: Any) -> dict[str, Any]:
    actual = actual_output.detach().to(device="cpu", copy=True)
    expected = trace["expected_output"]
    difference = actual.float() - expected.float()
    strategy = str(trace["strategy"])
    rtol = 0.0 if strategy.endswith("identity") else 5.0e-3
    atol = 0.0 if strategy.endswith("identity") else 5.0e-3
    validation = {
        "shape_equal": tuple(actual.shape) == tuple(expected.shape),
        "dtype_equal": actual.dtype == expected.dtype,
        "exact_equal": torch.equal(actual, expected),
        "allclose": torch.allclose(
            actual.float(), expected.float(), rtol=rtol, atol=atol
        ),
        "rtol": rtol,
        "atol": atol,
        "max_abs_diff": float(difference.abs().max().item()),
        "mean_abs_diff": float(difference.abs().mean().item()),
        "actual_all_finite": bool(torch.isfinite(actual.float()).all().item()),
        "expected_all_finite": bool(torch.isfinite(expected.float()).all().item()),
    }
    if strategy == "attn_nha":
        validation["nha_component_checks"] = _validate_nha_components(trace)
    trace["input"] = trace["call_inputs"]["hidden_states"]
    trace["actual_output"] = actual
    trace["output"] = actual
    trace["validation"] = validation
    return validation


__all__ = ["capture_reference", "finalize_reference"]
