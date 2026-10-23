"""Architecture-neutral reversible interventions over semantic decoder handles."""

from __future__ import annotations

import inspect
from typing import Any, MutableMapping, Sequence

from mllm_comp.analysis.common.tensors import hidden_tensor, replace_hidden_tensor
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError


def install_layer_update_skip(
    hooks: HookSession,
    *,
    layer: Any,
    layout: TokenGroupLayout,
    target_group: str,
) -> None:
    """Make selected token positions see the decoder layer as identity."""

    captured: dict[str, Any] = {}

    def capture_input(_module: Any, args: tuple[Any, ...]) -> None:
        captured["input"] = hidden_tensor(args)

    def replace_output(_module: Any, _args: Any, output: Any) -> Any:
        layer_input = captured.pop("input", None)
        if layer_input is None:
            raise BackendIntegrationError("Layer intervention missed its input hook.")
        layer_output = hidden_tensor(output)
        _require_matching_hidden(layer_input, layer_output)
        mask = token_group_mask(layout, layer_output, target_group).unsqueeze(-1)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("Layer interventions require PyTorch.") from exc
        mixed = torch.where(mask, layer_input, layer_output)
        return replace_hidden_tensor(output, mixed)

    hooks.register_forward_pre_hook(layer, capture_input)
    hooks.register_forward_hook(layer, replace_output)


def install_ffn_update_skip(
    hooks: HookSession,
    *,
    ffn: Any,
    layout: TokenGroupLayout,
    target_group: str,
) -> None:
    """Zero selected per-token FFN residual updates while preserving Attention."""

    install_sublayer_update_skip(
        hooks,
        sublayer=ffn,
        layout=layout,
        target_group=target_group,
    )


def install_sublayer_update_skip(
    hooks: HookSession,
    *,
    sublayer: Any,
    layout: TokenGroupLayout,
    target_group: str,
) -> None:
    """Zero selected residual updates emitted by Attention or FFN."""

    def replace_update(_module: Any, _args: Any, output: Any) -> Any:
        update = hidden_tensor(output)
        mask = token_group_mask(layout, update, target_group).unsqueeze(-1)
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise BackendIntegrationError("FFN interventions require PyTorch.") from exc
        changed = torch.where(mask, torch.zeros_like(update), update)
        return replace_hidden_tensor(output, changed)

    hooks.register_forward_hook(sublayer, replace_update)


def install_attention_edge_block(
    hooks: HookSession,
    *,
    attention: Any,
    layout: TokenGroupLayout,
    source_group: str,
    target_group: str | None,
    non_source_queries: bool = False,
    head_indices: Sequence[int] | None = None,
    num_heads: int | None = None,
    evidence: MutableMapping[str, Any] | None = None,
) -> None:
    """Block target-query to source-key edges in the provider's causal mask."""

    def replace_mask(
        module: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        hidden = _call_argument(module, args, kwargs, "hidden_states")
        base_mask = _call_argument(module, args, kwargs, "attention_mask")
        changed_mask, details = block_attention_edges(
            base_mask,
            hidden_states=hidden,
            layout=layout,
            source_group=source_group,
            target_group=target_group,
            non_source_queries=non_source_queries,
            head_indices=head_indices,
            num_heads=num_heads,
        )
        changed_args, changed_kwargs = _replace_call_argument(
            module,
            args,
            kwargs,
            "attention_mask",
            changed_mask,
        )
        if evidence is not None:
            evidence.update(details)
        return changed_args, changed_kwargs

    hooks.register_forward_pre_hook(attention, replace_mask, with_kwargs=True)


def install_attention_head_ablation(
    hooks: HookSession,
    *,
    output_projection: Any,
    head_indices: Sequence[int],
    num_heads: int,
    head_dim: int,
) -> None:
    """Zero selected per-head contributions immediately before ``o_proj``."""

    selected = _validated_indices(head_indices, upper=num_heads, name="Head")

    def replace_input(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            raise BackendIntegrationError("Head ablation received no o_proj input.")
        hidden = hidden_tensor(args[0])
        if len(hidden.shape) != 3 or int(hidden.shape[-1]) != num_heads * head_dim:
            raise BackendIntegrationError(
                "Head ablation input does not match bound head geometry."
            )
        changed = hidden.view(
            int(hidden.shape[0]),
            int(hidden.shape[1]),
            num_heads,
            head_dim,
        ).clone()
        changed[:, :, list(selected), :] = 0
        return (changed.reshape_as(hidden), *args[1:])

    hooks.register_forward_pre_hook(output_projection, replace_input)


def install_ffn_neuron_ablation(
    hooks: HookSession,
    *,
    down_projection: Any,
    neuron_indices: Sequence[int],
) -> None:
    """Zero selected intermediate FFN activations before ``down_proj``."""

    input_width = getattr(down_projection, "in_features", None)
    if input_width is None:
        raise BackendIntegrationError(
            "FFN neuron ablation requires down_proj.in_features."
        )
    selected = _validated_indices(
        neuron_indices,
        upper=int(input_width),
        name="FFN neuron",
    )

    def replace_input(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            raise BackendIntegrationError(
                "FFN neuron ablation received no down_proj input."
            )
        activation = hidden_tensor(args[0])
        if len(activation.shape) != 3 or int(activation.shape[-1]) != int(input_width):
            raise BackendIntegrationError(
                "FFN neuron ablation input does not match down_proj.in_features."
            )
        changed = activation.clone()
        changed[:, :, list(selected)] = 0
        return (changed, *args[1:])

    hooks.register_forward_pre_hook(down_projection, replace_input)


def install_token_hidden_replacement(
    hooks: HookSession,
    *,
    layer: Any,
    layout: TokenGroupLayout,
    target_group: str,
    replacement: Any,
    evidence: MutableMapping[str, Any] | None = None,
) -> None:
    """Replace selected token states at a provider decoder-layer input."""

    def replace_input(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
        if not args:
            raise BackendIntegrationError(
                "Token reinjection requires positional decoder hidden states."
            )
        hidden = hidden_tensor(args)
        mask = token_group_mask(layout, hidden, target_group)
        if len(replacement.shape) == 2:
            value = replacement.unsqueeze(0)
        elif len(replacement.shape) == 3:
            value = replacement
        else:
            raise BackendIntegrationError(
                "Token reinjection replacement must be [tokens, hidden] or "
                "[batch, tokens, hidden]."
            )
        token_count = int(mask.sum().detach().cpu().item())
        if (
            int(value.shape[0]) != int(hidden.shape[0])
            or int(value.shape[1]) != token_count
            or int(value.shape[2]) != int(hidden.shape[2])
        ):
            raise BackendIntegrationError(
                "Token reinjection shape does not match selected hidden states: "
                f"replacement={tuple(value.shape)}, selected_tokens={token_count}, "
                f"hidden={tuple(hidden.shape)}."
            )
        changed = hidden.clone()
        changed[mask] = value.to(device=hidden.device, dtype=hidden.dtype).reshape(
            token_count,
            int(hidden.shape[2]),
        )
        if evidence is not None:
            evidence.update(
                {
                    "reinjection_contract": "decoder_layer_input_token_reset_v1",
                    "reinjected_token_count": token_count,
                    "target_group": target_group,
                }
            )
        return (changed, *args[1:])

    hooks.register_forward_pre_hook(layer, replace_input)


def block_attention_edges(
    attention_mask: Any,
    *,
    hidden_states: Any,
    layout: TokenGroupLayout,
    source_group: str,
    target_group: str | None,
    non_source_queries: bool = False,
    head_indices: Sequence[int] | None = None,
    num_heads: int | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return a cloned four-dimensional mask with requested causal edges blocked."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Attention interventions require PyTorch.") from exc
    if attention_mask is None or not hasattr(attention_mask, "shape"):
        raise BackendIntegrationError(
            "Attention edge intervention requires the provider's explicit causal mask."
        )
    if len(attention_mask.shape) != 4:
        raise BackendIntegrationError(
            "Attention edge intervention requires a four-dimensional causal mask; "
            f"got {tuple(attention_mask.shape)}. Use eager or SDPA attention."
        )
    if len(hidden_states.shape) != 3:
        raise BackendIntegrationError("Attention hidden states must be [batch, query, hidden].")
    batch_size, query_length, _ = (int(value) for value in hidden_states.shape)
    key_length = int(attention_mask.shape[-1])
    if int(attention_mask.shape[0]) != batch_size or int(attention_mask.shape[-2]) != query_length:
        raise BackendIntegrationError(
            "Attention causal-mask batch/query shape does not match hidden states."
        )
    if len(layout.labels_by_sample) != batch_size or any(
        len(row) != query_length or len(row) != key_length
        for row in layout.labels_by_sample
    ):
        raise BackendIntegrationError(
            "Attention edge intervention requires exact prefill query/key token layouts."
        )

    source = token_group_mask(layout, hidden_states, source_group)
    if non_source_queries:
        valid = token_group_mask(layout, hidden_states, "all")
        queries = valid & ~source
        query_semantics = "valid_non_source"
    else:
        if target_group is None:
            raise BackendIntegrationError("Attention target_group is required.")
        queries = token_group_mask(layout, hidden_states, target_group)
        query_semantics = f"target:{target_group}"
    edge_mask = queries[:, None, :, None] & source[:, None, None, :]
    changed = attention_mask.clone()
    selected_heads: tuple[int, ...] | None = None
    if head_indices is not None:
        if num_heads is None or num_heads <= 0:
            raise BackendIntegrationError(
                "Per-head attention edge blocking requires a positive num_heads."
            )
        selected_heads = _validated_indices(
            head_indices,
            upper=int(num_heads),
            name="Attention mask Head",
        )
        mask_heads = int(changed.shape[1])
        if mask_heads == 1:
            changed = changed.expand(
                batch_size,
                int(num_heads),
                query_length,
                key_length,
            ).clone()
        elif mask_heads != int(num_heads):
            raise BackendIntegrationError(
                "Per-head attention mask does not have one broadcast head or the "
                f"bound head count: {mask_heads} vs {num_heads}."
            )
        head_mask = torch.zeros(
            (1, int(num_heads), 1, 1),
            dtype=torch.bool,
            device=changed.device,
        )
        head_mask[:, list(selected_heads)] = True
        edge_mask = edge_mask & head_mask
    if changed.dtype == torch.bool:
        allowed_before = changed
        blocked_value: bool | float = False
    elif changed.dtype.is_floating_point:
        minimum = torch.finfo(changed.dtype).min
        allowed_before = torch.isfinite(changed) & (changed > minimum / 2)
        blocked_value = minimum
    else:
        raise BackendIntegrationError(
            f"Unsupported attention mask dtype {changed.dtype}; expected bool or float."
        )
    requested = edge_mask.expand_as(allowed_before)
    selected_allowed = allowed_before & requested
    remaining_allowed = allowed_before & ~requested
    degenerate_rows = selected_allowed.any(dim=-1, keepdim=True) & ~(
        remaining_allowed.any(dim=-1, keepdim=True)
    )
    applied_edges = selected_allowed & ~degenerate_rows
    changed = changed.masked_fill(applied_edges, blocked_value)
    degenerate_count = int(degenerate_rows.sum().detach().cpu().item())
    return changed, {
        "mask_contract": "safe_partial_explicit_4d_causal_mask_v2",
        "source_group": source_group,
        "query_semantics": query_semantics,
        "candidate_edge_count": int(requested.sum().detach().cpu().item()),
        "causally_visible_candidate_edge_count": int(
            selected_allowed.sum().detach().cpu().item()
        ),
        "newly_blocked_edge_count": int(
            applied_edges.sum().detach().cpu().item()
        ),
        "degenerate_query_head_count": degenerate_count,
        "intervention_scope": (
            "complete" if degenerate_count == 0 else "safe_partial"
        ),
        "degenerate_rows_left_unchanged": True,
        "head_indices": list(selected_heads) if selected_heads is not None else None,
    }


def token_group_mask(
    layout: TokenGroupLayout,
    hidden_states: Any,
    group: str,
) -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover
        raise BackendIntegrationError("Token-group interventions require PyTorch.") from exc
    if len(hidden_states.shape) != 3:
        raise BackendIntegrationError("Intervention hidden states must be [batch, sequence, hidden].")
    batch_size, sequence_length, _ = (int(value) for value in hidden_states.shape)
    if len(layout.labels_by_sample) != batch_size or any(
        len(row) != sequence_length for row in layout.labels_by_sample
    ):
        raise BackendIntegrationError(
            "Intervention token-group layout does not match hidden states."
        )

    def selected(label: str) -> bool:
        if group == "all":
            return label != "padding"
        if group.startswith("non") and len(group) > 3:
            positive_group = group[3:]
            return label != "padding" and not _label_in_group(
                label,
                positive_group,
            )
        return _label_in_group(label, group)

    return torch.tensor(
        [[selected(label) for label in row] for row in layout.labels_by_sample],
        dtype=torch.bool,
        device=hidden_states.device,
    )


def _label_in_group(label: str, group: str) -> bool:
    if group == "vision":
        return label in {"image", "video"}
    return label == group


def _require_matching_hidden(left: Any, right: Any) -> None:
    if tuple(left.shape) != tuple(right.shape) or len(left.shape) != 3:
        raise BackendIntegrationError(
            "Layer intervention input/output hidden shapes disagree: "
            f"{tuple(left.shape)} vs {tuple(right.shape)}."
        )


def _validated_indices(
    values: Sequence[int],
    *,
    upper: int,
    name: str,
) -> tuple[int, ...]:
    selected = tuple(sorted({int(value) for value in values}))
    if not selected:
        raise BackendIntegrationError(f"{name} ablation selection cannot be empty.")
    if selected[0] < 0 or selected[-1] >= upper:
        raise BackendIntegrationError(
            f"{name} ablation indices {selected} exceed [0,{upper})."
        )
    return selected


def _call_argument(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
) -> Any:
    if name in kwargs:
        return kwargs[name]
    parameters = list(inspect.signature(module.forward).parameters)
    if name in parameters:
        index = parameters.index(name)
        if index < len(args):
            return args[index]
    raise BackendIntegrationError(
        f"Attention hook could not resolve forward argument {name!r}."
    )


def _replace_call_argument(
    module: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    name: str,
    value: Any,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    changed_kwargs = dict(kwargs)
    if name in changed_kwargs:
        changed_kwargs[name] = value
        return args, changed_kwargs
    parameters = list(inspect.signature(module.forward).parameters)
    if name in parameters:
        index = parameters.index(name)
        if index < len(args):
            changed_args = list(args)
            changed_args[index] = value
            return tuple(changed_args), changed_kwargs
        changed_kwargs[name] = value
        return args, changed_kwargs
    raise BackendIntegrationError(
        f"Attention hook could not replace forward argument {name!r}."
    )


__all__ = [
    "block_attention_edges",
    "install_attention_edge_block",
    "install_attention_head_ablation",
    "install_ffn_neuron_ablation",
    "install_ffn_update_skip",
    "install_layer_update_skip",
    "install_sublayer_update_skip",
    "install_token_hidden_replacement",
    "token_group_mask",
]
