"""One-backward hierarchical Taylor capture over semantic decoder handles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from mllm_comp.analysis.bindings import AttentionGeometry, DecoderLayerBinding
from mllm_comp.analysis.common.taylor import (
    DIRECT_REMOVAL_SCORE,
    NORMALIZED_SCORE,
    PATH_NORMALIZED_SCORE,
    taylor_component_groups,
    taylor_path_groups,
    tensor_taylor_reductions,
)
from mllm_comp.analysis.common.tensors import hidden_tensor
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError


class HierarchicalTaylorCapture:
    """Capture all hierarchy levels for one bounded layer batch and backward."""

    def __init__(
        self,
        layers: Sequence[DecoderLayerBinding],
        geometry: AttentionGeometry,
        *,
        target_modality: str = "vision",
    ) -> None:
        self.layers = tuple(layers)
        self.geometry = geometry
        self.target_modality = target_modality
        self.component_groups = taylor_component_groups(target_modality)
        self.path_groups = taylor_path_groups(target_modality)
        self.layout: TokenGroupLayout | None = None
        self.activations: dict[str, dict[int, Any]] = {
            "layer": {},
            "attention": {},
            "ffn": {},
            "head": {},
            "value": {},
            "ffn_neuron": {},
        }
        self.gradients: dict[str, dict[int, Any]] = {
            "head": {},
        }
        self.results: dict[str, dict[int, Any]] = {
            "layer": {},
            "attention": {},
            "ffn": {},
            "ffn_neuron": {},
        }
        self._errors: list[str] = []
        self._installed = False

    def install(self, hooks: HookSession, *, input_embeddings: Any) -> None:
        if self._installed:
            raise BackendIntegrationError("Taylor capture hooks were already installed.")
        self._installed = True

        def enable_embedding_grad(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = hidden_tensor(output)
            if not bool(getattr(tensor, "requires_grad", False)):
                requires_grad = getattr(tensor, "requires_grad_", None)
                if not callable(requires_grad):
                    raise BackendIntegrationError(
                        "Embedding output cannot enable activation gradients."
                    )
                requires_grad(True)
            return output

        hooks.register_forward_hook(input_embeddings, enable_embedding_grad)
        for layer in self.layers:
            index = layer.index
            hooks.register_forward_hook(
                layer.layer,
                self._layer_hook(index),
            )
            hooks.register_forward_hook(
                layer.attention,
                self._component_hook("attention", index),
            )
            hooks.register_forward_hook(
                layer.ffn,
                self._component_hook("ffn", index),
            )
            hooks.register_forward_pre_hook(
                layer.attention_output_projection,
                self._activation_pre_hook("head", index, retain_gradient=True),
            )
            hooks.register_forward_hook(
                layer.attention_value_projection,
                self._activation_forward_hook("value", index),
            )
            hooks.register_forward_pre_hook(
                layer.ffn_down_projection,
                self._ffn_neuron_hook(index),
            )

    def bind_layout(self, layout: TokenGroupLayout) -> None:
        if self.layout is not None:
            raise BackendIntegrationError("Taylor token layout was already bound.")
        sequence_lengths = {
            int(tensor.shape[1])
            for group in self.activations.values()
            for tensor in group.values()
            if hasattr(tensor, "shape") and len(tensor.shape) == 3
        }
        if sequence_lengths != {len(layout.labels_by_sample[0])}:
            raise BackendIntegrationError(
                "Taylor captured inconsistent expanded sequence lengths: "
                f"captured={sorted(sequence_lengths)}, layout="
                f"{len(layout.labels_by_sample[0])}."
            )
        self.layout = layout
        for layer, activation in self.activations["ffn_neuron"].items():
            self.results["ffn_neuron"].setdefault(layer, {})[
                "activation_magnitude"
            ] = _ffn_activation_report(
                activation,
                layout,
                component_groups=self.component_groups,
                target_modality=self.target_modality,
            )

    def finish(self, attentions: Mapping[int, Any]) -> dict[str, Any]:
        if self.layout is None:
            raise BackendIntegrationError("Taylor capture has no token layout.")
        if self._errors:
            raise BackendIntegrationError(
                "Taylor activation capture failed: " + "; ".join(self._errors[:5])
            )
        expected = {layer.index for layer in self.layers}
        for kind in ("layer", "attention", "ffn", "ffn_neuron"):
            missing = sorted(expected - set(self.results[kind]))
            if missing:
                raise BackendIntegrationError(
                    f"Taylor backward did not produce {kind} results for layers {missing}."
                )
        missing_heads = sorted(expected - set(self.gradients["head"]))
        if missing_heads:
            raise BackendIntegrationError(
                f"Taylor backward did not produce Head gradients for layers {missing_heads}."
            )
        missing_attentions = sorted(expected - set(attentions))
        if missing_attentions:
            raise BackendIntegrationError(
                "Taylor attention capture does not cover selected decoder layers: "
                f"{missing_attentions}."
            )
        layer_results: dict[str, Any] = {}
        for layer in self.layers:
            index = layer.index
            attention = attentions[index]
            if attention is None:
                raise BackendIntegrationError(
                    f"Taylor layer {index} returned no exact attention weights."
                )
            heads, paths = _head_and_path_reports(
                attention=attention,
                value_projection=self.activations["value"][index],
                head_activation=self.activations["head"][index],
                head_gradient=self.gradients["head"][index],
                layout=self.layout,
                geometry=self.geometry,
                component_groups=self.component_groups,
                path_groups=self.path_groups,
                target_modality=self.target_modality,
            )
            neuron = self.results["ffn_neuron"][index]
            layer_results[str(index)] = {
                "path": paths,
                "head": heads,
                "ffn_neuron": neuron,
                "sublayer": {
                    "attention": self.results["attention"][index],
                    "ffn": self.results["ffn"][index],
                },
                "layer": self.results["layer"][index],
            }
        return {
            "score_definitions": {
                "normalized": NORMALIZED_SCORE,
                "path_normalized": PATH_NORMALIZED_SCORE,
                "direct_removal": DIRECT_REMOVAL_SCORE,
            },
            "target_modality": self.target_modality,
            "component_groups": list(self.component_groups),
            "path_types": list(self.path_groups),
            "layers": layer_results,
        }

    def clear(self) -> None:
        self.layout = None
        for values in self.activations.values():
            values.clear()
        for values in self.gradients.values():
            values.clear()
        for values in self.results.values():
            values.clear()
        self._errors.clear()

    def _layer_hook(self, layer: int):
        def hook(_module: Any, inputs: Any, output: Any) -> Any:
            try:
                input_tensor = _first_input_tensor(inputs, name=f"layer {layer}")
                output_tensor = _validated_activation(
                    hidden_tensor(output), name=f"layer {layer} output"
                )
                if tuple(input_tensor.shape) != tuple(output_tensor.shape):
                    raise BackendIntegrationError(
                        f"Taylor layer {layer} input/output shapes differ: "
                        f"{tuple(input_tensor.shape)} vs {tuple(output_tensor.shape)}."
                    )
                update = output_tensor.detach() - input_tensor.detach()
                self._store_unique("layer", layer, update)
                self._register_component_gradient("layer", layer, output_tensor, update)
            except Exception as exc:
                self._errors.append(f"L{layer} layer: {type(exc).__name__}: {exc}")
            return output

        return hook

    def _component_hook(self, kind: str, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            try:
                tensor = _validated_activation(
                    hidden_tensor(output), name=f"layer {layer} {kind} output"
                )
                activation = tensor.detach()
                self._store_unique(kind, layer, activation)
                self._register_component_gradient(kind, layer, tensor, activation)
            except Exception as exc:
                self._errors.append(f"L{layer} {kind}: {type(exc).__name__}: {exc}")
            return output

        return hook

    def _activation_pre_hook(self, kind: str, layer: int, *, retain_gradient: bool):
        def hook(_module: Any, inputs: Any) -> None:
            try:
                tensor = _first_input_tensor(inputs, name=f"layer {layer} {kind}")
                self._store_unique(kind, layer, tensor.detach())
                if retain_gradient:
                    _require_activation_grad(tensor, name=f"layer {layer} {kind}")

                    def on_gradient(gradient: Any) -> Any:
                        self.gradients[kind][layer] = gradient.detach()
                        return gradient

                    tensor.register_hook(on_gradient)
            except Exception as exc:
                self._errors.append(f"L{layer} {kind}: {type(exc).__name__}: {exc}")

        return hook

    def _activation_forward_hook(self, kind: str, layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            try:
                tensor = _validated_activation(
                    hidden_tensor(output), name=f"layer {layer} {kind} output"
                )
                self._store_unique(kind, layer, tensor.detach())
            except Exception as exc:
                self._errors.append(f"L{layer} {kind}: {type(exc).__name__}: {exc}")
            return output

        return hook

    def _ffn_neuron_hook(self, layer: int):
        def hook(_module: Any, inputs: Any) -> None:
            try:
                tensor = _first_input_tensor(inputs, name=f"layer {layer} FFN neuron")
                activation = tensor.detach()
                self._store_unique("ffn_neuron", layer, activation)
                _require_activation_grad(tensor, name=f"layer {layer} FFN neuron")

                def on_gradient(gradient: Any) -> Any:
                    if self.layout is None:
                        self._errors.append(
                            f"L{layer} FFN neuron gradient arrived before token layout."
                        )
                    else:
                        taylor = _ffn_taylor_report(
                            activation,
                            gradient,
                            self.layout,
                            component_groups=self.component_groups,
                            target_modality=self.target_modality,
                        )
                        current = self.results["ffn_neuron"].setdefault(layer, {})
                        current["taylor"] = taylor
                    return gradient

                tensor.register_hook(on_gradient)
            except Exception as exc:
                self._errors.append(
                    f"L{layer} FFN neuron: {type(exc).__name__}: {exc}"
                )

        return hook

    def _register_component_gradient(
        self,
        kind: str,
        layer: int,
        tensor: Any,
        activation: Any,
    ) -> None:
        _require_activation_grad(tensor, name=f"layer {layer} {kind}")

        def on_gradient(gradient: Any) -> Any:
            if self.layout is None:
                self._errors.append(
                    f"L{layer} {kind} gradient arrived before token layout."
                )
            else:
                self.results[kind][layer] = _component_report(
                    activation,
                    gradient,
                    self.layout,
                    component_groups=self.component_groups,
                    target_modality=self.target_modality,
                )
            return gradient

        tensor.register_hook(on_gradient)

    def _store_unique(self, kind: str, layer: int, tensor: Any) -> None:
        if layer in self.activations[kind]:
            raise BackendIntegrationError(
                f"Taylor {kind} hook fired more than once for layer {layer}."
            )
        self.activations[kind][layer] = tensor


def _component_report(
    activation: Any,
    gradient: Any,
    layout: TokenGroupLayout,
    *,
    component_groups: Sequence[str],
    target_modality: str,
) -> dict[str, Any]:
    if tuple(activation.shape) != tuple(gradient.shape):
        raise BackendIntegrationError(
            "Taylor component activation/gradient shapes do not match."
        )
    reports: dict[str, Any] = {}
    for group in component_groups:
        positions = _group_positions(
            layout,
            group,
            target_modality=target_modality,
        )
        if not positions:
            reports[group] = {"status": "absent", "token_count": 0}
            continue
        selected_activation = _select_tokens(activation, positions)
        selected_gradient = _select_tokens(gradient, positions)
        reductions = tensor_taylor_reductions(
            selected_activation.float() * selected_gradient.float(),
            reduce_dims=(0, 1, 2),
        )
        reports[group] = _scalar_reduction_report(
            reductions,
            token_count=len(positions),
        )
    return reports


def _ffn_taylor_report(
    activation: Any,
    gradient: Any,
    layout: TokenGroupLayout,
    *,
    component_groups: Sequence[str],
    target_modality: str,
) -> dict[str, Any]:
    if tuple(activation.shape) != tuple(gradient.shape):
        raise BackendIntegrationError(
            "FFN neuron activation/gradient shapes do not match."
        )
    reports: dict[str, Any] = {}
    for group in component_groups:
        positions = _group_positions(
            layout,
            group,
            target_modality=target_modality,
        )
        if not positions:
            reports[group] = {"status": "absent", "token_count": 0}
            continue
        product = _select_tokens(activation, positions).float() * _select_tokens(
            gradient, positions
        ).float()
        reductions = tensor_taylor_reductions(
            product,
            reduce_dims=(0, 1),
            normalization_count=len(positions),
        )
        reports[group] = {
            "status": "finite",
            "token_count": len(positions),
            "normalized_score": _vector_to_list(reductions["normalized"]),
            "direct_removal_score": _vector_to_list(
                reductions["direct_removal"]
            ),
            "signed_sum": _vector_to_list(reductions["signed_sum"]),
            "neuron_count": int(product.shape[-1]),
        }
    return reports


def _ffn_activation_report(
    activation: Any,
    layout: TokenGroupLayout,
    *,
    component_groups: Sequence[str],
    target_modality: str,
) -> dict[str, Any]:
    reports: dict[str, Any] = {}
    for group in component_groups:
        positions = _group_positions(
            layout,
            group,
            target_modality=target_modality,
        )
        if not positions:
            reports[group] = {"status": "absent", "token_count": 0}
            continue
        selected = _select_tokens(activation, positions).float()
        values = selected.abs().mean(dim=(0, 1))
        if not bool(_torch().isfinite(values).all().item()):
            raise BackendIntegrationError("FFN activation magnitude is non-finite.")
        reports[group] = {
            "status": "finite",
            "token_count": len(positions),
            "mean_abs_activation": _vector_to_list(values),
            "neuron_count": int(selected.shape[-1]),
        }
    return reports


def _head_and_path_reports(
    *,
    attention: Any,
    value_projection: Any,
    head_activation: Any,
    head_gradient: Any,
    layout: TokenGroupLayout,
    geometry: AttentionGeometry,
    component_groups: Sequence[str],
    path_groups: Mapping[str, tuple[str, str]],
    target_modality: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    torch = _torch()
    if len(attention.shape) != 4 or int(attention.shape[0]) != 1:
        raise BackendIntegrationError(
            "Taylor attention weights must have shape [1, head, query, key]."
        )
    batch, sequence, width = (int(value) for value in head_activation.shape)
    if batch != 1 or width != geometry.hidden_size:
        raise BackendIntegrationError(
            "Taylor Head activation does not match bound Attention geometry."
        )
    if tuple(head_activation.shape) != tuple(head_gradient.shape):
        raise BackendIntegrationError("Taylor Head activation/gradient shapes differ.")
    if int(attention.shape[1]) != geometry.num_query_heads or int(
        attention.shape[-2]
    ) != sequence or int(attention.shape[-1]) != sequence:
        raise BackendIntegrationError(
            "Taylor attention weights do not match Head/sequence geometry."
        )
    expected_value_width = geometry.num_key_value_heads * geometry.head_dim
    if tuple(value_projection.shape[:2]) != (1, sequence) or int(
        value_projection.shape[-1]
    ) != expected_value_width:
        raise BackendIntegrationError(
            "Taylor value projection does not match KV-head geometry."
        )

    head_values = head_activation.float().view(
        1, sequence, geometry.num_query_heads, geometry.head_dim
    ).permute(0, 2, 1, 3)[0]
    head_grads = head_gradient.float().view(
        1, sequence, geometry.num_query_heads, geometry.head_dim
    ).permute(0, 2, 1, 3)[0]
    kv_values = value_projection.float().view(
        1, sequence, geometry.num_key_value_heads, geometry.head_dim
    ).permute(0, 2, 1, 3)[0]
    if geometry.query_heads_per_key_value_head > 1:
        kv_values = kv_values.repeat_interleave(
            geometry.query_heads_per_key_value_head,
            dim=0,
        )
    if int(kv_values.shape[0]) != geometry.num_query_heads:
        raise BackendIntegrationError("Taylor GQA/MQA value-head expansion failed.")

    head_reports: dict[str, Any] = {}
    for group in component_groups:
        positions = _group_positions(
            layout,
            group,
            target_modality=target_modality,
        )
        if not positions:
            head_reports[group] = {"status": "absent", "token_count": 0}
            continue
        index = torch.tensor(positions, dtype=torch.long, device=head_values.device)
        product = head_values.index_select(1, index) * head_grads.index_select(1, index)
        reductions = tensor_taylor_reductions(
            product,
            reduce_dims=(1, 2),
        )
        head_reports[group] = {
            "status": "finite",
            "token_count": len(positions),
            "normalized_score": _vector_to_list(reductions["normalized"]),
            "direct_removal_score": _vector_to_list(
                reductions["direct_removal"]
            ),
            "signed_sum": _vector_to_list(reductions["signed_sum"]),
            "head_count": geometry.num_query_heads,
        }

    path_reports: dict[str, Any] = {}
    attention_heads = attention.detach().float()[0]
    for path, (query_group, key_group) in path_groups.items():
        query_positions = _group_positions(
            layout,
            query_group,
            target_modality=target_modality,
        )
        key_positions = _group_positions(
            layout,
            key_group,
            target_modality=target_modality,
        )
        if not query_positions or not key_positions:
            path_reports[path] = {
                "status": "structurally_empty",
                "query_count": len(query_positions),
                "key_count": len(key_positions),
            }
            continue
        query_index = torch.tensor(
            query_positions,
            dtype=torch.long,
            device=attention_heads.device,
        )
        key_index = torch.tensor(
            key_positions,
            dtype=torch.long,
            device=attention_heads.device,
        )
        attention_block = attention_heads.index_select(1, query_index).index_select(
            2, key_index
        )
        values_block = kv_values.to(attention_heads.device).index_select(1, key_index)
        gradients_block = head_grads.to(attention_heads.device).index_select(
            1, query_index
        )
        contribution = torch.bmm(attention_block, values_block)
        product = contribution * gradients_block
        if not bool(torch.isfinite(product).all().item()):
            raise BackendIntegrationError(
                f"Taylor {path.upper()} block contains non-finite values."
            )
        signed_sum = product.sum(dim=(1, 2))
        path_reports[path] = {
            "status": "finite",
            "query_count": len(query_positions),
            "key_count": len(key_positions),
            "normalized_score": _vector_to_list(
                signed_sum.abs() / len(query_positions)
            ),
            "direct_removal_score": _vector_to_list(signed_sum.abs()),
            "signed_sum": _vector_to_list(signed_sum),
            "elementwise_mean_abs": _vector_to_list(product.abs().mean(dim=(1, 2))),
            "head_count": geometry.num_query_heads,
        }
    return head_reports, path_reports


def _group_positions(
    layout: TokenGroupLayout,
    group: str,
    *,
    target_modality: str,
) -> tuple[int, ...]:
    labels = layout.labels_by_sample[0]
    if group == "all":
        return tuple(index for index, label in enumerate(labels) if label != "padding")
    if group == target_modality:
        return layout.positions(0, target_modality)
    if group == "text":
        return tuple(
            index
            for index, label in enumerate(labels)
            if label != "padding"
            and index not in layout.positions(0, target_modality)
        )
    raise BackendIntegrationError(f"Unknown Taylor token group {group!r}.")


def _select_tokens(value: Any, positions: Sequence[int]) -> Any:
    torch = _torch()
    indices = torch.tensor(positions, dtype=torch.long, device=value.device)
    return value.index_select(1, indices)


def _scalar_reduction_report(
    reductions: Mapping[str, Any],
    *,
    token_count: int,
) -> dict[str, Any]:
    return {
        "status": "finite",
        "token_count": token_count,
        "normalized_score": _scalar(reductions["normalized"]),
        "direct_removal_score": _scalar(reductions["direct_removal"]),
        "signed_sum": _scalar(reductions["signed_sum"]),
        "normalization_count": int(reductions["normalization_count"]),
    }


def _first_input_tensor(inputs: Any, *, name: str) -> Any:
    if not isinstance(inputs, (tuple, list)) or not inputs:
        raise BackendIntegrationError(f"Taylor {name} hook received no positional input.")
    return _validated_activation(hidden_tensor(inputs[0]), name=name)


def _validated_activation(value: Any, *, name: str) -> Any:
    if len(value.shape) != 3 or int(value.shape[0]) != 1:
        raise BackendIntegrationError(
            f"Taylor {name} must have shape [1, sequence, width], got "
            f"{tuple(value.shape)}."
        )
    return value


def _require_activation_grad(value: Any, *, name: str) -> None:
    if not bool(getattr(value, "requires_grad", False)):
        raise BackendIntegrationError(
            f"Taylor {name} does not require activation gradients."
        )


def _scalar(value: Any) -> float:
    result = float(value.detach().cpu().item())
    if not _is_finite(result):
        raise BackendIntegrationError("Taylor scalar result is non-finite.")
    return result


def _vector_to_list(value: Any) -> list[float]:
    result = [float(item) for item in value.detach().cpu().tolist()]
    if any(not _is_finite(item) for item in result):
        raise BackendIntegrationError("Taylor vector result contains non-finite values.")
    return result


def _is_finite(value: float) -> bool:
    import math

    return math.isfinite(value)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runtime requires torch
        raise BackendIntegrationError("Hierarchical Taylor capture requires PyTorch.") from exc
    return torch


__all__ = ["HierarchicalTaylorCapture"]
