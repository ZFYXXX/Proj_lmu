"""Qwen3-VL feed-forward replacement modules."""

from __future__ import annotations

from typing import Any

from mllm_comp.core.errors import BackendIntegrationError

try:  # Keep schema-only imports usable without the optional torch extra.
    import torch
    from torch import nn
except ImportError:  # pragma: no cover - exercised by no-torch contract jobs.
    torch = None
    nn = None


def _missing_torch() -> BackendIntegrationError:
    return BackendIntegrationError(
        "Qwen3-VL FFN replacement requires PyTorch in the runtime environment."
    )


def _shape(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    return None if shape is None else tuple(int(dim) for dim in shape)


if nn is not None:

    class Qwen3VLLowRankLinear(nn.Module):
        """Two-factor linear layer initialized from sampled checkpoint columns."""

        def __init__(
            self,
            original_linear: Any,
            *,
            rank: int,
            initialization: str = "column_sampling",
        ) -> None:
            super().__init__()
            required = ("in_features", "out_features", "weight")
            missing = [name for name in required if not hasattr(original_linear, name)]
            if missing:
                raise BackendIntegrationError(
                    f"Low-rank replacement received an incompatible linear layer; missing {missing}."
                )
            self.in_features = int(original_linear.in_features)
            self.out_features = int(original_linear.out_features)
            self.rank = int(rank)
            if self.rank < 1 or self.rank > min(self.in_features, self.out_features):
                raise ValueError(
                    "low-rank projection rank must satisfy "
                    f"1 <= rank <= {min(self.in_features, self.out_features)}, got {self.rank}."
                )
            if initialization != "column_sampling":
                raise ValueError(
                    "low-rank initialization must be 'column_sampling', "
                    f"got {initialization!r}."
                )
            self.initialization = initialization
            self.input_factor = nn.Linear(self.in_features, self.rank, bias=False)
            self.output_factor = nn.Linear(
                self.rank,
                self.out_features,
                bias=getattr(original_linear, "bias", None) is not None,
            )
            reference_weight = original_linear.weight
            self.to(device=reference_weight.device, dtype=reference_weight.dtype)
            indices = (
                torch.linspace(
                    0,
                    self.in_features - 1,
                    steps=self.rank,
                    device=reference_weight.device,
                )
                .round()
                .to(dtype=torch.long)
            )
            with torch.no_grad():
                self.input_factor.weight.zero_()
                self.input_factor.weight[
                    torch.arange(self.rank, device=reference_weight.device),
                    indices,
                ] = 1
                self.output_factor.weight.copy_(reference_weight[:, indices])
                if self.output_factor.bias is not None:
                    self.output_factor.bias.copy_(original_linear.bias)
            self.register_buffer("sampled_input_indices", indices, persistent=False)

        @property
        def compression_ratio(self) -> float:
            original = self.in_features * self.out_features
            factored = self.rank * (self.in_features + self.out_features)
            return factored / original

        def forward(self, hidden_states: Any) -> Any:
            return self.output_factor(self.input_factor(hidden_states))

    class Qwen3VLLowRankMLP(nn.Module):
        """Qwen3-VL gated MLP with low-rank gate/up/down projections."""

        strategy_name = "ffn_low_rank"

        def __init__(
            self,
            original_mlp: Any,
            *,
            rank: int = 256,
            initialization: str = "column_sampling",
            debug: bool = False,
        ) -> None:
            super().__init__()
            required = ("gate_proj", "up_proj", "down_proj", "act_fn")
            missing = [name for name in required if not hasattr(original_mlp, name)]
            if missing:
                raise BackendIntegrationError(
                    f"Qwen3-VL low-rank FFN received an incompatible MLP; missing {missing}."
                )
            self.rank = int(rank)
            self.initialization = initialization
            self.gate_proj = Qwen3VLLowRankLinear(
                original_mlp.gate_proj,
                rank=self.rank,
                initialization=initialization,
            )
            self.up_proj = Qwen3VLLowRankLinear(
                original_mlp.up_proj,
                rank=self.rank,
                initialization=initialization,
            )
            self.down_proj = Qwen3VLLowRankLinear(
                original_mlp.down_proj,
                rank=self.rank,
                initialization=initialization,
            )
            self.act_fn = original_mlp.act_fn
            self.debug = bool(debug)
            self.forward_calls = 0
            self.last_debug: dict[str, Any] = {}
            self.train(bool(getattr(original_mlp, "training", False)))

        @property
        def parameter_compression_ratio(self) -> float:
            factors = (self.gate_proj, self.up_proj, self.down_proj)
            original = sum(item.in_features * item.out_features for item in factors)
            factored = sum(
                item.rank * (item.in_features + item.out_features) for item in factors
            )
            return factored / original

        def forward(self, hidden_states: Any) -> Any:
            output = self.down_proj(
                self.act_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
            )
            self.forward_calls += 1
            self.last_debug = {
                "strategy": self.strategy_name,
                "rank": self.rank,
                "initialization": self.initialization,
                "forward_calls": self.forward_calls,
                "input_shape": _shape(hidden_states),
                "output_shape": _shape(output),
                "dtype": str(getattr(output, "dtype", None)),
                "device": str(getattr(output, "device", None)),
                "parameter_compression_ratio": self.parameter_compression_ratio,
            }
            if self.debug:
                finite = torch.isfinite(output.detach()).all()
                self.last_debug["all_finite"] = bool(finite.item())
            return output

    class Qwen3VLTwoLayerMLP(nn.Module):
        """Shape-preserving bottleneck MLP: Linear -> SiLU -> Linear."""

        strategy_name = "ffn_two_layer_mlp"

        def __init__(
            self,
            original_mlp: Any,
            *,
            bottleneck_dim: int = 1024,
            bias: bool = False,
            debug: bool = False,
        ) -> None:
            super().__init__()
            required = ("gate_proj", "down_proj")
            missing = [name for name in required if not hasattr(original_mlp, name)]
            if missing:
                raise BackendIntegrationError(
                    "Qwen3-VL two-layer FFN received an incompatible MLP; "
                    f"missing {missing}."
                )
            gate_proj = original_mlp.gate_proj
            down_proj = original_mlp.down_proj
            for name, projection in (
                ("gate_proj", gate_proj),
                ("down_proj", down_proj),
            ):
                if not hasattr(projection, "weight"):
                    raise BackendIntegrationError(
                        f"Qwen3-VL two-layer FFN requires {name}.weight."
                    )
            self.hidden_size = int(gate_proj.in_features)
            output_size = int(down_proj.out_features)
            if output_size != self.hidden_size:
                raise BackendIntegrationError(
                    "Qwen3-VL two-layer FFN must preserve the hidden size; "
                    f"got input={self.hidden_size}, output={output_size}."
                )
            self.bottleneck_dim = int(bottleneck_dim)
            if self.bottleneck_dim < 1 or self.bottleneck_dim > self.hidden_size:
                raise ValueError(
                    "two-layer FFN bottleneck_dim must satisfy "
                    f"1 <= bottleneck_dim <= {self.hidden_size}, "
                    f"got {self.bottleneck_dim}."
                )
            self.bias = bool(bias)
            reference_weight = gate_proj.weight
            linear_kwargs = {
                "bias": self.bias,
                "device": reference_weight.device,
                "dtype": reference_weight.dtype,
            }
            self.reduce_proj = nn.Linear(
                self.hidden_size,
                self.bottleneck_dim,
                **linear_kwargs,
            )
            self.expand_proj = nn.Linear(
                self.bottleneck_dim,
                self.hidden_size,
                **linear_kwargs,
            )
            self.act_fn = torch.nn.functional.silu
            self.original_parameter_count = sum(
                int(parameter.numel()) for parameter in original_mlp.parameters()
            )
            self.debug = bool(debug)
            self.forward_calls = 0
            self.last_debug: dict[str, Any] = {}
            self.train(bool(getattr(original_mlp, "training", False)))

        @property
        def parameter_compression_ratio(self) -> float:
            replacement_parameter_count = sum(
                int(parameter.numel()) for parameter in self.parameters()
            )
            if self.original_parameter_count == 0:
                return 0.0
            return replacement_parameter_count / self.original_parameter_count

        def forward(self, hidden_states: Any) -> Any:
            reduced = self.reduce_proj(hidden_states)
            activated = self.act_fn(reduced)
            output = self.expand_proj(activated)
            self.forward_calls += 1
            self.last_debug = {
                "strategy": self.strategy_name,
                "bottleneck_dim": self.bottleneck_dim,
                "bias": self.bias,
                "forward_calls": self.forward_calls,
                "input_shape": _shape(hidden_states),
                "reduced_shape": _shape(reduced),
                "output_shape": _shape(output),
                "dtype": str(getattr(output, "dtype", None)),
                "device": str(getattr(output, "device", None)),
                "parameter_compression_ratio": self.parameter_compression_ratio,
            }
            if self.debug:
                finite = torch.isfinite(output.detach()).all()
                self.last_debug["all_finite"] = bool(finite.item())
            return output

    class Qwen3VLIdentityMLP(nn.Module):
        """Identity FFN replacement that preserves the module input exactly."""

        strategy_name = "ffn_identity"

        def __init__(self, original_mlp: Any, *, debug: bool = False) -> None:
            super().__init__()
            self.debug = bool(debug)
            self.forward_calls = 0
            self.last_debug: dict[str, Any] = {}
            self.train(bool(getattr(original_mlp, "training", False)))

        def forward(self, hidden_states: Any) -> Any:
            output = hidden_states
            self.forward_calls += 1
            self.last_debug = {
                "strategy": self.strategy_name,
                "forward_calls": self.forward_calls,
                "input_shape": _shape(hidden_states),
                "output_shape": _shape(output),
                "dtype": str(getattr(output, "dtype", None)),
                "device": str(getattr(output, "device", None)),
            }
            if self.debug:
                finite = torch.isfinite(output.detach()).all()
                self.last_debug["all_finite"] = bool(finite.item())
            return output

else:

    class Qwen3VLLowRankLinear:  # pragma: no cover - import-only fallback.
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLLowRankMLP:  # pragma: no cover - import-only fallback.
        strategy_name = "ffn_low_rank"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLTwoLayerMLP:  # pragma: no cover - import-only fallback.
        strategy_name = "ffn_two_layer_mlp"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()

    class Qwen3VLIdentityMLP:  # pragma: no cover - import-only fallback.
        strategy_name = "ffn_identity"

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise _missing_torch()


__all__ = [
    "Qwen3VLIdentityMLP",
    "Qwen3VLLowRankLinear",
    "Qwen3VLLowRankMLP",
    "Qwen3VLTwoLayerMLP",
]
