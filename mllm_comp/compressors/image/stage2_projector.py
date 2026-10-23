"""Trainable Stage-2 projectors for fixed feature-anchor clusters.

The module is deliberately independent of the Qwen provider bridge.  It takes
one post-merger cluster at a time and returns one LLM-width token.  Every
output branch is zero-initialised so attaching an untrained projector is
exactly equivalent to the Stage-1 FP32 cluster mean.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Stage2ProjectorSpec:
    hidden_size: int
    variant: str = "meta_residual"
    bottleneck_dim: int = 64

    def __post_init__(self) -> None:
        if self.variant not in {"mean_only", "meta", "dispersion", "meta_residual"}:
            raise ValueError(
                "variant must be mean_only, meta, dispersion, or meta_residual"
            )
        if isinstance(self.hidden_size, bool) or self.hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer")
        if isinstance(self.bottleneck_dim, bool) or self.bottleneck_dim < 1:
            raise ValueError("bottleneck_dim must be a positive integer")


def _zero_linear(layer: Any) -> None:
    import torch

    torch.nn.init.zeros_(layer.weight)
    if layer.bias is not None:
        torch.nn.init.zeros_(layer.bias)


def create_stage2_projector(spec: Stage2ProjectorSpec) -> Any:
    """Create the torch module lazily so CPU-only metadata imports stay cheap."""

    import torch

    class Stage2VisualProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            width = spec.bottleneck_dim
            hidden = spec.hidden_size
            self.spec = spec
            self.mean_branch = torch.nn.Sequential(
                torch.nn.Linear(hidden, width),
                torch.nn.GELU(),
                torch.nn.Linear(width, hidden),
            )
            _zero_linear(self.mean_branch[-1])
            self.meta_branch = None
            if spec.variant in {"meta", "meta_residual"}:
                self.meta_branch = torch.nn.Sequential(
                    torch.nn.Linear(5, width),
                    torch.nn.GELU(),
                    torch.nn.Linear(width, hidden),
                )
                _zero_linear(self.meta_branch[-1])
            self.dispersion_branch = None
            if spec.variant == "dispersion":
                self.dispersion_branch = torch.nn.Sequential(
                    torch.nn.Linear(hidden, width),
                    torch.nn.GELU(),
                    torch.nn.Linear(width, hidden),
                )
                _zero_linear(self.dispersion_branch[-1])
            self.residual_down = None
            self.residual_up = None
            if spec.variant == "meta_residual":
                self.residual_down = torch.nn.Linear(hidden, width)
                self.residual_up = torch.nn.Linear(width, hidden)
                _zero_linear(self.residual_up)

        def forward(
            self,
            members: Any,
            *,
            rows: Any,
            cols: Any,
            grid_height: int,
            grid_width: int,
            source_count: int,
        ) -> Any:
            if members.ndim != 2 or int(members.shape[-1]) != spec.hidden_size:
                raise ValueError("cluster members do not match projector hidden_size")
            # The statistical surface is FP32 even under BF16 training.  The
            # tiny trainable module follows its registered parameter dtype.
            mean_fp32 = members.float().mean(dim=0)
            parameter = next(self.parameters())
            mean = mean_fp32.to(device=parameter.device, dtype=parameter.dtype)
            correction = self.mean_branch(mean)
            if self.meta_branch is not None:
                row = rows.float()
                col = cols.float()
                denom_h = float(max(1, grid_height - 1))
                denom_w = float(max(1, grid_width - 1))
                mass = torch.log(
                    torch.tensor(
                        float(members.shape[0]), device=row.device, dtype=torch.float32
                    )
                ) / max(1.0, __import__("math").log(float(max(2, source_count))))
                meta = torch.stack(
                    (
                        mass,
                        row.mean() / denom_h,
                        col.mean() / denom_w,
                        row.std(unbiased=False) / denom_h,
                        col.std(unbiased=False) / denom_w,
                    )
                ).to(device=parameter.device, dtype=parameter.dtype)
                correction = correction + self.meta_branch(meta)
            if self.dispersion_branch is not None:
                std = members.float().std(dim=0, unbiased=False).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                correction = correction + self.dispersion_branch(std)
            if self.residual_down is not None and self.residual_up is not None:
                residual = (members.float() - mean_fp32).to(
                    device=parameter.device, dtype=parameter.dtype
                )
                pooled = torch.nn.functional.gelu(self.residual_down(residual)).mean(0)
                correction = correction + self.residual_up(pooled)
            return (mean + correction).to(dtype=members.dtype, device=members.device)

        def config_dict(self) -> dict[str, Any]:
            return asdict(self.spec)

    return Stage2VisualProjector()


def save_stage2_projector(module: Any, path: str | Path) -> None:
    from safetensors.torch import save_file

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    state = {name: value.detach().cpu().contiguous() for name, value in module.state_dict().items()}
    save_file(state, str(destination), metadata={"format": "mllm_comp_stage2_projector_v1"})


def load_stage2_projector(module: Any, path: str | Path) -> None:
    from safetensors.torch import load_file

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Stage-2 projector checkpoint does not exist: {source}")
    module.load_state_dict(load_file(str(source), device="cpu"), strict=True)


__all__ = [
    "Stage2ProjectorSpec",
    "create_stage2_projector",
    "load_stage2_projector",
    "save_stage2_projector",
]
