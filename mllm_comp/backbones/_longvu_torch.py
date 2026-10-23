"""Lazy PyTorch implementation of LongVU's trainable SVA and frozen DINO."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from mllm_comp.core.errors import BackendIntegrationError


class _MultiKVCrossAttention(nn.Module):
    def __init__(self, hidden_size: int, *, num_heads: int = 16) -> None:
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("LongVU hidden size must be divisible by 16 heads")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.query = nn.Sequential(
            nn.LayerNorm(hidden_size), nn.Linear(hidden_size, hidden_size, bias=False)
        )
        self.keys = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
            for _ in range(2)
        )
        self.values = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size, bias=False),
            )
            for _ in range(2)
        )
        self.output = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, queries: Any, first: Any, second: Any) -> Any:
        batch, query_count, _ = queries.shape
        query = self.query(queries)
        keys = torch.cat((self.keys[0](first), self.keys[1](second)), dim=1)
        values = torch.cat((self.values[0](first), self.values[1](second)), dim=1)
        key_count = int(keys.shape[1])
        query = query.view(
            batch, query_count, self.num_heads, self.head_size
        ).transpose(1, 2)
        keys = keys.view(batch, key_count, self.num_heads, self.head_size).transpose(
            1, 2
        )
        values = values.view(
            batch, key_count, self.num_heads, self.head_size
        ).transpose(1, 2)
        attended = torch.nn.functional.scaled_dot_product_attention(query, keys, values)
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .reshape(batch, query_count, self.hidden_size)
        )
        return self.output(attended)


class _SVALayer(nn.Module):
    def __init__(self, hidden_size: int, local_tokens: int) -> None:
        super().__init__()
        self.context_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.input_projection = nn.Linear(hidden_size * 2, hidden_size, bias=False)
        self.raw_position = nn.Parameter(torch.randn(local_tokens, hidden_size))
        self.dino_position = nn.Parameter(torch.randn(local_tokens, hidden_size))
        self.attention = _MultiKVCrossAttention(hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=False),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size, bias=False),
        )

    def forward(self, queries: Any, context: Any, raw: Any, dino: Any) -> Any:
        residual = queries
        combined = torch.cat((queries, self.context_projection(context)), dim=-1)
        hidden = self.input_projection(combined)
        raw = raw + self.raw_position[None, :, :].to(raw.dtype)
        dino = dino + self.dino_position[None, :, :].to(dino.dtype)
        hidden = self.norm(hidden + self.attention(hidden, raw, dino))
        return residual + self.mlp(hidden)


class LongVUFusionModule(nn.Module):
    """Reference-initialized three-layer joint SVA registered before optimizer."""

    supported_frame_selection_sites = ("pre_sva", "pre_output_projector", "post_output_projector")

    def __init__(
        self,
        *,
        raw_hidden_size: int,
        dino_hidden_size: int,
        output_hidden_size: int,
        vision_hidden_size: int = 1024,
        dino_interp_size: int = 576,
        query_num: int = 144,
        connector_depth: int = 3,
    ) -> None:
        super().__init__()
        interp_side = math.isqrt(dino_interp_size)
        query_side = math.isqrt(query_num)
        if (
            interp_side * interp_side != dino_interp_size
            or query_side * query_side != query_num
            or interp_side % query_side
        ):
            raise ValueError("LongVU interpolation/query sizes are incompatible")
        local_side = interp_side // query_side
        self.interp_side = interp_side
        self.query_side = query_side
        self.query_num = query_num
        self.local_tokens = local_side * local_side
        self.raw_projector = nn.Sequential(
            nn.Linear(raw_hidden_size, vision_hidden_size),
            nn.GELU(),
            nn.Linear(vision_hidden_size, vision_hidden_size),
            nn.LayerNorm(vision_hidden_size),
        )
        self.dino_projector = nn.Sequential(
            nn.Linear(dino_hidden_size, vision_hidden_size),
            nn.GELU(),
            nn.Linear(vision_hidden_size, vision_hidden_size),
            nn.LayerNorm(vision_hidden_size),
        )
        self.layers = nn.ModuleList(
            _SVALayer(vision_hidden_size, self.local_tokens)
            for _ in range(connector_depth)
        )
        standard_deviation = 1.0 / math.sqrt(vision_hidden_size)
        self.vision_query = nn.Parameter(
            torch.randn(1, vision_hidden_size) * standard_deviation
        )
        self.output_projector = nn.Sequential(
            nn.Linear(vision_hidden_size, output_hidden_size),
            nn.GELU(),
            nn.Linear(output_hidden_size, output_hidden_size),
        )

    def forward(
        self, raw_features: Any, dino_features: Any, *,
        selected_frame_indices: Any | None = None,
        frame_selection_site: str = "pre_sva",
    ) -> Any:
        if frame_selection_site not in self.supported_frame_selection_sites:
            raise BackendIntegrationError(f"Unsupported LongVU frame selection site: {frame_selection_site!r}")
        if frame_selection_site != "pre_sva" and selected_frame_indices is None:
            raise BackendIntegrationError("Moved LongVU frame selection requires selected frame indices")
        if raw_features.ndim != 3 or dino_features.ndim != 3:
            raise BackendIntegrationError(
                "LongVU SVA inputs must be rank-three frame rasters"
            )
        expected = self.interp_side * self.interp_side
        if (
            int(raw_features.shape[0]) != int(dino_features.shape[0])
            or int(raw_features.shape[1]) != expected
            or int(dino_features.shape[1]) != expected
        ):
            raise BackendIntegrationError(
                "LongVU SVA raw and DINO rasters are not spatially aligned"
            )
        selected = None if selected_frame_indices is None else torch.as_tensor(
            selected_frame_indices, device=raw_features.device, dtype=torch.long,
        )
        if frame_selection_site == "pre_sva" and selected is not None:
            raw_features = raw_features.index_select(0, selected)
            dino_features = dino_features.index_select(0, selected)
        raw = self.raw_projector(raw_features)
        dino = self.dino_projector(dino_features)
        context = raw.mean(dim=1)
        raw = self._query_cells(raw)
        dino = self._query_cells(dino)
        frame_count = int(raw_features.shape[0])
        queries = self.vision_query.view(1, 1, -1).expand(
            frame_count * self.query_num, 1, -1
        )
        context = (
            context[:, None, None, :]
            .expand(-1, self.query_num, 1, -1)
            .reshape(frame_count * self.query_num, 1, -1)
        )
        for layer in self.layers:
            queries = layer(queries, context, raw, dino)
        # SVA always retains its learned vision width. Only the frame axis
        # crosses the original nonlinear output projector boundary.
        queries = queries.squeeze(1)
        if frame_selection_site == "pre_output_projector":
            queries = queries.reshape(frame_count, self.query_num, -1).index_select(0, selected)
            frame_count = int(queries.shape[0])
            queries = queries.reshape(frame_count * self.query_num, -1)
        output = self.output_projector(queries).reshape(frame_count, self.query_num, -1)
        if frame_selection_site == "post_output_projector":
            output = output.index_select(0, selected)
        return output

    def _query_cells(self, features: Any) -> Any:
        frame_count, _tokens, hidden = features.shape
        local_side = self.interp_side // self.query_side
        return (
            features.reshape(
                frame_count,
                self.query_side,
                local_side,
                self.query_side,
                local_side,
                hidden,
            )
            .permute(0, 1, 3, 2, 4, 5)
            .reshape(frame_count * self.query_num, self.local_tokens, hidden)
        )


class FrozenDinov2Encoder:
    """Local-only frozen DINOv2 feature provider with 24x24 interpolation."""

    def __init__(self, source: str | Path, *, interp_size: int = 576) -> None:
        path = Path(source)
        if not path.is_dir():
            raise BackendIntegrationError(
                f"LongVU DINOv2 path is not a readable local directory: {path}"
            )
        try:
            from transformers import Dinov2Model
        except ImportError as exc:
            raise BackendIntegrationError(
                "LongVU requires the existing Transformers DINOv2 implementation"
            ) from exc
        try:
            model = Dinov2Model.from_pretrained(path, local_files_only=True)
        except Exception as exc:
            raise BackendIntegrationError(
                f"LongVU could not load the local DINOv2 checkpoint {path}: {exc}"
            ) from exc
        model.requires_grad_(False)
        model.eval()
        self.source = path
        self.model = model
        self.interp_size = interp_size
        self.hidden_size = int(model.config.hidden_size)

    def __call__(self, pixels: Any) -> Any:
        if pixels.ndim != 4 or int(pixels.shape[1]) != 3:
            raise BackendIntegrationError(
                "LongVU DINOv2 input must be [frames,3,height,width]"
            )
        parameters = tuple(self.model.parameters())
        model_dtype = parameters[0].dtype if parameters else torch.float32
        model_device = parameters[0].device if parameters else pixels.device
        if model_device != pixels.device:
            self.model.to(device=pixels.device)
            model_device = pixels.device
        with torch.no_grad():
            output = self.model(pixels.to(device=model_device, dtype=model_dtype))
            features = output.last_hidden_state[:, 1:].to(pixels.dtype)
            token_count = int(features.shape[1])
            patch_size = getattr(self.model.config, "patch_size", None)
            if (
                isinstance(patch_size, bool)
                or not isinstance(patch_size, int)
                or patch_size < 1
            ):
                raise BackendIntegrationError(
                    "LongVU DINOv2 config does not expose a positive patch size"
                )
            source_height = int(pixels.shape[-2]) // patch_size
            source_width = int(pixels.shape[-1]) // patch_size
            target_side = math.isqrt(self.interp_size)
            if (
                source_height * source_width != token_count
                or target_side * target_side != self.interp_size
            ):
                raise BackendIntegrationError(
                    "LongVU DINOv2 patch output does not match the input raster"
                )
            if (source_height, source_width) != (target_side, target_side):
                features = features.reshape(
                    int(features.shape[0]), source_height, source_width, -1
                ).permute(0, 3, 1, 2)
                features = torch.nn.functional.interpolate(
                    features.float(),
                    size=(target_side, target_side),
                    mode="bilinear",
                    align_corners=False,
                ).to(pixels.dtype)
                features = features.permute(0, 2, 3, 1).reshape(
                    int(features.shape[0]), self.interp_size, -1
                )
        return features


__all__ = ["FrozenDinov2Encoder", "LongVUFusionModule"]
