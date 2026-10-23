"""Frozen local-only SigLIP capability used by canonical METok."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from mllm_comp.core.errors import BackendIntegrationError


class FrozenSigLIPEncoder:
    """Outside-checkpoint SigLIP text or text+vision feature provider.

    LLaVA retains only the text components after loading because its native
    visual tower is already SigLIP.  Qwen retains the full double tower so
    reconstructed temporal tubelets can be embedded in the same feature
    space.  Every load is local-only and every parameter remains frozen.
    """

    def __init__(self, model_path: str | Path, *, include_vision: bool) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_dir():
            raise BackendIntegrationError(
                f"METok SigLIP directory does not exist: {self.model_path}"
            )
        try:
            from transformers import AutoTokenizer, SiglipModel
        except ImportError as exc:
            raise BackendIntegrationError(
                "METok requires SigLIP support in the existing Transformers install"
            ) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
            full_model = SiglipModel.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        except Exception as exc:
            raise BackendIntegrationError(
                "METok could not load the required local SigLIP capability"
            ) from exc

        full_model.requires_grad_(False)
        full_model.eval()
        self.include_vision = bool(include_vision)
        if self.include_vision:
            self._full_model = full_model
            self.text_model = None
            self.text_projection = None
        else:
            self._full_model = None
            self.text_model = full_model.text_model
            self.text_projection = getattr(full_model, "text_projection", None)
            self.text_model.requires_grad_(False)
            self.text_model.eval()
            if self.text_projection is not None:
                self.text_projection.requires_grad_(False)
                self.text_projection.eval()

    @property
    def vision_enabled(self) -> bool:
        return self._full_model is not None

    @torch.no_grad()
    def text_features(
        self,
        text: str,
        *,
        device: Any,
        dtype: Any,
        max_length: int = 64,
    ) -> Any:
        if not isinstance(text, str) or not text.strip():
            raise BackendIntegrationError("METok requires non-empty prompt text")
        owner = self._full_model if self._full_model is not None else self.text_model
        if owner is None:
            raise BackendIntegrationError("METok SigLIP text model is unavailable")
        owner.to(device=device, dtype=dtype)
        if self.text_projection is not None:
            self.text_projection.to(device=device, dtype=dtype)
        encoded = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {
            name: value.to(device=device)
            for name, value in encoded.items()
            if name in {"input_ids", "attention_mask"}
        }
        if self._full_model is not None:
            features = self._full_model.get_text_features(**encoded)
        else:
            output = self.text_model(**encoded)
            features = output.pooler_output
            if self.text_projection is not None:
                features = self.text_projection(features)
        return features.to(dtype=dtype)

    @torch.no_grad()
    def native_frame_similarity(
        self,
        text: str,
        raw_features: Any,
        *,
        frame_count: int,
        tokens_per_frame: int,
    ) -> Any:
        if not isinstance(raw_features, torch.Tensor) or raw_features.ndim != 2:
            raise BackendIntegrationError(
                "METok native SigLIP features must be rank-two"
            )
        if int(raw_features.shape[0]) != frame_count * tokens_per_frame:
            raise BackendIntegrationError(
                "METok native SigLIP raster does not match its frames"
            )
        frame_features = raw_features.reshape(frame_count, tokens_per_frame, -1).mean(
            dim=1
        )
        text_features = self.text_features(
            text,
            device=frame_features.device,
            dtype=frame_features.dtype,
        )
        if int(text_features.shape[-1]) != int(frame_features.shape[-1]):
            raise BackendIntegrationError(
                "METok SigLIP text and native visual dimensions differ"
            )
        return torch.nn.functional.cosine_similarity(
            frame_features,
            text_features.expand(frame_count, -1),
            dim=-1,
        )

    @torch.no_grad()
    def pixel_frame_similarity(self, text: str, pixels: Any) -> Any:
        if self._full_model is None:
            raise BackendIntegrationError(
                "METok Qwen provider requires the frozen SigLIP vision tower"
            )
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise BackendIntegrationError(
                "METok SigLIP pixels must be [frames, channels, height, width]"
            )
        parameter = next(self._full_model.parameters())
        self._full_model.to(device=pixels.device, dtype=pixels.dtype)
        if parameter.device != pixels.device or parameter.dtype != pixels.dtype:
            parameter = next(self._full_model.parameters())
        image_features = self._full_model.get_image_features(pixel_values=pixels)
        text_features = self.text_features(
            text,
            device=image_features.device,
            dtype=image_features.dtype,
        )
        if int(text_features.shape[-1]) != int(image_features.shape[-1]):
            raise BackendIntegrationError(
                "METok SigLIP text and reconstructed visual dimensions differ"
            )
        return torch.nn.functional.cosine_similarity(
            image_features,
            text_features.expand(int(image_features.shape[0]), -1),
            dim=-1,
        ).to(dtype=pixels.dtype)


__all__ = ["FrozenSigLIPEncoder"]
