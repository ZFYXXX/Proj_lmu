"""Frozen local LLaVA-OneVision capability provider for DyToK detection."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.compressors.video.positions import video_detection_layers


class FrozenDyToKDetector:
    """Capture configured attention modules of a separate frozen detector.

    The detector is external capability state: it is never registered as a
    child of the trainable base backbone and never enters compression
    checkpoints.  Loading accepts local directories only and overrides the
    detector's vision-tower path with the explicitly supplied local SigLIP
    directory, preventing an implicit hub lookup.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        siglip_path: str | Path,
        target_layers: tuple[int, ...] = tuple(range(16, 24)),
        device_map: Any = "cpu",
        torch_dtype: Any = "bfloat16",
        source_loader: Any | None = None,
        owner_model: Any | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.siglip_path = Path(siglip_path)
        self.target_layers = video_detection_layers(None, target_layers)
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self._source_loader = source_loader
        self._owner_model = owner_model
        self._artifacts: Any | None = None
        self._api: Any | None = None
        self._calls = 0

    @property
    def prepared(self) -> bool:
        return self._artifacts is not None

    def prepare(self) -> None:
        if self.prepared:
            return
        if not self.checkpoint_path.is_dir():
            raise BackendIntegrationError(
                "DyToK requires runtime.metadata['dytok_detection_model_path'] "
                "naming an existing local LLaVA-OneVision detector directory; "
                "downloads are forbidden"
            )
        if not self.siglip_path.is_dir():
            raise BackendIntegrationError(
                "DyToK requires runtime.metadata['dytok_detection_siglip_path'] "
                "naming an existing local SigLIP directory; downloads are forbidden"
            )
        from mllm_comp.backbones.llava_onevision.implementation import (
            LlavaOneVisionSourceLoader,
        )

        loader = self._source_loader or LlavaOneVisionSourceLoader()
        artifacts = loader.load(
            str(self.checkpoint_path),
            multimodal=True,
            attn_implementation="sdpa",
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            overwrite_config={
                "mm_spatial_pool_stride": 2,
                "mm_spatial_pool_mode": "bilinear",
                "mm_vision_tower": str(self.siglip_path),
            },
        )
        if artifacts.model is self._owner_model:
            raise BackendIntegrationError(
                "DyToK detection must use a separate frozen model instance"
            )
        model = artifacts.model
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _resolve_components,
        )

        layers = tuple(_resolve_components(model).core.layers)
        try:
            video_detection_layers(None, self.target_layers, num_layers=len(layers))
        except ValueError as exc:
            raise BackendIntegrationError(str(exc)) from exc
        self._artifacts = artifacts
        self._api = loader.api()

    def configure_target_layers(self, target_layers: tuple[int, ...]) -> None:
        """Apply requested capture modules, including to an injected prepared detector."""
        depth = None
        if self.prepared:
            from mllm_comp.backbones.llava_onevision.provider_bridge import _resolve_components
            depth = len(_resolve_components(self._artifacts.model).core.layers)
        self.target_layers = video_detection_layers(None, target_layers, num_layers=depth)

    def frame_importance(self, text: str, frames: Any) -> Any:
        """Return one normalized importance value per detector frame."""

        torch = _torch()
        if not isinstance(text, str) or not text.strip():
            raise BackendIntegrationError(
                "DyToK detection requires prompt text without the assistant answer"
            )
        if not isinstance(frames, torch.Tensor) or frames.ndim != 4:
            raise BackendIntegrationError(
                "DyToK detection requires one rank-four frame tensor"
            )
        if int(frames.shape[0]) < 1 or int(frames.shape[1]) != 3:
            raise BackendIntegrationError(
                "DyToK detector frames must be non-empty RGB tensors"
            )
        if not frames.is_floating_point() or not bool(
            torch.isfinite(frames).all().item()
        ):
            raise ValueError("DyToK detector frames must be finite floating tensors")
        self.prepare()
        assert self._artifacts is not None and self._api is not None
        model = self._artifacts.model
        parameter = next(model.parameters())
        device = parameter.device
        dtype = parameter.dtype
        detector_frames = frames.to(device=device, dtype=dtype)
        input_ids = self._detector_input_ids(text).to(device=device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        prepare = getattr(model, "prepare_inputs_labels_for_multimodal", None)
        if not callable(prepare):
            raise BackendIntegrationError(
                "DyToK detector lacks multimodal input preparation"
            )
        height, width = (int(detector_frames.shape[-2]), int(detector_frames.shape[-1]))
        with torch.no_grad():
            prepared = prepare(
                input_ids=input_ids,
                position_ids=None,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=[detector_frames],
                modalities=("video",),
                image_sizes=[(width, height)],
            )
            result = self._frame_importance_from_prepared(
                prepared,
                input_ids=input_ids,
                frame_count=int(detector_frames.shape[0]),
            )
        self._calls += 1
        return result

    def runtime_metadata(self) -> dict[str, Any]:
        model = self._artifacts.model if self._artifacts is not None else None
        frozen = None
        if model is not None:
            frozen = all(
                not parameter.requires_grad for parameter in model.parameters()
            )
        return {
            "provider": "frozen_llava_onevision_dytok_detector",
            "checkpoint_path": str(self.checkpoint_path),
            "siglip_path": str(self.siglip_path),
            "target_layers": self.target_layers,
            "prepared": self.prepared,
            "frozen": frozen,
            "calls": self._calls,
            "local_files_only": True,
            "assistant_answer_excluded": True,
            "attention_query": "last_prompt_token",
            "attention_key_normalization": "visual_keys_only",
        }

    def _detector_input_ids(self, text: str) -> Any:
        from mllm_comp.backbones.llava_onevision.implementation import (
            DEFAULT_CONVERSATION_MODE,
            _render_prompt,
        )

        assert self._artifacts is not None and self._api is not None
        prompt = _render_prompt(
            text.strip(),
            media_count=1,
            conversation_mode=DEFAULT_CONVERSATION_MODE,
            api=self._api,
            model_config=self._artifacts.model.config,
            assistant_text=None,
        )
        ids = self._api.tokenizer_image_token(
            prompt,
            self._artifacts.tokenizer,
            self._api.image_token_index,
            return_tensors="pt",
        )
        if getattr(ids, "ndim", None) == 1:
            ids = ids.unsqueeze(0)
        if getattr(ids, "ndim", None) != 2 or int(ids.shape[0]) != 1:
            raise BackendIntegrationError(
                "DyToK detector tokenizer did not return batch-one input IDs"
            )
        placeholders = (ids[0] == self._api.image_token_index).nonzero(as_tuple=False)
        if int(placeholders.numel()) != 1:
            raise BackendIntegrationError(
                "DyToK detector prompt must contain exactly one visual placeholder"
            )
        return ids

    def _frame_importance_from_prepared(
        self,
        prepared: Any,
        *,
        input_ids: Any,
        frame_count: int,
    ) -> Any:
        torch = _torch()
        if not isinstance(prepared, tuple) or len(prepared) != 6:
            raise BackendIntegrationError(
                "DyToK detector multimodal preparation returned an unknown structure"
            )
        _ids, position_ids, padding_mask, past, inputs_embeds, labels = prepared
        if _ids is not None or past is not None or labels is not None:
            raise BackendIntegrationError(
                "DyToK detector requires the non-cached embedding prefill path"
            )
        if getattr(inputs_embeds, "ndim", None) != 3 or tuple(
            inputs_embeds.shape[:1]
        ) != (1,):
            raise BackendIntegrationError(
                "DyToK detector did not produce batch-one embeddings"
            )
        from mllm_comp.backbones.llava_onevision.provider_bridge import (
            _decoder_masks,
            _decoder_query_keys,
            _resolve_components,
            _transformers_decoder_helpers,
        )
        from mllm_comp.backbones.llava_onevision.video_compression import (
            LlavaOneVisionVideoPacking,
            _pooled_grid_shape,
        )
        from mllm_comp.compressors.video.dytok import (
            aggregate_dytok_frame_importance,
            dytok_frame_attention_from_qk,
        )

        model = self._artifacts.model
        core = _resolve_components(model).core
        grid_height, grid_width = _pooled_grid_shape(model, model.config)
        packing = LlavaOneVisionVideoPacking.from_native(
            frame_count=frame_count,
            grid_height=grid_height,
            grid_width=grid_width,
            patch_merge_type=str(getattr(model.config, "mm_patch_merge_type", "flat")),
            newline_position=str(
                getattr(model.config, "mm_newline_position", "one_token")
            ),
        )
        placeholder = int(
            (input_ids[0] == self._api.image_token_index)
            .nonzero(as_tuple=False)
            .flatten()[0]
            .item()
        )
        sequence_length = int(inputs_embeds.shape[1])
        expected = int(input_ids.shape[1]) - 1 + packing.packed_token_count
        if sequence_length != expected:
            raise BackendIntegrationError(
                "DyToK detector visual packing differs from the pinned LLaVA contract"
            )
        visual_positions = tuple(
            placeholder + position for position in packing.patch_positions
        )
        if len(visual_positions) != frame_count * grid_height * grid_width:
            raise InvariantError("DyToK detector patch-to-frame layout changed")
        frame_token_counts = tuple(grid_height * grid_width for _ in range(frame_count))
        visual_index = torch.tensor(
            visual_positions,
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        if padding_mask is None:
            padding_mask = torch.ones(
                1,
                sequence_length,
                dtype=torch.bool,
                device=inputs_embeds.device,
            )
        if tuple(padding_mask.shape) != (1, sequence_length):
            raise BackendIntegrationError(
                "DyToK detector requires a batch-one padding mask"
            )
        active = padding_mask[0].bool().nonzero(as_tuple=False).flatten()
        query_position = int(active[-1].item())
        if query_position < placeholder + packing.packed_token_count:
            raise BackendIntegrationError(
                "DyToK detector requires a post-visual prompt query"
            )
        query_index = torch.tensor(
            (query_position,),
            dtype=torch.long,
            device=inputs_embeds.device,
        )
        cache_position = torch.arange(sequence_length, device=inputs_embeds.device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if tuple(position_ids.shape) != (1, sequence_length):
            raise BackendIntegrationError(
                "DyToK detector requires rank-two native positions"
            )
        (
            _base_output,
            create_causal_mask,
            create_sliding_window_causal_mask,
        ) = _transformers_decoder_helpers()
        hidden_states = inputs_embeds
        masks = _decoder_masks(
            core,
            hidden_states,
            padding_mask,
            position_ids,
            cache_position,
            create_causal_mask,
            create_sliding_window_causal_mask,
        )
        position_embeddings = core.rotary_emb(hidden_states, position_ids)
        target_set = set(self.target_layers)
        collected = []
        for layer_index, layer in enumerate(tuple(core.layers)):
            if layer_index in target_set:
                query, visual_keys, _all_keys = _decoder_query_keys(
                    layer,
                    hidden_states,
                    instruction_indices=query_index,
                    image_indices=visual_index,
                    position_embeddings=position_embeddings,
                )
                collected.append(
                    dytok_frame_attention_from_qk(
                        query,
                        visual_keys,
                        frame_token_counts=frame_token_counts,
                    )
                )
                if layer_index == self.target_layers[-1]:
                    break
            layer_mask = masks[getattr(layer, "attention_type", "full_attention")]
            output = layer(
                hidden_states,
                attention_mask=layer_mask,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            hidden_states = output[0] if isinstance(output, tuple) else output
        if len(collected) != len(self.target_layers):
            raise InvariantError("DyToK detector missed a target attention layer")
        return aggregate_dytok_frame_importance(collected)


def _torch() -> Any:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - existing environment owns this
        raise BackendIntegrationError(
            "DyToK detector requires the existing PyTorch environment"
        ) from exc
    return torch


__all__ = ["FrozenDyToKDetector"]
