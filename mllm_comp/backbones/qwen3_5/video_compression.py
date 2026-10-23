"""Qwen3.5/Qwen3.8 VQToken video projector-boundary compression."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.layout import layout_delta_record


class Qwen3_5VideoVQTokenActivation:
    """Apply the canonical per-video VQToken kernel before the LLM scatter.

    The activation is intentionally limited to one pure-video sample. It
    precomputes native Qwen3.5 video features, clusters each video's projected
    stream, and compacts the video placeholders and M-RoPE positions in lockstep.
    """

    algorithm = "vqtoken"
    binding_name = "vqtoken.video.qwen3_5"

    def __init__(
        self,
        *,
        model: Any,
        runtime: Any = None,
        facade: Any = None,
        stage: Any = None,
        binding: Any,
        binding_config: Any = None,
        execution_phase: Any = None,
        contract_projection: Any = None,
    ) -> None:
        del runtime, facade, execution_phase, contract_projection
        if getattr(binding, "name", None) != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 VQToken received a foreign binding: "
                f"{getattr(binding, 'name', None)!r}."
            )
        self.model = model
        self.binding = binding
        merged: dict[str, Any] = {}
        for source in (
            getattr(stage, "operation", None),
            getattr(stage, "criterion", None),
            getattr(stage, "schedule", None),
            getattr(stage, "config", None),
            getattr(binding_config, "config", None),
        ):
            if isinstance(source, Mapping):
                merged.update(source)
        self.config = {
            "num_clusters": int(merged.get("num_clusters", 32)),
            "adaptive": bool(merged.get("adaptive", False)),
            "use_cross_attention": bool(merged.get("use_cross_attention", False)),
        }
        if self.config != {
            "num_clusters": 32,
            "adaptive": False,
            "use_cross_attention": False,
        }:
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken uses the canonical fixed-K=32 recipe."
            )
        self._installed = False
        self._had_generate = "generate" in getattr(model, "__dict__", {})
        self._original_generate = model.generate
        self._request_records: list[dict[str, Any]] = []
        self._layout_records: list[dict[str, Any]] = []

    def install(self) -> None:
        if self._installed:
            return
        import types

        owner = self

        def generate_hook(_model: Any, *args: Any, **kwargs: Any) -> Any:
            if args or kwargs.get("pixel_values_videos") is None:
                return owner._original_generate(*args, **kwargs)
            original_ids = kwargs.get("input_ids")
            if original_ids is None:
                return owner._original_generate(*args, **kwargs)
            previous_rope = getattr(owner.model.model, "rope_deltas", None)
            prepared, record = owner._compress_inputs(kwargs)
            try:
                raw = owner._original_generate(**prepared)
            finally:
                owner.model.model.rope_deltas = previous_rope
            sequences = getattr(raw, "sequences", raw)
            compact_width = int(prepared["input_ids"].shape[1])
            import torch

            restored = torch.cat(
                (original_ids, sequences[:, compact_width:]), dim=-1
            )
            if hasattr(raw, "sequences"):
                raw.sequences = restored
            else:
                raw = restored
            record["generated"] = True
            owner._request_records.append(record)
            return raw

        self.model.generate = types.MethodType(generate_hook, self.model)
        self._installed = True

    def uninstall(self) -> None:
        if not self._installed:
            return
        if self._had_generate:
            self.model.generate = self._original_generate
        else:
            delattr(self.model, "generate")
        self._installed = False

    def clear_diagnostic_records(self) -> None:
        self._request_records.clear()
        self._layout_records.clear()

    def _compress_inputs(
        self, source: Mapping[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch
        from transformers.models.qwen3_vl.modeling_qwen3_vl import (
            BaseModelOutputWithDeepstackFeatures,
        )

        from mllm_comp.backbones.qwen3_vl.video_compression import Qwen3VLVideoGrid
        from mllm_comp.compressors.video import VQTokenVideoKernel

        kwargs = dict(source)
        input_ids = kwargs.get("input_ids")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        video_grid_thw = kwargs.get("video_grid_thw")
        mm_types = kwargs.get("mm_token_type_ids")
        if input_ids is None or pixel_values_videos is None or video_grid_thw is None or mm_types is None:
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken requires input_ids, pixel_values_videos, "
                "video_grid_thw, and mm_token_type_ids."
            )
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 video VQToken requires batch_size=1.")
        if video_grid_thw.ndim != 2 or video_grid_thw.shape[0] != 1:
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken currently supports one video per sample."
            )
        if kwargs.get("pixel_values") is not None or kwargs.get("image_grid_thw") is not None:
            raise BackendIntegrationError(
                "Qwen3.5 VQToken currently requires video-only visual input; "
                "mixed image/video prompts are not enabled."
            )
        deepstack_indexes = getattr(
            getattr(self.model.config, "vision_config", None),
            "deepstack_visual_indexes",
            (),
        )
        if deepstack_indexes:
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken currently expects an empty DeepStack configuration."
            )

        video_outputs = self.model.get_video_features(
            pixel_values_videos,
            video_grid_thw,
            return_dict=True,
        )
        feature_groups = video_outputs.pooler_output
        if torch.is_tensor(feature_groups):
            feature_groups = (feature_groups,)
        feature_groups = tuple(feature_groups)
        if len(feature_groups) != 1:
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken expected one post-merger feature stream."
            )

        video_token_id = getattr(self.model.config, "video_token_id", None)
        if video_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config does not expose video_token_id.")
        video_positions = torch.nonzero(
            input_ids[0] == int(video_token_id), as_tuple=False
        ).flatten()
        source_count = int(feature_groups[0].shape[0])
        if int(video_positions.numel()) != source_count:
            raise BackendIntegrationError(
                f"Video placeholder/features mismatch: {int(video_positions.numel())} "
                f"placeholders for {source_count} merged features."
            )

        spatial_merge_size = int(self.model.config.vision_config.spatial_merge_size)
        grid = Qwen3VLVideoGrid.from_native(
            video_grid_thw[0].tolist(), spatial_merge_size=spatial_merge_size
        )
        if grid.token_count != source_count:
            raise BackendIntegrationError(
                f"Qwen3.5 video grid implies {grid.token_count} tokens, "
                f"but the native merger returned {source_count}."
            )
        payload = grid.payload(feature_groups[0], media_id="qwen3_5:video:0")
        result = VQTokenVideoKernel()(payload, config=self.config)
        representatives = torch.tensor(
            result.representative_indices[0],
            dtype=torch.long,
            device=video_positions.device,
        )
        compressed_features = result.features
        if int(compressed_features.shape[0]) != int(representatives.numel()):
            raise BackendIntegrationError(
                "Qwen3.5 video VQToken output and representative anchors disagree."
            )

        original_positions, original_delta = self.model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_types,
            video_grid_thw=video_grid_thw,
            attention_mask=kwargs.get("attention_mask"),
        )
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep[video_positions] = False
        kept_video_positions = video_positions.index_select(0, representatives)
        keep[kept_video_positions] = True
        source_mask = kwargs.get("attention_mask")
        if source_mask is None:
            source_mask = torch.ones_like(input_ids)
        removed_valid_tokens = int((~keep & source_mask[0].bool()).sum().item())

        compressed_ids = input_ids[:, keep]
        compressed_mask = source_mask[:, keep]
        compressed_types = mm_types[:, keep]
        compressed_positions = original_positions[:, :, keep]
        self.model.model.rope_deltas = original_delta + removed_valid_tokens
        kwargs["input_ids"] = compressed_ids
        kwargs["attention_mask"] = compressed_mask
        kwargs["mm_token_type_ids"] = compressed_types
        kwargs["position_ids"] = compressed_positions
        kwargs["mm_encoder_outputs"] = {
            "video": BaseModelOutputWithDeepstackFeatures(
                pooler_output=(compressed_features,), deepstack_features=[]
            )
        }
        for name in ("pixel_values_videos", "video_grid_thw", "video_metadata", "second_per_grid_ts"):
            kwargs.pop(name, None)

        self._layout_records.extend(
            layout_delta_record(delta) for delta in result.layout_deltas
        )
        record = {
            "algorithm": self.algorithm,
            "placement": "post_merger_pre_llm_scatter",
            "input_prompt_tokens": int(input_ids.shape[1]),
            "output_prompt_tokens": int(compressed_ids.shape[1]),
            "video_tokens_before": source_count,
            "video_tokens_after": int(compressed_features.shape[0]),
            "video_grid_thw": [int(value) for value in video_grid_thw[0].tolist()],
            "mrope_positions_synchronized": True,
            "mm_encoder_outputs_used": True,
            "layout_deltas": [layout_delta_record(delta) for delta in result.layout_deltas],
        }
        return kwargs, record

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": "vqtoken.video",
            "binding": self.binding_name,
            "target_site": "post_merger_pre_llm_scatter",
            "num_clusters": int(self.config["num_clusters"]),
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "video_support": True,
        }


def create_qwen3_5_video_compression_executor(**kwargs: Any) -> Qwen3_5VideoVQTokenActivation:
    return Qwen3_5VideoVQTokenActivation(**kwargs)
