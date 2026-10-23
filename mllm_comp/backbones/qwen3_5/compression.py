"""Qwen3.5 projector-boundary image compression activations."""

from __future__ import annotations

from collections.abc import Mapping
import math
from typing import Any

from mllm_comp.compressors.image.tome import tome_merge_image
from mllm_comp.compressors.image.divprune import DivPruneSpec, divprune_image
from mllm_comp.compressors.image.aim import AIMSpec, aim_image_tokens
from mllm_comp.compressors.image.restore import RestoreSpec, restore_image_tokens
from mllm_comp.compressors.image.otprune import OTPruneSpec, otprune_image
from mllm_comp.compressors.image.prumerge import PruMergeSpec, prumerge_image_tokens
from mllm_comp.compressors.image.mmtok import MMTokSpec, mmtok_image_tokens
from mllm_comp.compressors.image.gap import GAPSpec, gap_image_tokens
from mllm_comp.compressors.image.vispruner import VisPrunerSpec, vispruner_image_tokens
from mllm_comp.compressors.image.visionzip import visionzip_image
from mllm_comp.compressors.image.bifocal import BifocalSpec, bifocal_image_tokens
from mllm_comp.compressors.image.zoo_prune import ZOOPruneSpec, zoo_prune_image_tokens
from mllm_comp.compressors.image.saint import SAINTSpec, saint_image_tokens
from mllm_comp.compressors.image.nuwa import (
    NUWA_CONFIG_KEYS,
    NuwaSpec,
    nuwa_stage1_image_tokens,
)
from mllm_comp.core.layout import layout_delta_record
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.backbones.qwen3_5.vision_attention import (
    capture_qwen3_5_final_vision_attention,
    capture_qwen3_5_premerger_features,
    capture_qwen3_5_semantic_hidden_range,
    build_qwen3_5_merger_inputs,
    pool_patch_statistics_to_merger_tokens,
    pool_premerger_features_to_merger_tokens,
)


class _Qwen3_5ImageCompressionActivation:
    """Compress merger outputs and compact prompt/M-RoPE metadata atomically.

    The removable runtime hook wraps the model's generate method.
    It precomputes final merged image features, applies the bound method, compacts the
    matching image placeholders and aligned masks/positions, then supplies the
    precomputed compressed features through Transformers' native
    ``mm_encoder_outputs`` input. No Qwen3-VL visual or DeepStack hooks are used.
    """

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
        binding_name = getattr(binding, "name", None)
        if binding_name != self.binding_name:
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} received a foreign binding: {binding_name!r}."
            )
        self.model = model
        self.binding = binding
        merged: dict[str, Any] = {}
        for source in (getattr(stage, "operation", None), getattr(stage, "criterion", None),
                       getattr(stage, "schedule", None), getattr(stage, "config", None),
                       getattr(binding_config, "config", None)):
            if isinstance(source, Mapping):
                merged.update(source)
        self.retain_ratio = float(merged.get("retain_ratio", 0.5))
        self.target_count = merged.get("target_count")
        self.prune_layer = int(merged.get("prune_layer", 0))
        self.distance_chunk_size = int(merged.get("distance_chunk_size", 512))
        self.distance = str(merged.get("distance", "cosine"))
        self.selection = str(merged.get("selection", "official_greedy_maxmin"))
        self.min_visual_tokens = int(merged.get("min_visual_tokens", 1))
        self.attention_chunk_size = int(merged.get("attention_chunk_size", 128))
        self.gamma = float(merged.get("gamma", 0.01))
        self.prumerge_cluster_size = int(merged.get("cluster_size", 32))
        self.mmtok_alpha = float(merged.get("alpha", 0.5))
        self.mmtok_text_temperature = float(merged.get("text_temperature", 0.02))
        self.mmtok_vision_temperature = float(merged.get("vision_temperature", 0.2))
        self.vispruner_important_ratio = float(merged.get("important_ratio", 0.5))
        self.vispruner_removal_step = int(merged.get("removal_step", 8))
        self.restore_pruning_ratio = float(merged.get("pruning_ratio", 0.5))
        self.restore_important_ratio = float(merged.get("important_ratio", 0.5))
        self.restore_removal_step = int(merged.get("removal_step", 8))
        self.zoo_num_directions = int(merged.get("num_directions", 64))
        self.saint_spec = SAINTSpec(
            keep_ratio=self.retain_ratio,
            target_count=self.target_count,
            prune_level=str(merged.get("prune_level", "token")),
            group_size=int(merged.get("group_size", 4)),
            similarity_threshold=float(merged.get("similarity_threshold", 0.7)),
            neighbour_count=int(merged.get("neighbour_count", 5)),
            gamma=float(merged.get("gamma", 10.0)),
            max_iterations=int(merged.get("max_iterations", 32)),
        ) if self.algorithm == "saint" else None
        self.nuwa_spec = NuwaSpec(
            **{key: merged[key] for key in NUWA_CONFIG_KEYS if key in merged}
        ) if self.algorithm == "nuwa" else None
        self.rules = merged.get("rules")
        if not 0.0 < self.retain_ratio <= 1.0:
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} retain_ratio must be in (0, 1]."
            )
        if self.target_count is not None and (
            isinstance(self.target_count, bool)
            or not isinstance(self.target_count, int)
            or self.target_count < 1
        ):
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} target_count must be a positive integer."
            )
        if self.prune_layer != 0:
            raise BackendIntegrationError("Qwen3.5 DivPrune currently requires prune_layer=0.")
        if self.distance_chunk_size < 1:
            raise BackendIntegrationError("Qwen3.5 DivPrune distance_chunk_size must be positive.")
        if self.min_visual_tokens < 1:
            raise BackendIntegrationError("Qwen3.5 DivPrune min_visual_tokens must be positive.")
        if self.attention_chunk_size < 1:
            raise BackendIntegrationError("Qwen3.5 VisionZip attention_chunk_size must be positive.")
        if self.algorithm == "visionzip" and self.target_count is not None and self.target_count < 2:
            raise BackendIntegrationError("Qwen3.5 VisionZip target_count must be at least 2.")
        if self.algorithm == "divprune" and (
            self.distance != "cosine" or self.selection != "official_greedy_maxmin"
        ):
            raise BackendIntegrationError(
                "Qwen3.5 DivPrune requires cosine distance and official_greedy_maxmin selection."
            )
        if self.algorithm == "otprune" and self.gamma < 0:
            raise BackendIntegrationError("Qwen3.5 OTPrune gamma must be non-negative.")
        if self.algorithm == "prumerge" and self.prumerge_cluster_size < 1:
            raise BackendIntegrationError("Qwen3.5 PruMerge cluster_size must be positive.")
        if self.algorithm == "mmtok" and (
            not math.isfinite(self.mmtok_alpha)
            or self.mmtok_alpha < 0
            or not math.isfinite(self.mmtok_text_temperature)
            or self.mmtok_text_temperature <= 0
            or not math.isfinite(self.mmtok_vision_temperature)
            or self.mmtok_vision_temperature <= 0
        ):
            raise BackendIntegrationError("Qwen3.5 MMTok has invalid coverage parameters.")
        if self.algorithm == "vispruner" and (
            not math.isfinite(self.vispruner_important_ratio)
            or not 0.0 <= self.vispruner_important_ratio <= 1.0
            or self.vispruner_removal_step < 1
        ):
            raise BackendIntegrationError("Qwen3.5 VisPruner has invalid selection parameters.")
        if self.algorithm == "restore" and (
            not math.isfinite(self.restore_pruning_ratio)
            or not 0.0 < self.restore_pruning_ratio < 1.0
            or not math.isfinite(self.restore_important_ratio)
            or not 0.0 <= self.restore_important_ratio <= 1.0
            or self.restore_removal_step < 1
        ):
            raise BackendIntegrationError("Qwen3.5 RESTORE has invalid merge parameters.")
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
            if args or kwargs.get("pixel_values") is None:
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
            if hasattr(raw, "sequences"):
                import torch

                restored = torch.cat((original_ids, sequences[:, compact_width:]), dim=-1)
                raw.sequences = restored
            else:
                import torch

                raw = torch.cat((original_ids, sequences[:, compact_width:]), dim=-1)
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

    def _compress_inputs(self, source: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        import torch
        from transformers.modeling_outputs import BaseModelOutputWithPooling

        kwargs = dict(source)
        pixel_values = kwargs.get("pixel_values")
        input_ids = kwargs.get("input_ids")
        grid = kwargs.get("image_grid_thw")
        mm_types = kwargs.get("mm_token_type_ids")
        if self.algorithm in {"visionzip", "otprune", "prumerge", "mmtok", "gap", "vispruner", "aim", "restore", "saint", "nuwa"} and kwargs.get("pixel_values_videos") is not None:
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} image binding does not accept video inputs."
            )
        if pixel_values is None or input_ids is None or grid is None or mm_types is None:
            raise BackendIntegrationError(
                f"Qwen3.5 projector {self.algorithm} requires pixel_values, image_grid_thw, "
                "input_ids and mm_token_type_ids."
            )
        if input_ids.shape[0] != 1:
            raise BackendIntegrationError("Qwen3.5 projector ToMe currently requires batch_size=1.")

        vision_statistics = None
        vision_reference_groups = None
        zoo_projector_input_groups = None
        nuwa_semantic_groups = None
        if self.algorithm == "nuwa":
            if self.nuwa_spec is None:
                raise BackendIntegrationError("Qwen3.8 Nüwa has no validated method spec.")
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise BackendIntegrationError("Qwen3.8 Nüwa requires the native visual tower.")
            visual_config = getattr(self.model.config, "vision_config", None)
            spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
            if spatial_merge_size < 1:
                raise BackendIntegrationError("Qwen3.8 Nüwa requires vision_config.spatial_merge_size.")
            with capture_qwen3_5_premerger_features(visual) as captured_patch_features:
                with capture_qwen3_5_semantic_hidden_range(
                    visual,
                    start_layer=self.nuwa_spec.semantic_hidden_start,
                    end_layer=self.nuwa_spec.semantic_hidden_end,
                ) as semantic_hidden_states:
                    with capture_qwen3_5_final_vision_attention(
                        visual, query_chunk_size=self.attention_chunk_size
                    ) as vision_statistics:
                        image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
            if len(captured_patch_features) != 1 or not semantic_hidden_states:
                raise BackendIntegrationError(
                    "Qwen3.8 Nüwa requires one image and all configured semantic vision layers."
                )
            semantic_patches = torch.stack(semantic_hidden_states, dim=0).mean(dim=0)
            nuwa_semantic_groups = pool_premerger_features_to_merger_tokens(
                semantic_patches,
                grid,
                spatial_merge_size=spatial_merge_size,
            )
        elif self.algorithm in {"visionzip", "prumerge", "gap", "bifocal", "saint"}:
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise BackendIntegrationError("Qwen3.5 VisionZip requires the native visual tower.")
            with capture_qwen3_5_final_vision_attention(
                visual, query_chunk_size=self.attention_chunk_size
            ) as vision_statistics:
                image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
        elif self.algorithm == "mmtok":
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise BackendIntegrationError("Qwen3.5 MMTok requires the native visual tower.")
            visual_config = getattr(self.model.config, "vision_config", None)
            spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
            if spatial_merge_size < 1:
                raise BackendIntegrationError(
                    "Qwen3.5 MMTok requires vision_config.spatial_merge_size."
                )
            with capture_qwen3_5_premerger_features(visual) as captured_patch_features:
                image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
            if len(captured_patch_features) != 1:
                raise BackendIntegrationError(
                    "Qwen3.5 MMTok expected one packed pre-merger feature capture."
                )
            vision_reference_groups = pool_premerger_features_to_merger_tokens(
                captured_patch_features[0],
                grid,
                spatial_merge_size=spatial_merge_size,
            )
        elif self.algorithm == "zoo_prune":
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise BackendIntegrationError("Qwen3.5 ZOO-Prune requires the native visual tower.")
            visual_config = getattr(self.model.config, "vision_config", None)
            spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
            if spatial_merge_size < 1:
                raise BackendIntegrationError("Qwen3.5 ZOO-Prune requires vision_config.spatial_merge_size.")
            with capture_qwen3_5_premerger_features(visual) as captured_patch_features:
                image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
            merger = getattr(visual, "merger", None)
            if merger is None:
                raise BackendIntegrationError("Qwen3.5 ZOO-Prune requires the native patch merger.")
            zoo_projector_input_groups = build_qwen3_5_merger_inputs(
                captured_patch_features[0], grid,
                spatial_merge_size=spatial_merge_size, merger=merger,
            )
        elif self.algorithm in {"vispruner", "restore"}:
            visual = getattr(getattr(self.model, "model", None), "visual", None)
            if visual is None:
                raise BackendIntegrationError(f"Qwen3.5 {self.algorithm} requires the native visual tower.")
            visual_config = getattr(self.model.config, "vision_config", None)
            spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
            if spatial_merge_size < 1:
                raise BackendIntegrationError(
                    f"Qwen3.5 {self.algorithm} requires vision_config.spatial_merge_size."
                )
            with capture_qwen3_5_premerger_features(visual) as captured_patch_features:
                with capture_qwen3_5_final_vision_attention(
                    visual, query_chunk_size=self.attention_chunk_size
                ) as vision_statistics:
                    image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
            if len(captured_patch_features) != 1:
                raise BackendIntegrationError(
                    f"Qwen3.5 {self.algorithm} expected one packed pre-merger feature capture."
                )
            vision_reference_groups = pool_premerger_features_to_merger_tokens(
                captured_patch_features[0],
                grid,
                spatial_merge_size=spatial_merge_size,
            )
        else:
            image_outputs = self.model.get_image_features(pixel_values, grid, return_dict=True)
        feature_groups = image_outputs.pooler_output
        if torch.is_tensor(feature_groups):
            feature_groups = (feature_groups,)
        feature_groups = tuple(feature_groups)
        if vision_statistics is not None and len(vision_statistics) != len(feature_groups):
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} attention capture produced "
                f"{len(vision_statistics)} visual sequences for {len(feature_groups)} images."
            )
        if vision_reference_groups is not None and len(vision_reference_groups) != len(feature_groups):
            raise BackendIntegrationError(
                f"Qwen3.5 {self.algorithm} pre-merger features and image feature groups disagree."
            )
        if nuwa_semantic_groups is not None and len(nuwa_semantic_groups) != len(feature_groups):
            raise BackendIntegrationError("Qwen3.8 Nüwa semantic features and merger outputs disagree.")
        image_token_id = getattr(self.model.config, "image_token_id", None)
        if image_token_id is None:
            raise BackendIntegrationError("Qwen3.5 config does not expose image_token_id.")
        all_positions = torch.nonzero(input_ids[0] == int(image_token_id), as_tuple=False).flatten()
        expected = sum(int(features.shape[0]) for features in feature_groups)
        if int(all_positions.numel()) != expected:
            raise BackendIntegrationError(
                f"Image placeholder/features mismatch: {int(all_positions.numel())} "
                f"placeholders for {expected} merged features."
            )
        text_embeddings = None
        if self.algorithm == "mmtok":
            text_mask = input_ids[0] != int(image_token_id)
            prompt_mask = kwargs.get("attention_mask")
            if prompt_mask is not None and prompt_mask.ndim == 2:
                text_mask = text_mask & prompt_mask[0].bool()
            prompt_ids = input_ids[0][text_mask]
            if int(prompt_ids.numel()) < 1:
                raise BackendIntegrationError("Qwen3.5 MMTok found no prompt text tokens.")
            text_embeddings = self.model.get_input_embeddings()(prompt_ids)

        original_positions, original_delta = self.model.model.get_rope_index(
            input_ids=input_ids,
            mm_token_type_ids=mm_types,
            image_grid_thw=grid,
            attention_mask=kwargs.get("attention_mask"),
        )
        keep = torch.ones(input_ids.shape[1], dtype=torch.bool, device=input_ids.device)
        keep[all_positions] = False
        compressed_groups = []
        representative_positions = []
        per_image_records = []
        cursor = 0
        for image_index, features in enumerate(feature_groups):
            source_count = int(features.shape[0])
            positions = all_positions[cursor : cursor + source_count]
            cursor += source_count
            target_count = (
                min(source_count, int(self.target_count))
                if self.target_count is not None
                else max(1, math.ceil(source_count * self.retain_ratio))
            )
            if self.algorithm == "visionzip" and source_count > 1:
                target_count = min(source_count, max(2, target_count))
            remove_count = source_count - target_count
            if self.algorithm == "tome":
                result = tome_merge_image(
                    features,
                    features,
                    r=remove_count,
                    media_unit="image_feature",
                    clamp_r=False,
                )
                compressed_features = result.hidden_states
                representatives = tuple(
                    int(value)
                    for value in result.layout_delta.result_layout.get(
                        "representative_indices", range(source_count)
                    )
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "divprune":
                result = divprune_image(
                    features,
                    spec=DivPruneSpec(
                        retain_ratio=self.retain_ratio,
                        target_count=target_count,
                        prune_layer=self.prune_layer,
                        distance=self.distance,
                        selection=self.selection,
                        min_visual_tokens=self.min_visual_tokens,
                        distance_chunk_size=self.distance_chunk_size,
                    ),
                    media_unit="image_feature",
                    rules=self.rules,
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "aim":
                result = aim_image_tokens(
                    features,
                    spec=AIMSpec(
                        retain_ratio=self.retain_ratio,
                        target_count=target_count,
                        min_visual_tokens=self.min_visual_tokens,
                    ),
                    media_unit="image_feature",
                    source_layout={"backbone": "qwen3_5", "image_index": image_index},
                    result_layout={
                        "backbone": "qwen3_5",
                        "image_index": image_index,
                        "target_site": "post_merger_pre_llm_scatter",
                    },
                    rules=self.rules,
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value)
                    for value in result.representative_indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "otprune":
                result = otprune_image(
                    features,
                    spec=OTPruneSpec(
                        retain_ratio=self.retain_ratio,
                        target_count=target_count,
                        gamma=self.gamma,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "gap":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                if spatial_merge_size < 1:
                    raise BackendIntegrationError(
                        "Qwen3.5 GAP requires vision_config.spatial_merge_size."
                    )
                received, _ = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                # GAP expects a dense attention matrix and reduces it to mean
                # received importance. Repeating this vector preserves the
                # same score without reconstructing a patch attention matrix.
                attention = received.unsqueeze(0).expand(source_count, -1)
                result = gap_image_tokens(
                    features,
                    attention,
                    spec=GAPSpec(
                        retain_ratio=self.retain_ratio,
                        target_count=target_count,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "prumerge":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                if spatial_merge_size < 1:
                    raise BackendIntegrationError(
                        "Qwen3.5 PruMerge+ requires vision_config.spatial_merge_size."
                    )
                received, keys = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                # The shared PruMerge kernel expects an attention matrix and
                # averages over query rows. Repeating the Qwen3.8 mean-received
                # vector preserves exactly that importance score without
                # reconstructing or storing a dense attention map.
                attention = received.unsqueeze(0).expand(source_count, -1)
                result = prumerge_image_tokens(
                    features,
                    attention,
                    keys,
                    spec=PruMergeSpec(
                        variant="plus",
                        adaptive=False,
                        reduction_ratio=self.retain_ratio,
                        target_count=max(2, target_count),
                        cluster_size=self.prumerge_cluster_size,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "mmtok":
                result = mmtok_image_tokens(
                    features,
                    vision_reference_groups[image_index].to(features.device),
                    text_embeddings.to(features.device),
                    spec=MMTokSpec(
                        retain_ratio=self.retain_ratio,
                        target_count=self.target_count,
                        alpha=self.mmtok_alpha,
                        text_temperature=self.mmtok_text_temperature,
                        vision_temperature=self.mmtok_vision_temperature,
                        min_visual_tokens=self.min_visual_tokens,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "vispruner":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                received, _ = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                result = vispruner_image_tokens(
                    features,
                    vision_reference_groups[image_index].to(features.device),
                    received.to(features.device),
                    spec=VisPrunerSpec(
                        target_count=min(
                            source_count,
                            max(self.min_visual_tokens, target_count),
                        ),
                        important_ratio=self.vispruner_important_ratio,
                        removal_step=self.vispruner_removal_step,
                        min_visual_tokens=min(self.min_visual_tokens, source_count),
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "restore":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                received, _ = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                min_visual_tokens = min(max(2, self.min_visual_tokens), source_count)
                effective_target = min(source_count, max(min_visual_tokens, target_count))
                result = restore_image_tokens(
                    features,
                    vision_reference_groups[image_index].to(features.device),
                    received.to(features.device),
                    spec=RestoreSpec(
                        target_count=effective_target,
                        pruning_ratio=self.restore_pruning_ratio,
                        important_ratio=self.restore_important_ratio,
                        removal_step=self.restore_removal_step,
                        min_visual_tokens=min_visual_tokens,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value)
                    for value in result.representative_indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "bifocal":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                if spatial_merge_size < 1:
                    raise BackendIntegrationError(
                        "Qwen3.5 Bifocal requires vision_config.spatial_merge_size."
                    )
                received, _ = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                t, h, w = (int(value) for value in grid[image_index].tolist())
                grid_height, grid_width = h // spatial_merge_size, w // spatial_merge_size
                spec = BifocalSpec(
                    s1_frac=min(0.95, max(0.05, self.retain_ratio)),
                    metric="vit_attn",
                    metric_agg="mass",
                )
                text_mask = input_ids[0] != int(image_token_id)
                prompt_mask = kwargs.get("attention_mask")
                if prompt_mask is not None and prompt_mask.ndim == 2:
                    text_mask = text_mask & prompt_mask[0].bool()
                text_ids = input_ids[0][text_mask]
                word_embeddings = self.model.get_input_embeddings()(text_ids)
                result = bifocal_image_tokens(
                    features,
                    {"mass": received},
                    grid_height=grid_height,
                    grid_width=grid_width,
                    word_embeddings=word_embeddings,
                    spec=spec,
                    media_unit="image_feature",
                    source_layout={"backbone": "qwen3_5", "image_index": image_index},
                    result_layout={"backbone": "qwen3_5", "image_index": image_index},
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "zoo_prune":
                visual = self.model.model.visual
                merger = visual.merger
                if zoo_projector_input_groups is None:
                    raise BackendIntegrationError("Qwen3.5 ZOO-Prune did not capture native merger inputs.")
                result = zoo_prune_image_tokens(
                    features,
                    zoo_projector_input_groups[image_index].to(features.device),
                    (merger.linear_fc1, merger.act_fn, merger.linear_fc2),
                    spec=ZOOPruneSpec(
                        retention_ratio=self.retain_ratio,
                        target_count=target_count,
                        num_directions=self.zoo_num_directions,
                        direction_batch_size=4,
                        low_rank_rank=64,
                        min_visual_tokens=self.min_visual_tokens,
                    ),
                    media_unit="image_feature",
                )
                compressed_features = result.tokens
                representatives = tuple(int(value) for value in result.indices.detach().cpu().tolist())
                layout_delta = result.layout_delta
            elif self.algorithm == "saint":
                if self.saint_spec is None:
                    raise BackendIntegrationError("Qwen3.5 SAINT has no validated method spec.")
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                if spatial_merge_size < 1:
                    raise BackendIntegrationError("Qwen3.5 SAINT requires vision_config.spatial_merge_size.")
                _, pooled_keys = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                keys = pooled_keys.permute(1, 0, 2).contiguous()
                result = saint_image_tokens(
                    features,
                    keys,
                    spec=self.saint_spec,
                    media_unit="image_feature",
                    source_layout={"backbone": "qwen3_5", "image_index": image_index,
                                   "key_source": "native_final_vision_rotary_keys"},
                    result_layout={"backbone": "qwen3_5", "image_index": image_index,
                                  "target_site": "post_merger_pre_llm_scatter"},
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.selected_indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            elif self.algorithm == "nuwa":
                if self.nuwa_spec is None or nuwa_semantic_groups is None or vision_statistics is None:
                    raise BackendIntegrationError("Qwen3.8 Nüwa did not capture its Stage 1 inputs.")
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                time_size, grid_h, grid_w = (int(value) for value in grid[image_index].tolist())
                if time_size != 1 or grid_h % spatial_merge_size or grid_w % spatial_merge_size:
                    raise BackendIntegrationError("Qwen3.8 Nüwa requires one image with a merger-aligned spatial grid.")
                attention, keys = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                result = nuwa_stage1_image_tokens(
                    features,
                    nuwa_semantic_groups[image_index].to(features.device),
                    attention.to(features.device),
                    keys.to(features.device),
                    grid_height=grid_h // spatial_merge_size,
                    grid_width=grid_w // spatial_merge_size,
                    spec=self.nuwa_spec,
                    media_unit="image_feature",
                    source_layout={"backbone": "qwen3_5", "image_index": image_index,
                                   "key_source": "native_final_vision_rotary_keys"},
                    result_layout={"backbone": "qwen3_5", "image_index": image_index,
                                   "target_site": "post_merger_pre_llm_scatter",
                                   "position_policy": "source_representative_rpme"},
                )
                compressed_features = result.tokens
                representatives = tuple(int(value) for value in result.selected_indices.detach().cpu().tolist())
                layout_delta = result.layout_delta
            elif self.algorithm == "visionzip":
                visual_config = getattr(self.model.config, "vision_config", None)
                spatial_merge_size = int(getattr(visual_config, "spatial_merge_size", 0))
                if spatial_merge_size < 1:
                    raise BackendIntegrationError(
                        "Qwen3.5 VisionZip requires vision_config.spatial_merge_size."
                    )
                attention, keys = pool_patch_statistics_to_merger_tokens(
                    vision_statistics[image_index],
                    merged_token_count=source_count,
                    spatial_merge_size=spatial_merge_size,
                )
                result = visionzip_image(
                    features,
                    attention,
                    keys,
                    budget=target_count,
                    attention_mode="mean_received",
                    media_unit="image_feature",
                    rules=self.rules,
                )
                compressed_features = result.tokens
                representatives = tuple(
                    int(value) for value in result.indices.detach().cpu().tolist()
                )
                layout_delta = result.layout_delta
            if len(representatives) != int(compressed_features.shape[0]):
                raise BackendIntegrationError(
                    f"{self.algorithm} output and LayoutDelta representatives disagree."
                )
            compressed_groups.append(compressed_features)
            representative_positions.append(
                positions[torch.tensor(representatives, dtype=torch.long, device=positions.device)]
            )
            keep[representative_positions[-1]] = True
            self._layout_records.append(layout_delta_record(layout_delta))
            per_image_records.append(
                {
                    "image_index": image_index,
                    "input_tokens": source_count,
                    "output_tokens": int(compressed_features.shape[0]),
                    "r_requested": remove_count,
                    "r_effective": (
                        int(result.metadata["r_effective"])
                        if self.algorithm == "tome"
                        else remove_count
                    ),
                    "media_unit": "image_feature",
                    **(
                        {
                            "selection_method": result.metadata["selection"],
                            "gamma": self.gamma,
                        }
                        if self.algorithm == "otprune"
                        else {}
                    ),
                    **(
                        {
                            "attention_mode": "mean_received_no_cls",
                            "key_position_encoding": "qwen3_5_rotary",
                            "patches_per_merger_token": spatial_merge_size**2,
                            "center_count": int(result.metadata["center_count"]),
                            "global_count": int(result.metadata["global_count"]),
                            "cluster_size": int(result.metadata["cluster_size"]),
                            "merge_mode": result.metadata["merge_mode"],
                            "budget_mode": "fixed_total_output_tokens",
                        }
                        if self.algorithm == "prumerge"
                        else {}
                    ),
                    **(
                        {
                            "text_token_count": int(result.metadata["text_token_count"]),
                            "alpha": self.mmtok_alpha,
                            "text_temperature": self.mmtok_text_temperature,
                            "vision_temperature": self.mmtok_vision_temperature,
                            "vision_source": "mean_pooled_pre_merger_patch_features",
                            "text_source": "prompt_input_embeddings",
                        }
                        if self.algorithm == "mmtok"
                        else {}
                    ),
                    **(
                        {
                            "attention_mode": "mean_received_no_cls",
                            "selection": "received_attention_topk",
                            "position_policy": "preserve_original",
                            "patches_per_merger_token": spatial_merge_size**2,
                        }
                        if self.algorithm == "gap"
                        else {}
                    ),
                    **(
                        {
                            "key_source": "native_final_vision_rotary_keys",
                            "key_position_encoding": "qwen3_5_rotary",
                            "patches_per_merger_token": spatial_merge_size**2,
                            "prune_level": self.saint_spec.prune_level,
                            "group_size": self.saint_spec.group_size,
                        }
                        if self.algorithm == "saint" and self.saint_spec is not None
                        else {}
                    ),
                    **(
                        {
                            "stage": "nuwa_stage1",
                            "semantic_layers": [self.nuwa_spec.semantic_hidden_start,
                                                self.nuwa_spec.semantic_hidden_end],
                            "stage1_target_tokens": self.nuwa_spec.stage1_total_tokens,
                            "stage2_after_layer": self.nuwa_spec.stage2_after_layer,
                        }
                        if self.algorithm == "nuwa" and self.nuwa_spec is not None
                        else {}
                    ),
                    **(
                        {
                            "attention_mode": "mean_received_no_cls",
                            "visual_reference": "mean_pooled_pre_merger_patch_features",
                            "important_ratio": self.vispruner_important_ratio,
                            "important_count": int(result.metadata["important_count"]),
                            "diverse_count": int(result.metadata["diverse_count"]),
                            "removal_step": self.vispruner_removal_step,
                            "patches_per_merger_token": spatial_merge_size**2,
                        }
                        if self.algorithm == "vispruner"
                        else {}
                    ),
                    **(
                        {
                            "merge_round_count": len(result.merge_rounds),
                            "representative_indices": representatives,
                            "token_sizes": tuple(
                                float(value)
                                for value in result.token_sizes.detach().cpu().tolist()
                            ),
                        }
                        if self.algorithm == "aim"
                        else {}
                    ),
                    **(
                        {
                            "attention_mode": "mean_received_no_cls",
                            "visual_reference": "mean_pooled_pre_merger_patch_features",
                            "pristine_count": int(result.metadata["pristine_count"]),
                            "anchor_count": int(result.metadata["anchor_count"]),
                            "pruning_ratio": self.restore_pruning_ratio,
                            "anchor_mask_mode": result.metadata["anchor_mask_mode"],
                            "position_strategy": result.metadata["position_strategy"],
                            "patches_per_merger_token": spatial_merge_size**2,
                        }
                        if self.algorithm == "restore"
                        else {}
                    ),
                    **(
                        {
                            "attention_mode": "mean_received",
                            "key_position_encoding": "qwen3_5_rotary",
                            "patches_per_merger_token": spatial_merge_size**2,
                            "attention_chunk_size": self.attention_chunk_size,
                        }
                        if self.algorithm == "visionzip"
                        else {}
                    ),
                }
            )

        source_mask = kwargs.get("attention_mask")
        if source_mask is None:
            source_mask = torch.ones_like(input_ids)
        removed_valid_tokens = int((~keep & source_mask[0].bool()).sum())
        compressed_ids = input_ids[:, keep]
        compressed_mask = kwargs.get("attention_mask")
        if compressed_mask is not None:
            compressed_mask = compressed_mask[:, keep]
        compressed_types = mm_types[:, keep]
        compressed_positions = original_positions[:, :, keep]
        # Generated positions are based on the compact cache length. Add back
        # the removed prompt slots so decode-time M-RoPE continues from the
        # original text position after the image.
        self.model.model.rope_deltas = original_delta + removed_valid_tokens

        kwargs["input_ids"] = compressed_ids
        kwargs["attention_mask"] = compressed_mask
        kwargs["mm_token_type_ids"] = compressed_types
        kwargs["position_ids"] = compressed_positions
        kwargs["mm_encoder_outputs"] = {
            "image": BaseModelOutputWithPooling(pooler_output=tuple(compressed_groups))
        }
        kwargs.pop("pixel_values", None)
        kwargs.pop("image_grid_thw", None)
        kwargs.pop("pixel_values_videos", None)
        kwargs.pop("video_grid_thw", None)
        for name, value in tuple(kwargs.items()):
            if name in {"input_ids", "attention_mask", "mm_token_type_ids", "position_ids"}:
                continue
            if torch.is_tensor(value) and value.ndim >= 2 and value.shape[-1] == keep.shape[0]:
                kwargs[name] = value[..., keep]

        record = {
            "algorithm": self.algorithm,
            "placement": "post_merger_pre_llm_scatter",
            "input_prompt_tokens": int(input_ids.shape[1]),
            "output_prompt_tokens": int(compressed_ids.shape[1]),
            "image_tokens_before": expected,
            "image_tokens_after": sum(int(item.shape[0]) for item in compressed_groups),
            "images": per_image_records,
            "mrope_positions_synchronized": True,
            "mm_encoder_outputs_used": True,
        }
        return kwargs, record

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "backbone": "qwen3_5",
            "algorithm": self.algorithm,
            "variant": {
                "tome": "tome.image.projector_feature",
                "divprune": "divprune.image.projector_feature",
                "visionzip": "visionzip.image.projector_feature",
                "otprune": "otprune.image.projector_feature",
                "prumerge": "prumerge.image",
                "mmtok": "mmtok.image",
                "gap": "gap.image",
                "vispruner": "vispruner.image",
                "aim": "aim.image",
                "restore": "restore.image",
                "bifocal": "bifocal.image",
                "zoo_prune": "zoo_prune.image",
                "saint": "saint.image.projector_feature",
                "nuwa": "nuwa.image.qwen3_5",
            }[self.algorithm],
            "binding": self.binding_name,
            "target_site": "post_merger_pre_llm_scatter",
            "retain_ratio": self.retain_ratio,
            "target_count": self.target_count,
            "installed": self._installed,
            "generation_calls": len(self._request_records),
            "layout_delta_records_count": len(self._layout_records),
            "layout_delta_records": list(self._layout_records),
            "requests": list(self._request_records),
            "video_support": False,
        }


class Qwen3_5ToMeActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "tome"
    binding_name = "tome.image.qwen3_5"


class Qwen3_5DivPruneActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "divprune"
    binding_name = "divprune.image.qwen3_5"


class Qwen3_5VisionZipActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "visionzip"
    binding_name = "visionzip.image.qwen3_5"


class Qwen3_5OTPruneActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "otprune"
    binding_name = "otprune.image.qwen3_5"


class Qwen3_5PruMergeActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "prumerge"
    binding_name = "prumerge.image.qwen3_5"


class Qwen3_5MMTokActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "mmtok"
    binding_name = "mmtok.image.qwen3_5"


class Qwen3_5GAPActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "gap"
    binding_name = "gap.image.qwen3_5"


class Qwen3_5VisPrunerActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "vispruner"
    binding_name = "vispruner.image.qwen3_5"


class Qwen3_5AIMActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "aim"
    binding_name = "aim.image.qwen3_5"


class Qwen3_5RestoreActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "restore"
    binding_name = "restore.image.qwen3_5"


class Qwen3_5BifocalActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "bifocal"
    binding_name = "bifocal.image.qwen3_5"


class Qwen3_5ZOOPruneActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "zoo_prune"
    binding_name = "zoo_prune.image.qwen3_5"


class Qwen3_5SAINTActivation(_Qwen3_5ImageCompressionActivation):
    algorithm = "saint"
    binding_name = "saint.image.qwen3_5"


class Qwen3_5NuwaActivation(_Qwen3_5ImageCompressionActivation):
    """Composite Nüwa adapter: native vision Stage 1 plus decoder Stage 2."""

    algorithm = "nuwa"
    binding_name = "nuwa.image.qwen3_5"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5NuwaDecoderActivation

        self._rollback_errors: list[str] = []
        self._uninstall_errors: list[str] = []
        self.decoder = Qwen3_5NuwaDecoderActivation(**kwargs)

    def install(self) -> None:
        if self._installed:
            return
        import types

        owner = self

        def generate_hook(_model: Any, *args: Any, **kwargs: Any) -> Any:
            call_kwargs = dict(kwargs)
            if args and "input_ids" not in call_kwargs:
                call_kwargs["input_ids"] = args[0]
            input_ids = call_kwargs.get("input_ids")
            if input_ids is None or call_kwargs.get("pixel_values") is None:
                return owner._original_generate(*args, **kwargs)
            original_ids = input_ids
            previous_rope = getattr(owner.model.model, "rope_deltas", None)
            prepared, record = owner._compress_inputs(call_kwargs)
            try:
                output = owner.decoder._generate_with_decoder_selection(
                    (), prepared, dict(prepared)
                )
            finally:
                owner.model.model.rope_deltas = previous_rope
            sequences = getattr(output, "sequences", output)
            compact_width = int(prepared["input_ids"].shape[1])
            restored = __import__("torch").cat(
                (original_ids, sequences[:, compact_width:]), dim=-1
            )
            if hasattr(output, "sequences"):
                output.sequences = restored
            else:
                output = restored
            record["generated"] = True
            record["stage2_after_layer"] = owner.nuwa_spec.stage2_after_layer
            owner._request_records.append(record)
            return output

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
        super().clear_diagnostic_records()
        self.decoder._request_records.clear()
        self.decoder._layout_records.clear()

    def runtime_metadata(self) -> dict[str, Any]:
        metadata = super().runtime_metadata()
        metadata.update(
            {
                "composite_binding": True,
                "target_sites": ["post_merger_pre_llm_scatter", "decoder_prune_boundary"],
                "stage1_spec": {
                    "stage1_total_tokens": self.nuwa_spec.stage1_total_tokens,
                    "semantic_hidden_range": [self.nuwa_spec.semantic_hidden_start,
                                               self.nuwa_spec.semantic_hidden_end],
                    "position_mode": self.nuwa_spec.position_mode,
                },
                "stage2_spec": {
                    "stage2_after_layer": self.nuwa_spec.stage2_after_layer,
                    "stage2_score_layer": self.nuwa_spec.stage2_score_layer,
                    "stage2_retain_ratio": self.nuwa_spec.stage2_retain_ratio,
                    "cache_policy": "no_cache_full_prefix_recompute",
                },
                "layout_delta_records": [
                    *self._layout_records, *self.decoder._layout_records
                ],
                "layout_delta_records_count": (
                    len(self._layout_records) + len(self.decoder._layout_records)
                ),
                "stage2_requests": list(self.decoder._request_records),
                "rollback_errors": [*self._rollback_errors, *self.decoder._rollback_errors],
                "uninstall_errors": [*self._uninstall_errors, *self.decoder._uninstall_errors],
            }
        )
        return metadata


def create_qwen3_5_image_compression_executor(
    **kwargs: Any,
) -> _Qwen3_5ImageCompressionActivation:
    binding = kwargs.get("binding")
    name = getattr(binding, "name", None)
    if name == "tome.image.qwen3_5":
        return Qwen3_5ToMeActivation(**kwargs)
    if name == "divprune.image.qwen3_5":
        return Qwen3_5DivPruneActivation(**kwargs)
    if name == "visionzip.image.qwen3_5":
        return Qwen3_5VisionZipActivation(**kwargs)
    if name == "otprune.image.qwen3_5":
        return Qwen3_5OTPruneActivation(**kwargs)
    if name == "prumerge.image.qwen3_5":
        return Qwen3_5PruMergeActivation(**kwargs)
    if name == "mmtok.image.qwen3_5":
        return Qwen3_5MMTokActivation(**kwargs)
    if name == "gap.image.qwen3_5":
        return Qwen3_5GAPActivation(**kwargs)
    if name == "vispruner.image.qwen3_5":
        return Qwen3_5VisPrunerActivation(**kwargs)
    if name == "aim.image.qwen3_5":
        return Qwen3_5AIMActivation(**kwargs)
    if name == "restore.image.qwen3_5":
        return Qwen3_5RestoreActivation(**kwargs)
    if name == "bifocal.image.qwen3_5":
        return Qwen3_5BifocalActivation(**kwargs)
    if name == "zoo_prune.image.qwen3_5":
        return Qwen3_5ZOOPruneActivation(**kwargs)
    if name == "saint.image.qwen3_5":
        return Qwen3_5SAINTActivation(**kwargs)
    if name == "nuwa.image.qwen3_5":
        return Qwen3_5NuwaActivation(**kwargs)
    if name == "sparsevlm.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5SparseVLMActivation

        return Qwen3_5SparseVLMActivation(**kwargs)
    if name == "finepruner.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5FinePrunerActivation

        return Qwen3_5FinePrunerActivation(**kwargs)
    if name == "transprune.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5TransPruneActivation

        return Qwen3_5TransPruneActivation(**kwargs)
    if name == "fastv.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5FastVActivation

        return Qwen3_5FastVActivation(**kwargs)
    if name == "pyramid_drop.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5PyramidDropActivation

        return Qwen3_5PyramidDropActivation(**kwargs)
    if name == "dart.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5DARTActivation

        return Qwen3_5DARTActivation(**kwargs)
    if name == "ivc_prune.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5IVCPruneActivation

        return Qwen3_5IVCPruneActivation(**kwargs)
    if name == "feather.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5FEATHERActivation

        return Qwen3_5FEATHERActivation(**kwargs)
    if name == "btp.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5BTPActivation

        return Qwen3_5BTPActivation(**kwargs)
    if name == "halfv.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5HalfVActivation

        return Qwen3_5HalfVActivation(**kwargs)
    if name == "sgl.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5SGLActivation

        return Qwen3_5SGLActivation(**kwargs)
    if name == "d2pruner.image.qwen3_5":
        from mllm_comp.backbones.qwen3_5.decoder_fastv import Qwen3_5D2PrunerActivation

        return Qwen3_5D2PrunerActivation(**kwargs)
    raise BackendIntegrationError(f"Unsupported Qwen3.5 image binding: {name!r}.")
