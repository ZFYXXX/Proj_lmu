"""Comparison selectors at the native OneVision packed-image boundary."""
from dataclasses import replace
from importlib import import_module

from mllm_comp.core.layout import LayoutDelta
from mllm_comp.core.vocabulary import ExecutionPhase


class LlavaProjectorComparison:
    def __init__(self, model, stage, binding, phase):
        self.model, self.method, self.binding = model, stage.algorithm, binding
        self.phase = ExecutionPhase(phase)
        module = import_module("." + ("cdpruner" if self.method == "dppruner" else self.method), __package__)
        defaults = {"anyres_assembly": "onevision_unpad"} if self.method in {"cdpruner", "dppruner"} else {}
        self.config = getattr(module, f"normalize_{self.method}_config")({**defaults, **dict(stage.config)})
        if self.config.get("anyres_assembly", "onevision_unpad") != "onevision_unpad":
            raise ValueError("The native OneVision binding requires anyres_assembly=onevision_unpad")
        if self.config.get("position_id_mode", "reindex") != "reindex":
            raise ValueError("This packed-image binding currently requires position_id_mode=reindex")
        self.bridge = self.handle = None
        self.owns_bridge = False
        self.records = []

    def install(self):
        if self.handle is not None:
            return
        from mllm_comp.backbones.llava_onevision.image_compression import _resolve_or_install_bridge, VISIONZIP_SITE
        self.bridge, self.owns_bridge = _resolve_or_install_bridge(self.model)

        def callback(request):
            return self.select(request)

        callback._mllm_algorithm = self.method
        callback._mllm_activation_id = id(self)
        callback._mllm_execution_phase = self.phase
        callback._mllm_requires_vision_source = True
        if self.method in {"cdpruner", "eadp"}:
            callback._mllm_instruction_selector = instruction_ids
        if self.method == "hiprune":
            callback._mllm_vision_attention_layers = (self.config["object_layer"] - 1, self.config["vision_layer"])
        if self.method == "dop":
            callback._mllm_vision_attention_layers = (self.config["vision_layer"],)
        try:
            self.handle = self.bridge.register_callback(VISIONZIP_SITE, callback)
        except BaseException:
            self.uninstall()
            raise

    def uninstall(self):
        if self.handle is not None:
            self.handle.remove()
            self.handle = None
        if self.owns_bridge and self.bridge is not None:
            self.bridge.detach()
        self.bridge = None

    def select(self, request):
        import torch
        from .cdpruner import select_cdpruner_tokens
        from mllm_comp.backbones.llava_onevision.image_compression import (
            LlavaPackedImageResponse, _crop_source_ids_in_packed_layout,
        )
        crops = request.projected_crops
        if crops is None:
            raise ValueError("Comparison selection requires native projected crop features")
        count = int(request.tokens.shape[0])
        sources = _crop_source_ids_in_packed_layout(
            crop_count=int(crops.shape[0]), patch_count=int(crops.shape[1]),
            patches_per_side=request.patches_per_side, grid_width=request.grid_width,
            grid_height=request.grid_height, patch_merge_type=request.patch_merge_type,
            image_size=request.image_sizes[0], source_count=count,
            newline_indices=request.newline_indices, device=request.tokens.device)
        if self.method in {"cdpruner", "dppruner"}:
            _, selected, stats = select_cdpruner_tokens(crops, request.text_embeddings, cfg=self.config)
        elif self.method == "eadp":
            from .eadp import select_eadp_indices
            valid = torch.zeros(crops.shape[:2], dtype=torch.bool, device=crops.device)
            valid.view(-1)[sources[sources >= 0].to(crops.device)] = True
            selected, stats = select_eadp_indices(crops, request.text_embeddings,
                patches_per_side=request.patches_per_side, cfg=self.config, valid_masks=valid)
        elif self.method == "dop":
            from mllm_comp.compressors.image.vispruner import select_vispruner_indices
            selected = [select_vispruner_indices(features,
                request.vision_attention_layers[self.config["vision_layer"]][crop],
                min(self.config["tokens_per_crop"], len(features)),
                important_ratio=self.config["important_ratio"], removal_step=self.config["removal_step"]).sort().values
                for crop, features in enumerate(request.vision_crops)]
            stats = {"token_reducer": "vispruner"}
        else:
            from .hiprune import select_hiprune_indices
            layers = request.vision_attention_layers
            selected, stats = select_hiprune_indices(layers[self.config["object_layer"] - 1],
                layers[self.config["vision_layer"]], tokens_per_crop=self.config["tokens_per_crop"],
                alpha=self.config["alpha"], patches_per_side=request.patches_per_side)
        selected_sources = torch.cat([indices + crop * crops.shape[1]
            for crop, indices in enumerate(selected)]).to(sources.device)
        keep_mask = torch.isin(sources, selected_sources)
        keep_mask[list(request.newline_indices)] = True
        indices = keep_mask.nonzero(as_tuple=True)[0]
        if indices.numel() <= len(request.newline_indices):
            raise ValueError("No selected visual patches survived native AnyRes unpadding; increase tokens_per_crop")
        delta = LayoutDelta.drop(count, (~keep_mask).nonzero(as_tuple=True)[0].tolist(),
            media_unit="image_feature", protected_indices=request.newline_indices,
            source_layout=dict(request.anyres_layout))
        newlines = tuple(delta.old_to_new[i] for i in request.newline_indices)
        layout = {**dict(request.anyres_layout), "packed_token_count": delta.result_count,
            "newline_indices": newlines, "crop_boundaries": ((0, delta.result_count),),
            "algorithm": self.method, "layout_operation": "drop",
            "compression_site": "post_anyres_pack_pre_llm_splice"}
        delta = replace(delta, result_layout=layout)
        self.records.append({"source_count": count, "result_count": delta.result_count, **stats})
        return LlavaPackedImageResponse(
            tokens=request.tokens.index_select(0, indices), layout_delta=delta,
            image_sizes=request.image_sizes, modalities=request.modalities,
            newline_indices=newlines, anyres_layout=layout, image_index=request.image_index,
            metadata={"algorithm": self.method, "native_anyres_unpad": True, **stats})

    def runtime_metadata(self):
        return {"algorithm": self.method, "binding": self.binding.name,
            "installed": self.handle is not None, "layout_delta_records": list(self.records),
            "support_activation_claimed": False, "real_checkpoint_verified": False}

    def clear_diagnostic_records(self):
        self.records.clear()


def instruction_ids(ids, mask, labels):
    """Read only the current user message, including during teacher forcing."""
    from .eadp import extract_qwen2_user_instruction_ids
    boundary = None
    if labels is not None:
        answer = (labels[0] != -100).nonzero(as_tuple=True)[0]
        if answer.numel():
            boundary = int(answer[0])
    return extract_qwen2_user_instruction_ids(ids[0],
        attention_mask=None if mask is None else mask[0], answer_start=boundary)
