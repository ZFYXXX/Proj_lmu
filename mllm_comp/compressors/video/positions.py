"""Resolve video positions through the existing compound binding targets."""

from dataclasses import replace
from typing import Any, Mapping

from mllm_comp.compressors.layer_positions import LayerPosition, resolve_layer_positions
from .config import resolve_canonical_video_config


_PROJECTOR_INPUT_SITES = {
    "llava_onevision": "video_pre_vision_projector_pooled",
    "qwen25_vl": "video_pre_vision_merger",
    "qwen3_vl": "video_pre_main_and_deepstack_mergers",
    "qwen3_5": "video_pre_vision_merger",
}
_PROJECTOR_OUTPUT_SITES = {
    "llava_onevision": "video_projected_features_pre_llm_splice",
    "qwen25_vl": "video_features_pre_placeholder_scatter",
    "qwen3_vl": "video_features_pre_placeholder_scatter_and_deepstack",
    "qwen3_5": "video_features_pre_placeholder_scatter",
}
_COMPOUND_TARGETS = (
    frozenset({"detection_budget_provider", "dynamic_visionzip_reduce"}),
    frozenset({"ttm_visual_merge", "decode_kv_view_prune"}),
    frozenset({"event_pool", "prefill_attention_prune"}),
    frozenset({"temporal_spatial_reduce", "vtp_prefill_prune"}),
)
_DECODE_SITES = {
    "llava_onevision": "llm_decoder_video_tokens_prefill_or_decode",
    "qwen25_vl": "llm_decoder_video_tokens_sparse_mrope",
    "qwen3_vl": "llm_decoder_video_tokens_sparse_mrope_visual_pos_masks",
}


def video_projector_site(backbone: str, position: str) -> str:
    return (_PROJECTOR_INPUT_SITES if position == "before" else _PROJECTOR_OUTPUT_SITES)[backbone]


def _target_operation(stage: Any, target_name: str, *, projector: bool = False) -> Mapping[str, Any]:
    operation = getattr(stage, "operation", {}) or {}
    targets = operation.get("targets", {})
    allowed = next((names for names in _COMPOUND_TARGETS if target_name in names), {target_name})
    if not isinstance(targets, Mapping) or set(targets) - allowed:
        raise ValueError(f"Video operation.targets must name existing steps: {sorted(allowed)}.")
    scoped = targets.get(target_name, {})
    keys = {"target_site"} if projector else {"target_site", "layers"}
    if target_name == "dino_frame_select_sva_fusion":
        keys.add("frame_selection_site")
    if target_name in {"prefill_attention_prune", "vtp_prefill_prune"}:
        keys.add("layer_edge")
    if not isinstance(scoped, Mapping) or set(scoped) - keys:
        raise ValueError(f"Video target {target_name} accepts {sorted(keys)} only.")
    if "layers" in operation or "layer_edge" in operation:
        raise ValueError("Use scoped video operation.targets for auxiliary or KV layers; Projector has no layer edge.")
    if projector and "target_site" in operation and "target_site" in scoped:
        raise ValueError(f"Use either operation.target_site or the scoped {target_name} position.")
    return {**scoped, "target_site": operation["target_site"]} if projector and "target_site" in operation else scoped


def video_projector_position(stage: Any, backbone: str, *, target_name: str) -> str:
    values = _target_operation(stage, target_name, projector=True)
    site = values.get("target_site", "projector_output")
    if site in {"projector_input", _PROJECTOR_INPUT_SITES[backbone]}:
        return "before"
    if site in {"projector_output", _PROJECTOR_OUTPUT_SITES[backbone]}:
        return "after"
    raise ValueError(f"Video Projector target {target_name} cannot execute at {site!r}.")


def video_learned_projector_site(algorithm: str, position: str) -> str:
    sites = {"llamavid": {"before": "llamavid_pre_output_projectors", "after": "llamavid_post_output_projectors"}}
    return sites[algorithm][position]


def video_learned_projector_position(stage: Any, backbone: str, *, target_name: str = "qformer_context_content_pool") -> str:
    """LLaMA-VID pools before its own output projectors in the original recipe."""
    values = _target_operation(stage, target_name, projector=True)
    site = values.get("target_site", "projector_input")
    if site in {"projector_input", _PROJECTOR_OUTPUT_SITES[backbone], video_learned_projector_site("llamavid", "before")}:
        return "before"
    if site in {"projector_output", video_learned_projector_site("llamavid", "after")}:
        return "after"
    raise ValueError(f"LLaMA-VID pooling cannot execute at {site!r}; positions refer to its own output projectors.")


def video_longvu_frame_selection_site(stage: Any, backbone: str) -> str:
    values = _target_operation(stage, "dino_frame_select_sva_fusion", projector=True)
    site = values.get("target_site", "projector_input")
    if site not in {"projector_input", _PROJECTOR_OUTPUT_SITES[backbone], "longvu_pre_output_projector"}:
        raise ValueError("LongVU SVA requires its learned vision-hidden query/KV/residual dimensions before its nonlinear output projector; only frame selection can move across that projector.")
    frame_site = values.get("frame_selection_site", "pre_sva")
    if frame_site not in {"pre_sva", "pre_output_projector", "post_output_projector"}:
        raise ValueError("LongVU frame_selection_site must be pre_sva, pre_output_projector, or post_output_projector.")
    return frame_site


def _layers(values: Mapping[str, Any], defaults: tuple[int, ...], *, num_layers: int | None) -> tuple[int, ...]:
    layers = values.get("layers", defaults)
    if (not isinstance(layers, (list, tuple)) or not layers
            or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in layers)
            or tuple(sorted(set(layers))) != tuple(layers)):
        raise ValueError("Video layers must be non-empty, increasing unique non-negative integers.")
    if num_layers is not None and any(i >= num_layers for i in layers):
        raise ValueError("Video layers exceed the actual model depth.")
    return tuple(layers)


def video_detection_layers(stage: Any, default_layers: tuple[int, ...], num_layers: int | None = None) -> tuple[int, ...]:
    values = _target_operation(stage, "detection_budget_provider")
    if values.get("target_site", "auxiliary_detection_model_attention_budget") != "auxiliary_detection_model_attention_budget":
        raise ValueError("DyTok detection layers capture auxiliary-model Attention, not a Token compression boundary.")
    return _layers(values, default_layers, num_layers=num_layers)


def video_decode_prune_layer(stage: Any, default_layer: int, num_layers: int | None = None, *, backbone: str) -> int:
    values = _target_operation(stage, "decode_kv_view_prune")
    if values.get("target_site", _DECODE_SITES[backbone]) not in {_DECODE_SITES[backbone], "decode_kv_before_attention"}:
        raise ValueError("DyCoKe KV views apply before decode Attention, not at a Token/FFN boundary.")
    layers = _layers(values, (default_layer,), num_layers=num_layers)
    if len(layers) != 1 or layers[0] == 0:
        raise ValueError("DyCoKe requires one first sparse layer >= 1; its previous layer supplies full-cache Attention.")
    return layers[0]


def video_decoder_positions(stage: Any, source_layers: tuple[int, ...], num_layers: int | None = None,
                            *, algorithm: str, backbone: str) -> tuple[LayerPosition, ...]:
    target = {"metok": "prefill_attention_prune", "prunevid": "vtp_prefill_prune"}[algorithm]
    values = dict(_target_operation(stage, target))
    site = values.get("target_site", _DECODE_SITES[backbone])
    if site not in {_DECODE_SITES[backbone], "llm_decoder_layer_output_video_tokens", "llm_decoder_layer_post_attention_video_tokens"}:
        raise ValueError(f"{algorithm} requires a decoder layer boundary, received {site!r}.")
    if site == _DECODE_SITES[backbone] and algorithm == "metok":
        values.setdefault("layer_edge", "before")
    values.setdefault("layers", list(source_layers))
    points = resolve_layer_positions(values, source_layers, num_layers=num_layers,
        boundary="post_attention_pre_ffn" if site == "llm_decoder_layer_post_attention_video_tokens" else "post_layer")
    if len(points) != len(source_layers):
        raise ValueError(f"{algorithm} requires one application position per existing pruning step.")
    # METok's final stage drops all remaining visual tokens without scores.
    score_layers = source_layers[:2] if algorithm == "metok" else source_layers
    if any(source > point.index or (num_layers is not None and source >= num_layers)
           for source, point in zip(score_layers, points)):
        raise ValueError(f"{algorithm} requires each scoring-layer input before its application boundary.")
    return points


def resolve_video_position(*, stage: Any, binding: Any, algorithm: Any = None,
                           variant: Any = None) -> Any:
    effective = variant.resolve_config(getattr(stage, "config", {}) or {}) if variant is not None else None
    targets = []
    for target in binding.targets:
        if target.name in {"prefill_attention_prune", "vtp_prefill_prune"}:
            algorithm = "metok" if target.name == "prefill_attention_prune" else "prunevid"
            config = effective if effective is not None else resolve_canonical_video_config(algorithm, getattr(stage, "config", {}) or {})
            canonical = resolve_canonical_video_config(algorithm)
            sources = tuple(config["pruning_layers"]) if algorithm == "metok" else (config["vtp_selected_layer"],)
            defaults = tuple(canonical["pruning_layers"]) if algorithm == "metok" else (canonical["vtp_selected_layer"],)
            points = video_decoder_positions(stage, sources, algorithm=algorithm, backbone=binding.backbone)
            original = tuple(LayerPosition("inter_layer", layer + (algorithm == "prunevid")) for layer in defaults)
            score_sources = sources[:2] if algorithm == "metok" else sources
            original_scores = defaults[:2] if algorithm == "metok" else defaults
            if points != original or score_sources != original_scores:
                site = "llm_decoder_layer_post_attention_video_tokens" if points[0].boundary == "post_attention_pre_ffn" else "llm_decoder_layer_output_video_tokens"
                target = replace(target, target_site=site, metadata={**target.metadata,
                    "application_positions": [p.as_dict() for p in points],
                    "score_layers": list(score_sources)})
        elif target.name == "qformer_context_content_pool":
            position = video_learned_projector_position(stage, binding.backbone, target_name=target.name)
            if position != "before":
                site = video_learned_projector_site("llamavid", position)
                target = replace(target, target_site=site, metadata={**target.metadata,
                    "application_positions": [{"boundary": site, "projector_owner": "compression_method"}]})
        elif target.name == "dino_frame_select_sva_fusion":
            frame_site = video_longvu_frame_selection_site(stage, binding.backbone)
            if frame_site != "pre_sva":
                target = replace(target, metadata={**target.metadata, "application_positions": [
                    {"operation": "dino_frame_selection", "boundary": frame_site},
                    {"operation": "sva_fusion", "boundary": "longvu_pre_output_projector"}]})
        elif target.name == "detection_budget_provider":
            config = resolve_canonical_video_config("dytok")
            defaults = tuple(config["llm_config"]["target_layers"])
            sources = tuple(effective["llm_config"]["target_layers"]) if effective is not None else defaults
            layers = video_detection_layers(stage, sources)
            if layers != defaults:
                target = replace(target, metadata={**target.metadata, "score_layers": list(layers)})
        elif target.name == "decode_kv_view_prune":
            default = int(resolve_canonical_video_config("dycoke")["decoder_prune_layer"])
            source = int(effective["decoder_prune_layer"]) if effective is not None else default
            layer = video_decode_prune_layer(stage, source, backbone=binding.backbone)
            if layer != default:
                target = replace(target, metadata={**target.metadata, "score_layer": layer - 1,
                    "application_positions": [{"boundary": "decode_kv_before_attention", "index": layer}]})
        else:
            position = video_projector_position(stage, binding.backbone, target_name=target.name)
            site = video_projector_site(binding.backbone, position)
            if site != target.target_site:
                target = replace(target, target_site=site, metadata={**target.metadata, "projector_position": position})
        targets.append(target)
    if tuple(targets) == binding.targets:
        return binding
    return replace(binding, targets=tuple(targets), metadata={**binding.metadata, "position_changed": True})
