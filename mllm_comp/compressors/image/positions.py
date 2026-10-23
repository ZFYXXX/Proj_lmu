"""Physical image boundaries, retaining the steps of compound bindings."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Any

from mllm_comp.compressors.layer_positions import LayerPosition, resolve_layer_positions
from mllm_comp.compressors.image.decoder_config import (
    merge_decoder_executor_config, resolve_decoder_method_config,
)


_STAGE_BINDING_CONFIG = object()
_PROJECTOR_SITES = frozenset({"projector_input", "projector_output", "pre_vision_projector", "post_vision_projector",
    "post_anyres_pack_pre_llm_splice", "pre_vision_merger", "post_vision_merger_pre_llm_scatter",
    "pre_final_merger", "post_final_merger_pre_llm_scatter", "pre_deepstack_mergers",
    "post_deepstack_mergers_pre_deepstack_injection", "post_merger_pre_scatter"})
_TOME_OUTPUT_SITES = {
    "llava_onevision": ("pre_vision_projector", "post_anyres_pack_pre_llm_splice_tome"),
    "qwen25_vl": ("pre_vision_merger", "post_vision_merger_pre_llm_scatter_tome"),
    "qwen3_vl": ("pre_visual_mergers_tome", "post_visual_mergers_pre_llm_scatter_tome"),
}
_TOME_RESTORE_SITES = {
    "llava_onevision": {"dense_restore": "vision_encoder_output_pre_anyres_pack"},
    "qwen25_vl": {"dense_restore": "visual_output_pre_spatial_merger"},
    "qwen3_vl": {"deepstack_dense_restore": "visual_pre_deepstack_mergers",
                 "final_dense_restore": "visual_output_pre_final_merger"},
}


def image_projector_sites(stage: Any, algorithm: str, backbone: str) -> tuple[str, ...]:
    """Resolve existing projector steps; Qwen3 main and DeepStack stay distinct."""
    if algorithm == "saint" and backbone == "qwen25_vl":
        raise ValueError("Qwen2.5-VL SAINT uses encoder positions, not projector positions.")
    if backbone == "qwen3_vl":
        names = {"visionzip": ("main_image_features", "deepstack_image_features"),
            "aim": ("main_image_features", "deepstack_image_features"),
            "saint": ("main_image_features", "deepstack_image_features"),
            "mmtok": ("multisource_main_image_features", "deepstack_image_features"),
            "zoo_prune": ("projector_sensitivity_main_image_features", "deepstack_image_features"),
            "nuwa": ("stage1_regional_visual_aggregation", "stage1_deepstack_aggregation"),
            "restore": ("vispruner_pristine_plus_distinctive_anchor_merge", "shared_deepstack_distinctive_anchor_merge"),
            "vispruner": ("visual_attention_main_image_features", "deepstack_image_features"),
            "divprune": ("main_image_features", "deepstack_image_features"),
            "prumerge": ("main_adaptive_merge", "deepstack_adaptive_merge"),
            "gap": ("main_grounding_aware_prune", "deepstack_grounding_aware_prune")}[algorithm]
        before = ("pre_final_merger", "pre_deepstack_mergers")
        after = ("post_final_merger_pre_llm_scatter", "post_deepstack_mergers_pre_deepstack_injection")
    else:
        name = (("visual_attention_crop_features" if backbone == "llava_onevision" else "visual_attention_merged_image_features") if algorithm == "vispruner" else "adaptive_merge" if algorithm == "prumerge" else "grounding_aware_prune" if algorithm == "gap"
            else "packed_image_features" if backbone == "llava_onevision" else "merged_image_features")
        if algorithm == "mmtok":
            name = "multisource_anyres_features" if backbone == "llava_onevision" else "multisource_merged_image_features"
        elif algorithm == "zoo_prune":
            name = "projector_sensitivity_crop_features" if backbone == "llava_onevision" else "projector_sensitivity_merged_image_features"
        elif algorithm == "nuwa":
            name = "stage1_regional_visual_aggregation"
        elif algorithm == "restore":
            name = "vispruner_pristine_plus_distinctive_anchor_merge"
        names = (name,)
        before = ("pre_vision_projector",) if backbone == "llava_onevision" else ("pre_vision_merger",)
        after = ("post_anyres_pack_pre_llm_splice",) if backbone == "llava_onevision" else ("post_vision_merger_pre_llm_scatter",)
    defaults = after
    if backbone == "llava_onevision" and algorithm in {"nuwa", "restore"}:
        # These legacy callbacks aggregate raw vision values and project the
        # result internally. Their callback name is not a post-value boundary.
        after = ("post_vision_projector",)
    operation = dict(getattr(stage, "operation", {}) or {})
    if algorithm in {"nuwa", "restore"}:
        auxiliary = ("stage2_last_prompt_attention_selection" if algorithm == "nuwa" else
            "token_mass_and_rope_distance_attention_calibration" if backbone == "llava_onevision" else
            "token_mass_and_mrope_distance_attention_calibration")
        targets = operation.get("targets", {})
        if not isinstance(targets, Mapping) or set(targets) - {*names, auxiliary}:
            raise ValueError(f"{algorithm} positions must name its existing compound targets.")
        if algorithm == "restore" and auxiliary in targets:
            calibration = targets[auxiliary]
            if (not isinstance(calibration, Mapping) or set(calibration) - {"target_site"}
                    or calibration.get("target_site", "llm_decoder_attention_calibration") != "llm_decoder_attention_calibration"):
                raise ValueError("RESTORE calibration modifies every decoder layer's attention logits; it is not a token-compression boundary.")
        operation["targets"] = {name: value for name, value in targets.items() if name in names}
        if algorithm == "nuwa" and operation.get("target_site") not in _PROJECTOR_SITES:
            for key in ("target_site", "layers", "layer_edge"):
                operation.pop(key, None)
    if "layers" in operation or "layer_edge" in operation:
        raise ValueError("Projector image positions do not accept layer indices or layer_edge.")
    targets = operation.get("targets", {})
    if not isinstance(targets, Mapping) or set(targets) - set(names):
        raise ValueError(f"{algorithm} projector targets must be drawn from {names}.")
    if "target_site" in operation and targets:
        raise ValueError("Use either operation.target_site or per-target projector sites.")
    shared = operation.get("target_site")
    if backbone == "qwen3_vl" and shared == "post_merger_pre_scatter":
        shared = "projector_output"
    if shared in before:
        shared = "projector_input"
    elif shared in after:
        shared = "projector_output"
    sites = []
    for name, input_site, output_site, default_site in zip(names, before, after, defaults):
        scoped = targets.get(name, {})
        if not isinstance(scoped, Mapping) or set(scoped) - {"target_site"}:
            raise ValueError(f"Projector target {name} accepts target_site only.")
        site = shared if shared is not None else scoped.get("target_site", default_site)
        if algorithm == "visionzip" and backbone == "qwen25_vl" and site == "post_encoder_pre_llm":
            # Existing forward-probe YAML names the original post-merger callback.
            site = default_site
        site = input_site if site == "projector_input" else output_site if site == "projector_output" else site
        if site not in {input_site, output_site, default_site}:
            raise ValueError(f"{algorithm} projector target {name} cannot execute at {site!r}.")
        sites.append(site)
    return tuple(sites)


def _target_operation(stage: Any, target_name: str, *, other_targets: tuple[str, ...] = ()) -> dict[str, Any]:
    operation = getattr(stage, "operation", {}) or {}
    targets = operation.get("targets", {})
    if not isinstance(targets, Mapping) or set(targets) - {target_name, *other_targets}:
        raise ValueError(f"operation.targets must name {target_name} or another existing compound step.")
    scoped = targets.get(target_name, {})
    if not isinstance(scoped, Mapping):
        raise ValueError(f"operation.targets.{target_name} must be a mapping.")
    keys = {"target_site", "layers", "layer_edge"}
    if set(scoped) - keys:
        raise ValueError(f"Unknown {target_name} position fields.")
    values = {key: operation[key] for key in keys if key in operation}
    if target_name == "stage2_last_prompt_attention_selection" and values.get("target_site") in _PROJECTOR_SITES:
        values = {}
    if values and scoped:
        raise ValueError(f"Use either operation position fields or {target_name} fields.")
    values.update(scoped)
    return values


def saint_encoder_operation(stage: Any) -> dict[str, Any]:
    values = _target_operation(stage, "encoder_progressive_prune")
    default = "visual_block_output_progressive_patch_prune"
    values.setdefault("target_site", default)
    if values["target_site"] == "visual_block_output":
        values["target_site"] = default
    sites = {default: "post_layer", "visual_block_post_attention_residual_pre_mlp": "post_attention_pre_ffn"}
    if not isinstance(values["target_site"], str) or values["target_site"] not in sites:
        raise ValueError("SAINT encoder positions require a visual layer boundary.")
    resolve_layer_positions(values, (0,), boundary=sites[values["target_site"]])
    return values


def saint_encoder_positions(stage: Any, layer_indices: Any, num_layers: int | None = None) -> tuple[LayerPosition, ...]:
    """Keep each norm/QKV module while applying its rule to the chosen values."""
    sources = tuple(layer_indices)
    if (not sources or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in sources)
            or sources != tuple(sorted(set(sources)))):
        raise ValueError("SAINT key projection layers must be unique increasing non-negative integers.")
    if num_layers is not None and any(i >= num_layers for i in sources):
        raise ValueError("SAINT key projection layer is outside the visual stack.")
    values = saint_encoder_operation(stage)
    points = resolve_layer_positions(values, sources, num_layers=num_layers,
        boundary="post_attention_pre_ffn" if values["target_site"] == "visual_block_post_attention_residual_pre_mlp" else "post_layer")
    if len(points) != len(sources):
        raise ValueError("SAINT application positions must match the progressive step count.")
    return points


def tome_output_site(stage: Any, backbone: str) -> str:
    """Resolve the independent output merge, leaving structural restores fixed."""
    targets = (getattr(stage, "operation", {}) or {}).get("targets", {})
    restores = _TOME_RESTORE_SITES[backbone]
    if not isinstance(targets, Mapping) or set(targets) - {"encoder_merge", "permanent_output_merge", *restores}:
        raise ValueError("ToMe operation.targets must name its existing compound steps.")
    for name, fixed in restores.items():
        values = targets.get(name, {})
        if not isinstance(values, Mapping) or set(values) - {"target_site"} or values.get("target_site", fixed) != fixed:
            raise ValueError(f"ToMe {name} restores the native layout before its structural consumer at {fixed}.")
    values = targets.get("permanent_output_merge", {})
    if not isinstance(values, Mapping) or set(values) - {"target_site"}:
        raise ValueError("ToMe permanent_output_merge accepts a Projector target_site only.")
    before, after = _TOME_OUTPUT_SITES[backbone]
    site = values.get("target_site", after)
    if site == "projector_input":
        site = before
    elif site == "projector_output":
        site = after
    if site not in (before, after):
        raise ValueError(f"ToMe permanent_output_merge cannot execute at {site!r}.")
    return site


def tome_encoder_operation(stage: Any, backbone: str) -> dict[str, Any]:
    tome_output_site(stage, backbone)
    values = _target_operation(stage, "encoder_merge", other_targets=("permanent_output_merge", *_TOME_RESTORE_SITES[backbone]))
    prefix = "vision" if backbone == "llava_onevision" else "visual"
    default = f"{prefix}_block_post_attention_residual_pre_mlp"
    values.setdefault("target_site", default)
    # Existing Qwen2.5-VL YAML recipes use this generic vision spelling.
    if values["target_site"] == "vision_block" or (
        backbone != "llava_onevision"
        and values["target_site"] == "vision_block_post_attention_residual_pre_mlp"
    ):
        values["target_site"] = default
    sites = {default: "post_attention_pre_ffn", f"{prefix}_block_output": "post_layer"}
    site = values["target_site"]
    if not isinstance(site, str) or site not in sites:
        raise ValueError(f"ToMe encoder execution is not implemented at {site!r}.")
    # Validate syntax even before a model supplies its depth.
    resolve_layer_positions(values, (0,), boundary=sites[site])
    return values


def decoder_token_operation(stage: Any, algorithm: str = "fastv", *, backbone: str | None = None) -> dict[str, Any]:
    target_name = "decoder_guide_prune_boundary" if algorithm == "sgl" else "decoder_prune_boundary"
    default_site = "llm_decoder_pre_prune_layer_image_tokens"
    if algorithm == "ivc_prune":
        target_name = "native_rope_coordinate_prune" if backbone == "llava_onevision" else "native_mrope_coordinate_prune"
        if backbone != "llava_onevision":
            default_site = "llm_decoder_pre_ivc_prune_layer_image_tokens"
    elif algorithm == "feather":
        target_name = "two_stage_no_rope_prune"
        if backbone != "llava_onevision":
            default_site = "llm_decoder_pre_feather_stage_image_tokens"
    elif algorithm == "nuwa":
        target_name = "stage2_last_prompt_attention_selection"
        if backbone != "llava_onevision":
            default_site = "llm_decoder_pre_stage2_layer_image_tokens"
    elif algorithm == "btp":
        target_name = "decoder_balanced_pruning_boundaries"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "halfv":
        target_name = "decoder_ivr_ssr_boundaries"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "transprune":
        target_name = "decoder_ttv_iga_boundaries"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "finepruner":
        target_name = "decoder_iac_arp_boundaries"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "pyramid_drop":
        target_name = "decoder_stage_boundary"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "pact":
        target_name = "decoder_pact_boundary"
        default_site = "llm_decoder_layer_output_image_tokens"
    elif algorithm == "sparsevlm":
        target_name = "decoder_merge_boundaries"
        default_site = "llm_decoder_layer_output_image_tokens"
    others = (("stage1_regional_visual_aggregation", "stage1_deepstack_aggregation") if backbone == "qwen3_vl"
        else ("stage1_regional_visual_aggregation",)) if algorithm == "nuwa" else ()
    values = _target_operation(stage, target_name, other_targets=others)
    if "layer_edge" in values and "layers" not in values:
        raise ValueError("operation.layer_edge requires explicit operation.layers.")
    values.setdefault("target_site", default_site)
    site = values["target_site"]
    if algorithm == "pyramid_drop" and backbone == "qwen25_vl" and site == "llm_decoder_layer_loop":
        # The legacy loop callback applies each stage after its complete layer.
        site = values["target_site"] = default_site
    if site == default_site and algorithm not in {"btp", "halfv", "pyramid_drop", "finepruner", "transprune", "pact", "sparsevlm"}:
        if values.get("layer_edge", "before") != "before":
            raise ValueError("A pre-layer target requires layer_edge=before.")
        values["layer_edge"] = "before"
    elif site not in {"llm_decoder_layer_output_image_tokens", "llm_decoder_layer_post_attention_image_tokens"}:
        raise ValueError(f"{algorithm} execution is not implemented at {site!r}.")
    return values


def decoder_token_position(stage: Any, prune_layer: int, num_layers: int | None = None, *, algorithm: str = "fastv", backbone: str | None = None) -> LayerPosition:
    values = decoder_token_operation(stage, algorithm, backbone=backbone)
    points = resolve_layer_positions({**values, "layers": values.get("layers", [prune_layer])}, (),
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer", num_layers=num_layers)
    if len(points) != 1:
        raise ValueError(f"{algorithm} requires exactly one application boundary.")
    point = points[0]
    if algorithm in {"dart", "ivc_prune"} and prune_layer > point.index:
        raise ValueError(f"{algorithm.upper()} requires its completed scoring layer before the application boundary.")
    if algorithm != "sgl" and num_layers is not None and prune_layer > 0:
        source = prune_layer - 1
        if source >= num_layers or source > point.index:
            raise ValueError(f"{algorithm}'s configured scoring-layer input is unavailable at this boundary.")
    return point


def pact_decoder_position(stage: Any, spec: Any, num_layers: int | None = None) -> LayerPosition:
    values = decoder_token_operation(stage, "pact")
    points = resolve_layer_positions(values, (spec.boundary_layer,),
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != 1:
        raise ValueError("PACT requires exactly one application boundary.")
    point = points[0]
    if point.index <= spec.boundary_layer or (num_layers is not None and spec.reduction_layer >= num_layers):
        raise ValueError("PACT requires its completed scoring layer and the next layer's Q/K weights before application.")
    return point


def feather_decoder_positions(stage: Any, rule_layers: tuple[int, ...], num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "feather", backbone=backbone)
    points = resolve_layer_positions({**values, "layers": values.get("layers", list(rule_layers))}, (),
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(rule_layers):
        raise ValueError("FEATHER requires one application boundary for each of its two stages.")
    if any(source > point.index or (num_layers is not None and source > num_layers)
           for source, point in zip(rule_layers, points)):
        raise ValueError("FEATHER requires each completed scoring layer before its application boundary.")
    return points


def sparsevlm_decoder_positions(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "sparsevlm", backbone=backbone)
    points = resolve_layer_positions(values, spec.prune_layers,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(spec.prune_layers):
        raise ValueError("SparseVLM requires one application boundary per select/recycle stage.")
    if any(source > point.index or (num_layers is not None and source >= num_layers)
           for source, point in zip(spec.prune_layers, points)):
        raise ValueError("SparseVLM requires each configured scoring-layer input before application.")
    return points


def nuwa_decoder_position(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> LayerPosition:
    values = decoder_token_operation(stage, "nuwa", backbone=backbone)
    points = resolve_layer_positions({**values, "layers": values.get("layers", [spec.stage2_after_layer])}, (),
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != 1:
        raise ValueError("Nüwa Stage 2 requires exactly one application boundary.")
    point = points[0]
    source = spec.stage2_score_layer
    if source is not None:
        if source > point.index or (num_layers is not None and source >= num_layers):
            raise ValueError("Nüwa Stage 2 scoring-layer input is unavailable at this boundary.")
    return point


def btp_decoder_positions(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "btp", backbone=backbone)
    points = resolve_layer_positions(values, spec.boundary_after_layer_indices,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(spec.prune_before_layers):
        raise ValueError("BTP requires one application boundary per stage, including FinalWipe.")
    if any(source > point.index or (num_layers is not None and source >= num_layers)
           for source, point in zip(spec.boundary_after_layer_indices[:-1], points[:-1])):
        raise ValueError("BTP scoring-layer input is unavailable at its application boundary.")
    return points


def halfv_decoder_positions(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "halfv", backbone=backbone)
    points = resolve_layer_positions(values, spec.boundary_after_layer_indices,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != 2:
        raise ValueError("HalfV requires one application boundary for IVR and one for SSR.")
    sources = spec.boundary_after_layer_indices
    if (points[0].index <= sources[0] or points[1].index < sources[1]
            or (num_layers is not None and sources[1] >= num_layers)):
        raise ValueError("HalfV needs the completed IVR scoring layer and the SSR scoring-layer input before application.")
    return points


def transprune_decoder_positions(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "transprune", backbone=backbone)
    points = resolve_layer_positions(values, spec.boundary_after_layer_indices,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(spec.prune_after_layers):
        raise ValueError("TransPrune requires one application boundary per pruning stage.")
    if num_layers is not None:
        spec.validate_decoder_depth(num_layers)
    if any(point.index <= source for source, point in zip(spec.boundary_after_layer_indices, points)):
        raise ValueError("TransPrune requires the completed TTV/IGA scoring layer before application.")
    return points


def finepruner_decoder_positions(stage: Any, spec: Any, num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "finepruner", backbone=backbone)
    defaults = tuple(i - 1 for i in (*spec.iac_after_layers, spec.arp_apply_after_layer))
    points = resolve_layer_positions(values, defaults,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(defaults):
        raise ValueError("FinePruner requires one boundary per IAC stage and one for ARP.")
    # IAC hashes the features at application. ARP consumes the configured
    # Attention sources; its last source's input must already be available.
    source = spec.arp_apply_after_layer - 1
    if points[-1].index < source or (num_layers is not None and source >= num_layers):
        raise ValueError("FinePruner ARP requires every configured scoring-layer input before application.")
    return points


def pyramid_decoder_positions(stage: Any, rule_layers: tuple[int, ...], num_layers: int | None = None, *, backbone: str) -> tuple[LayerPosition, ...]:
    values = decoder_token_operation(stage, "pyramid_drop", backbone=backbone)
    points = resolve_layer_positions(values, rule_layers,
        boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer",
        num_layers=num_layers)
    if len(points) != len(rule_layers):
        raise ValueError("PyramidDrop requires one application boundary for each stage.")
    if any(point.index <= source or (num_layers is not None and source >= num_layers)
           for source, point in zip(rule_layers, points)):
        raise ValueError("PyramidDrop requires the completed scoring layer before application.")
    return points


def tome_encoder_positions(
    stage: Any, backbone: str, num_layers: int | None,
) -> tuple[LayerPosition, ...]:
    values = tome_encoder_operation(stage, backbone)
    boundary = "post_layer" if values["target_site"].endswith("_output") else "post_attention_pre_ffn"
    defaults = tuple(range(num_layers)) if num_layers is not None else ()
    return resolve_layer_positions(values, defaults, boundary=boundary, num_layers=num_layers)


def tome_legacy_budget_layers(config: Mapping[str, Any]) -> tuple[int, ...] | None:
    layers = config.get("layers")
    if layers is None and isinstance(config.get("r_by_layer"), Mapping):
        layers = config["r_by_layer"]
    return tuple(sorted({int(i) for i in layers})) if layers is not None else None


def tome_encoder_score_layer(
    stage: Any, backbone: str, num_layers: int | None = None, *, config: Mapping[str, Any] | None = None,
) -> int | None:
    """An explicit source is captured at that layer's input, before compression."""
    criterion = getattr(stage, "criterion", {}) or {}
    if "score_layer" not in criterion:
        return None
    source = criterion["score_layer"]
    if isinstance(source, bool) or not isinstance(source, int) or source < 0:
        raise ValueError("ToMe criterion.score_layer must be a non-negative integer.")
    if num_layers is not None and source >= num_layers:
        raise ValueError("ToMe score_layer is outside the vision encoder.")
    # LLaVA public binding selections can override the legacy layer budgets.
    # Check their availability once the provider has the final config and depth.
    if backbone == "llava_onevision" and num_layers is None:
        return source
    operation = tome_encoder_operation(stage, backbone)
    positions = tome_encoder_positions(stage, backbone, num_layers) if num_layers is not None or "layers" in operation else (LayerPosition("post_attention_pre_ffn", 0),)
    if backbone == "llava_onevision":
        config = config if config is not None else (getattr(stage, "config", {}) or {})
        legacy = tome_legacy_budget_layers(config)
        if "layers" not in operation and legacy is not None:
            positions = resolve_layer_positions({**operation, "layers": list(legacy)}, (),
                boundary="post_layer" if operation["target_site"].endswith("_output") else "post_attention_pre_ffn") if legacy else ()
        budgets = config.get("r_by_layer")
        rule_layers = legacy if legacy is not None else tuple(p.index if p.boundary == "post_attention_pre_ffn" else max(0, p.index - 1) for p in positions)
        positions = tuple(p for p, layer in zip(positions, rule_layers) if (
            budgets.get(layer, budgets.get(str(layer), 0)) if isinstance(budgets, Mapping) else config.get("r", 0)
        ) != 0)
    for position in positions:
        available = position.index
        if source > available:
            raise ValueError("ToMe score_layer is a future layer at the requested application boundary.")
    return source


def tome_position_metadata(stage: Any, backbone: str) -> dict[str, Any]:
    values = tome_encoder_operation(stage, backbone)
    return {"encoder_merge_position": values, "encoder_score_layer": tome_encoder_score_layer(stage, backbone),
            "permanent_output_merge_site": tome_output_site(stage, backbone)}


def resolve_image_position(*, stage: Any, binding: Any, binding_config: Any = _STAGE_BINDING_CONFIG, algorithm: Any = None, variant: Any = None) -> Any:
    """Change the selected step only; structural restoration remains ordered."""
    if algorithm is not None:
        algorithm_name = algorithm.name
    elif variant is not None:
        algorithm_name = variant.algorithm
    else:
        from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts

        algorithm_name = builtin_multimodal_compression_contracts().variant(binding.variant).algorithm
    # A pure stage projection reads its selection. Public factories pass their
    # actual argument, including None, which means no binding-local override.
    if binding_config is _STAGE_BINDING_CONFIG:
        binding_config = next((item for item in (getattr(stage, "bindings", ()) or ())
                               if item.binding == binding.name), None)
    method_config = None
    if variant is not None and binding.strategy_factory is not None:
        method_config = resolve_decoder_method_config(
            stage=stage, binding=binding, variant=variant, binding_config=binding_config)
    if binding.variant in {"visionzip.image", "feature_anchor_mean.image", "grid_mean.image", "prumerge.image", "gap.image", "divprune.image", "aim.image", "vispruner.image", "saint.image.packed_feature_prune", "mmtok.image", "zoo_prune.image"}:
        sites = image_projector_sites(stage, binding.variant.split(".")[0], binding.backbone)
        if sites == tuple(target.target_site for target in binding.targets):
            return binding
        return replace(binding, targets=tuple(replace(target, target_site=site,
            metadata={**target.metadata, "projector_position": "before" if site.startswith("pre_") else "after"})
            for target, site in zip(binding.targets, sites)), metadata={**binding.metadata, "position_changed": True})
    if binding.variant == "saint.image.encoder_progressive_prune":
        sources = (15, 23)
        for config in (*(getattr(stage, name, None) for name in ("operation", "criterion", "schedule", "config")), getattr(binding_config, "config", None)):
            if isinstance(config, Mapping) and "layer_indices" in config:
                sources = config["layer_indices"]
        points = saint_encoder_positions(stage, sources)
        values = saint_encoder_operation(stage)
        target = binding.targets[0]
        changed = points != tuple(LayerPosition("inter_layer", index + 1) for index in sources)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata={
            **target.metadata, "application_positions": [point.as_dict() for point in points],
            "key_projection_layers": list(sources),
        }),), metadata={**binding.metadata, "position_changed": changed})
    if binding.variant == "pact.image":
        from mllm_comp.compressors.image.pact import PACTSpec

        spec = PACTSpec(**merge_decoder_executor_config("pact", stage=stage, binding_config=binding_config))
        point = pact_decoder_position(stage, spec)
        values = decoder_token_operation(stage, "pact")
        target = binding.targets[0]
        changed = point != LayerPosition("inter_layer", spec.reduction_layer)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata={
            **target.metadata, "application_positions": [point.as_dict()], "score_layers": [spec.boundary_layer],
        }),), metadata={**binding.metadata, "position_changed": changed})
    if algorithm_name == "pyramid_drop":
        values = decoder_token_operation(stage, "pyramid_drop", backbone=binding.backbone)
        target = binding.targets[0]
        if values["target_site"] == target.target_site and "layers" not in values:
            return binding
        metadata = dict(target.metadata)
        rule_layers = None
        if binding.backbone != "llava_onevision":
            if method_config is not None:
                rule_layers = method_config.get("layer_indices")
            else:
                for config in (*(getattr(stage, name, None) for name in ("operation", "criterion", "schedule", "config")), getattr(binding_config, "config", None)):
                    if isinstance(config, Mapping) and "layer_indices" in config:
                        rule_layers = config["layer_indices"]
        if rule_layers is not None:
            rule_layers = tuple(rule_layers)
            if (not rule_layers or any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in rule_layers)
                    or tuple(sorted(set(rule_layers))) != rule_layers):
                raise ValueError("PyramidDrop scoring layers must be unique increasing non-negative integers.")
            points = pyramid_decoder_positions(stage, rule_layers, backbone=binding.backbone)
            metadata.update(application_positions=[point.as_dict() for point in points], score_layers=list(rule_layers))
            changed = points != tuple(LayerPosition("inter_layer", i + 1) for i in rule_layers)
        elif "layers" in values:
            points = resolve_layer_positions(values, (), boundary="post_attention_pre_ffn" if values["target_site"] == "llm_decoder_layer_post_attention_image_tokens" else "post_layer")
            metadata["application_positions"] = [point.as_dict() for point in points]
            changed = True
        else:
            changed = True
        # Only the uniform default needs a loaded decoder depth for comparison.
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata=metadata),),
            metadata={**binding.metadata, "position_changed": changed})
    if algorithm_name in {"btp", "halfv", "transprune"} or binding.variant in {"finepruner.image", "sparsevlm.image"}:
        from mllm_comp.compressors.image.btp import BalancedTokenPruningSpec
        from mllm_comp.compressors.image.halfv import HalfVSpec
        from mllm_comp.compressors.image.finepruner import FinePrunerSpec
        from mllm_comp.compressors.image.transprune import TransPruneSpec
        from mllm_comp.compressors.image.sparsevlm import SparseVLMSpec

        algorithm = algorithm_name
        spec_type, resolve = {
            "btp": (BalancedTokenPruningSpec, btp_decoder_positions),
            "halfv": (HalfVSpec, halfv_decoder_positions),
            "finepruner": (FinePrunerSpec, finepruner_decoder_positions),
            "transprune": (TransPruneSpec, transprune_decoder_positions),
            "sparsevlm": (SparseVLMSpec, sparsevlm_decoder_positions),
        }[algorithm]
        config = method_config if method_config is not None else merge_decoder_executor_config(
            algorithm, stage=stage, binding_config=binding_config)
        spec = spec_type(**config)
        values = decoder_token_operation(stage, algorithm, backbone=binding.backbone)
        points = resolve(stage, spec, backbone=binding.backbone)
        target = binding.targets[0]
        rule_layers = (spec.prune_layers if algorithm == "sparsevlm" else
            tuple(i - 1 for i in (*spec.iac_after_layers, spec.arp_apply_after_layer)) if algorithm == "finepruner" else spec.boundary_after_layer_indices)
        score_layers = tuple(i - 1 for i in spec.arp_target_after_layers) if algorithm == "finepruner" else (rule_layers[:-1] if algorithm == "btp" else rule_layers)
        changed = points != tuple(LayerPosition("inter_layer", i + 1) for i in rule_layers)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata={
            **target.metadata, "application_positions": [point.as_dict() for point in points],
            "score_layers": list(score_layers),
        }),), metadata={**binding.metadata, "position_changed": changed})
    if binding.variant == "restore.image":
        sites = image_projector_sites(stage, "restore", binding.backbone)
        if sites == tuple(target.target_site for target in binding.targets[:len(sites)]):
            return binding
        return replace(binding, targets=tuple(replace(target, target_site=sites[index], metadata={
            **target.metadata, "projector_position": "before" if sites[index].startswith("pre_") else "after",
        }) if index < len(sites) else target for index, target in enumerate(binding.targets)),
            metadata={**binding.metadata, "position_changed": True})
    if binding.variant == "nuwa.image":
        from mllm_comp.compressors.image.nuwa import NuwaSpec, merge_nuwa_executor_config

        spec = NuwaSpec(**merge_nuwa_executor_config(stage=stage, binding_config=binding_config))
        values = decoder_token_operation(stage, "nuwa", backbone=binding.backbone)
        point = nuwa_decoder_position(stage, spec, backbone=binding.backbone)
        target = next(target for target in binding.targets if target.name == "stage2_last_prompt_attention_selection")
        sites = image_projector_sites(stage, "nuwa", binding.backbone)
        projector_changed = sites != tuple(item.target_site for item in binding.targets[:len(sites)])
        changed = projector_changed or point != LayerPosition("inter_layer", spec.stage2_after_layer)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=tuple(replace(item, target_site=values["target_site"], metadata={
            **item.metadata, "application_positions": [point.as_dict()], "score_layer": spec.stage2_score_layer,
        }) if item.name == target.name else replace(item, target_site=sites[index], metadata={
            **item.metadata, "projector_position": "before" if sites[index].startswith("pre_") else "after",
        }) if index < len(sites) else item for index, item in enumerate(binding.targets)),
            metadata={**binding.metadata, "position_changed": changed})
    if binding.variant == "feather.image":
        from mllm_comp.compressors.image.feather import FEATHERSpec

        config = merge_decoder_executor_config("feather", stage=stage, binding_config=binding_config)
        config.pop("single_image_only", None)
        if "stage_layers" in config:
            config["stage_layers"] = tuple(int(value) for value in config["stage_layers"])
        spec = FEATHERSpec(**config)
        values = decoder_token_operation(stage, "feather", backbone=binding.backbone)
        points = feather_decoder_positions(stage, spec.stage_layers, backbone=binding.backbone)
        target = binding.targets[0]
        changed = points != tuple(LayerPosition("inter_layer", i) for i in spec.stage_layers)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata={
            **target.metadata, "application_positions": [point.as_dict() for point in points],
            "score_layers": [i - 1 for i in spec.stage_layers],
        }),), metadata={**binding.metadata, "position_changed": changed})
    if binding.variant == "sgl.image":
        from mllm_comp.compressors.image.sgl import SGLSpec

        values = decoder_token_operation(stage, "sgl")
        target = binding.targets[0]
        if values["target_site"] == target.target_site and "layers" not in values:
            return binding
        spec = SGLSpec(**merge_decoder_executor_config("sgl", stage=stage, binding_config=binding_config))
        metadata = {**target.metadata, "prune_layer_fraction": spec.prune_layer_fraction}
        if "layers" in values:
            point = decoder_token_position(stage, 0, algorithm="sgl")
            metadata["application_positions"] = [point.as_dict()]
        # The fractional default requires the loaded target's decoder depth.
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata=metadata),),
            metadata={**binding.metadata, "position_changed": True})
    if algorithm_name == "fastv" or binding.variant in {"dart.image", "d2pruner.image", "ivc_prune.image"}:
        from mllm_comp.compressors.image.fastv import FastVSpec
        from mllm_comp.compressors.image.dart import DARTSpec
        from mllm_comp.compressors.image.d2pruner import D2PrunerSpec
        from mllm_comp.compressors.image.ivc_prune import IVCPruneSpec

        algorithm = algorithm_name
        values = decoder_token_operation(stage, algorithm, backbone=binding.backbone)
        config = method_config if method_config is not None else merge_decoder_executor_config(
            algorithm, stage=stage, binding_config=binding_config)
        if algorithm == "ivc_prune":
            config.pop("single_image_only", None)
            prune_layer = IVCPruneSpec(**config).prune_layer
        else:
            prune_layer = FastVSpec(**config).prune_layer if algorithm == "fastv" else (DARTSpec(**config) if algorithm == "dart" else D2PrunerSpec(**config)).prune_after_layer
        point = decoder_token_position(stage, prune_layer, algorithm=algorithm, backbone=binding.backbone)
        target = binding.targets[0]
        changed = point != LayerPosition("inter_layer", prune_layer)
        if values["target_site"] == target.target_site and not changed and "layers" not in values:
            return binding
        return replace(binding, targets=(replace(target, target_site=values["target_site"], metadata={
            **target.metadata, "application_positions": [point.as_dict()],
            "score_layer": prune_layer - 1 if prune_layer else None,
        }),), metadata={**binding.metadata, "position_changed": changed})
    if algorithm_name != "tome":
        raise ValueError(
            f"The existing image position resolver has no rule for algorithm {algorithm_name!r}; "
            "provide a method-specific position_resolver or use a fixed binding without one."
        )
    values = tome_encoder_operation(stage, binding.backbone)
    source = tome_encoder_score_layer(stage, binding.backbone)
    target = next(t for t in binding.targets if t.name == "encoder_merge")
    metadata = dict(target.metadata)
    if source is not None:
        metadata["score_layer"] = source
    layers_changed = "layers" in values
    if layers_changed:
        boundary = "post_layer" if values["target_site"].endswith("_output") else "post_attention_pre_ffn"
        points = resolve_layer_positions(values, (), boundary=boundary)
        metadata["application_positions"] = [p.as_dict() for p in points]
        if binding.backbone == "llava_onevision":
            config = {**(getattr(stage, "config", {}) or {}), **(getattr(binding_config, "config", {}) or {})}
            legacy_layers = tome_legacy_budget_layers(config)
            if legacy_layers is not None:
                layers_changed = points != tuple(LayerPosition("post_attention_pre_ffn", i) for i in legacy_layers)
    output_site = tome_output_site(stage, binding.backbone)
    output_target = next(t for t in binding.targets if t.name == "permanent_output_merge")
    changed = (values["target_site"] != target.target_site or layers_changed or source is not None
               or output_site != output_target.target_site)
    if not changed and "layers" not in values:
        return binding
    return replace(binding, targets=tuple(
        replace(t, target_site=values["target_site"], metadata=metadata) if t.name == "encoder_merge" else
        replace(t, target_site=output_site, metadata={**t.metadata,
            "projector_position": "before" if output_site.startswith("pre_") else "after"})
        if t.name == "permanent_output_merge" and output_site != t.target_site else t
        for t in binding.targets
    ), metadata={**binding.metadata, "position_changed": changed})
