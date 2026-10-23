"""Comparison decisions consumed by the existing native decoder lifecycle."""
from dataclasses import replace
from types import SimpleNamespace

from mllm_comp.compressors.image.decoder_selection import DecoderSelectionCall, AttentionCapture
from mllm_comp.compressors.image.decoder_lifecycle import CacheRepairPolicy
from mllm_comp.compressors.image.token_selection import ImageTokenSelection, ImageTokenSelectionResult
from mllm_comp.compressors.layer_positions import LayerPosition


class VLMPrunerMethod:
    algorithm = "vlm_pruner"
    cache_policy = CacheRepairPolicy.COMPACT_PREFIX
    position_policy = "sparse_source"
    single_image_only = True
    target_site = "llm_decoder_pre_prune_layer_image_tokens"
    legacy_config_keys = ()

    def __init__(self, *, stage, config, rules, backbone):
        from .vlm_pruner import normalize_vlm_pruner_config
        self.config = normalize_vlm_pruner_config(config)
        self.spec = SimpleNamespace(**self.config)
        self.rules = rules
        self.resolve_depth(None)

    def resolve_depth(self, depth):
        boundary = self.config["prune_after_layer"]
        if depth is not None and boundary >= depth:
            raise ValueError("VLM-Pruner needs a layer after its pruning boundary")
        self.positions = (LayerPosition("inter_layer", boundary),)
        self.score_layers = (boundary - 1,)

    def new_call(self):
        return VLMPrunerCall(self)


class VLMPrunerCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.keys = None
        self.segments = None

    def set_visual_layout(self, segments):
        self.segments = segments

    def capture_input(self, layer_index, image_mask, *, padding_mask, labels):
        if layer_index == self.method.score_layers[0]:
            import torch
            return AttentionCapture(layer_index, layer_index,
                torch.tensor([image_mask.numel() - 1], device=image_mask.device))

    def record_attention(self, capture, *, attention_keys, **unused):
        self.keys = attention_keys.flatten(1).unsqueeze(0)

    def select(self, position, hidden_states, *, image_indices, protected_indices=(), **kwargs):
        import torch
        from .vlm_pruner import build_vlm_pruner_plan
        if self.keys is None or self.segments is None:
            raise ValueError("VLM-Pruner requires native keys and image spatial segments")
        start, end = int(image_indices[0]), int(image_indices[-1]) + 1
        if end - start != image_indices.numel():
            raise ValueError("VLM-Pruner requires one contiguous visual span")
        plan = build_vlm_pruner_plan(hidden_states.unsqueeze(0), self.keys,
            visual_span=(start, end), cfg=self.method.config, spatial_segments=self.segments)
        # Like the supplied decoder port, the visual span includes serialized
        # newlines. The provider preserves only surviving image-newline tokens.
        result = ImageTokenSelectionResult.from_selection(
            ImageTokenSelection(start + plan["selected_visual_indices"], metadata={"algorithm": "vlm_pruner"}),
            hidden_states, image_indices=image_indices, protected_indices=(), **kwargs)
        rows = [[[int(index)], [1.0]] for index in result.kept_indices.tolist()]
        row_for_source = {source: row for row, source in enumerate(result.kept_indices.tolist())}
        weights = plan["fusion_weights"].detach()
        for column, representative in enumerate(plan["selection_order"].tolist()):
            active = (weights[:, column] != 0).nonzero(as_tuple=True)[0]
            if active.numel():
                members = (start + plan["discarded_visual_indices"].index_select(0, active)).tolist()
                coefficients = (self.method.config["aggregated_state_weight"] * weights[active, column]).tolist()
                rows[row_for_source[start + representative]] = [
                    [start + representative, *members],
                    [self.method.config["retained_state_weight"], *coefficients]]
        delta = replace(result.layout_delta,
            result_layout={**dict(result.layout_delta.result_layout), "weighted_replay_rows": rows})
        return replace(result, hidden_states=plan["hidden_states"][0], layout_delta=delta)


def llava_visual_segments(*, count, newlines, side, merge_type):
    """Describe actual base/local grids; never infer a square AnyRes sequence."""
    patch_count = side * side
    if merge_type == "flat":
        if count % patch_count:
            raise ValueError("Flat OneVision visual layout is not a complete crop sequence")
        return [{"start": start, "end": start + patch_count, "grid_width": side}
                for start in range(0, count, patch_count)]
    segments = [{"start": 0, "end": patch_count, "grid_width": side}]
    if count > patch_count:
        tail = [i for i in newlines if i >= patch_count]
        if tail:
            width = tail[0] - patch_count + 1
        elif (count - patch_count) % side == 0:
            width = side
        else:
            raise ValueError("VLM-Pruner needs native AnyRes row boundaries")
        segments.append({"start": patch_count, "end": count, "grid_width": width})
    return segments


class CoreMatchingMethod(VLMPrunerMethod):
    algorithm = "core_matching"

    def __init__(self, *, stage, config, rules, backbone):
        from .core_matching import normalize_core_matching_config
        self.config = normalize_core_matching_config(config)
        self.spec = SimpleNamespace(**self.config)
        self.rules = rules
        self.runtime = None
        self.resolve_depth(None)

    def new_call(self):
        return CoreMatchingCall(self)


class CoreMatchingCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.runtime = method.runtime

    def initialize(self, input_embeddings, image_mask, *, padding_mask, labels):
        self.runtime = {"has_image": bool(image_mask.any())}
        self.method.runtime = self.runtime

    def capture_input(self, layer_index, image_mask, **kwargs):
        indices = image_mask.nonzero(as_tuple=True)[0]
        self.runtime["expanded_visual_spans"] = [[int(indices[0]), int(indices[-1]) + 1]]

    def select(self, position, hidden_states, *, image_indices, **kwargs):
        from .core_matching import build_core_token_plan
        plan = build_core_token_plan(sequence_length=len(hidden_states),
            visual_indices=self.runtime["_core_matching_visual_indices"],
            visual_scores=self.runtime["_core_matching_visual_scores"],
            minimum_visual_retain_ratio=self.method.config["minimum_visual_retain_ratio"],
            maximum_visual_retain_ratio=self.method.config["maximum_visual_retain_ratio"])
        return ImageTokenSelection(plan["selected_visual_indices"], metadata={
            "algorithm": self.method.algorithm, "knee_index": plan["knee_index"],
            "threshold": plan["threshold"]})


class TwigVLMMethod(VLMPrunerMethod):
    algorithm = "twigvlm"

    def __init__(self, *, stage, config, rules, backbone):
        from .twigvlm import normalize_twigvlm_config
        self.config = normalize_twigvlm_config(config)
        self.spec = SimpleNamespace(**self.config)
        self.rules = rules
        self.positions = (LayerPosition("inter_layer", self.config["twig_start_layer"]),)
        self.score_layers = ()

    def resolve_depth(self, depth):
        from .twigvlm import resolve_twigvlm_schedule
        self.depth = depth
        self.start, self.count, self.wipe = resolve_twigvlm_schedule(depth, self.config)
        self.positions = (LayerPosition("inter_layer", self.start),)
        if self.config["enable_final_wipe"]:
            self.positions += (LayerPosition("inter_layer", self.wipe),)

    def new_call(self):
        return TwigVLMCall(self)


class TwigVLMCall(DecoderSelectionCall):
    def __init__(self, method):
        self.method = method
        self.plan = None

    def initialize(self, input_embeddings, image_mask, *, padding_mask, labels):
        from mllm_comp.compressors.image.fastv import fastv_scorer_query_indices
        self.prompt_query = int(fastv_scorer_query_indices(image_mask,
            score_source="last_token", attention_mask=padding_mask, labels=labels)[-1])
        self.image_indices = image_mask.nonzero(as_tuple=True)[0]

    def select(self, position, hidden_states, *, image_indices, protected_indices=(), **kwargs):
        if position.index == self.method.start:
            if self.plan is None:
                raise ValueError("TwigVLM needs its native initialization branch observation")
            selected = int(self.image_indices[0]) + self.plan["selected_visual_indices"]
        else:
            selected = image_indices[:0]
        selection = ImageTokenSelection(selected, metadata={"algorithm": "twigvlm",
            "adaptation": "initialization_only", "final_wipe": position.index != self.method.start})
        return ImageTokenSelectionResult.from_selection(selection, hidden_states,
            image_indices=image_indices, protected_indices=(), **kwargs)
