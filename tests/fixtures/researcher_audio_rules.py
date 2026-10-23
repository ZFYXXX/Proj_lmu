"""Stateless research rules and their minimal audio configurations."""

MODULE = "tests.fixtures.researcher_audio_rules"


def increasing_scores(request):
    import torch
    values = request["values"]
    if "adjacent_similarity" in request:
        return {"adjacent_similarity": torch.arange(len(values) - 1, device=values.device, dtype=torch.float32),
                "merge_weights": torch.arange(1, len(values) + 1, device=values.device, dtype=torch.float32)}
    if "merge" in request:
        return {key: torch.arange(1, len(values) + 1, device=values.device, dtype=torch.float32) for key in ("merge", "prune")}
    scores = request["scores"]
    return torch.arange(len(values) if scores is None else len(scores), device=values.device, dtype=torch.float32)


def last_indices(request):
    if "groups" in request:
        return list(range(request["keep_count"]))
    if "segment_boundaries" in request:
        return [i for (start, end), count in zip(request["segment_boundaries"], request["segment_budgets"])
                for i in range(end - count, end)]
    length = len(request.get("values", request.get("scores")))
    return list(range(length - request["keep_count"], length))


def latest_pairs(request):
    return list(request["pair_starts"][-request["merge_count"]:])


def chronological_groups(request):
    length = len(request["values"])
    count = request.get("keep_count", max(1, length // 2))
    if "merge" in request:
        count = min(length, count + 1)
    first = length - count + 1
    return [tuple(range(first))] + [(i,) for i in range(first, length)]


def first_weighted(request, *, first_weight=3.0):
    values = request["values"]
    return (first_weight * values[0] + values[1:].sum(0)) / (first_weight + len(values) - 1)


def scaled_hta(request, *, scale=1.5):
    return request["abstractor"](request["values"], keep_count=request["keep_count"])[0] * scale


def first_layers(request):
    return list(range(request["remove_count"]))


NAMES = {
    "segmentwise_topk_audio": {"scorer": "increasing_scores", "selector": "last_indices"},
    "uniform_sampling_audio": {"selector": "last_indices"},
    "encoder_attention_topk_audio": {"scorer": "increasing_scores", "selector": "last_indices"},
    "progressive_attention_pruning_audio": {"scorer": "increasing_scores", "selector": "last_indices"},
    "adjacent_token_merging_audio": {"scorer": "increasing_scores", "grouper": "latest_pairs", "aggregator": "first_weighted"},
    "uniform_average_pooling_audio": {"grouper": "chronological_groups", "aggregator": "first_weighted"},
    "unsupervised_segmentation_audio": {"scorer": "increasing_scores", "grouper": "chronological_groups", "aggregator": "first_weighted"},
    "similarity_attention_merge_prune_audio": {"scorer": "increasing_scores", "grouper": "chronological_groups", "aggregator": "first_weighted", "selector": "last_indices"},
    "fastadasp_weighted_token_merging": {"scorer": "increasing_scores", "grouper": "chronological_groups", "aggregator": "first_weighted"},
    "audio_pyramid_attention_pruning": {"scorer": "increasing_scores", "selector": "last_indices"},
    "headrouter_dynamic_audio_pruning": {"scorer": "increasing_scores", "selector": "last_indices"},
    "fastslm_hta_audio": {"transform": "scaled_hta"},
    "structured_llm_layer_pruning": {"decision": "first_layers"},
}


def specifications(algorithm):
    return {name: {"callable": MODULE + ":" + function, "params": {}} for name, function in NAMES[algorithm].items()}


def stage_for(algorithm, family=None):
    from tests.fixtures.qwen25_omni_audio import (
        projector_stage, encoder_stage, encoder_extension_stage, llm_stage,
        _operation_stage,
    )
    from mllm_comp.core.vocabulary import ExecutionPhase, Placement
    if algorithm == "structured_llm_layer_pruning":
        stage = _operation_stage()
    elif algorithm in {"audio_pyramid_attention_pruning", "headrouter_dynamic_audio_pruning", "fastadasp_weighted_token_merging"}:
        stage = llm_stage(phase="forward")
        stage.plugin = stage.algorithm = algorithm
        stage.variant = algorithm + ".audio"
        stage.binding = stage.variant + ".qwen25_omni"
        stage.activation_scope["layers"] = [0]
        if algorithm == "fastadasp_weighted_token_merging":
            stage.criterion = {}
            stage.operation["type"] = "merge"
            stage.schedule = {"merge_ratio": 0.5}
        elif algorithm == "headrouter_dynamic_audio_pruning":
            from mllm_comp.compressors.llm_level.headrouter import PROFILE_NAMES
            stage.activation_scope["layers"] = [2]
            stage.criterion = {}
            stage.schedule = {"pruning_ratio": 0.5}
            stage.config = {"profiles": {name: [0.25, 0.75] for name in PROFILE_NAMES}}
    elif algorithm == "adjacent_token_merging_audio":
        stage = encoder_stage()
        stage.activation_scope = {"layers": [0]}
        stage.schedule = {"merge_ratio": 0.5}
    elif algorithm in {"encoder_attention_topk_audio", "progressive_attention_pruning_audio"}:
        stage = encoder_extension_stage(algorithm)
        stage.activation_scope = {"layers": [0]}
    else:
        stage = projector_stage(algorithm)
    if algorithm != "structured_llm_layer_pruning":
        stage.execution_phase = ExecutionPhase.FORWARD
    if family is not None:
        stage.placement = Placement(family + "_level")
        stage.operation["target_site"] = {"projector": "audio_encoder_output_pre_projector",
            "encoder": "audio_encoder_layer_post_mhsa_pre_ffn", "llm": "llm_decoder_layer_post_attention_audio_tokens"}[family]
        if family != "projector":
            stage.operation["layers"] = [3 if family == "encoder" else 1]
    stage.operation["rules"] = specifications(algorithm)
    return stage
