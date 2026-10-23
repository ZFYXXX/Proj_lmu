"""Answer-blind reuse of the framework's Qwen projector comparison kernels.

These are Qwen2.5 adaptations, not claims to reproduce another backbone's
paper numbers. Generation preserves original text and retained-token M-RoPE.
"""

from contextlib import contextmanager

METHODS = ("cdpruner", "dppruner", "eadp", "hiprune")
SOURCES = {
    "cdpruner": "https://github.com/Theia-4869/CDPruner",
    "dppruner": "https://github.com/Theia-4869/CDPruner",
    "eadp": "https://github.com/SJTU-DeepVisionLab/EADP",
    "hiprune": "https://github.com/Danielement321/HiPrune",
}


def prepare_prompt_state(bundle, entry, image_data, question):
    """Prepare exactly the historical image/chat protocol, without an answer."""
    from io import BytesIO
    from PIL import Image
    from mllm_comp.backbones.qwen25_vl import DEFAULT_SYSTEM_PROMPT, Qwen25VLSample
    from mllm_comp.analysis.common.learned_hybrid_runtime import prepare_state
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    source = BytesIO(image_data["bytes"]) if image_data.get("bytes") else image_data["path"]
    with Image.open(source) as original:
        image = original.convert("RGB")
    try:
        if media_fingerprint([image]) != entry["image_fingerprint"]:
            raise ValueError("Image content fingerprint changed")
        sample = Qwen25VLSample(entry["sample_id"], [
            {"role": "system", "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": image, "min_pixels": 200704, "max_pixels": 1605632},
                {"type": "text", "text": question}]}])
        prompt = bundle["facade"].processor_adapter.prepare_inputs(
            bundle["processor"], sample, phase="eval").model_inputs
        with capture_hiprune(bundle, prompt["image_grid_thw"]) as (capture, config):
            state = prepare_state(bundle, prompt)
        return state, {"attention": capture["attention"], "config": config}
    finally:
        image.close()


def method_config(method, budget=128):
    from mllm_comp.compressors.comparison import cdpruner, eadp, hiprune
    normalizer = {
        "cdpruner": cdpruner.normalize_cdpruner_config,
        "dppruner": cdpruner.normalize_dppruner_config,
        "eadp": eadp.normalize_eadp_config,
        "hiprune": hiprune.normalize_hiprune_config,
    }[method]
    return normalizer({"tokens_per_crop": budget})


@contextmanager
def capture_hiprune(bundle, grid, budget=128):
    """Use the native framework window/RoPE-aware vision attention capture."""
    from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
    capture = QwenProjectorComparison(bundle["model"], "hiprune",
                                     {"tokens_per_crop": budget}, "qwen25_vl")
    source = {"grid": grid.to(bundle["device"]), "attention": {}}
    token = capture.source.set(source)
    try:
        capture.install_vision_capture()
        yield source, capture.config
    finally:
        capture.uninstall()
        capture.source.reset(token)


def user_embeddings(bundle, state):
    """Match the native binding's user-span extraction; never read labels."""
    from mllm_comp.compressors.comparison.llava import instruction_ids
    model = bundle["model"]
    prompt = state["prompt_inputs"]
    ids = prompt["input_ids"].clone()
    for value in (model.config.image_token_id, model.config.vision_start_token_id,
                  model.config.vision_end_token_id):
        ids[ids == value] = -200
    selected = instruction_ids(ids, prompt.get("attention_mask"), None)
    return bundle["llm"].embed_tokens(selected)


def select_indices(bundle, state, method, *, attention=None, capture_config=None,
                   budget=128):
    import torch
    features = state["features"]
    config = method_config(method, budget)
    text = user_embeddings(bundle, state) if method in {"cdpruner", "eadp"} else None
    if method in {"cdpruner", "dppruner"}:
        from mllm_comp.compressors.comparison.cdpruner import select_cdpruner_tokens
        _, selected, stats = select_cdpruner_tokens(features[None], text, cfg=config)
    elif method == "eadp":
        from mllm_comp.compressors.comparison.eadp import select_eadp_indices
        selected, stats = select_eadp_indices(features[None], text, cfg=config,
            patches_per_side=state["width"], grid_height=state["height"])
    elif method == "hiprune":
        from mllm_comp.compressors.comparison.hiprune import select_hiprune_indices
        if attention is None or capture_config is None:
            raise ValueError("HiPrune requires captured native shallow/deep vision attention")
        depth = len(bundle["model"].model.visual.blocks)
        config = capture_config
        selected, stats = select_hiprune_indices(
            attention[config["object_layer"] - 1],
            attention[config["vision_layer"] % depth], tokens_per_crop=budget,
            alpha=config["alpha"], patches_per_side=state["width"],
            grid_height=state["height"])
    else:
        raise ValueError(method)
    selected = selected[0].to(features.device, dtype=torch.long)
    expected = min(budget, len(features))
    if len(selected) != expected or len(torch.unique(selected)) != expected:
        raise ValueError(f"{method}: wrong count or duplicate token selection")
    if bool((selected < 0).any()) or bool((selected >= len(features)).any()):
        raise ValueError(f"{method}: out-of-range token selection")
    selected = selected.sort().values
    return selected, {"algorithm": method, "binding": f"{method}.image.qwen25_vl",
        "placement": "projector_level", "target_site": "post_vision_merger_pre_llm",
        "adaptation": "framework_native_rectangular_post_merger_grid",
        "source_count": len(features), "result_count": len(selected),
        "selected_indices": selected.tolist(), "native_grid": [state["height"], state["width"]],
        "position_policy": "original_retained_token_mrope_text_unchanged",
        "config": config, "statistics": stats, "trained": False, "source": SOURCES[method]}


def compact_prompt(state, selected):
    """Physically prune a gold-free prompt and verify all aligned tensors."""
    import torch
    selected = torch.as_tensor(selected, dtype=torch.long, device=state["hidden"].device)
    visual = state["visual"]
    if selected.ndim != 1 or not len(selected) or len(selected.unique()) != len(selected):
        raise ValueError("Selection must contain unique token indices")
    if selected.tolist() != sorted(selected.tolist()) or int(selected.min()) < 0 or int(selected.max()) >= len(visual):
        raise ValueError("Selection must be sorted and in range")
    prefix = state["prefix"]
    if any(i >= prefix for i in visual):
        raise ValueError("Visual tokens must be in the gold-free prompt")
    device = selected.device
    keep = torch.ones(prefix, dtype=torch.bool, device=device)
    source = torch.tensor(visual, device=device)
    keep[source] = False
    keep[source[selected]] = True
    gather = keep.nonzero(as_tuple=True)[0]
    hidden = state["hidden"][:, :prefix].index_select(1, gather)
    positions = state["positions"][:, :, :prefix].index_select(-1, gather)
    prompt = state["prompt_inputs"]
    ids = prompt["input_ids"].index_select(1, gather)
    mask = prompt["attention_mask"].index_select(1, gather)
    visual_id = prompt["input_ids"][0, visual[0]]
    text_source = torch.ones(prefix, dtype=torch.bool, device=device)
    text_source[source] = False
    text_dest = ids[0] != visual_id
    if (hidden.shape[1] != prefix - len(visual) + len(selected) or
            ids.shape != mask.shape or not bool((mask == 1).all()) or
            int((ids == visual_id).sum()) != len(selected) or
            not torch.equal(positions[:, :, text_dest], state["positions"][:, :, :prefix][:, :, text_source]) or
            not torch.equal(ids[:, text_dest], prompt["input_ids"][:, text_source]) or
            not bool(torch.isfinite(hidden).all())):
        raise ValueError("Pruned prompt alignment, positions, count or finite-value check failed")
    return {"hidden": hidden, "positions": positions, "input_ids": ids,
            "attention_mask": mask, "prefix": hidden.shape[1], "gather": gather}


def decode_prompt(bundle, state, *, max_new, selected=None):
    from mllm_comp.analysis.probes.visual_evidence_interaction import cached_decode
    compact = compact_prompt(state, selected) if selected is not None else {
        "hidden": state["hidden"][:, :state["prefix"]],
        "positions": state["positions"][:, :, :state["prefix"]]}
    eos = bundle["model"].generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else list(eos)
    result = cached_decode(llm=bundle["llm"], head=bundle["model"].lm_head,
        embedding=bundle["model"].get_input_embeddings(), hidden=compact["hidden"],
        positions=compact["positions"], eos_ids=eos, max_new=max_new)
    expected = int(compact["positions"].max()) + 1
    if result["first_appended_token_position"] != expected:
        raise ValueError("Decode position is not prompt maximum M-RoPE plus one")
    result.update(prediction=bundle["processor"].tokenizer.decode(
        result["generated_token_ids"], skip_special_tokens=True).strip(),
        retained_visual_tokens=len(state["visual"]) if selected is None else len(selected),
        source_visual_tokens=len(state["visual"]), text_positions_preserved=True,
        placeholder_mask_labels_alignment_passed=True)
    return result
