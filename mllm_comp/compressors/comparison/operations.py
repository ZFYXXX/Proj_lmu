"""Operation comparisons on native decoder modules, with request-local layouts.

Only the selected Attention/FFN computations are replaced. Native model loops,
RoPE construction, multimodal preparation and checkpointing remain in charge.
"""
from contextvars import ContextVar
from importlib import import_module

from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.fixed_mask.runtime import _call, make_layout

KEY = "comparison_operation_context"
ATTRIBUTE = "_mllm_comparison_operations"


def has_cache(past):
    if past is None:
        return False
    return any(past.get_seq_length(i) > 0 for i in range(len(past.layers)))


class OperationComparison:
    def __init__(self, model, method, config, backbone, checkpoint_path=None):
        if backbone not in {"llava_onevision", "qwen25_vl", "qwen3_vl"}:
            raise ValueError("Unknown comparison backbone")
        self.model, self.method, self.backbone = model, method, backbone
        self.decoder = model.get_model() if backbone == "llava_onevision" else model.model.language_model
        self.is_adallava = method in {"adallava", "adallava_l", "adallava_h"}
        self.is_projector = method in {"cdpruner", "dppruner", "eadp", "hiprune", "prunesid"}
        module = import_module("." + ("adallava" if self.is_adallava else "cdpruner" if method == "dppruner" else method), __package__)
        config = dict(config)
        if method == "yopo":
            config.setdefault("head_mask_source", "file")
        if self.is_adallava:
            config.setdefault("method", method)
            config.setdefault("scheduler_variant", "H" if method == "adallava_h" else "L")
            config.setdefault("original_layer_count", len(self.decoder.layers))
            config.setdefault("num_prefix_layers", len(self.decoder.layers) // 2)
            config.setdefault("num_attention_heads", self.decoder.config.num_attention_heads)
            config.setdefault("num_key_value_heads", self.decoder.config.num_key_value_heads)
            config.setdefault("head_group_size", self.decoder.config.num_attention_heads // self.decoder.config.num_key_value_heads)
            config.setdefault("depth_adaptation", "target_checkpoint_geometry")
        self.config = getattr(module, "normalize_" + ("adallava" if self.is_adallava else method) + "_config")(config)
        self.layers = self.decoder.layers
        if self.is_adallava:
            for key, actual in (("original_layer_count", len(self.layers)),
                                ("num_attention_heads", self.decoder.config.num_attention_heads),
                                ("num_key_value_heads", self.decoder.config.num_key_value_heads)):
                if self.config[key] != actual:
                    raise ValueError(f"AdaLLaVA {key} policy does not match this checkpoint")
        for field in ("replaced_layers", "attention_reduction_layers", "ffn_reduction_layers", "pruned_layers"):
            if any(i >= len(self.layers) for i in self.config.get(field, [])):
                raise ValueError(f"{method} {field} exceeds this checkpoint's depth")
        if method == "dop":
            if max(self.config["mha_depth"], self.config["mlp_depth"]) > len(self.layers):
                raise ValueError("DOP depth exceeds this checkpoint's depth")
        self.head_mask = None
        if method == "yopo":
            self.head_mask = module.load_yopo_head_mask(self.config, layer_count=len(self.layers),
                head_count=self.decoder.config.num_attention_heads)
        if method in {"structural_pruning", "short_lvlm"}:
            if len(self.config["pruned_layers"]) >= len(self.layers):
                raise ValueError("Layer pruning must retain at least one decoder block")
            if self.config.get("original_layer_count", self.config.get("flops_reference_decoder_layers")) != len(self.layers):
                raise ValueError("Layer policy belongs to a different decoder depth")
        self.checkpoint_path = checkpoint_path
        self.scheduler = None
        self.latency_position = None
        self.skip_layouts = None
        self.labels = ContextVar(f"comparison_labels_{id(self)}", default=None)
        self.incoming = ContextVar(f"comparison_input_{id(self)}", default=None)
        self.current = ContextVar(f"comparison_layer_{id(self)}", default=None)
        self.request = None
        self.patches = []
        self.calls = 0
        self.projector = None

    def patch(self, owner, name, function):
        self.patches.append(MethodPatch.apply(owner, name, function, getattr(owner, name), label=self.method))

    def install(self):
        if self.patches:
            return self
        if any(getattr(self.model, name, None) is not None for name in
               (ATTRIBUTE, "_mllm_fixed_mask", "_modality_split_session")):
            raise ValueError("Operation comparisons require exclusive ownership of decoder modules")
        if any(layer.self_attn.config._attn_implementation not in {"eager", "sdpa"} for layer in self.layers):
            raise ValueError("Operation comparisons require eager or sdpa attention")
        setattr(self.model, ATTRIBUTE, self)
        try:
            if self.method == "short_lvlm":
                self.install_scp()
            if self.is_adallava:
                from .adallava import ensure_adallava_scheduler
                self.scheduler = ensure_adallava_scheduler(self.decoder, self.config)
                reference = self.decoder.embed_tokens.weight
                self.scheduler.to(device=reference.device, dtype=reference.dtype)
                self.scheduler.train(self.model.training)
                self.install_scheduler_checkpoint()
            if self.method == "skip_vision":
                from .skip_vision import ensure_skip_vision_summary
                self.scheduler = ensure_skip_vision_summary(self.decoder)
                reference = self.decoder.embed_tokens.weight
                self.scheduler.to(device=reference.device, dtype=reference.dtype)
                self.scheduler.train(self.model.training)
                self.install_scheduler_checkpoint()
            self.install_input()
            self.install_decoder()
            for index, layer in enumerate(self.layers):
                self.install_layer(layer, index)
            if self.is_projector:
                self.install_vision_capture()
            if self.method == "dop" and self.config["token_reducer"] == "vispruner" and self.backbone == "llava_onevision":
                from types import SimpleNamespace
                from .llava import LlavaProjectorComparison
                self.projector = LlavaProjectorComparison(self.model,
                    SimpleNamespace(algorithm="dop", config=self.config),
                    SimpleNamespace(name="dop.image.llava_onevision"), "generate")
                self.projector.install()
        except BaseException:
            self.uninstall()
            raise
        return self

    def uninstall(self):
        if self.projector is not None:
            self.projector.uninstall()
            self.projector = None
        for patch in reversed(self.patches):
            patch.restore()
        self.patches.clear()
        if getattr(self.model, ATTRIBUTE, None) is self:
            delattr(self.model, ATTRIBUTE)
        self.request = None

    def install_input(self):
        if self.backbone != "llava_onevision":
            self.install_qwen_input()
            return
        original = self.model.prepare_inputs_labels_for_multimodal

        def prepare(_model, *args, **kwargs):
            import torch
            call = _call(original, args, kwargs)
            ids = call["input_ids"]
            if ids.shape[0] != 1 or any(m != "image" for m in call.get("modalities", ["image"])):
                raise ValueError("Operation comparisons require batch-one image inputs")
            if has_cache(call.get("past_key_values")):
                return original(*args, **kwargs)
            labels = call.get("labels")
            call["labels"] = torch.arange(ids.shape[-1], device=ids.device)[None]
            result = list(original(**call))
            if result[4] is None:
                self.incoming.set(make_layout(torch.zeros_like(ids, dtype=torch.bool), call.get("attention_mask")))
                result[5] = labels
                return tuple(result)
            mapping = result[5]
            self.incoming.set(make_layout(mapping == -100, result[2]))
            result[5] = None if labels is None else labels.gather(1, mapping.clamp_min(0)).masked_fill(mapping == -100, -100)
            if self.is_adallava:
                self.insert_latency(result)
            if self.method == "skip_vision":
                self.compress_skip_vision(result)
            return tuple(result)
        self.patch(self.model, "prepare_inputs_labels_for_multimodal", prepare)

    def install_decoder(self):
        original = self.decoder.forward

        def forward(_decoder, *args, **kwargs):
            import torch
            call = _call(original, args, kwargs)
            cached = has_cache(call.get("past_key_values"))
            hidden = call.get("inputs_embeds")
            if hidden is None:
                hidden = self.decoder.embed_tokens(call["input_ids"])
            if not cached and self.backbone != "llava_onevision" and (self.is_adallava or self.method == "skip_vision" or self.is_projector):
                hidden = self.prepare_qwen_embeddings(call, hidden)
            length = hidden.shape[1]
            if not cached:
                layout = self.incoming.get()
                if layout is None or layout.vision.shape[-1] != length:
                    raise ValueError("Operation comparison is missing its prepared multimodal layout")
                indices = layout.vision[0].nonzero(as_tuple=True)[0]
                spans = []
                for value in indices.tolist():
                    if spans and spans[-1][1] == value:
                        spans[-1][1] += 1
                    else:
                        spans.append([value, value + 1])
                if len(spans) > 1:
                    raise ValueError("This operation comparison binding requires one image span")
                runtime = {"has_image": bool(spans), "batch_size": 1,
                    "expanded_prefill_seq_len": length, "expanded_visual_spans": spans,
                    "visual_start": spans[0][0] if spans else 0,
                    "visual_end": spans[0][1] if spans else 0,
                    "visual_tokens": len(indices), "orig_seq_len": length - len(indices) + int(bool(spans)),
                    "placeholder_positions": [spans[0][0]] if spans else [],
                    "prefill_length": length}
                if self.method == "skip_vision":
                    runtime["skip_vision_layouts"] = self.skip_layouts
                self.request = {"layout": layout, "runtime": runtime, "cache_positions": {}, "next_position": 0}
                self.request["latency_position"] = self.latency_position
            elif self.request is None:
                raise ValueError("A comparison cache must be decoded in the session that created it")
            state = self.request
            offset = state["next_position"]
            call["cache_position"] = torch.arange(offset, offset + length, device=hidden.device)
            if self.backbone == "llava_onevision" and (call.get("position_ids") is None or cached):
                call["position_ids"] = call["cache_position"][None]
            # This immutable per-call tuple travels through the native checkpoint
            # closure, so backward never reads the layout of a later request.
            call[KEY] = (state, cached, call["cache_position"])
            self.calls += 1
            try:
                output = original(**call)
                if self.method == "skip_vision" and not cached and output.past_key_values is not None:
                    from .skip_vision import skip_vision_operation_indices
                    from mllm_comp.compressors.image.decoder_lifecycle import compact_dynamic_cache_prefix
                    _, keep = skip_vision_operation_indices(state["runtime"], length, device=hidden.device)
                    compact_dynamic_cache_prefix(output.past_key_values, keep, source_length=length,
                        result_length=len(keep), layer_count=len(self.layers), boundary_layer=len(self.layers))
                    for index in state["cache_positions"]:
                        state["cache_positions"][index] = state["cache_positions"][index].index_select(0, keep)
                state["next_position"] = offset + length
                return output
            finally:
                self.incoming.set(None)
        self.patch(self.decoder, "forward", forward)

    def install_layer(self, layer, index):
        original_layer, original_attention, original_ffn = layer.forward, layer.self_attn.forward, layer.mlp.forward

        def forward(_layer, *args, **kwargs):
            import torch
            context = kwargs.pop(KEY, None)
            token = self.current.set(context)
            try:
                call = _call(original_layer, args, kwargs)
                hidden = call["hidden_states"]
                if self.is_adallava and context is not None:
                    state, cached, _ = context
                    if self.model.training and self.decoder.gradient_checkpointing:
                        raise ValueError("AdaLLaVA scheduler training currently requires gradient_checkpointing=False")
                    if index == self.config["num_prefix_layers"] and not cached:
                        budget = self.config["active_units" if self.config["scheduler_variant"] == "H" else "active_layers"]
                        state["plan"] = self.scheduler(hidden[:, state["latency_position"]],
                            torch.tensor([budget], device=hidden.device))
                        state["runtime"]["scheduler_active_budget"] = budget
                if self.method in {"structural_pruning", "short_lvlm"} and index in self.config["pruned_layers"]:
                    if context is not None:
                        context[0]["runtime"]["skipped_layers"] = list(self.config["pruned_layers"])
                    self.empty_cache_slot(layer, call, index)
                    return self.skipped_output(hidden, call)
                if self.is_adallava and context is not None and not self.model.training:
                    plan = context[0].get("plan")
                    if plan is not None and not bool(plan[:, index].any()):
                        self.empty_cache_slot(layer, call, index)
                        return self.skipped_output(hidden, call)
                return original_layer(*args, **kwargs)
            finally:
                self.current.reset(token)

        def attention(module, *args, **kwargs):
            context = self.current.get()
            if context is None:
                return original_attention(*args, **kwargs)
            state, cached, positions = context
            if not state["runtime"]["has_image"]:
                return original_attention(*args, **kwargs)
            selected = (
                self.method == "skip_vision"
                or self.method == "dop" and index >= self.config["mha_depth"]
                or self.method == "shortv" and index in self.config["replaced_layers"] and not cached
                or self.method == "redundancy_lens" and index in self.config["attention_reduction_layers"] and not cached
                or self.method == "yopo" and (not cached or not self.config["prefill_only"]))
            if not selected:
                if not self.is_adallava or "plan" not in state:
                    return original_attention(*args, **kwargs)
                gate = state["plan"][:, index]
                if self.config["scheduler_variant"] == "L":
                    output, weights = original_attention(*args, **kwargs)
                    return output * gate[:, None, None], weights
                def gate_heads(_projection, inputs):
                    value = inputs[0]
                    heads = module.config.num_attention_heads
                    keep = gate[:, 0]
                    return ((value.view(*value.shape[:2], heads, -1) * keep[:, None, :, None]).reshape_as(value),)
                handle = module.o_proj.register_forward_pre_hook(gate_heads)
                try:
                    return original_attention(*args, **kwargs)
                finally:
                    handle.remove()
            return self.attention(module, _call(original_attention, args, kwargs), index, context)

        def ffn(module, hidden):
            import torch
            context = self.current.get()
            if self.is_adallava and context is not None and "plan" in context[0]:
                gate = context[0]["plan"][:, index]
                if self.config["scheduler_variant"] == "L":
                    return original_ffn(hidden) * gate[:, None, None]
                intermediate = module.act_fn(module.gate_proj(hidden)) * module.up_proj(hidden)
                groups = gate.shape[-1]
                channels = torch.arange(intermediate.shape[-1], device=hidden.device)
                group_ids = channels * groups // intermediate.shape[-1]
                keep = gate[:, 1].index_select(-1, group_ids)
                return module.down_proj(intermediate * keep[:, None])
            if context is None or context[1] or not context[0]["runtime"]["has_image"]:
                return original_ffn(hidden)
            state = context[0]
            visual = state["layout"].vision.to(hidden.device)
            if self.method == "skip_vision":
                from .skip_vision import skip_vision_operation_indices
                indices, _ = skip_vision_operation_indices(state["runtime"], hidden.shape[1], device=hidden.device)
                return torch.zeros_like(hidden).index_copy(1, indices, original_ffn(hidden.index_select(1, indices)))
            skip = (self.method == "dop" and index >= self.config["mlp_depth"]
                    or self.method == "shortv" and index in self.config["replaced_layers"])
            if skip:
                state["runtime"]["operation_ffn_calls"] = state["runtime"].get("operation_ffn_calls", 0) + 1
                indices = (~visual[0]).nonzero(as_tuple=True)[0]
                values = original_ffn(hidden.index_select(1, indices))
                return torch.zeros_like(hidden).index_copy(1, indices, values)
            if self.method == "redundancy_lens" and index in self.config["ffn_reduction_layers"]:
                from .redundancy_lens import probe_activated_structural_ffn, apply_probe_activated_ffn_mask
                if self.config["operation_backend"] == "indexed_structural":
                    result = probe_activated_structural_ffn(hidden, gate_proj=module.gate_proj,
                        up_proj=module.up_proj, down_proj=module.down_proj, act_fn=module.act_fn,
                        cfg=self.config, runtime=state["runtime"])
                    if result is None:
                        raise ValueError("Indexed RedundancyLens requires supported linear or LoRA projections")
                    return result
                intermediate = module.act_fn(module.gate_proj(hidden)) * module.up_proj(hidden)
                return module.down_proj(apply_probe_activated_ffn_mask(intermediate,
                    cfg=self.config, runtime=state["runtime"]))
            if self.method == "yopo":
                from .yopo import apply_yopo_ffn_mask
                gate, up = apply_yopo_ffn_mask(module.gate_proj(hidden), module.up_proj(hidden),
                    cfg=self.config, runtime=state["runtime"], layer_idx=index)
                return module.down_proj(module.act_fn(gate) * up)
            return original_ffn(hidden)

        self.patch(layer, "forward", forward)
        self.patch(layer.self_attn, "forward", attention)
        self.patch(layer.mlp, "forward", ffn)

    def attention(self, module, call, index, context):
        import torch
        from torch.nn import functional as F
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
        state, cached, positions = context
        hidden = call["hidden_states"]
        batch, length, _ = hidden.shape
        runtime = state["runtime"]
        all_indices = torch.arange(length, device=hidden.device)
        visual = state["layout"].vision[0].to(hidden.device)
        nonvisual = all_indices if cached else (~visual).nonzero(as_tuple=True)[0]
        q_indices = kv_indices = all_indices
        if self.method == "dop":
            q_indices = kv_indices = nonvisual
        elif self.method == "shortv":
            q_indices = nonvisual
        elif self.method == "yopo":
            from .yopo import yopo_uses_indexed_text_only
            if yopo_uses_indexed_text_only(self.config, index):
                q_indices = kv_indices = nonvisual
        heads, kv_heads = module.config.num_attention_heads, module.config.num_key_value_heads
        dim = module.head_dim
        q = module.q_proj(hidden.index_select(1, q_indices)).view(batch, -1, heads, dim)
        k = module.k_proj(hidden.index_select(1, kv_indices)).view(batch, -1, kv_heads, dim)
        if self.backbone == "qwen3_vl":
            q, k = module.q_norm(q), module.k_norm(k)
        q, k = q.transpose(1, 2), k.transpose(1, 2)
        v = module.v_proj(hidden.index_select(1, kv_indices)).view(batch, -1, kv_heads, dim).transpose(1, 2)
        cos, sin = call["position_embeddings"]
        if self.backbone == "qwen25_vl":
            from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb
            def rotate(value, indices):
                return apply_multimodal_rotary_pos_emb(value, value, cos.index_select(-2, indices),
                    sin.index_select(-2, indices), module.rope_scaling["mrope_section"])[0]
        else:
            def rotate(value, indices):
                return apply_rotary_pos_emb(value, value, cos.index_select(-2, indices), sin.index_select(-2, indices))[0]
        q, k = rotate(q, q_indices), rotate(k, kv_indices)
        key_positions = positions.index_select(0, kv_indices)
        cache = call.get("past_key_values")
        if cache is not None:
            if cached:
                key_positions = torch.cat((state["cache_positions"][index].to(key_positions.device), key_positions))
            k, v = cache.update(k, v, index, {"cos": cos, "sin": sin,
                "cache_position": positions.index_select(0, kv_indices)})
            state["cache_positions"][index] = key_positions.detach()
        k, v = repeat_kv(k, heads // kv_heads), repeat_kv(v, heads // kv_heads)
        query_positions = positions.index_select(0, q_indices)
        allowed = key_positions[None, :] <= query_positions[:, None]
        window = getattr(module, "sliding_window", None)
        if window is not None:
            allowed &= key_positions[None, :] > query_positions[:, None] - window
        valid = state["layout"].valid[0].to(hidden.device)
        key_valid = torch.ones_like(key_positions, dtype=torch.bool)
        prefix = key_positions < valid.numel()
        key_valid[prefix] = valid[key_positions[prefix]]
        allowed &= key_valid[None, :]
        mask = hidden.new_zeros((1, 1, q.shape[-2], k.shape[-2])).masked_fill(~allowed[None, None], float("-inf"))
        if self.method == "redundancy_lens":
            from .redundancy_lens import prepare_hollow_attention_mask
            mask = prepare_hollow_attention_mask(mask, runtime=runtime, attention_range=self.config["attention_range"])
        logits = q @ k.transpose(-1, -2) * getattr(module, "scaling", dim ** -.5)
        yopo_masks = self.method == "yopo" and not yopo_uses_indexed_text_only(self.config, index)
        if yopo_masks:
            from .yopo import (prepare_yopo_attention_mask, apply_yopo_layer_drop_to_logits,
                apply_yopo_head_mask_to_probs, apply_yopo_visual_query_head_mask)
            mask = prepare_yopo_attention_mask(mask, attn_logits=logits, cfg=self.config,
                runtime=runtime, past_len=positions[0].item(), layer_idx=index)
            logits = apply_yopo_layer_drop_to_logits(logits + mask, cfg=self.config,
                runtime=runtime, layer_idx=index, past_len=positions[0].item())
        else:
            logits = logits + mask
        probabilities = torch.softmax(logits.float(), dim=-1).to(q.dtype).nan_to_num()
        if yopo_masks:
            probabilities = apply_yopo_head_mask_to_probs(probabilities, cfg=self.config,
                runtime=runtime, active_head_mask=self.head_mask, layer_idx=index,
                past_len=positions[0].item())
        probabilities = F.dropout(probabilities, p=module.attention_dropout, training=module.training)
        output = probabilities @ v
        if yopo_masks:
            output = apply_yopo_visual_query_head_mask(output, cfg=self.config,
                runtime=runtime, active_head_mask=self.head_mask, layer_idx=index,
                past_len=positions[0].item())
        output = module.o_proj(output.transpose(1, 2).reshape(batch, -1, heads * dim))
        if q_indices.numel() != length:
            output = torch.zeros_like(hidden).index_copy(1, q_indices, output)
        runtime["operation_attention_calls"] = runtime.get("operation_attention_calls", 0) + 1
        return output, probabilities if call.get("output_attentions", False) else None

    def skipped_output(self, hidden, call):
        if self.backbone == "qwen25_vl":
            return (hidden, None) if call.get("output_attentions", False) else (hidden,)
        return hidden

    def install_qwen_input(self):
        original = self.model.forward
        def forward(_model, *args, **kwargs):
            import torch
            call = _call(original, args, kwargs)
            ids = call.get("input_ids")
            if call.get("pixel_values_videos") is not None:
                raise ValueError("These comparison bindings support images only")
            if has_cache(call.get("past_key_values")):
                offset = self.request["next_position"]
                call["cache_position"] = torch.arange(offset, offset + ids.shape[1], device=ids.device)
                if call.get("position_ids") is None:
                    call["position_ids"] = (call["cache_position"][None] + self.model.model.rope_deltas).unsqueeze(0).expand(3, -1, -1)
            else:
                if ids is None:
                    raise ValueError("Qwen comparisons require input_ids to identify image positions")
                self.incoming.set(make_layout(ids == self.model.config.image_token_id, call.get("attention_mask")))
                if self.is_projector:
                    self.prepare_source_inputs(call)
            changing = self.is_adallava or self.method == "skip_vision" or self.is_projector
            labels = call.get("labels")
            token = self.labels.set(labels)
            if changing and labels is not None:
                call["labels"] = None
                call["logits_to_keep"] = 0
                call["return_dict"] = True
            try:
                output = original(**call)
                if changing and labels is not None:
                    from mllm_comp.backbones.image_training import attach_shifted_causal_loss
                    output = attach_shifted_causal_loss(output, self.labels.get(), operation=self.method)
                return output
            finally:
                self.labels.reset(token)
        self.patch(self.model, "forward", forward)

    def runtime_metadata(self):
        runtime = {} if self.request is None else {k: v for k, v in self.request["runtime"].items() if not k.startswith("_")}
        executed = bool(runtime.get("operation_attention_calls") or runtime.get("operation_ffn_calls") or runtime.get("skipped_layers")
                        or runtime.get("scheduler_active_budget"))
        return {"method": self.method, "backbone": self.backbone, "config": self.config,
            "installed": bool(self.patches), "forward_calls": self.calls,
            "layout_delta_records": [] if self.projector is None else self.projector.runtime_metadata()["layout_delta_records"],
            "runtime": runtime,
            "operation_records": ([{"algorithm": self.method, "comparison_operation_executed": True,
                                     "runtime": runtime}] if executed else []),
            "support_activation_claimed": False, "real_checkpoint_verified": False}

    def clear_diagnostic_records(self):
        if self.projector is not None:
            self.projector.clear_diagnostic_records()
        # Cache and scheduler state belong to the active request, not diagnostics.
        if self.request is not None:
            for key in ("operation_attention_calls", "operation_ffn_calls", "skipped_layers", "scheduler_active_budget"):
                self.request["runtime"].pop(key, None)

    def empty_cache_slot(self, layer, call, index):
        cache = call.get("past_key_values")
        if cache is not None and not cache.get_seq_length(index):
            hidden = call["hidden_states"]
            empty = hidden.new_empty((hidden.shape[0], layer.self_attn.config.num_key_value_heads, 0, layer.self_attn.head_dim))
            cache.update(empty, empty, index, {"cache_position": call.get("cache_position")})

    def install_scp(self):
        """Apply W(I+VVᵀ) through an equivalent, reversible input transform."""
        import torch
        asset = torch.load(self.config["calibration_asset_path"], map_location="cpu", weights_only=True)
        if asset.get("format") != "short_lvlm_scp_asset_v1" or asset.get("projection_pairs") != self.config["projection_pairs"]:
            raise ValueError("Short-LVLM calibration asset and layer policy disagree")
        if asset.get("metadata", {}).get("hidden_size") != self.decoder.config.hidden_size:
            raise ValueError("Short-LVLM calibration hidden size does not match this checkpoint")
        for pair in self.config["projection_pairs"]:
            index = pair["retained_layer"]
            if not 0 <= index < len(self.layers):
                raise ValueError("Short-LVLM SCP retained layer exceeds this checkpoint's depth")
            basis = asset["bases"][str(index)]
            if basis.shape != (self.decoder.config.hidden_size, self.config["subspace_rank"]):
                raise ValueError("Short-LVLM calibration subspace has incompatible dimensions")
            layer = self.layers[index]
            for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.mlp.gate_proj, layer.mlp.up_proj):
                self.install_scp_projection(projection, basis)

    def install_scp_projection(self, projection, basis):
        original = projection.forward
        basis = basis.to(device=projection.weight.device, dtype=projection.weight.dtype)
        def projected(_module, value):
            return original(value + (value @ basis) @ basis.transpose(0, 1))
        self.patch(projection, "forward", projected)

    def insert_latency(self, prepared):
        import torch
        hidden, labels = prepared[4], prepared[5]
        layout = self.incoming.get()
        boundary = hidden.shape[1]
        if labels is not None:
            targets = (labels[0] != -100).nonzero(as_tuple=True)[0]
            if targets.numel():
                boundary = int(targets[0])
        budget = self.config["active_units" if self.config["scheduler_variant"] == "H" else "active_layers"]
        latency = self.scheduler.latency_encoding(torch.tensor([budget], device=hidden.device), dtype=hidden.dtype)
        prepared[4] = torch.cat((hidden[:, :boundary], latency[:, None], hidden[:, boundary:]), dim=1)
        vision = torch.cat((layout.vision[:, :boundary], torch.zeros_like(layout.vision[:, :1]), layout.vision[:, boundary:]), dim=1)
        valid = torch.cat((layout.valid[:, :boundary], torch.ones_like(layout.valid[:, :1]), layout.valid[:, boundary:]), dim=1)
        self.incoming.set(make_layout(vision, valid))
        if labels is not None:
            prepared[5] = torch.cat((labels[:, :boundary], labels.new_full((1, 1), -100), labels[:, boundary:]), dim=1)
        if prepared[2] is not None:
            prepared[2] = valid.to(prepared[2].dtype)
        if prepared[1] is not None:
            prepared[1] = valid.long().cumsum(-1) - 1
        self.latency_position = boundary

    def install_scheduler_checkpoint(self):
        from pathlib import Path
        from safetensors.torch import load_file, save_file
        sidecar = "comparison_skip_vision.safetensors" if self.method == "skip_vision" else "comparison_adallava.safetensors"
        if self.checkpoint_path:
            path = Path(self.checkpoint_path) / sidecar
            if path.is_file():
                self.scheduler.load_state_dict(load_file(str(path)))
        original = self.model.save_pretrained
        def save(_model, directory, *args, **kwargs):
            result = original(directory, *args, **kwargs)
            save_file({k: v.detach().cpu().contiguous() for k, v in self.scheduler.state_dict().items()}, str(Path(directory) / sidecar))
            return result
        self.patch(self.model, "save_pretrained", save)

    def compress_skip_vision(self, prepared):
        import torch
        from .skip_vision import compress_skip_vision_features
        hidden, labels = prepared[4], prepared[5]
        layout = self.incoming.get()
        ids = layout.vision[0].nonzero(as_tuple=True)[0]
        if not len(ids) or int(ids[-1]) - int(ids[0]) + 1 != len(ids):
            raise ValueError("Skip-Vision requires one contiguous image span")
        start, end = int(ids[0]), int(ids[-1]) + 1
        block, metadata = compress_skip_vision_features(hidden[0, start:end], cfg=self.config,
            base_crop_tokens=self.model.get_vision_tower().num_patches_per_side ** 2,
            summary_module=self.scheduler)
        self.skip_mixing = metadata.pop("_mixing_weights")
        self.skip_layouts = [metadata]
        prepared[4] = torch.cat((hidden[:, :start], block[None], hidden[:, end:]), dim=1)
        vision = torch.cat((layout.vision[:, :start], torch.ones((1, len(block)), dtype=torch.bool, device=hidden.device), layout.vision[:, end:]), dim=1)
        valid = torch.cat((layout.valid[:, :start], torch.ones((1, len(block)), dtype=torch.bool, device=hidden.device), layout.valid[:, end:]), dim=1)
        self.incoming.set(make_layout(vision, valid))
        if labels is not None:
            prepared[5] = torch.cat((labels[:, :start], labels.new_full((1, len(block)), -100), labels[:, end:]), dim=1)
        if prepared[2] is not None:
            prepared[2] = valid.to(prepared[2].dtype)
        if prepared[1] is not None:
            prepared[1] = valid.long().cumsum(-1) - 1

    def prepare_qwen_embeddings(self, call, hidden):
        import torch
        layout = self.incoming.get()
        labels = self.labels.get()
        source_length = hidden.shape[1]
        positions = call["position_ids"]
        prepared = [None, None, layout.valid, None, hidden, labels]
        if self.is_adallava:
            self.insert_latency(prepared)
            boundary = self.latency_position
            coordinate = positions[..., boundary:boundary + 1] if boundary < source_length else positions[..., -1:] + 1
            call["position_ids"] = torch.cat((positions[..., :boundary], coordinate, positions[..., boundary:] + 1), dim=-1)
        else:
            from .skip_vision import compress_skip_vision_features
            ids = layout.vision[0].nonzero(as_tuple=True)[0]
            if not len(ids) or int(ids[-1]) - int(ids[0]) + 1 != len(ids):
                raise ValueError("Skip-Vision requires one contiguous image span")
            start, end = int(ids[0]), int(ids[-1]) + 1
            if labels is not None and bool((labels[:, start:end] != -100).any()):
                raise ValueError("Skip-Vision may only replace ignored visual labels")
            block, metadata = compress_skip_vision_features(hidden[0, start:end], cfg=self.config,
                base_crop_tokens=len(ids), summary_module=self.scheduler)
            mixing = metadata.pop("_mixing_weights")
            self.skip_layouts = [metadata]
            prepared[4] = torch.cat((hidden[:, :start], block[None], hidden[:, end:]), dim=1)
            count = len(block)
            vision = torch.cat((layout.vision[:, :start], torch.ones((1, count), device=hidden.device, dtype=torch.bool), layout.vision[:, end:]), dim=1)
            valid = torch.cat((layout.valid[:, :start], torch.ones((1, count), device=hidden.device, dtype=torch.bool), layout.valid[:, end:]), dim=1)
            self.incoming.set(make_layout(vision, valid))
            prepared[2] = valid
            if labels is not None:
                prepared[5] = torch.cat((labels[:, :start], labels.new_full((1, count), -100), labels[:, end:]), dim=1)
            dominant = metadata["dominant_source_indices"]
            local = metadata["contextual_representative_source_indices"]
            representatives = torch.tensor([*dominant, dominant[-1], *local, local[-1]], device=positions.device) + start
            call["position_ids"] = torch.cat((positions[..., :start], positions.index_select(-1, representatives), positions[..., end:]), dim=-1)
            # Native three-axis coordinates survive token reordering. The text
            # causal axis, when present, describes the new physical sequence.
            if call["position_ids"].shape[0] == 4:
                call["position_ids"][0] = torch.arange(prepared[4].shape[1], device=positions.device)[None]
            deepstack = call.get("deepstack_visual_embeds")
            if deepstack is not None:
                call["deepstack_visual_embeds"] = [mixing.to(branch) @ branch for branch in deepstack]
            self.model.model.rope_deltas = self.model.model.rope_deltas + source_length - prepared[4].shape[1]
        self.labels.set(prepared[5])
        call["inputs_embeds"] = prepared[4]
        call["input_ids"] = None
        call["attention_mask"] = prepared[2]
        if "visual_pos_masks" in call:
            call["visual_pos_masks"] = self.incoming.get().vision
        return prepared[4]
