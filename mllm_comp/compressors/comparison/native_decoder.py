"""Native-module execution for the CoreMatching and TwigVLM comparisons."""
from contextvars import ContextVar

from mllm_comp.backbones._provider_patching import MethodPatch

CORE_CONTEXT = "comparison_core_call"


def qwen_visual_segments(grid, merge_size):
    if grid is None or len(grid) != 1 or int(grid[0, 0]) != 1:
        raise ValueError("Comparison spatial selection requires one native image grid")
    height, width = (int(value) // merge_size for value in grid[0, 1:])
    return [{"start": 0, "end": height * width, "grid_width": width}]


def layer_context(selection):
    from .decoder import CoreMatchingCall
    return {CORE_CONTEXT: selection} if isinstance(selection, CoreMatchingCall) else {}


def observe_comparison_input(selection, index, hidden, layers, *, attention_mask,
                             position_ids, position_embeddings, cache_position,
                             deepstack_visual_embeds=None, visual_pos_masks=None):
    from .decoder import TwigVLMCall
    if not isinstance(selection, TwigVLMCall) or index != selection.method.start:
        return
    import torch
    from .twigvlm import twigvlm_intermediate_budget, build_twigvlm_ttp_plan
    method = selection.method
    cfg = method.config
    captured = []
    last = layers[method.start + method.count - 1].self_attn
    implementation = last.config._attn_implementation
    handle = last.register_forward_hook(lambda module, args, output: captured.append(output[1]))
    try:
        last.config._attn_implementation = "eager"
        with torch.no_grad():
            branch = hidden.detach()
            for offset, layer in enumerate(layers[method.start:method.start + method.count]):
                # Call forward directly: the branch is an observation, not a
                # training checkpoint or an update to the caller's KV cache.
                output = layer.forward(branch, attention_mask=attention_mask,
                    position_ids=position_ids, position_embeddings=position_embeddings,
                    cache_position=cache_position, past_key_values=None, use_cache=False,
                    output_attentions=True)
                branch = output[0] if isinstance(output, tuple) else output
                branch_index = method.start + offset
                if deepstack_visual_embeds is not None and branch_index < len(deepstack_visual_embeds):
                    branch = branch.clone()
                    branch[visual_pos_masks] += deepstack_visual_embeds[branch_index].to(branch)
        budget, _ = twigvlm_intermediate_budget(len(selection.image_indices), method.depth,
            method.start, method.wipe if cfg["enable_final_wipe"] else method.depth,
            cfg["average_retention_ratio"], min_visual_tokens=cfg["min_visual_tokens"])
        selection.plan = build_twigvlm_ttp_plan(captured[-1], sequence_length=hidden.shape[1],
            visual_span=(int(selection.image_indices[0]), int(selection.image_indices[-1]) + 1),
            retained_visual_tokens=budget, score_source=cfg["score_source"], device=hidden.device,
            prompt_query_index=selection.prompt_query)
    finally:
        handle.remove()
        last.config._attn_implementation = implementation


class CoreMatchingActivation:
    def __init__(self, native, decoder):
        self.native, self.decoder = native, decoder
        self.patches = []
        self.context = ContextVar(f"core_matching_{id(self)}", default=None)

    def install(self):
        if self.patches:
            return
        self.native.install()
        try:
            for index, layer in enumerate(self.decoder.layers):
                self.install_layer(layer, index)
        except BaseException:
            self.uninstall()
            raise

    def install_layer(self, layer, index):
        original_layer, original_mlp = layer.forward, layer.mlp.forward
        def forward(_layer, *args, **kwargs):
            call = kwargs.pop(CORE_CONTEXT, None)
            token = self.context.set(call)
            try:
                return original_layer(*args, **kwargs)
            finally:
                self.context.reset(token)

        def mlp(module, hidden):
            from .core_matching import capture_core_matching_prefill, core_matching_decode_ffn
            call = self.context.get()
            if call is None:
                return original_mlp(hidden)
            if hidden.shape[1] == 1:
                value = core_matching_decode_ffn(hidden, gate_proj=module.gate_proj,
                    up_proj=module.up_proj, down_proj=module.down_proj, act_fn=module.act_fn,
                    cfg=call.method.config, runtime=call.runtime, layer_idx=index)
                if value is None:
                    raise ValueError("CoreMatching decode requires plain linear FFN projections")
                return value
            intermediate = module.act_fn(module.gate_proj(hidden)) * module.up_proj(hidden)
            capture_core_matching_prefill(intermediate, cfg=call.method.config,
                runtime=call.runtime, layer_idx=index)
            return module.down_proj(intermediate)

        self.patches.append(MethodPatch.apply(layer, "forward", forward, original_layer, label="core_matching"))
        self.patches.append(MethodPatch.apply(layer.mlp, "forward", mlp, original_mlp, label="core_matching"))

    def uninstall(self):
        for patch in reversed(self.patches):
            patch.restore()
        self.patches.clear()
        self.native.uninstall()

    def runtime_metadata(self):
        return self.native.runtime_metadata()

    def clear_diagnostic_records(self):
        self.native.clear_diagnostic_records()
