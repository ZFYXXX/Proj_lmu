"""Compare cached and cache-free logits under the SAME autoregressive prefix."""


def compare_cache_logits(*, llm, head, embedding, hidden, positions, eos_ids, max_new):
    import torch
    import torch.nn.functional as F
    from mllm_comp.analysis.probes.visual_token_phenomena import _run_decoder_suffix
    prefix, full_positions = hidden, positions
    state, past = hidden, None
    first_position, initial_length = int(positions.max()) + 1, hidden.shape[1]
    generated, steps = [], []
    for step in range(max_new):
        ids = positions if step == 0 else torch.full((3, 1, 1), first_position + step - 1, device=hidden.device, dtype=positions.dtype)
        output = llm(inputs_embeds=state, position_ids=ids, attention_mask=torch.ones((1, initial_length + step), device=hidden.device, dtype=torch.long),
                     past_key_values=past, use_cache=True, output_attentions=False, return_dict=True)
        cached = head(output.last_hidden_state[:, -1:])[0, -1].float()
        plain = _run_decoder_suffix(llm=llm, lm_head=head, hidden=prefix, position_ids=full_positions, start_layer=0, last_token_only=True)[0, -1].float()
        if not bool(torch.isfinite(cached).all() and torch.isfinite(plain).all()):
            raise ValueError("Nonfinite cache parity logits.")
        a, b = F.log_softmax(cached, -1), F.log_softmax(plain, -1)
        chosen, alternate = int(cached.argmax()), int(plain.argmax())
        steps.append({"step": step, "cached_top1": chosen, "cache_free_top1": alternate,
                      "max_abs_logit_difference": float((cached - plain).abs().max()),
                      "kl": float((a.exp() * (a - b)).sum().clamp_min(0)),
                      "max_top1_regret": max(float(cached.max() - cached[alternate]), float(plain.max() - plain[chosen]))})
        generated.append(chosen)
        if chosen in eos_ids or step + 1 == max_new:
            break
        past = output.past_key_values
        if past is None or past.get_seq_length() != initial_length + step:
            raise ValueError("Cache parity sequence length mismatch.")
        state = embedding(torch.tensor([[chosen]], device=hidden.device))
        prefix = torch.cat((prefix, state), dim=1)
        full_positions = torch.cat((full_positions, torch.full((3, 1, 1), first_position + step, device=hidden.device, dtype=positions.dtype)), dim=2)
    return {"generated_token_ids": generated, "steps": steps, "same_prefix_comparison": True,
            "max_kl": max(s["kl"] for s in steps), "max_abs_logit_difference": max(s["max_abs_logit_difference"] for s in steps),
            "max_top1_regret": max(s["max_top1_regret"] for s in steps)}


def verify_bf16_boundary(*, llm, head, embedding, hidden, positions, eos_ids, max_new):
    """A BF16 greedy tie is acceptable only after FP32 lifecycle verification.

    Converting BF16 weights to FP32 and back is exact. No optimizer or weight
    update occurs. Both modules are restored even if reference checking fails.
    """
    import torch
    kwargs = dict(llm=llm, head=head, embedding=embedding, positions=positions, eos_ids=eos_ids, max_new=max_new)
    bf16 = compare_cache_logits(hidden=hidden, **kwargs)
    llm_dtype = next(llm.parameters()).dtype
    head_dtype = next(head.parameters()).dtype
    # In particular M-RoPE inv_freq must remain FP32 after restoring weights;
    # a blanket module.to(BF16) would otherwise silently quantize its buffer.
    buffers = [(module, {name: value.clone() for name, value in module.named_buffers()}) for module in [llm, head]]
    try:
        llm.float()
        head.float()
        fp32 = compare_cache_logits(hidden=hidden.float(), **kwargs)
    finally:
        llm.to(dtype=llm_dtype)
        head.to(dtype=head_dtype)
        for module, original in buffers:
            for name, value in original.items():
                parent_name, _, attribute = name.rpartition(".")
                parent = module.get_submodule(parent_name) if parent_name else module
                setattr(parent, attribute, value)
    passed = (bf16["max_kl"] <= .01 and bf16["max_top1_regret"] <= .25
              and fp32["max_kl"] <= 1e-6 and fp32["max_abs_logit_difference"] <= .002)
    return {"passed": passed, "bf16": bf16, "fp32": fp32,
            "limits": {"bf16_kl": .01, "bf16_top1_regret": .25, "fp32_kl": 1e-6, "fp32_abs_logits": .002},
            "weight_dtype_restored": str(next(llm.parameters()).dtype) == str(llm_dtype)}
