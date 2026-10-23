"""Isolated CUDA kernel measurements on actual encoder Q/K/V, not suite latency."""
from __future__ import annotations
from contextlib import contextmanager
import numpy as np
import torch
from .interventions import range_mask


@contextmanager
def capture_qkv(tower, layer, enabled):
    values, handles = {}, []
    if enabled:
        attention = tower.layers[layer].self_attn
        for name in ('q', 'k', 'v'):
            def hook(module, args, output, name=name):
                values[name] = output.detach().reshape(len(output), attention.num_heads, -1)
            handles.append(getattr(attention, name+'_proj').register_forward_hook(hook))
        values['scaling'] = attention.scaling
    try:
        yield values
    finally:
        for handle in handles:
            handle.remove()


def dense_masked(q, k, v, boundaries, scaling, masks):
    pieces = []
    for i, (a, b) in enumerate(zip(boundaries, boundaries[1:])):
        qc, kc, vc = [t[a:b].transpose(0, 1) for t in (q, k, v)]
        logits = (qc @ kc.transpose(-1, -2)) * scaling
        p = torch.softmax(logits.masked_fill(~masks[i], -torch.inf), -1, dtype=torch.float32)
        pieces.append((p.to(v.dtype) @ vc).transpose(0, 1))
    return torch.cat(pieces)


@torch.no_grad()
def benchmark_reference(values, boundaries, step_ms, windows_ms, repeats=10):
    q, k, v = [values[name].contiguous() for name in ('q', 'k', 'v')]
    if q.device.type != 'cuda':
        return {'status': 'not_measured', 'reason': 'FlashAttention-2 kernel comparison requires CUDA'}
    from flash_attn import flash_attn_varlen_func
    scaling = values['scaling']
    cu = torch.tensor(boundaries, dtype=torch.int32, device=q.device)
    maximum = int(max(np.diff(boundaries)))
    def flash(radius):
        return flash_attn_varlen_func(q, k, v, cu, cu, maximum, maximum, dropout_p=0,
            softmax_scale=scaling, causal=False, window_size=(radius, radius))
    def timed(fn):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(q.device)
        memory_before = torch.cuda.memory_allocated(q.device)
        torch.cuda.reset_peak_memory_stats(q.device)
        durations = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(torch.cuda.current_stream(q.device))
            result = fn()
            end.record(torch.cuda.current_stream(q.device))
            end.synchronize()
            durations.append(start.elapsed_time(end))
            del result
        return dict(median_ms=float(np.median(durations)), repeats=repeats,
                    peak_extra_allocated_bytes=max(0, torch.cuda.max_memory_allocated(q.device)-memory_before))
    rows = [{'backend': 'flash_attention_2_full', 'radius_ms': None, **timed(lambda: flash(-1))}]
    for window in sorted(set(windows_ms)):
        radius = int(window//step_ms)
        masks = [torch.as_tensor(range_mask(b-a, radius, 'window', 0), device=q.device)
                 for a, b in zip(boundaries, boundaries[1:])]
        dense = lambda: dense_masked(q, k, v, boundaries, scaling, masks)
        native = lambda: flash(radius)
        reference, sparse = dense(), native()
        error = (sparse.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12)
        relative_error = float(error)
        del reference, sparse
        rows.extend([
            dict(backend='dense_masked', radius_ms=window, **timed(dense)),
            dict(backend='flash_attention_2_window', radius_ms=window,
                 av_relative_l2_vs_dense=relative_error, **timed(native)),
        ])
    return dict(status='measured', scope='attention kernels only; excludes projections, masks construction and generation',
                device=torch.cuda.get_device_name(q.device), dtype=str(q.dtype),
                chunk_lengths=np.diff(boundaries).tolist(), rows=rows)
