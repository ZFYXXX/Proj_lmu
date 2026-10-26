"""Opt-in native request timing; disabled unless an experiment supplies a path."""
from contextlib import contextmanager
import json
import os
import time


def indexed_device(value):
    import torch
    device = torch.device(value)
    if device.type == 'cuda' and device.index is None:
        device = torch.device('cuda', torch.cuda.current_device())
    return device


@contextmanager
def request_performance(runtime, processed, sample_id):
    from .request_context import sample_request
    with sample_request(sample_id), _request_performance(runtime, processed, sample_id):
        yield


@contextmanager
def _request_performance(runtime, processed, sample_id):
    filename = os.environ.get('MLLM_TOPK_PERFORMANCE')
    if not filename:
        yield
        return
    import torch
    from mllm_comp.analysis.common.learned_topk_experiment import owned_output
    from mllm_comp.analysis.common.learned_topk_experiment import cooperative_pause, shared_policy
    path = owned_output(filename)
    device = indexed_device(runtime.device)
    cuda = device.type == 'cuda'
    policy = shared_policy()
    if cuda and policy:
        torch.cuda.set_per_process_memory_fraction(policy['allocator_fraction'], device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        if free_bytes < policy['minimum_free_mib_at_step'] * 1024**2:
            raise RuntimeError('Shared GPU headroom fell below 8 GiB; stopping evaluation')
    if cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    inputs = processed.model_inputs
    image_id = runtime.model.config.image_token_id
    original = int((inputs['input_ids'] == image_id).sum())
    def finite_logits(_module, _inputs, output):
        if not bool(torch.isfinite(output).all()):
            raise ValueError('Nonfinite native generation logits')
    head = getattr(runtime.model, 'lm_head', None)
    handle = head.register_forward_hook(finite_logits) if head is not None else None
    start, status = time.perf_counter(), 'failed'
    try:
        yield
        status = 'complete'
    finally:
        if handle is not None:
            handle.remove()
        if cuda:
            torch.cuda.synchronize(device)
        record = {'sample_id': sample_id, 'status': status, 'original_visual_tokens': original,
            'generation_s': time.perf_counter() - start,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if cuda else None,
            'scope': 'synchronized native generation with finite-logit validation; includes vision/compressor/prefill/decode, excludes CPU processor/model load'}
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a') as stream:
            stream.write(json.dumps(record, allow_nan=False) + '\n')
        if status == 'complete':
            cooperative_pause(record['generation_s'])
