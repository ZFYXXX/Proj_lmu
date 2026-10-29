"""Read-only model audit: distinguish state errors from BF16 LM-head shape effects.

Imports the exact failed experiment's frozen source. Never changes its source,
protocol, teacher labels, checkpoint or status. Writes separate diagnostic data.
"""
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text())


def comparison(a, b):
    import torch
    if a.shape != b.shape:
        return {'same_shape': False, 'left_shape': list(a.shape), 'right_shape': list(b.shape)}
    x, y = a.float(), b.float()
    close = torch.isclose(x, y, rtol=1e-3, atol=1e-3)
    return {'same_shape': True, 'bitwise_equal': torch.equal(a, b),
            'finite': bool(torch.isfinite(x).all() and torch.isfinite(y).all()),
            'max_abs': float((x-y).abs().max()), 'mean_abs': float((x-y).abs().mean()),
            'failed_original_tolerance': int((~close).sum()), 'elements': x.numel()}


@contextmanager
def capture_llm(llm):
    captured = {}
    def before(module, args, kwargs):
        for key in ('inputs_embeds', 'position_ids', 'attention_mask'):
            if kwargs.get(key) is not None:
                captured[key] = kwargs[key].detach().clone()
    def after(module, args, output):
        captured['last_hidden_state'] = output.last_hidden_state.detach().clone()
    pre = llm.register_forward_pre_hook(before, with_kwargs=True)
    post = llm.register_forward_hook(after)
    try:
        yield captured
    finally:
        pre.remove(); post.remove()


def audit(args):
    p = read(args.run/'protocol.json')
    source = Path(p['source']['source_root'])
    sys.path.insert(0, str(source))
    import torch
    import torch.nn.functional as F
    from mllm_comp.analysis.common.anchor7b_runtime import load_runtime, file_sha
    from mllm_comp.analysis.common.learned_topk_experiment import gpu_processes, gpu_free_mib, owned_output
    from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions
    from mllm_comp.core.serialization import atomic_write_json as write
    from scripts.train.qwen25_vl.run_query_interaction_2x2 import validate
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples, prepare_example
    output = owned_output(args.output)
    output.mkdir(parents=True, exist_ok=False)
    lockpath = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main/outputs/runtime/qwen25_gpu_locks/gpu0.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
            raise ValueError('Audit must explicitly select only physical GPU0')
        if any(r['uid'] == os.getuid() and r['gpu'] == 0 for r in gpu_processes()) or gpu_free_mib()[0] < 46000:
            raise RuntimeError('GPU0 busy; no diagnostic model loaded')
        validate(args.run)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.use_deterministic_algorithms(True)
        started = time.time()
        report = {'status': 'running', 'run': str(args.run), 'source': p['source'],
                  'protocol_sha256': file_sha(args.run/'protocol.json'), 'indices': args.indices,
                  'torch': torch.__version__, 'rows': [], 'production_artifacts_modified': False}
        write(output/'audit.json', report)
        bundle = load_runtime(read(p['identity_path']), torch.device('cuda:0'))
        model, llm = bundle['model'], bundle['llm']
        examples = load_examples(Path(p['training_manifest']))
        with torch.inference_mode():
            for index in args.indices:
                example = examples[index]
                state = prepare_example(bundle, example)
                predictors, targets = answer_predictor_positions(state['teacher_inputs']['labels'])
                inputs = {k:v for k,v in state['teacher_inputs'].items() if k != 'labels'}
                # Native full-sequence head vs the reduced-M head in answer_logits.
                with capture_llm(llm) as native:
                    result = model(**inputs, use_cache=False)
                    native_logits = result.logits[0, predictors].float()
                    del result
                with capture_llm(llm) as manual:
                    result = llm(inputs_embeds=state['hidden'], position_ids=state['positions'],
                                 attention_mask=torch.ones(state['hidden'].shape[:2], dtype=torch.long, device='cuda:0'),
                                 use_cache=False, output_attentions=False, return_dict=True)
                    selected_logits = model.lm_head(result.last_hidden_state[:, predictors])[0].float()
                    full_shape_logits = model.lm_head(result.last_hidden_state)[0, predictors].float()
                    del result
                # Official logits_to_keep applies exactly the same answer-row gather
                # before lm_head, without changing the decoder or visual inputs.
                matched_logits = model(**inputs, use_cache=False,
                    logits_to_keep=torch.tensor(predictors, device='cuda:0')).logits[0].float()
                a = F.linear(manual['last_hidden_state'][:, predictors].float(), model.lm_head.weight.float())[0]
                b = F.linear(native['last_hidden_state'][:, predictors].float(), model.lm_head.weight.float())[0]
                target = torch.tensor(targets, device='cuda:0')
                row = {'training_index': index, 'sample_id': example['sample_id'],
                    'sequence_length': state['hidden'].shape[1], 'answer_length': len(predictors),
                    'visual_tokens': len(state['visual']),
                    'decoder_inputs': {key: comparison(manual[key], native[key])
                        for key in ('inputs_embeds', 'position_ids', 'attention_mask')},
                    'decoder_final_hidden': comparison(manual['last_hidden_state'], native['last_hidden_state']),
                    'original_check': comparison(selected_logits, native_logits),
                    'same_full_shape_head': comparison(full_shape_logits, native_logits),
                    'official_matched_shape_head': comparison(selected_logits, matched_logits),
                    'fp32_head_same_shape': comparison(a, b),
                    'answer_nll_original_native': float(F.cross_entropy(native_logits, target)),
                    'answer_nll_selected': float(F.cross_entropy(selected_logits, target)),
                    'argmax_flips': int((selected_logits.argmax(-1) != native_logits.argmax(-1)).sum()),
                    'teacher_kl_selected_to_native': float(F.kl_div(F.log_softmax(selected_logits, -1),
                        F.softmax(native_logits, -1), reduction='batchmean'))}
                report['rows'].append(row)
                write(output/'audit.json', report)
                print(json.dumps(row), flush=True)
                del native, manual, state, native_logits, selected_logits, full_shape_logits, matched_logits, a, b
        report.update(status='complete', elapsed_s=time.time()-started,
                      peak_allocated_bytes=torch.cuda.max_memory_allocated())
        report['localization'] = {
            'all_decoder_states_bitwise_equal': all(r['decoder_final_hidden']['bitwise_equal'] and
                all(v['bitwise_equal'] for v in r['decoder_inputs'].values()) for r in report['rows']),
            'all_original_full_shape_heads_bitwise_equal': all(r['same_full_shape_head']['bitwise_equal'] for r in report['rows']),
            'all_official_matched_shape_heads_bitwise_equal': all(r['official_matched_shape_head']['bitwise_equal'] for r in report['rows']),
            'all_fp32_heads_bitwise_equal': all(r['fp32_head_same_shape']['bitwise_equal'] for r in report['rows']),
            'any_original_check_failed': any(r['original_check']['failed_original_tolerance'] for r in report['rows'])}
        write(output/'audit.json', report)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 1, 2, 3, 10, 11, 2048, 2049])
    audit(parser.parse_args())
