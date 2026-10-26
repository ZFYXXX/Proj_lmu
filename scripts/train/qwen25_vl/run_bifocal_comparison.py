"""Queue Bifocal Stage-1 K128 and Stage-1+2 r25, with native eval only."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event, Lock
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import source_identity, dependency_identity, owned_output, available_gpus
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_context_visual_topk as context
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import sample_rows, task_contract

METHODS = ('bifocal_stage1_k128', 'bifocal_stage12_r25')
TASKS = ('chartqa', 'ocrbench', 'ai2d')


def read(path):
    return json.loads(Path(path).read_text())


def config(reference, protocol, method, phase, task, run):
    cfg = context.eval_config(reference, protocol, 'dense', phase, run)
    full = method == METHODS[1]
    algorithm = 'bifocal_two_stage' if full else 'bifocal'
    settings = ({'average_fraction': .25, 'stage2_layer': 15, 'stage2_fraction': 32/576,
        'stage2_min_keep': 8, 'density_weight': 1., 'stage1_config': {'vit_block': 31}} if full else
        {'target_count': 128, 'vit_block': 31})
    cfg['pipeline']['stages'] = [{'name': method, 'plugin': algorithm, 'algorithm': algorithm,
        'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl',
        'placement': 'llm_level' if full else 'projector_level', 'method': 'token_compression',
        'object_family': 'tensor_slot', 'execution_phase': 'generate',
        'modalities': ['image'], 'target_modalities': ['image'],
        'conditioning_modalities': ['text'], 'preserved_modalities': ['text', 'video', 'audio'],
        'coupling_policy': 'independent', 'media_alignment': 'spatial', 'config': settings}]
    cfg['experiment']['name'] = f'{method}_{phase}_{task}'
    cfg['eval'].update(tasks=[f'context_topk_{task}'], sample_limit=8 if phase == 'smoke' else None,
        compression_mode='compressed', target='compressed')
    return cfg


def check(run, method, task, phase):
    n = 8 if phase == 'smoke' else context.COUNTS[task]
    task_name = f'context_topk_{task}'
    _, rows = sample_rows(run, task_name, n, context.METRICS[task])
    files = list(run.glob('**/*_results.json'))
    if len(files) != 1:
        raise ValueError('Missing or duplicate native result file')
    metrics = read(files[0])['results'][task_name]
    metric_key = next(k for k in metrics if k.split(',')[0] == context.METRICS[task])
    score = sum(float(r[context.METRICS[task]]) for r in rows) / n
    if abs(score - metrics[metric_key]) > 1e-10:
        raise ValueError('Native aggregate score and predictions disagree')
    performance = [json.loads(s) for s in (run / 'request_performance.jsonl').read_text().splitlines()]
    traces = [json.loads(s) for s in (run / 'compression_runtime.requests.jsonl').read_text().splitlines()]
    def key(value):
        parts = value.split(':')
        if parts[0] != task_name:
            raise ValueError('Unexpected request task')
        return int(parts[2])
    perf = {key(r['sample_id']): r for r in performance}
    indexed = {key(r['request']['sample_id']): r for r in traces}
    if len(perf) != len(performance) or len(indexed) != len(traces) or set(perf) != set(range(n)) or set(indexed) != set(perf):
        raise ValueError('Incomplete, duplicate or unknown performance/trace requests')
    token_rows = []
    for idx in range(n):
        p = perf[idx]
        if p['status'] != 'complete' or not math.isfinite(p['generation_s']):
            raise ValueError('Failed or nonfinite timed generation request')
        activation, = indexed[idx]['runtime']['activations']
        delta = activation['latest_layout_delta']
        source, first = delta['source_count'], delta['result_count']
        reps = delta['bifocal_representatives']
        if source != p['original_visual_tokens'] or len(reps) != first or len(set(reps)) != first or any(not 0 <= i < source for i in reps):
            raise ValueError('Bifocal Stage-1 source/budget/membership mismatch')
        lifecycle = activation['projector_generate']['last']
        if lifecycle['decode_position_verified'] is False or (lifecycle['decode_steps'] and lifecycle['decode_position_verified'] is not True):
            raise ValueError('Bifocal Stage-1 decode position verification failed')
        final, average, ratio = first, float(first), first / source
        if method == METHODS[0]:
            if first != min(128, source):
                raise ValueError('Stage-1-only output must be min(128,N)')
        else:
            stage2 = activation['stage2']
            expected_first = round(source * (0.25 * 28 - 13 * (32/576)) / 15)
            expected_final = min(first, max(8, round(source * 32/576)))
            if first != expected_first or stage2['stage2_count'] != expected_final:
                raise ValueError('Two-stage actual budgets differ from the frozen schedule')
            final = stage2['stage2_count']
            counts = stage2['observed_visual_tokens_per_decoder_layer']
            if counts != [first] * 15 + [final] * 13:
                raise ValueError('Two-stage observed layer token schedule differs')
            if stage2['decode_position_verified'] is False or (stage2['decode_steps'] and stage2['decode_position_verified'] is not True):
                raise ValueError('Two-stage decode position verification failed')
            average, ratio = sum(counts) / 28, sum(counts) / (28 * source)
        token_rows.append({'doc_id': idx, 'source': source, 'stage1': first,
            'stage2': final, 'layer_average': average, 'layer_average_ratio': ratio})
    result = {'status': 'complete', 'method': method, 'task': task, 'phase': phase,
        'requests': n, 'score': score, 'token_counts': token_rows,
        'mean_original_tokens': sum(r['source'] for r in token_rows) / n,
        'mean_stage1_tokens': sum(r['stage1'] for r in token_rows) / n,
        'mean_stage2_tokens': sum(r['stage2'] for r in token_rows) / n,
        'mean_layer_average_tokens': sum(r['layer_average'] for r in token_rows) / n,
        'mean_layer_average_ratio': sum(r['layer_average_ratio'] for r in token_rows) / n,
        'generation_seconds_sum': sum(r['generation_s'] for r in performance),
        'peak_allocated_bytes': max(r['peak_allocated_bytes'] or 0 for r in performance)}
    if task == 'ai2d':
        result['leading_option_auxiliary'] = sum(float(r['leading_option_accuracy']) for r in rows) / n
    write_json(run / 'protocol_checks.json', result)
    return result


def prepare(root, reference, prerequisite):
    if os.environ.get('MLLM_COMP_METHOD_PROFILE') != 'full':
        raise ValueError('This experiment explicitly requires MLLM_COMP_METHOD_PROFILE=full for Stage 2')
    old = read(reference / 'protocol.json')
    identity = read(old['identity_path'])
    validate_identity(identity)
    if dependency_identity() != old['dependencies']:
        raise ValueError('Evaluator dependencies differ from existing results')
    for task in TASKS:
        prior = Path(old['source']['source_root']) / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        current = ROOT / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        cfg = config(reference, old, METHODS[0], 'full', task, root / 'unused')
        if task_contract(prior, cfg) != task_contract(current, cfg):
            raise ValueError('Native task prompt/generation/scoring contract changed')
    ref_source = Path('/home/YinPeiran/vlm_model/qwen2_5_vl/modeling_qwen2_5_vl.py')
    protocol = {'source': source_identity(ROOT), 'dependencies': old['dependencies'],
        'reference': str(reference), 'reference_protocol_sha256': file_sha(reference / 'protocol.json'),
        'identity_path': old['identity_path'], 'base_identity_sha256': identity['identity_sha256'],
        'wait_for': str(prerequisite), 'methods': list(METHODS), 'tasks': list(TASKS),
        'method_profile': 'full (only for this process; default projector profile is unchanged)',
        'full_requests': 2 * sum(context.COUNTS.values()), 'smoke_requests': 48,
        'reference_stage2_source': str(ref_source), 'reference_stage2_sha256': file_sha(ref_source),
        'reference_access': 'read only; source/method parity test, not full-model replacement',
        'stage1_budget': 'min(128,N); projector only; NOT full Bifocal',
        'two_stage_budget': {'average_fraction': .25, 'layers': 28, 'boundary': 15,
            'stage1_fraction': (28 * .25 - 13 * (32/576)) / 15,
            'stage2_fraction': 32/576, 'stage2_min_keep': 8},
        'preserved': ['Qwen2.5-VL-7B/BF16/eager/native resolution/JPEG85', 'same native task prompts, greedy generation and scoring',
            'no training, no changes to own-method K128 or existing external evaluation source/results'],
        'limits': ['Stage-1 K128 is a component control, not the full paper method.',
            'Two-stage r25 has different per-image and per-layer budgets; do not rank as a fixed K128 projector-only method.',
            'The two rows are NOT a matched-budget causal ablation of Stage 2.',
            'Controlled framework adaptations, not certified paper-number reproduction.',
            'Twenty-three optional original Stage-1 parity tests lack the old local reference; do not claim them passed.',
            'Stage-2 FP32/BF16 operator matches the read-only Qwen reference; native cache/identity tests cover the new loop.']}
    path = root / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Source/config/dependencies changed; choose a new experiment directory')
    write_json(path, protocol)
    if not (root / 'manifest.json').exists():
        write_json(root / 'manifest.json', {'status': 'prepared', 'source': protocol['source'],
            'protocol_sha256': file_sha(path), 'completed_cells': [], 'created_utc': context.native.now()})
    return protocol


def report(root, protocol):
    records = []
    reference = Path(protocol['reference'])
    plan = read(reference / 'evaluation_plan.json')
    cohort = read(reference / 'cohort.json')
    for task in TASKS:
        dense = next(c for c in plan['cells'] if (c['method'], c['dataset']) == ('dense', task))
        _, baseline = sample_rows(Path(dense['run']), dense['task_name'], context.COUNTS[task], context.METRICS[task])
        for method in METHODS:
            pointer = root / 'eval/full' / method / task / 'complete.json'
            if not pointer.exists():
                continue
            run = Path(read(pointer)['run'])
            current = read(run / 'protocol_checks.json')
            _, rows = sample_rows(run, f'context_topk_{task}', context.COUNTS[task], context.METRICS[task])
            if any((a['input'], a['target']) != (b['input'], b['target']) for a, b in zip(rows, baseline)):
                raise ValueError('Bifocal and reused dense prompts/targets differ')
            current.pop('token_counts')
            current['versus_dense_image_bootstrap'] = context.clustered_difference(
                [r[context.METRICS[task]] for r in rows], [r[context.METRICS[task]] for r in baseline],
                [r['image_fingerprint'] for r in cohort[task]['rows']])
            current['run'] = str(run)
            records.append(current)
    write_json(root / 'summary.json', {'completed_cells': len(records), 'total_cells': 6,
        'results': records, 'limits': protocol['limits']})
    lines = ['# Bifocal component and two-stage controls', '', f'Complete full cells: {len(records)}/6.', '',
        '| Task | Version | Score | Mean Stage-1 tokens | Mean Stage-2 tokens | Mean layer-average tokens |',
        '|---|---|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['task']} | {r['method']} | {100*r['score']:.2f}% | {r['mean_stage1_tokens']:.2f} | {r['mean_stage2_tokens']:.2f} | {r['mean_layer_average_tokens']:.2f} |")
    lines += ['', 'OCRBench score is normalized official points/1000. AI2D headline is strict matching; leading-option is auxiliary in JSON.', '', *protocol['limits']]
    write_text(root / 'report.md', '\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--wait-for', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = owned_output(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(root, args.reference.resolve(), args.wait_for.resolve())
        if args.prepare_only:
            print('Prepared six full cells; two explicitly different Bifocal budgets.', flush=True)
            return
        if not args.gpus or len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {0, 1}:
            raise ValueError('Invalid GPU selection')
        manifest = read(root / 'manifest.json')
        mutex, failed = Lock(), Event()
        try:
            manifest.update(status='queued', stage='waiting_for_external_reruns', gpus=args.gpus, updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            while not (args.wait_for / 'full_complete.json').exists():
                if read(args.wait_for / 'manifest.json')['status'] == 'failed':
                    raise RuntimeError('Earlier external comparison failed; Bifocal queue stops without taking its GPUs')
                failed.wait(20)
            if read(args.wait_for / 'manifest.json')['status'] not in ('running', 'complete'):
                raise ValueError('Unexpected prerequisite state')

            def phase(name):
                manifest.update(status='running', stage=name, updated_utc=context.native.now())
                write_json(root / 'manifest.json', manifest)
                jobs = Queue()
                for task in TASKS:
                    for method in METHODS:
                        jobs.put((method, task))
                def worker(gpu):
                    while not failed.is_set():
                        try:
                            method, task = jobs.get_nowait()
                        except Empty:
                            return
                        parent = root / 'eval' / name / method / task
                        try:
                            marker = parent / 'complete.json'
                            if marker.exists():
                                saved = read(marker)
                                if saved['source'] != protocol['source'] or file_sha(saved['config']) != saved['config_sha256']:
                                    raise ValueError('Existing Bifocal completion provenance changed')
                                check(Path(saved['run']), method, task, name)
                                continue
                            while available_gpus([gpu]) != [gpu]:
                                if failed.wait(20):
                                    return
                            if source_identity(ROOT) != protocol['source']:
                                raise ValueError('Frozen Bifocal source changed')
                            run = parent / f'attempt_{time.time_ns()}'
                            cfg = config(args.reference.resolve(), read(args.reference / 'protocol.json'), method, name, task, run)
                            filename = root / 'configs' / f'{name}_{method}_{task}_{time.time_ns()}.yaml'
                            write_text(filename, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                            print(f'START GPU{gpu}: {name}/{method}/{task}', flush=True)
                            context.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(filename)],
                                parent / f'launch_{time.time_ns()}.log', [gpu], {
                                    'MLLM_COMP_METHOD_PROFILE': 'full',
                                    'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                                    'MLLM_TOPK_COHORT': str(args.reference / 'cohort.json'),
                                    'HF_HOME': str(root / 'cache/huggingface'), 'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
                                    'XDG_CACHE_HOME': str(root / 'cache/xdg')})
                            validated = check(run, method, task, name)
                            write_json(marker, {'run': str(run), 'source': protocol['source'], 'requests': validated['requests'],
                                'config': str(filename), 'config_sha256': file_sha(filename), 'completed_utc': context.native.now()})
                            print(f'COMPLETE GPU{gpu}: {name}/{method}/{task}', flush=True)
                            with mutex:
                                if name == 'full':
                                    manifest['completed_cells'].append(f'{method}/{task}')
                                    report(root, protocol)
                                manifest.update(updated_utc=context.native.now())
                                write_json(root / 'manifest.json', manifest)
                        except BaseException:
                            failed.set()
                            write_json(parent / f'failure_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
                            raise
                        finally:
                            jobs.task_done()
                with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
                    for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
                        future.result()
                write_json(root / f'{name}_complete.json', {'status': 'complete', 'cells': 6, 'time': context.native.now()})
            phase('smoke')
            phase('full')
            report(root, protocol)
            manifest.update(status='complete', stage='complete', finished_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc), updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            raise


if __name__ == '__main__':
    main()
