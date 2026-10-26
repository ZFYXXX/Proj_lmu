"""Re-evaluate only the three audited external controls, from a frozen snapshot."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
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
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import sample_rows, task_contract, method_contract

METHODS = ('mmtok', 'prunesid', 'eadp')
TASKS = ('chartqa', 'ai2d', 'ocrbench')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, reference, audit):
    old = read(reference / 'protocol.json')
    if dependency_identity() != old['dependencies']:
        raise ValueError('Dependencies changed relative to existing comparisons')
    validate_identity(read(old['identity_path']))
    if read(audit / 'status.json') != {'status': 'complete', 'images': 24}:
        raise ValueError('Complete the real-feature audit first')
    checked = read(audit / 'results.json')
    for row in checked:
        for dtype in ('torch.bfloat16', 'torch.float32'):
            if not all(row['selectors'][dtype][m]['same_set'] for m in METHODS):
                raise ValueError('Real-feature official selected-set mismatch')
            if not row['selectors'][dtype]['prunesid']['same_order']:
                raise ValueError('Real-feature official PruneSID order mismatch')
    source = source_identity(ROOT)
    repo = reference.parents[3]
    # Resolve old source/result pointers without rewriting those experiments.
    prior = repo / 'outputs/eval/qwen25_vl/learned_visual_topk_step320_20261023_v1'
    external = repo / 'outputs/eval/qwen25_vl/context_topk_ai2d_external_20261024_v1'
    cells = []
    base_identity = read(old['identity_path'])['identity_sha256']
    for task in TASKS:
        for method in METHODS:
            pointer = (external if task == 'ai2d' else prior) / 'eval/full' / method / 'complete.json'
            saved = read(pointer)
            if file_sha(saved['config']) != saved['config_sha256']:
                raise ValueError('Historical config hash changed')
            cfg = yaml.safe_load(Path(saved['config']).read_text())
            old_task = Path(cfg['eval']['args']['include_path']) / f'{task}.yaml'
            name = f'context_topk_{task}' if task == 'ai2d' else f'learned_topk_{task}'
            samples, _ = sample_rows(Path(saved['run']), name, context.COUNTS[task], context.METRICS[task])
            expected = context.eval_config(reference, old, method, 'full', root / 'unused')
            if method_contract(cfg, base_identity) != method_contract(expected, base_identity):
                raise ValueError(f'Historical method settings differ: {method}/{task}')
            if task_contract(old_task, cfg) != task_contract(ROOT / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml', expected):
                raise ValueError(f'Historical task protocol differs: {method}/{task}')
            cells.append({'method': method, 'dataset': task, 'old_run': saved['run'], 'old_task_name': name,
                'old_samples': str(samples), 'old_samples_sha256': file_sha(samples),
                'old_config': saved['config'], 'old_config_sha256': saved['config_sha256']})
    root.mkdir(parents=True, exist_ok=True)
    protocol = {'source': source, 'dependencies': old['dependencies'], 'reference': str(reference),
        'reference_protocol_sha256': file_sha(reference / 'protocol.json'),
        'identity_path': old['identity_path'], 'base_identity_sha256': base_identity,
        'audit': str(audit), 'audit_results_sha256': file_sha(audit / 'results.json'),
        'cells': cells, 'full_requests': 3 * sum(context.COUNTS.values()),
        'smoke_requests': 3 * 3 * 8, 'target_count': 128,
        'generation': old['eval']['generation'],
        'corrections': ['EADP input-dtype visual normalization and input-dtype text mean',
            'MMTok input-dtype normalization before FP32 coverage',
            'PruneSID input-dtype NMS threshold, official global score sorting and output order',
            'PruneSID embedding/position/mask/label gather preserves that output order'],
        'preserved': ['same Qwen2.5-VL-7B weights, BF16, eager, seed42',
            'same native dynamic resolution, JPEG85, task prompts, greedy generation and scoring',
            'same fixed min(128,N), not the papers\' ratio protocols',
            'old predictions and existing own-method results remain untouched; no training'],
        'limits': ['Controlled Qwen adaptations with corrected kernels, not paper-number reproduction certification.',
            'AI2D strict and leading-option results remain separate.',
            'Existing own-method scores can be reused for quality, not for matched latency claims.']}
    path = root / 'protocol.json'
    if path.exists() and read(path) != protocol:
        raise ValueError('Immutable re-evaluation protocol changed; use a new run directory')
    write_json(path, protocol)
    if not (root / 'manifest.json').exists():
        write_json(root / 'manifest.json', {'status': 'prepared', 'stage': 'smoke', 'source': source,
            'protocol_sha256': file_sha(path), 'created_utc': context.native.now(), 'completed_full_cells': []})
    return protocol


def check_runtime(run, method):
    for line in (run / 'compression_runtime.requests.jsonl').read_text().splitlines():
        activation, = json.loads(line)['runtime']['activations']
        delta = activation.get('selection') or activation.get('latest_layout_delta')
        if method == 'prunesid':
            if delta['gathered_visual_indices'] != delta['selected_indices'] or delta['position_adapter'] != 'official_order_original_sparse_mrope':
                raise ValueError('PruneSID official order did not reach the decoder')


def summary(root, protocol):
    cohort = read(Path(protocol['reference']) / 'cohort.json')
    records = []
    for cell in protocol['cells']:
        method, task = cell['method'], cell['dataset']
        pointer = root / 'eval/full' / method / task / 'complete.json'
        if not pointer.exists():
            continue
        run = Path(read(pointer)['run'])
        metric = context.METRICS[task]
        _, rows = sample_rows(run, f'context_topk_{task}', context.COUNTS[task], metric)
        _, previous = sample_rows(Path(cell['old_run']), cell['old_task_name'], context.COUNTS[task], metric)
        if any((x['input'], x['target']) != (y['input'], y['target']) for x, y in zip(rows, previous)):
            raise ValueError('Paired old/new prompts or targets differ')
        new = [float(r[metric]) for r in rows]
        old = [float(r[metric]) for r in previous]
        ids = [r['image_fingerprint'] for r in cohort[task]['rows']]
        item = {'method': method, 'dataset': task, 'questions': len(rows),
            'old_score': sum(old) / len(old), 'corrected_score': sum(new) / len(new),
            'paired_image_bootstrap': context.clustered_difference(new, old, ids), 'run': str(run)}
        if task == 'ai2d':
            item['old_leading_option'] = sum(float(r['leading_option_accuracy']) for r in previous) / len(previous)
            item['corrected_leading_option'] = sum(float(r['leading_option_accuracy']) for r in rows) / len(rows)
        records.append(item)
    write_json(root / 'summary.json', {'complete_cells': len(records), 'total_cells': 9, 'comparisons': records,
        'limits': protocol['limits']})
    lines = ['# Corrected external controls — fixed K128', '',
        f'Completed full method/task cells: {len(records)}/9. Incomplete cells are not assigned scores.', '',
        '| Task | Method | Previous | Corrected | Delta pp | 95% image-bootstrap CI pp |',
        '|---|---|---:|---:|---:|---:|']
    for row in records:
        paired = row['paired_image_bootstrap']
        lines.append(f"| {row['dataset']} | {row['method']} | {100*row['old_score']:.2f}% | {100*row['corrected_score']:.2f}% | {100*paired['delta']:+.2f} | [{100*paired['ci95'][0]:+.2f}, {100*paired['ci95'][1]:+.2f}] |")
    lines += ['', 'OCRBench percentages are normalized official points/1000. AI2D leading-option scores are auxiliary fields in summary.json.', '', *protocol['limits']]
    write_text(root / 'report.md', '\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--audit', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = owned_output(args.root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = prepare(root, args.reference.resolve(), args.audit.resolve())
        if args.prepare_only:
            print('Prepared 9 full cells / 19764 requests; smoke required first.', flush=True)
            return
        if len(set(args.gpus)) != len(args.gpus) or not set(args.gpus) <= {0, 1}:
            raise ValueError('Invalid or duplicate GPUs')
        mutex = Lock()
        failed = Event()
        manifest = read(root / 'manifest.json')
        manifest.update(status='running', started_utc=context.native.now(), gpus=args.gpus)
        write_json(root / 'manifest.json', manifest)

        def phase(name):
            jobs = Queue()
            for cell in protocol['cells']:
                jobs.put(cell)
            def worker(gpu):
                while not failed.is_set():
                    try:
                        cell = jobs.get_nowait()
                    except Empty:
                        return
                    method, task = cell['method'], cell['dataset']
                    parent = root / 'eval' / name / method / task
                    try:
                        marker = parent / 'complete.json'
                        if marker.exists():
                            saved = read(marker)
                            if saved['source'] != protocol['source'] or file_sha(saved['config']) != saved['config_sha256']:
                                raise ValueError('Completed cell provenance changed')
                            context.native.check_eval(Path(saved['run']), method, name, tasks=(task,), counts=context.COUNTS, prefix='context_topk')
                            check_runtime(Path(saved['run']), method)
                            continue
                        while available_gpus([gpu]) != [gpu]:
                            if failed.wait(30):
                                return
                            print(f'GPU{gpu}: waiting for safe free memory; no other process modified', flush=True)
                        if source_identity(ROOT) != protocol['source']:
                            raise ValueError('Frozen execution source changed')
                        run = parent / f'attempt_{time.time_ns()}'
                        cfg = context.eval_config(Path(protocol['reference']), read(Path(protocol['reference']) / 'protocol.json'), method, name, run)
                        cfg['eval']['tasks'] = [f'context_topk_{task}']
                        cfg['eval']['sample_limit'] = 8 if name == 'smoke' else None
                        config = root / 'configs' / f'{name}_{method}_{task}_{time.time_ns()}.yaml'
                        write_text(config, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                        print(f'START GPU{gpu}: {name}/{method}/{task}', flush=True)
                        context.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config)],
                            parent / f'launch_{time.time_ns()}.log', [gpu], {
                                'MLLM_TOPK_COHORT': str(Path(protocol['reference']) / 'cohort.json'),
                                'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                                'HF_HOME': str(root / 'cache/huggingface'),
                                'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
                                'XDG_CACHE_HOME': str(root / 'cache/xdg')})
                        check = context.native.check_eval(run, method, name, tasks=(task,), counts=context.COUNTS, prefix='context_topk')
                        check_runtime(run, method)
                        write_json(marker, {'run': str(run), 'requests': check['requests'], 'config': str(config),
                            'config_sha256': file_sha(config), 'source': protocol['source'], 'completed_utc': context.native.now()})
                        print(f'COMPLETE GPU{gpu}: {name}/{method}/{task}', flush=True)
                        with mutex:
                            if name == 'full':
                                manifest['completed_full_cells'].append(f'{method}/{task}')
                                summary(root, protocol)
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
            write_json(root / f'{name}_complete.json', {'status': 'complete', 'cells': 9, 'time': context.native.now()})
        try:
            phase('smoke')
            manifest.update(stage='full', updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            phase('full')
            summary(root, protocol)
            manifest.update(status='complete', stage='complete', finished_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc), updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            raise


if __name__ == '__main__':
    main()
