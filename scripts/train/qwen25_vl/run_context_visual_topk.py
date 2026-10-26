"""Two matched context arms, then three-task native eval. No automatic commits.

Each GPU runs one normal, unthrottled process. The frozen visual selector is
warm-started in BOTH arms; an image-global context matches the question arm's
capacity and extra training budget. Recovery and dynamic K are deliberately off.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import fcntl
from io import BytesIO
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity, development_indices, owned_output, reject_overlap, shared_policy,
    source_identity, validate_training_entries,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
from scripts.train.qwen25_vl import run_learned_visual_topk as native

COUNTS = {'chartqa': 2500, 'ocrbench': 1000, 'ai2d': 3088}
METRICS = {'chartqa': 'relaxed_overall', 'ocrbench': 'ocrbench_accuracy', 'ai2d': 'exact_match'}
ARMS = ('image', 'question')
METHODS = ('dense', 'old_mlp', 'frozen_visual', 'image_context', 'question_context')


def read(path):
    return json.loads(Path(path).read_text())


def prepare(root, reference):
    import pyarrow.parquet as pq
    from PIL import Image
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    from mllm_comp.compressors.image.learned_visual_topk import load_selector
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
    if root.exists():
        raise ValueError('Preserve prior runs; choose a new root')
    old = read(reference / 'protocol.json')
    if dependency_identity() != old['dependencies']:
        raise ValueError('Evaluator/software drifted from the reference recipe')
    for filename, checksum in old['input_hashes'].items():
        if file_sha(filename) != checksum:
            raise ValueError(f'Reference input changed: {filename}')
    entries = read(old['training_manifest'])['entries']
    validate_training_entries(entries)
    identity = read(old['identity_path'])
    validate_identity(identity)
    if read(reference / 'training/manifest.json')['status'] != 'complete':
        raise ValueError('Warm-start training is not complete')
    checkpoint = native.latest(reference / 'training')
    if read(reference / 'training/latest.json')['step'] != 320:
        raise ValueError('Warm start must be the previously fixed step320')
    visual = load_selector(checkpoint, base_identity=identity['identity_sha256'], device='cpu')
    initial = parameter_digest(ContextVisualTopK(ContextVisualTopKSpec(visual.spec.hidden_size), visual=visual))
    cohort = read(reference / 'cohort.json')
    ai2d_paths = sorted(Path('/home/YinPeiran/datasets/lmms_eval_datasets/ai2d/data').glob('test-*.parquet'))
    if len(ai2d_paths) != 2:
        raise ValueError('Expected both AI2D test shards')
    rows = []
    for filename in ai2d_paths:
        for batch in pq.ParquetFile(filename).iter_batches(batch_size=16):
            for doc in batch.to_pylist():
                raw = doc['image']
                with Image.open(BytesIO(raw['bytes']) if raw.get('bytes') else raw['path']) as image:
                    fingerprint = media_fingerprint([image.convert('RGB')])
                rows.append({'row': len(rows), 'image_fingerprint': fingerprint})
    if len(rows) != COUNTS['ai2d']:
        raise ValueError('AI2D full split size changed')
    cohort['ai2d'] = {'paths': [str(p) for p in ai2d_paths], 'rows': rows,
        'independent_images': len({r['image_fingerprint'] for r in rows})}
    for task in COUNTS:
        reject_overlap(entries, cohort[task]['rows'])
        cohort[task]['development_indices'] = development_indices(cohort[task]['rows'], count=500, seed=20261024)
        print(f'{task}: {len(cohort[task]["rows"])} questions / {cohort[task]["independent_images"]} images; train overlap=0', flush=True)
    root.mkdir(parents=True)
    atomic_write_json(root / 'cohort.json', cohort)
    inputs = {**old['input_hashes'], str(reference / 'protocol.json'): file_sha(reference / 'protocol.json'),
        str(root / 'cohort.json'): file_sha(root / 'cohort.json')}
    for filename in [*ai2d_paths, checkpoint / 'selector.safetensors', checkpoint / 'selector_config.json']:
        inputs[str(filename)] = file_sha(filename)
    source = source_identity(ROOT)
    common = {**old, 'created_utc': native.now(), 'source': source, 'input_hashes': inputs,
        'selector_type': 'context_visual_topk', 'warmstart_checkpoint': str(checkpoint),
        'initial_parameter_sha256': initial, 'resource_policy': shared_policy(),
        'eval': {**old['eval'], 'counts': COUNTS, 'development_unique_images_per_task': 500,
            'max_new_tokens': {'chartqa': 16, 'ocrbench': 128, 'ai2d': 16}},
        'methods': list(METHODS), 'reference_train': str(reference),
        'claims': 'Single-seed method development; prompt conditioning is prior art, not a novelty claim',
        'limits': ['Two arms both warm-start the same visual selector and train one additional pass (320 steps).',
            'No recovery, dynamic budget, Qwen3.5, task labels, extra dense LLM prompt pass, or test tuning.',
            'Input-embedding mean is not a contextual question encoder; a null result does not reject all query-aware selectors.',
            'Dev/full are exploratory and overlap; no independent confirmation or model selection is claimed.']}
    common.pop('trained_run_root', None)
    common.pop('training_reference', None)
    for mode in ARMS:
        arm = root / 'arms' / mode
        arm.mkdir(parents=True)
        protocol = {**common, 'training': {**old['training'], 'training_mask_mode': 'soft', 'context_mode': mode,
            'initialization': 'same frozen step320 visual warm start; zero-initialized context branch',
            'data_order': 'original 2560 manifest order; one additional pass; new optimizer'}}
        atomic_write_json(arm / 'protocol.json', protocol)
    atomic_write_json(root / 'protocol.json', common)
    atomic_write_json(root / 'manifest.json', {'status': 'prepared', 'created_utc': native.now(),
        'arm_protocol_sha256': {m: file_sha(root / 'arms' / m / 'protocol.json') for m in ARMS},
        'protocol_sha256': file_sha(root / 'protocol.json')})


def validate(root):
    protocol = native.validate(root)
    saved = read(root / 'manifest.json')
    if saved['protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Master protocol changed')
    for mode in ARMS:
        arm = root / 'arms' / mode
        if saved['arm_protocol_sha256'][mode] != file_sha(arm / 'protocol.json'):
            raise ValueError('Arm protocol changed')
        current = native.validate(arm)
        if current['training']['context_mode'] != mode or current['initial_parameter_sha256'] != protocol['initial_parameter_sha256']:
            raise ValueError('Matched initialization/context contract changed')
    return protocol


def eval_config(root, protocol, method, phase, run):
    base_method = 'trained_selector' if method == 'frozen_visual' else method
    reference_protocol = {**protocol, 'trained_run_root': protocol['reference_train']}
    cfg = native.eval_config(root, reference_protocol, 'dense' if method.endswith('_context') else base_method, phase, run)
    if method.endswith('_context'):
        mode = method.removesuffix('_context')
        trained = root / 'arms' / mode / 'training'
        if read(trained / 'manifest.json')['status'] != 'complete' or read(trained / 'latest.json')['step'] != 320:
            raise ValueError('Only completed fixed step320 context weights may be evaluated')
        algorithm = 'context_visual_topk'
        cfg['pipeline']['stages'] = [{'name': method, 'plugin': algorithm, 'algorithm': algorithm,
            'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl',
            'placement': 'projector_level', 'method': 'token_compression', 'object_family': 'tensor_slot',
            'execution_phase': 'generate', 'modalities': ['image'], 'target_modalities': ['image'],
            'config': {'context_mode': mode, 'target_count': 128, 'width': 256, 'seed': 42,
                'checkpoint_path': str(native.latest(trained)), 'identity_path': protocol['identity_path']}}]
        cfg['eval'].update(compression_mode='compressed', target='compressed')
    cfg['experiment']['name'] = f'context_topk_{phase}_{method}'
    cfg['eval']['tasks'] = [f'context_topk_{t}' + ('_dev' if phase == 'dev' else '') for t in COUNTS]
    cfg['eval']['args']['include_path'] = str(ROOT / 'configs/eval/lmms_eval/tasks/context_visual_topk')
    cfg['data']['dataset_id'] = 'context_topk_native_tasks'
    return cfg


def check_eval(run, method, phase):
    result = native.check_eval(run, method, phase, tasks=COUNTS, counts=COUNTS,
        development_count=500, prefix='context_topk')
    if method.endswith('_context'):
        for line in (run / 'compression_runtime.requests.jsonl').read_text().splitlines():
            row = json.loads(line)
            delta = row['runtime']['activations'][0]['selection']
            if delta['context_mode'] != method.removesuffix('_context'):
                raise ValueError('Wrong context arm evaluated')
            if method == 'question_context' and (not delta['reads_question'] or not delta['question_tokens'] or not delta['question_ids_sha256']):
                raise ValueError('Missing prompt-only question audit')
    return result


def evaluate(root, protocol, phase, gpus):
    import yaml
    def worker(gpu, methods):
        for method in methods:
            pointer = root / 'eval' / phase / method / 'complete.json'
            if pointer.exists():
                saved = read(pointer)
                if saved['source'] != protocol['source'] or file_sha(saved['config']) != saved['config_sha256']:
                    raise ValueError('Completed evaluation provenance changed')
                check_eval(Path(saved['run']), method, phase)
                continue
            run = pointer.parent / f'attempt_{time.time_ns()}'
            config = root / 'configs' / f'{phase}_{method}_{time.time_ns()}.yaml'
            config.parent.mkdir(exist_ok=True)
            atomic_write_text(config, yaml.safe_dump(eval_config(root, protocol, method, phase, run), sort_keys=False))
            native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config)],
                pointer.parent / f'launch_{time.time_ns()}.log', [gpu],
                {'MLLM_TOPK_COHORT': str(root / 'cohort.json'), 'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                    'HF_HOME': str(root / 'cache/huggingface'), 'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
                    'XDG_CACHE_HOME': str(root / 'cache/xdg')})
            checks = check_eval(run, method, phase)
            atomic_write_json(pointer, {'run': str(run), 'source': protocol['source'], 'config': str(config),
                'config_sha256': file_sha(config), 'requests': checks['requests'], 'completed_utc': native.now()})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, METHODS[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    atomic_write_json(root / f'eval_{phase}_complete.json', {'status': 'complete', 'methods': METHODS, 'time': native.now()})


def clustered_difference(a, b, image_ids, *, resamples=10000, seed=42):
    """Question-weighted estimand; cluster resample independent images, not QAs."""
    import numpy as np
    if not len(a) or len(a) != len(b) or len(a) != len(image_ids) or resamples < 1:
        raise ValueError('Bootstrap requires nonempty aligned questions/images and positive resamples')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Bootstrap scores must be finite')
    groups = defaultdict(list)
    for x, y, image in zip(a, b, image_ids):
        groups[image].append(float(x) - float(y))
    sums = np.array([sum(v) for v in groups.values()])
    counts = np.array([len(v) for v in groups.values()])
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        idx = rng.integers(0, len(sums), size=len(sums))
        draws.append(sums[idx].sum() / counts[idx].sum())
    return {'delta': float(sums.sum() / counts.sum()), 'ci95': np.quantile(draws, [.025, .975]).tolist(),
        'questions': int(counts.sum()), 'independent_images': len(sums), 'resamples': resamples}


def report(root, protocol, phase):
    cohort = read(root / 'cohort.json')
    scores, comparisons = {}, {}
    for task, metric in METRICS.items():
        name = f'context_topk_{task}' + ('_dev' if phase == 'dev' else '')
        by_method = {}
        for method in METHODS:
            run = Path(read(root / 'eval' / phase / method / 'complete.json')['run'])
            check_eval(run, method, phase)
            filename, = run.glob(f'**/*_samples_{name}.jsonl')
            rows = sorted((json.loads(line) for line in filename.read_text().splitlines()), key=lambda r: r['doc_id'])
            by_method[method] = rows
        dense = by_method['dense']
        scores[task] = {m: sum(float(r[metric]) for r in rows) / len(rows) for m, rows in by_method.items()}
        ids = [cohort[task]['rows'][cohort[task]['development_indices'][i] if phase == 'dev' else i]['image_fingerprint']
            for i in range(len(dense))]
        for left, right in [('question_context', 'image_context'), ('question_context', 'frozen_visual'),
                ('question_context', 'old_mlp'), ('question_context', 'dense')]:
            a, b = by_method[left], by_method[right]
            if any((x['input'], x['target']) != (y['input'], y['target']) for x, y in zip(a, b)):
                raise ValueError('Paired question/target mismatch')
            comparisons[f'{task}:{left}_minus_{right}'] = clustered_difference(
                [r[metric] for r in a], [r[metric] for r in b], ids)
    atomic_write_json(root / f'summary_{phase}.json', {'scores': scores, 'comparisons': comparisons, 'limits': protocol['limits']})
    lines = [f'# Context selector ({phase}; fixed K128; fixed step320)', '',
        '| Task | Dense | Old MLP | Frozen visual | Image context | Question context |',
        '|---|---:|---:|---:|---:|---:|']
    for task in COUNTS:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{100*scores[task][m]:.2f}%' for m in METHODS) + ' |')
    lines += ['', 'OCRBench percentages are native sum/1000 normalized to percent, not an extra accuracy definition.',
        'AI2D headline uses upstream multi-choice filter exact_match; leading-option score remains supplementary.',
        'Paired bootstrap CIs use image clusters and the same question-weighted estimand as headline means.', '', *protocol['limits']]
    for name, row in comparisons.items():
        lines.append(f"\n{name}: {100*row['delta']:+.2f} pp; 95% CI [{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}] pp.")
    atomic_write_text(root / f'report_{phase}.md', '\n'.join(lines) + '\n')


def pipeline(root, gpus):
    protocol = validate(root)
    status = read(root / 'manifest.json')
    def update(stage, **kwargs):
        status.update(status='running', stage=stage, updated_utc=native.now(), **kwargs)
        atomic_write_json(root / 'manifest.json', status)
    try:
        update('real_7b_gates_and_training', gpu_ids=gpus)
        def train(mode, gpu):
            arm = root / 'arms' / mode
            if not (arm / 'gates_passed.json').exists():
                native.gates(arm, [gpu])
            marker = arm / 'training/manifest.json'
            if not marker.exists() or read(marker)['status'] != 'complete':
                native.execute(native.training_command(arm, arm / 'training', [gpu], resume=(arm / 'training/latest.json').exists()),
                    arm / f'train_launch_{time.time_ns()}.log', [gpu])
        if len(gpus) == 1:
            for mode in ARMS:
                train(mode, gpus[0])
        else:
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [pool.submit(train, mode, gpu) for mode, gpu in zip(ARMS, gpus)]
                for future in futures:
                    future.result()
        for mode in ARMS:
            manifest = read(root / 'arms' / mode / 'training/manifest.json')
            if manifest['steps_completed'] != 320 or manifest['initial_parameter_sha256'] != protocol['initial_parameter_sha256']:
                raise ValueError('Matched training completion/initialization mismatch')
        for phase in ('smoke', 'dev', 'full'):
            update(f'eval:{phase}')
            evaluate(root, protocol, phase, gpus)
            if phase != 'smoke':
                report(root, protocol, phase)
        status.update(status='complete', stage='complete', finished_utc=native.now())
        atomic_write_json(root / 'manifest.json', status)
    except BaseException as exc:
        status.update(status='failed', error=str(exc), updated_utc=native.now())
        atomic_write_json(root / 'manifest.json', status)
        atomic_write_json(root / f'failure_{time.time_ns()}.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'pipeline'])
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--gpus', nargs='+', type=int, choices=[0, 1], default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == 'prepare':
        if not args.reference:
            parser.error('--reference is required for preparation')
        prepare(root, args.reference.resolve())
    else:
        if len(set(args.gpus)) != len(args.gpus):
            parser.error('Duplicate GPU IDs')
        with (root / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(root, args.gpus)
