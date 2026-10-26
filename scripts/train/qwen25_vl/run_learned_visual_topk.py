"""Version-pinned experiment orchestration; all generation uses mllm-comp eval.

Commands: prepare -> gates -> train -> eval (smoke/dev/full) -> report.
Busy resources produce blocked_resource.json and exit 75, never a background
process consuming another user's GPUs. Reinvoke the same command when free.
"""
import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, development_indices, owned_output, reject_overlap, shared_policy, source_identity, validate_training_entries,
)
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text

LEGACY = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
OLD_TRAIN = LEGACY / 'outputs/train/qwen25_vl/anchor_mlp_7b_k128_ddp_20261021_v1'
TASK_DIR = ROOT / 'configs/eval/lmms_eval/tasks/learned_visual_topk'
DATA = {
    'chartqa': '/home/YinPeiran/datasets/lmms_eval_datasets/ChartQA/data/test-00000-of-00001.parquet',
    'ocrbench': '/home/YinPeiran/datasets/lmms_eval_datasets/OCRBench/data/test-00000-of-00001.parquet',
}
COUNTS = {'chartqa': 2500, 'ocrbench': 1000}
METHODS = ('dense', 'anchor_mean', 'old_mlp', 'random11', 'random23', 'random47',
           'initial_selector', 'trained_selector', 'eadp', 'prunesid', 'mmtok')


def now():
    return datetime.now(timezone.utc).isoformat()


def prepare(root, trained_run=None):
    import pyarrow.parquet as pq
    from PIL import Image
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    source = source_identity(ROOT)
    root = owned_output(root)
    if root.exists():
        raise ValueError('Experiment directory exists: preserve it, use a new run ID')
    root.mkdir(parents=True)
    try:
        manifest_path = OLD_TRAIN / 'training_samples.json'
        entries = json.loads(manifest_path.read_text())['entries']
        validate_training_entries(entries)
        identity_path = OLD_TRAIN / 'model_identity.json'
        identity = json.loads(identity_path.read_text())
        validate_identity(identity)
        cohort = {}
        for task, filename in DATA.items():
            rows = []
            for batch in pq.ParquetFile(filename).iter_batches(batch_size=16):
                for doc in batch.to_pylist():
                    raw = doc['image']
                    with Image.open(BytesIO(raw['bytes']) if raw.get('bytes') else raw['path']) as image:
                        fingerprint = media_fingerprint([image.convert('RGB')])
                    rows.append({'row': len(rows), 'image_fingerprint': fingerprint})
            if len(rows) != COUNTS[task]:
                raise ValueError(f'{task}: unexpected full split size {len(rows)}')
            reject_overlap(entries, rows)
            cohort[task] = {'path': filename, 'rows': rows, 'independent_images': len({r['image_fingerprint'] for r in rows}),
                            'development_indices': development_indices(rows)}
            print(f'{task}: {len(rows)} questions, {cohort[task]["independent_images"]} unique images; overlap=0', flush=True)
        atomic_write_json(root / 'cohort.json', cohort)
        adapter = OLD_TRAIN / 'model/checkpoints/step_000320'
        inputs = [manifest_path, identity_path, root / 'cohort.json', adapter / 'projector.safetensors',
                  adapter / 'projector_config.json', *map(Path, DATA.values()),
                  *sorted({Path(e['path']) for e in entries})]
        hashes = {}
        for path in inputs:
            print(f'Pinning input: {path.name}', flush=True)
            hashes[str(path)] = file_sha(path)
        training_reference = None
        if trained_run is not None:
            trained_run = trained_run.resolve()
            training_manifest = json.loads((trained_run / 'training/manifest.json').read_text())
            if training_manifest['status'] != 'complete' or training_manifest['steps_completed'] != 320:
                raise ValueError('External training run must have completed fixed step320')
            checkpoint = latest(trained_run / 'training')
            from mllm_comp.compressors.image.learned_visual_topk import load_selector
            selector = load_selector(checkpoint, base_identity=identity['identity_sha256'], device='cpu')
            if selector.spec.target_count != 128 or selector.spec.width != 256 or selector.spec.seed != 42:
                raise ValueError('External step320 selector has a different frozen specification')
            pointer = json.loads((trained_run / 'training/latest.json').read_text())
            if pointer['step'] != 320:
                raise ValueError('External pointer does not identify step320')
            references = [trained_run / 'protocol.json', trained_run / 'training/manifest.json',
                          trained_run / 'training/latest.json', checkpoint / 'selector.safetensors',
                          checkpoint / 'selector_config.json', checkpoint / 'complete.json']
            hashes.update({str(p): file_sha(p) for p in references})
            training_reference = {'run': str(trained_run), 'checkpoint': str(checkpoint),
                'weights_sha256': file_sha(checkpoint / 'selector.safetensors'),
                'scope': 'Evaluation-only restart: fixed step320 weights, no retraining or checkpoint selection'}
        protocol = {'created_utc': now(), 'source': source, 'allowed_gpus': [0, 1], 'input_hashes': hashes,
            'resource_policy': shared_policy(),
            'dependencies': dependency_identity(), 'python': sys.version,
            'gpu_inventory': subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,uuid,driver_version,memory.total', '--format=csv'], text=True),
            'identity_path': str(identity_path), 'training_manifest': str(manifest_path),
            'old_adapter': str(adapter), 'methods': list(METHODS),
            'training': {'seed': 42, 'steps': 320, 'global_batch': 8, 'learning_rate': 2e-4,
                'weight_decay': .01, 'warmup': 16, 'schedule': 'cosine', 'gradient_clip': 1.,
                'width': 256, 'target_count': 128, 'teacher_kl_weight': .5,
                'bce_weight_start': .1, 'bce_weight_end': 3., 'data_order': 'original manifest order; one pass',
                'dtype': 'BF16 frozen backbone / FP32 selector', 'attention': 'eager',
                'soft_sequence': 'full N visual sequence; sigmoid mask has mass min(128,N)',
                'hard_sequence': 'min(128,N) ORIGINAL tokens in source order; original M-RoPE'},
            'eval': {'counts': COUNTS, 'development_unique_images_per_task': 250,
                'generation': {'do_sample': False, 'repetition_penalty': 1.05},
                'max_new_tokens': {'chartqa': 16, 'ocrbench': 128},
                'native_dynamic_resolution': True, 'processor_pixels': [200704, 1605632],
                'bootstrap': '10000 paired image-cluster resamples; fixed model step320, no selection'},
            'claims': 'single-training-seed exploration; Qwen adaptations, not certified paper reproductions'}
        if training_reference is not None:
            protocol.update(trained_run_root=str(trained_run), training_reference=training_reference)
        atomic_write_json(root / 'protocol.json', protocol)
        atomic_write_json(root / 'manifest.json', {'status': 'prepared', 'source': source, 'protocol_sha256': file_sha(root / 'protocol.json')})
    except BaseException as exc:
        atomic_write_json(root / 'preparation_failure.json', {'error': str(exc), 'traceback': traceback.format_exc()})
        raise


def validate(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    if source_identity(ROOT) != protocol['source']:
        raise ValueError('Source snapshot changed or a different import location was used')
    if shared_policy() != protocol.get('resource_policy'):
        raise ValueError('Resource sharing policy changed')
    if dependency_identity() != protocol['dependencies']:
        raise ValueError('Software or editable evaluator changed')
    for path, checksum in protocol['input_hashes'].items():
        if file_sha(path) != checksum:
            raise ValueError(f'Pinned input changed: {path}')
    validate_identity(json.loads(Path(protocol['identity_path']).read_text()))
    return protocol


def pick_gpus(root, requested):
    free = available_gpus(requested)
    if not free:
        atomic_write_json(root / 'blocked_resource.json', {'status': 'blocked_resource', 'checked_utc': now(),
            'requested_gpus': requested, 'reason': 'Requested GPUs have active compute processes; nothing launched'})
        manifest = json.loads((root / 'manifest.json').read_text())
        manifest.update(status='blocked_resource', updated_utc=now())
        atomic_write_json(root / 'manifest.json', manifest)
        raise SystemExit(75)
    return free


def training_command(root, output, gpus, *, stop=320, resume=False):
    prefix = [sys.executable]
    if len(gpus) == 2:
        prefix += ['-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
    command = prefix + [str(ROOT / 'scripts/train/qwen25_vl/train_learned_visual_topk.py'),
        '--protocol', str(root / 'protocol.json'), '--output', str(output), '--stop-after', str(stop)]
    return command + (['--resume'] if resume else [])


def execute(command, log, gpus, extra_env=None):
    # Check again immediately before spawning; no GPU context was made above.
    if available_gpus(gpus) != gpus:
        raise RuntimeError('GPU became busy before launch; no job started')
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, gpus)), 'OMP_NUM_THREADS': '2',
           'PYTHONPATH': str(ROOT), 'TOKENIZERS_PARALLELISM': 'false', 'CUBLAS_WORKSPACE_CONFIG': ':4096:8', **(extra_env or {})}
    cache_overrides = extra_env or {}
    env.update(HF_HOME=str(owned_output(cache_overrides.get('HF_HOME', log.parent / 'cache/huggingface'))),
               HF_DATASETS_CACHE=str(owned_output(cache_overrides.get('HF_DATASETS_CACHE', log.parent / 'cache/datasets'))),
               XDG_CACHE_HOME=str(owned_output(cache_overrides.get('XDG_CACHE_HOME', log.parent / 'cache/xdg'))))
    for key in ('HF_HOME', 'HF_DATASETS_CACHE', 'XDG_CACHE_HOME'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('x') as stream:
        subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)


def latest(root):
    pointer = json.loads((root / 'latest.json').read_text())
    return root / pointer['checkpoint']


def gates(root, gpus):
    """Real identical 4 steps: uninterrupted vs 2 + process restart + 2."""
    import torch
    folder = root / f'gates_{time.time_ns()}'
    folder.mkdir()
    for name, stop, resume in [('continuous', 4, False), ('resumed', 2, False), ('resumed', 4, True)]:
        execute(training_command(root, folder / name, gpus, stop=stop, resume=resume),
                folder / f'{name}_{stop}.log', gpus)
    from safetensors.torch import load_file
    a = load_file(str(latest(folder / 'continuous') / 'selector.safetensors'))
    b = load_file(str(latest(folder / 'resumed') / 'selector.safetensors'))
    if set(a) != set(b) or not all(torch.equal(a[k], b[k]) for k in a):
        raise ValueError('Real 7B uninterrupted/resumed selector parameters are not bitwise identical')
    x = torch.load(latest(folder / 'continuous') / 'training_state.pt', weights_only=False, map_location='cpu')
    y = torch.load(latest(folder / 'resumed') / 'training_state.pt', weights_only=False, map_location='cpu')
    if x['step'] != y['step'] or x['scheduler'] != y['scheduler']:
        raise ValueError('Resumed optimizer schedule drifted')
    for key, state in x['optimizer']['state'].items():
        for field, value in state.items():
            other = y['optimizer']['state'][key][field]
            if isinstance(value, torch.Tensor) and not torch.equal(value, other):
                raise ValueError('Resumed optimizer moment/state drifted')
    atomic_write_json(root / 'gates_passed.json', {'status': 'passed', 'time': now(), 'world_size': len(gpus),
        'artifact': str(folder), 'protocol_sha256': file_sha(root / 'protocol.json'),
        'checks': ['real7B identity logits', 'hard K128 decode/cache', 'soft scorer/backbone gradients',
                   'continuous4 vs restart2+2 parameter equality']})


def freeze_release(root, protocol):
    if 'local_changes_sha256' in protocol['source']:
        atomic_write_json(root / 'release.json', {'status': 'hash_frozen_local_source_not_git_committed',
            'source': protocol['source'], 'protocol_sha256': file_sha(root / 'protocol.json'), 'time': now()})
        return
    tag = 'learned-visual-topk-v1.0.0'
    commit = protocol['source']['commit']
    existing = subprocess.run(['git', '-C', str(ROOT), 'rev-parse', '--verify', f'refs/tags/{tag}^{{commit}}'],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if existing.returncode == 0:
        if existing.stdout.strip() != commit:
            raise ValueError('Release tag already points to another version; never move it')
    else:
        subprocess.run(['git', '-C', str(ROOT), 'tag', '-a', tag, commit, '-m',
            'Real 7B identity/cache and exact restart gates passed; fixed experiment source'], check=True)
    atomic_write_json(root / 'release.json', {'tag': tag, 'commit': commit,
        'execution_worktree': str(ROOT), 'protocol_sha256': file_sha(root / 'protocol.json'), 'time': now()})


def eval_config(root, protocol, method, phase, output):
    identity = json.loads(Path(protocol['identity_path']).read_text())
    algorithm, cfg = None, {}
    if method == 'anchor_mean':
        algorithm, cfg = 'feature_anchor_mean', {'target_count': 128}
    elif method == 'old_mlp':
        algorithm, cfg = 'feature_anchor_adapter', {'tokens_per_crop': 128, 'adapter_path': protocol['old_adapter'], 'identity_path': protocol['identity_path']}
    elif method.startswith('random'):
        algorithm, cfg = 'random_visual_pruning', {'target_count': 128, 'seed': int(method[6:])}
    elif method in ('initial_selector', 'trained_selector'):
        algorithm, cfg = 'learned_visual_topk', {'target_count': 128, 'width': 256, 'seed': 42}
        if method == 'trained_selector':
            trained = Path(protocol.get('trained_run_root', root)) / 'training'
            manifest = json.loads((trained / 'manifest.json').read_text())
            if manifest['status'] != 'complete' or manifest['steps_completed'] != 320:
                raise ValueError('Evaluate only the fixed completed step320 selector')
            cfg.update(checkpoint_path=str(latest(trained)), identity_path=protocol['identity_path'])
    elif method in ('eadp', 'prunesid', 'mmtok'):
        algorithm = method
        cfg = {'tokens_per_crop': 128} if method != 'mmtok' else {'target_count': 128}
        if method == 'eadp':
            cfg.update(alpha=.5, beta=2.)
    elif method != 'dense':
        raise ValueError('Unknown method')
    stages = [] if algorithm is None else [{'name': method, 'plugin': algorithm, 'algorithm': algorithm,
        'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl', 'placement': 'projector_level',
        'method': 'token_compression', 'object_family': 'tensor_slot', 'execution_phase': 'generate',
        'modalities': ['image'], 'target_modalities': ['image'], 'config': cfg}]
    return {'experiment': {'name': f'learned_topk_{phase}_{method}', 'seed': 42},
        'backbone': {'name': 'qwen25_vl', 'model_path': identity['checkpoint'], 'model_revision': identity['revision'],
            'trust_remote_code': False, 'config': {'torch_dtype': 'bfloat16', 'attn_implementation': 'eager',
                'model_load_kwargs': {'local_files_only': True}, 'processor_load_kwargs': {
                    'local_files_only': True, 'use_fast': False, 'min_pixels': 200704, 'max_pixels': 1605632}}},
        'pipeline': {'stages': stages}, 'analysis': {}, 'train': {'compression_mode': 'none'},
        'eval': {'backend': 'lmms_eval', 'compression_mode': 'none' if not stages else 'compressed',
            'target': 'baseline' if not stages else 'compressed', 'checkpoint_path': identity['checkpoint'],
            'tasks': [f'learned_topk_{t}' + ('_dev' if phase == 'dev' else '') for t in DATA],
            'runtime_batch_size': 1, 'num_processes': 1, 'sample_limit': 8 if phase == 'smoke' else None,
            'generation': {'do_sample': False, 'repetition_penalty': 1.05},
            'args': {'force_simple': True, 'log_samples': True, 'verbosity': 'INFO', 'include_path': str(TASK_DIR)}},
        'data': {'dataset_id': 'learned_topk_native_tasks', 'sample_split': 'test'},
        'output': {'run_dir': str(output), 'manifest_path': str(output / 'manifest.json')}}


def check_eval(run, method, phase, *, tasks=None, counts=None, development_count=250, prefix='learned_topk'):
    import math
    results = list(run.glob('**/*_results.json'))
    if len(results) != 1:
        raise ValueError('Expected one native result file')
    native = json.loads(results[0].read_text())['results']
    sample_count, sample_keys = 0, set()
    for task in (DATA if tasks is None else tasks):
        name = f'{prefix}_{task}' + ('_dev' if phase == 'dev' else '')
        files = list(run.glob(f'**/*_samples_{name}.jsonl'))
        if len(files) != 1 or name not in native:
            raise ValueError(f'Missing native samples/metric: {name}')
        rows = [json.loads(s) for s in files[0].read_text().splitlines()]
        n = 8 if phase == 'smoke' else development_count if phase == 'dev' else (COUNTS if counts is None else counts)[task]
        if len(rows) != n or {r['doc_id'] for r in rows} != set(range(n)):
            raise ValueError('Missing or duplicate native predictions')
        metric = {'chartqa': 'relaxed_overall', 'ocrbench': 'ocrbench_accuracy', 'ai2d': 'exact_match'}[task]
        if not all(math.isfinite(float(r[metric])) for r in rows):
            raise ValueError('Nonfinite native score')
        sample_keys.update((name, int(r['doc_id'])) for r in rows)
        sample_count += n
    counts, trace_keys, verified = [], set(), 0
    performance = [json.loads(line) for line in (run / 'request_performance.jsonl').read_text().splitlines()]
    source_counts = {}
    for record in performance:
        parts = record['sample_id'].split(':')
        key = (parts[0], int(parts[2]))
        if key in source_counts or key not in sample_keys or record['status'] != 'complete':
            raise ValueError('Duplicate, failed, or unknown performance request')
        if record['original_visual_tokens'] < 1 or not math.isfinite(record['generation_s']):
            raise ValueError('Invalid source token count or timing')
        source_counts[key] = record['original_visual_tokens']
    if set(source_counts) != sample_keys:
        raise ValueError('Missing native generation/performance requests')
    if method != 'dense':
        files = list(run.glob('compression_runtime*.requests.jsonl'))
        if len(files) != 1:
            raise ValueError('Missing compression request trace')
        for line in files[0].read_text().splitlines():
            row = json.loads(line)
            parts = row['request']['sample_id'].split(':')
            key = (parts[0], int(parts[2]))
            if key not in sample_keys or key in trace_keys:
                raise ValueError('Duplicate or unknown trace request')
            trace_keys.add(key)
            activation, = row['runtime']['activations']
            delta = activation.get('selection') or activation.get('latest_layout_delta')
            n, k = delta['source_count'], delta['result_count']
            if n != source_counts[key] or k != min(128, n):
                raise ValueError('Actual output token budget differs from min(128,N)')
            indices = delta.get('selected_indices')
            if indices is not None and (len(indices) != len(set(indices)) or len(indices) != k or
                                        any(not 0 <= i < n for i in indices)):
                raise ValueError('Token indices duplicated or dropped')
            layer_counts = delta.get('observed_visual_tokens_per_decoder_layer')
            if layer_counts is not None and any(value != k for value in layer_counts):
                raise ValueError('Decoder layer received a different visual token count')
            lifecycle = activation.get('projector_generate', {}).get('last', {})
            checked = delta.get('decode_position_verified', lifecycle.get('decode_position_verified'))
            if checked is False or (lifecycle.get('decode_steps', 0) and checked is not True):
                raise ValueError('Decode position validation failed')
            verified += int(checked is True)
            counts.append({'task': key[0], 'doc_id': key[1], 'source': n, 'retained': k})
        if trace_keys != sample_keys:
            raise ValueError('Missing compression trace requests')
    checked = {'status': 'complete', 'method': method, 'phase': phase, 'requests': sample_count,
        'native_metrics': native, 'token_counts': counts, 'first_decode_verified_requests': verified}
    atomic_write_json(run / 'protocol_checks.json', checked)
    return checked


def evaluate(root, protocol, gpus, phase, methods=None):
    import yaml
    methods = tuple(METHODS if methods is None else methods)
    if not methods or len(set(methods)) != len(methods) or set(methods) - set(METHODS):
        raise ValueError('Unknown, empty, or duplicate evaluation methods')
    prerequisite = 'smoke' if phase == 'dev' else 'dev' if phase == 'full' else None
    if prerequisite and not (root / f'eval_{prerequisite}_complete.json').exists():
        raise ValueError(f'Complete {prerequisite} before {phase}')
    def worker(gpu, methods):
        for method in methods:
            pointer = root / 'eval' / phase / method / 'complete.json'
            if pointer.exists():
                saved = json.loads(pointer.read_text())
                if saved['source'] != protocol['source'] or file_sha(saved['config']) != saved['config_sha256']:
                    raise ValueError('Completed evaluation provenance changed')
                check_eval(Path(saved['run']), method, phase)
                continue
            run = pointer.parent / f'attempt_{time.time_ns()}'
            cfg = eval_config(root, protocol, method, phase, run)
            config_path = root / 'configs' / f'{phase}_{method}_{time.time_ns()}.yaml'
            config_path.parent.mkdir(exist_ok=True)
            atomic_write_text(config_path, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
            execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config_path)],
                pointer.parent / f'launch_{time.time_ns()}.log', [gpu], {'MLLM_TOPK_COHORT': str(root / 'cohort.json'),
                    'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl')})
            checks = check_eval(run, method, phase)
            atomic_write_json(pointer, {'run': str(run), 'requests': checks['requests'], 'source': protocol['source'],
                'config': str(config_path), 'config_sha256': file_sha(config_path), 'completed_utc': now()})
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, list(methods[i::len(gpus)])) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()
    atomic_write_json(root / f'eval_{phase}_complete.json', {'status': 'complete', 'methods': list(methods), 'time': now()})


def run_command(args):
    root = owned_output(args.root)
    if args.command == 'prepare':
        return prepare(root, args.trained_run)
    if args.command in ('pipeline', 'evaluate-all'):
        from types import SimpleNamespace
        stages = [('eval', 'smoke'), ('eval', 'dev'), ('eval', 'full'), ('report', 'full')]
        if args.command == 'pipeline':
            stages = [('gates', 'smoke'), ('train', 'smoke'), *stages]
        for command, phase in stages:
            marker = root / ('gates_passed.json' if command == 'gates' else
                f'eval_{phase}_complete.json' if command == 'eval' else 'summary.json' if command == 'report' else 'training/manifest.json')
            if marker.exists() and (command != 'train' or json.loads(marker.read_text()).get('status') == 'complete'):
                continue
            stage_args = SimpleNamespace(**{**vars(args), 'command': command, 'phase': phase})
            stage_args.resume = command == 'train' and (root / 'training/latest.json').exists()
            atomic_write_json(root / 'pipeline_stage.json', {'command': command, 'phase': phase, 'status': 'running', 'time': now()})
            run_command(stage_args)
            atomic_write_json(root / 'pipeline_stage.json', {'command': command, 'phase': phase, 'status': 'complete', 'time': now()})
        return
    protocol = validate(root)
    if args.command == 'report':
        from scripts.eval.lmms_eval.report_learned_visual_topk import report
        return report(root)
    gpus = pick_gpus(root, args.gpus)
    if args.command == 'gates':
        return gates(root, gpus)
    if args.command == 'train':
        gate = json.loads((root / 'gates_passed.json').read_text())
        if gate['world_size'] != len(gpus) or gate['protocol_sha256'] != file_sha(root / 'protocol.json'):
            raise ValueError('Real restore gates must match the training world size and protocol')
        freeze_release(root, protocol)
        return execute(training_command(root, root / 'training', gpus, resume=args.resume),
            root / f'train_launch_{time.time_ns()}.log', gpus)
    if args.command == 'eval':
        return evaluate(root, protocol, gpus, args.phase)


def main(args):
    root = owned_output(args.root)
    if args.command == 'prepare':
        return run_command(args)
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(status='running', current_command=args.command, current_phase=args.phase, updated_utc=now())
            atomic_write_json(root / 'manifest.json', manifest)
            result = run_command(args)
            manifest.update(status='complete' if args.command in ('report', 'pipeline', 'evaluate-all') else f'{args.command}_complete', updated_utc=now())
            atomic_write_json(root / 'manifest.json', manifest)
            return result
        except Exception as exc:
            atomic_write_json(root / f'runner_failure_{time.time_ns()}.json', {
                'command': args.command, 'phase': args.phase, 'error': str(exc), 'traceback': traceback.format_exc(), 'time': now()})
            manifest = json.loads((root / 'manifest.json').read_text())
            manifest.update(status='failed', error=str(exc), updated_utc=now())
            atomic_write_json(root / 'manifest.json', manifest)
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'gates', 'train', 'eval', 'report', 'pipeline', 'evaluate-all'])
    parser.add_argument('--trained-run', type=Path, help='Read-only completed step320 training run for an evaluation-only restart')
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', choices=[0, 1], default=[0, 1])
    parser.add_argument('--phase', choices=['smoke', 'dev', 'full'], default='smoke')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if len(set(args.gpus)) != len(args.gpus):
        parser.error('Duplicate GPU indices')
    main(args)
