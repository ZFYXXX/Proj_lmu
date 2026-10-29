"""Evaluation-only POPE/MMStar extension for the fixed Qwen2.5/3.5 heads.

Independent, hash-frozen source; no commits, retraining, legacy result edits or
shared-directory writes. Existing image-context training/evaluation has priority.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity, gpu_free_mib, owned_output, reject_overlap, source_identity,
)
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from configs.eval.lmms_eval.tasks.projector_extra import extra_utils as tasks

BASE = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
Q25 = BASE / 'outputs/train/qwen25_vl/context_visual_topk_20261024_v2'
Q35 = BASE / 'outputs/eval/qwen35_vl/transfer_k128_20261026_v1'
IMAGE = BASE / 'outputs/eval/qwen35_vl/image_context_k128_20261027_v1'
METHODS = ('dense', 'old_mlp', 'visual_selector', 'image_context_selector',
           'question_selector', 'protected_selector', 'eadp', 'prunesid', 'mmtok', 'bifocal_stage1')
COUNTS = {'mmstar': 1500, 'pope': 9000}
MMSTAR_REVISION = 'bc98d668301da7b14f648724866e57302778ab27'
DATA = {'mmstar': [Path('/home/ZhouFengyu/fastv-repro/datasets/MMStar/mmstar.parquet')],
        'pope': [Path('/home/YinPeiran/datasets/lmms_eval_datasets/POPE/Full') / f'{c}-00000-of-00001.parquet'
                 for c in ('adversarial', 'popular', 'random')]}


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def template_configs():
    """Reuse actual completed head/config recipes, not historical predictions."""
    import yaml
    from scripts.train.qwen25_vl import run_context_visual_topk as context
    from scripts.eval.lmms_eval import run_qwen35_transfer as transfer
    from scripts.eval.lmms_eval import run_qwen35_reference as reference
    q25, q35 = read(Q25 / 'protocol.json'), read(Q35 / 'protocol.json')
    result, provenance = {'qwen25': {}, 'qwen35': {}}, {}
    aliases = {'visual_selector': 'frozen_visual', 'image_context_selector': 'image_context',
               'question_selector': 'question_context'}
    for method in METHODS:
        if method == 'protected_selector':
            marker = BASE / 'outputs/analysis/protected_visual_topk/protected_k128_20261026_v1/eval/full/learned_word/chartqa/complete.json'
        elif method == 'bifocal_stage1':
            marker = BASE / 'outputs/eval/qwen25_vl/bifocal_stage1_stage12_20261024_v2/eval/full/bifocal_stage1_k128/chartqa/complete.json'
        elif method in ('eadp', 'prunesid', 'mmtok'):
            marker = BASE / f'outputs/eval/qwen25_vl/projector_official_fidelity_k128_20261024_v1/eval/full/{method}/chartqa/complete.json'
        else:
            marker = None
        if marker:
            saved = read(marker)
            if file_sha(saved['config']) != saved['config_sha256']:
                raise ValueError('Historical method config changed')
            cfg = yaml.safe_load(Path(saved['config']).read_text())
            provenance[str(marker)] = file_sha(marker)
            provenance[saved['config']] = saved['config_sha256']
        else:
            cfg = context.eval_config(Q25, q25, aliases.get(method, method), 'full', BASE / 'unused')
        # Historical shuffled-question audits cover only the old three tasks.
        # They do not affect true-question selection; omit them for new tasks.
        for stage in cfg['pipeline']['stages']:
            stage['config'].pop('audit_conditions', None)
            stage['config'].pop('audit_conditions_sha256', None)
        result['qwen25'][method] = cfg
        if method == 'image_context_selector':
            cfg = transfer.config(Q35, q35, 'visual_selector', 'full', 'chartqa', BASE / 'unused')
            stage, = cfg['pipeline']['stages']
            name = 'qwen35_image_context_projector'
            stage.update(name=method, algorithm=name, plugin=name, variant=f'{name}.image',
                         binding=f'{name}.image.qwen3_5', conditioning_modalities=[])
            stage['config'] = {'method': method, 'target_count': 128, 'identity_path': q35['identity_path'],
                               'checkpoint_path': '__pending_image_step320__'}
        elif method in ('eadp', 'mmtok', 'bifocal_stage1'):
            cfg = reference.config(Q35, q35, method, 'full', 'chartqa', BASE / 'unused')
        else:
            cfg = transfer.config(Q35, q35, method, 'full', 'chartqa', BASE / 'unused')
        result['qwen35'][method] = cfg
    return result, provenance


def config(p, model, method, phase, task, run):
    cfg = copy.deepcopy(p['templates'][model][method])
    cfg['experiment']['name'] = f'extra_{model}_{phase}_{method}_{task}'
    cfg['eval'].update(tasks=[f'projector_extra_{task}'], sample_limit=8 if phase == 'smoke' else None)
    cfg['eval']['args']['include_path'] = str(ROOT / 'configs/eval/lmms_eval/tasks/projector_extra')
    cfg['data'] = {'dataset_id': 'projector_extra_native_tasks', 'sample_split': 'test'}
    cfg['output'] = {'run_dir': str(run), 'manifest_path': str(run / 'manifest.json')}
    if model == 'qwen35' and method == 'image_context_selector':
        folder = IMAGE / 'training/image'
        pointer = read(folder / 'latest.json')
        if pointer['step'] != 320 or read(folder / 'manifest.json')['status'] != 'complete':
            raise ValueError('Wait for the fixed image-context step320; never use an interim checkpoint')
        cfg['pipeline']['stages'][0]['config']['checkpoint_path'] = str(folder / pointer['checkpoint'])
    return cfg


def prepare(root):
    import pyarrow.parquet as pq
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    if root.exists():
        raise ValueError('Use a fresh run directory; old results are never overwritten')
    root.mkdir(parents=True)
    write_json(root / 'manifest.json', {'status': 'preparing', 'started_utc': now()})
    try:
        training_manifest = read(Q25 / 'protocol.json')['training_manifest']
        entries = read(training_manifest)['entries']
        cohort, pins, image_cache = {}, {training_manifest: file_sha(training_manifest)}, {}
        for task, paths in DATA.items():
            rows = []
            for path in paths:
                pins[str(path)] = file_sha(path)
                for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                    for doc in batch.to_pylist():
                        data = doc['image']
                        raw = data.get('bytes') if isinstance(data, dict) else data
                        key = hashlib.sha256(raw).hexdigest() if isinstance(raw, bytes) else str(data)
                        if key not in image_cache:
                            image_cache[key] = media_fingerprint([tasks.image_from_doc(doc)])
                        row = {'row': len(rows), 'image_fingerprint': image_cache[key],
                               'category': doc['category'], 'target': doc['answer'],
                               'sample_id': str(doc.get('index', doc.get('question_id', doc.get('id')))),
                               'prompt': (tasks.mmstar_doc_to_text(doc, {'post_prompt': "\nAnswer with the option's letter from the given choices directly"})
                                          if task == 'mmstar' else tasks.pope_doc_to_text(doc, {'post_prompt': '\nAnswer the question using a single word or phrase.'}))}
                        if task == 'mmstar':
                            row['l2_category'] = doc['l2_category']
                            row['source_benchmark'] = doc['meta_info']['source']
                        rows.append(row)
            if len(rows) != COUNTS[task]:
                raise ValueError(f'Wrong full-split size: {task}')
            overlap = sorted({e['image_fingerprint'] for e in entries} & {r['image_fingerprint'] for r in rows})
            write_json(root / f'{task}_training_overlap.json', {'overlap': overlap, 'overlap_images': len(overlap)})
            reject_overlap(entries, rows)
            cohort[task] = {'paths': [str(p) for p in paths], 'rows': rows,
                            'independent_images': len({r['image_fingerprint'] for r in rows}),
                            'smoke_indices': tasks.smoke_indices([r['category'] for r in rows])}
            print(f'{task}: {len(rows)} questions / {cohort[task]["independent_images"]} unique images; train overlap=0', flush=True)
        if Counter(r['category'] for r in cohort['pope']['rows']) != {'adversarial': 3000, 'popular': 3000, 'random': 3000}:
            raise ValueError('POPE must contain the complete three strategies')
        write_json(root / 'cohort.json', cohort)
        pins[str(root / 'cohort.json')] = file_sha(root / 'cohort.json')
        templates, provenance = template_configs()
        pins.update(provenance)
        for configs in templates.values():
            for cfg in configs.values():
                for stage in cfg['pipeline']['stages']:
                    opts = stage['config']
                    for key in ('checkpoint_path', 'adapter_path'):
                        folder = opts.get(key)
                        if folder and folder != '__pending_image_step320__':
                            files = [f for f in Path(folder).iterdir() if f.suffix in ('.safetensors', '.json')]
                            if not files:
                                raise ValueError(f'Missing frozen checkpoint: {folder}')
                            for f in files:
                                pins[str(f)] = file_sha(f)
                    if opts.get('identity_path'):
                        pins[opts['identity_path']] = file_sha(opts['identity_path'])
        for path in (Q25 / 'protocol.json', Q35 / 'protocol.json', IMAGE / 'protocol.json'):
            pins[str(path)] = file_sha(path)
        identities = {model: read(cfgs['old_mlp']['pipeline']['stages'][0]['config']['identity_path'])
                      for model, cfgs in templates.items()}
        for identity in identities.values():
            validate_identity(identity)
        p = {'created_utc': now(), 'source': source_identity(ROOT), 'dependencies': dependency_identity(),
             'templates': templates, 'identities': identities, 'input_hashes': pins,
             'counts': COUNTS, 'methods': list(METHODS), 'models': ['qwen25', 'qwen35'],
             'wait_for': str(IMAGE / 'manifest.json'), 'no_training': True, 'no_development_pass': True,
             'mmstar_source': {'repo': 'Lin-Chen/MMStar', 'revision': MMSTAR_REVISION},
             'generation': {'greedy': True, 'repetition_penalty': 1.05, 'max_new_tokens': 16},
             'budget': 'min(128,N) post-merger, native dynamic resolution, not cross-layer ratio',
             'primary_metrics': {'pope': 'framework native strict exact yes/no accuracy',
                                 'mmstar': 'framework native average: equal detailed-category weights'},
             'limits': ['No retraining, test-based checkpoint selection, noise gate, OT, or new method.',
                        'MMStar shares source benchmarks with earlier tests; it is not an independent confirmation dataset.',
                        'POPE questions share images across strategies; bootstrap clusters image content, not questions.',
                        'External rows are current Qwen adaptations at matched K128, not certified paper-number reproductions.',
                        'Bifocal Stage1 only; no unimplemented Qwen3.5 Stage2 claim.',
                        'Timings include vision/compressor/prefill/decode but exclude CPU processor/model loading; GPUs are shared.']}
        write_json(root / 'protocol.json', p)
        write_json(root / 'manifest.json', {'status': 'prepared', 'source': p['source'],
                   'protocol_sha256': file_sha(root / 'protocol.json'), 'total_full_cells': 40,
                   'full_requests': 210000, 'updated_utc': now()})
        report(root, p)
    except BaseException as error:
        write_json(root / 'manifest.json', {'status': 'failed_preparation', 'error': str(error), 'updated_utc': now()})
        raise


def validate(root):
    p = read(root / 'protocol.json')
    if read(root / 'manifest.json')['protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Protocol changed')
    if p['source'] != source_identity(ROOT) or p['dependencies'] != dependency_identity():
        raise ValueError('Source or evaluator changed; start a new run')
    for path, checksum in p['input_hashes'].items():
        if file_sha(path) != checksum:
            raise ValueError(f'Pinned input changed: {path}')
    for identity in p['identities'].values():
        validate_identity(identity)
    return p


def check_trace(activation, original, layers):
    record = activation.get('selection') or activation.get('latest_layout_delta')
    if not record:
        raise ValueError('No projector selection trace')
    n, k = record['source_count'], record['result_count']
    if n != original or k != min(128, n):
        raise ValueError('Source/token budget mismatch')
    indices = record.get('selected_indices', record.get('bifocal_representatives', record.get('representative_indices')))
    if indices is not None and (len(indices) != k or len(set(indices)) != k or any(i < 0 or i >= n for i in indices)):
        raise ValueError('Invalid/duplicate visual representatives')
    lifecycle = activation.get('projector_generate', {}).get('last', {})
    verified = record.get('decode_position_verified', lifecycle.get('decode_position_verified'))
    if verified is False or (lifecycle.get('decode_steps', 0) and verified is not True):
        raise ValueError('Invalid decode M-RoPE position')
    counts = record.get('observed_visual_tokens_per_decoder_layer')
    if counts is not None and counts != [k] * layers:
        raise ValueError('Wrong visual count in decoder layers')
    return k


def check(root, model, method, phase, task, run):
    cohort = read(root / 'cohort.json')[task]
    expected = [cohort['rows'][i] for i in cohort['smoke_indices']] if phase == 'smoke' else cohort['rows']
    n, name = len(expected), f'projector_extra_{task}'
    files = list(run.glob(f'**/*_samples_{name}.jsonl'))
    result_files = list(run.glob('**/*_results.json'))
    if len(files) != 1 or len(result_files) != 1:
        raise ValueError('Missing/duplicate native prediction artifacts')
    rows = [json.loads(line) for line in files[0].read_text().splitlines()]
    if len(rows) != n or {r['doc_id'] for r in rows} != set(range(n)):
        raise ValueError('Missing/duplicate predictions')
    rows.sort(key=lambda r: r['doc_id'])
    if any(r['input'] != e['prompt'] or r['target'] != e['target'] for r, e in zip(rows, expected)):
        raise ValueError('Native question/target alignment changed')
    primary = 'average' if task == 'mmstar' else 'pope_accuracy'
    score = tasks.macro_score([r['average'] for r in rows]) if task == 'mmstar' else sum(r['pope_accuracy'] for r in rows) / n
    native = read(result_files[0])['results'][name]
    key, = [k for k in native if k.split(',')[0] == primary]
    if not math.isfinite(score) or abs(score - native[key]) > 1e-10:
        raise ValueError('Native aggregate and per-question scores disagree')
    performance = [json.loads(s) for s in (run / 'request_performance.jsonl').read_text().splitlines()]
    def docid(sid):
        fields = sid.split(':')
        if fields[0] != name or fields[1] != 'test':
            raise ValueError('Unknown request/task')
        return int(fields[2])
    perf = {docid(r['sample_id']): r for r in performance}
    if len(performance) != n or set(perf) != set(range(n)):
        raise ValueError('Missing/duplicate performance requests')
    if any(r['status'] != 'complete' or r['original_visual_tokens'] <= 0 or
           not math.isfinite(r['generation_s']) for r in performance):
        raise ValueError('Failed/nonfinite generation')
    kept = [perf[i]['original_visual_tokens'] for i in range(n)]
    if method != 'dense':
        traces = [json.loads(s) for s in (run / 'compression_runtime.requests.jsonl').read_text().splitlines()]
        indexed = {docid(r['request']['sample_id']): r for r in traces}
        if len(traces) != n or set(indexed) != set(range(n)):
            raise ValueError('Missing/duplicate compression traces')
        for i in range(n):
            activation, = indexed[i]['runtime']['activations']
            kept[i] = check_trace(activation, perf[i]['original_visual_tokens'], 28 if model == 'qwen25' else 32)
    result = {'requests': n, 'score': score, 'samples': str(files[0]),
              'samples_sha256': file_sha(files[0]), 'native_metrics': native,
              'mean_source_tokens': sum(r['original_visual_tokens'] for r in performance) / n,
              'mean_retained_tokens': sum(kept) / n,
              'mean_retained_fraction': sum(k / perf[i]['original_visual_tokens'] for i, k in enumerate(kept)) / n,
              'generation_seconds': sum(r['generation_s'] for r in performance),
              'peak_allocated_bytes': max(r['peak_allocated_bytes'] or 0 for r in performance)}
    if task == 'pope':
        result['strategy_metrics'] = {}
        for category in sorted({e['category'] for e in expected}):
            subset = [r['pope_precision'] for r, e in zip(rows, expected) if e['category'] == category]
            result['strategy_metrics'][category] = tasks.pope_statistics(subset)
    else:
        result['micro_accuracy'] = sum(r['mmstar_micro_accuracy'] for r in rows) / n
    write_json(run / 'protocol_checks.json', result)
    return result


def child_env(root, model, gpu, phase, run):
    # The two backbones require isolated Transformers versions.
    extra = '/home/ZhouFengyu/fastv-repro/qwen38-triton38:/home/ZhouFengyu/fastv-repro/qwen38-transformers-519:' if model == 'qwen35' else ''
    cache = root / 'cache'
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': str(gpu), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2',
           'PYTHONPATH': extra + str(ROOT), 'TOKENIZERS_PARALLELISM': 'false',
           'CUBLAS_WORKSPACE_CONFIG': ':4096:8', 'HF_HUB_OFFLINE': '1', 'HF_DATASETS_OFFLINE': '1',
           'WANDB_MODE': 'disabled', 'MLLM_TOPK_HASH_LOCAL_SOURCE': '1', 'MLLM_TOPK_NORMAL_GPU': '1',
           'MLLM_EXTRA_SMOKE': '1' if phase == 'smoke' else '0',
           'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
           'HF_HOME': str(cache / 'huggingface'), 'HF_DATASETS_CACHE': str(cache / 'datasets'),
           'XDG_CACHE_HOME': str(cache / 'xdg')}
    env.pop('MLLM_TOPK_SHARED_GPU', None)
    for key in ('HF_HOME', 'HF_DATASETS_CACHE', 'XDG_CACHE_HOME'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return env


def evaluate(root, p, model, method, phase, task, gpu):
    import yaml
    directory = root / 'eval' / phase / model / method / task
    marker = directory / 'complete.json'
    if marker.exists():
        saved = read(marker)
        if file_sha(saved['config']) != saved['config_sha256'] or file_sha(saved['samples']) != saved['samples_sha256']:
            raise ValueError('Completed result provenance changed')
        return
    stamp = time.time_ns()
    run = directory / f'attempt_{stamp}'
    filename = root / 'configs' / f'{phase}_{model}_{method}_{task}_{stamp}.yaml'
    cfg = config(p, model, method, phase, task, run)
    write_text(filename, yaml.safe_dump(cfg, sort_keys=False))
    if model == 'qwen35' and method == 'image_context_selector':
        checkpoint = Path(cfg['pipeline']['stages'][0]['config']['checkpoint_path'])
        pins = {str(checkpoint / f): file_sha(checkpoint / f) for f in ('head.safetensors', 'head_config.json')}
        seal = root / 'resolved_image_context.json'
        if seal.exists() and read(seal) != pins:
            raise ValueError('Resolved image-context checkpoint changed')
        write_json(seal, pins)
    while gpu_free_mib().get(gpu, 0) < 40960:
        write_json(root / f'worker_gpu{gpu}.json', {'status': 'waiting_for_memory', 'updated_utc': now()})
        time.sleep(30)
    directory.mkdir(parents=True, exist_ok=True)
    log = directory / f'launch_{stamp}.log'
    command = [sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(filename)]
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=ROOT, env=child_env(root, model, gpu, phase, run), stdout=stream, stderr=subprocess.STDOUT)
        write_json(root / f'worker_gpu{gpu}.json', {'status': 'running', 'model': model, 'method': method,
                   'task': task, 'phase': phase, 'pid': child.pid, 'log': str(log), 'updated_utc': now()})
        code = child.wait()
    if code:
        raise RuntimeError(f'Native eval failed ({code}); preserve attempt/log: {log}')
    checks = check(root, model, method, phase, task, run)
    write_json(marker, {**checks, 'run': str(run), 'config': str(filename),
                       'config_sha256': file_sha(filename), 'finished_utc': now()})


def report(root, p):
    records, lines = [], ['# Qwen projector extension: full POPE + MMStar', '',
                         '| Model | Method | MMStar native | POPE accuracy |', '|---|---|---:|---:|']
    for model in p['models']:
        for method in METHODS:
            scores = []
            for task in COUNTS:
                marker = root / 'eval/full' / model / method / task / 'complete.json'
                if marker.exists():
                    item = read(marker)
                    records.append({'model': model, 'method': method, 'task': task, **item})
                    scores.append(f'{item["score"] * 100:.2f}%')
                else:
                    scores.append('pending')
            lines.append('| ' + ' | '.join((model, method, *scores)) + ' |')
    write_json(root / 'summary.json', {'complete_cells': len(records), 'total_cells': 40, 'results': records})
    write_text(root / 'report.md', '\n'.join(lines + ['', *p['limits']]) + '\n')


def bootstrap(delta, image_ids, strata=None, resamples=10000, seed=42):
    """Image-cluster paired CI, preserving MMStar's detailed-category estimand."""
    import numpy as np
    if len(delta) != len(image_ids) or not len(delta):
        raise ValueError('Misaligned bootstrap input')
    groups = {value: i for i, value in enumerate(sorted(set(image_ids)))}
    categories = {value: i for i, value in enumerate(sorted(set(strata)))} if strata else {'all': 0}
    sums = np.zeros((len(groups), len(categories)))
    counts = np.zeros_like(sums)
    for i, (value, image) in enumerate(zip(delta, image_ids)):
        c = categories[strata[i]] if strata else 0
        sums[groups[image], c] += value
        counts[groups[image], c] += 1
    rng, draws = np.random.default_rng(seed), []
    for _ in range(resamples):
        indices = rng.integers(0, len(groups), len(groups))
        denominator = counts[indices].sum(0)
        if (denominator == 0).any():
            raise ValueError('Bootstrap draw lost a scoring stratum')
        draws.append(float((sums[indices].sum(0) / denominator).mean()))
    return {'delta': float((sums.sum(0) / counts.sum(0)).mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'resamples': resamples,
            'independent_images': len(groups), 'questions': len(delta)}


def final_statistics(root, p):
    cohort, comparisons = read(root / 'cohort.json'), []
    for model in p['models']:
        for task in COUNTS:
            def sample_rows(method):
                marker = read(root / 'eval/full' / model / method / task / 'complete.json')
                rows = [json.loads(s) for s in Path(marker['samples']).read_text().splitlines()]
                return sorted(rows, key=lambda r: r['doc_id'])
            baseline = sample_rows('dense')
            def score(row):
                return row['average']['score'] if task == 'mmstar' else row['pope_accuracy']
            rows = cohort[task]['rows']
            for method in METHODS[1:]:
                actual = sample_rows(method)
                comparisons.append({'model': model, 'task': task, 'method': method,
                    **bootstrap([score(a) - score(b) for a, b in zip(actual, baseline)],
                                [r['image_fingerprint'] for r in rows],
                                [r['l2_category'] for r in rows] if task == 'mmstar' else None)})
    write_json(root / 'paired_bootstrap.json', {'comparisons': comparisons})


def pipeline(root, gpus):
    p = validate(root)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = read(root / 'manifest.json')
        def stage(status, **extra):
            state.update(status=status, updated_utc=now(), pid=os.getpid(), gpus=gpus, **extra)
            write_json(root / 'manifest.json', state)
        try:
            prerequisite = Path(p['wait_for'])
            while read(prerequisite)['status'] != 'complete':
                prior = read(prerequisite)
                if prior['status'] in ('failed', 'failed_preparation'):
                    raise RuntimeError('Existing image-context queue failed; inspect before starting dependent jobs')
                stage('queued', reason='Existing Qwen3.5 image-context training and three-task evaluation has GPU priority',
                      prerequisite=str(prerequisite))
                time.sleep(30)
            stop = Event()
            # A model/method cell passes both task smokes before either full run.
            queue = Queue()
            for method in METHODS:
                for model in p['models']:
                    queue.put((model, method))
            def worker(gpu):
                try:
                    while not stop.is_set():
                        try:
                            model, method = queue.get_nowait()
                        except Empty:
                            break
                        for phase in ('smoke', 'full'):
                            for task in COUNTS:
                                if stop.is_set():
                                    return
                                evaluate(root, p, model, method, phase, task, gpu)
                        write_json(root / f'worker_gpu{gpu}.json', {'status': 'cell_complete',
                                   'model': model, 'method': method, 'updated_utc': now()})
                except BaseException as error:
                    stop.set()
                    write_json(root / f'worker_gpu{gpu}.json', {'status': 'failed', 'error': str(error),
                               'traceback': traceback.format_exc(), 'updated_utc': now()})
                    raise
            stage('running')
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                jobs = [pool.submit(worker, gpu) for gpu in gpus]
                while not all(job.done() for job in jobs):
                    report(root, p)
                    time.sleep(20)
                for job in jobs:
                    job.result()
            report(root, p)
            if read(root / 'summary.json')['complete_cells'] != 40:
                raise ValueError('Full benchmark matrix is incomplete')
            stage('final_statistics')
            final_statistics(root, p)
            stage('complete')
        except BaseException as error:
            stage('failed', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'pipeline', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or set(args.gpus) - {0, 1}:
        raise ValueError('Only distinct GPU0/GPU1 workers are supported')
    if args.command == 'prepare':
        prepare(root)
    elif args.command == 'pipeline':
        pipeline(root, args.gpus)
    else:
        report(root, read(root / 'protocol.json'))


if __name__ == '__main__':
    main()
