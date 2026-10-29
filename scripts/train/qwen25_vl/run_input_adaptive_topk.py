"""Fixed-K input-gate experiment; reuse verified controls, native small benches.

Prepare and execute from an isolated hash-frozen source snapshot. No automatic
commits. Existing word-head training is the matched fixed-mix reference; its
data, initialization, frozen prior, schedule and losses are reused exactly.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, owned_output, reject_overlap,
    shared_policy, source_identity, validate_training_entries,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_learned_visual_topk as native
from scripts.train.qwen25_vl import run_context_visual_topk as context
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import (
    method_contract, sample_rows, task_contract,
)

BASE = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
REFERENCE = BASE / 'outputs/train/qwen25_vl/word_visual_topk_20261025_v2'
CONTEXT = BASE / 'outputs/train/qwen25_vl/context_visual_topk_20261024_v2'
EXTRA = BASE / 'outputs/eval/projector_extra/pope_mmstar_k128_20261027_v1'
COUNTS = {'ocrbench': 1000, 'mmstar': 1500, 'ai2d': 3088}
METRICS = {'ocrbench': 'ocrbench_accuracy', 'mmstar': 'mmstar_micro_accuracy', 'ai2d': 'leading_option_accuracy'}
METHODS = ('dense', 'old_mlp', 'visual', 'image_context', 'question_mean', 'word_fixed', 'input_adaptive')


def read(path):
    return json.loads(Path(path).read_text())


def task_info(task):
    if task == 'mmstar':
        return 'projector_extra_mmstar', ROOT / 'configs/eval/lmms_eval/tasks/projector_extra/mmstar.yaml'
    return f'context_topk_{task}', ROOT / f'configs/eval/lmms_eval/tasks/context_visual_topk/{task}.yaml'


def config(root, p, method, phase, task, run):
    cp = read(CONTEXT / 'protocol.json')
    alias = {'visual': 'frozen_visual', 'question_mean': 'question_context'}
    cfg = context.eval_config(CONTEXT, cp, alias.get(method, method) if method not in ('word_fixed', 'input_adaptive') else 'dense', phase, run)
    if method in ('word_fixed', 'input_adaptive'):
        adaptive = method == 'input_adaptive'
        algorithm = 'input_adaptive_topk' if adaptive else 'word_visual_topk'
        checkpoint = native.latest(root / 'training') if adaptive and (root / 'training/latest.json').exists() else (
            root / 'initial' if adaptive else native.latest(REFERENCE / 'training'))
        opts = {'target_count': 128, 'width': 256, 'seed': 42, 'match_width': 128,
            'checkpoint_path': str(checkpoint), 'identity_path': p['identity_path'], 'visual_only': False}
        if adaptive:
            opts['gate_width'] = p['word_spec']['gate_width']
        cfg['pipeline']['stages'] = [{'name': method, 'plugin': algorithm, 'algorithm': algorithm,
            'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl',
            'placement': 'projector_level', 'method': 'token_compression', 'object_family': 'tensor_slot',
            'execution_phase': 'generate', 'modalities': ['image'], 'target_modalities': ['image'],
            'conditioning_modalities': ['text'], 'config': opts}]
        cfg['eval'].update(compression_mode='compressed', target='compressed')
    name, task_file = task_info(task)
    cfg['experiment']['name'] = f'input_adaptive_{phase}_{method}_{task}'
    cfg['eval'].update(tasks=[name], sample_limit=8 if phase == 'smoke' else None)
    cfg['eval']['args']['include_path'] = str(task_file.parent)
    cfg['data'] = {'dataset_id': 'input_adaptive_native_tasks', 'sample_split': 'test'}
    cfg['output'] = {'run_dir': str(run), 'manifest_path': str(run / 'manifest.json')}
    return cfg


def historical(method, task):
    if task == 'mmstar':
        aliases = {'visual': 'visual_selector', 'question_mean': 'question_selector',
            'image_context': 'image_context_selector'}
        if method == 'word_fixed':
            return None
        marker = EXTRA / 'eval/full/qwen25' / aliases.get(method, method) / 'mmstar/complete.json'
        saved = read(marker)
        return {'run': saved['run'], 'config': saved['config'], 'marker': str(marker),
            'config_sha256': saved['config_sha256'], 'task_name': 'projector_extra_mmstar'}
    if method == 'word_fixed':
        marker = REFERENCE / f'eval/full/trained/{task}/complete.json'
        saved = read(marker)
        return {**saved, 'marker': str(marker), 'task_name': f'context_topk_{task}'}
    alias = {'visual': 'frozen_visual', 'question_mean': 'question_context'}.get(method, method)
    cell = next(c for c in read(CONTEXT / 'evaluation_plan.json')['cells'] if (c['method'], c['dataset']) == (alias, task))
    if cell['action'] == 'reuse':
        cfg = next(f for f in cell['artifacts_sha256'] if f.endswith('.yaml') and '/outputs/' in f and '/configs/' in f)
        return {**cell, 'config': cfg, 'config_sha256': file_sha(cfg),
            'marker': next(f for f in cell['artifacts_sha256'] if f.endswith('complete.json') or f.endswith('protocol_checks.json'))}
    marker = CONTEXT / f'eval/full_selective/{alias}/{task}/complete.json'
    saved = read(marker)
    return {**saved, 'marker': str(marker), 'task_name': f'context_topk_{task}'}


def audit_history(root, p, method, task):
    cell = historical(method, task)
    if cell is None:
        return {'method': method, 'dataset': task, 'action': 'evaluate'}
    if file_sha(cell['config']) != cell['config_sha256']:
        raise ValueError('Historical config changed')
    old = yaml.safe_load(Path(cell['config']).read_text())
    expected = config(root, p, method, 'full', task, root / 'unused')
    # Old scorer-only shuffled/generic audits do not change true-query inference.
    for cfg in (old, expected):
        for stage in cfg['pipeline']['stages']:
            stage['config'].pop('audit_conditions', None)
            stage['config'].pop('audit_conditions_sha256', None)
    identity = read(p['identity_path'])['identity_sha256']
    if method_contract(old, identity) != method_contract(expected, identity):
        raise ValueError(f'Historical method/backbone/processor mismatch: {method}/{task}')
    old_task = Path(old['eval']['args']['include_path']) / f'{task}.yaml'
    if task_contract(old_task, old) != task_contract(task_info(task)[1], expected):
        raise ValueError(f'Historical prompt/scoring/generation mismatch: {method}/{task}')
    # The task wrappers reuse these exact primary scoring and prompt functions.
    old_source = old_task.parents[5]
    for relative in ('configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio/utils.py',
                     'configs/eval/lmms_eval/tasks/projector_extra/extra_utils.py'):
        before = old_source / relative
        if before.exists() and file_sha(before) != file_sha(ROOT / relative):
            raise ValueError(f'Historical scoring implementation changed: {relative}')
    sample, rows = sample_rows(Path(cell['run']), cell['task_name'], COUNTS[task], METRICS[task])
    results, = Path(cell['run']).glob('**/*_results.json')
    result = read(results)['results'][cell['task_name']]
    key = next(k for k in result if k.split(',')[0] == METRICS[task])
    score = sum(float(r[METRICS[task]]) for r in rows) / len(rows)
    if abs(score - result[key]) > 1e-9:
        raise ValueError('Historical predictions do not reproduce native metric')
    return {**cell, 'method': method, 'dataset': task, 'action': 'reuse', 'samples': str(sample),
        'artifacts_sha256': {str(f): file_sha(f) for f in (cell['config'], cell['marker'], sample, results, old_task)},
        'audit': 'same checkpoint, processor, method, prompt, native scorer and generation',
        'timing_note': 'Historical timing is not a matched acceleration comparison'}


def prepare(root):
    import pyarrow.parquet as pq
    import torch
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    from configs.eval.lmms_eval.tasks.projector_extra.extra_utils import image_from_doc
    from mllm_comp.compressors.image.learned_visual_topk import load_selector as load_visual
    from mllm_comp.compressors.image.input_adaptive_topk import InputAdaptiveTopK, InputAdaptiveTopKSpec, save_selector
    from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec
    if root.exists():
        raise ValueError('Preserve previous experiment; choose a new run ID')
    root.mkdir(parents=True)
    old = read(REFERENCE / 'protocol.json')
    if read(REFERENCE / 'training/manifest.json')['status'] != 'complete':
        raise ValueError('Fixed-word reference training incomplete')
    if dependency_identity() != old['dependencies']:
        raise ValueError('Evaluator/software drift from matched reference')
    identity = read(old['identity_path']); validate_identity(identity)
    entries = read(old['training_manifest'])['entries']; validate_training_entries(entries)
    cohort = read(CONTEXT / 'cohort.json')
    cohort = {t: cohort[t] for t in ('ocrbench', 'ai2d')}
    mmstar = Path('/home/ZhouFengyu/fastv-repro/datasets/MMStar/mmstar.parquet')
    rows = []
    for batch in pq.ParquetFile(mmstar).iter_batches(batch_size=16):
        for doc in batch.to_pylist():
            rows.append({'row': len(rows), 'image_fingerprint': media_fingerprint([image_from_doc(doc)])})
    cohort['mmstar'] = {'path': str(mmstar), 'rows': rows}
    for task in COUNTS:
        if len(cohort[task]['rows']) != COUNTS[task]:
            raise ValueError(f'Unexpected dataset length: {task}')
        reject_overlap(entries, cohort[task]['rows'])
        print(f'{task}: {len(cohort[task]["rows"])} questions, '
              f'{len({r["image_fingerprint"] for r in cohort[task]["rows"]})} images, train overlap=0', flush=True)
    visual = load_visual(old['warmstart_checkpoint'], base_identity=identity['identity_sha256'], device='cpu')
    spec = InputAdaptiveTopKSpec(**old['word_spec'])
    module = InputAdaptiveTopK(spec, visual=visual)
    fixed = WordVisualTopK(WordVisualTopKSpec(**old['word_spec']), visual=visual)
    fixed_initial = read(REFERENCE / 'initial/selector_config.json')
    from safetensors.torch import load_file
    fixed.load_state_dict(load_file(str(REFERENCE / 'initial/selector.safetensors')), strict=True)
    if any(not torch.equal(value, fixed.state_dict()[key]) for key, value in module.state_dict().items()
           if not key.startswith('input_gate.')):
        raise ValueError('Initial shared parameters differ from the fixed-word reference')
    if fixed_initial['base_identity_sha256'] != identity['identity_sha256']:
        raise ValueError('Fixed-word reference base changed')
    save_selector(module, root / 'initial', base_identity=identity['identity_sha256'], provenance={'matched_word_reference': str(REFERENCE)})
    write_json(root / 'cohort.json', cohort)
    paths = [Path(old['training_manifest']), Path(old['identity_path']), root / 'cohort.json',
        REFERENCE / 'protocol.json', REFERENCE / 'initial/selector.safetensors',
        REFERENCE / 'initial/selector_config.json', root / 'initial/selector.safetensors', root / 'initial/selector_config.json',
        Path(old['warmstart_checkpoint']) / 'selector.safetensors', Path(old['warmstart_checkpoint']) / 'selector_config.json',
        mmstar, *{Path(e['path']) for e in entries}]
    for task in ('ocrbench', 'ai2d'):
        paths.extend(map(Path, cohort[task].get('paths', [cohort[task].get('path')])))
    p = {**old, 'source': source_identity(ROOT), 'created_utc': native.now(), 'resource_policy': shared_policy(),
        'selector_type': 'input_adaptive_topk', 'word_spec': asdict(spec), 'allowed_gpus': [0, 1],
        'initial_parameter_sha256': parameter_digest(module), 'input_hashes': {str(f): file_sha(f) for f in paths},
        'dependencies': dependency_identity(), 'methods': METHODS,
        'training': {**old['training'], 'initialization': 'identical fixed-word initial trunk; input gate initially one',
            'input_gate_width': spec.gate_width, 'evidence_weight': 0.},
        'eval': {'counts': COUNTS, 'primary_metrics': METRICS, 'no_development_pass': True},
        'limits': ['Single training seed, fixed step320, same 2560 training images as the fixed-word reference.',
            'Gate predicts query-relevance strength; no explicit task-type or complexity label claim.',
            'Question word embeddings are not contextual: word order and negation understanding are not established.',
            'Fixed min(128,N), native dynamic resolution, original token content and M-RoPE.',
            'Shared GPUs; historical latency is not a matched speed benchmark. No checkpoint selection on these tests.']}
    p.pop('evidence_training_manifest', None)
    plan = []
    for method in METHODS:
        for task in COUNTS:
            cell = {'method': method, 'dataset': task, 'action': 'evaluate'} if method == 'input_adaptive' else audit_history(root, p, method, task)
            plan.append(cell)
            p['input_hashes'].update(cell.get('artifacts_sha256', {}))
            opts = config(root, p, method, 'full', task, root / 'unused')['pipeline']['stages']
            for stage in opts:
                settings = stage['config']
                for key in ('checkpoint_path', 'adapter_path'):
                    if settings.get(key):
                        folder = Path(settings[key])
                        for artifact in folder.iterdir():
                            if artifact.suffix in ('.json', '.safetensors'):
                                p['input_hashes'][str(artifact)] = file_sha(artifact)
    write_json(root / 'protocol.json', p)
    write_json(root / 'evaluation_plan.json', {'cells': plan, 'limits': p['limits']})
    write_json(root / 'manifest.json', {'status': 'prepared', 'pid': os.getpid(), 'source': p['source'],
        'protocol_sha256': file_sha(root / 'protocol.json'), 'plan_sha256': file_sha(root / 'evaluation_plan.json'),
        'reused_cells': sum(c['action'] == 'reuse' for c in plan), 'new_cells': sum(c['action'] == 'evaluate' for c in plan),
        'trainable_parameters': sum(x.numel() for x in module.parameters() if x.requires_grad), 'updated_utc': native.now()})


def check(run, method, task, phase):
    from scripts.eval.lmms_eval.run_projector_extra_benchmarks import check_trace
    name, _ = task_info(task)
    n = 8 if phase == 'smoke' else COUNTS[task]
    sample, rows = sample_rows(run, name, n, METRICS[task])
    results, = run.glob('**/*_results.json')
    perf = [json.loads(s) for s in (run / 'request_performance.jsonl').read_text().splitlines()]
    if len(perf) != n or any(r['status'] != 'complete' or not math.isfinite(r['generation_s']) for r in perf):
        raise ValueError('Missing or invalid performance records')
    by_id = {r['sample_id']: r for r in perf}
    if len(by_id) != n:
        raise ValueError('Duplicate performance records')
    traces = [json.loads(s) for s in (run / 'compression_runtime.requests.jsonl').read_text().splitlines()]
    seen, gate_values, original, retained = set(), [], [], []
    for record in traces:
        key = record['request']['sample_id']
        if key not in by_id or key in seen:
            raise ValueError('Unknown or duplicated compression trace')
        seen.add(key)
        activation, = record['runtime']['activations']
        source = by_id[key]['original_visual_tokens']
        k = check_trace(activation, source, 28)
        selection = activation['selection']
        if not selection.get('reads_question') or not selection.get('question_tokens'):
            raise ValueError('Missing answer-free question audit')
        if method == 'input_adaptive':
            gate = selection.get('input_gate')
            if not selection.get('input_adaptive_gate') or gate is None or not math.isfinite(gate) or not 0 <= gate <= 2:
                raise ValueError('Missing or invalid input gate')
            gate_values.append(gate)
        original.append(source); retained.append(k)
    if len(seen) != n:
        raise ValueError('Missing compression trace')
    score = sum(float(r[METRICS[task]]) for r in rows) / n
    native_metrics = read(results)['results'][name]
    native_key = next(k for k in native_metrics if k.split(',')[0] == METRICS[task])
    if abs(score - native_metrics[native_key]) > 1e-9:
        raise ValueError('Per-sample score differs from native aggregate')
    checked = {'status': 'complete', 'requests': n, 'samples': str(sample), 'task_name': name,
        'score': score, 'native_metrics': native_metrics, 'mean_original_tokens': sum(original) / n,
        'mean_retained_tokens': sum(retained) / n, 'generation_seconds': sum(r['generation_s'] for r in perf),
        'input_gates': gate_values}
    write_json(run / 'protocol_checks.json', checked)
    return checked


def wait_gpu(gpu, root, stage):
    while available_gpus([gpu]) != [gpu]:
        write_json(root / f'waiting_gpu{gpu}.json', {'stage': stage, 'status': 'waiting_for_memory', 'updated_utc': native.now()})
        time.sleep(20)


def evaluate(root, p, method, task, gpu):
    for phase in ('smoke', 'full'):
        pointer = root / f'eval/{phase}/{method}/{task}/complete.json'
        if pointer.exists():
            saved = read(pointer)
            if saved['config_sha256'] != file_sha(saved['config']) or saved['source'] != p['source']:
                raise ValueError('Completed evaluation provenance changed')
            check(Path(saved['run']), method, task, phase)
            continue
        run = pointer.parent / f'attempt_{time.time_ns()}'
        filename = root / f'configs/{phase}_{method}_{task}_{time.time_ns()}.yaml'
        filename.parent.mkdir(exist_ok=True)
        cfg = config(root, p, method, phase, task, run)
        write_text(filename, yaml.safe_dump(cfg, sort_keys=False))
        wait_gpu(gpu, root, f'{phase}_{method}_{task}')
        native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(filename)],
            pointer.parent / f'launch_{time.time_ns()}.log', [gpu],
            {'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
             'HF_HOME': str(root / 'cache/huggingface'), 'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
             'XDG_CACHE_HOME': str(root / 'cache/xdg')})
        checked = check(run, method, task, phase)
        write_json(pointer, {**checked, 'run': str(run), 'config': str(filename), 'config_sha256': file_sha(filename),
            'source': p['source'], 'finished_utc': native.now()})
        print(f'{phase} {method}/{task}: {checked["score"]:.6f}', flush=True)


def paired_mmstar(a, b, image_ids, *, resamples=10000, seed=42):
    """Recompute category denominators on every image-cluster bootstrap draw."""
    import numpy as np
    categories = sorted({r['average']['l2_category'] for r in a})
    images = sorted(set(image_ids))
    ci, ii = {c: j for j, c in enumerate(categories)}, {v: j for j, v in enumerate(images)}
    sums = np.zeros((len(images), len(categories)))
    counts = np.zeros_like(sums)
    for x, y, image in zip(a, b, image_ids):
        if x['average']['l2_category'] != y['average']['l2_category']:
            raise ValueError('MMStar paired categories differ')
        i, c = ii[image], ci[x['average']['l2_category']]
        sums[i, c] += float(x['average']['score']) - float(y['average']['score'])
        counts[i, c] += 1
    rng, draws = np.random.default_rng(seed), []
    while len(draws) < resamples:
        idx = rng.integers(0, len(images), size=len(images))
        denominator = counts[idx].sum(0)
        if (denominator == 0).any():
            continue
        draws.append(float((sums[idx].sum(0) / denominator).mean()))
    return {'delta': float((sums.sum(0) / counts.sum(0)).mean()),
        'ci95': np.quantile(draws, [.025, .975]).tolist(), 'questions': len(a),
        'independent_images': len(images), 'resamples': resamples, 'estimand': 'native detailed-category macro'}


def report(root, p):
    from configs.eval.lmms_eval.tasks.projector_extra.extra_utils import macro_score
    from collections import defaultdict
    from itertools import combinations
    cohort = read(root / 'cohort.json')
    scores, records, gates, question_audits = {}, {}, {}, {}
    for cell in read(root / 'evaluation_plan.json')['cells']:
        method, task = cell['method'], cell['dataset']
        saved = cell if cell['action'] == 'reuse' else (
            read(root / f'eval/full/{method}/{task}/complete.json')
            if (root / f'eval/full/{method}/{task}/complete.json').exists() else None)
        if saved is None:
            continue
        _, rows = sample_rows(Path(saved['run']), saved['task_name'], COUNTS[task], METRICS[task])
        records[method, task] = rows
        score = macro_score([r['average'] for r in rows]) if task == 'mmstar' else sum(float(r[METRICS[task]]) for r in rows) / len(rows)
        scores.setdefault(task, {})[method] = score
        if method == 'input_adaptive':
            values = saved['input_gates']
            gates[task] = {'mean': sum(values) / len(values), 'min': min(values), 'max': max(values),
                'std': (sum((v - sum(values) / len(values)) ** 2 for v in values) / len(values)) ** .5}
            groups = defaultdict(list)
            for line in (Path(saved['run']) / 'compression_runtime.requests.jsonl').read_text().splitlines():
                record = json.loads(line)
                index = int(record['request']['sample_id'].split(':')[2])
                groups[cohort[task]['rows'][index]['image_fingerprint']].append(record['runtime']['activations'][0]['selection'])
            audits = []
            for image, entries in groups.items():
                pairs = []
                for a, b in combinations(entries, 2):
                    if a['question_ids_sha256'] == b['question_ids_sha256']:
                        continue
                    if a['source_count'] != b['source_count']:
                        raise ValueError('Same image has inconsistent native visual grid')
                    pairs.append({'gate_change': abs(a['input_gate'] - b['input_gate']),
                        'changed_tokens': len(set(a['selected_indices']) - set(b['selected_indices']))})
                if pairs:
                    audits.append({'image_fingerprint': image, 'question_pairs': len(pairs),
                        'mean_gate_change': sum(v['gate_change'] for v in pairs) / len(pairs),
                        'mean_changed_tokens': sum(v['changed_tokens'] for v in pairs) / len(pairs)})
            question_audits[task] = {'independent_multiquestion_images': len(audits), 'images': audits,
                'meaning': 'Natural same-image different prompts; changes alone do not establish task-relevant evidence selection.'}
    comparisons = {}
    for task in COUNTS:
        for other in METHODS[:-1]:
            if ('input_adaptive', task) not in records or (other, task) not in records:
                continue
            a, b = records['input_adaptive', task], records[other, task]
            if any((x['input'], x['target']) != (y['input'], y['target']) for x, y in zip(a, b)):
                raise ValueError('Paired prompts/targets differ')
            # Preserve MMStar's category-macro estimand in bootstrap too.
            if task == 'mmstar':
                comparison = paired_mmstar(a, b, [r['image_fingerprint'] for r in cohort[task]['rows']])
            else:
                av, bv = ([float(r[METRICS[task]]) for r in rows] for rows in (a, b))
                comparison = context.clustered_difference(av, bv, [r['image_fingerprint'] for r in cohort[task]['rows']])
            comparisons[f'{task}:input_adaptive_minus_{other}'] = comparison
    strict_ai2d = {m: sum(float(r['exact_match']) for r in rows) / len(rows)
        for (m, t), rows in records.items() if t == 'ai2d'}
    summary = {'scores': scores, 'paired_comparisons': comparisons, 'input_gate_statistics': gates,
        'same_image_question_audits': question_audits, 'ai2d_strict_match_diagnostic_only': strict_ai2d, 'limits': p['limits']}
    write_json(root / 'summary.json', summary)
    lines = ['# Input-adaptive fixed-K selector', '', '| Dataset | ' + ' | '.join(METHODS) + ' |',
        '|---|' + '---:|' * len(METHODS)]
    for task in COUNTS:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{100*scores[task][m]:.2f}%' if m in scores.get(task, {}) else 'pending' for m in METHODS) + ' |')
    lines.extend(['', 'OCRBench percentages correspond to sum/1000 on the full split. MMStar uses its native category macro average.',
        'AI2D headline uses the previously corrected leading-option accuracy; strict whole-output matching is diagnostic only.', '', *p['limits']])
    write_text(root / 'report.md', '\n'.join(lines) + '\n')


def pipeline(root, p, gpus):
    state = read(root / 'manifest.json')
    def status(value, **more):
        state.update(status=value, pid=os.getpid(), updated_utc=native.now(), **more)
        write_json(root / 'manifest.json', state)
    with (root / 'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            native.validate(root)
            if state['protocol_sha256'] != file_sha(root / 'protocol.json') or state['plan_sha256'] != file_sha(root / 'evaluation_plan.json'):
                raise ValueError('Frozen experiment definition changed')
            status('training_gates', gpus=gpus)
            def train():
                if not (root / 'gates_passed.json').exists():
                    wait_gpu(gpus[0], root, 'real_training_gates')
                    native.gates(root, [gpus[0]])
                if (root / 'training/manifest.json').exists() and read(root / 'training/manifest.json')['status'] == 'complete':
                    return
                status('training', gpus=gpus)
                wait_gpu(gpus[0], root, 'training')
                native.execute(native.training_command(root, root / 'training', [gpus[0]], resume=(root / 'training/latest.json').exists()),
                    root / f'train_launch_{time.time_ns()}.log', [gpus[0]])
            with ThreadPoolExecutor(max_workers=2 if len(gpus) > 1 else 1) as pool:
                training = pool.submit(train)
                controls = None
                if len(gpus) > 1:
                    controls = pool.submit(evaluate, root, p, 'word_fixed', 'mmstar', gpus[1])
                training.result()
                if controls:
                    controls.result()
            if len(gpus) == 1:
                evaluate(root, p, 'word_fixed', 'mmstar', gpus[0])
            status('evaluating')
            report(root, p)
            def worker(gpu, tasks):
                for task in tasks:
                    evaluate(root, p, 'input_adaptive', task, gpu)
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                jobs = [pool.submit(worker, gpu, list(COUNTS)[i::len(gpus)]) for i, gpu in enumerate(gpus)]
                for job in jobs:
                    job.result()
            report(root, p)
            status('complete', finished_utc=native.now())
        except BaseException as error:
            status('failed', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'pipeline', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', nargs='+', type=int, default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == 'prepare':
        prepare(root)
    elif args.command == 'pipeline':
        pipeline(root, read(root / 'protocol.json'), args.gpus)
    else:
        report(root, read(root / 'protocol.json'))


if __name__ == '__main__':
    main()
