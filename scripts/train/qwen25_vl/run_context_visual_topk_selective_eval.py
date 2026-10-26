"""Reuse audited full predictions and evaluate only missing method/task cells.

The model source remains the original hash-frozen context-selector worktree.
This orchestration amendment has its own hash and never changes old results.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event
import time
import traceback

import yaml


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


class TaskLoader(yaml.SafeLoader):
    pass


TaskLoader.add_constructor('!function', lambda loader, node:
    '@function:' + loader.construct_scalar(node).rsplit('.', 1)[-1])


def task_contract(path, config):
    value = yaml.load(path.read_text(), Loader=TaskLoader)
    if 'include' in value:
        inherited = task_contract(path.parent / value['include'], config)
        value = {**inherited, **value}
    value = {k: v for k, v in value.items() if k not in ('task', 'include', 'metadata')}
    checkpoint_generation = read(Path(config['backbone']['model_path']) / 'generation_config.json')
    value['generation_kwargs'] = {
        'repetition_penalty': checkpoint_generation.get('repetition_penalty', 1.),
        **value.get('generation_kwargs', {}), **config['eval'].get('generation', {})}
    return value


def method_contract(config, base_identity):
    stages = []
    for stage in config['pipeline']['stages']:
        stage = {k: v for k, v in stage.items() if k not in ('name', 'enabled')}
        settings = dict(stage['config'])
        if 'identity_path' in settings:
            settings['identity_path'] = read(settings['identity_path'])['identity_sha256']
        stages.append({**stage, 'config': settings})
    backbone = {**config['backbone'], 'model_revision': f'content-sha256:{base_identity}'}
    return {'backbone': backbone, 'stages': stages,
            'seed': config['experiment']['seed'],
            'batch_size': config['eval']['runtime_batch_size']}


def sample_rows(run, name, count, metric):
    import math
    files = list(run.glob(f'**/*_samples_{name}.jsonl'))
    if len(files) != 1:
        raise ValueError(f'Expected exactly one full sample file: {run}/{name}')
    rows = sorted((json.loads(line) for line in files[0].read_text().splitlines()), key=lambda r: r['doc_id'])
    if len(rows) != count or [r['doc_id'] for r in rows] != list(range(count)):
        raise ValueError(f'Incomplete/duplicate full predictions: {name}')
    if not all(math.isfinite(float(row[metric])) for row in rows):
        raise ValueError(f'Nonfinite score: {name}')
    return files[0], rows


def check_new(runner, run, method, task):
    checks = runner.native.check_eval(run, method, 'full', tasks=(task,),
        counts={task: runner.COUNTS[task]}, prefix='context_topk')
    if method.endswith('_context'):
        for line in (run / 'compression_runtime.requests.jsonl').read_text().splitlines():
            delta = json.loads(line)['runtime']['activations'][0]['selection']
            if delta['context_mode'] != method.removesuffix('_context'):
                raise ValueError('Wrong context arm')
            if method == 'question_context' and not (delta['reads_question'] and
                    delta['question_tokens'] and delta['question_ids_sha256']):
                raise ValueError('Missing prompt-only question audit')
    return checks


def audit_reuse(root, protocol, runner, method, task):
    repo = Path(__file__).resolve().parents[3]
    if task == 'ai2d':
        prior = repo / 'outputs/eval/qwen25_vl/projector_dynamic_k128_platform_20261021_v1'
        old_method = 'feature_anchor_adapter_k128' if method == 'old_mlp' else 'dense'
        run = prior / 'full' / old_method
        config = prior / 'configs' / f'full_{old_method}.yaml'
        evidence = run / 'protocol_checks.json'
        saved = read(evidence)
        name = 'qwen25_7b_ratio_ai2d'
        if saved['status'] != 'complete' or saved['counts'][name] != 3088:
            raise ValueError('Historical AI2D is not complete')
        old_protocol = read(prior / 'protocol.json')
        if sha(config) != old_protocol['configs'][str(config)]:
            raise ValueError('Historical AI2D config hash changed')
        utilities = Path(protocol['source']['source_root']) / 'configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio/utils.py'
        if sha(utilities) != old_protocol['implementation_files']['configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio/utils.py']:
            raise ValueError('Historical native prompt/scoring functions changed')
        # Resolve via the recorded include_path rather than an inferred directory.
        old_cfg = yaml.safe_load(config.read_text())
        old_task = Path(old_cfg['eval']['args']['include_path']) / 'ai2d.yaml'
    else:
        prior = repo / 'outputs/eval/qwen25_vl/learned_visual_topk_step320_20261023_v1'
        old_method = 'trained_selector' if method == 'frozen_visual' else method
        evidence = prior / 'eval/full' / old_method / 'complete.json'
        saved = read(evidence)
        run, config = Path(saved['run']), Path(saved['config'])
        if saved['requests'] != 3500 or sha(config) != saved['config_sha256']:
            raise ValueError('Prior full result provenance changed')
        old_protocol = read(prior / 'protocol.json')
        if old_protocol['dependencies'] != protocol['dependencies']:
            raise ValueError('Prior evaluator dependencies differ')
        name = f'learned_topk_{task}'
        old_cfg = yaml.safe_load(config.read_text())
        old_task = Path(old_cfg['eval']['args']['include_path']) / f'{task}.yaml'
        old_source = Path(saved['source']['source_root'])
        for relative in ('mllm_comp/compressors/comparison/qwen_projector.py',
                         'mllm_comp/compressors/comparison/anchor_adapter.py',
                         'mllm_comp/compressors/image/learned_visual_topk.py',
                         'configs/eval/lmms_eval/tasks/qwen25_7b_projector_ratio/utils.py'):
            if sha(old_source / relative) != sha(Path(protocol['source']['source_root']) / relative):
                raise ValueError(f'Prior method/prompt implementation changed: {relative}')
    expected = runner.eval_config(root, protocol, method, 'full', root / 'unused_config_reference')
    current_task = Path(expected['eval']['args']['include_path']) / f'{task}.yaml'
    identity = read(protocol['identity_path'])['identity_sha256']
    old_identity = (old_protocol['base_identity_sha256'] if task == 'ai2d' else
                    read(old_protocol['identity_path'])['identity_sha256'])
    if old_identity != identity or method_contract(old_cfg, old_identity) != method_contract(expected, identity):
        raise ValueError(f'Historical method/base identity differs: {method}/{task}')
    if task_contract(old_task, old_cfg) != task_contract(current_task, expected):
        raise ValueError(f'Historical native task/generation differs: {method}/{task}')
    samples, rows = sample_rows(run, name, runner.COUNTS[task], runner.METRICS[task])
    results, = run.glob('**/*_results.json')
    native = read(results)['results'][name]
    score_key = next(k for k in native if k.split(',')[0] == runner.METRICS[task])
    score = sum(float(row[runner.METRICS[task]]) for row in rows) / len(rows)
    if abs(score - float(native[score_key])) > 1e-10:
        raise ValueError('Full predictions do not reproduce the recorded native metric')
    smoke_pointer = read(root / 'eval/smoke' / method / 'complete.json')
    _, smoke = sample_rows(Path(smoke_pointer['run']), f'context_topk_{task}', 8, runner.METRICS[task])
    for row in smoke:
        before = rows[row['doc_id']]
        if (before['input'], before['target']) != (row['input'], row['target']):
            raise ValueError('Historical question/target differs from current smoke')
    artifacts = [config, evidence, samples, results, old_task]
    return {'method': method, 'dataset': task, 'action': 'reuse', 'run': str(run),
        'task_name': name, 'samples': str(samples), 'score': score,
        'questions': len(rows), 'artifacts_sha256': {str(p): sha(p) for p in artifacts},
        'audit': {'method_backbone_processor_match': True, 'native_task_generation_match': True,
            'base_content_identity': identity,
            'historical_revision': old_cfg['backbone']['model_revision'],
            'current_revision': expected['backbone']['model_revision'],
            'complete_predictions_metric_match': True, 'smoke_prompt_target_match': len(smoke),
            'smoke_prediction_match': sum(rows[s['doc_id']]['filtered_resps'] == s['filtered_resps'] for s in smoke),
            'historical_source_preserved': True,
            'latency_note': 'Historical run; timing is not a matched speed benchmark.'}}


def report(root, plan, runner, write_json, write_text):
    cohort = read(root / 'cohort.json')
    scores, comparisons = {}, {}
    for task, metric in runner.METRICS.items():
        rows = {}
        for method in runner.METHODS:
            cell = next(c for c in plan['cells'] if (c['method'], c['dataset']) == (method, task))
            if cell['action'] == 'reuse':
                run, name = Path(cell['run']), cell['task_name']
            else:
                pointer = root / 'eval/full_selective' / method / task / 'complete.json'
                run = Path(read(pointer)['run'])
                name = f'context_topk_{task}'
            _, rows[method] = sample_rows(run, name, runner.COUNTS[task], metric)
        scores[task] = {m: sum(float(r[metric]) for r in rs) / len(rs) for m, rs in rows.items()}
        ids = [row['image_fingerprint'] for row in cohort[task]['rows']]
        for other in ('image_context', 'frozen_visual', 'old_mlp', 'dense'):
            a, b = rows['question_context'], rows[other]
            if any((x['input'], x['target']) != (y['input'], y['target']) for x, y in zip(a, b)):
                raise ValueError('Paired full question/target mismatch')
            comparisons[f'{task}:question_context_minus_{other}'] = runner.clustered_difference(
                [r[metric] for r in a], [r[metric] for r in b], ids)
    write_json(root / 'summary_full_selective.json', {'scores': scores, 'comparisons': comparisons,
        'evaluation_plan': str(root / 'evaluation_plan.json'), 'limits': plan['limits']})
    lines = ['# Full context-selector comparison', '',
        '| Dataset | Dense | Old MLP | Frozen visual | Image context | Question context |',
        '|---|---:|---:|---:|---:|---:|']
    for task in runner.COUNTS:
        lines.append('| ' + task + ' | ' + ' | '.join(f'{100*scores[task][m]:.2f}%' for m in runner.METHODS) + ' |')
    lines += ['', *plan['limits'], '', 'Eight existing method/task results are linked with provenance; seven missing cells are newly evaluated.']
    for key, row in comparisons.items():
        lines.append(f"\n{key}: {100*row['delta']:+.2f} pp, image-bootstrap 95% CI [{100*row['ci95'][0]:+.2f}, {100*row['ci95'][1]:+.2f}] pp.")
    write_text(root / 'report_full_selective.md', '\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / 'protocol.json')
    sys.path.insert(0, protocol['source']['source_root'])
    from scripts.train.qwen25_vl import run_context_visual_topk as runner
    from mllm_comp.analysis.common.learned_topk_experiment import available_gpus, owned_output
    from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
    owned_output(root)
    with (root / 'selective_eval.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if runner.source_identity(Path(protocol['source']['source_root'])) != protocol['source']:
            raise ValueError('Frozen method source differs')
        if runner.dependency_identity() != protocol['dependencies']:
            raise ValueError('Evaluator dependencies changed')
        smoke = read(root / 'eval_smoke_complete.json')
        if smoke['status'] != 'complete' or set(smoke['methods']) != set(runner.METHODS):
            raise ValueError('All five smoke methods must be complete')
        cells = []
        for method in runner.METHODS:
            for task in runner.COUNTS:
                reuse = method in ('dense', 'old_mlp') or (method == 'frozen_visual' and task != 'ai2d')
                cells.append(audit_reuse(root, protocol, runner, method, task) if reuse else
                    {'method': method, 'dataset': task, 'action': 'evaluate', 'questions': runner.COUNTS[task]})
        plan = {'source': protocol['source'], 'protocol_sha256': sha(root / 'protocol.json'),
            'orchestration_sha256': sha(__file__), 'cells': cells,
            'reused_cells': sum(c['action'] == 'reuse' for c in cells),
            'new_cells': sum(c['action'] == 'evaluate' for c in cells),
            'new_requests': sum(c['questions'] for c in cells if c['action'] == 'evaluate'),
            'development_evaluation': 'Skipped by user preference; smoke already passed.',
            'prior_partial_attempts': 'Preserved; not treated as complete results.',
            'limits': [*protocol['limits'], 'Historical latency and memory are not matched speed comparisons.',
                'AI2D baseline/old MLP use existing native full predictions; their original source and audit records remain linked.']}
        plan_path = root / 'evaluation_plan.json'
        if plan_path.exists() and read(plan_path) != plan:
            raise ValueError('Selective evaluation plan or referenced artifacts changed')
        write_json(plan_path, plan)
        lines = ['# Full evaluation reuse audit', '', '| Method | ChartQA (2500) | OCRBench (1000) | AI2D (3088) |',
            '|---|---|---|---|']
        for method in runner.METHODS:
            lines.append('| ' + method + ' | ' + ' | '.join(next(c['action'] for c in cells
                if (c['method'], c['dataset']) == (method, task)) for task in runner.COUNTS) + ' |')
        lines += ['', f"Reuse: {plan['reused_cells']} cells. New: {plan['new_cells']} cells / {plan['new_requests']} requests."]
        write_text(root / 'evaluation_plan.md', '\n'.join(lines) + '\n')
        print(f"Reuse {plan['reused_cells']}; evaluate {plan['new_cells']} cells / {plan['new_requests']} requests", flush=True)
        if args.prepare_only:
            return
        if len(set(args.gpus)) != len(args.gpus) or available_gpus(args.gpus) != args.gpus:
            raise RuntimeError('Insufficient free GPU memory for requested devices')
        manifest = read(root / 'manifest.json')
        write_json(root / f'manifest_before_selective_{time.time_ns()}.json', manifest)
        manifest.pop('error', None)
        manifest.update(status='running', stage='eval:selective_full', evaluation_plan=str(plan_path),
            reused_cells=plan['reused_cells'], new_cells=plan['new_cells'], updated_utc=runner.native.now())
        write_json(root / 'manifest.json', manifest)
        jobs = Queue()
        for task in runner.COUNTS:
            for method in ('question_context', 'image_context', 'frozen_visual'):
                cell = next(c for c in cells if (c['method'], c['dataset']) == (method, task))
                if cell['action'] == 'evaluate':
                    jobs.put(cell)
        failed = Event()

        def worker(gpu):
            while not failed.is_set():
                try:
                    cell = jobs.get_nowait()
                except Empty:
                    return
                method, task = cell['method'], cell['dataset']
                directory = root / 'eval/full_selective' / method / task
                pointer = directory / 'complete.json'
                try:
                    if pointer.exists():
                        saved = read(pointer)
                        if saved['source'] != protocol['source'] or sha(saved['config']) != saved['config_sha256']:
                            raise ValueError('Completed cell provenance changed')
                        check_new(runner, Path(saved['run']), method, task)
                        continue
                    run = directory / f'attempt_{time.time_ns()}'
                    cfg = runner.eval_config(root, protocol, method, 'full', run)
                    cfg['eval']['tasks'] = [f'context_topk_{task}']
                    config = root / 'configs' / f'full_selective_{method}_{task}_{time.time_ns()}.yaml'
                    write_text(config, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                    print(f'START GPU{gpu}: {method}/{task}, {cell["questions"]} questions', flush=True)
                    runner.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config)],
                        directory / f'launch_{time.time_ns()}.log', [gpu], {
                            'MLLM_TOPK_COHORT': str(root / 'cohort.json'),
                            'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                            'HF_HOME': str(root / 'cache/huggingface'),
                            'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
                            'XDG_CACHE_HOME': str(root / 'cache/xdg')})
                    checks = check_new(runner, run, method, task)
                    write_json(pointer, {'run': str(run), 'method': method, 'dataset': task,
                        'source': protocol['source'], 'config': str(config), 'config_sha256': sha(config),
                        'evaluation_plan_sha256': sha(plan_path), 'requests': checks['requests'],
                        'completed_utc': runner.native.now()})
                    print(f'COMPLETE GPU{gpu}: {method}/{task}', flush=True)
                except BaseException:
                    failed.set()
                    write_json(directory / f'failure_{time.time_ns()}.json', {'method': method, 'dataset': task,
                        'traceback': traceback.format_exc(), 'time': runner.native.now()})
                    raise
                finally:
                    jobs.task_done()

        try:
            with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
                for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
                    future.result()
            report(root, plan, runner, write_json, write_text)
            write_json(root / 'eval_full_selective_complete.json', {'status': 'complete',
                'evaluation_plan_sha256': sha(plan_path), 'new_cells': plan['new_cells'],
                'reused_cells': plan['reused_cells'], 'time': runner.native.now()})
            manifest.update(status='complete', stage='complete', finished_utc=runner.native.now())
            write_json(root / 'manifest.json', manifest)
        except BaseException as exc:
            manifest.update(status='failed', stage='eval:selective_full', error=str(exc), updated_utc=runner.native.now())
            write_json(root / 'manifest.json', manifest)
            raise


if __name__ == '__main__':
    main()
