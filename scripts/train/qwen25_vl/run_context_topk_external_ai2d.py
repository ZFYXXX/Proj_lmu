"""Complete external K128 AI2D controls; reuse existing ChartQA/OCRBench full runs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event
import time
import traceback

import yaml

METHODS = ('eadp', 'prunesid', 'mmtok')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    reference, root = args.reference.resolve(), args.root.resolve()
    protocol = json.loads((reference / 'protocol.json').read_text())
    sys.path.insert(0, protocol['source']['source_root'])
    from scripts.train.qwen25_vl import run_context_visual_topk as runner
    from mllm_comp.analysis.common.learned_topk_experiment import available_gpus, owned_output
    from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
    # Orchestration helpers are outside the frozen model source, so load them explicitly.
    import runpy
    helper = runpy.run_path(str(Path(__file__).with_name('run_context_visual_topk_selective_eval.py')))
    read, sha = helper['read'], helper['sha']
    rows_for, method_contract, task_contract = helper['sample_rows'], helper['method_contract'], helper['task_contract']
    owned_output(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if runner.source_identity(Path(protocol['source']['source_root'])) != protocol['source']:
            raise ValueError('Frozen model source changed')
        if runner.dependency_identity() != protocol['dependencies']:
            raise ValueError('Evaluator dependency snapshot changed')
        repo = Path(__file__).resolve().parents[3]
        prior = repo / 'outputs/eval/qwen25_vl/learned_visual_topk_step320_20261023_v1'
        old_protocol = read(prior / 'protocol.json')
        if old_protocol['dependencies'] != protocol['dependencies']:
            raise ValueError('Prior external-control evaluator differs')
        identity = read(protocol['identity_path'])['identity_sha256']
        references = []
        for method in METHODS:
            pointer = prior / 'eval/full' / method / 'complete.json'
            saved = read(pointer)
            if saved['requests'] != 3500 or sha(saved['config']) != saved['config_sha256']:
                raise ValueError(f'Prior external full run changed: {method}')
            run, cfg_path = Path(saved['run']), Path(saved['config'])
            old_cfg = yaml.safe_load(cfg_path.read_text())
            expected = runner.eval_config(reference, protocol, method, 'full', root / 'unused')
            if method_contract(old_cfg, identity) != method_contract(expected, identity):
                raise ValueError(f'Prior external method specification differs: {method}')
            for task in ('chartqa', 'ocrbench'):
                old_task = Path(old_cfg['eval']['args']['include_path']) / f'{task}.yaml'
                new_task = Path(expected['eval']['args']['include_path']) / f'{task}.yaml'
                if task_contract(old_task, old_cfg) != task_contract(new_task, expected):
                    raise ValueError(f'Prior native task definition differs: {method}/{task}')
                samples, rows = rows_for(run, f'learned_topk_{task}', runner.COUNTS[task], runner.METRICS[task])
                references.append({'method': method, 'dataset': task, 'run': str(run),
                    'samples': str(samples), 'task_name': f'learned_topk_{task}',
                    'score': sum(float(row[runner.METRICS[task]]) for row in rows) / len(rows),
                    'artifacts_sha256': {str(p): sha(p) for p in (pointer, cfg_path, samples)}})
        plan = {'source': protocol['source'], 'dependencies': protocol['dependencies'],
            'reference': str(reference), 'reference_protocol_sha256': sha(reference / 'protocol.json'),
            'orchestration_sha256': sha(__file__),
            'helper_sha256': sha(Path(__file__).with_name('run_context_visual_topk_selective_eval.py')),
            'methods': list(METHODS), 'new_dataset': 'ai2d', 'questions': 3088,
            'new_requests': 9264, 'target_count': 128, 'reuse': references,
            'eadp': {'alpha': .5, 'beta': 2.,
                'historical_ai2d_excluded': 'Old full run has beta=1; current ChartQA/OCRBench controls have beta=2.'},
            'metrics': ['exact_match', 'leading_option_accuracy'],
            'generation': protocol['eval']['generation'], 'max_new_tokens': 16,
            'limits': ['Qwen2.5-VL-7B adaptations of external methods, not certified reproductions of paper numbers.',
                'Native exact-match remains primary; leading-option score is a separately labeled format sensitivity check.',
                'Reused and new timings are not matched speed measurements.']}
        plan_file = root / 'protocol.json'
        if plan_file.exists() and read(plan_file) != plan:
            raise ValueError('External-control evaluation protocol or referenced artifacts changed')
        write_json(plan_file, plan)
        print('Reuse six ChartQA/OCRBench cells; evaluate three AI2D full cells (9264 requests).', flush=True)
        if args.prepare_only:
            return
        if len(set(args.gpus)) != len(args.gpus) or available_gpus(args.gpus) != args.gpus:
            raise RuntimeError('Insufficient GPU memory for requested devices')
        manifest = {'status': 'running', 'stage': 'smoke', 'source': protocol['source'],
            'protocol_sha256': sha(plan_file), 'started_utc': runner.native.now(), 'gpu_ids': args.gpus}
        write_json(root / 'manifest.json', manifest)
        (root / 'configs').mkdir(exist_ok=True)

        def evaluate_phase(phase):
            jobs, failed = Queue(), Event()
            for method in METHODS:
                jobs.put(method)

            def worker(gpu):
                while not failed.is_set():
                    try:
                        method = jobs.get_nowait()
                    except Empty:
                        return
                    cell = root / 'eval' / phase / method
                    try:
                        marker = cell / 'complete.json'
                        if marker.exists():
                            saved = read(marker)
                            if saved['source'] != protocol['source'] or sha(saved['config']) != saved['config_sha256']:
                                raise ValueError('Completed external-control provenance changed')
                            runner.native.check_eval(Path(saved['run']), method, phase, tasks=('ai2d',),
                                counts={'ai2d': 3088}, prefix='context_topk')
                            continue
                        run = cell / f'attempt_{time.time_ns()}'
                        config = root / 'configs' / f'{phase}_{method}_{time.time_ns()}.yaml'
                        cfg = runner.eval_config(reference, protocol, method, phase, run)
                        cfg['eval']['tasks'] = ['context_topk_ai2d']
                        write_text(config, yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
                        print(f'START GPU{gpu}: {phase}/{method}/ai2d', flush=True)
                        runner.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config)],
                            cell / f'launch_{time.time_ns()}.log', [gpu], {
                                'MLLM_TOPK_COHORT': str(reference / 'cohort.json'),
                                'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                                'HF_HOME': str(reference / 'cache/huggingface'),
                                'HF_DATASETS_CACHE': str(reference / 'cache/datasets'),
                                'XDG_CACHE_HOME': str(reference / 'cache/xdg')})
                        checks = runner.native.check_eval(run, method, phase, tasks=('ai2d',),
                            counts={'ai2d': 3088}, prefix='context_topk')
                        if method == 'eadp':
                            for line in (run / 'compression_runtime.requests.jsonl').read_text().splitlines():
                                effective = json.loads(line)['runtime']['activations'][0]['config']
                                if effective['beta'] != 2. or effective['alpha'] != .5:
                                    raise ValueError('EADP effective alpha/beta differs from existing two-task controls')
                        write_json(marker, {'run': str(run), 'requests': checks['requests'],
                            'source': protocol['source'], 'config': str(config), 'config_sha256': sha(config),
                            'protocol_sha256': sha(plan_file), 'completed_utc': runner.native.now()})
                        print(f'COMPLETE GPU{gpu}: {phase}/{method}/ai2d', flush=True)
                    except BaseException:
                        failed.set()
                        write_json(cell / f'failure_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
                        raise
                    finally:
                        jobs.task_done()
            with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
                for future in [pool.submit(worker, gpu) for gpu in args.gpus]:
                    future.result()
            write_json(root / f'eval_{phase}_complete.json', {'status': 'complete',
                'methods': list(METHODS), 'dataset': 'ai2d', 'time': runner.native.now()})

        try:
            evaluate_phase('smoke')
            manifest.update(stage='full', updated_utc=runner.native.now())
            write_json(root / 'manifest.json', manifest)
            evaluate_phase('full')
            scores = read(reference / 'summary_full_selective.json')['scores']
            ai2d_aux = {}
            current_plan = read(reference / 'evaluation_plan.json')
            for method in runner.METHODS:
                cell = next(c for c in current_plan['cells'] if (c['method'], c['dataset']) == (method, 'ai2d'))
                if cell['action'] == 'reuse':
                    run, name = Path(cell['run']), cell['task_name']
                else:
                    run = Path(read(reference / 'eval/full_selective' / method / 'ai2d/complete.json')['run'])
                    name = 'context_topk_ai2d'
                _, rows = rows_for(run, name, 3088, 'exact_match')
                ai2d_aux[method] = sum(float(r['leading_option_accuracy']) for r in rows) / len(rows)
            for ref in references:
                scores[ref['dataset']][ref['method']] = ref['score']
            for method in METHODS:
                run = Path(read(root / 'eval/full' / method / 'complete.json')['run'])
                _, rows = rows_for(run, 'context_topk_ai2d', 3088, 'exact_match')
                scores['ai2d'][method] = sum(float(r['exact_match']) for r in rows) / len(rows)
                ai2d_aux[method] = sum(float(r['leading_option_accuracy']) for r in rows) / len(rows)
            write_json(root / 'summary.json', {'scores': scores, 'ai2d_leading_option_accuracy': ai2d_aux,
                'protocol': str(plan_file), 'limits': plan['limits']})
            methods = (*runner.METHODS, *METHODS)
            lines = ['# Qwen2.5-VL-7B K128: three-task comparison', '',
                '| Method | ChartQA | OCRBench /1000 | AI2D exact-match | AI2D leading-option (auxiliary) |',
                '|---|---:|---:|---:|---:|']
            for method in methods:
                lines.append(f"| {method} | {100*scores['chartqa'][method]:.2f}% | {1000*scores['ocrbench'][method]:.0f} | "
                    f"{100*scores['ai2d'][method]:.2f}% | {100*ai2d_aux[method]:.2f}% |")
            lines += ['', *plan['limits'], '', 'EADP beta=2 is used consistently across the three datasets.']
            write_text(root / 'report.md', '\n'.join(lines) + '\n')
            manifest.update(status='complete', stage='complete', finished_utc=runner.native.now())
            write_json(root / 'manifest.json', manifest)
        except BaseException as exc:
            manifest.update(status='failed', error=str(exc), updated_utc=runner.native.now())
            write_json(root / 'manifest.json', manifest)
            raise


if __name__ == '__main__':
    main()
