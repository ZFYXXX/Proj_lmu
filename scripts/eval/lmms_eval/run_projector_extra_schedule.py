"""Scheduling-only override of a previously frozen benchmark experiment.

Import model/evaluation code from the original immutable source snapshot. A
separate hashed schedule restricts tasks without changing checkpoints, the
original protocol, or any completed predictions. POPE is explicitly deferred.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
from queue import Empty, Queue
import sys
from threading import Event
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text())


def task_scope(protocol, tasks):
    if not tasks or len(set(tasks)) != len(tasks) or set(tasks) - set(protocol['counts']):
        raise ValueError('Select distinct known tasks')
    cells = len(tasks) * len(protocol['models']) * len(protocol['methods'])
    requests = sum(protocol['counts'][task] for task in tasks) * len(protocol['models']) * len(protocol['methods'])
    return {'active_tasks': list(tasks), 'deferred_tasks': [t for t in protocol['counts'] if t not in tasks],
            'active_full_cells': cells, 'active_full_requests': requests}


def run(root, gpus, selected):
    protocol = read(root / 'protocol.json')
    scope = task_scope(protocol, selected)
    sys.path.insert(0, protocol['source']['source_root'])
    runner = importlib.import_module('scripts.eval.lmms_eval.run_projector_extra_benchmarks')
    from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
    runner.owned_output(root)
    runner.validate(root)
    schedule = {**scope, 'gpus': gpus, 'model_source': protocol['source'],
                'protocol_sha256': runner.file_sha(root / 'protocol.json'),
                'orchestration_path': str(Path(__file__).resolve()),
                'orchestration_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'policy': 'User requested MMStar first; deferred tasks do not start automatically.',
                'wait_for': protocol['wait_for']}
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        schedule_path = root / 'schedules' / ('_'.join(selected) + '_v1.json')
        if schedule_path.exists() and read(schedule_path) != schedule:
            raise ValueError('Schedule changed; preserve previous scheduling provenance')
        write_json(schedule_path, schedule)
        state = read(root / 'manifest.json')
        def stage(status, **extra):
            state.update(status=status, stage='selected_tasks_only', updated_utc=runner.now(), pid=os.getpid(),
                         gpus=gpus, **scope, schedule_path=str(schedule_path),
                         original_full_requests=210000, full_requests=scope['active_full_requests'], **extra)
            write_json(root / 'manifest.json', state)
        def report():
            runner.report(root, protocol)
            records, lines = [], ['# MMStar full — Qwen2.5/3.5', '',
                                 'POPE is deferred by user request; no new POPE jobs are scheduled.', '',
                                 '| Model | Method | Native MMStar average | Micro accuracy |',
                                 '|---|---|---:|---:|']
            for model in protocol['models']:
                for method in protocol['methods']:
                    marker = root / 'eval/full' / model / method / 'mmstar/complete.json'
                    if marker.exists():
                        item = read(marker)
                        records.append({'model': model, 'method': method, **item})
                        values = [f'{100 * item["score"]:.2f}%', f'{100 * item["micro_accuracy"]:.2f}%']
                    else:
                        values = ['pending', 'pending']
                    lines.append('| ' + ' | '.join((model, method, *values)) + ' |')
            write_json(root / 'mmstar_summary.json', {'complete_cells': len(records),
                       'total_cells': scope['active_full_cells'], 'deferred_tasks': scope['deferred_tasks'], 'results': records})
            write_text(root / 'mmstar_report.md', '\n'.join(lines) + '\n')
        try:
            report()
            prerequisite = Path(protocol['wait_for'])
            while read(prerequisite)['status'] != 'complete':
                if read(prerequisite)['status'] in ('failed', 'failed_preparation'):
                    raise RuntimeError('Preceding image-context queue failed; inspect before dependent jobs')
                stage('queued', reason='Waiting for existing Qwen3.5 image-context queue; only MMStar follows')
                time.sleep(30)
            queue, stop = Queue(), Event()
            for method in protocol['methods']:
                for model in protocol['models']:
                    queue.put((model, method))
            def worker(gpu):
                try:
                    while not stop.is_set():
                        try:
                            model, method = queue.get_nowait()
                        except Empty:
                            return
                        for phase in ('smoke', 'full'):
                            for task in selected:
                                if stop.is_set():
                                    return
                                runner.evaluate(root, protocol, model, method, phase, task, gpu)
                except BaseException as error:
                    stop.set()
                    write_json(root / f'worker_gpu{gpu}.json', {'status': 'failed', 'error': str(error),
                               'traceback': traceback.format_exc(), 'updated_utc': runner.now()})
                    raise
            stage('running')
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                jobs = [pool.submit(worker, gpu) for gpu in gpus]
                while not all(job.done() for job in jobs):
                    report()
                    time.sleep(20)
                for job in jobs:
                    job.result()
            report()
            if read(root / 'mmstar_summary.json')['complete_cells'] != scope['active_full_cells']:
                raise ValueError('Selected-task matrix incomplete')
            stage('final_statistics')
            cohort, comparisons = read(root / 'cohort.json'), []
            for model in protocol['models']:
                def samples(method):
                    saved = read(root / 'eval/full' / model / method / 'mmstar/complete.json')
                    if runner.file_sha(saved['samples']) != saved['samples_sha256']:
                        raise ValueError('Prediction artifact changed')
                    return sorted([json.loads(s) for s in Path(saved['samples']).read_text().splitlines()], key=lambda r: r['doc_id'])
                baseline = samples('dense')
                rows = cohort['mmstar']['rows']
                for method in protocol['methods'][1:]:
                    actual = samples(method)
                    comparisons.append({'model': model, 'task': 'mmstar', 'method': method,
                        **runner.bootstrap([a['average']['score'] - b['average']['score'] for a, b in zip(actual, baseline)],
                                           [r['image_fingerprint'] for r in rows], [r['l2_category'] for r in rows])})
            write_json(root / 'paired_bootstrap_mmstar.json', {'comparisons': comparisons})
            stage('selected_tasks_complete', reason='MMStar complete; POPE remains deferred, not automatically resumed')
        except BaseException as error:
            stage('failed', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    if not args.gpus or len(set(args.gpus)) != len(args.gpus) or set(args.gpus) - {0, 1}:
        raise ValueError('Use distinct GPU0/GPU1 workers')
    # This override intentionally has one scope; do not accidentally restore POPE.
    run(args.root.resolve(), args.gpus, ['mmstar'])


if __name__ == '__main__':
    main()
