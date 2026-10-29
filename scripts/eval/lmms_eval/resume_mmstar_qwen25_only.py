"""Resume only the existing Qwen2.5 MMStar cells after Qwen3.5 is paused."""
import argparse
from concurrent.futures import ThreadPoolExecutor
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


def alive(pid):
    stat = Path(f'/proc/{pid}/stat')
    return stat.exists() and stat.read_text().split()[2] != 'Z'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--wait-pid', type=int, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / 'protocol.json')
    sys.path.insert(0, protocol['source']['source_root'])
    runner = importlib.import_module('scripts.eval.lmms_eval.run_projector_extra_benchmarks')
    from mllm_comp.core.serialization import atomic_write_json as write_json
    runner.owned_output(root)
    plan = {'model': 'qwen25', 'task': 'mmstar', 'gpus': [0, 1],
            'paused_models': ['qwen35'], 'deferred_tasks': ['pope'],
            'previous_supervisor': args.wait_pid, 'source': protocol['source'],
            'orchestrator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'protocol_sha256': runner.file_sha(root / 'protocol.json')}
    write_json(root / 'schedules/qwen25_only_after_qwen35_pause.json', plan)
    status_file = root / 'qwen25_continuation.json'
    while alive(args.wait_pid):
        write_json(status_file, {**plan, 'status': 'waiting_for_inflight_qwen25', 'updated_utc': runner.now()})
        time.sleep(20)
    with (root / 'pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        runner.validate(root)
        state = read(root / 'manifest.json')
        def status(value, **more):
            state.update(status=value, stage='qwen25_mmstar_only', pid=os.getpid(), updated_utc=runner.now(),
                         active_models=['qwen25'], paused_models=['qwen35'], active_tasks=['mmstar'],
                         deferred_tasks=['pope'], **more)
            write_json(root / 'manifest.json', state)
            write_json(status_file, {**plan, 'status': value, 'updated_utc': runner.now(), **more})
        queue, stop = Queue(), Event()
        for method in protocol['methods']:
            queue.put(method)
        def worker(gpu):
            try:
                while not stop.is_set():
                    try:
                        method = queue.get_nowait()
                    except Empty:
                        return
                    for phase in ('smoke', 'full'):
                        if stop.is_set():
                            return
                        runner.evaluate(root, protocol, 'qwen25', method, phase, 'mmstar', gpu)
            except BaseException as error:
                stop.set()
                write_json(root / f'qwen25_worker_gpu{gpu}.json', {'status': 'failed', 'error': str(error),
                           'traceback': traceback.format_exc(), 'updated_utc': runner.now()})
                raise
        try:
            status('running_qwen25_only')
            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(worker, gpu) for gpu in (0, 1)]
                while not all(job.done() for job in jobs):
                    runner.report(root, protocol)
                    time.sleep(20)
                for job in jobs:
                    job.result()
            runner.report(root, protocol)
            completed = [m for m in protocol['methods'] if (root / 'eval/full/qwen25' / m / 'mmstar/complete.json').exists()]
            if len(completed) != len(protocol['methods']):
                raise ValueError('Qwen2.5 MMStar matrix is incomplete')
            status('qwen25_complete_qwen35_paused', completed_methods=completed)
        except BaseException as error:
            status('qwen25_failed_qwen35_paused', error=str(error), traceback=traceback.format_exc())
            raise


if __name__ == '__main__':
    main()
