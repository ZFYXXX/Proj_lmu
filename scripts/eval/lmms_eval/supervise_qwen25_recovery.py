"""One owner for the interrupted Qwen2.5 GPU queues.

Run independent experiment pipelines in sequence; each pipeline may use both
GPUs internally. Global file locks prevent a second recovery supervisor, and
the process check also detects older workers that do not know about the locks.
Model, scorer, checkpoint and completed-cell provenance stay in each pipeline.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
ADAPTIVE_SOURCE = Path('/home/ZhouFengyu/fastv-repro/worktrees/input-adaptive-topk-20261028-v2')
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.learned_topk_experiment import gpu_free_mib, gpu_processes, owned_output, source_identity
from mllm_comp.core.serialization import atomic_write_json


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def jobs():
    return [
        ('input_adaptive', ADAPTIVE_SOURCE, 'scripts/train/qwen25_vl/run_input_adaptive_topk.py',
         BASE / 'outputs/train/qwen25_vl/input_adaptive_topk_20261028_v2'),
        ('pope_grounding', ROOT, 'scripts/eval/lmms_eval/run_qwen25_pope_refcocoplus.py',
         BASE / 'outputs/eval/qwen25_vl/pope_refcocoplus_k128_20261027_v2'),
        ('okvqa', ROOT, 'scripts/eval/lmms_eval/run_qwen25_okvqa.py',
         BASE / 'outputs/eval/qwen25_vl/okvqa_full_k128_20261027_v1'),
    ]


def competing_workers(processes, gpus, uid):
    return [p for p in processes if p['gpu'] in gpus and p['uid'] == uid]


def retryable(log):
    with Path(log).open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 16000))
        tail = stream.read().decode(errors='replace')
    return any(text in tail for text in (
        'headroom fell below 8 GiB', 'GPU became busy before launch',
        'CUDA out of memory', 'torch.OutOfMemoryError',
    ))


def pipeline(root):
    root = owned_output(root)
    root.mkdir(parents=True, exist_ok=True)
    plan = jobs()
    state = {'supervisor_pid': os.getpid(), 'started_utc': now(), 'source': source_identity(ROOT),
             'gpus': [0, 1], 'completed': [], 'failures': [],
             'policy': 'one pipeline at a time; at most one model worker per GPU within each pipeline',
             'plan': [{'name': n, 'source': str(s), 'run': str(r)} for n, s, _, r in plan]}
    child = None

    def update(status, **extra):
        state.update(status=status, updated_utc=now(), **extra)
        atomic_write_json(root / 'status.json', state)

    def stop(signum, _frame):
        update('interrupted', signal=signum)
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    with ExitStack() as locks:
        folder = BASE / 'outputs/runtime/qwen25_gpu_locks'
        folder.mkdir(parents=True, exist_ok=True)
        for gpu in (0, 1):
            lock = locks.enter_context((folder / f'gpu{gpu}.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        update('starting')
        try:
            for name, source, script, run in plan:
                if name == 'okvqa' and read(plan[1][3] / 'manifest.json')['status'] != 'complete':
                    state['failures'].append({'name': name, 'reason': 'POPE/Grounding prerequisite incomplete'})
                    update('prerequisite_incomplete', current=name)
                    continue
                attempt = 0
                while True:
                    if read(run / 'manifest.json')['status'] == 'complete':
                        state['completed'].append(name)
                        update('stage_complete', current=name)
                        break
                    while True:
                        workers = competing_workers(gpu_processes(), [0, 1], os.getuid())
                        memory = gpu_free_mib()
                        if not workers and all(memory.get(gpu, 0) >= 46000 for gpu in (0, 1)):
                            break
                        update('waiting_for_resources', current=name, competing_workers=workers, free_mib=memory)
                        time.sleep(20)
                    attempt += 1
                    log = root / f'{name}_attempt_{attempt}_{time.time_ns()}.log'
                    env = {**os.environ, 'PYTHONPATH': str(source), 'MLLM_TOPK_HASH_LOCAL_SOURCE': '1',
                           'MLLM_TOPK_NORMAL_GPU': '1', 'OMP_NUM_THREADS': '2',
                           'TOKENIZERS_PARALLELISM': 'false'}
                    env.pop('MLLM_TOPK_SHARED_GPU', None)
                    command = [sys.executable, '-u', str(source / script), 'pipeline', '--root', str(run), '--gpus', '0', '1']
                    launched = time.time()
                    with log.open('x') as output:
                        child = subprocess.Popen(command, cwd=source, env=env, stdout=output,
                                                 stderr=subprocess.STDOUT, start_new_session=True)
                        update('running', current=name, child_pid=child.pid, attempt=attempt, log=str(log))
                        while child.poll() is None:
                            update('running', current=name, child_pid=child.pid,
                                   pipeline_state=read(run / 'manifest.json'), free_mib=gpu_free_mib())
                            time.sleep(10)
                        code = child.returncode
                    child = None
                    actual = read(run / 'manifest.json')
                    if code == 0 and actual['status'] == 'complete':
                        state['completed'].append(name)
                        update('stage_complete', current=name)
                        break
                    # Deep child tracebacks live in per-cell launch logs.
                    candidates = [log, *run.glob('eval/**/launch_*.log')]
                    recent = [p for p in candidates if p.stat().st_mtime >= launched - 1]
                    if any(retryable(p) for p in recent):
                        update('resource_retry', current=name, returncode=code, error=actual.get('error'))
                        time.sleep(30)
                        continue
                    state['failures'].append({'name': name, 'returncode': code, 'log': str(log), 'error': actual.get('error')})
                    update('stage_failed', current=name)
                    break
            update('complete' if not state['failures'] else 'incomplete', finished_utc=now())
        except BaseException as error:
            update('failed', error=str(error))
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    pipeline(parser.parse_args().root)


if __name__ == '__main__':
    main()
