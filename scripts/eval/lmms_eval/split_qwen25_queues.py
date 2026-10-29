"""Hand off GPU0 after the current MLP cell; keep legacy evaluation on GPU1.

The old supervisor has no live reallocation API. Retire only its CPU process,
park its CPU dispatcher, and let its in-flight GPU1 native evaluator finish.
Validate that native artifact independently, then resume the old pipeline with
--gpus 1. Do not interrupt either in-flight model evaluation.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
BASE = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
OLD = Path('/home/ZhouFengyu/fastv-repro/worktrees/qwen25-eval-recovery-20261028-v1')
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.learned_topk_experiment import gpu_processes, owned_output
from mllm_comp.core.serialization import atomic_write_json as write_json
from scripts.eval.lmms_eval.supervise_qwen25_recovery import now, read, retryable


def identity(pid):
    directory = Path(f'/proc/{pid}')
    try:
        stat = (directory/'stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'uid': directory.stat().st_uid, 'state': stat[0],
                'start_ticks': stat[19], 'pgid': int(stat[2]),
                'command': (directory/'cmdline').read_bytes().replace(b'\0', b' ').decode()}
    except FileNotFoundError:
        return None


def checked_signal(saved, sig):
    actual = identity(saved['pid'])
    if actual is None:
        return
    if actual['uid'] != os.getuid() or actual['uid'] != saved['uid'] or actual['start_ticks'] != saved['start_ticks'] or actual['command'] != saved['command']:
        raise ValueError('PID identity changed; refusing signal')
    try:
        os.kill(saved['pid'], sig)
    except ProcessLookupError:
        # A verified process may exit between the identity check and signal.
        return


def running(pid):
    info = identity(pid)
    return info is not None and info['state'] != 'Z'


def orchestrate(root, new_run):
    root.mkdir(parents=True, exist_ok=False)
    legacy = BASE/'outputs/eval/qwen25_vl/pope_refcocoplus_k128_20261027_v2'
    old_ops = BASE/'outputs/operations/qwen25_recovery_20261028_v1'
    old_new = BASE/'outputs/train/qwen25_vl/query_interaction_2x2_20261028_v1'
    old_state = read(old_ops/'status.json')
    pipeline_state = read(legacy/'manifest.json')
    queued_state = read(old_new/'manifest.json')
    supervisor = identity(old_state['supervisor_pid'])
    dispatcher = identity(old_state['child_pid'])
    queued = identity(queued_state['pid'])
    if supervisor is None or dispatcher is None or supervisor['uid'] != os.getuid() or dispatcher['uid'] != os.getuid():
        raise ValueError('Expected owner-local live legacy supervisors')
    if 'supervise_qwen25_recovery.py' not in supervisor['command'] or 'run_qwen25_pope_refcocoplus.py' not in dispatcher['command']:
        raise ValueError('Unexpected supervisor identity')
    if pipeline_state['status'] != 'full_refcocoplus_testA':
        raise ValueError('Expected current testA stage; re-audit before reallocation')
    if queued_state['status'] != 'queued_gpu_reservation' or (queued and 'run_query_interaction_2x2.py' not in queued['command']):
        raise ValueError('Old new-method queue is no longer an idle waiter')
    state = {'status': 'starting', 'pid': os.getpid(), 'started_utc': now(),
             'allocation': {'0': 'new query-interaction 2x2', '1': 'legacy grounding then OK-VQA'},
             'new_run': str(new_run), 'old_supervisor': supervisor, 'old_dispatcher': dispatcher,
             'old_waiter': queued}
    def update(status, **more):
        state.update(status=status, updated_utc=now(), **more)
        if (new_run/'manifest.json').exists():
            state['new_method_state'] = read(new_run/'manifest.json')
        write_json(root/'status.json', state)
    update('retiring_old_waiter')
    write_json(root/'legacy_manifest.before.json', pipeline_state)
    write_json(root/'legacy_supervisor.before.json', old_state)
    write_json(root/'old_new_waiter.before.json', queued_state)
    new_child = None
    with ExitStack() as stack:
        try:
            if queued:
                checked_signal(queued, signal.SIGTERM)
                while running(queued['pid']):
                    time.sleep(1)
            queued_state.update(status='superseded_scheduling_only', updated_utc=now(),
                                replacement=str(new_run), handoff=str(root))
            write_json(old_new/'manifest.json', queued_state)
            mlp_marker = legacy/'eval/full/old_mlp/refcoco_plus_testA/complete.json'
            while not mlp_marker.exists():
                if not running(dispatcher['pid']):
                    raise RuntimeError('Legacy dispatcher stopped before MLP completed')
                update('waiting_for_mlp_cell')
                time.sleep(5)
            if read(legacy/'manifest.json')['status'] != 'full_refcocoplus_testA':
                raise RuntimeError('Legacy stage advanced before handoff; refuse to interrupt new model work')
            worker = read(legacy/'worker_gpu1.json')
            if worker.get('method') != 'eadp' or worker.get('task') != 'refcoco_plus_testA':
                raise ValueError('Expected in-flight EADP worker on GPU1')
            carry = identity(worker['pid'])
            if carry is None or carry['uid'] != os.getuid() or 'mllm_comp.cli.main eval' not in carry['command']:
                raise ValueError('Unexpected in-flight GPU1 evaluator')
            gpu_rows = gpu_processes()
            if any(r['pid'] in (supervisor['pid'], dispatcher['pid']) for r in gpu_rows):
                raise RuntimeError('Scheduling process unexpectedly owns a CUDA context')
            if any(r['gpu']==0 and r['uid']==os.getuid() for r in gpu_rows):
                raise RuntimeError('GPU0 still has an owner model process after MLP marker')
            # Stop dispatch only. Its child evaluator is a separate PID and keeps
            # running on GPU1. TERM on the old supervisor forwards to the whole
            # evaluator group, so retire only that CPU PID with KILL instead.
            checked_signal(dispatcher, signal.SIGSTOP)
            checked_signal(supervisor, signal.SIGKILL)
            while running(supervisor['pid']):
                time.sleep(1)
            old_state.update(status='superseded_by_gpu_split', updated_utc=now(), handoff=str(root))
            write_json(old_ops/'status.json', old_state)
            lock_path = BASE/'outputs/runtime/qwen25_gpu_locks/gpu1.lock'
            lock = stack.enter_context(lock_path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            update('gpu_split_active', mlp_score=read(mlp_marker)['score'], carried_worker=carry)
            env = {**os.environ, 'PYTHONPATH': str(ROOT), 'MLLM_TOPK_HASH_LOCAL_SOURCE': '1',
                   'MLLM_TOPK_NORMAL_GPU': '1', 'OMP_NUM_THREADS': '2', 'TOKENIZERS_PARALLELISM': 'false',
                   'CUBLAS_WORKSPACE_CONFIG': ':4096:8'}
            env.pop('MLLM_TOPK_SHARED_GPU', None)
            with (root/'new_method_launch.log').open('x') as stream:
                new_child = subprocess.Popen([sys.executable, '-u', str(ROOT/'scripts/train/qwen25_vl/run_query_interaction_2x2.py'),
                    'pipeline', '--root', str(new_run), '--gpus', '0'], cwd=ROOT, env=env,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            while running(carry['pid']):
                if new_child.poll() is not None and read(new_run/'manifest.json')['status'] != 'complete':
                    update('new_method_failed_legacy_running', new_method_pid=new_child.pid)
                else:
                    update('new_gpu0_legacy_inflight_gpu1', new_method_pid=new_child.pid)
                time.sleep(10)
            # The parked dispatcher cannot write its complete marker. Its native
            # output is validated with the original frozen checker before reuse.
            log = Path(worker['log'])
            stamp = log.stem.removeprefix('launch_')
            run = log.parent/f'attempt_{stamp}'
            marker = log.parent/'complete.json'
            native_result = read(run/'result.json') if (run/'result.json').exists() else {}
            old_env = {**env, 'PYTHONPATH': str(OLD)}
            if native_result.get('status') == 'succeeded' and native_result.get('native_returncode') == 0:
                code = """import sys
from pathlib import Path
from scripts.eval.lmms_eval import run_qwen25_pope_refcocoplus as r
root,run,marker,config=map(Path,sys.argv[1:])
p=r.validate(root)
checked=r.check(root,'eadp','full','refcoco_plus_testA',run)
r.write_json(marker,{**checked,'run':str(run),'config':str(config),'config_sha256':r.file_sha(config),'finished_utc':r.now()})
r.report(root,p)
"""
                config = legacy/f'configs/full_eadp_refcoco_plus_testA_{stamp}.yaml'
                with (root/'finalize_carried_eadp.log').open('x') as stream:
                    subprocess.run([sys.executable, '-c', code, str(legacy), str(run), str(marker), str(config)],
                                   cwd=OLD, env=old_env, stdout=stream, stderr=subprocess.STDOUT, check=True)
            checked_signal(dispatcher, signal.SIGTERM)
            checked_signal(dispatcher, signal.SIGCONT)
            while running(dispatcher['pid']):
                time.sleep(1)
            for name, script, runroot in [
                ('grounding', 'run_qwen25_pope_refcocoplus.py', legacy),
                ('okvqa', 'run_qwen25_okvqa.py', BASE/'outputs/eval/qwen25_vl/okvqa_full_k128_20261027_v1'),
            ]:
                attempt = 0
                while read(runroot/'manifest.json')['status'] != 'complete':
                    attempt += 1
                    started = time.time()
                    logfile = root/f'gpu1_{name}_{attempt}_{time.time_ns()}.log'
                    with logfile.open('x') as stream:
                        child = subprocess.Popen([sys.executable, '-u', str(OLD/'scripts/eval/lmms_eval'/script),
                            'pipeline', '--root', str(runroot), '--gpus', '1'], cwd=OLD, env=old_env,
                            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                        while child.poll() is None:
                            update('split_running', legacy_task=name, legacy_pid=child.pid, new_method_pid=new_child.pid,
                                   legacy_state=read(runroot/'manifest.json'))
                            time.sleep(10)
                    if child.returncode == 0 and read(runroot/'manifest.json')['status'] == 'complete':
                        break
                    logs = [logfile, *runroot.glob('eval/**/launch_*.log')]
                    if any(retryable(p) for p in logs if p.stat().st_mtime >= started-1):
                        update('gpu1_resource_retry', legacy_task=name);time.sleep(30)
                        continue
                    raise RuntimeError(f'GPU1 {name} failed; preserved {logfile}')
            fcntl.flock(lock, fcntl.LOCK_UN)
            while new_child.poll() is None:
                update('legacy_complete_new_method_running', new_method_pid=new_child.pid)
                time.sleep(10)
            update('complete' if read(new_run/'manifest.json')['status']=='complete' else 'new_method_failed')
        except BaseException as error:
            update('handoff_failed', error=str(error), traceback=traceback.format_exc())
            # Do not kill in-flight model processes on an orchestration error.
            # State/PIDs are retained for explicit recovery.
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--new-run', required=True, type=Path)
    args = parser.parse_args()
    orchestrate(owned_output(args.root), owned_output(args.new_run))


if __name__ == '__main__':
    main()
