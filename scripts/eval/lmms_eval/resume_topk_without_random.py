"""Explicit queue amendment: preserve the active eval, omit pending full random arms.

Do not edit the frozen evaluator snapshot or original protocol. Only the owned
orchestrator is replaced; its active CLI/model children continue unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import sys
import time
from types import SimpleNamespace


def process(pid):
    path = Path('/proc') / str(pid)
    try:
        cmd = (path / 'cmdline').read_bytes().replace(b'\0', b' ').decode()
        status = (path / 'status').read_text()
        ppid = int(next(line.split()[1] for line in status.splitlines() if line.startswith('PPid:')))
        state = next(line.split()[1] for line in status.splitlines() if line.startswith('State:'))
        return {'uid': path.stat().st_uid, 'cmd': cmd, 'ppid': ppid, 'state': state}
    except FileNotFoundError:
        return None


def main(args):
    root = args.root.resolve()
    owner = Path('/home/ZhouFengyu').resolve()
    if not root.is_relative_to(owner) or root == owner:
        raise ValueError('Only operate on the owned experiment tree')
    protocol = json.loads((root / 'protocol.json').read_text())
    source = Path(protocol['source']['source_root'])
    sys.path.insert(0, str(source))
    from mllm_comp.core.serialization import atomic_write_json
    from scripts.train.qwen25_vl import run_learned_visual_topk as native
    native.validate(root)
    omitted = ('random11', 'random23', 'random47')
    active = tuple(m for m in protocol['methods'] if m not in omitted)
    if set(protocol['methods']) != set(native.METHODS) or len(active) != 8:
        raise ValueError('Unexpected original matrix; refuse implicit schedule changes')
    parent, child = process(args.parent_pid), process(args.cli_pid)
    if not parent or parent['uid'] != os.getuid() or str(root) not in parent['cmd'] or 'evaluate-all' not in parent['cmd']:
        raise ValueError('Original orchestrator identity mismatch')
    if not child or child['uid'] != os.getuid() or child['ppid'] != args.parent_pid or 'full_old_mlp_' not in child['cmd']:
        raise ValueError('Expected the currently active old-MLP CLI child')
    for method in omitted:
        if any((root / 'eval/full' / method).glob('attempt_*')):
            raise ValueError('A random arm already started; do not silently discard it')
    attempts = list((root / 'eval/full/old_mlp').glob('attempt_*'))
    if len(attempts) != 1:
        raise ValueError('Cannot resolve the exact active old-MLP attempt')
    config = Path(child['cmd'].split('--config ', 1)[1].split()[0])
    if not config.resolve().is_relative_to(root):
        raise ValueError('Active config is outside the experiment')
    frozen = False
    taken_over = False
    try:
        os.kill(args.parent_pid, signal.SIGSTOP)
        frozen = True
        confirmed = process(args.cli_pid)
        if confirmed is None or any(confirmed[key] != child[key] for key in ('uid', 'cmd', 'ppid')):
            raise ValueError('Active child changed before takeover')
        amendment = {'created_utc': native.now(), 'reason': 'User cancelled three pending random-pruning full evaluations',
            'scope': 'Full evaluation only; completed smoke/development artifacts preserved',
            'omitted_methods': list(omitted), 'active_methods': list(active),
            'original_protocol_sha256': native.file_sha(root / 'protocol.json'),
            'original_source': protocol['source'], 'original_parent_pid': args.parent_pid,
            'preserved_active_cli_pid': args.cli_pid, 'preserved_attempt': str(attempts[0]),
            'continuation_pid': os.getpid(), 'continuation_script': str(Path(__file__).resolve()),
            'continuation_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'status': 'waiting_for_preserved_active_evaluation'}
        atomic_write_json(root / 'schedule_amendment_no_full_random.json', amendment)
        for method in omitted:
            folder = root / 'eval/full' / method
            folder.mkdir(parents=True, exist_ok=True)
            atomic_write_json(folder / 'cancelled.json', {'status': 'cancelled_before_launch',
                'method': method, 'reason': amendment['reason'], 'created_utc': native.now(),
                'completed_development_preserved': True})
        # Signal only the exact owned parent, never its process group/children.
        os.kill(args.parent_pid, signal.SIGTERM)
        os.kill(args.parent_pid, signal.SIGCONT)
        taken_over = True
        print(json.dumps(amendment), flush=True)
        while (current := process(args.cli_pid)) is not None and current['state'] != 'Z':
            if current['uid'] != child['uid'] or current['cmd'] != child['cmd']:
                raise ValueError('CLI PID reused; refusing to monitor another process')
            time.sleep(5)
        checked = native.check_eval(attempts[0], 'old_mlp', 'full')
        atomic_write_json(root / 'eval/full/old_mlp/complete.json', {
            'run': str(attempts[0]), 'requests': checked['requests'], 'source': protocol['source'],
            'config': str(config), 'config_sha256': native.file_sha(config), 'completed_utc': native.now(),
            'continuation': 'Active process completed without restart during explicit queue amendment'})
        native.METHODS = active
        amendment.update(status='continuing_revised_full_schedule', updated_utc=native.now())
        atomic_write_json(root / 'schedule_amendment_no_full_random.json', amendment)
        native.main(SimpleNamespace(command='evaluate-all', root=root, gpus=[1], phase='full', resume=False, trained_run=None))
        amendment.update(status='complete', updated_utc=native.now())
        atomic_write_json(root / 'schedule_amendment_no_full_random.json', amendment)
        summary = json.loads((root / 'summary.json').read_text())
        summary['schedule_amendment'] = amendment
        atomic_write_json(root / 'summary.json', summary)
    except BaseException as exc:
        if taken_over:
            atomic_write_json(root / 'queue_continuation_failure.json', {
                'status': 'failed', 'error': str(exc), 'updated_utc': native.now(),
                'active_evaluation_was_not_terminated': True})
        raise
    finally:
        if frozen and not taken_over and process(args.parent_pid) is not None:
            os.kill(args.parent_pid, signal.SIGCONT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--cli-pid', type=int, required=True)
    main(parser.parse_args())
