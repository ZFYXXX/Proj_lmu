import json
from pathlib import Path

from scripts.eval.lmms_eval import supervise_qwen25_recovery as recovery


def test_resource_wait_recognizes_only_transient_errors(tmp_path):
    log = tmp_path / 'failed.log'
    for message, expected in [
        ('Shared GPU headroom fell below 8 GiB; stopping evaluation', True),
        ('CUDA out of memory', True),
        ('Pinned input changed: checkpoint', False),
        ('Nonfinite native generation logits', False),
    ]:
        log.write_text(message)
        assert recovery.retryable(log) is expected


def test_process_guard_does_not_confuse_other_users_with_duplicate_owner_workers():
    rows = [{'gpu': 0, 'uid': 5, 'pid': 10}, {'gpu': 1, 'uid': 6, 'pid': 11},
            {'gpu': 2, 'uid': 5, 'pid': 12}]
    assert recovery.competing_workers(rows, [0, 1], 5) == rows[:1]


def test_pipeline_serializes_stages_and_retries_resource_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, 'BASE', tmp_path)
    monkeypatch.setattr(recovery, 'owned_output', lambda p: Path(p))
    monkeypatch.setattr(recovery, 'source_identity', lambda p: {'source': 'test'})
    monkeypatch.setattr(recovery, 'gpu_processes', lambda: [])
    monkeypatch.setattr(recovery, 'gpu_free_mib', lambda: {0: 60000, 1: 60000})
    monkeypatch.setattr(recovery.time, 'sleep', lambda _: None)
    monkeypatch.setattr(recovery.signal, 'signal', lambda *args: None)
    plan = []
    for name in ('input_adaptive', 'pope_grounding', 'okvqa'):
        run = tmp_path / name
        run.mkdir()
        (run / 'manifest.json').write_text(json.dumps({'status': 'failed'}))
        plan.append((name, tmp_path, 'fake.py', run))
    monkeypatch.setattr(recovery, 'jobs', lambda: plan)
    calls = []

    class Child:
        def __init__(self, command, *, stdout, **kwargs):
            self.pid = 123456
            run = Path(command[command.index('--root') + 1])
            calls.append(run.name)
            if run.name == 'input_adaptive' and calls.count(run.name) == 1:
                stdout.write('Shared GPU headroom fell below 8 GiB; stopping evaluation\n')
                stdout.flush()
                self.returncode = 2
            else:
                index = [job[3] for job in plan].index(run)
                assert all(recovery.read(job[3] / 'manifest.json')['status'] == 'complete'
                           for job in plan[:index])
                (run / 'manifest.json').write_text(json.dumps({'status': 'complete'}))
                self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(recovery.subprocess, 'Popen', Child)
    recovery.pipeline(tmp_path / 'supervisor')
    status = recovery.read(tmp_path / 'supervisor/status.json')
    assert calls == ['input_adaptive', 'input_adaptive', 'pope_grounding', 'okvqa']
    assert status['status'] == 'complete'
    assert status['completed'] == ['input_adaptive', 'pope_grounding', 'okvqa']
    # Completed runs are skipped when the supervisor itself is restarted.
    calls.clear()
    recovery.pipeline(tmp_path / 'supervisor')
    assert calls == []
