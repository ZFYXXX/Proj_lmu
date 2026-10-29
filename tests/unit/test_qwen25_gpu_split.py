import os
import signal

import pytest

from scripts.train.qwen25_vl.run_query_interaction_2x2 import teacher_assignment
from scripts.eval.lmms_eval import split_qwen25_queues as split


@pytest.mark.parametrize('gpus,expected', [
    ([0], [(0, 0), (1, 0)]),
    ([1], [(0, 1), (1, 1)]),
    ([0, 1], [(0, 0), (1, 1)]),
    ([1, 0], [(0, 1), (1, 0)]),
])
def test_teacher_data_shards_unchanged_under_single_gpu_execution(gpus, expected):
    assert teacher_assignment(gpus) == expected
    assert sorted(s for s, _ in expected) == [0, 1]


@pytest.mark.parametrize('gpus', [[], [0, 0], [2], [-1]])
def test_bad_gpu_allocations_fail(gpus):
    with pytest.raises(ValueError):
        teacher_assignment(gpus)


def test_signal_is_exact_pid_and_owner_checked(monkeypatch):
    saved = {'pid': 123, 'uid': os.getuid(), 'start_ticks': '100', 'command': 'owned dispatcher'}
    sent = []
    monkeypatch.setattr(split, 'identity', lambda pid: dict(saved))
    monkeypatch.setattr(split.os, 'kill', lambda pid, sig: sent.append((pid, sig)))
    split.checked_signal(saved, signal.SIGSTOP)
    assert sent == [(123, signal.SIGSTOP)]
    for changed in ({'uid': os.getuid()+1}, {'start_ticks': '101'}, {'command': 'unrelated worker'}):
        monkeypatch.setattr(split, 'identity', lambda pid: {**saved, **changed})
        with pytest.raises(ValueError, match='refusing signal'):
            split.checked_signal(saved, signal.SIGTERM)
    assert len(sent) == 1


def test_exited_process_and_signal_race_are_harmless(monkeypatch):
    saved = {'pid': 123, 'uid': os.getuid(), 'start_ticks': '100', 'command': 'owned dispatcher'}
    monkeypatch.setattr(split, 'identity', lambda pid: None)
    split.checked_signal(saved, signal.SIGTERM)
    monkeypatch.setattr(split, 'identity', lambda pid: dict(saved))
    def exited(pid, sig):
        raise ProcessLookupError()
    monkeypatch.setattr(split.os, 'kill', exited)
    split.checked_signal(saved, signal.SIGTERM)


def test_identity_and_zombie_check():
    actual = split.identity(os.getpid())
    assert actual['pid'] == os.getpid() and actual['uid'] == os.getuid()
    assert actual['start_ticks'] and split.running(os.getpid())
