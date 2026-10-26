import hashlib
import json
from pathlib import Path

import pytest

from scripts.train.qwen25_vl.noise_ot_reporting import resolve_completed_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(tmp_path):
    plan = {'cells': [{'action': 'evaluate', 'method': 'question_context', 'dataset': 'chartqa', 'questions': 2500}]}
    (tmp_path / 'evaluation_plan.json').write_text(json.dumps(plan))
    config = tmp_path / 'eval.yaml'
    config.write_text('fixed config')
    run = tmp_path / 'run'
    run.mkdir()
    marker = tmp_path / 'eval/full_selective/question_context/chartqa/complete.json'
    marker.parent.mkdir(parents=True)
    saved = {'method': 'question_context', 'dataset': 'chartqa', 'requests': 2500,
        'source': {'commit': 'original'}, 'evaluation_plan_sha256': sha(tmp_path / 'evaluation_plan.json'),
        'config': str(config), 'config_sha256': sha(config), 'run': str(run)}
    marker.write_text(json.dumps(saved))
    return plan, marker, saved


def test_schedule_without_run_uses_completion_marker_without_mutating_inputs(tmp_path):
    plan, marker, _ = fixture(tmp_path)
    before = (tmp_path / 'evaluation_plan.json').read_bytes()
    actual = resolve_completed_plan(tmp_path, plan, file_sha=sha, expected_source={'commit': 'original'})
    assert actual['cells'][0]['run'] == str(tmp_path / 'run')
    assert actual['cells'][0]['task_name'] == 'context_topk_chartqa'
    assert actual['cells'][0]['completion_marker_sha256'] == sha(marker)
    assert 'run' not in plan['cells'][0]
    assert (tmp_path / 'evaluation_plan.json').read_bytes() == before


@pytest.mark.parametrize('key,value', [('source', {'commit': 'wrong'}), ('requests', 2499),
    ('method', 'image_context'), ('dataset', 'ocrbench'), ('evaluation_plan_sha256', 'wrong'),
    ('config_sha256', 'wrong')])
def test_rejects_mismatched_completion_provenance(tmp_path, key, value):
    plan, marker, saved = fixture(tmp_path)
    marker.write_text(json.dumps({**saved, key: value}))
    with pytest.raises(ValueError):
        resolve_completed_plan(tmp_path, plan, file_sha=sha, expected_source={'commit': 'original'})


def test_missing_completion_fails_explicitly(tmp_path):
    plan = {'cells': [{'action': 'evaluate', 'method': 'question_context', 'dataset': 'chartqa', 'questions': 2500}]}
    with pytest.raises(ValueError, match='completion marker'):
        resolve_completed_plan(tmp_path, plan, file_sha=sha, expected_source={})


def test_existing_reused_runs_unchanged():
    plan = {'cells': [{'action': 'reuse', 'run': '/historical', 'task_name': 'original_task'}]}
    assert resolve_completed_plan('/unused', plan, file_sha=sha, expected_source={}) == plan
