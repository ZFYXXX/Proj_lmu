import json
import multiprocessing as mp
from pathlib import Path
import time
import pytest

from scripts.eval.lmms_eval.qwen25_portable_queue import (
    claim_next_work, dump, report, seed_completed_cells,
)
from scripts.eval.lmms_eval.portable_results import sample_signature
from scripts.eval.lmms_eval.resident_lmms import model_key


def fake_worker(root, index, reference_only=False):
    root = Path(root)
    methods = ['importance_fps', 'importance_fps_pool'] if reference_only else ['dense', 'old_mlp']
    tasks = ['ai2d', 'ocrbench']
    deadline = time.monotonic() + 20
    preferred = None
    while time.monotonic() < deadline:
        claimed = claim_next_work(root, methods, tasks, 2, None, index, preferred, reference_only=reference_only)
        if claimed is None:
            if all((root/'cells'/m/t/'complete.json').exists() for m in methods for t in tasks):
                return
            time.sleep(.01)
            continue
        method, task, cell, lock, start = claimed
        preferred = method
        try:
            if start is None:
                dump(cell/'complete.json', {'score': {'samples': 12, 'processed_total': 12, 'score': .5}})
            else:
                chunk = cell/f'chunk_{start:09d}'
                with (chunk/'executions.txt').open('a') as stream:
                    stream.write(str(index)+'\n')
                time.sleep(.025)
                dump(chunk/'complete.json', {'chunk': {'total':12, 'start':start, 'stop':start+2}})
        finally:
            lock.close()
    raise RuntimeError('Shared queue deadlocked')


@pytest.mark.parametrize('reference_only', [False, True])
def test_four_processes_share_chunks_exactly_once(tmp_path, reference_only):
    ctx=mp.get_context('fork')
    workers=[ctx.Process(target=fake_worker,args=(str(tmp_path),index,reference_only)) for index in range(4)]
    for worker in workers: worker.start()
    for worker in workers:
        worker.join(25)
        if worker.is_alive(): worker.terminate()
        assert worker.exitcode==0
    executions=list(tmp_path.glob('cells/*/*/chunk_*/executions.txt'))
    assert len(executions)==24
    assert all(len(p.read_text().splitlines())==1 for p in executions)
    assert len({p.read_text().strip() for p in executions})==4
    assert len(list(tmp_path.glob('cells/*/*/complete.json')))==4


def test_full_reuse_allows_different_chunk_size_and_added_methods(tmp_path):
    previous, current = tmp_path/'previous', tmp_path/'new'
    previous.mkdir();current.mkdir()
    plan={'data_root':'/data','model_identity_sha256':'base','limit':None,
          'mmmu_single_image_only':True,'first_shard_only':False,'checkpoint_hashes':{},'chunk_size':512}
    dump(previous/'plan.json',plan)
    dump(current/'plan.json',{**plan,'chunk_size':1024,'methods':['dense','importance_fps']})
    cell=previous/'cells/dense/ai2d';cell.mkdir(parents=True)
    dump(cell/'complete.json',{'score':{'score':.8,'samples':3088,'processed_total':3088}})
    seed_completed_cells(previous,current,['dense','importance_fps'],['ai2d'])
    record=json.loads((current/'cells/dense/ai2d/complete.json').read_text())
    assert record['score']['reused_from']==str(cell)
    work=claim_next_work(current,['dense','importance_fps'],['ai2d'],1024,None,0)
    assert work[0]=='importance_fps' and work[-1]==0
    work[3].close()


def test_reports_preserve_raw_and_relative_and_pending_scores(tmp_path):
    values={'dense':{'ocrbench':.774,'ai2d':.8067,'hallusionbench':None},
            'old_mlp':{'ocrbench':.628,'ai2d':.7895,'hallusionbench':None}}
    for method,tasks in values.items():
        for task,score in tasks.items():
            cell=tmp_path/'cells'/method/task;cell.mkdir(parents=True)
            dump(cell/'complete.json',{'score':{'score':score,'metric':'test','samples':1000}})
    report(tmp_path,list(values),list(values['dense']))
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['raw_scores']['old_mlp']['ocrbench']['score']==.628
    assert summary['raw_display']['old_mlp']['ocrbench']==628
    assert abs(summary['normalized_percent']['old_mlp']['ocrbench']-100*.628/.774)<1e-9
    assert summary['normalized_percent']['old_mlp']['hallusionbench'] is None
    assert 'Raw task scores' in (tmp_path/'report.md').read_text()
    assert 'raw_score' in (tmp_path/'scores.csv').read_text()


def test_sample_identity_includes_prompt_and_target():
    row={'doc_hash':'same-native-hash','input':'question one','target':'A'}
    assert sample_signature(row)!=sample_signature({**row,'input':'question two'})
    assert sample_signature(row)!=sample_signature({**row,'target':'B'})


def test_reference_only_never_schedules_dense_or_waits_for_missing_baseline(tmp_path):
    work=claim_next_work(tmp_path,['importance_fps','importance_fps_pool'],['ai2d'],1024,None,0,reference_only=True)
    assert work[0]=='importance_fps' and work[-1]==0
    work[3].close()


def test_worker_task_preferences_fall_back_without_duplicate_claims(tmp_path):
    from scripts.eval.lmms_eval.qwen25_portable_queue import prioritize_tasks
    tasks = ['gqa', 'refcoco+_bbox_rec_testB', 'refcocog_bbox_rec_test']
    ref_tasks = prioritize_tasks(tasks, tasks[1:])
    general_tasks = prioritize_tasks(tasks, ['gqa'])
    first = claim_next_work(tmp_path, ['importance_fps'], ref_tasks, 512, None, 0, reference_only=True)
    second = claim_next_work(tmp_path, ['importance_fps'], general_tasks, 512, None, 1, reference_only=True)
    try:
        assert first[1] == tasks[1] and second[1] == 'gqa'
        fallback = claim_next_work(tmp_path, ['importance_fps'], general_tasks, 512, None, 1, reference_only=True)
        assert fallback[1] == 'refcocog_bbox_rec_test'
        fallback[3].close()
    finally:
        first[3].close()
        second[3].close()
    with pytest.raises(ValueError, match='unique members'):
        prioritize_tasks(tasks, ['missing'])


def test_resident_identity_ignores_output_tasks_but_not_method():
    spec={'checkpoint_path':'/model','effective_config':{'backbone':{'name':'qwen25_vl'},'pipeline':{'stages':[]}}}
    changed={**spec,'compression_manifest_path':'/next/file.json','effective_config':{
        **spec['effective_config'],'eval':{'tasks':['different']},'output':{'run_dir':'/next'}}}
    assert model_key(spec)==model_key(changed)
    assert model_key(spec)!=model_key({**spec,'effective_config':{
        **spec['effective_config'],'pipeline':{'stages':[{'algorithm':'importance_anchor_adapter'}]}}})
