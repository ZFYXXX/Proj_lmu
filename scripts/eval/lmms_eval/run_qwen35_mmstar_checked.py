"""Validate MMStar scoring and Qwen3.5 dense baseline before any compressed runs.

Reuses the independently frozen model/bindings; writes a separate protocol,
task snapshot and results. This is a controlled non-thinking, short-output
protocol, not a claim to reproduce the unpublished official MMStar recipe.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
from threading import Event, Lock
import time
import traceback

REPO = Path('/home/ZhouFengyu/fastv-repro/MLLM-Comp-Platform/MLLM-Comp-Platform-main')
PREVIOUS = REPO / 'outputs/eval/projector_extra/pope_mmstar_k128_20261027_v1'
TASK_SOURCE = REPO / 'configs/eval/lmms_eval/tasks/mmstar_qwen35_check'
DATA = Path('/home/ZhouFengyu/fastv-repro/datasets/MMStar/mmstar.parquet')
SUFFIX = '\nAnswer with exactly one letter: A, B, C, or D. Do not explain.'
MODEL_SOURCE = Path('/home/ZhouFengyu/fastv-repro/worktrees/projector-extra-benchmarks-20261027-v1')
VERIFIED_BASELINE = REPO / 'outputs/eval/qwen35_vl/mmstar_protocol_check_20261027_v2'
NATIVE_AUDIT = REPO / 'outputs/analysis/qwen35_mmstar_native_audit_20261027_v1/results.json'
sys.path.insert(0, str(MODEL_SOURCE))
from scripts.eval.lmms_eval import run_projector_extra_benchmarks as old
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return old.file_sha(path)


def pilot_indices(rows, count=20):
    """Stratify by capability and image fingerprint without reading outcomes."""
    categories = sorted({row['category'] for row in rows})
    chosen, used = [], set()
    for category in categories:
        candidates = [row for row in rows if row['category'] == category]
        candidates.sort(key=lambda x: hashlib.sha256(('mmstar-q35-format-v1:' + x['image_fingerprint'] + ':' + str(x['row'])).encode()).hexdigest())
        added = 0
        for row in candidates:
            if row['image_fingerprint'] in used:
                continue
            chosen.append(row['row'])
            used.add(row['image_fingerprint'])
            added += 1
            if added == count:
                break
        if added != count:
            raise ValueError('Insufficient independent images for stratified validation')
    return sorted(chosen)


def prepare(root):
    import shutil
    old.owned_output(root)
    if root.exists():
        raise ValueError('Use a fresh result identity')
    source_protocol = old.validate(PREVIOUS)
    audit = read(NATIVE_AUDIT)
    if audit.get('status') != 'complete' or len(audit.get('samples', [])) != 6 or not all(
            x.get('native_platform_input_and_token_ids_equal') and x.get('saved_evaluation_answer_equal')
            for x in audit['samples']):
        raise ValueError('Independent native model/processor parity audit must pass before launch')
    cohort = read(PREVIOUS / 'cohort.json')['mmstar']
    if len(cohort['rows']) != 1500:
        raise ValueError('Wrong official MMStar split')
    root.mkdir(parents=True)
    tasks = root / 'task_snapshot'
    tasks.mkdir()
    for name in ('mmstar.yaml', 'mmstar_check_utils.py'):
        shutil.copy2(TASK_SOURCE / name, tasks / name)
    indices = pilot_indices(cohort['rows'])
    write_json(root / 'pilot_indices.json', {'indices': indices, 'selection': '20 independent images per category, hashed before outcomes'})
    for name in ('mmstar.yaml','mmstar_check_utils.py'):
        if digest(tasks/name) != digest(VERIFIED_BASELINE/'task_snapshot'/name):
            raise ValueError('Cannot reuse baseline with different task/scoring code')
    if digest(root/'pilot_indices.json') != digest(VERIFIED_BASELINE/'pilot_indices.json'):
        raise ValueError('Cannot reuse baseline with different stratified sample IDs')
    input_hashes = {str(path): digest(path) for path in (DATA, PREVIOUS/'protocol.json', PREVIOUS/'cohort.json',
        root/'pilot_indices.json', tasks/'mmstar.yaml', tasks/'mmstar_check_utils.py', Path(__file__).resolve(),
        NATIVE_AUDIT, VERIFIED_BASELINE/'protocol.json')}
    for phase in ('pilot','full'):
        previous = VERIFIED_BASELINE/'eval'/phase/'dense/complete.json'
        saved = read(previous)
        if digest(saved['config']) != saved['config_sha256'] or digest(saved['samples']) != saved['samples_sha256']:
            raise ValueError('Previous baseline artifacts changed')
        cfg = __import__('yaml').safe_load(Path(saved['config']).read_text())
        if cfg['eval']['generation'] != {'do_sample':False,'repetition_penalty':1.0}:
            raise ValueError('Prior baseline generation protocol differs')
        for path in (previous, Path(saved['config']), Path(saved['samples'])):
            input_hashes[str(path)] = digest(path)
        reused = {**saved,'reused_from':str(previous),'reuse_reason':'same model, task, prompts, generation and strict scorer'}
        write_json(root/'eval'/phase/'dense/complete.json', reused)
    protocol = {'model_source': source_protocol['source'], 'source_experiment': str(PREVIOUS),
        'inputs': input_hashes, 'models': ['qwen35'], 'methods': source_protocol['methods'],
        'pilot_count': len(indices), 'full_count': 1500, 'target_tokens': 128,
        'generation': {'thinking': False, 'max_new_tokens': 128, 'do_sample': False, 'repetition_penalty': 1.0},
        'prompt': SUFFIX, 'scoring': 'whole response single letter or single-field answer JSON, otherwise invalid',
        'validation_gate': {'pilot_valid_fraction_min': .98, 'baseline_full_valid_fraction_min': .98,
                            'native_parity_samples':6,
                            'score_guard_change':'Removed prior 74.7% threshold: official MMStar evaluation protocol is not established to match this fixed non-thinking protocol; it is not an implementation correctness test.'},
        'limits': ['Only launch compression methods after pilot format, full dense format and independent native parity gates pass.',
                   'Official 79.7 uses an unpublished exact per-benchmark protocol; this run is not certified reproduction.',
                   'Qwen2.5 legacy outputs use a different prompt/limit/scorer and are not directly comparable.',
                   'No test tuning, no additional training, no changes to historical results.']}
    write_json(root/'protocol.json', protocol)
    write_json(root/'manifest.json', {'status':'prepared', 'protocol_sha256':digest(root/'protocol.json'),
        'created_utc':old.now(), 'stage':'wait_for_qwen25'})
    return protocol


def validate(root):
    p = read(root/'protocol.json')
    if read(root/'manifest.json')['protocol_sha256'] != digest(root/'protocol.json'):
        raise ValueError('Protocol changed')
    for path, expected in p['inputs'].items():
        if digest(path) != expected:
            raise ValueError(f'Pinned input changed: {path}')
    original = old.validate(PREVIOUS)
    if original['source'] != p['model_source']:
        raise ValueError('Frozen model source changed')
    return p, original


def config(original, root, method, phase, run):
    cfg = old.config(original, 'qwen35', method, 'full', 'mmstar', run)
    cfg['experiment']['name'] = f'qwen35_mmstar_checked_{phase}_{method}'
    cfg['eval']['tasks'] = ['mmstar_qwen35_checked']
    cfg['eval']['sample_limit'] = None
    cfg['eval']['args']['include_path'] = str(root/'task_snapshot')
    cfg['eval']['generation'] = {'do_sample':False, 'repetition_penalty':1.0}
    return cfg


def evaluate(root, p, original, method, phase, gpu):
    import pyarrow.parquet as pq
    import yaml
    marker = root/'eval'/phase/method/'complete.json'
    if marker.exists():
        done = read(marker)
        if digest(done['samples']) != done['samples_sha256'] or digest(done['config']) != done['config_sha256']:
            raise ValueError('Completed result changed')
        return done
    stamp = time.time_ns()
    folder = marker.parent
    run = folder/f'attempt_{stamp}'
    cfg_file = root/'configs'/f'{phase}_{method}_{stamp}.yaml'
    cfg = config(original, root, method, phase, run)
    write_text(cfg_file, yaml.safe_dump(cfg, sort_keys=False))
    env = old.child_env(root, 'qwen35', gpu, phase, run)
    if phase == 'pilot':
        env['MMSTAR_Q35_CHECK_INDICES'] = str(root/'pilot_indices.json')
    else:
        env.pop('MMSTAR_Q35_CHECK_INDICES', None)
    while old.gpu_free_mib().get(gpu, 0) < 40960:
        write_json(root/f'worker_gpu{gpu}.json', {'status':'waiting_for_memory', 'updated_utc':old.now()})
        time.sleep(30)
    folder.mkdir(parents=True, exist_ok=True)
    log = folder/f'launch_{stamp}.log'
    command = [sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(cfg_file)]
    with log.open('x') as stream:
        child = subprocess.Popen(command, cwd=MODEL_SOURCE, env=env, stdout=stream, stderr=subprocess.STDOUT)
        write_json(root/f'worker_gpu{gpu}.json', {'status':'running', 'pid':child.pid, 'phase':phase,
            'method':method, 'log':str(log), 'updated_utc':old.now()})
        code = child.wait()
    if code:
        raise RuntimeError(f'Evaluation failed ({code}), see {log}')
    files = list(run.glob('**/*_samples_mmstar_qwen35_checked.jsonl'))
    if len(files) != 1:
        raise ValueError('Missing/duplicate sample file')
    rows = sorted([json.loads(s) for s in files[0].read_text().splitlines()], key=lambda x:x['doc_id'])
    ids = read(root/'pilot_indices.json')['indices'] if phase == 'pilot' else list(range(1500))
    if len(rows) != len(ids) or [r['doc_id'] for r in rows] != list(range(len(ids))):
        raise ValueError('Sample missing/duplicated')
    table = pq.read_table(DATA, columns=['question', 'answer', 'category'])
    docs = table.take(ids).to_pylist()
    if any(r['target'] != d['answer'] or r['input'] != d['question'].strip() + SUFFIX for r,d in zip(rows,docs)):
        raise ValueError('Prompt/target mismatch')
    # Independent verification of the task parser, not the general LMMS fallback.
    import re
    for r in rows:
        text = r['filtered_resps'].strip()
        match = re.fullmatch(r'(?:([A-Da-d])|\(([A-Da-d])\))[.!]?', text)
        ans = (match.group(1) or match.group(2)).upper() if match else None
        if ans is None:
            try:
                obj = json.loads(text)
                if isinstance(obj,dict) and set(obj)=={'answer'} and isinstance(obj['answer'],str):
                    candidate=obj['answer'].strip().upper()
                    ans=candidate if candidate in ('A','B','C','D') else None
            except (ValueError, TypeError):
                pass
        if float(ans is not None) != float(r['valid_answer']) or float(ans == r['target']) != float(r['strict_accuracy']):
            raise ValueError('Independent score recomputation mismatch')
    perf = [json.loads(s) for s in (run/'request_performance.jsonl').read_text().splitlines()]
    if len(perf) != len(rows) or any(x['status'] != 'complete' or not math.isfinite(x['generation_s']) for x in perf):
        raise ValueError('Generation failed/nonfinite')
    if method != 'dense':
        traces = [json.loads(s) for s in (run/'compression_runtime.requests.jsonl').read_text().splitlines()]
        if len(traces) != len(rows):
            raise ValueError('Compression trace count mismatch')
        originals = {x['sample_id']:x['original_visual_tokens'] for x in perf}
        seen = set()
        for trace in traces:
            sid = trace['request']['sample_id']
            if sid in seen or sid not in originals:
                raise ValueError('Missing/duplicate compression trace')
            seen.add(sid)
            activation, = trace['runtime']['activations']
            old.check_trace(activation, originals[sid], 32)
    categories = {}
    for category in sorted({d['category'] for d in docs}):
        part = [r for r,d in zip(rows,docs) if d['category'] == category]
        categories[category] = {'count':len(part), 'accuracy':sum(r['strict_accuracy'] for r in part)/len(part),
                                'valid_fraction':sum(r['valid_answer'] for r in part)/len(part)}
    result = {'requests':len(rows), 'accuracy':sum(r['strict_accuracy'] for r in rows)/len(rows),
        'valid_fraction':sum(r['valid_answer'] for r in rows)/len(rows), 'categories':categories,
        'invalid_examples':[{'doc_id':r['doc_id'], 'response':r['filtered_resps']} for r in rows if not r['valid_answer']][:20],
        'samples':str(files[0]), 'samples_sha256':digest(files[0]), 'config':str(cfg_file),
        'config_sha256':digest(cfg_file), 'run':str(run), 'finished_utc':old.now()}
    write_json(marker, result)
    return result


def pipeline(root):
    p, original = validate(root)
    manifest = read(root/'manifest.json')
    def state(status, stage, **more):
        manifest.update(status=status, stage=stage, updated_utc=old.now(), pid=os.getpid(), **more)
        write_json(root/'manifest.json', manifest)
    try:
        prior = PREVIOUS/'qwen25_continuation.json'
        while read(prior)['status'] != 'qwen25_complete_qwen35_paused':
            value = read(prior)['status']
            if 'failed' in value:
                raise RuntimeError(f'Qwen2.5 queue failed: {value}')
            state('queued', 'wait_for_qwen25', reason='Preserve active 2.5 full MMStar evaluations')
            time.sleep(30)
        state('running','stratified_format_validation')
        pilot = evaluate(root,p,original,'dense','pilot',0)
        if pilot['valid_fraction'] < p['validation_gate']['pilot_valid_fraction_min']:
            state('gate_failed','pilot_format', result=pilot, reason='Too many invalid or long-form answers; no full compression runs')
            return
        state('running','full_dense_validation', pilot=pilot)
        dense = evaluate(root,p,original,'dense','full',0)
        if dense['valid_fraction'] < p['validation_gate']['baseline_full_valid_fraction_min']:
            state('gate_failed','full_dense_sanity', pilot=pilot, baseline=dense,
                  reason='Baseline fails format sanity guard; compressed comparisons are withheld')
            return
        state('running','compressed_full_evaluations', pilot=pilot, baseline=dense)
        queue, stop, report_lock = Queue(), Event(), Lock()
        def report():
            records=[]
            lines=['# Qwen3.5 MMStar, checked non-thinking protocol', '',
                   '| Method | Accuracy | Valid outputs | Samples |', '|---|---:|---:|---:|']
            for name in p['methods']:
                marker=root/'eval/full'/name/'complete.json'
                if marker.exists():
                    item=read(marker)
                    records.append({'method':name, **item})
                    lines.append(f"| {name} | {item['accuracy']*100:.2f}% | {item['valid_fraction']*100:.2f}% | {item['requests']} |")
                else:
                    lines.append(f'| {name} | pending | pending | — |')
            write_json(root/'summary.json', {'results':records,'complete_cells':len(records),'total_cells':len(p['methods']),
                'warning':'Same non-thinking protocol; not certified reproduction of the Qwen official 79.7 score.'})
            write_text(root/'report.md','\n'.join(lines+['',*p['limits']])+'\n')
        report()
        for method in p['methods']:
            if method != 'dense':queue.put(method)
        def worker(gpu):
            try:
                while not stop.is_set():
                    try: method=queue.get_nowait()
                    except Empty:return
                    evaluate(root,p,original,method,'pilot',gpu)
                    current=read(root/'eval/pilot'/method/'complete.json')
                    if current['valid_fraction'] < p['validation_gate']['pilot_valid_fraction_min']:
                        raise ValueError(f'{method} format gate failed')
                    full=evaluate(root,p,original,method,'full',gpu)
                    if full['valid_fraction'] < p['validation_gate']['baseline_full_valid_fraction_min']:
                        raise ValueError(f'{method} full format gate failed')
                    with report_lock:report()
            except BaseException:
                stop.set()
                raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(worker,gpu) for gpu in (0,1)]
            for future in futures:future.result()
        report()
        dense_rows=sorted([json.loads(s) for s in Path(dense['samples']).read_text().splitlines()],key=lambda r:r['doc_id'])
        image_ids=[r['image_fingerprint'] for r in read(PREVIOUS/'cohort.json')['mmstar']['rows']]
        comparisons=[]
        for method in p['methods']:
            if method=='dense':continue
            result=read(root/'eval/full'/method/'complete.json')
            rows=sorted([json.loads(s) for s in Path(result['samples']).read_text().splitlines()],key=lambda r:r['doc_id'])
            if [r['doc_id'] for r in rows] != [r['doc_id'] for r in dense_rows]:
                raise ValueError('Paired comparisons require identical sample IDs')
            comparisons.append({'method':method,**old.bootstrap(
                [a['strict_accuracy']-b['strict_accuracy'] for a,b in zip(rows,dense_rows)],image_ids)})
        write_json(root/'paired_bootstrap.json',{'comparisons':comparisons,'resampling':'independent image, 10000 draws'})
        state('complete','all_comparisons_complete')
    except BaseException as error:
        state('failed','error',error=str(error),traceback=traceback.format_exc())
        raise


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=('prepare','pipeline'))
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    if args.command=='prepare':prepare(root)
    else:pipeline(root)


if __name__ == '__main__':main()
