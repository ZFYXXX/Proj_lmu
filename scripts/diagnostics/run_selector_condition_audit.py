"""What does a fixed selector use? Native full eval, selector-only treatments.

No training or method tuning. Reuse true-condition full results, validate them
with a native true-condition smoke, then run matched wrong-question and generic
conditions. The LLM always receives its original image/question/options.
"""
import argparse
from collections import Counter
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, owned_output, shared_policy, source_identity,
)
from mllm_comp.analysis.common.selector_conditioning import FORMAT, GENERIC, ids_sha, matched_donors, text_ids, request_key
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_context_visual_topk as context
from scripts.train.qwen25_vl.noise_ot_reporting import resolve_completed_plan
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import sample_rows, task_contract

TASKS = ('chartqa', 'ocrbench', 'ai2d')
MODES = ('shuffled', 'generic')


def read(path):
    return json.loads(Path(path).read_text())


def trace_rows(run, task):
    rows = {}
    for line in (Path(run) / 'compression_runtime.requests.jsonl').read_text().splitlines():
        data = json.loads(line)
        name, split, index, request = data['request']['sample_id'].split(':')
        if name != f'context_topk_{task}' or split != 'test' or not request.isdigit() or int(index) in rows:
            raise ValueError('Unexpected/duplicate native request identity')
        rows[int(index)] = data['runtime']['activations'][0]['selection']
    return rows


def reference_runs(reference):
    p = resolve_completed_plan(reference, read(reference / 'evaluation_plan.json'), file_sha=file_sha,
        expected_source=read(reference / 'protocol.json')['source'])
    return {t: next(c for c in p['cells'] if c['method'] == 'question_context' and c['dataset'] == t) for t in TASKS}


def config(root, protocol, mode, phase, task, run):
    reference = Path(protocol['reference'])
    cfg = context.eval_config(reference, read(reference / 'protocol.json'), 'question_context', phase, run)
    algorithm = 'selector_condition_audit'
    stage, = cfg['pipeline']['stages']
    stage.update(name=mode, algorithm=algorithm, plugin=algorithm, variant=f'{algorithm}.image',
        binding=f'{algorithm}.image.qwen25_vl', conditioning_modalities=['text'])
    stage['config'].update(condition_mode=mode, condition_manifest=str(root / 'conditions.json'),
        condition_manifest_sha256=protocol['conditions_sha256'])
    cfg['experiment']['name'] = f'selector_question_usage_{phase}_{mode}_{task}'
    cfg['eval'].update(tasks=[f'context_topk_{task}'], sample_limit=8 if phase == 'smoke' else None)
    return cfg


def prepare(root, reference):
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    import scipy
    if root.exists():
        raise ValueError('Preserve old results; use a new experiment root')
    prior = read(reference / 'protocol.json')
    if read(reference / 'manifest.json')['status'] != 'complete' or dependency_identity() != prior['dependencies']:
        raise ValueError('Reference is incomplete or evaluator/software changed')
    identity = read(prior['identity_path'])
    validate_identity(identity)
    checkpoint = context.native.latest(reference / 'arms/question/training')
    # Integrity checked by the unchanged checkpoint loader as well as pins.
    from mllm_comp.compressors.image.context_visual_topk import load_selector
    selector = load_selector(checkpoint, base_identity=identity['identity_sha256'], device='cpu')
    if selector.spec.context_mode != 'question' or selector.spec.target_count != 128:
        raise ValueError('Wrong fixed selector checkpoint')
    tokenizer = AutoTokenizer.from_pretrained(identity['checkpoint'], use_fast=False, local_files_only=True)
    cohort = read(reference / 'cohort.json')
    runs = reference_runs(reference)
    pins = {str(p): file_sha(p) for p in (reference / 'protocol.json', reference / 'evaluation_plan.json',
        reference / 'cohort.json', checkpoint / 'selector.safetensors', checkpoint / 'selector_config.json')}
    rows, stats = [], {}
    for task in TASKS:
        cell = runs[task]
        run = Path(cell['run'])
        samples, baseline = sample_rows(run, cell['task_name'], context.COUNTS[task], context.METRICS[task])
        traces = trace_rows(run, task)
        if set(traces) != set(range(context.COUNTS[task])):
            raise ValueError('Reference traces are incomplete')
        for path in (Path(cell['completion_marker']), samples, run / 'compression_runtime.requests.jsonl'):
            pins[str(path)] = file_sha(path)
        strata = ['all'] * len(baseline)
        if task == 'ocrbench':
            path = '/home/YinPeiran/datasets/lmms_eval_datasets/OCRBench/data/test-00000-of-00001.parquet'
            strata = pq.read_table(path, columns=['question_type']).column(0).to_pylist()
            pins[path] = file_sha(path)
        for item in baseline:
            i = int(item['doc_id'])
            ids = text_ids(tokenizer, item['input'])
            if ids_sha(ids) != traces[i]['question_ids_sha256']:
                raise ValueError(f'Native question tokenization mismatch: {task}/{i}')
            rows.append({'task': task, 'doc_id': i, 'true_ids': ids, 'stratum': strata[i],
                'image_fingerprint': cohort[task]['rows'][i]['image_fingerprint']})
        cfg = context.eval_config(reference, prior, 'question_context', 'full', root / 'unused')
        old_task = Path(prior['source']['source_root']) / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        new_task = ROOT / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        if task_contract(old_task, cfg) != task_contract(new_task, cfg):
            raise ValueError('Native task/generation/scoring contract changed')
    assignment = matched_donors(rows, seed=42)
    manifest_rows = {}
    for row, donor in zip(rows, assignment):
        other = rows[donor['donor_index']]
        sample = request_key(f"context_topk_{row['task']}:test:{row['doc_id']}:0")
        manifest_rows[sample] = {**row, 'donor_ids': other['true_ids'], 'donor_doc_id': other['doc_id'],
            'donor_image_fingerprint': other['image_fingerprint'],
            'generic_ids': text_ids(tokenizer, GENERIC[row['task']]),
            'informative': donor['informative'], 'reason': donor['reason']}
    for task in TASKS:
        selected = [v for v in manifest_rows.values() if v['task'] == task]
        stats[task] = {'questions': len(selected), 'unique_question_ids': len({ids_sha(v['true_ids']) for v in selected}),
            'informative_wrong_questions': sum(v['informative'] for v in selected),
            'reasons': dict(Counter(v['reason'] for v in selected)),
            'mean_true_condition_tokens': sum(len(v['true_ids']) for v in selected) / len(selected),
            'mean_donor_length_difference': sum(abs(len(v['true_ids']) - len(v['donor_ids'])) for v in selected) / len(selected)}
    root.mkdir(parents=True)
    write_json(root / 'conditions.json', {'format': FORMAT, 'base_identity_sha256': identity['identity_sha256'],
        'selector_weights_sha256': file_sha(checkpoint / 'selector.safetensors'), 'rows': manifest_rows})
    protocol = {'source': source_identity(ROOT), 'dependencies': prior['dependencies'], 'resource_policy': shared_policy(),
        'reference': str(reference), 'reference_runs': runs, 'reference_hashes': pins,
        'conditions_sha256': file_sha(root / 'conditions.json'), 'identity_path': prior['identity_path'],
        'selector_checkpoint': str(checkpoint), 'methods': MODES, 'tasks': TASKS,
        'assignment_seed': 42, 'assignment_scipy_version': scipy.__version__, 'condition_stats': stats,
        'generic_text': GENERIC, 'created_utc': context.native.now(),
        'limits': ['No training, no new weights, no noise gate, OT, pooling, dynamic K or Qwen3.5.',
            'Only selector text is replaced; the LLM always receives its actual original image/question/options.',
            'True-condition full predictions are reused; 24 new true smokes must reproduce predictions and token sets.',
            'Matched wrong questions are permuted within dataset (OCRBench additionally within question_type), from a different image.',
            'Assignment maximizes changed questions then matches length, with seeded tie-break; not uniform random shuffling.',
            'Identical question substitutions are non-informative, not evidence that question conditioning is unused.',
            'Generic instructions are not length-matched; AI2D replacement also removes options from SELECTOR input only.',
            'Question use is not evidence of correct semantic grounding; different questions may legitimately use the same visual tokens.',
            'Single fixed checkpoint/assignment seed; image-bootstrap intervals do not cover training or assignment seed uncertainty.',
            'All full samples retained, with informative and identical-selection subgroups reported separately.',
            'Existing full datasets have been inspected; this is exploratory diagnosis, not independent confirmation.',
            'Timings include extra diagnostic scorer passes and are not matched production-speed claims.']}
    write_json(root / 'protocol.json', protocol)
    write_json(root / 'manifest.json', {'status': 'prepared', 'source': protocol['source'], 'completed_cells': [],
        'protocol_sha256': file_sha(root / 'protocol.json'), 'created_utc': context.native.now()})
    print(json.dumps(stats, ensure_ascii=False), flush=True)


def validate(root):
    p = read(root / 'protocol.json')
    if read(root / 'manifest.json')['protocol_sha256'] != file_sha(root / 'protocol.json') or p['source'] != source_identity(ROOT):
        raise ValueError('Frozen protocol/source changed')
    if p['dependencies'] != dependency_identity() or p['resource_policy'] != shared_policy():
        raise ValueError('Evaluator/software/resource policy changed')
    if p['conditions_sha256'] != file_sha(root / 'conditions.json'):
        raise ValueError('Frozen selector conditions changed')
    for path, checksum in p['reference_hashes'].items():
        if file_sha(path) != checksum:
            raise ValueError(f'Reference input changed: {path}')
    validate_identity(read(p['identity_path']))
    return p


def check(root, protocol, run, mode, phase, task):
    checks = context.native.check_eval(run, mode, phase, tasks=(task,), counts=context.COUNTS, prefix='context_topk')
    n = 8 if phase == 'smoke' else context.COUNTS[task]
    _, actual = sample_rows(run, f'context_topk_{task}', n, context.METRICS[task])
    base = protocol['reference_runs'][task]
    _, baseline = sample_rows(Path(base['run']), base['task_name'], context.COUNTS[task], context.METRICS[task])
    original_traces, traces = trace_rows(base['run'], task), trace_rows(run, task)
    conditions = read(root / 'conditions.json')['rows']
    for i, row in enumerate(actual):
        original, current = original_traces[i], traces[i]
        expected = conditions[request_key(f'context_topk_{task}:test:{i}:0')]
        key = {'true': 'true_ids', 'shuffled': 'donor_ids', 'generic': 'generic_ids'}[mode]
        if (row['input'], row['target']) != (baseline[i]['input'], baseline[i]['target']):
            raise ValueError('Actual LLM question/target changed')
        if current['condition_mode'] != mode or current['llm_question_replaced'] or current['condition_ids_sha256'] != ids_sha(expected[key]):
            raise ValueError('Wrong selector-only treatment')
        if current['question_ids_sha256'] != original['question_ids_sha256'] or current['true_selected_indices'] != original['selected_indices']:
            raise ValueError('Underlying true-condition selector/vision features drifted')
        if current['selection_identical'] and row['filtered_resps'] != baseline[i]['filtered_resps']:
            raise ValueError('Same token set produced different answers; do not attribute this to selector conditions')
        if mode == 'true' and (not current['selection_identical'] or current['selected_indices'] != original['selected_indices']):
            raise ValueError('True smoke did not reproduce original selection')
    return checks


def report(root, protocol):
    cohort = read(Path(protocol['reference']) / 'cohort.json')
    records = []
    for task in TASKS:
        ref = protocol['reference_runs'][task]
        _, baseline = sample_rows(Path(ref['run']), ref['task_name'], context.COUNTS[task], context.METRICS[task])
        for mode in MODES:
            marker = root / f'eval/full/{mode}/{task}/complete.json'
            if not marker.exists():
                continue
            run = Path(read(marker)['run'])
            _, actual = sample_rows(run, f'context_topk_{task}', context.COUNTS[task], context.METRICS[task])
            traces = trace_rows(run, task)
            ids = [r['image_fingerprint'] for r in cohort[task]['rows']]
            a, b = [[float(r[context.METRICS[task]]) for r in data] for data in (actual, baseline)]
            groups = {'all': list(range(len(a))), 'selection_changed': [i for i in range(len(a)) if not traces[i]['selection_identical']],
                'selection_identical': [i for i in range(len(a)) if traces[i]['selection_identical']]}
            if mode == 'shuffled':
                groups['informative_wrong_question'] = [i for i in range(len(a)) if traces[i]['shuffle_informative']]
            item = {'task': task, 'mode': mode, 'questions': len(a), 'run': str(run),
                'score': sum(a)/len(a), 'control_score': sum(b)/len(b),
                'mean_selection_overlap': sum(traces[i]['selection_overlap_fraction'] for i in range(len(a))) / len(a),
                'mean_selection_jaccard': sum(traces[i]['selection_jaccard'] for i in range(len(a))) / len(a),
                'selected_set_changed_questions': len(groups['selection_changed']),
                'prediction_changed_questions': sum(x['filtered_resps'] != y['filtered_resps'] for x, y in zip(actual, baseline)),
                'score_wins': sum(x>y for x,y in zip(a,b)), 'score_losses': sum(x<y for x,y in zip(a,b)), 'subgroups': {}}
            cases = []
            for i in range(len(a)):
                cases.append({'doc_id': i, 'image_fingerprint': ids[i], 'question': actual[i]['input'],
                    'baseline_prediction': baseline[i]['filtered_resps'], 'condition_prediction': actual[i]['filtered_resps'],
                    'baseline_score': b[i], 'condition_score': a[i], 'score_delta': a[i]-b[i],
                    'selection_overlap_fraction': traces[i]['selection_overlap_fraction'],
                    'condition_ids_sha256': traces[i]['condition_ids_sha256'],
                    'donor_doc_id': traces[i]['donor_doc_id'], 'shuffle_informative': traces[i]['shuffle_informative']})
            write_text(root / f'cases/{mode}_{task}.jsonl', ''.join(json.dumps(c, ensure_ascii=False, allow_nan=False)+'\n' for c in cases))
            for name, indices in groups.items():
                item['subgroups'][name] = None if not indices else context.clustered_difference(
                    [a[i] for i in indices], [b[i] for i in indices], [ids[i] for i in indices])
            records.append(item)
    write_json(root / 'summary.json', {'completed_cells': len(records), 'total_cells': 6, 'results': records,
        'condition_stats': protocol['condition_stats'], 'limits': protocol['limits']})
    lines = ['# What does the fixed question selector use?', '',
        '| Dataset | Selector condition | Task score | True control | Delta (pp), 95% image CI | Mean token overlap | Changed selections |',
        '|---|---|---:|---:|---|---:|---:|']
    for r in records:
        d = r['subgroups']['all']
        lines.append(f"| {r['task']} | {r['mode']} | {100*r['score']:.2f}% | {100*r['control_score']:.2f}% | {100*d['delta']:+.2f} [{100*d['ci95'][0]:+.2f}, {100*d['ci95'][1]:+.2f}] | {100*r['mean_selection_overlap']:.2f}% | {r['selected_set_changed_questions']}/{r['questions']} |")
    write_text(root / 'report.md', '\n'.join(lines + ['', 'OCRBench percentages are native points/1000; AI2D headline is strict matching.', '', *protocol['limits']])+'\n')


def pipeline(root, gpu):
    p = validate(root)
    manifest = read(root / 'manifest.json')
    def evaluate(mode, phase, task):
        parent = root / f'eval/{phase}/{mode}/{task}'
        marker = parent / 'complete.json'
        if marker.exists():
            saved = read(marker)
            if saved['source'] != p['source'] or file_sha(saved['config']) != saved['config_sha256']:
                raise ValueError('Completed cell provenance changed')
            check(root, p, Path(saved['run']), mode, phase, task)
            return
        while available_gpus([gpu]) != [gpu]:
            manifest.update(status='queued', stage='waiting_for_gpu', updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            time.sleep(20)
        validate(root)
        manifest.update(status='running', stage=f'{phase}:{mode}:{task}', gpu=gpu, updated_utc=context.native.now())
        write_json(root / 'manifest.json', manifest)
        run = parent / f'attempt_{time.time_ns()}'
        filename = root / 'configs' / f'{phase}_{mode}_{task}_{time.time_ns()}.yaml'
        write_text(filename, yaml.safe_dump(config(root, p, mode, phase, task, run), sort_keys=False))
        context.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(filename)],
            parent / f'launch_{time.time_ns()}.log', [gpu], {'MLLM_TOPK_COHORT': str(Path(p['reference'])/'cohort.json'),
                'MLLM_TOPK_PERFORMANCE': str(run/'request_performance.jsonl'),
                'HF_HOME': str(root/'cache/huggingface'), 'HF_DATASETS_CACHE': str(root/'cache/datasets'),
                'XDG_CACHE_HOME': str(root/'cache/xdg')})
        result = check(root, p, run, mode, phase, task)
        write_json(marker, {'run': str(run), 'source': p['source'], 'config': str(filename),
            'config_sha256': file_sha(filename), 'requests': result['requests'], 'completed_utc': context.native.now()})
        if phase == 'full':
            manifest['completed_cells'] = sorted(set(manifest['completed_cells']) | {f'{mode}/{task}'})
            report(root, p)
            write_json(root/'manifest.json', manifest)
    try:
        for mode in ('true', *MODES):
            for task in TASKS:
                evaluate(mode, 'smoke', task)
        write_json(root/'smoke_complete.json', {'status': 'complete', 'requests': 72, 'time': context.native.now()})
        for mode in MODES:
            for task in TASKS:
                evaluate(mode, 'full', task)
        manifest.update(status='complete', stage='complete', finished_utc=context.native.now())
        write_json(root/'manifest.json', manifest)
        write_json(root/'full_complete.json', {'status': 'complete', 'new_requests': 13176,
            'reused_true_requests': 6588, 'time': context.native.now()})
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc), updated_utc=context.native.now())
        write_json(root/'manifest.json', manifest)
        write_json(root/f'failure_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'pipeline'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--gpu', type=int, choices=(0, 1), default=1)
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == 'prepare':
        if args.reference is None:
            parser.error('--reference is required')
        prepare(root, args.reference.resolve())
    else:
        with (root/'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(root, args.gpu)
