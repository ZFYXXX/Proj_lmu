"""Independent noise-training and OT-inference arms; reuse old full controls.

No automatic commits, no development sweep, no combined noise+OT arm. All task
generation/scoring goes through the native MLLM-Comp evaluator.
"""
import argparse
import fcntl
import json
import math
from pathlib import Path
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    available_gpus, dependency_identity, owned_output, shared_policy, source_identity,
)
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_context_visual_topk as context
from scripts.train.qwen25_vl.run_context_visual_topk_selective_eval import sample_rows, task_contract

METHODS = ('ot', 'vp_noise')
TASKS = ('chartqa', 'ocrbench', 'ai2d')


def read(path):
    return json.loads(Path(path).read_text())


def config(root, protocol, method, phase, task, run):
    if method not in METHODS or task not in TASKS or phase not in ('smoke', 'full'):
        raise ValueError('Unknown noise/OT evaluation cell')
    reference = Path(protocol['reference'])
    cfg = context.eval_config(reference, read(reference / 'protocol.json'), 'question_context', phase, run)
    stage, = cfg['pipeline']['stages']
    stage['conditioning_modalities'] = ['text']
    if method == 'ot':
        algorithm = 'context_visual_topk_ot'
        stage.update(name=method, plugin=algorithm, algorithm=algorithm,
            variant=f'{algorithm}.image', binding=f'{algorithm}.image.qwen25_vl')
        stage['config'].update({'ot_' + k: v for k, v in protocol['ot'].items()})
    else:
        train = root / 'noise/training'
        if read(train / 'manifest.json')['status'] != 'complete' or read(train / 'latest.json')['step'] != 320:
            raise ValueError('Noise eval requires completed fixed step320, not a smoke checkpoint')
        stage['config']['checkpoint_path'] = str(context.native.latest(train))
    cfg['experiment']['name'] = f'noise_ot_{phase}_{method}_{task}'
    cfg['eval'].update(tasks=[f'context_topk_{task}'], sample_limit=8 if phase == 'smoke' else None)
    return cfg


def prepare(root, reference):
    from mllm_comp.compressors.image.query_transport import QueryTransportSpec
    from dataclasses import asdict
    if root.exists():
        raise ValueError('Use a new run directory; old artifacts must not be overwritten')
    old = read(reference / 'protocol.json')
    arm = read(reference / 'arms/question/protocol.json')
    if read(reference / 'manifest.json')['status'] != 'complete':
        raise ValueError('Reference full comparison is not complete')
    if dependency_identity() != arm['dependencies']:
        raise ValueError('Evaluator/software differs from the reference')
    for name, checksum in arm['input_hashes'].items():
        if file_sha(name) != checksum:
            raise ValueError(f'Reference input changed: {name}')
    validate_identity(read(arm['identity_path']))
    for task in TASKS:
        prior = Path(old['source']['source_root']) / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        current = ROOT / 'configs/eval/lmms_eval/tasks/context_visual_topk' / f'{task}.yaml'
        cfg = context.eval_config(reference, old, 'question_context', 'full', root / 'unused')
        if task_contract(prior, cfg) != task_contract(current, cfg):
            raise ValueError('Native prompt/scoring/generation changed')
    checkpoint = context.native.latest(reference / 'arms/question/training')
    pins = {str(p): file_sha(p) for p in (reference / 'protocol.json', reference / 'evaluation_plan.json',
        reference / 'cohort.json', reference / 'arms/question/protocol.json',
        checkpoint / 'selector_config.json', checkpoint / 'selector.safetensors')}
    protocol = {'created_utc': context.native.now(), 'source': source_identity(ROOT),
        'dependencies': arm['dependencies'], 'resource_policy': shared_policy(),
        'reference': str(reference), 'reference_hashes': pins, 'identity_path': arm['identity_path'],
        'methods': METHODS, 'tasks': TASKS, 'ot': asdict(QueryTransportSpec()),
        'limits': ['Single-seed exploratory experiment; no test-based checkpoint/hyperparameter selection.',
            'Noise arm changes ONLY the training gate; same scorer initialization/data/320 steps/loss as question-context control.',
            'Noise arm is inspired by DiffPrune, without its denoising block or altered architecture: NOT a reproduction.',
            'Standard-normal noise does not guarantee raw-feature variance preservation or train/deploy equivalence.',
            'OT arm uses the existing question selector; no retraining or nonlinear cross-modal alignment.',
            'Fixed query marginals do not guarantee that the final hard subset covers every question entity.',
            'No dynamic K, token recovery, Qwen3.5, or combined noise+OT arm.',
            'Reuse existing full quality results only; historical latency is NOT a matched speed comparison.',
            'No 500-image development pass; eight-request smoke is engineering validation only.']}
    root.mkdir(parents=True)
    noise = root / 'noise'
    noise.mkdir()
    arm = {**arm, 'source': protocol['source'], 'resource_policy': protocol['resource_policy'],
        'created_utc': context.native.now(), 'claims': 'Matched training-gate change only; same CE+0.5KL+maskBCE',
        'training': {**arm['training'], 'training_mask_mode': 'vp_noise'}, 'limits': protocol['limits']}
    arm['input_hashes'] = {**arm['input_hashes'], **pins}
    write_json(noise / 'protocol.json', arm)
    protocol['noise_protocol_sha256'] = file_sha(noise / 'protocol.json')
    write_json(root / 'protocol.json', protocol)
    write_json(root / 'manifest.json', {'status': 'prepared', 'source': protocol['source'],
        'protocol_sha256': file_sha(root / 'protocol.json'), 'completed_cells': [], 'created_utc': context.native.now()})


def validate(root):
    protocol = read(root / 'protocol.json')
    if read(root / 'manifest.json')['protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Master protocol changed')
    if protocol['source'] != source_identity(ROOT) or protocol['dependencies'] != dependency_identity():
        raise ValueError('Frozen source/software changed')
    if protocol['resource_policy'] != shared_policy():
        raise ValueError('Resource policy changed')
    for p, expected in protocol['reference_hashes'].items():
        if file_sha(p) != expected:
            raise ValueError('Reference artifact changed')
    if protocol['noise_protocol_sha256'] != file_sha(root / 'noise/protocol.json'):
        raise ValueError('Noise training protocol changed')
    context.native.validate(root / 'noise')
    return protocol


def check(run, method, task, phase):
    checks = context.native.check_eval(run, method, phase, tasks=(task,), counts=context.COUNTS, prefix='context_topk')
    for line in (run / 'compression_runtime.requests.jsonl').read_text().splitlines():
        trace = json.loads(line)['runtime']['activations'][0]['selection']
        if not trace['reads_question'] or not trace.get('question_ids_sha256') or trace['context_mode'] != 'question':
            raise ValueError('Missing answer-free question audit')
        if trace['decode_position_verified'] is False:
            raise ValueError('Invalid decode position')
        if method == 'ot' and trace['source_count'] > 128:
            if not trace['ot_applied'] or trace['ot_row_marginal_error'] > 1e-5:
                raise ValueError('OT was not applied or failed its marginal check')
    return checks


def report(root, protocol):
    from scripts.train.qwen25_vl.noise_ot_reporting import resolve_completed_plan
    reference = Path(protocol['reference'])
    plan, cohort = read(reference / 'evaluation_plan.json'), read(reference / 'cohort.json')
    plan = resolve_completed_plan(reference, plan, file_sha=file_sha,
        expected_source=read(reference / 'protocol.json')['source'])
    records = []
    for task in TASKS:
        cell = next(c for c in plan['cells'] if (c['method'], c['dataset']) == ('question_context', task))
        _, control = sample_rows(Path(cell['run']), cell['task_name'], context.COUNTS[task], context.METRICS[task])
        for method in METHODS:
            marker = root / f'eval/full/{method}/{task}/complete.json'
            if not marker.exists():
                continue
            run = Path(read(marker)['run'])
            _, rows = sample_rows(run, f'context_topk_{task}', context.COUNTS[task], context.METRICS[task])
            if any((a['input'], a['target']) != (b['input'], b['target']) for a, b in zip(rows, control)):
                raise ValueError('New/control native prompts/targets differ')
            scores = [[float(r[context.METRICS[task]]) for r in data] for data in (rows, control)]
            if not all(math.isfinite(v) for values in scores for v in values):
                raise ValueError('Nonfinite report score')
            records.append({'task': task, 'method': method, 'requests': len(rows), 'run': str(run),
                'score': sum(scores[0]) / len(rows), 'control_score': sum(scores[1]) / len(rows),
                'versus_question_control': context.clustered_difference(*scores,
                    [r['image_fingerprint'] for r in cohort[task]['rows']])})
    write_json(root / 'summary.json', {'completed_cells': len(records), 'total_cells': 6,
        'results': records, 'limits': protocol['limits']})
    lines = ['# Independent noise / OT experiments, fixed K128', '',
        '| Task | Arm | Score | Existing question control | Delta (pp) | Image-bootstrap 95% CI (pp) |',
        '|---|---|---:|---:|---:|---|']
    for r in records:
        diff = r['versus_question_control']
        lines.append(f"| {r['task']} | {r['method']} | {100*r['score']:.2f}% | {100*r['control_score']:.2f}% | {100*diff['delta']:+.2f} | [{100*diff['ci95'][0]:+.2f}, {100*diff['ci95'][1]:+.2f}] |")
    write_text(root / 'report.md', '\n'.join(lines + ['', *protocol['limits']]) + '\n')


def pipeline(root, gpu):
    protocol = validate(root)
    manifest = read(root / 'manifest.json')
    def stage(name):
        manifest.update(status='running', stage=name, gpu=gpu, updated_utc=context.native.now())
        write_json(root / 'manifest.json', manifest)
    def idle():
        while available_gpus([gpu]) != [gpu]:
            manifest.update(status='queued', stage='waiting_for_gpu', gpu=gpu, updated_utc=context.native.now())
            write_json(root / 'manifest.json', manifest)
            time.sleep(20)
    def evaluate(method, phase, task):
        parent = root / f'eval/{phase}/{method}/{task}'
        marker = parent / 'complete.json'
        if marker.exists():
            saved = read(marker)
            if saved['source'] != protocol['source'] or file_sha(saved['config']) != saved['config_sha256']:
                raise ValueError('Existing evaluation provenance changed')
            check(Path(saved['run']), method, task, phase)
            return
        idle()
        validate(root)
        stage(f'{phase}:{method}:{task}')
        run = parent / f'attempt_{time.time_ns()}'
        filename = root / 'configs' / f'{phase}_{method}_{task}_{time.time_ns()}.yaml'
        write_text(filename, yaml.safe_dump(config(root, protocol, method, phase, task, run), sort_keys=False))
        context.native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(filename)],
            parent / f'launch_{time.time_ns()}.log', [gpu], {
                'MLLM_TOPK_COHORT': str(Path(protocol['reference']) / 'cohort.json'),
                'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl'),
                'HF_HOME': str(root / 'cache/huggingface'), 'HF_DATASETS_CACHE': str(root / 'cache/datasets'),
                'XDG_CACHE_HOME': str(root / 'cache/xdg')})
        checks = check(run, method, task, phase)
        write_json(marker, {'source': protocol['source'], 'run': str(run), 'config': str(filename),
            'config_sha256': file_sha(filename), 'requests': checks['requests'], 'completed_utc': context.native.now()})
        if phase == 'full':
            manifest['completed_cells'] = sorted(set(manifest['completed_cells']) | {f'{method}/{task}'})
            report(root, protocol)
            write_json(root / 'manifest.json', manifest)
    try:
        # First validate OT on the actual 7B, not only a random tiny model.
        for task in TASKS:
            evaluate('ot', 'smoke', task)
        idle()
        stage('noise_real7b_restore_gates')
        noise = root / 'noise'
        if not (noise / 'gates_passed.json').exists():
            context.native.gates(noise, [gpu])
        idle()
        stage('noise_training_320_steps')
        marker = noise / 'training/manifest.json'
        if not marker.exists() or read(marker)['status'] != 'complete':
            context.native.execute(context.native.training_command(noise, noise / 'training', [gpu],
                resume=(noise / 'training/latest.json').exists()), noise / f'train_launch_{time.time_ns()}.log', [gpu])
        for task in TASKS:
            evaluate('vp_noise', 'smoke', task)
        write_json(root / 'smoke_complete.json', {'status': 'complete', 'requests': 48, 'time': context.native.now()})
        for method in METHODS:
            for task in TASKS:
                evaluate(method, 'full', task)
        stage('complete')
        manifest.update(status='complete', finished_utc=context.native.now())
        write_json(root / 'manifest.json', manifest)
        write_json(root / 'full_complete.json', {'status': 'complete', 'requests': 13176, 'time': context.native.now()})
    except BaseException as exc:
        manifest.update(status='failed', error=str(exc), updated_utc=context.native.now())
        write_json(root / 'manifest.json', manifest)
        write_json(root / f'failure_{time.time_ns()}.json', {'traceback': traceback.format_exc()})
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
        with (root / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(root, args.gpu)
