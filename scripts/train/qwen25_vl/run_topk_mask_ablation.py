"""Frozen matched soft/STE arms; native MLLM-Comp hard-deployment evaluation.

No automatic commits/tags. Resume reuses only matching hash-pinned artifacts.
"""
import argparse
from collections import defaultdict
import fcntl
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity, digest_json, owned_output, shared_policy, source_identity, validate_training_entries,
)
from mllm_comp.analysis.common.topk_training_modes import MODES, parameter_digest
from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
from scripts.train.qwen25_vl import run_learned_visual_topk as native


def read(path):
    return json.loads(Path(path).read_text())


def matched_settings(protocols):
    settings = []
    for mode in MODES:
        current = dict(protocols[mode]['training'])
        if current.pop('training_mask_mode') != mode:
            raise ValueError('Arm name and training mask disagree')
        settings.append(current)
    if settings[0] != settings[1]:
        raise ValueError('Non-mask training settings differ between arms')
    for key in ('source', 'dependencies', 'identity_path', 'training_manifest', 'initial_parameter_sha256', 'eval', 'resource_policy'):
        if protocols['soft'][key] != protocols['ste'][key]:
            raise ValueError(f'Matched-arm contract differs: {key}')
    return digest_json(settings[0])


def prepare(root, reference_train, reference_eval):
    from mllm_comp.compressors.image.learned_visual_topk import LearnedVisualTopK, LearnedVisualTopKSpec
    if root.exists():
        raise ValueError('Preserve existing experiments; choose a new root')
    reference_path = reference_train / 'protocol.json'
    reference = read(reference_path)
    if dependency_identity() != reference['dependencies']:
        raise ValueError('Software changed relative to the reference recipe')
    for filename, checksum in reference['input_hashes'].items():
        if file_sha(filename) != checksum:
            raise ValueError(f'Reference data changed: {filename}')
    validate_training_entries(read(reference['training_manifest'])['entries'])
    identity = read(reference['identity_path'])
    validate_identity(identity)
    model_config = read(Path(identity['checkpoint']) / 'config.json')
    dimension = model_config.get('text_config', model_config)['hidden_size']
    initial = parameter_digest(LearnedVisualTopK(LearnedVisualTopKSpec(dimension)))
    source = source_identity(ROOT)
    root.mkdir(parents=True)
    atomic_write_json(root / 'cohort.json', read(reference_train / 'cohort.json'))
    protocols, pins = {}, {}
    for mode in MODES:
        arm = root / 'arms' / mode
        arm.mkdir(parents=True)
        atomic_write_json(arm / 'cohort.json', read(root / 'cohort.json'))
        protocol = dict(reference)
        protocol.pop('trained_run_root', None)
        protocol.pop('training_reference', None)
        protocol.update(created_utc=native.now(), source=source, resource_policy=shared_policy(),
            initial_parameter_sha256=initial, methods=['trained_selector'],
            claims='Single-seed matched soft versus hard-forward/sigmoid-backward; NOT Top-K versus no Top-K',
            training={**reference['training'], 'training_mask_mode': mode})
        protocol['input_hashes'] = {**reference['input_hashes'],
            str(reference_path): file_sha(reference_path), str(arm / 'cohort.json'): file_sha(arm / 'cohort.json')}
        atomic_write_json(arm / 'protocol.json', protocol)
        protocols[mode] = protocol
        pins[mode] = file_sha(arm / 'protocol.json')
    pair = {'created_utc': native.now(), 'source': source, 'arm_protocol_sha256': pins,
        'matched_training_sha256': matched_settings(protocols), 'initial_parameter_sha256': initial,
        'reference_train': str(reference_train), 'reference_eval': str(reference_eval),
        'reference_protocol_sha256': file_sha(reference_path), 'cohort_sha256': file_sha(root / 'cohort.json'),
        'schedule': 'Both arms freshly initialized; 320 steps/2560 images each; sequential single-GPU execution',
        'primary_contrast': 'soft minus STE hard-deployment native task score, paired by image',
        'backward': 'Both use the SAME continuous sigmoid-threshold Jacobian; only forward gate changes',
        'common_loss': 'answer CE + 0.5 teacher KL + identical annealed BCE on continuous weights',
        'limits': ['STE forward keeps N positions and zeroes embeddings, not physical token deletion',
            'Both deploy identical source-order hard min(128,N) original tokens',
            'Comparison identifies training forward-mask effect, not all possible Top-K estimator effects',
            'Existing development sets were previously inspected; no tuning/checkpoint selection',
            'Single seed gives image-sampling uncertainty, not training-seed uncertainty']}
    atomic_write_json(root / 'protocol.json', pair)
    atomic_write_json(root / 'manifest.json', {'status': 'prepared', 'created_utc': native.now()})
    print(json.dumps({'status': 'prepared', 'root': str(root), 'initial_parameter_sha256': initial}), flush=True)


def validate(root):
    pair = read(root / 'protocol.json')
    if pair['source'] != source_identity(ROOT) or pair['cohort_sha256'] != file_sha(root / 'cohort.json'):
        raise ValueError('Frozen source or cohort changed')
    protocols = {}
    for mode in MODES:
        arm = root / 'arms' / mode
        if pair['arm_protocol_sha256'][mode] != file_sha(arm / 'protocol.json'):
            raise ValueError('Frozen arm protocol changed')
        protocols[mode] = native.validate(arm)
    if matched_settings(protocols) != pair['matched_training_sha256']:
        raise ValueError('Matched settings changed')
    return pair, protocols


def verify_training_pair(root, pair):
    records = {}
    orders = []
    for mode in MODES:
        arm = root / 'arms' / mode
        record = read(arm / 'training/manifest.json')
        if record['status'] != 'complete' or record['steps_completed'] != 320:
            raise ValueError('Both arms must complete fixed step320')
        if record['initial_parameter_sha256'] != pair['initial_parameter_sha256']:
            raise ValueError('Arm initialization drifted')
        if record['training_mask_mode'] != mode:
            raise ValueError('Wrong training mode recorded')
        steps = {}
        for path in sorted((arm / 'training').glob('training_from_step_*.jsonl')):
            for line in path.read_text().splitlines():
                item = json.loads(line)
                # Retried uncheckpointed steps must retain exactly the same order.
                if item['step'] in steps and steps[item['step']]['global_sample_ids'] != item['global_sample_ids']:
                    raise ValueError('Data order changed in a retry')
                steps[item['step']] = item
        if set(steps) != set(range(1, 321)):
            raise ValueError('Incomplete step logs')
        orders.append([steps[i]['global_sample_ids'] for i in range(1, 321)])
        records[mode] = {'manifest': record, 'checkpoint': str(native.latest(arm / 'training')),
            'weights_sha256': file_sha(native.latest(arm / 'training') / 'selector.safetensors'),
            'final_step': steps[320]}
    if orders[0] != orders[1] or sum(map(len, orders[0])) != 2560:
        raise ValueError('Matched arms did not see the same 2560 samples in order')
    atomic_write_json(root / 'training_pair_checks.json', {
        'status': 'passed', 'identical_initialization': True, 'identical_data_order': True,
        'data_order_sha256': digest_json(orders[0]), 'arms': records})


def load_samples(arm, phase, method, task):
    pointer = read(arm / 'eval' / phase / method / 'complete.json')
    run = Path(pointer['run'])
    if file_sha(pointer['config']) != pointer['config_sha256']:
        raise ValueError('Evaluation config changed')
    native.check_eval(run, method, phase)
    name = f'learned_topk_{task}' + ('_dev' if phase == 'dev' else '')
    path, = run.glob(f'**/*_samples_{name}.jsonl')
    rows = {int(row['doc_id']): row for line in path.read_text().splitlines() if (row := json.loads(line))}
    checks = read(run / 'protocol_checks.json')
    counts = {c['doc_id']: c['source'] for c in checks['token_counts'] if c['task'] == name}
    return rows, counts


def paired_scores(a, b, cohort, phase, task):
    from scripts.eval.lmms_eval.report_learned_visual_topk import paired_image_bootstrap
    if set(a) != set(b):
        raise ValueError('Paired sample IDs differ')
    key = 'relaxed_overall' if task == 'chartqa' else 'ocrbench_accuracy'
    groups = defaultdict(list)
    wins = losses = ties = 0
    for i in sorted(a):
        if any(a[i][k] != b[i][k] for k in ('input', 'target')):
            raise ValueError('Paired prompts/targets differ')
        original = cohort['development_indices'][i] if phase == 'dev' else i
        image = cohort['rows'][original]['image_fingerprint']
        diff = float(a[i][key]) - float(b[i][key])
        groups[image].append(diff)
        wins += int(diff > 0); losses += int(diff < 0); ties += int(diff == 0)
    result = paired_image_bootstrap([sum(v) / len(v) for v in groups.values()])
    result.update(question_weighted_delta=sum(float(a[i][key]) - float(b[i][key]) for i in a) / len(a),
        question_wins=wins, question_losses=losses, question_ties=ties)
    return result


def report(root, pair, phase):
    cohort = read(root / 'cohort.json')
    result = {'phase': phase, 'status': 'complete', 'metrics': {}, 'comparisons': {}, 'limits': pair['limits']}
    for task in native.DATA:
        key = 'relaxed_overall' if task == 'chartqa' else 'ocrbench_accuracy'
        soft, nsoft = load_samples(root / 'arms/soft', phase, 'trained_selector', task)
        ste, nste = load_samples(root / 'arms/ste', phase, 'trained_selector', task)
        initial, ninitial = load_samples(root / 'arms/soft', phase, 'initial_selector', task)
        if nsoft != nste or nsoft != ninitial:
            raise ValueError('Native source token counts differ between arms')
        result['metrics'][task] = {name: sum(float(v[key]) for v in rows.values()) / len(rows)
            for name, rows in [('soft', soft), ('ste', ste), ('initial', initial)]}
        for name, a, b in [('soft_minus_ste', soft, ste), ('soft_minus_initial', soft, initial), ('ste_minus_initial', ste, initial)]:
            result['comparisons'][f'{task}:{name}'] = paired_scores(a, b, cohort[task], phase, task)
    atomic_write_json(root / f'summary_{phase}.json', result)
    lines = [f'# Matched Top-K mask ablation ({phase}, fixed step320)', '',
        '| Task | Initial | Soft | STE | Soft−STE (pp) | Image-bootstrap 95% CI (pp) |',
        '|---|---:|---:|---:|---:|---|']
    for task, scores in result['metrics'].items():
        delta = result['comparisons'][f'{task}:soft_minus_ste']
        lo, hi = delta['ci95']
        lines.append(f'| {task} | {100*scores["initial"]:.2f}% | {100*scores["soft"]:.2f}% | '
            f'{100*scores["ste"]:.2f}% | {100*delta["question_weighted_delta"]:+.2f} | [{100*lo:+.2f}, {100*hi:+.2f}] |')
    lines += ['', 'Headline differences weight questions; CIs weight independent image means (ChartQA has repeated images).',
        'OCRBench percentages here are normalized native scores, not the full /1000 sum for development subsets.', '', *pair['limits']]
    atomic_write_text(root / f'report_{phase}.md', '\n'.join(lines) + '\n')


def pipeline(root, gpu):
    pair, protocols = validate(root)
    stage = 'validate'
    try:
        for mode in MODES:
            arm = root / 'arms' / mode
            stage = f'gates:{mode}'
            if not (arm / 'gates_passed.json').exists():
                atomic_write_json(root / 'manifest.json', {'status': 'running', 'stage': stage, 'gpu': gpu, 'updated_utc': native.now()})
                native.gates(arm, [gpu])
        for mode in MODES:
            arm = root / 'arms' / mode
            stage = f'train:{mode}'
            atomic_write_json(root / 'manifest.json', {'status': 'running', 'stage': stage, 'gpu': gpu, 'updated_utc': native.now()})
            marker = arm / 'training/manifest.json'
            if marker.exists() and read(marker)['status'] == 'complete':
                continue
            native.execute(native.training_command(arm, arm / 'training', [gpu], resume=(arm / 'training/latest.json').exists()),
                arm / f'train_launch_{time.time_ns()}.log', [gpu])
        verify_training_pair(root, pair)
        for phase in ('smoke', 'dev', 'full'):
            for mode in MODES:
                arm = root / 'arms' / mode
                stage = f'eval:{phase}:{mode}'
                atomic_write_json(root / 'manifest.json', {'status': 'running', 'stage': stage, 'gpu': gpu, 'updated_utc': native.now()})
                native.evaluate(arm, protocols[mode], [gpu], phase,
                    methods=('initial_selector', 'trained_selector') if mode == 'soft' else ('trained_selector',))
            if phase != 'smoke':
                report(root, pair, phase)
        atomic_write_json(root / 'manifest.json', {'status': 'complete', 'updated_utc': native.now(), 'gpu': gpu})
    except BaseException as exc:
        atomic_write_json(root / 'manifest.json', {'status': 'failed', 'stage': stage, 'error': str(exc), 'updated_utc': native.now()})
        atomic_write_json(root / f'failure_{time.time_ns()}.json', {'stage': stage, 'error': str(exc), 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['prepare', 'pipeline'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--reference-train', type=Path)
    parser.add_argument('--reference-eval', type=Path)
    parser.add_argument('--gpu', type=int, choices=[0, 1], default=0)
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == 'prepare':
        if not args.reference_train or not args.reference_eval:
            parser.error('prepare requires both reference roots')
        prepare(root, args.reference_train.resolve(), args.reference_eval.resolve())
    else:
        with (root / 'runner.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            pipeline(root, args.gpu)
