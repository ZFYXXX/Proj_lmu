"""Frozen-source 2x2: word/contextual interaction x task-set teacher supervision."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import asdict
import fcntl
import json
import math
import os
from pathlib import Path
from queue import Empty, Queue
import signal
import subprocess
import sys
import time
import traceback

import yaml

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha, validate_identity
from mllm_comp.analysis.common.learned_topk_experiment import (
    dependency_identity, gpu_free_mib, gpu_processes, owned_output, reject_overlap, shared_policy, source_identity,
)
from mllm_comp.analysis.common.topk_training_modes import parameter_digest
from mllm_comp.core.serialization import atomic_write_json as write_json, atomic_write_text as write_text
from scripts.train.qwen25_vl import run_input_adaptive_topk as history
from scripts.train.qwen25_vl import run_learned_visual_topk as native

BASE = history.BASE
REFERENCE = BASE / 'outputs/train/qwen25_vl/input_adaptive_topk_20261028_v2'
ARMS = {'word_teacher': ('word', 1.), 'context_ce': ('contextual', 0.), 'context_teacher': ('contextual', 1.)}
TASKS = history.COUNTS


def read(path):
    return json.loads(Path(path).read_text())


def validate(root):
    p = read(root / 'protocol.json')
    if p['source'] != source_identity(ROOT) or p['dependencies'] != dependency_identity():
        raise ValueError('Frozen source/dependencies changed')
    if read(root / 'manifest.json')['protocol_sha256'] != file_sha(root / 'protocol.json'):
        raise ValueError('Frozen experiment definition changed')
    for path, checksum in p['input_hashes'].items():
        if file_sha(path) != checksum:
            raise ValueError(f'Pinned input changed: {path}')
    validate_identity(read(p['identity_path']))
    return p


def prepare(root):
    import torch
    from mllm_comp.compressors.image.learned_visual_topk import load_selector as load_visual
    from mllm_comp.compressors.image.query_interaction_topk import QueryInteractionTopK, QueryInteractionSpec
    root.mkdir(parents=True, exist_ok=False)
    reference = read(REFERENCE / 'protocol.json')
    cohort = read(REFERENCE / 'cohort.json')
    entries = read(reference['training_manifest'])['entries']
    for task in TASKS:
        reject_overlap(entries, cohort[task]['rows'])
    validate_identity(read(reference['identity_path']))
    if dependency_identity() != reference['dependencies']:
        raise ValueError('Evaluator drift from existing controls')
    write_json(root / 'cohort.json', cohort)
    source = source_identity(ROOT)
    p = {**reference, 'source': source, 'dependencies': dependency_identity(), 'created_utc': native.now(),
         'selector_type': 'query_interaction_topk', 'resource_policy': shared_policy(),
         'teacher': {'count': 8, 'temperature': .25, 'target_count': 128,
                     'definition': 'same-budget candidate-set answer NLL soft retention targets, not causal labels or EvoComp reproduction'},
         'arms': ARMS, 'eval': {'counts': TASKS, 'no_development_pass': True},
         'limits': ['Single seed42; frozen original 2560 training images; fixed step320.',
                    'Teacher uses training answers offline; inference only sees prompt words and image.',
                    'K128, native dynamic resolution, original content/positions, no recycling or variable budget.',
                    'Candidate-pool task labels are proxies; shared-GPU timing is not a matched speed benchmark.']}
    p.pop('initial_parameter_sha256', None)
    p.pop('evidence_training_manifest', None)
    p['training'] = {k: v for k, v in p['training'].items() if k not in ('input_gate_width', 'initialization')}
    p['training']['initialization'] = 'identical old word trunk and frozen visual prior; contextual arm adds fixed-seed attention blocks'
    p['references'] = {}
    pins = {**reference['input_hashes'], str(root / 'cohort.json'): file_sha(root / 'cohort.json')}
    for task in TASKS:
        p['references'][task] = {}
        for method in ('dense', 'old_mlp', 'image_context', 'word_fixed'):
            cell = history.historical(method, task)
            if cell is None:
                marker = REFERENCE / f'eval/full/word_fixed/{task}/complete.json'
                cell = {**read(marker), 'marker': str(marker), 'task_name': 'projector_extra_mmstar'}
            sample, rows = history.sample_rows(Path(cell['run']), cell['task_name'], TASKS[task], history.METRICS[task])
            cell['samples'] = str(sample)
            for path in (cell['config'], cell['marker'], sample):
                pins[str(path)] = file_sha(path)
            p['references'][task][method] = cell
    p['input_hashes'] = pins
    write_json(root / 'protocol.json', p)
    write_json(root / 'manifest.json', {'status': 'prepared', 'protocol_sha256': file_sha(root / 'protocol.json'),
        'source': source, 'new_training_arms': list(ARMS), 'reuse_control': 'matched word CE/KL step320',
        'teacher_images': len(entries), 'new_full_eval_cells': len(ARMS)*len(TASKS), 'updated_utc': native.now()})


def teacher(root, gpu):
    import torch
    import torch.nn.functional as F
    from mllm_comp.analysis.common.anchor7b_runtime import load_runtime
    from mllm_comp.analysis.common.learned_hybrid_runtime import answer_logits, compact_state
    from mllm_comp.analysis.common.task_loss_supervision import candidate_sets, label_key, retention_target
    from mllm_comp.compressors.image.learned_visual_topk import load_selector
    from mllm_comp.compressors.image.word_visual_topk import prompt_word_question
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples, prepare_example
    p = validate(root)
    shard = root / f'teacher/shard{gpu}'
    shard.mkdir(parents=True, exist_ok=True)
    index = read(shard / 'index.json') if (shard / 'index.json').exists() else {'rows': {}}
    expected = {e['sample_id'] for e in read(p['training_manifest'])['entries'][gpu::2]}
    if set(index['rows']) == expected:
        for item in index['rows'].values():
            if file_sha(item['path']) != item['sha256']:
                raise ValueError('Completed teacher label changed')
        return
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    bundle = load_runtime(read(p['identity_path']), device)
    prior = load_selector(p['warmstart_checkpoint'], base_identity=read(p['identity_path'])['identity_sha256'], device=device)
    examples = load_examples(Path(p['training_manifest']))
    patterns = p['word_spec']['template_token_ids']
    started = time.time()
    for i in range(gpu, len(examples), 2):
        example = examples[i]
        item = index['rows'].get(example['sample_id'])
        if item:
            if file_sha(item['path']) != item['sha256']:
                raise ValueError('Saved teacher label changed')
            continue
        torch.cuda.empty_cache()
        if torch.cuda.mem_get_info(device)[0] < 8*1024**3:
            raise RuntimeError('Shared GPU headroom fell below 8 GiB; stopping teacher')
        state = prepare_example(bundle, example)
        question, audit = prompt_word_question(bundle['model'], state['prompt_inputs'], patterns)
        with torch.inference_mode():
            logits, targets = answer_logits(bundle, state['hidden'], state['positions'], state['teacher_inputs']['labels'])
            if not index['rows']:
                from mllm_comp.analysis.common.visual_token_phenomena import answer_predictor_positions
                predictors, _ = answer_predictor_positions(state['teacher_inputs']['labels'])
                native_inputs = {key: value for key, value in state['teacher_inputs'].items() if key != 'labels'}
                native_logits = bundle['model'](**native_inputs, use_cache=False, logits_to_keep=torch.tensor(predictors, device=device)).logits[0].float()
                torch.testing.assert_close(logits, native_logits, rtol=1e-3, atol=1e-3)
                del native_logits
            dense_loss = float(F.cross_entropy(logits, targets))
            if not math.isfinite(dense_loss):
                raise ValueError('Nonfinite dense teacher loss')
            candidates = candidate_sets(prior(state['features'])['scores'], state['height'], state['width'],
                count=p['teacher']['count'], seed=42+int(label_key(example['sample_id'])[:8], 16))
            losses = []
            for members in candidates:
                indices = torch.tensor(members, device=device)
                compact = compact_state(state, {'indices': members, 'tokens': state['features'].index_select(0, indices)})
                actual, answer = answer_logits(bundle, compact['hidden'], compact['positions'], compact['labels'])
                if not torch.equal(answer, targets):
                    raise ValueError('Teacher compaction changed answer target')
                loss = float(F.cross_entropy(actual, answer))
                if not math.isfinite(loss):
                    raise ValueError('Nonfinite task teacher loss')
                losses.append(loss)
                del actual, compact
            target = retention_target(candidates, losses, len(state['visual']), temperature=p['teacher']['temperature'])
            if abs(float(target.sum())-128) > 1e-3:
                raise ValueError('Teacher retention mass changed')
        path = shard / f'{label_key(example["sample_id"])}.json'
        row = {'sample_id': example['sample_id'], 'image_fingerprint': example['image_fingerprint'],
               'question_ids_sha256': audit['question_ids_sha256'], 'source_count': len(state['visual']),
               'height': state['height'], 'width': state['width'], 'candidates': candidates,
               'losses': losses, 'dense_answer_nll': dense_loss, 'best_improvement_over_prior': losses[0]-min(losses),
               'protocol_sha256': file_sha(root / 'protocol.json')}
        write_json(path, row)
        index['rows'][example['sample_id']] = {'path': str(path), 'sha256': file_sha(path)}
        write_json(shard / 'index.json', index)
        write_json(shard / 'status.json', {'status': 'running', 'complete': len(index['rows']),
            'total': len(examples[gpu::2]), 'pid': os.getpid(), 'updated_utc': native.now(), 'elapsed_s': time.time()-started})
        print(json.dumps({'teacher_sample': i, 'complete': len(index['rows']), 'nll_improvement': row['best_improvement_over_prior']}), flush=True)
        del state, question, logits, targets, target
    write_json(shard / 'status.json', {'status': 'complete', 'complete': len(index['rows']), 'updated_utc': native.now()})


def prepare_arm(root, p, arm):
    from mllm_comp.compressors.image.learned_visual_topk import load_selector as load_visual
    from mllm_comp.compressors.image.query_interaction_topk import QueryInteractionSpec, QueryInteractionTopK
    folder = root / 'arms' / arm
    if (folder / 'protocol.json').exists():
        return folder
    folder.mkdir(parents=True, exist_ok=True)
    interaction, weight = ARMS[arm]
    spec_data = {k: v for k, v in p['word_spec'].items() if k != 'gate_width'}
    spec = QueryInteractionSpec(**spec_data, interaction=interaction)
    visual = load_visual(p['warmstart_checkpoint'], base_identity=read(p['identity_path'])['identity_sha256'], device='cpu')
    module = QueryInteractionTopK(spec, visual=visual)
    index = root / 'teacher/index.json'
    protocol = {**p, 'word_spec': asdict(spec), 'initial_parameter_sha256': parameter_digest(module),
        'task_teacher_index': str(index) if weight else None,
        'training': {**p['training'], 'evidence_weight': 0., 'task_teacher_weight': weight},
        'input_hashes': {**p['input_hashes'], **({str(index): file_sha(index)} if weight else {})}}
    write_json(folder / 'protocol.json', protocol)
    write_json(folder / 'manifest.json', {'status': 'prepared', 'protocol_sha256': file_sha(folder / 'protocol.json'),
        'trainable_parameters': sum(x.numel() for x in module.parameters() if x.requires_grad), 'interaction': interaction,
        'task_teacher_weight': weight, 'updated_utc': native.now()})
    return folder


def evaluate(root, p, arm, gpu):
    folder = root / 'arms' / arm
    ap = read(folder / 'protocol.json')
    checkpoint = native.latest(folder / 'training')
    for task in TASKS:
        for phase in ('smoke', 'full'):
            directory = folder / f'eval/{phase}/{task}'
            marker = directory / 'complete.json'
            if marker.exists():
                saved = read(marker)
                if saved['config_sha256'] != file_sha(saved['config']):
                    raise ValueError('Completed evaluation config changed')
                history.check(Path(saved['run']), arm, task, phase)
                continue
            run = directory / f'attempt_{time.time_ns()}'
            cfg = history.config(root, p, 'word_fixed', phase, task, run)
            stage, = cfg['pipeline']['stages']
            name = 'query_interaction_topk'
            stage.update(name=arm, plugin=name, algorithm=name, variant=f'{name}.image', binding=f'{name}.image.qwen25_vl')
            stage['config'].update(checkpoint_path=str(checkpoint), interaction=ap['word_spec']['interaction'], heads=4)
            config = folder / f'configs/{phase}_{task}_{time.time_ns()}.yaml'
            config.parent.mkdir(parents=True, exist_ok=True)
            write_text(config, yaml.safe_dump(cfg, sort_keys=False))
            while gpu_free_mib().get(gpu, 0) < 46000:
                time.sleep(20)
            native.execute([sys.executable, '-m', 'mllm_comp.cli.main', 'eval', '--config', str(config)],
                directory / f'launch_{time.time_ns()}.log', [gpu], {'MLLM_TOPK_PERFORMANCE': str(run / 'request_performance.jsonl')})
            checked = history.check(run, arm, task, phase)
            write_json(marker, {**checked, 'run': str(run), 'config': str(config), 'config_sha256': file_sha(config),
                'checkpoint': str(checkpoint), 'source': p['source'], 'finished_utc': native.now()})


def report(root, p):
    from configs.eval.lmms_eval.tasks.projector_extra.extra_utils import macro_score
    cohort = read(root / 'cohort.json')
    scores, predictions = {}, {}
    for task in TASKS:
        scores[task] = {}
        for method, cell in p['references'][task].items():
            _, rows = history.sample_rows(Path(cell['run']), cell['task_name'], TASKS[task], history.METRICS[task])
            alias = 'word_ce' if method == 'word_fixed' else method
            predictions[task, alias] = rows
            scores[task][alias] = macro_score([r['average'] for r in rows]) if task == 'mmstar' else sum(float(r[history.METRICS[task]]) for r in rows)/len(rows)
        for arm in ARMS:
            marker = root / f'arms/{arm}/eval/full/{task}/complete.json'
            if marker.exists():
                cell = read(marker)
                _, rows = history.sample_rows(Path(cell['run']), cell['task_name'], TASKS[task], history.METRICS[task])
                predictions[task, arm] = rows
                scores[task][arm] = macro_score([r['average'] for r in rows]) if task == 'mmstar' else cell['score']
    comparisons = {}
    for task in TASKS:
        for a, b in [('context_ce', 'word_ce'), ('word_teacher', 'word_ce'), ('context_teacher', 'context_ce'),
                     ('context_teacher', 'word_teacher'), ('context_teacher', 'old_mlp'), ('context_teacher', 'image_context')]:
            if (task, a) not in predictions or (task, b) not in predictions:
                continue
            ra, rb = predictions[task, a], predictions[task, b]
            if any((x['input'], x['target']) != (y['input'], y['target']) for x, y in zip(ra, rb)):
                raise ValueError('Paired benchmark prompt/target mismatch')
            ids = [r['image_fingerprint'] for r in cohort[task]['rows']]
            comparisons[f'{task}:{a}-{b}'] = history.paired_mmstar(ra, rb, ids) if task == 'mmstar' else history.context.clustered_difference(
                [float(r[history.METRICS[task]]) for r in ra], [float(r[history.METRICS[task]]) for r in rb], ids)
        if all((task, arm) in predictions for arm in ('context_teacher', 'context_ce', 'word_teacher', 'word_ce')):
            ct, cc, wt, wc = [predictions[task, arm] for arm in ('context_teacher', 'context_ce', 'word_teacher', 'word_ce')]
            ids = [r['image_fingerprint'] for r in cohort[task]['rows']]
            if task == 'mmstar':
                a = [{'average': {**x['average'], 'score': float(x['average']['score'])-float(y['average']['score'])}} for x,y in zip(ct,cc)]
                b = [{'average': {**x['average'], 'score': float(x['average']['score'])-float(y['average']['score'])}} for x,y in zip(wt,wc)]
                comparison = history.paired_mmstar(a, b, ids)
            else:
                metric = history.METRICS[task]
                comparison = history.context.clustered_difference([float(x[metric])-float(y[metric]) for x,y in zip(ct,cc)],
                    [float(x[metric])-float(y[metric]) for x,y in zip(wt,wc)], ids)
            comparisons[f'{task}:structure_x_supervision'] = comparison
    methods = ('dense', 'old_mlp', 'image_context', 'word_ce', *ARMS)
    write_json(root / 'summary.json', {'scores': scores, 'paired_comparisons': comparisons, 'limits': p['limits']})
    lines = ['# Query interaction × task-set supervision', '', '| Dataset | '+' | '.join(methods)+' |', '|---|'+'---:|'*len(methods)]
    for task in TASKS:
        lines.append('| '+task+' | '+' | '.join(f'{100*scores[task][m]:.2f}%' if m in scores[task] else 'pending' for m in methods)+' |')
    write_text(root / 'report.md', '\n'.join(lines+['', 'MMStar: native category macro. AI2D: corrected leading-option accuracy.', *p['limits']])+'\n')


def teacher_assignment(gpus):
    """Two immutable data shards; execution may use one or two physical GPUs."""
    if not gpus or len(set(gpus)) != len(gpus) or set(gpus)-{0, 1}:
        raise ValueError('Expected distinct physical GPU0/GPU1')
    return [(shard, gpus[shard % len(gpus)]) for shard in (0, 1)]


def pipeline(root, gpus=(0, 1)):
    assignments = teacher_assignment(gpus)
    p = validate(root)
    manifest = read(root / 'manifest.json')
    def status(state, **more):
        manifest.update(status=state, pid=os.getpid(), updated_utc=native.now(), **more)
        write_json(root / 'manifest.json', manifest)
    with ExitStack() as stack:
        try:
            for gpu in gpus:
                lockpath = BASE / f'outputs/runtime/qwen25_gpu_locks/gpu{gpu}.lock'
                lockpath.parent.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context(lockpath.open('a'))
                while True:
                    try:
                        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        status('queued_gpu_reservation', waiting_gpu=gpu,
                               prerequisite=str(BASE / 'outputs/operations/qwen25_recovery_20261028_v1/status.json'))
                        time.sleep(20)
            while any(x['uid'] == os.getuid() for x in gpu_processes() if x['gpu'] in gpus) or any(gpu_free_mib().get(g, 0)<46000 for g in gpus):
                status('waiting_for_resources'); time.sleep(20)
            status('real_7b_preflight', gpus=list(gpus))
            first_arm = prepare_arm(root, p, 'context_ce')
            if not (first_arm/'gates_passed.json').exists():
                native.gates(first_arm, [gpus[0]])
            def launch(command, log, gpu):
                native.execute(command, log, [gpu])
            status('teacher_labels')
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(launch, [sys.executable, str(Path(__file__).resolve()), 'teacher', '--root', str(root), '--gpu', str(shard)],
                                       root / f'teacher_shard{shard}_gpu{gpu}_{time.time_ns()}.log', gpu) for shard, gpu in assignments]
                for f in futures:
                    f.result()
            rows = {}
            for gpu in (0, 1):
                part = read(root / f'teacher/shard{gpu}/index.json')['rows']
                if set(rows)&set(part):
                    raise ValueError('Duplicate teacher image')
                rows.update(part)
            if set(rows) != {e['sample_id'] for e in read(p['training_manifest'])['entries']}:
                raise ValueError('Missing teacher images')
            index = root / 'teacher/index.json'
            write_json(index, {'rows': rows, 'temperature': p['teacher']['temperature'], 'protocol_sha256': file_sha(root/'protocol.json')})
            quality = [read(item['path']) for item in rows.values()]
            write_json(root / 'teacher/quality.json', {'images': len(quality),
                'fraction_best_better_than_prior': sum(r['best_improvement_over_prior']>1e-6 for r in quality)/len(quality),
                'mean_best_nll_improvement': sum(r['best_improvement_over_prior'] for r in quality)/len(quality),
                'labels_are_causal_ground_truth': False})
            status('training_and_evaluation')
            queue = Queue()
            for arm in ARMS:
                prepare_arm(root, p, arm); queue.put(arm)
            def worker(gpu):
                while True:
                    try:
                        arm = queue.get_nowait()
                    except Empty:
                        return
                    folder = root / 'arms' / arm
                    if not (folder/'gates_passed.json').exists():
                        native.gates(folder, [gpu])
                    if not (folder/'training/manifest.json').exists() or read(folder/'training/manifest.json')['status'] != 'complete':
                        if (folder/'training').exists() and not (folder/'training/latest.json').exists():
                            (folder/'training').rename(folder/f'training_failed_before_checkpoint_{time.time_ns()}')
                        native.execute(native.training_command(folder, folder/'training', [gpu], resume=(folder/'training/latest.json').exists()),
                            folder/f'train_{time.time_ns()}.log', [gpu])
                    evaluate(root, p, arm, gpu)
            with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
                futures = [pool.submit(worker, gpu) for gpu in gpus]
                for f in futures:
                    f.result()
            report(root, p)
            status('complete', finished_utc=native.now())
        except BaseException as error:
            status('failed', error=str(error), traceback=traceback.format_exc())
            raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=('prepare', 'pipeline', 'teacher', 'report'))
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1))
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    args = parser.parse_args()
    root = owned_output(args.root)
    if args.command == 'prepare':
        prepare(root)
    elif args.command == 'pipeline':
        from scripts.eval.lmms_eval.supervise_qwen25_recovery import retryable
        while True:
            started = time.time()
            try:
                pipeline(root, args.gpus)
                break
            except Exception as error:
                state = read(root/'manifest.json')
                state.update(status='failed', error=str(error), traceback=traceback.format_exc(), pid=os.getpid(), updated_utc=native.now())
                write_json(root/'manifest.json', state)
                logs = [p for p in root.rglob('*.log') if p.stat().st_mtime >= started-1]
                if not any(retryable(p) for p in logs):
                    raise
                state.update(status='resource_retry', updated_utc=native.now())
                write_json(root/'manifest.json', state)
                time.sleep(30)
    elif args.command == 'teacher':
        teacher(root, args.gpu)
    else:
        report(root, read(root/'protocol.json'))


if __name__ == '__main__':
    main()
