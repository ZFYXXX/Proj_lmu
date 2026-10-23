"""Pinned 7B dense versus trained anchor MLP on local OCRBench and DocVQA.

This runner has its own frozen protocol and leaves earlier 7B results untouched.
OCRBench uses the upstream 1000-point scoring rule; DocVQA uses upstream ANLS.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from mllm_comp.analysis.common.anchor7b_runtime import (
    identify_checkpoint, validate_identity, file_sha, load_runtime, load_adapter,
    anchor_base, compact_anchor, explicit_generate,
)
from mllm_comp.analysis.common.compression_training_check import evaluate_state
from mllm_comp.analysis.common.condition_ledger import append_json, records
from mllm_comp.analysis.common.evidence_interaction import digest
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.core.serialization import atomic_write_json
from scripts.eval.lmms_eval import run_anchor7b_full as previous
from scripts.train.qwen25_vl.train_learned_hybrid import prepare_example

DATA_ROOT = Path('/home/YinPeiran/datasets/lmms_eval_datasets')
DATASETS = {
    'ocrbench': (1000, 'OCRBench/data/test-*.parquet', 128),
    'docvqa': (5349, 'DocVQA/DocVQA/validation-*.parquet', 32),
}
METHODS = ('dense', 'feature_anchor_adapter')


def code_hash():
    import inspect
    from lmms_eval.tasks.ocrbench import utils as ocr
    from lmms_eval.tasks.docvqa import utils as doc
    from lmms_eval.api import metrics
    h = hashlib.sha256(previous.code_hash().encode())
    for path in (Path(__file__), Path(inspect.getsourcefile(ocr)), Path(inspect.getsourcefile(doc)),
                 Path(inspect.getsourcefile(metrics))):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def prompt_target(dataset, doc):
    from lmms_eval.tasks.ocrbench import utils as ocr
    from lmms_eval.tasks.docvqa import utils as dvqa
    if dataset == 'ocrbench':
        prompt = ocr.ocrbench_doc_to_text(doc, {'pre_prompt': '', 'post_prompt': ''})
        answers = doc['answer']
    elif dataset == 'docvqa':
        prompt = dvqa.docvqa_doc_to_text(doc, {'pre_prompt': '',
            'post_prompt': '\nAnswer the question using a single word or phrase.'})
        answers = doc['answers']
    else:
        raise ValueError(dataset)
    if not isinstance(answers, list):
        answers = [answers]
    answers = [str(a).strip() for a in answers if str(a).strip()]
    if not answers or not prompt.strip():
        raise ValueError('Question or answer list is empty')
    return prompt, answers[0], answers


def score_prediction(dataset, prediction, doc):
    if dataset == 'ocrbench':
        from lmms_eval.tasks.ocrbench.utils import ocrbench_process_results
        native = ocrbench_process_results(doc, [prediction])['ocrbench_accuracy']
        return float(native['score']), native
    if dataset == 'docvqa':
        from lmms_eval.api.metrics import anls
        native = anls(doc['answers'], [prediction])
        return float(native['anls']), native
    raise ValueError(dataset)


def prepare(root, source):
    import pyarrow.parquet as pq
    from PIL import Image
    if not (source/'model/checkpoints/step_000320/projector.safetensors').exists():
        raise ValueError('Missing completed 7B MLP checkpoint')
    training = json.loads((source/'training_samples.json').read_text())
    train_hashes = {e['image_fingerprint'] for e in training['entries']}
    identity = identify_checkpoint()
    datasets = {}
    for dataset, (expected, pattern, _) in DATASETS.items():
        folder = root/dataset
        folder.mkdir(parents=True, exist_ok=True)
        files = sorted(DATA_ROOT.glob(pattern))
        if not files:
            raise ValueError(f'No local data: {dataset}')
        entries, first = [], {}
        for path in files:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16):
                for doc in batch.to_pylist():
                    image = doc['image']
                    with Image.open(BytesIO(image['bytes']) if image.get('bytes') else image['path']) as loaded:
                        rgb = loaded.convert('RGB')
                        fingerprint = media_fingerprint([rgb])
                        rgb.close()
                    first.setdefault(fingerprint, len(entries))
                    entries.append({'sample_id': f'{dataset}:{len(entries)}', 'doc_id': len(entries),
                        'path': str(path), 'row': index, 'image_fingerprint': fingerprint,
                        'shard': int(digest(fingerprint)[:8], 16) % 2,
                        'training_image_overlap': fingerprint in train_hashes})
                    index += 1
        if len(entries) != expected:
            raise ValueError(f'{dataset}: expected {expected}, found {len(entries)}')
        smoke = set(sorted(first, key=lambda fp: digest(dataset+':7b-ocr-doc-smoke:'+fp))[:16])
        for entry in entries:
            entry['smoke'] = (entry['image_fingerprint'] in smoke and
                              entry['doc_id'] == first[entry['image_fingerprint']])
        atomic_write_json(folder/'samples.json', {'entries': entries,
            'sampling': 'Entire OCRBench test / DocVQA validation; no outcome filtering'})
        datasets[dataset] = {'n_questions': len(entries), 'n_images': len(first),
            'training_overlap_questions': sum(e['training_image_overlap'] for e in entries),
            'training_overlap_images': len(set(first) & train_hashes),
            'samples_sha256': file_sha(folder/'samples.json'),
            'data_files': {str(p): file_sha(p) for p in files}}
        print(json.dumps({'prepared': dataset, 'n_questions': len(entries), 'n_images': len(first),
                          'training_overlap_questions': datasets[dataset]['training_overlap_questions']}), flush=True)
    atomic_write_json(root/'model_identity.json', identity)
    adapter = source/'model/checkpoints/step_000320'
    protocol = {'checkpoint': identity['checkpoint'], 'revision': identity['revision'],
        'base_identity_sha256': identity['identity_sha256'], 'identity_file_sha256': file_sha(root/'model_identity.json'),
        'source_training_run': str(source), 'training_manifest_sha256': file_sha(source/'training_samples.json'),
        'adapter_path': str(adapter), 'adapter_sha256': file_sha(adapter/'projector.safetensors'),
        'adapter_config_sha256': file_sha(adapter/'projector_config.json'),
        'implementation_hash': code_hash(), 'methods': METHODS, 'budget': 128, 'training': False,
        'dtype': 'bfloat16', 'attention': 'eager', 'seed': 42,
        'processor': {'use_fast': False, 'min_pixels': 200704, 'max_pixels': 1605632},
        'generation': {'do_sample': False, 'repetition_penalty': 1.,
                       'ocrbench_max_new': 128, 'docvqa_max_new': 32},
        'scoring': {'ocrbench': 'upstream ocrbench_process_results; score out of 1000',
                    'docvqa': 'upstream lmms_eval.api.metrics.anls; validation split'},
        'datasets': datasets,
        'comparison_scope': 'Same 7B checkpoint, same questions, dense versus trained MLP K128',
        'timing_caveat': 'Includes diagnostic teacher NLL and cached generation; not an optimized deployment speed measurement',
        'created_utc': datetime.now(timezone.utc).isoformat()}
    atomic_write_json(root/'protocol.json', protocol)
    return protocol


def validate(root):
    protocol = json.loads((root/'protocol.json').read_text())
    if protocol['implementation_hash'] != code_hash():
        raise ValueError('Frozen evaluation implementation changed')
    if file_sha(root/'model_identity.json') != protocol['identity_file_sha256']:
        raise ValueError('Base model identity file changed')
    validate_identity(json.loads((root/'model_identity.json').read_text()))
    adapter = Path(protocol['adapter_path'])
    if (file_sha(adapter/'projector.safetensors') != protocol['adapter_sha256'] or
            file_sha(adapter/'projector_config.json') != protocol['adapter_config_sha256']):
        raise ValueError('Frozen adapter changed')
    if file_sha(Path(protocol['source_training_run'])/'training_samples.json') != protocol['training_manifest_sha256']:
        raise ValueError('Frozen training manifest changed')
    for dataset, spec in protocol['datasets'].items():
        if file_sha(root/dataset/'samples.json') != spec['samples_sha256']:
            raise ValueError('Frozen dataset manifest changed')
    return protocol


def worker(root, dataset, stage, shard, device):
    import pyarrow.parquet as pq
    import torch
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    protocol = validate(root)
    identity = json.loads((root/'model_identity.json').read_text())
    entries = [e for e in json.loads((root/dataset/'samples.json').read_text())['entries']
               if e['shard'] == shard and (stage == 'full' or e['smoke'])]
    wanted = defaultdict(dict)
    for e in entries:
        wanted[e['path']][e['row']] = e
    config_hash = digest(json.dumps(protocol, sort_keys=True)+dataset+stage)
    status_path = root/dataset/f'{stage}_{shard}_manifest.json'
    status = {'status': 'loading', 'dataset': dataset, 'stage': stage, 'shard': shard,
              'device': device, 'expected_questions': len(entries), 'completed_questions': 0,
              'config_hash': config_hash, 'started_utc': datetime.now(timezone.utc).isoformat()}
    atomic_write_json(status_path, status)
    bundle = None
    try:
        bundle = load_runtime(identity, device)
        module = load_adapter(protocol['adapter_path'], identity, device).eval()
        status['status'] = 'running'
        atomic_write_json(status_path, status)
        for path, rows in wanted.items():
            if file_sha(path) != protocol['datasets'][dataset]['data_files'][path]:
                raise ValueError('Parquet bytes changed')
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=8):
                for doc in batch.to_pylist():
                    entry = rows.get(index)
                    index += 1
                    if entry is None:
                        continue
                    folder = root/dataset/'samples'/stage/entry['sample_id'].replace(':', '_')
                    folder.mkdir(parents=True, exist_ok=True)
                    ledger = folder/'conditions.jsonl'
                    saved = records(ledger, config_hash)
                    if len(saved) != len(METHODS):
                        prompt, target, references = prompt_target(dataset, doc)
                        with torch.inference_mode():
                            state = prepare_example(bundle, {**entry, 'image': doc['image'],
                                'question': prompt, 'target': target})
                            base = anchor_base(state)
                            for method in METHODS:
                                key = (entry['sample_id'], method)
                                if key in saved:
                                    continue
                                start = time.perf_counter()
                                output, compact, trace = None, None, None
                                if method == 'feature_anchor_adapter':
                                    with torch.autocast('cuda', dtype=torch.bfloat16):
                                        output = module(base)
                                    compact = compact_anchor(state, output)
                                    trace = {k: output[k] for k in
                                        ['indices', 'members', 'position_policy', 'partition_hash']}
                                max_new = DATASETS[dataset][2]
                                result = evaluate_state(bundle, state, compact, max_new=max_new, native_check=False)
                                if stage == 'smoke' and method == 'dense':
                                    if explicit_generate(bundle, state, max_new=max_new) != result['generated_token_ids']:
                                        raise ValueError('Native dense generate differs from cached diagnostic path')
                                    result['native_generation_identity_passed'] = True
                                selected = state if compact is None else compact
                                expected_decode = int(selected['positions'][:, :, :selected['prefix']].max())+1
                                if result['first_appended_token_position'] != expected_decode:
                                    raise ValueError('First decode M-RoPE position mismatch')
                                if (method != 'dense' and
                                        (len(output['indices']) != 128 or result['retained_visual_tokens'] != 128)):
                                    raise ValueError('K128 budget mismatch')
                                if (not result['text_positions_preserved'] or
                                        not result['placeholder_mask_labels_alignment_passed'] or
                                        result['gold_answer_in_decode_context'] or
                                        not math.isfinite(result['answer_nll'])):
                                    raise ValueError('Alignment, finite NLL, or gold-free decode check failed')
                                score, native = score_prediction(dataset, result['prediction'], doc)
                                if not math.isfinite(score) or not 0 <= score <= 1:
                                    raise ValueError('Invalid native task score')
                                record = {**result, **entry, 'dataset': dataset, 'variant': method,
                                    'config_hash': config_hash, 'status': 'completed',
                                    'checkpoint': identity['checkpoint'], 'revision': identity['revision'],
                                    'base_identity_sha256': identity['identity_sha256'],
                                    'adapter_sha256': protocol['adapter_sha256'] if method != 'dense' else None,
                                    'question': prompt, 'target': target, 'references': references,
                                    'primary_task_score': score, 'native_task_metrics': native,
                                    'ocrbench_category': doc.get('question_type') if dataset == 'ocrbench' else None,
                                    'compression_trace': trace, 'elapsed_s': time.perf_counter()-start}
                                append_json(ledger, record)
                                saved[key] = record
                        del state, base, output, compact
                    if len(saved) != 2 or {key[1] for key in saved} != set(METHODS):
                        raise ValueError('Incomplete or duplicate condition rows')
                    atomic_write_json(folder/'summary.json', {**entry, 'status': 'completed',
                        'condition_count': 2, 'config_hash': config_hash})
                    status['completed_questions'] += 1
                    atomic_write_json(status_path, status)
                    print(json.dumps({'dataset': dataset, 'stage': stage, 'shard': shard,
                        'completed': status['completed_questions'], 'expected': len(entries)}), flush=True)
        if status['completed_questions'] != len(entries):
            raise ValueError('Missing question rows')
        status.update(status='complete', peak_allocated_bytes=torch.cuda.max_memory_allocated())
    except BaseException as exc:
        status.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        atomic_write_json(status_path, status)
        if bundle:
            bundle['runtime'].close()


def report(root, dataset, stage):
    protocol = validate(root)
    entries = [e for e in json.loads((root/dataset/'samples.json').read_text())['entries']
               if stage == 'full' or e['smoke']]
    groups = {'all': defaultdict(list), 'train_overlap_free': defaultdict(list)}
    for e in entries:
        folder = root/dataset/'samples'/stage/e['sample_id'].replace(':', '_')
        summary = json.loads((folder/'summary.json').read_text())
        raw = [json.loads(s) for s in (folder/'conditions.jsonl').read_text().splitlines()]
        rows = {x['variant']: x for x in raw}
        if (len(raw) != 2 or set(rows) != set(METHODS) or summary['condition_count'] != 2 or
                any(x['config_hash'] != summary['config_hash'] for x in raw)):
            raise ValueError('Incomplete or mixed question records')
        if stage == 'smoke' and not rows['dense'].get('native_generation_identity_passed'):
            raise ValueError('Native smoke gate absent')
        for name, row in rows.items():
            if (row['image_fingerprint'] != e['image_fingerprint'] or
                    row['base_identity_sha256'] != protocol['base_identity_sha256'] or
                    row['gold_answer_in_decode_context'] or
                    (name != 'dense' and row['retained_visual_tokens'] != 128)):
                raise ValueError('Mixed checkpoint, image, answer or token budget')
        groups['all'][e['image_fingerprint']].append(rows)
        if not e['training_image_overlap']:
            groups['train_overlap_free'][e['image_fingerprint']].append(rows)
    tables = {}
    for subset, images in groups.items():
        if not images:
            tables[subset] = {'status': 'no_eligible_images'}
            continue
        table = {}
        for name in METHODS:
            rows = [q[name] for qs in images.values() for q in qs]
            accuracy = sum(x['primary_task_score'] for x in rows)/len(rows)
            item = {'n_questions': len(rows), 'n_images': len(images),
                'score': accuracy, 'mean_visual_tokens': sum(x['retained_visual_tokens'] for x in rows)/len(rows),
                'mean_nll_first_reference': sum(x['answer_nll'] for x in rows)/len(rows),
                'score_delta_vs_dense': previous.clustered_delta(list(images.values()), name)}
            if dataset == 'ocrbench':
                item['ocrbench_points_out_of_1000'] = sum(x['primary_task_score'] for x in rows) if subset == 'all' else None
                by_category = defaultdict(list)
                for x in rows:
                    by_category[x['ocrbench_category']].append(x['primary_task_score'])
                item['categories'] = {c: {'correct': sum(v), 'count': len(v)} for c,v in sorted(by_category.items())}
            table[name] = item
        tables[subset] = table
    result = {'status': 'complete', 'dataset': dataset, 'stage': stage,
        'n_questions': len(entries), 'tables': tables, 'checkpoint': protocol['checkpoint'],
        'adapter_sha256': protocol['adapter_sha256'],
        'interpretation': 'Same 7B weights, questions and scoring; paired effects resample independent image contents.'}
    atomic_write_json(root/dataset/f'{stage}_report.json', result)
    lines = [f'# Qwen2.5-VL-7B {dataset}: {stage}', '', result['interpretation'], '',
        '| Subset | Method | Questions / images | Native score | Visual tokens |',
        '|---|---|---:|---:|---:|']
    for subset, table in tables.items():
        if 'status' in table:
            continue
        for name in METHODS:
            row = table[name]
            label = f"{row['ocrbench_points_out_of_1000']:.0f}/1000" if dataset == 'ocrbench' and subset == 'all' else f"{100*row['score']:.2f}%"
            lines.append(f"| {subset} | {name} | {row['n_questions']} / {row['n_images']} | {label} | {row['mean_visual_tokens']:.1f} |")
    lines.extend(['', protocol['timing_caveat'], '',
                  'Our AI2D 7B dense baseline differs from the referenced Bifocal table; paper-level comparisons require a harmonized baseline.'])
    (root/dataset/f'{stage}_report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return result


def run(root, source):
    root.mkdir(parents=True, exist_ok=True)
    with (root/'runner.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = {'status': 'preparing', 'pid': os.getpid(),
            'started_utc': datetime.now(timezone.utc).isoformat(),
            'source_training_run': str(source), 'stages': {}}
        atomic_write_json(root/'run_manifest.json', state)
        logs = root/'logs'
        logs.mkdir(exist_ok=True)
        stop = threading.Event()
        mutex = threading.Lock()
        children = []

        def execute(dataset, stage, shard):
            validate(root)
            if stop.is_set():
                raise RuntimeError('Sibling worker failed')
            name = f'{dataset}_{stage}_{shard}'
            device = f'cuda:{shard}'
            log = logs/f'{name}_{int(time.time())}.log'
            with log.open('w') as stream:
                command = [sys.executable, str(Path(__file__)), '--root', str(root),
                    '--worker', dataset, '--stage', stage, '--shard', str(shard), '--device', device]
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    start_new_session=True, env={**os.environ, 'PYTHONUNBUFFERED': '1',
                                                 'OMP_NUM_THREADS': '4', 'TOKENIZERS_PARALLELISM': 'false'})
                with mutex:
                    children.append(child)
                    state['stages'][name] = {'status': 'running', 'pid': child.pid, 'device': device, 'log': str(log)}
                    atomic_write_json(root/'run_manifest.json', state)
                while child.poll() is None:
                    if stop.wait(2):
                        os.killpg(child.pid, signal.SIGTERM)
                        child.wait()
                        raise RuntimeError('Stopped owned worker after sibling failure')
                with mutex:
                    state['stages'][name]['status'] = 'complete' if child.returncode == 0 else 'failed'
                    atomic_write_json(root/'run_manifest.json', state)
                if child.returncode:
                    stop.set()
                    raise RuntimeError(f'{name} failed; see {log}')

        try:
            protocol = prepare(root, source)
            state.update(status='running', checkpoint=protocol['checkpoint'],
                         implementation_hash=protocol['implementation_hash'])
            atomic_write_json(root/'run_manifest.json', state)
            for dataset in DATASETS:
                state['active_dataset'] = dataset
                atomic_write_json(root/'run_manifest.json', state)
                for stage in ('smoke', 'full'):
                    with ThreadPoolExecutor(max_workers=2) as pool:
                        jobs = [pool.submit(execute, dataset, stage, shard) for shard in (0, 1)]
                        for job in jobs:
                            job.result()
                    report(root, dataset, stage)
            state.update(status='complete', completed_utc=datetime.now(timezone.utc).isoformat())
        except BaseException as exc:
            stop.set()
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid, signal.SIGTERM)
                    child.wait()
            state.update(status='failed', error=str(exc), traceback=traceback.format_exc())
            raise
        finally:
            atomic_write_json(root/'run_manifest.json', state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--source-run', type=Path)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--worker', choices=DATASETS)
    parser.add_argument('--stage', choices=('smoke', 'full'), default='smoke')
    parser.add_argument('--shard', type=int, choices=(0, 1), default=0)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.worker:
        worker(root, args.worker, args.stage, args.shard, args.device)
    else:
        if args.source_run is None:
            parser.error('--source-run is required for supervisor')
        source = args.source_run.resolve()
        root.mkdir(parents=True, exist_ok=True)
        if args.launch:
            with (root/'supervisor.log').open('a') as stream:
                child = subprocess.Popen([sys.executable, str(Path(__file__)), '--root', str(root),
                    '--source-run', str(source)], cwd=ROOT, stdout=stream,
                    stderr=subprocess.STDOUT, start_new_session=True,
                    env={**os.environ, 'PYTHONUNBUFFERED': '1'})
            print(json.dumps({'pid': child.pid, 'root': str(root)}))
        else:
            run(root, source)
