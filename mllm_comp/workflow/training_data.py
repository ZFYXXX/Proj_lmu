"""Build an auditable, train-only balanced QA pool and launch the paired MLP study."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import random
import re
import time

from .catalog import ROOT, load_tasks
from .runner import digest, read, write

NEXT_REPO = 'lmms-lab/LLaVA-NeXT-Data'
NEXT_REVISION = 'c8aef391ce214167c4ebc7f06bc05a50dee2e75f'
NEXT_JSON = 'llava_next_raw_format/llava_next_raw_format_processed.json'
REF_REVISION = '9fba8200c5326e996f789191f095bd464ef1d09e'
COUNTS = {'ocrvqa_next_train': 320, 'chartqa_next_train': 320, 'docvqa_next_train': 256,
          'gqa_next_train': 384, 'llava_next_coco_train': 256, 'refcoco_train': 512,
          'scienceqa_train': 256, 'ai2d_next_train': 256}
DOMAINS = {'ocrvqa_next_train': 'ocr', 'chartqa_next_train': 'ocr', 'docvqa_next_train': 'ocr',
           'gqa_next_train': 'general', 'llava_next_coco_train': 'general',
           'refcoco_train': 'grounding', 'scienceqa_train': 'science', 'ai2d_next_train': 'science'}


def order_key(value):
    return hashlib.sha256(f'42:{value}'.encode()).hexdigest()


def validate_manifest(path, expected_hash, expected_sources=None):
    if digest(path) != expected_hash:
        raise ValueError('Training manifest checksum changed')
    payload = read(path)
    entries = payload['entries']
    if len(entries) != 2560 or len({e['image_fingerprint'] for e in entries}) != 2560:
        raise ValueError('Training requires 2560 distinct images')
    if payload.get('selection_reads_answers') is not False or not payload.get('excluded_all_evaluation_source_images'):
        raise ValueError('Training manifest needs answer-independent sampling and evaluation exclusion')
    if expected_sources and Counter(e['source'] for e in entries) != expected_sources:
        raise ValueError('Training source quotas differ')
    for entry in entries:
        if not Path(entry['path']).is_file():
            raise ValueError(f"Training shard missing: {entry['path']}")
    return payload


def load_normalized_examples(path):
    import pyarrow.parquet as pq
    payload = read(path)
    if payload.get('format') != 'normalized_visual_qa_v1':
        from scripts.train.qwen25_vl.train_learned_hybrid import load_examples
        return load_examples(path)
    rows = {}
    for filename in {e['path'] for e in payload['entries']}:
        rows[filename] = pq.read_table(filename).to_pylist()
    examples = []
    for entry in payload['entries']:
        value = rows[entry['path']][entry['row']]
        if value['image_fingerprint'] != entry['image_fingerprint'] or value['sample_id'] != entry['sample_id']:
            raise ValueError('Normalized QA row identity changed')
        examples.append({**entry, **value})
    return examples


def first_qa(row):
    turns = row.get('conversations') or []
    if len(turns) < 2 or turns[0]['from'] != 'human' or turns[1]['from'] != 'gpt':
        raise ValueError('Expected a self-contained initial user/assistant pair')
    if not all(isinstance(turn.get('value'), str) for turn in turns[:2]):
        raise ValueError('Unsupported first-turn content')
    question = turns[0]['value'].replace('<image>', '').strip()
    question = re.sub(r'\nAnswer the question with GPT-T-COCO format\.?$', '', question)
    if not question or not turns[1]['value'].strip():
        raise ValueError('Empty question/answer')
    return question, turns[1]['value'].strip()


def balanced_order(rows):
    """Exact 35/25/20/20 quotas per forty examples; one image is visited once."""
    pools = defaultdict(list)
    for row in sorted(rows, key=lambda r: order_key(r['sample_id'])):
        pools[DOMAINS[row['source']]].append(row)
    quotas = {'ocr': 14, 'general': 10, 'grounding': 8, 'science': 8}
    if {key: len(value) for key, value in pools.items()} != {key: value * 64 for key, value in quotas.items()}:
        raise ValueError('Domain quotas do not sum to the registered 2560-image mixture')
    result = []
    for block in range(64):
        chunk = [row for key, count in quotas.items() for row in pools[key][block * count:(block + 1) * count]]
        random.Random(42 + block).shuffle(chunk)
        result.extend(chunk)
    return result


def iter_json_array(path):
    """Stream the upstream 845 MB annotation file without materializing all conversations."""
    decoder = json.JSONDecoder()
    with Path(path).open() as stream:
        buffer, ended = '', False
        while True:
            buffer = buffer.lstrip(' \r\n\t[,')
            if buffer.startswith(']'):
                return
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    if ended or buffer.strip():
                        raise ValueError('Truncated upstream JSON array')
                    return
                buffer += chunk
                ended = True
                continue
            yield value
            buffer = buffer[end:]


def fingerprint(image):
    from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
    return media_fingerprint([image.convert('RGB')])


def open_image(value):
    from PIL import Image
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray)):
        return Image.open(BytesIO(value)).convert('RGB')
    if isinstance(value, str):
        return Image.open(value).convert('RGB')
    if not value.get('bytes') and not value.get('path'):
        return None
    return Image.open(BytesIO(value['bytes']) if value.get('bytes') else value['path']).convert('RGB')


def grounding_target(row, processed_size):
    """The HF bbox column is already xyxy; raw COCO annotations are xywh."""
    import math
    raw = json.loads(row['raw_anns'])
    info = json.loads(row['raw_image_info'])
    x, y, w, h = raw['bbox']
    if str(raw['image_id']) != str(row['image_id']) or w <= 0 or h <= 0:
        raise ValueError('Invalid RefCOCO annotation/image identity')
    xyxy = [x, y, x + w, y + h]
    if not all(math.isclose(a, b, abs_tol=1e-4) for a, b in zip(xyxy, row['bbox'])):
        raise ValueError('Unexpected RefCOCO exported bbox coordinate convention')
    rw, rh = processed_size
    bbox = [round(min(max(v / bound, 0), 1) * size)
            for v, bound, size in zip(xyxy, [info['width'], info['height']] * 2, [rw, rh] * 2)]
    expression = sorted(row['captions'], key=order_key)[0]
    question = f'Locate {expression} in this image and output the bbox coordinates in JSON format.'
    return question, json.dumps({'bbox_2d': bbox}), {
        'bbox_xywh_original': raw['bbox'], 'original_size': [info['width'], info['height']],
        'processed_size': [rw, rh], 'coordinate_format': 'processed_image_pixels'}


def clean_control(profile, folder, excluded):
    """Preserve the historical list and explicitly replace only overlapping images."""
    import pyarrow.parquet as pq
    historical = read(profile['training_manifest'])
    source_hash = digest(profile['training_manifest'])
    target = folder / 'clean_control_manifest.json'
    if target.exists():
        saved = read(target)
        if saved['historical_manifest_sha256'] != source_hash:
            raise ValueError('Historical control changed after cleaning')
        if excluded & {e['image_fingerprint'] for e in saved['entries']}:
            raise ValueError('Cleaned control still overlaps evaluation')
        return target
    bad = [e for e in historical['entries'] if e['image_fingerprint'] in excluded]
    used = {e['image_fingerprint'] for e in historical['entries']}
    replacements = {}
    for source, amount in Counter(e['source'] for e in bad).items():
        candidates = {}
        paths = sorted({e['path'] for e in historical['entries'] if e['source'] == source})
        for path in paths:
            index = 0
            for batch in pq.ParquetFile(path).iter_batches(batch_size=16, columns=['image']):
                for value in batch.column(0).to_pylist():
                    image = open_image(value)
                    key = fingerprint(image)
                    if key not in used and key not in excluded and key not in candidates:
                        candidates[key] = {'source': source, 'path': path, 'row': index,
                            'sample_id': f'{source}:{Path(path).name}:{index}', 'image_fingerprint': key}
                    index += 1
        keys = sorted(candidates, key=lambda key: order_key(f'control-replacement:{source}:{key}'))[:amount]
        if len(keys) != amount:
            raise ValueError(f'Not enough clean replacements for {source}')
        for old, key in zip([e for e in bad if e['source'] == source], keys):
            replacements[old['sample_id']] = candidates[key]
            used.add(key)
    entries = [replacements.get(e['sample_id'], e) for e in historical['entries']]
    audit = [{'removed': old, 'replacement': replacements[old['sample_id']]} for old in bad]
    write(folder / 'control_replacements.json', {'historical_manifest_sha256': source_hash,
        'replacement_count': len(audit), 'reason': 'decoded RGB image overlaps evaluation', 'entries': audit})
    write(target, {**historical, 'entries': entries, 'excluded_all_evaluation_source_images': True,
        'historical_manifest': profile['training_manifest'], 'historical_manifest_sha256': source_hash,
        'control_replacements': str(folder / 'control_replacements.json'),
        'evaluation_exclusion': profile.get('exclusion_cache_dir', str(folder)),
        'sampling': 'Historical order, replacing only explicitly audited evaluation-overlap images'})
    validate_manifest(target, digest(target), {'textvqa_train': 2048, 'scienceqa_train': 512})
    return target


def evaluation_exclusion(data_root, folder, extra_env):
    import pyarrow.parquet as pq
    tasks = load_tasks()
    files = set()
    for name, task in tasks.items():
        if name.startswith('vqav2'):
            continue
        files.update(Path(data_root).glob(task['pattern']))
    files.update((Path(data_root) / 'GQA/testdev_all_images').glob('*.parquet'))
    if extra_env.get('MLLM_MMSTAR_PARQUET'):
        files.add(Path(extra_env['MLLM_MMSTAR_PARQUET']))
    vqa_files = sorted((Path(data_root) / 'VQAv2/data').glob('validation-*.parquet'))
    pins = {str(p): [p.stat().st_size, p.stat().st_mtime_ns] for p in sorted(files | set(vqa_files))}
    cache = folder / 'evaluation_exclusion.json'
    if cache.exists():
        cached = read(cache)
        if cached['files'] != pins:
            raise ValueError('Evaluation sources changed during preparation')
        return set(cached['rgb_hashes']), set(cached['coco_ids'])
    parts = folder / 'exclusion_parts'
    parts.mkdir(exist_ok=True)
    def scan(path):
        part = parts / (order_key(f'{path}:{pins[str(path)]}') + '.json')
        if part.exists():
            return set(read(part)['rgb_hashes'])
        hashes = set()
        pf = pq.ParquetFile(path)
        columns = [key for key in pf.schema_arrow.names if key == 'image' or re.fullmatch(r'image_\d+', key)]
        if columns:
            for batch in pf.iter_batches(batch_size=16, columns=columns):
                for column in batch.columns:
                    for value in column.to_pylist():
                        image = open_image(value)
                        if image is not None:
                            hashes.add(fingerprint(image))
        write(part, {'rgb_hashes': sorted(hashes), 'path': str(path), 'stat': pins[str(path)]})
        print(f'EXCLUSION {path.name}: {len(hashes)} images', flush=True)
        return hashes
    excluded = set()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for hashes in pool.map(scan, sorted(files)):
            excluded.update(hashes)
    # COCO IDs exclude recompressed/resized variants of VQAv2 validation images.
    coco = set()
    for path in vqa_files:
        pf = pq.ParquetFile(path)
        column = next((key for key in ['image_id', 'imageId'] if key in pf.schema_arrow.names), None)
        if column is None:
            raise ValueError('Cannot identify VQAv2 validation COCO image IDs')
        for batch in pf.iter_batches(columns=[column]):
            coco.update(str(int(value)) for value in batch.column(0).to_pylist())
    for path in files:
        if 'RefCOCO' not in str(path):
            continue
        pf = pq.ParquetFile(path)
        if 'file_name' in pf.schema_arrow.names:
            for batch in pf.iter_batches(columns=['file_name']):
                coco.update(str(int(Path(value).stem.split('_')[-1])) for value in batch.column(0).to_pylist())
    write(cache, {'files': pins, 'rgb_hashes': sorted(excluded), 'coco_ids': sorted(coco),
                  'vqav2_exclusion': 'all validation COCO IDs, including resized image variants'})
    return excluded, coco


def prepare(profile, folder):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, HfFileSystem
    from transformers import AutoImageProcessor
    from qwen_vl_utils.vision_process import smart_resize

    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'balanced_manifest.json').exists():
        return folder / 'balanced_manifest.json'
    status_path = folder / 'preparation_status.json'
    def status(phase, **values):
        write(status_path, {'phase': phase, 'updated_unix': time.time(), **values})
        print(phase, values, flush=True)
    status('checking_evaluation_overlap')
    exclusion_folder = Path(profile.get('exclusion_cache_dir', str(folder)))
    excluded, coco_excluded = evaluation_exclusion(profile['data_root'], exclusion_folder, profile.get('environment', {}))
    control_manifest = clean_control(profile, folder, excluded)
    write(folder / 'prepared_profile.json', {**profile, 'training_manifest': str(control_manifest)})
    images = folder / 'images'; images.mkdir(exist_ok=True)
    partial = folder / 'selected_rows.json'
    selected = read(partial) if partial.exists() else []
    used, rejects = {r['image_fingerprint'] for r in selected}, Counter()
    if used & excluded or len(used) != len(selected):
        raise ValueError('Partial training data contains overlap/duplicates')
    processor = AutoImageProcessor.from_pretrained(profile['model'], local_files_only=True, use_fast=False)
    def add(source, sample_id, image, question, target, provenance):
        key = fingerprint(image)
        if key in excluded or key in used:
            rejects['evaluation_or_duplicate_image'] += 1; return False
        if min(image.size) < 128:
            rejects['very_small_source_image'] += 1; return False
        height, width = smart_resize(image.height, image.width, factor=processor.patch_size * processor.merge_size,
                                     min_pixels=200704, max_pixels=1605632)
        tokens = height * width // (processor.patch_size * processor.merge_size) ** 2
        if tokens <= 192:
            rejects['insufficient_visual_tokens'] += 1; return False
        path = images / f'{key}.png'
        if not path.exists():
            temporary = path.with_suffix('.tmp')
            image.save(temporary, format='PNG')
            os.replace(temporary, path)
        selected.append({'sample_id': sample_id, 'source': source, 'image_fingerprint': key,
                         'image': {'path': str(path), 'bytes': None}, 'question': question, 'target': target,
                         'source_visual_tokens': tokens, 'provenance': json.dumps(provenance, sort_keys=True)})
        used.add(key)
        return True
    # Local train splits retain their existing question formatting.
    from scripts.train.qwen25_vl.train_learned_hybrid import load_examples
    old_examples = load_examples(control_manifest)
    for source in ['scienceqa_train']:
        if Counter(r['source'] for r in selected)[source] >= COUNTS[source]:
            continue
        candidates = sorted([e for e in old_examples if e['source'] == source], key=lambda e: order_key(e['sample_id']))
        for e in candidates:
            add(source, e['sample_id'], open_image(e['image']), e['question'], e['target'], {'path': e['path'], 'row': e['row'], 'split': 'train'})
            if Counter(r['source'] for r in selected)[source] == COUNTS[source]:
                break
    del old_examples
    write(partial, selected)
    status('local_training_sources_ready', selected=len(selected))
    # Raw-format and parquet releases differ in row count (raw excludes text-only
    # examples). Match by ID + first question, never by raw JSON row offset.
    raw_path = profile.get('next_metadata')
    if not raw_path or not Path(raw_path).is_file():
        raw_path = hf_hub_download(NEXT_REPO, NEXT_JSON, repo_type='dataset', revision=NEXT_REVISION,
                                  token=False, local_dir=folder / 'downloads')
    wanted = {'coco': 'llava_next_coco_train', 'gqa': 'gqa_next_train',
              'chartqa': 'chartqa_next_train', 'ai2d': 'ai2d_next_train',
              'ocr_vqa': 'ocrvqa_next_train', 'docvqa': 'docvqa_next_train'}
    annotations, invalid = {}, []
    for row in iter_json_array(raw_path):
        image_path = str(row.get('image', ''))
        prefix = image_path.split('/')[0]
        if prefix not in wanted:
            continue
        if prefix in {'chartqa', 'coco'} and '/train' not in image_path:
            continue
        try:
            question, answer = first_qa(row)
        except ValueError as exc:
            invalid.append({'id': row.get('id'), 'source': prefix, 'reason': str(exc)})
            continue
        annotations[(str(row['id']), row['conversations'][0]['value'])] = (wanted[prefix], image_path, question, answer)
    write(folder / 'invalid_upstream_annotations.json', invalid)
    rejects['invalid_first_qa'] = len(invalid)
    status('next_annotation_index_ready', valid=len(annotations), filtered=len(invalid))
    # Pin a modest, declared set of upstream train shards; no 92 GB pool download.
    # These ranges include the source boundaries, verified through IDs above.
    source_shards = {'llava_next_coco_train': [0, 1, 2, 3], 'gqa_next_train': [128, 129, 130, 131],
                     'ocrvqa_next_train': [182, 183, 184], 'docvqa_next_train': [208, 209, 210],
                     'ai2d_next_train': [217, 218, 219], 'chartqa_next_train': [219, 220, 221]}
    ref_file = profile.get('ref_metadata')
    if not ref_file or not Path(ref_file).is_file():
        ref_file = hf_hub_download('jxu124/refcoco', 'data/train-00000-of-00001-94431d5f4bd5b93f.parquet',
                                  repo_type='dataset', revision=REF_REVISION, token=False, local_dir=folder / 'downloads/refcoco')
    references = {}
    for row in pq.read_table(ref_file).to_pylist():
        if row['split'] != 'train':
            raise ValueError('RefCOCO training candidate split mismatch')
        key = str(int(row['image_id']))
        if key not in references or order_key(str(row['ref_id'])) < order_key(str(references[key]['ref_id'])):
            references[key] = row

    def add_reference(image, coco_id, provenance):
        row = references[coco_id]
        rh, rw = smart_resize(image.height, image.width, factor=28, min_pixels=200704, max_pixels=1605632)
        question, target, box_record = grounding_target(row, (rw, rh))
        return add('refcoco_train', f'refcoco_train:{row["ref_id"]}', image, question, target,
            {'repo': 'jxu124/refcoco', 'revision': REF_REVISION, 'split': 'train', 'image_id': row['image_id'],
             **box_record, 'image_source': provenance})

    def quota_done(source):
        counts = Counter(r['source'] for r in selected)
        return counts[source] >= COUNTS[source] and (source != 'llava_next_coco_train' or counts['refcoco_train'] >= COUNTS['refcoco_train'])

    fs = HfFileSystem(token=False)
    for source, shards in source_shards.items():
        for shard in shards:
            if quota_done(source):
                break
            filename = f'data/train-{shard:05d}-of-00250.parquet'
            remote = f'datasets/{NEXT_REPO}@{NEXT_REVISION}/{filename}'
            status('reading_next_training_shard', source=source, shard=shard, selected=len(selected))
            with fs.open(remote, 'rb', block_size=65536) as stream:
                pf = pq.ParquetFile(stream)
                for group in range(pf.num_row_groups):
                    metadata = pf.read_row_group(group, columns=['id', 'conversations']).to_pylist()
                    eligible = []
                    for index, row in enumerate(metadata):
                        turns = row.get('conversations') or []
                        key = (str(row['id']), turns[0]['value'] if turns else '')
                        ann = annotations.get(key)
                        if ann and ann[0] == source:
                            if source == 'llava_next_coco_train' and str(int(Path(ann[1]).stem)) in coco_excluded:
                                rejects['vqav2_validation_coco_id'] += 1; continue
                            eligible.append((index, row, ann))
                    if not eligible:
                        continue
                    values = pf.read_row_group(group, columns=['image']).column('image').to_pylist()
                    for index, row, ann in sorted(eligible, key=lambda x: order_key(f'{source}:{x[1]["id"]}')):
                        provenance = {'repo': NEXT_REPO, 'revision': NEXT_REVISION, 'file': filename, 'row_group': group,
                                      'offset': index, 'raw_image_path': ann[1], 'split': 'upstream_train'}
                        image = open_image(values[index])
                        coco_id = str(int(Path(ann[1]).stem)) if source == 'llava_next_coco_train' else None
                        if (coco_id in references and Counter(r['source'] for r in selected)['refcoco_train'] < COUNTS['refcoco_train']):
                            add_reference(image, coco_id, provenance)
                        elif Counter(r['source'] for r in selected)[source] < COUNTS[source]:
                            add(source, f'{source}:{row["id"]}', image, ann[2], ann[3], provenance)
                        if quota_done(source):
                            break
                    write(partial, selected)
                    status('extracting_next_subset', source=source, shard=shard, row_group=group,
                           selected=len(selected), counts=dict(Counter(r['source'] for r in selected)))
                    if quota_done(source):
                        break
        if Counter(r['source'] for r in selected)[source] != COUNTS[source]:
            raise ValueError(f'Not enough verified {source} candidates in declared train shards')
    del annotations
    if Counter(r['source'] for r in selected) != COUNTS:
        raise ValueError(f'Incomplete balanced mixture: {Counter(r["source"] for r in selected)}')
    rows = balanced_order(selected)
    normalized = folder / 'balanced_qa.parquet'
    pq.write_table(pa.Table.from_pylist(rows), normalized)
    entries = [{key: row[key] for key in ['sample_id', 'source', 'image_fingerprint']} |
               {'path': str(normalized), 'row': index} for index, row in enumerate(rows)]
    payload = {'format': 'normalized_visual_qa_v1', 'seed': 42, 'entries': entries,
               'sources': [{'path': str(normalized), 'sha256': digest(normalized), 'split': 'train'}],
               'source_counts': COUNTS, 'selection_reads_answers': False,
               'excluded_all_evaluation_source_images': True, 'exclusion_audit': str(exclusion_folder / 'evaluation_exclusion.json'),
               'upstream': {'repo': NEXT_REPO, 'revision': NEXT_REVISION, 'raw_annotation_sha256': digest(raw_path)},
               'candidate_scope': {'next_parquet_shards': source_shards,
                                   'scienceqa': 'subset of the cleaned control train cohort'},
               'next_training_images': 1792, 'supplemental_training_images': 768,
               'frozen_scorer_note': 'Existing scorer is shared, not retrained. Its historical training provenance remains unchanged.',
               'sampling': 'fixed seed/hash order, one first QA per image, exact RGB dedup and COCO validation ID exclusion',
               'rejections': dict(rejects)}
    write(folder / 'balanced_manifest.json', payload)
    status('ready', selected=len(rows), counts=COUNTS)
    return folder / 'balanced_manifest.json'


def launch(profile_path, manifest, run_id):
    from types import SimpleNamespace
    import torch
    from .anchor_training import start, initialize_adapter, parameter_hash
    profile_path = manifest.parent / 'prepared_profile.json'
    profile = read(profile_path)
    config = read(ROOT / 'configs/workflow/training/anchor_adaptation.json')
    config.update(run_id=run_id, initialization='same_seed_zero_residual_mlp',
                  training_manifest_sha256=digest(profile['training_manifest']),
                  evaluation_tasks=['chartqa', 'ocrbench', 'ai2d', 'gqa', 'scienceqa_img',
                                    'refcoco+_bbox_rec_testA', 'refcoco+_bbox_rec_testB'])
    config['arms'] = {
        'R1_clean_control_scratch': {'method': 'importance_fps', 'pooling': 'mean', 'train_pool': False},
        'R1_balanced_data_scratch': {'method': 'importance_fps', 'pooling': 'mean', 'train_pool': False,
            'training_manifest': str(manifest), 'training_manifest_sha256': digest(manifest), 'source_counts': COUNTS}}
    torch.manual_seed(config['seed'])
    old = Path(profile['checkpoint_root']) / 'checkpoints/old_mlp'
    config['initial_parameter_sha256'] = parameter_hash(initialize_adapter(config, old, {}, 'cpu'))
    config['pool_training'] = 'none; existing image-context anchor scorer frozen in both arms'
    path = manifest.parent / 'training_config.json'
    write(path, config)
    return start(SimpleNamespace(profile=profile_path, training_config=path, run_id=run_id,
        action='train', run_dir=None, resume=False, wait_for_gpu=True, launch=True))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--profile', type=Path, default=ROOT / 'configs/workflow/profiles/p6000.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--launch-training', action='store_true')
    parser.add_argument('--run-id', default='r1_next_subset_20261104_v2')
    args = parser.parse_args()
    folder = args.output.resolve()
    if not folder.is_relative_to(ROOT / 'outputs'):
        raise ValueError('Preparation writes must stay in this worktree outputs')
    import fcntl
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'preparation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            manifest = prepare(read(args.profile), folder)
            validate_manifest(manifest, digest(manifest), COUNTS)
            if args.launch_training:
                raise SystemExit(launch(args.profile, manifest, args.run_id))
        except Exception as exc:
            import traceback
            write(folder / 'preparation_status.json', {'phase': 'failed', 'error': str(exc),
                'traceback': traceback.format_exc(), 'updated_unix': time.time()})
            raise
