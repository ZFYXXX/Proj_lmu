import json
from io import BytesIO
from collections import Counter

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from mllm_comp.workflow import training_data as data
from mllm_comp.workflow import anchor_training as training
from mllm_comp.workflow.runner import write
from mllm_comp.analysis.common.anchor7b_runtime import AnchorAdapterSpec, create_anchor_adapter, save_adapter


def test_fresh_initialization_is_identical_and_does_not_load_or_modify_old_weights(tmp_path):
    identity = {'hidden_size': 16, 'identity_sha256': 'toy-base'}
    old = create_anchor_adapter(AnchorAdapterSpec(16, 8))
    with torch.no_grad():
        old.adapter[-1].weight.fill_(2)
    save_adapter(old, tmp_path, identity)
    protected = data.digest(tmp_path / 'projector.safetensors')
    hashes = []
    for _ in range(2):
        torch.manual_seed(42)
        module = training.initialize_adapter({'initialization': 'same_seed_zero_residual_mlp'}, tmp_path, identity, 'cpu')
        hashes.append(training.parameter_hash(module))
        inputs = torch.randn(128, 16)
        torch.testing.assert_close(module({'tokens': inputs})['tokens'], inputs, rtol=0, atol=0)
    assert hashes[0] == hashes[1] != training.parameter_hash(old)
    assert data.digest(tmp_path / 'projector.safetensors') == protected


def test_domain_blocks_visit_every_image_once_with_exact_registered_quotas():
    rows = [{'sample_id': f'{source}:{index}', 'source': source}
            for source, count in data.COUNTS.items() for index in range(count)]
    first = data.balanced_order(rows)
    assert first == data.balanced_order(list(reversed(rows)))
    assert len({row['sample_id'] for row in first}) == 2560
    for start in range(0, 2560, 40):
        assert Counter(data.DOMAINS[row['source']] for row in first[start:start + 40]) == {
            'ocr': 14, 'general': 10, 'grounding': 8, 'science': 8}


def test_first_qa_never_uses_later_turn_context():
    row = {'conversations': [{'from': 'human', 'value': '<image>\nWhat color?\nAnswer the question with GPT-T-COCO format.'},
                             {'from': 'gpt', 'value': 'Red'}, {'from': 'human', 'value': 'Later question'},
                             {'from': 'gpt', 'value': 'Later answer'}]}
    assert data.first_qa(row) == ('What color?', 'Red')
    with pytest.raises(ValueError):
        data.first_qa({'conversations': row['conversations'][1:]})


def test_evaluation_decoder_accepts_both_raw_bytes_and_huggingface_image_struct():
    from PIL import Image
    image = Image.new('RGB', (24, 32), 'blue')
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    assert data.fingerprint(data.open_image(buffer.getvalue())) == data.fingerprint(image)
    assert data.fingerprint(data.open_image({'bytes': buffer.getvalue(), 'path': None})) == data.fingerprint(image)


def test_refcoco_exported_xyxy_is_not_accidentally_treated_as_xywh():
    row = {'image_id': 8, 'raw_anns': json.dumps({'image_id': 8, 'bbox': [20, 30, 40, 50]}),
           'raw_image_info': json.dumps({'width': 100, 'height': 100}),
           'bbox': [20, 30, 60, 80], 'captions': ['the blue person']}
    question, answer, record = data.grounding_target(row, (200, 200))
    assert json.loads(answer) == {'bbox_2d': [40, 60, 120, 160]}
    assert 'the blue person' in question and 'bbox_2d' not in question
    assert record['bbox_xywh_original'] == [20, 30, 40, 50]
    row['bbox'] = [20, 30, 40, 50]
    with pytest.raises(ValueError, match='coordinate convention'):
        data.grounding_target(row, (200, 200))


def test_streaming_metadata_matches_json_and_rejects_truncation(tmp_path):
    rows = [{'id': '一', 'text': 'a [ bracket ], comma ,'}, {'id': 'two', 'text': 'x' * (1024 * 1024 + 40)}]
    path = tmp_path / 'source.json'
    write(path, rows)
    assert list(data.iter_json_array(path)) == rows
    path.write_text('[{"id": "broken"')
    with pytest.raises(ValueError, match='Truncated'):
        list(data.iter_json_array(path))


def test_normalized_loader_preserves_manifest_order_and_rejects_row_substitution(tmp_path):
    rows = [{'sample_id': f'sample-{i}', 'image_fingerprint': str(i),
             'question': f'Question {i}', 'target': f'Answer {i}', 'image': {'path': f'{i}.png', 'bytes': None}}
            for i in range(2)]
    parquet = tmp_path / 'qa.parquet'
    pq.write_table(pa.Table.from_pylist(rows), parquet)
    entries = [{'sample_id': row['sample_id'], 'image_fingerprint': row['image_fingerprint'],
                'path': str(parquet), 'row': index} for index, row in reversed(list(enumerate(rows)))]
    manifest = tmp_path / 'manifest.json'
    write(manifest, {'format': 'normalized_visual_qa_v1', 'entries': entries})
    assert [row['sample_id'] for row in data.load_normalized_examples(manifest)] == ['sample-1', 'sample-0']
    entries[0]['row'] = 0
    write(manifest, {'format': 'normalized_visual_qa_v1', 'entries': entries})
    with pytest.raises(ValueError, match='identity'):
        data.load_normalized_examples(manifest)


def test_paired_plan_pins_both_datasets_and_rejects_a_changed_balanced_manifest(tmp_path):
    source = tmp_path / 'source.parquet'
    source.touch()
    historical = [{'source': 'textvqa_train' if i < 2048 else 'scienceqa_train',
                   'image_fingerprint': f'old-{i}', 'path': str(source)} for i in range(2560)]
    balanced = [{'source': key, 'image_fingerprint': f'{key}-{i}', 'path': str(source)}
                for key, count in data.COUNTS.items() for i in range(count)]
    common = {'selection_reads_answers': False, 'excluded_all_evaluation_source_images': True}
    old_path, new_path = tmp_path / 'old.json', tmp_path / 'balanced.json'
    write(old_path, {**common, 'entries': historical})
    write(new_path, {**common, 'entries': balanced, 'sources': [{'path': str(source)}]})
    identity = tmp_path / 'identity.json'
    write(identity, {'checkpoint': str(tmp_path)})
    profile = tmp_path / 'profile.json'
    write(profile, {'training_manifest': str(old_path), 'training_identity': str(identity),
                    'model': str(tmp_path), 'checkpoint_root': str(tmp_path)})
    config = {'schema_version': 1, 'run_id': 'paired', 'steps': 320, 'global_batch': 8,
              'initialization': 'same_seed_zero_residual_mlp', 'training_manifest_sha256': data.digest(old_path),
              'evaluation_tasks': ['ai2d'], 'arms': {
                  'old': {'method': 'importance_fps', 'pooling': 'mean', 'train_pool': False},
                  'balanced': {'method': 'importance_fps', 'pooling': 'mean', 'train_pool': False,
                               'training_manifest': str(new_path), 'training_manifest_sha256': data.digest(new_path),
                               'source_counts': data.COUNTS}}}
    config_path = tmp_path / 'config.json'
    write(config_path, config)
    planned = training.plan(profile, config_path)
    assert {str(old_path), str(new_path), str(source)} <= set(planned['extra_input_files'])
    changed = json.loads(new_path.read_text())
    changed['entries'][0]['source'] = 'unexpected'
    write(new_path, changed)
    with pytest.raises(ValueError, match='checksum'):
        training.plan(profile, config_path)


def test_control_replacement_preserves_original_manifest_counts_and_other_examples(tmp_path):
    from PIL import Image
    images = []
    for color in ['blue', 'red']:
        image = Image.new('RGB', (16, 16), color)
        stream = BytesIO()
        image.save(stream, format='PNG')
        images.append({'bytes': stream.getvalue(), 'path': None})
    path = tmp_path / 'science_train.parquet'
    pq.write_table(pa.Table.from_pylist([{'image': value} for value in images]), path)
    excluded = {data.fingerprint(data.open_image(images[0]))}
    entries = [{'source': 'textvqa_train' if i < 2048 else 'scienceqa_train',
                'image_fingerprint': f'original-{i}', 'sample_id': f'sample-{i}',
                'path': str(path), 'row': 0} for i in range(2560)]
    entries[2048]['image_fingerprint'] = next(iter(excluded))
    historical = tmp_path / 'historical.json'
    write(historical, {'entries': entries, 'selection_reads_answers': False})
    original_hash = data.digest(historical)
    cleaned = data.clean_control({'training_manifest': str(historical)}, tmp_path, excluded)
    after = json.loads(cleaned.read_text())['entries']
    assert data.digest(historical) == original_hash
    assert len(after) == len({e['image_fingerprint'] for e in after}) == 2560
    assert Counter(e['source'] for e in after) == {'textvqa_train': 2048, 'scienceqa_train': 512}
    assert after[:2048] == entries[:2048] and after[2049:] == entries[2049:]
    assert after[2048]['image_fingerprint'] == data.fingerprint(data.open_image(images[1]))
    assert json.loads((tmp_path / 'control_replacements.json').read_text())['replacement_count'] == 1
