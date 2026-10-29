"""Import an immutable, completed teacher shard after a validation-only repair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from mllm_comp.analysis.common.anchor7b_runtime import file_sha
from mllm_comp.analysis.common.learned_topk_experiment import owned_output
from mllm_comp.analysis.common.task_loss_supervision import retention_target
from mllm_comp.core.serialization import atomic_write_json
from scripts.train.qwen25_vl.run_query_interaction_2x2 import read, validate

OLD_LINE = "                native_logits = bundle['model'](**state['teacher_inputs'], use_cache=False).logits[0, predictors].float()\n"
NEW_LINES = ("                native_inputs = {key: value for key, value in state['teacher_inputs'].items() if key != 'labels'}\n"
             "                native_logits = bundle['model'](**native_inputs, use_cache=False, logits_to_keep=torch.tensor(predictors, device=device)).logits[0].float()\n")


def import_shard(old_root, new_root, shard):
    old_root, new_root = owned_output(old_root), owned_output(new_root)
    new = validate(new_root)
    old = read(old_root/'protocol.json')
    old_source = Path(old['source']['source_root'])
    env = {**os.environ, 'PYTHONPATH': str(old_source), 'MLLM_TOPK_HASH_LOCAL_SOURCE': '1'}
    subprocess.run([sys.executable, '-c',
        'from pathlib import Path; from scripts.train.qwen25_vl.run_query_interaction_2x2 import validate; import sys; validate(Path(sys.argv[1]))',
        str(old_root)], cwd=old_source, env=env, check=True)
    # Source snapshot must differ only in the validated reference-head call;
    # newly added files have no influence on previously generated labels.
    names = subprocess.check_output(['git', '-C', str(old_source), 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).decode().split('\0')
    runner = 'scripts/train/qwen25_vl/run_query_interaction_2x2.py'
    for name in filter(None, names):
        a, b = old_source/name, ROOT/name
        if not a.is_file() or not b.is_file():
            raise ValueError(f'Missing original source file: {name}')
        if name == runner:
            before = a.read_text()
            if before.count(OLD_LINE) != 1 or before.replace(OLD_LINE, NEW_LINES) != b.read_text():
                raise ValueError('Teacher generation source changed beyond the native-logit validation repair')
        elif file_sha(a) != file_sha(b):
            raise ValueError(f'Original source file changed: {name}')
    # Ignore snapshot location, creation time and file-pin paths only; all
    # data, model, teacher, training and evaluation semantics must match.
    ignored = {'source', 'created_utc', 'input_hashes'}
    if {k:v for k,v in old.items() if k not in ignored} != {k:v for k,v in new.items() if k not in ignored}:
        raise ValueError('Experiment protocols differ beyond source location and creation time')
    if file_sha(old_root/'cohort.json') != file_sha(new_root/'cohort.json'):
        raise ValueError('Evaluation cohort changed')
    entries = read(old['training_manifest'])['entries']
    expected = {e['sample_id']: e for e in entries[shard::2]}
    old_index = read(old_root/f'teacher/shard{shard}/index.json')
    if set(old_index['rows']) != set(expected):
        raise ValueError('Old shard is incomplete or has extra samples')
    old_protocol_sha = file_sha(old_root/'protocol.json')
    for sample_id, item in old_index['rows'].items():
        path = Path(item['path']).resolve()
        if not path.is_relative_to((old_root/f'teacher/shard{shard}').resolve()) or file_sha(path) != item['sha256']:
            raise ValueError(f'Label path/hash mismatch: {sample_id}')
        row = read(path)
        if (row['sample_id'] != sample_id or row['image_fingerprint'] != expected[sample_id]['image_fingerprint']
                or row['protocol_sha256'] != old_protocol_sha or row['height']*row['width'] != row['source_count']):
            raise ValueError(f'Label provenance/alignment mismatch: {sample_id}')
        target = retention_target(row['candidates'], row['losses'], row['source_count'], temperature=new['teacher']['temperature'])
        if abs(float(target.sum()) - min(new['teacher']['target_count'], row['source_count'])) > 1e-3:
            raise ValueError(f'Label retention mass mismatch: {sample_id}')
    folder = new_root/f'teacher/shard{shard}'
    folder.mkdir(parents=True, exist_ok=False)
    atomic_write_json(folder/'index.json', old_index)
    atomic_write_json(folder/'status.json', {'status': 'complete', 'complete': len(expected),
        'reused_from': str(old_root), 'old_protocol_sha256': old_protocol_sha,
        'new_protocol_sha256': file_sha(new_root/'protocol.json'),
        'policy': 'read-only original label paths; validated source, protocol, image, hashes and candidate sets'})
    print(f'Validated and reused {len(expected)} immutable teacher rows from {old_root}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--old-root', type=Path, required=True)
    parser.add_argument('--new-root', type=Path, required=True)
    parser.add_argument('--shard', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    import_shard(args.old_root, args.new_root, args.shard)
