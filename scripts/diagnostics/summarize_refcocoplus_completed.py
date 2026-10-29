"""Summarize only validated completed RefCOCO+ cells, without editing live runs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


def read(path):
    return json.loads(Path(path).read_text())


def summarize(args):
    protocol = read(args.run/'protocol.json')
    sys.path.insert(0, protocol['source']['source_root'])
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    from mllm_comp.analysis.common.learned_topk_experiment import owned_output
    from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
    from mllm_comp.integrations.lmms_eval.grounding import box_scores, PROTOCOL
    from scripts.eval.lmms_eval.run_projector_extra_benchmarks import bootstrap
    from scripts.eval.lmms_eval.run_qwen25_pope_refcocoplus import METHODS, _close_target
    out = owned_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    cohort = read(args.run/'cohort.json')
    result = {'created_utc': datetime.now(timezone.utc).isoformat(), 'run': str(args.run),
              'scope': 'fixed local image subsets, not the complete official RefCOCO+ benchmark',
              'grounding_protocol': PROTOCOL, 'primary_metric': 'ACC@0.5', 'tasks': {}}
    lines = ['# RefCOCO+ completed-result audit', '', result['scope']+'.',
             'Qwen2.5-VL-7B, native dynamic resolution, processed-pixel coordinates, K128 for compressed methods.', '']
    for task in ('refcoco_plus_testA', 'refcoco_plus_testB'):
        expected = cohort[task]['rows']
        images = [r['image_fingerprint'] for r in expected]
        cells, predictions = {}, {}
        for method in METHODS:
            marker = args.run/f'eval/full/{method}/{task}/complete.json'
            if not marker.exists():
                cells[method] = {'status': 'incomplete'}
                continue
            cell = read(marker)
            if file_sha(cell['samples']) != cell['samples_sha256'] or file_sha(cell['config']) != cell['config_sha256']:
                raise ValueError('Completed artifact integrity mismatch')
            rows = sorted([json.loads(s) for s in Path(cell['samples']).read_text().splitlines()], key=lambda r:r['doc_id'])
            if len(rows) != len(expected) or [r['doc_id'] for r in rows] != list(range(len(expected))):
                raise ValueError('Missing or duplicate predictions')
            values, invalid = [], 0
            for row, item in zip(rows, expected):
                if row['input'] != item['prompt'] or not _close_target(row['target'], item['target']):
                    raise ValueError('Paired sample alignment mismatch')
                payload = row['refcoco_ACC@0.5']
                if payload.get('grounding_protocol') != PROTOCOL or payload.get('coordinate_space') != 'processed_pixels':
                    raise ValueError('Wrong coordinate/scoring protocol')
                values.append(float(box_scores(payload['pred'], payload['bbox'])['ACC@0.5']))
                invalid += payload['parse_status'] != 'valid'
            if abs(sum(values)/len(values)-cell['score']) > 1e-10:
                raise ValueError('Recomputed aggregate disagrees with saved result')
            predictions[method] = values
            cells[method] = {'status': 'complete', 'marker': str(marker), **cell,
                             'recomputed_score': sum(values)/len(values), 'parse_invalid_count': invalid}
        for method, values in predictions.items():
            baseline = predictions['dense']
            cells[method]['versus_dense'] = {**bootstrap([a-b for a,b in zip(values,baseline)], images),
                'wins': sum(a>b for a,b in zip(values,baseline)),
                'losses': sum(a<b for a,b in zip(values,baseline))}
        result['tasks'][task] = {'requests': len(expected), 'independent_images': len(set(images)), 'cells': cells}
        lines += [f'## {task}: {len(set(images))} images / {len(expected)} expressions', '',
                  '| Method | ACC@0.5 | Delta vs dense (pp) | Paired image-bootstrap 95% CI (pp) | Mean IoU | Parse valid | Mean visual tokens |',
                  '|---|---:|---:|---:|---:|---:|---:|']
        for method, cell in cells.items():
            if cell['status'] != 'complete':
                lines.append(f'| {method} | pending | — | — | — | — | — |')
                continue
            pair, metrics = cell['versus_dense'], cell['metrics']
            low, high = pair['ci95']
            lines.append(f"| {method} | {100*cell['score']:.2f}% | {100*pair['delta']:+.2f} | [{100*low:+.2f}, {100*high:+.2f}] | {100*metrics['IoU']:.2f}% | {100*metrics['parse_valid_fraction']:.2f}% | {cell['mean_retained_tokens']:.2f} |")
        lines += ['', 'Bootstrap resamples=10000, seed42, resampling independent images rather than expressions.', '']
    lines += ['Incomplete cells are not assigned partial scores. EADP is the current Qwen adaptation, not a certified original-paper reproduction.',
              'The new query-interaction experiment has no completed benchmark result and is not included in this comparison.']
    atomic_write_json(out/'summary.json', result)
    atomic_write_text(out/'report.md', '\n'.join(lines)+'\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    summarize(parser.parse_args())
