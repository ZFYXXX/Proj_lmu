"""Native task scores and image-cluster paired confidence intervals."""
from collections import defaultdict
import json
from pathlib import Path

from mllm_comp.core.serialization import atomic_write_json, atomic_write_text


def paired_image_bootstrap(differences, *, seed=42, repeats=10000):
    import numpy as np
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError('Bootstrap needs finite independent image-level paired differences')
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, repeats, 100):
        index = rng.integers(0, len(values), (min(100, repeats - start), len(values)))
        draws.extend(values[index].mean(1))
    return {'mean_image_weighted_delta': float(values.mean()),
            'ci95': np.quantile(draws, [.025, .975]).tolist(), 'independent_images': len(values),
            'repeats': repeats, 'seed': seed}


def report(root):
    from scripts.train.qwen25_vl.run_learned_visual_topk import METHODS, DATA
    if not (root / 'eval_full_complete.json').exists():
        raise ValueError('Full native matrix is not complete')
    cohort = json.loads((root / 'cohort.json').read_text())
    summary, samples = {}, {}
    for method in METHODS:
        pointer = json.loads((root / 'eval/full' / method / 'complete.json').read_text())
        run = Path(pointer['run'])
        checks = json.loads((run / 'protocol_checks.json').read_text())
        metrics = {}
        for task in DATA:
            name = f'learned_topk_{task}'
            path, = run.glob(f'**/*_samples_{name}.jsonl')
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            key = 'relaxed_overall' if task == 'chartqa' else 'ocrbench_accuracy'
            metrics[task] = sum(float(r[key]) for r in rows) / len(rows)
            samples[method, task] = {int(r['doc_id']): r for r in rows}
        performance = [json.loads(line) for line in (run / 'request_performance.jsonl').read_text().splitlines()]
        if len(performance) != checks['requests'] or any(r['status'] != 'complete' for r in performance):
            raise ValueError('Incomplete performance records')
        total = sum(p['generation_s'] for p in performance)
        counts = checks['token_counts']
        metrics.update(mean_original_tokens=sum(p['original_visual_tokens'] for p in performance) / len(performance),
            mean_retained_tokens=sum(c['retained'] for c in counts) / len(counts) if counts else
                sum(p['original_visual_tokens'] for p in performance) / len(performance),
            native_generation_s=total, requests_per_generation_s=len(performance) / total,
            peak_allocated_bytes=max(p['peak_allocated_bytes'] for p in performance))
        summary[method] = metrics
    # All source grids must agree, rather than trusting a common nominal pixel cap.
    references = None
    for method in METHODS:
        run = Path(json.loads((root / 'eval/full' / method / 'complete.json').read_text())['run'])
        current = {}
        for line in (run / 'request_performance.jsonl').read_text().splitlines():
            p = json.loads(line); parts = p['sample_id'].split(':')
            current[parts[0], int(parts[2])] = p['original_visual_tokens']
        if references is None:
            references = current
        elif current != references:
            raise ValueError('Per-image native resolution/source visual counts differ across methods')
    comparisons = {}
    for task in DATA:
        metric = 'relaxed_overall' if task == 'chartqa' else 'ocrbench_accuracy'
        for control in METHODS:
            if control == 'trained_selector':
                continue
            grouped = defaultdict(list)
            wins = losses = ties = 0
            for row in cohort[task]['rows']:
                i = row['row']
                a, b = samples['trained_selector', task][i], samples[control, task][i]
                if any(a[key] != b[key] for key in ('input', 'target')):
                    raise ValueError('Paired evaluation prompt/target mismatch')
                diff = float(a[metric]) - float(b[metric])
                wins += int(diff > 0); losses += int(diff < 0); ties += int(diff == 0)
                grouped[row['image_fingerprint']].append(diff)
            result = paired_image_bootstrap([sum(v) / len(v) for v in grouped.values()])
            result.update(question_wins=wins, question_losses=losses, question_ties=ties,
                question_weighted_delta=summary['trained_selector'][task] - summary[control][task])
            comparisons[f'{task}:trained_minus_{control}'] = result
    result = {'status': 'complete', 'metrics': summary, 'paired_comparisons': comparisons,
        'ci_estimand': 'Equal-weight independent image means, not equal-weight questions; multiple questions are clustered',
        'limits': ['Single training seed; do not attribute effects solely to the Top-K operator',
            'Original anchor MLP and external methods have different training/selection scope',
            'Generation timings include differing answer lengths, are not a controlled fixed-length speed benchmark',
            'Soft training retains N tokens, so training CE improvements do not demonstrate hard deployment improvements']}
    atomic_write_json(root / 'summary.json', result)
    lines = ['# Learned visual Top-K: fixed step320, native Qwen2.5-VL-7B', '',
        '| Method | ChartQA % | OCRBench /1000 | Original N | Actual K | Generation s | Peak GB |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for method, row in summary.items():
        lines.append(f'| {method} | {row["chartqa"]*100:.2f} | {row["ocrbench"]*1000:.1f} | '
            f'{row["mean_original_tokens"]:.1f} | {row["mean_retained_tokens"]:.1f} | '
            f'{row["native_generation_s"]:.1f} | {row["peak_allocated_bytes"]/1e9:.2f} |')
    lines += ['', 'Confidence intervals and paired wins/losses are in summary.json. OCRBench /1000 is the full official sum, not a percent.',
              '', *result['limits']]
    atomic_write_text(root / 'report.md', '\n'.join(lines) + '\n')
