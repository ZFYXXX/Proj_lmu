"""Aggregate audio-level measurements and render observed representations."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np


def load_records(directory, *, metadata_only=False):
    rows = []
    for path in sorted(Path(directory).glob('samples.rank*.jsonl')):
        # All-layer JSONL can exceed gigabytes: avoid a second full string copy.
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if metadata_only:
                        row.pop('layers', None)
                    else:
                        # Dense numeric arrays avoid retaining millions of Python
                        # float/list objects when all 32 layers are observed.
                        for layer in row.get('layers', []):
                            for key, value in layer['heads'].items():
                                if isinstance(value, list) and key != 'distance_ms_edges':
                                    layer['heads'][key] = np.asarray(value, dtype=float)
                    rows.append(row)
    return sorted(rows, key=lambda x: x['doc_index'])


def coverage_report(rows, expected):
    ids = Counter(r['id'] for r in rows)
    doc_ids = Counter(r['doc_index'] for r in rows)
    duplicates = [k for k, n in ids.items() if n != 1]
    return {'expected_samples': expected, 'observed_samples': len(rows),
            'unique_ids': len(ids), 'duplicate_ids': duplicates,
            'missing_doc_indices': sorted(set(range(expected)) - set(doc_ids)),
            'unexpected_doc_indices': sorted(set(doc_ids) - set(range(expected))),
            'complete': len(rows) == len(ids) == expected and not duplicates
                        and set(doc_ids) == set(range(expected))}


def bootstrap_mean(values, groups, *, seed=42, repeats=1000):
    """Equal-audio point estimate; resample complete source groups for intervals."""
    values = np.asarray(values, dtype=float)
    indices = {}
    for i, group in enumerate(groups):
        indices.setdefault(group, []).append(i)
    unique = sorted(indices)
    sums, counts = [], []
    for group in unique:
        v = values[indices[group]]
        sums.append(np.nansum(v, axis=0))
        counts.append(np.isfinite(v).sum(axis=0))
    sums, counts = np.asarray(sums), np.asarray(counts)
    total = counts.sum(0)
    mean = np.divide(sums.sum(0), total, out=np.full(values.shape[1:], np.nan), where=total > 0)
    if len(unique) < 2:
        return mean, np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    rng = np.random.default_rng(seed)
    replicates = []
    for _ in range(repeats):
        idx = rng.integers(0, len(unique), len(unique))
        denominator = counts[idx].sum(0)
        replicates.append(np.divide(sums[idx].sum(0), denominator,
                          out=np.full_like(mean, np.nan), where=denominator > 0))
    replicates = np.asarray(replicates)
    lo, hi = np.full_like(mean, np.nan), np.full_like(mean, np.nan)
    valid_groups = (counts > 0).sum(0).reshape(-1)
    for i in range(mean.size):
        if valid_groups[i] < 2:
            continue
        v = replicates.reshape(repeats, -1)[:, i]
        v = v[np.isfinite(v)]
        if len(v):
            lo.flat[i], hi.flat[i] = np.quantile(v, [0.025, 0.975])
    return mean, lo, hi


def compare_baseline(rows, baseline):
    left, right = {r['id']: r for r in baseline}, {r['id']: r for r in rows}
    if len(left) != len(baseline) or len(right) != len(rows) or left.keys() != right.keys():
        raise ValueError('Baseline comparison requires identical unique sample IDs')
    result = Counter()
    changed = []
    for key, original in left.items():
        current = right[key]
        if original['audio_path'] != current['audio_path']:
            raise ValueError('Paired sample audio paths differ')
        a, b = bool(original['accuracy']), bool(current['accuracy'])
        result[('correct' if a else 'wrong') + '_to_' + ('correct' if b else 'wrong')] += 1
        if current['prediction'] != original['prediction']:
            changed.append(key)
    return {'paired_samples': len(rows), 'transitions': dict(result), 'changed_output_ids': changed,
            'outputs_identical': not changed,
            'accuracy_delta': float(np.mean([r['accuracy'] for r in rows]) - np.mean([r['accuracy'] for r in baseline]))}


def _save(fig, path):
    fig.savefig(path.with_suffix('.png'), dpi=180, facecolor='white')
    fig.savefig(path.with_suffix('.pdf'), facecolor='white')
    import matplotlib.pyplot as plt
    plt.close(fig)


def _cosine_matrix(x):
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    unit = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    gram = unit @ unit.T
    invalid = norm[:, 0] == 0
    gram[invalid, :] = np.nan
    gram[:, invalid] = np.nan
    return gram


def render_attention_distance(rows, destination):
    """Describe reading distance from the existing pass; no masking experiment."""
    import matplotlib.pyplot as plt

    layers = [layer['layer'] for layer in rows[0]['layers']]
    # Retain support for saved observations made before distance was recorded.
    if 'distance_mass' not in rows[0]['layers'][0]['heads']:
        return None
    keys = ('distance_mass', 'distance_uniform_mass', 'mean_distance_ms')
    result = {key: np.mean([[layer['heads'][key] for layer in row['layers']]
                           for row in rows], axis=0) for key in keys}
    edges = rows[0]['layers'][0]['heads']['distance_ms_edges']
    result.update(layers=layers, distance_ms_edges=edges, audio_records=len(rows))
    bounds = np.asarray(edges, dtype=float)
    labels = [f'{a:g}–{b:g}' if np.isfinite(b) else f'≥{a:g}'
              for a, b in zip(bounds, bounds[1:])]
    distance = result['distance_mass']
    excess = distance - result['distance_uniform_mass']
    result['plotted_layers'] = layers
    result['figure_pages'] = []
    limit = max(.01, float(np.abs(excess).max()))
    for start in range(0, len(layers), 4):
        selected = list(range(start, min(start+4, len(layers))))
        fig, axes = plt.subplots(len(selected), 2, figsize=(12, 3*len(selected)),
                                 layout='constrained', squeeze=False)
        for row_index, layer_index in enumerate(selected):
            for column, values, title, cmap, low, high in [
                (0, distance, 'Observed mass', 'viridis', 0, 1),
                (1, excess, 'Minus visibility-matched uniform', 'coolwarm', -limit, limit)]:
                axis = axes[row_index, column]
                im = axis.imshow(values[layer_index], origin='lower', aspect='auto',
                                 cmap=cmap, vmin=low, vmax=high)
                axis.set(title=f'Layer {layers[layer_index]}: {title}', ylabel='Head',
                         xlabel='Absolute attention distance within chunk (ms)')
                axis.set_xticks(range(len(labels)), labels, rotation=45, ha='right')
                axis.set_yticks(range(distance.shape[1]))
                axis.tick_params(axis='y', labelsize=7)
                fig.colorbar(im, ax=axis, label='Probability mass' if column == 0 else 'Excess mass')
        fig.suptitle(f'Attention distance | n={len(rows)} audio records | equal audio weight')
        name = f'attention_distance_l{layers[selected[0]]}_l{layers[selected[-1]]}'
        result['figure_pages'].append(name)
        _save(fig, destination / name)
    return result


def render_example(row, run_dir, destination):
    import matplotlib.pyplot as plt
    data = np.load(run_dir / row['example_file'], allow_pickle=False)
    layers = [r['layer'] for r in row['layers']]
    layer = layers[-1]
    time, bounds = data['time_s'], data['boundaries']
    fig, axes = plt.subplots(4, 1, figsize=(11, 9), layout='constrained', sharex=True)
    end = row['mel_length'] * float(data['mel_step_s'])
    mel = data['mel']
    mel_step = float(data['mel_step_s'])
    axes[0].imshow(mel, origin='lower', aspect='auto', cmap='magma',
                   extent=(-mel_step/2, (mel.shape[-1]-.5) * mel_step, -.5, len(mel)-.5))
    axes[0].set_ylabel('Log-Mel bin')
    r = data[f'r_{layer}'].mean(0)
    axes[1].plot(time, r, color='#0072B2', lw=1)
    top = np.argsort(-r, kind='stable')[:min(10, len(r))]
    axes[1].scatter(time[top], r[top], color='#D55E00', s=18, label='Top 10 markers')
    axes[1].axhline(1, color='0.5', ls='--', lw=0.8)
    axes[1].set_ylabel(f'Layer {layer}\nmean-head r')
    axes[1].set_ylim(0, max(2, float(r.max()) * 1.08))
    axes[1].ticklabel_format(axis='y', style='plain', useOffset=False)
    axes[1].legend(loc='upper right', fontsize=8)
    for name, label, color in [('k', 'Keys', '#0072B2'), ('v', 'Values', '#D55E00'), ('h', 'Post-MHSA hidden', '#009E73')]:
        axes[2].plot(time, data[f'{name}_adj_{layer}'], label=label, lw=0.9, color=color)
    axes[2].set_ylabel('Adjacent cosine')
    axes[2].set_ylim(-1.02, 1.02)
    axes[2].legend(loc='lower right', fontsize=8, ncols=3)
    if 'pooled_scores' in data:
        t = data['pooled_time_s']
        for y, key, color, label in [(1, 'virtual_topk', '#D55E00', 'Global TopK'),
                                     (0, 'virtual_segmentwise', '#0072B2', 'Segmentwise')]:
            ix = data[key]
            axes[3].scatter(t[ix], np.full(len(ix), y), marker='|', s=65, color=color)
        axes[3].set_yticks([0, 1], ['Segmentwise', 'Global TopK'])
        axes[3].set_ylim(-0.5, 1.5)
        axes[3].set_title('Diagnostic indices only; every token remains in the model', fontsize=10)
    for axis in axes:
        for position in bounds[1:-1]:
            axis.axvline(time[position], color='0.6', lw=0.5, ls=':')
        axis.spines[['top', 'right']].set_visible(False)
    axes[-1].set_xlim(0, end)
    axes[-1].set_xlabel('Audio time (s); dotted lines are encoder chunk boundaries')
    fig.suptitle(f'Uncompressed observation | doc {row["doc_index"]} | {row["id"]}', fontsize=11)
    _save(fig, destination / f'doc{row["doc_index"]}_timeline')

    c = row['example_chunk']
    if c is None or c >= len(bounds)-1:
        data.close()
        return
    a, b = bounds[c:c+2]
    t = time[a:b]
    if len(t) < 2:
        data.close()
        return
    edges = (t[0] - row['time_step_s']/2, t[-1] + row['time_step_s']/2)
    for layer in layers:
        keys, hidden = data[f'keys_{layer}'], data[f'hidden_{layer}']
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), layout='constrained')
        for axis, x, label in [(axes[0, 0], keys, 'Key cosine'), (axes[0, 1], hidden, 'Post-MHSA hidden cosine')]:
            im = axis.imshow(_cosine_matrix(x), vmin=-1, vmax=1, cmap='coolwarm', origin='lower',
                             extent=(*edges, *edges), aspect='auto')
            axis.set(xlabel='Token source time (s)', ylabel='Token source time (s)', title=label)
            fig.colorbar(im, ax=axis, shrink=0.8)
        centered = hidden.astype(np.float64) - hidden.mean(axis=0, dtype=np.float64)
        u, singular, _ = np.linalg.svd(centered, full_matrices=False)
        xy = u[:, :2] * singular[:2]
        axes[1, 0].plot(xy[:, 0], xy[:, 1], color='0.8', lw=0.6)
        im = axes[1, 0].scatter(xy[:, 0], xy[:, 1], c=t, cmap='viridis', s=16)
        axes[1, 0].set(xlabel='PC1', ylabel='PC2', title='PCA fitted to this chunk only')
        fig.colorbar(im, ax=axes[1, 0], label='Source time (s)', shrink=0.8)
        r = data[f'r_{layer}'].mean(0)
        cos = data[f'k_adj_{layer}'][a:b-1]
        axes[1, 1].scatter(cos, (r[a:b-1] + r[a+1:b]) / 2, s=12, alpha=0.65, color='#0072B2')
        axes[1, 1].axhline(1, color='0.5', ls='--', lw=0.7)
        axes[1, 1].set(xlabel='Adjacent key cosine', ylabel='Pair mean incoming r', title='Co-location, not causal contribution',
                       ylim=(0, max(2, float(r[a:b].max()) * 1.08)))
        axes[1, 1].ticklabel_format(axis='y', style='plain', useOffset=False)
        fig.suptitle(f'Doc {row["doc_index"]}, layer {layer}, chunk {c} | observed representations', fontsize=11)
        _save(fig, destination / f'doc{row["doc_index"]}_layer{layer}_representation')

        matrix = data[f'attention_{layer}']
        columns = min(5, len(matrix))
        grid_rows = (len(matrix)+columns-1)//columns
        fig, axes = plt.subplots(grid_rows, columns, figsize=(3*columns, 2.6*grid_rows),
                                 layout='constrained', squeeze=False, sharex=True, sharey=True)
        for head, axis in enumerate(axes.flat):
            if head >= len(matrix):
                axis.set_axis_off()
                continue
            im = axis.imshow(matrix[head], origin='lower', aspect='auto', cmap='viridis',
                             extent=(*edges, *edges), vmin=0, vmax=1)
            axis.set_title(f'Head {head}', fontsize=9)
        fig.supxlabel('Key source time (s)')
        fig.supylabel('Query source time (s)')
        fig.colorbar(im, ax=list(axes.flat), label='Attention probability (shared 0–1 scale)', shrink=.8)
        fig.suptitle(f'Doc {row["doc_index"]}, layer {layer}, chunk {c} | all heads')
        _save(fig, destination / f'doc{row["doc_index"]}_layer{layer}_all_heads')

    # This head is intentionally an extremal diagnostic, not a typical-head estimate.
    matrix = data[f'attention_{layers[-1]}']
    head = int(matrix.sum(axis=1).max(axis=1).argmax())
    fig, axis = plt.subplots(figsize=(6, 5), layout='constrained')
    im = axis.imshow(matrix[head], origin='lower', aspect='auto', cmap='viridis',
                     extent=(*edges, *edges), vmin=0)
    axis.set(xlabel='Key source time (s)', ylabel='Query source time (s)',
             title=f'Layer {layers[-1]}, head {head}: largest column mass in chunk {c}')
    fig.colorbar(im, ax=axis, label='Attention probability')
    _save(fig, destination / f'doc{row["doc_index"]}_attention')
    data.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--baseline-dir')
    parser.add_argument('--group-field', default='audio_path', help='Known source/scene group when available')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    run_dir = Path(args.run_dir).resolve()
    rows = load_records(run_dir)
    if not rows:
        parser.error('No observed samples found')
    settings = json.loads((run_dir / 'observation.json').read_text())
    coverage = coverage_report(rows, settings['expected_samples'])
    if not coverage['complete']:
        parser.error(f'Incomplete/duplicated sample set: {coverage}')
    if any(r.get(args.group_field) is None for r in rows):
        parser.error(f'Missing grouping field {args.group_field}')
    destination = run_dir / 'figures'
    destination.mkdir(exist_ok=True)
    summary = {'coverage': coverage, 'accuracy': float(np.mean([r['accuracy'] for r in rows])),
               'group_field': args.group_field, 'source_groups': len(set(r[args.group_field] for r in rows)),
               'interpretation': 'No compression; similarity and concentration are observational evidence.'}
    if args.baseline_dir:
        baseline_settings = json.loads((Path(args.baseline_dir) / 'observation.json').read_text())
        for key in ('checkpoint', 'seed'):
            if baseline_settings[key] != settings[key]:
                parser.error(f'Baseline {key} differs from the observation run')
        summary['baseline_comparison'] = compare_baseline(rows, load_records(args.baseline_dir, metadata_only=True))
    if 'layers' not in rows[0]:
        (run_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from .metrics import jsonable
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 11, 'pdf.fonttype': 42})
    layer_ids = [x['layer'] for x in rows[0]['layers']]
    if any([x['layer'] for x in r['layers']] != layer_ids for r in rows):
        raise ValueError('Layer selection differs between samples')
    groups = [str(r[args.group_field]) for r in rows]
    layer_ticks = np.unique(np.r_[np.arange(0, len(layer_ids), max(1, len(layer_ids)//8)), len(layer_ids)-1])
    tick_labels = [layer_ids[i] for i in layer_ticks]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    summaries = {}
    for space, lag, label, color in [('k', '1', 'Key adjacent', '#0072B2'),
                                     ('k', '16', 'Key lag 16', '#56B4E9'),
                                     ('v', '1', 'Value adjacent', '#D55E00'),
                                     ('post_mhsa', '1', 'Hidden adjacent', '#009E73')]:
        values = np.array([[l['similarity'][space]['lag_means'][lag] for l in row['layers']] for row in rows], dtype=float)
        mean, lo, hi = bootstrap_mean(values, groups, repeats=args.bootstrap)
        if np.isfinite(mean).any():
            axes[0, 0].plot(layer_ids, mean, label=label, color=color)
            axes[0, 0].fill_between(layer_ids, lo, hi, alpha=0.15, color=color)
        summaries[label] = {'mean': mean, 'ci95_low': lo, 'ci95_high': hi,
                            'valid_audio_count': np.isfinite(values).sum(0)}
    axes[0, 0].set(xlabel='Encoder layer (zero-based)', ylabel='Cosine similarity', title='Adjacent similarity with within-chunk distance control')
    axes[0, 0].legend(fontsize=8)
    peak = np.mean([[l['heads']['peak_r'] for l in row['layers']] for row in rows], axis=0)
    log_peak = np.log2(np.maximum(peak, 1e-12))
    im = axes[0, 1].imshow(log_peak, origin='lower', aspect='auto', cmap='magma',
                         vmin=0, vmax=max(1, float(log_peak.max())))
    axes[0, 1].set(xlabel='Head', ylabel='Selected layer index', title='Mean peak incoming r (equal audio weight)')
    axes[0, 1].set_yticks(layer_ticks, tick_labels)
    axes[0, 1].set_xticks(range(peak.shape[1]))
    fig.colorbar(im, ax=axes[0, 1], label='log2(r); uniform = 0')
    hist = np.mean([[l['heads']['peak_position_hist'] for l in row['layers']] for row in rows], axis=(0, 2))
    reference = np.mean([[l['heads']['uniform_peak_position_hist'] for l in row['layers']] for row in rows], axis=0)
    excess = hist - reference
    limit = max(.1, float(np.abs(excess).max()))
    im = axes[1, 0].imshow(excess, origin='lower', aspect='auto', cmap='coolwarm', vmin=-limit, vmax=limit,
                          extent=(0, 1, -0.5, len(layer_ids)-0.5))
    axes[1, 0].set(xlabel='Relative key position within chunk', ylabel='Encoder layer', title='Peak location minus length-matched uniform reference')
    axes[1, 0].set_yticks(layer_ticks, tick_labels)
    fig.colorbar(im, ax=axes[1, 0], label='Excess fraction per position bin')
    virtual = [row['layers'][-1].get('virtual_selection') for row in rows]
    if all(v is not None for v in virtual):
        labels = ['topk', 'random', 'segmentwise']
        values = np.array([[v[k]['segment_coverage'] for k in labels] for v in virtual])
        mean, lo, hi = bootstrap_mean(values, groups, repeats=args.bootstrap)
        axes[1, 1].bar(labels, mean, color=['#D55E00', '#999999', '#0072B2'])
        axes[1, 1].errorbar(labels, mean, yerr=[np.maximum(0, mean-lo), np.maximum(0, hi-mean)], fmt='none', color='black', capsize=4)
        axes[1, 1].set(ylabel='Fraction of temporal segments covered', ylim=(0, 1.08),
                       title='Virtual candidates only (Segmentwise enforces coverage)')
        summary['virtual_selection'] = {'coverage_mean': dict(zip(labels, mean)),
            'samples_with_ambiguous_cutoff_ties': sum(v['cutoff_tie_count'] > v['cutoff_slots'] for v in virtual)}
    else:
        axes[1, 1].set_axis_off()
    fig.suptitle(f'Uncompressed MMSU observations | n={len(rows)} audio records | grouped 95% bootstrap CI')
    _save(fig, destination / 'overview')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    concentration = {}
    for axis, metric, reference_metric, title in [
        (axes[0, 0], 'top10pct_mass', 'uniform_top10pct_mass', 'Top 10% incoming mass minus uniform'),
        (axes[0, 1], 'incoming_entropy', None, 'Incoming entropy / log(chunk length)'),
        (axes[1, 0], 'peak_query_above_uniform', None, 'Query fraction reading the peak above uniform'),
        (axes[1, 1], 'peak_start_weight', 'uniform_peak_start_weight', 'Peak at chunk start minus uniform')]:
        values = np.array([[l['heads'][metric] for l in row['layers']] for row in rows], dtype=float)
        if reference_metric:
            values -= np.array([[l['heads'][reference_metric] for l in row['layers']] for row in rows], dtype=float)
        finite = np.isfinite(values)
        counts = finite.sum(0)
        means = np.divide(np.where(finite, values, 0).sum(0), counts,
                          out=np.full(values.shape[1:], np.nan), where=counts > 0)
        cmap, vmin = ('coolwarm', -1) if metric == 'peak_start_weight' else ('viridis', 0)
        im = axis.imshow(means, origin='lower', aspect='auto', cmap=cmap, vmin=vmin, vmax=1)
        axis.set(xlabel='Head', ylabel='Encoder layer', title=title)
        axis.set_yticks(layer_ticks, tick_labels)
        axis.set_xticks(range(means.shape[1]))
        fig.colorbar(im, ax=axis, shrink=.85)
        # Intervals summarize a pre-specified mean over all heads, without choosing peak heads on B.
        audio_mean = np.divide(np.where(finite, values, 0).sum(2), finite.sum(2),
                               out=np.full(values.shape[:2], np.nan), where=finite.sum(2) > 0)
        mean, lo, hi = bootstrap_mean(audio_mean, groups, repeats=args.bootstrap)
        concentration[metric] = {'reference_subtracted': reference_metric,
            'head_mean': means, 'valid_audio_count': counts,
            'all_head_mean': mean, 'ci95_low': lo, 'ci95_high': hi}
    fig.suptitle('Concentration diagnostics across all audios; joint evidence, no universal sink cutoff')
    _save(fig, destination / 'sink_diagnostics')
    summary['concentration'] = concentration
    summary['peak_positions'] = {'observed': hist, 'uniform_reference': reference, 'excess': excess}
    summary['boundary_comparison'] = {}
    for metric in ('first_chunk_start_r', 'later_chunks_start_r'):
        values = np.array([[np.asarray(l['heads'][metric], dtype=float).mean() for l in row['layers']]
                           for row in rows], dtype=float)
        mean, lo, hi = bootstrap_mean(values, groups, repeats=args.bootstrap)
        summary['boundary_comparison'][metric] = {'all_head_mean': mean, 'ci95_low': lo,
            'ci95_high': hi, 'valid_audio_count': np.isfinite(values).sum(0), 'uniform_reference': 1}
    summary['layers'] = layer_ids
    summary['head_ids'] = list(range(peak.shape[1]))
    summary['observed_layer_head_pairs'] = len(layer_ids) * peak.shape[1]
    summary['similarity'] = summaries
    summary['attention_distance'] = render_attention_distance(rows, destination)
    checked = [l['shadow_check'] for row in rows for l in row['layers'] if l['shadow_check'] is not None]
    summary['shadow_validation'] = {'layer_checks': len(checked),
        'max_av_relative_l2': max((c['av_relative_l2'] for c in checked), default=None),
        'max_av_abs_error': max((c['av_max_abs_error'] for c in checked), default=None)}
    for row in rows:
        if row.get('example_file'):
            render_example(row, run_dir, destination)
    (run_dir / 'summary.json').write_text(json.dumps(jsonable(summary), indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(f'Wrote {destination} and {run_dir / "summary.json"}')


if __name__ == '__main__':
    main()
