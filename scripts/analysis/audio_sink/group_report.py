"""Offline MMSU 2x3 analysis: join labels and aggregate saved observations only."""
import argparse
import csv
import json
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

HERE = Path(__file__).resolve().parent
MAJORS = ['Perception', 'Reasoning']
DOMAINS = ['Semantic', 'Phonology', 'Paralinguistic']
SELECTED = [(0, 15), (23, 1), (23, 6), (23, 3), (31, 3), (7, 10), (15, 8)]
HEAD_KEYS = ['peak_r', 'start_r', 'top10pct_mass', 'incoming_entropy',
             'query_entropy', 'peak_query_above_uniform', 'peak_start_weight',
             'uniform_peak_start_weight', 'uniform_top10pct_mass',
             'first_chunk_start_r', 'later_chunks_start_r', 'mean_distance_ms']
METHODS = ['topk', 'random', 'segmentwise']
VKEYS = ['segment_coverage', 'largest_segment_share', 'longest_unselected_fraction']
BLUE, ORANGE, GREEN = '#0072B2', '#D55E00', '#009E73'
COLORS = [BLUE, ORANGE, GREEN]


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)


def boot_columns(x, repeats=1000):
    """Pointwise 95% record bootstrap, equal record weights, fixed seed."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    assert np.isfinite(x).all() and len(x)
    rng = np.random.default_rng(42)
    means = []
    for start in range(0, repeats, 20):
        ix = rng.integers(len(x), size=(min(20, repeats-start), len(x)))
        means.append(x[ix].mean(axis=1))
    bounds = np.quantile(np.concatenate(means), [.025, .975], axis=0)
    return [{'mean': float(v), 'ci95': bounds[:, j].tolist(), 'n': len(x)}
            for j, v in enumerate(x.mean(0))]


def aggregate(run, dataset_dir, out):
    taxonomy = json.loads((HERE / 'task_taxonomy.json').read_text())
    task_map = {}
    for major, domains in taxonomy['groups'].items():
        for domain, tasks in domains.items():
            for task in tasks:
                assert task not in task_map
                task_map[task] = (major, domain)
    assert len(task_map) == 47

    metadata = {}
    parquet_files = sorted(dataset_dir.glob('data/*.parquet'))
    assert parquet_files, f'No metadata Parquet files in {dataset_dir}'
    for path in parquet_files:
        for r in pq.read_table(path, columns=['id', 'category', 'sub_category', 'task_name']).to_pylist():
            assert r['id'] not in metadata
            metadata[r['id']] = r
    n = len(metadata)
    assert n == 5000
    paths = sorted(run.glob('samples.rank*.jsonl'))
    if not paths:
        raise ValueError(f'No saved observations in {run}')
    first = None
    for path in paths:
        with path.open() as stream:
            line = next((line for line in stream if line.strip()), None)
        if line is not None:
            first = json.loads(line)
            break
    if first is None or 'layers' not in first:
        raise ValueError('This report requires complete observe/suite reference records')
    layers = [layer['layer'] for layer in first['layers']]
    head_count = len(first['layers'][0]['heads']['peak_r'])
    layer_count = len(layers)
    selected = [(l, h) for l, h in SELECTED if l in layers and h < head_count]
    if selected != SELECTED or layers[-1] != 31:
        raise ValueError('The Qwen 7B group report requires L0/7/15/23/31 and heads 0–15; all layers are supported')
    heads = {k: np.full((n, layer_count, head_count), np.nan) for k in HEAD_KEYS}
    distance = np.empty((n, layer_count, head_count, 9), np.float32)
    distance_ref = np.empty_like(distance)
    position = np.empty((n, layer_count, head_count, 10), np.float32)
    position_ref = np.empty((n, layer_count, 10), np.float32)
    similarities = np.empty((n, layer_count, 3))
    virtual = np.empty((n, 3, 3))
    records = [None] * n
    for path in paths:
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                assert [l['layer'] for l in row['layers']] == layers
                i = row['doc_index']
                assert records[i] is None
                label = metadata[row['id']]
                major, domain = task_map[label['task_name']]
                assert major.lower() == row['question_type'] == label['category'].lower()
                assert label['task_name'] == row['task_name']
                assert label['sub_category'] == row['sub_category']
                expected = 'Paralinguistics' if domain == 'Paralinguistic' else 'Linguistics'
                assert label['sub_category'] == expected
                records[i] = {'doc_index': i, 'id': row['id'], 'audio_path': row['audio_path'],
                    'major': major, 'domain': domain, 'original_sub_category': label['sub_category'],
                    'task_name': label['task_name'], 'accuracy': row['accuracy'],
                    'duration_s': row['mel_length'] * row['time_step_s'] / 2,
                    'uniform_distance_ms': row['time_step_s'] * 1000 *
                        sum(nc*nc-1 for nc in row['chunk_lengths']) / (3*sum(row['chunk_lengths'])),
                    'chunks': len(row['chunk_lengths']),
                    'virtual_budget': row['layers'][-1]['virtual_selection']['budget']}
                for li, layer in enumerate(row['layers']):
                    assert layer['layer'] == layers[li]
                    for key in HEAD_KEYS:
                        heads[key][i, li] = layer['heads'][key]
                    distance[i, li] = layer['heads']['distance_mass']
                    distance_ref[i, li] = layer['heads']['distance_uniform_mass']
                    position[i, li] = layer['heads']['peak_position_hist']
                    position_ref[i, li] = layer['heads']['uniform_peak_position_hist']
                    for si, space in enumerate(['k', 'v', 'post_mhsa']):
                        similarities[i, li, si] = layer['similarity'][space]['lag_means']['1']
                v = row['layers'][-1]['virtual_selection']
                for mi, method in enumerate(METHODS):
                    virtual[i, mi] = [v[method][k] for k in VKEYS]
        print('Read', path.name, flush=True)
    assert all(r is not None for r in records)
    assert {r['id'] for r in records} == set(metadata)
    assert {r['task_name'] for r in records} == set(task_map)
    assert len({r['audio_path'] for r in records}) == n
    accuracy = np.array([r['accuracy'] for r in records])
    assert np.isin(accuracy, [0, 1]).all()
    # Saved GPU FP32 scatter-add histograms have small accumulation error.
    # Keep the original masses; record the observed error instead of renormalizing.
    max_distance_mass_error = float(np.abs(distance.sum(-1)-1).max())
    np.testing.assert_allclose(distance.sum(-1), 1, atol=1e-4)
    np.testing.assert_allclose(position.sum(-1), 1, atol=2e-5)
    for prefix, stop in [('below80', 3), ('below320', 5)]:
        heads[prefix + '_mass'] = distance[..., :stop].sum(-1)
        heads[prefix + '_uniform_mass'] = distance_ref[..., :stop].sum(-1)
    heads['top10_excess'] = heads['top10pct_mass'] - heads['uniform_top10pct_mass']
    uniform_distance = np.array([r['uniform_distance_ms'] for r in records])
    heads['distance_excess_ms'] = heads['mean_distance_ms'] - uniform_distance[:, None, None]
    duration = np.array([r['duration_s'] for r in records])
    multi = np.array([r['chunks'] > 1 for r in records])
    budgets = np.array([r['virtual_budget'] for r in records])
    write_csv(out / 'sample_groups.csv', records)
    counts = Counter((r['major'], r['domain'], r['task_name']) for r in records)
    write_csv(out / 'task_mapping.csv', [dict(major=m, domain=d, task_name=t, n=count)
              for (m, d, t), count in sorted(counts.items())])

    group_masks = [('All', np.ones(n, bool))]
    for major in MAJORS:
        m = np.array([r['major'] == major for r in records])
        group_masks.append((major + '/Avg', m))
        for domain in DOMAINS:
            group_masks.append((major + '/' + domain, m & np.array([r['domain'] == domain for r in records])))
    groups, layer_rows, head_rows, group_rows, boundary_rows = {}, [], [], [], []
    for name, ix in group_masks:
        layer_stats = []
        for li, layer in enumerate(layers):
            l = {'layer': layer}
            for key in heads:
                l[key] = float(np.nanmean(heads[key][ix, li]))
            l.update({f'{space}_adjacent_cosine': float(similarities[ix, li, si].mean())
                      for si, space in enumerate(['k', 'v', 'hidden'])})
            layer_rows.append({'group': name, 'n': int(ix.sum()), **l})
            l['distance_hist'] = distance[ix, li].mean((0, 1), dtype=np.float64).tolist()
            l['distance_uniform'] = distance_ref[ix, li].mean((0, 1), dtype=np.float64).tolist()
            l['position_hist'] = position[ix, li].mean((0, 1), dtype=np.float64).tolist()
            l['position_uniform'] = position_ref[ix, li].mean(0, dtype=np.float64).tolist()
            layer_stats.append(l)
        all_heads, selected_heads = [], []
        for li, layer in enumerate(layers):
            for h in range(head_count):
                hd = {'layer': layer, 'head': h}
                for key in heads:
                    hd[key] = float(np.nanmean(heads[key][ix, li, h]))
                q = np.quantile(heads['peak_r'][ix, li, h], [.1, .5, .9])
                hd.update(dict(zip(['peak_r_q10', 'peak_r_median', 'peak_r_q90'], q.tolist())))
                head_rows.append({'group': name, 'n': int(ix.sum()), **hd})
                hd['position_hist'] = position[ix, li, h].mean(0, dtype=np.float64).tolist()
                hd['position_uniform'] = position_ref[ix, li].mean(0, dtype=np.float64).tolist()
                all_heads.append(hd)
        for layer, h in SELECTED:
            selected_heads.append(next(x for x in all_heads if x['layer'] == layer and x['head'] == h))
        ci_names = ['accuracy', 'l31_top10_mass', 'l31_top10_excess', 'l31_peak_r',
                    'l31_distance_ms', 'l31_below80_mass', 'l0h15_peak_start',
                    'topk_minus_random_coverage', 'segmentwise_minus_topk_coverage']
        ci_arrays = [accuracy, heads['top10pct_mass'][:, -1].mean(-1),
                     heads['top10_excess'][:, -1].mean(-1), heads['peak_r'][:, -1].mean(-1),
                     heads['mean_distance_ms'][:, -1].mean(-1), heads['below80_mass'][:, -1].mean(-1),
                     heads['peak_start_weight'][:, 0, 15], virtual[:, 0, 0]-virtual[:, 1, 0],
                     virtual[:, 2, 0]-virtual[:, 0, 0]]
        ci = dict(zip(ci_names, boot_columns(np.stack(ci_arrays, axis=1)[ix])))
        boundary = []
        for layer, h in SELECTED[:3]:
            li = layers.index(layer)
            first = heads['first_chunk_start_r'][ix & multi, li, h]
            later = heads['later_chunks_start_r'][ix & multi, li, h]
            bci = boot_columns(np.stack([first, later, first-later], axis=1))
            b = {'layer': layer, 'head': h, 'n': len(first),
                 'first': bci[0], 'later': bci[1], 'first_minus_later': bci[2]}
            boundary.append(b)
            boundary_rows.append({'group': name, 'layer': layer, 'head': h, 'n': len(first),
                **{f'{part}_{k}': (v['mean'] if k == 'mean' else v['ci95'][0 if k=='ci_low' else 1])
                   for part, v in zip(['first', 'later', 'difference'], bci)
                   for k in ['mean', 'ci_low', 'ci_high']}})
        vstats = {}
        for subset, take in [('all', ix), ('budget_at_least_10', ix & (budgets >= 10))]:
            vstats[subset] = {'n': int(take.sum()),
                'means': {method: dict(zip(VKEYS, virtual[take, mi].mean(0).tolist()))
                          for mi, method in enumerate(METHODS)}}
        group = {'n': int(ix.sum()), 'n_tasks': len({r['task_name'] for r, take in zip(records, ix) if take}),
            'correct': int(accuracy[ix].sum()), 'accuracy': float(accuracy[ix].mean()),
            'duration_mean_s': float(duration[ix].mean()), 'duration_median_s': float(np.median(duration[ix])),
            'uniform_distance_ms': float(uniform_distance[ix].mean()),
            'mean_chunk_count': float(np.mean([r['chunks'] for r, take in zip(records, ix) if take])),
            'multi_chunk_n': int((ix & multi).sum()), 'layers': layer_stats,
            'heads': all_heads, 'selected_heads': selected_heads, 'ci': ci,
            'boundary': boundary, 'virtual': vstats}
        groups[name] = group
        flat = {k: v for k, v in group.items() if not isinstance(v, (dict, list))}
        for key, c in ci.items():
            flat.update({key+'_mean': c['mean'], key+'_ci_low': c['ci95'][0], key+'_ci_high': c['ci95'][1]})
        group_rows.append({'group': name, **flat})
        print('Aggregated', name, 'n=', group['n'], flush=True)
    for major in MAJORS:
        children = [groups[f'{major}/{d}'] for d in DOMAINS]
        parent = groups[major+'/Avg']
        assert sum(c['n'] for c in children) == parent['n']
        assert sum(c['correct'] for c in children) == parent['correct']
        for li in range(layer_count):
            weighted = sum(c['layers'][li]['top10pct_mass']*c['n'] for c in children) / parent['n']
            np.testing.assert_allclose(weighted, parent['layers'][li]['top10pct_mass'], rtol=1e-12)
    write_csv(out / 'group_summary.csv', group_rows)
    write_csv(out / 'layer_metrics.csv', layer_rows)
    write_csv(out / 'head_metrics.csv', head_rows)
    write_csv(out / 'boundary_metrics.csv', boundary_rows)
    result = {'run': str(run), 'metadata_dir': str(dataset_dir), 'n': n, 'layers': layers,
              'heads_per_layer': head_count, 'observed_layer_head_pairs': layer_count*head_count,
              'taxonomy_source': taxonomy, 'groups': groups,
              'bootstrap': {'repeats': 1000, 'seed': 42, 'unit': 'distinct audio_path record',
                            'interval': 'pointwise percentile 95%; no multiple-comparison correction'},
              'checks': {'id_join_complete': True, 'task_mapping_complete': True,
                         'category_consistent': True, 'no_model_inference': True,
                         'max_distance_mass_sum_error': max_distance_mass_error,
                         'parent_child_weighted_means_match': True}}
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return result


def plot(result, out):
    figdir = out / 'figures'; figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 12, 'axes.spines.top': False, 'axes.spines.right': False,
        'savefig.dpi': 180, 'pdf.fonttype': 42})
    gs = result['groups']
    layers = result['layers']
    layer_ticks = layers[::max(1, len(layers)//8)]
    if layer_ticks[-1] != layers[-1]:
        layer_ticks.append(layers[-1])

    def save(fig, name):
        for ext in ['png', 'pdf']:
            fig.savefig(figdir / f'{name}.{ext}', bbox_inches='tight', facecolor='white')
        plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout='constrained')
    for col, major in enumerate(MAJORS):
        names = ['Avg'] + DOMAINS
        groups = [gs[major+'/'+d] for d in names]
        for row, (metric, title) in enumerate([('accuracy', 'Task accuracy'),
                                              ('l31_top10_mass', 'L31 top-10% incoming mass')]):
            ax = axes[row, col]
            means = np.array([g['ci'][metric]['mean'] for g in groups])
            bounds = np.array([g['ci'][metric]['ci95'] for g in groups])
            ax.bar(range(4), means, color=['#777777']+COLORS, width=.65,
                   yerr=np.stack([means-bounds[:, 0], bounds[:, 1]-means]), capsize=4)
            for i, v in enumerate(means):
                annotation = f'{v:.2%}' + (f"\nn={groups[i]['n']:,}" if row == 0 else '')
                ax.text(i, bounds[i, 1]+.02, annotation, ha='center', fontsize=9)
            ax.set(xticks=range(4), xticklabels=names, ylim=(0, 1.08),
                   title=f'{major} | {title}')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            if row == 1:
                ref = [g['layers'][-1]['uniform_top10pct_mass'] for g in groups]
                ax.scatter(range(4), ref, marker='_', s=250, color='black', label='Group-specific uniform reference')
                ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('MMSU | two major groups and six subgroups | 95% record-bootstrap intervals', fontsize=14)
    save(fig, '01_group_overview')

    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for row, major in enumerate(MAJORS):
        for domain, color in zip(DOMAINS + ['Avg'], COLORS + ['#222222']):
            stats = gs[major+'/'+domain]['layers']
            for col, (key, ylabel) in enumerate([('peak_r', 'Mean chunk peak r'),
                    ('top10_excess', 'Top-10% mass minus uniform (pp)'),
                    ('incoming_entropy', 'Normalized incoming entropy')]):
                scale = 100 if key=='top10_excess' else 1
                axes[row, col].plot(layers, [x[key]*scale for x in stats],
                    marker='o', color=color, ls='--' if domain=='Avg' else '-', label=domain)
                axes[row, col].set(xticks=layer_ticks, xlabel='Encoder layer', ylabel=ylabel,
                                   title=f'{major} | {ylabel}')
        axes[row, 0].set_ylim(bottom=0); axes[row, 0].axhline(1, color='gray', lw=1, ls=':')
        axes[row, 1].set_ylim(bottom=0); axes[row, 2].set_ylim(0, 1)
        axes[row, 2].legend(frameon=False, fontsize=9, loc='lower left')
    fig.suptitle(f'Concentration across {len(layers)} observed encoder layers | equal record weights', fontsize=14)
    save(fig, '02_attention_mass_layers')

    group_names = [m+'/'+d for m in MAJORS for d in ['Avg']+DOMAINS]
    row_labels = [name.replace('/', ' / ') for name in group_names]
    col_labels = [f'L{l}\nH{h}' for l, h in SELECTED]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.1), layout='constrained')
    for ax, (key, title, vmax, pct) in zip(axes, [
            ('peak_r', 'A  Peak incoming r', None, False),
            ('peak_start_weight', 'B  Peak at exact chunk start (%)', 100, True),
            ('peak_query_above_uniform', 'C  Queries reading peak above uniform (%)', 100, True)]):
        values = np.array([[h[key] for h in gs[name]['selected_heads']] for name in group_names])
        if pct: values *= 100
        if vmax is None: vmax = max(1, float(values.max()))
        im = ax.imshow(values, cmap='YlGnBu', vmin=0, vmax=vmax, aspect='auto')
        ax.set(xticks=range(len(SELECTED)), xticklabels=col_labels, yticks=range(len(group_names)),
               yticklabels=row_labels if ax is axes[0] else [], title=title)
        ax.axhline(3.5, color='black', lw=1.5)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f'{values[i,j]:.1f}', ha='center', va='center', fontsize=8,
                        color='white' if values[i,j] > vmax*.60 else 'black')
        fig.colorbar(im, ax=ax, shrink=.65)
    fig.suptitle('The same selected heads in every group | exploratory head selection, no sink threshold', fontsize=14)
    save(fig, '03_sink_heads')

    distance_excess = {m: np.array([np.array(gs[m+'/'+d]['layers'][-1]['distance_hist']) -
                       gs[m+'/'+d]['layers'][-1]['distance_uniform'] for d in DOMAINS])*100 for m in MAJORS}
    distance_limit = max(1, max(float(np.abs(v).max()) for v in distance_excess.values()))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout='constrained')
    for col, major in enumerate(MAJORS):
        for domain, color in zip(DOMAINS+['Avg'], COLORS+['#222222']):
            stats = gs[major+'/'+domain]['layers']
            axes[0, col].plot(layers, [x['mean_distance_ms'] for x in stats], marker='o',
                color=color, ls='--' if domain=='Avg' else '-', label=domain)
        axes[0, col].set(xticks=layer_ticks, xlabel='Encoder layer', ylabel='Mean absolute distance (ms)',
                         title=major+' | within-chunk reading distance')
        axes[0, col].set_ylim(bottom=0)
        axes[0, col].legend(frameon=False, fontsize=9)
        excess = distance_excess[major]
        ax = axes[1, col]
        im = ax.imshow(excess, cmap='RdBu_r', vmin=-distance_limit, vmax=distance_limit, aspect='auto')
        ax.set(yticks=range(3), yticklabels=DOMAINS, xticks=range(9),
               xticklabels=['0–20','20–40','40–80','80–160','160–320','320–640','640–1280','1280–2560','≥2560'],
               xlabel='Absolute query–key distance bin (ms)', title=major+' | L31 distance mass minus uniform')
        ax.tick_params(axis='x', labelrotation=40)
        for i in range(3):
            for j in range(9):
                ax.text(j, i, f'{excess[i,j]:+.1f}', ha='center', va='center', fontsize=8,
                        color='white' if abs(excess[i,j])>distance_limit*.65 else 'black')
    fig.colorbar(im, ax=list(axes[1]), label='Percentage points', shrink=.75)
    fig.suptitle('Reading range is limited by the native ~2 s audio-encoder chunks', fontsize=14)
    save(fig, '04_attention_distance')

    boundary_high = max(b[key]['ci95'][1] for m in MAJORS for d in DOMAINS
                        for b in gs[m+'/'+d]['boundary'] for key in ('first', 'later'))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), layout='constrained')
    for row, major in enumerate(MAJORS):
        for col, domain in enumerate(DOMAINS):
            ax = axes[row, col]; g = gs[major+'/'+domain]
            xx = np.arange(3)
            for key, offset, color, label in [('first', -.18, BLUE, 'First chunk'),
                                              ('later', .18, ORANGE, 'Later chunks')]:
                means = np.array([b[key]['mean'] for b in g['boundary']])
                bounds = np.array([b[key]['ci95'] for b in g['boundary']])
                ax.bar(xx+offset, means, .34, color=color, label=label,
                    yerr=np.stack([means-bounds[:,0], bounds[:,1]-means]), capsize=3)
            ax.axhline(1, color='gray', ls=':', lw=1)
            ax.set(xticks=xx, xticklabels=['L0 H15','L23 H1','L23 H6'], ylim=(0, max(2, boundary_high*1.1)),
                ylabel='Incoming r at the first audio frame',
                title=f"{major} / {domain}\nMatched multi-chunk records: n={g['multi_chunk_n']:,}")
    axes[0, 0].legend(frameon=False, fontsize=9)
    fig.suptitle('Boundary attraction repeats in later chunks | 95% record-bootstrap intervals', fontsize=14)
    save(fig, '05_repeated_chunk_boundaries')

    # Every measured layer/head, split into the six task groups; no head sampling.
    for key, label, filename in [
            ('top10_excess', 'Top-10% mass minus uniform', '06_all_heads_mass'),
            ('peak_r', 'Mean peak incoming r', '07_all_heads_peak_r'),
            ('peak_start_weight', 'Peak at exact chunk start', '08_all_heads_start')]:
        matrices = {m+'/'+d: np.array([h[key] for h in gs[m+'/'+d]['heads']]).reshape(
                    len(layers), result['heads_per_layer']) for m in MAJORS for d in DOMAINS}
        high = max(float(v.max()) for v in matrices.values()) if key=='peak_r' else 1
        fig, axes = plt.subplots(2, 3, figsize=(15, max(9, .65*len(layers))), layout='constrained')
        for ri, major in enumerate(MAJORS):
            for ci, domain in enumerate(DOMAINS):
                ax = axes[ri, ci]
                im = ax.imshow(matrices[major+'/'+domain], origin='lower', aspect='auto',
                               cmap='viridis', vmin=0, vmax=high)
                ax.set(title=major+' / '+domain, xlabel='Head', ylabel='Encoder layer',
                       xticks=range(result['heads_per_layer']), yticks=range(len(layers)), yticklabels=layers)
                ax.tick_params(labelsize=8)
        fig.colorbar(im, ax=list(axes.flat), label=label, shrink=.6)
        fig.suptitle(f'{label} | all {len(layers)} layers × {result["heads_per_layer"]} heads', fontsize=14)
        save(fig, filename)


def write_tables(result, out):
    groups = result['groups']
    names = [m+'/'+d for m in MAJORS for d in ['Avg']+DOMAINS]
    lines = ['# 六组离线分析：完整数值表', '',
             f'观测层：{result["layers"]}；每层 {result["heads_per_layer"]} 个 head。',
             '由 scripts.analysis.audio_sink.group_report 从保存的观测生成；不加载模型。', '',
             '## 用户指定的八列表格', '',
             '| 指标 | Perception Avg | P Semantic | P Phonology | P Paralinguistic | Reasoning Avg | R Semantic | R Phonology | R Paralinguistic |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    specs = [('样本数', lambda g: str(g['n'])), ('准确率', lambda g: f"{g['accuracy']*100:.2f}%"),
             ('平均音频时长 / s', lambda g: f"{g['duration_mean_s']:.2f}"),
             ('L31 Top 10% mass', lambda g: f"{g['layers'][-1]['top10pct_mass']*100:.2f}%"),
             ('Top 10% 均匀参考', lambda g: f"{g['layers'][-1]['uniform_top10pct_mass']*100:.2f}%"),
             ('L31 mass − 均匀参考 / pp', lambda g: f"{g['layers'][-1]['top10_excess']*100:.2f}"),
             ('L31 peak r', lambda g: f"{g['layers'][-1]['peak_r']:.2f}"),
             ('L31 incoming entropy', lambda g: f"{g['layers'][-1]['incoming_entropy']:.3f}"),
             ('L31 读取距离 / ms', lambda g: f"{g['layers'][-1]['mean_distance_ms']:.2f}"),
             ('均匀读取距离 / ms', lambda g: f"{g['uniform_distance_ms']:.2f}"),
             ('L31 距离 − 均匀参考 / ms', lambda g: f"{g['layers'][-1]['distance_excess_ms']:.2f}"),
             ('L31 <80 ms mass', lambda g: f"{g['layers'][-1]['below80_mass']*100:.2f}%"),
             ('L31 <320 ms mass', lambda g: f"{g['layers'][-1]['below320_mass']*100:.2f}%"),
             ('L0–H15 块首峰值比例', lambda g: f"{g['selected_heads'][0]['peak_start_weight']*100:.2f}%")]
    for label, fn in specs:
        lines.append('| '+label+' | '+' | '.join(fn(groups[n]) for n in names)+' |')
    lines += ['', '## 核心指标的 95% bootstrap 区间', '',
              '| 组别 | 准确率 / % | L31 Top 10% mass / % | L31 距离 / ms |', '|---|---:|---:|---:|']
    def ci(g, key, scale):
        c=g['ci'][key]; lo,hi=c['ci95']
        return f"{c['mean']*scale:.2f} [{lo*scale:.2f}, {hi*scale:.2f}]"
    for name in names:
        g=groups[name]
        lines.append(f"| {name} | {ci(g,'accuracy',100)} | {ci(g,'l31_top10_mass',100)} | {ci(g,'l31_distance_ms',1)} |")
    lines += ['', '## 逐组 head 证据', '',
              '| 组别 | Head | peak r | Top 10% mass | 块首峰值 | 读取峰值超过均匀的 query |',
              '|---|---|---:|---:|---:|---:|']
    for name in names:
        for h in groups[name]['selected_heads']:
            lines.append(f"| {name} | L{h['layer']}–H{h['head']} | {h['peak_r']:.2f} | {h['top10pct_mass']:.2%} | {h['peak_start_weight']:.2%} | {h['peak_query_above_uniform']:.2%} |")
    lines += ['', '## 辅助观察：相邻表示与虚拟 TopK', '',
              '| 组别 | L31 K 相邻 cosine | L31 V 相邻 cosine | L31 hidden 相邻 cosine | TopK 覆盖 | 随机覆盖 | Segmentwise 覆盖 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name in names:
        g=groups[name]; l=g['layers'][-1]; v=g['virtual']['all']['means']
        lines.append(f"| {name} | {l['k_adjacent_cosine']:.3f} | {l['v_adjacent_cosine']:.3f} | {l['hidden_adjacent_cosine']:.3f} | {v['topk']['segment_coverage']:.2%} | {v['random']['segment_coverage']:.2%} | {v['segmentwise']['segment_coverage']:.2%} |")
    lines += ['', '虚拟选点不改变模型输入；不同组的长度与预算不同，覆盖率差异不能直接解释为语义损失。',
              '更多字段、逐层和全部 head 见 CSV 与 summary.json；所有置信区间为探索性的逐项区间，未作多重比较校正。']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--dataset-dir', type=Path, default=os.environ.get('MMSUFINAL_DATASET_DIR'))
    parser.add_argument('--plot-only', action='store_true', help='Redraw figures from the saved summary without rereading observations')
    args = parser.parse_args()
    if not args.plot_only and args.dataset_dir is None:
        parser.error('Set --dataset-dir or MMSUFINAL_DATASET_DIR for official ID/task labels')
    out = args.run_dir.resolve()/'six_group_analysis'
    out.mkdir(exist_ok=True)
    result = json.loads((out/'summary.json').read_text()) if args.plot_only else aggregate(args.run_dir, args.dataset_dir, out)
    plot(result, out)
    write_tables(result, out)
    print('Finished offline analysis:', out, flush=True)


if __name__ == '__main__':
    main()
