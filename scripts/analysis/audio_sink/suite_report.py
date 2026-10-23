"""Paired full-B reports; stream condition records rather than retaining their JSON."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from .metrics import jsonable
from .report import coverage_report, _save


def lines(directory, pattern):
    for path in sorted(Path(directory).glob(pattern)):
        with path.open(encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)


def condition_coverage(directory, reference, plan):
    ids = {r['id']: r['doc_index'] for r in reference}
    names = {c['id']: i for i, c in enumerate(plan['conditions'])}
    seen = np.zeros((len(names), len(reference)), dtype=bool)
    duplicates = np.zeros(len(names), dtype=int)
    for row in lines(directory, 'conditions.rank*.jsonl'):
        name, index = row['condition_id'], row['doc_index']
        if name not in names or row['id'] not in ids or ids[row['id']] != index or not 0 <= index < len(reference):
            raise ValueError('Unknown condition/sample or changed sample pairing')
        c = names[name]
        duplicates[c] += int(seen[c, index])
        seen[c, index] = True
    return {name: dict(observed=int(seen[c].sum()), expected=len(reference), duplicates=int(duplicates[c]),
                       complete=bool(seen[c].all() and not duplicates[c]))
            for name, c in names.items()}


def paired_intervals(values, groups, repeats=1000, seed=42):
    """Common source-group bootstrap draws, matrix multiplication in bounded chunks."""
    values = np.asarray(values, dtype=float)
    unique, group_ids = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    sums = np.zeros((len(values), len(unique)))
    counts = np.zeros_like(sums)
    finite = np.isfinite(values)
    row_ids = np.arange(len(values))[:, None]
    np.add.at(sums, (row_ids, group_ids[None, :]), np.where(finite, values, 0))
    np.add.at(counts, (row_ids, group_ids[None, :]), finite)
    totals = counts.sum(1)
    mean = np.divide(sums.sum(1), totals, out=np.full(len(values), np.nan), where=totals>0)
    lo, hi = np.full(len(values), np.nan), np.full(len(values), np.nan)
    if len(unique) < 2:
        return mean, lo, hi
    weights = np.random.default_rng(seed).multinomial(len(unique), np.full(len(unique), 1/len(unique)), size=repeats).T
    for start in range(0, len(values), 64):
        numerator, denominator = sums[start:start+64] @ weights, counts[start:start+64] @ weights
        samples = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator>0)
        for j in range(len(samples)):
            if (counts[start+j]>0).sum() < 2:
                continue
            valid = samples[j][np.isfinite(samples[j])]
            lo[start+j], hi[start+j] = np.quantile(valid, [.025, .975])
    return mean, lo, hi


def summarize(directory, *, repeats=1000, allow_smoke=False, group_field='audio_path'):
    directory = Path(directory)
    settings = json.loads((directory/'observation.json').read_text())
    plan = json.loads((directory/'suite_plan.json').read_text())
    reference, distance_sum, distance_uniform, kernels = [], None, None, []
    shadow_checks = []
    for row in lines(directory, 'samples.rank*.jsonl'):
        layers = row.pop('layers')
        shadow_checks.extend(l['shadow_check'] for l in layers if l.get('shadow_check') is not None)
        distance = np.asarray([l['heads']['distance_mass'] for l in layers])
        uniform = np.asarray([l['heads']['distance_uniform_mass'] for l in layers])
        distance_sum = distance.copy() if distance_sum is None else distance_sum+distance
        distance_uniform = uniform.copy() if distance_uniform is None else distance_uniform+uniform
        if 'kernel_benchmark' in row:
            kernels.append(dict(doc_index=row['doc_index'], **row.pop('kernel_benchmark')))
        reference.append(row)
    reference.sort(key=lambda r: r['doc_index'])
    coverage = coverage_report(reference, settings['expected_samples'])
    if not coverage['complete']:
        raise ValueError(f'Incomplete reference B: {coverage}')
    if len(reference) != 5000 and not allow_smoke:
        raise ValueError('Full-B report requires 5,000 samples; --allow-smoke labels a development test')
    n, configs = len(reference), plan['conditions']
    cids = {c['id']: i for i, c in enumerate(configs)}
    seen = np.zeros((len(configs), n), dtype=bool)
    scores = np.full((len(configs), n), np.nan, dtype=np.float32)
    seconds = np.full_like(scores, np.nan)
    drift = np.full_like(scores, np.nan)
    incoming = np.full_like(scores, np.nan)
    kept = np.full_like(scores, np.nan)
    connection_fraction = np.full_like(scores, np.nan)
    changed = np.zeros_like(seen)
    active = np.zeros_like(seen)
    following_sum, following_count = {}, {}
    probe_loss = None
    if plan['probe_weights']:
        attributes = reference[0]['probe']['attributes']
        probe_loss = np.full((len(configs), n, len(attributes)), np.nan, dtype=np.float32)
    for row in lines(directory, 'conditions.rank*.jsonl'):
        c, i = cids[row['condition_id']], row['doc_index']
        if not 0 <= i < n or reference[i]['id'] != row['id'] or seen[c, i]:
            raise ValueError('Duplicate or unpaired condition record')
        seen[c, i] = True
        scores[c, i] = row['accuracy'] if row['labels_preserved'] else row['original_label_agreement']
        seconds[c, i] = row['diagnostic_generation_seconds']
        changed[c, i] = row['prediction'] != reference[i]['prediction']
        details = row['intervention']
        active[c, i] = details.get('active', True)
        drift[c, i] = details.get('pooled_representation_relative_l2', np.nan)
        incoming[c, i] = details.get('selected_mean_r') if details.get('selected_mean_r') is not None else np.nan
        kept[c, i] = details.get('projector_tokens', np.nan)
        connection_fraction[c, i] = details.get('connection_keep_fraction', np.nan)
        if 'following' in details:
            for key in ('position_correlation', 'content_correlation'):
                data = np.asarray([v[key] for v in details['following'].values()], dtype=float)
                if data.ndim != 2:
                    continue
                name = f'{row["condition_id"]}/{key}'
                if name not in following_sum:
                    following_sum[name], following_count[name] = np.zeros_like(data), np.zeros_like(data)
                valid = np.isfinite(data)
                following_sum[name] += np.where(valid, data, 0)
                following_count[name] += valid
        if probe_loss is not None:
            probe_loss[c, i] = row['probe']['squared_error']
    if not seen.all():
        missing = {configs[c]['id']: int((~seen[c]).sum()) for c in range(len(configs)) if not seen[c].all()}
        raise ValueError(f'Conditions missing samples; no reduced-denominator results: {missing}')
    baseline = np.array([r['accuracy'] for r in reference])
    if any(r.get(group_field) is None for r in reference):
        raise ValueError(f'Missing bootstrap grouping field: {group_field}')
    groups = [r[group_field] for r in reference]
    differences = scores-baseline
    valid_labels = np.array([c['family'] != 'input' for c in configs])
    differences[~valid_labels] = np.nan
    means, low, high = paired_intervals(differences, groups, repeats, plan['seed'])
    stats = []
    for c, config in enumerate(configs):
        record = dict(config, samples=n, full_B=n==5000,
            accuracy=float(scores[c].mean()) if valid_labels[c] else None,
            original_label_agreement=float(scores[c].mean()) if not valid_labels[c] else None,
            accuracy_delta=means[c], ci95_low=low[c], ci95_high=high[c],
            active_samples=int(active[c].sum()),
            changed_predictions=int(changed[c].sum()), outputs_identical=bool(not changed[c].any()),
            diagnostic_seconds_mean=float(seconds[c].mean()),
            mean_projector_tokens=float(kept[c].mean()),
            pooled_representation_drift=float(np.nanmean(drift[c])),
            connection_keep_fraction=(float(np.nanmean(connection_fraction[c]))
                if np.isfinite(connection_fraction[c]).any() else None),
            selected_mean_r=float(np.nanmean(incoming[c])) if np.isfinite(incoming[c]).any() else None)
        if valid_labels[c]:
            record['transitions'] = {
                'correct_to_wrong': int(((baseline==1) & (scores[c]==0)).sum()),
                'wrong_to_correct': int(((baseline==0) & (scores[c]==1)).sum()),
                'both_correct': int(((baseline==1) & (scores[c]==1)).sum()),
                'both_wrong': int(((baseline==0) & (scores[c]==0)).sum()),
            }
        stats.append(record)
    taxonomy = json.loads(Path(__file__).with_name('task_taxonomy.json').read_text())
    task_groups = {task: f'{major}/{domain}' for major, domains in taxonomy['groups'].items()
                   for domain, tasks in domains.items() for task in tasks}
    for row in reference:
        row['task_group'] = task_groups.get(row['task_name'], 'Unknown')
        if row['task_group'] == 'Unknown' and n == 5000:
            raise ValueError(f'Unmapped MMSU task: {row["task_name"]}')
        if row['task_group'] != 'Unknown' and row['task_group'].split('/')[0].lower() != row['question_type'].lower():
            raise ValueError(f'MMSU category disagrees with official taxonomy: {row["task_name"]}')
    strata = {}
    for field in ('question_type', 'sub_category', 'task_name', 'task_group'):
        categories = sorted({str(r.get(field) or 'Unknown') for r in reference})
        for category in categories:
            indices = [i for i, r in enumerate(reference) if str(r.get(field) or 'Unknown') == category]
            strata[f'{field}:{category}'] = dict(n=len(indices), baseline=float(baseline[indices].mean()),
                delta=[float(differences[c, indices].mean()) if valid_labels[c] else None for c in range(len(configs))])
    contrasts, contrast_ids = [], []
    for config in configs:
        if config['family'] == 'head' and config['operation'] == 'zero' and config['random_control_heads']:
            c = cids[config['id']]
            controls = [cids[f'head_l{config["layer"]}_h{h}_zero'] for h in config['random_control_heads']]
            contrasts.append(scores[c]-scores[controls].mean(0))
            contrast_ids.append(config['id'])
    random_comparison = {}
    if contrasts:
        mean, lo, hi = paired_intervals(contrasts, groups, repeats, plan['seed'])
        random_comparison = {name: dict(delta_vs_random=mean[i], ci95_low=lo[i], ci95_high=hi[i])
                             for i, name in enumerate(contrast_ids)}
    recovery = {}
    for name, c in cids.items():
        config = configs[c]
        if config['family'] == 'refill' and config['operation'] != 'topk':
            anchor = cids[f'refill_{config["keep_fraction"]:g}_topk']
            mean, lo, hi = paired_intervals((scores[c]-scores[anchor])[None, :], groups, repeats, plan['seed'])
            recovery[name] = dict(anchor=configs[anchor]['id'], accuracy_delta=mean[0], ci95_low=lo[0], ci95_high=hi[0])
    paired_names, paired_values = [], []
    for name, c in cids.items():
        config = configs[c]
        if config['family']=='refill' and config['operation']=='refill_low':
            other = cids[f'refill_{config["keep_fraction"]:g}_refill_random']
            paired_names.append(f'{name}_minus_random')
            paired_values.append(scores[c]-scores[other])
        if config['family']=='sink' and config['operation']=='key_mask':
            other = cids[f'sink_l{config["layer"]}_h{config["head"]}_{config["selector"]}_v_zero']
            paired_names.append(f'{name}_minus_v_zero')
            paired_values.append(scores[c]-scores[other])
        if config['family']=='sink' and config['selector']=='high' and config['operation'] in ('v_zero', 'v_norm', 'key_mask'):
            other = cids[f'sink_l{config["layer"]}_h{config["head"]}_random_{config["operation"]}']
            paired_names.append(f'{name}_minus_random')
            paired_values.append(scores[c]-scores[other])
        if config['family']=='evidence' and config['selector']=='low' and config['operation'] != 'restore':
            for control in ('high', 'random'):
                other = cids[f'evidence_{control}_{config["fraction"]:g}_{config["operation"]}']
                paired_names.append(f'{name}_minus_{control}')
                paired_values.append(scores[c]-scores[other])
        if config['family']=='range' and config['selector']=='window':
            other = cids[f'range_l{config["layer"]}_random_{config["window_ms"]:g}ms_{config["placement"]}']
            paired_names.append(f'{name}_minus_random')
            paired_values.append(scores[c]-scores[other])
    mechanism_comparisons = {}
    if paired_values:
        mean, lo, hi = paired_intervals(paired_values, groups, repeats, plan['seed'])
        mechanism_comparisons = {name: dict(accuracy_delta=mean[i], ci95_low=lo[i], ci95_high=hi[i])
                                 for i, name in enumerate(paired_names)}
    following = {name: np.divide(value, following_count[name], out=np.full_like(value, np.nan),
                                where=following_count[name]>0) for name, value in following_sum.items()}
    restore = [s for s in stats if s.get('operation') in ('restore', 'restore_v', 'restore_head')]
    summary = dict(coverage=coverage, conditions=stats, condition_count=len(configs), full_B=n==5000,
        baseline_accuracy=float(baseline.mean()), task_strata=strata, head_random_comparison=random_comparison,
        refill_comparison=recovery, following=following, kernels=kernels,
        paired_mechanism_comparisons=mechanism_comparisons, bootstrap_group_field=group_field,
        distance_mass=distance_sum/n, distance_uniform_mass=distance_uniform/n,
        distance_ms_edges=[0, 20, 40, 80, 160, 320, 640, 1280, 2560, None],
        shadow_validation=dict(layer_checks=len(shadow_checks),
            max_av_relative_l2=max((r['av_relative_l2'] for r in shadow_checks), default=None),
            max_av_abs_error=max((r['av_max_abs_error'] for r in shadow_checks), default=None)),
        observer_layers=plan['observer_layers'],
        restore_checks_passed=all(s['outputs_identical'] for s in restore) if restore else None,
        inference='Exploratory effects on the full MMSU set; no head calibration or held-out selection.',
        property_scope='MMSU task categories are task strata, not independently labelled acoustic probes.',
        cost_scope='Diagnostic generation timings include hooks; only isolated kernel timings compare kernel costs.')
    if probe_loss is not None:
        original = np.array([r['probe']['squared_error'] for r in reference])
        changes = probe_loss-original[None, :, :]
        mean, lo, hi = paired_intervals(changes.transpose(0, 2, 1).reshape(-1, n), groups, repeats, plan['seed'])
        summary['fixed_probe'] = dict(attributes=attributes,
            mean_squared_error_change=mean.reshape(len(configs), -1),
            ci95_low=lo.reshape(len(configs), -1), ci95_high=hi.reshape(len(configs), -1),
            note='Fixed pre-projector mean-pooling linear readout; no fitting on B. Input transformations change targets.')
    return summary


def render(summary, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42})
    def save(fig, name):
        if not summary['full_B']:
            fig.suptitle(f'Development smoke test (n={summary["coverage"]["observed_samples"]}); not full MMSU', fontsize=10)
        _save(fig, destination/name)
    stats = summary['conditions']
    layers = summary['observer_layers']
    selected = sorted(set([0, len(layers)//2, len(layers)-1]))
    distance = np.asarray(summary['distance_mass'])
    excess = distance - np.asarray(summary['distance_uniform_mass'])
    limit = max(.01, float(np.abs(excess).max()))
    fig, axes = plt.subplots(2, len(selected), figsize=(5*len(selected), 8), layout='constrained', squeeze=False)
    labels = ['0–20', '20–40', '40–80', '80–160', '160–320', '320–640', '640–1280', '1280–2560', '≥2560']
    for column, index in enumerate(selected):
        axis = axes[0, column]
        im = axis.imshow(distance[index], aspect='auto', vmin=0, vmax=1, cmap='viridis', origin='lower')
        axis.set(title=f'Layer {layers[index]}', ylabel='Head', xlabel='Absolute attention distance (ms)')
        axis.set_xticks(range(len(labels)), labels, rotation=65)
        axis.set_yticks(range(distance.shape[1]))
        fig.colorbar(im, ax=axis, label='Attention probability mass')
        axis = axes[1, column]
        im = axis.imshow(excess[index], aspect='auto', vmin=-limit, vmax=limit, cmap='coolwarm', origin='lower')
        axis.set(title=f'Layer {layers[index]} minus visibility-matched uniform', ylabel='Head', xlabel='Distance (ms)')
        axis.set_xticks(range(len(labels)), labels, rotation=65)
        axis.set_yticks(range(distance.shape[1]))
        fig.colorbar(im, ax=axis, label='Excess probability mass')
    save(fig, 'attention_distance')
    head_rows = [r for r in stats if r['family']=='head' and r['operation']=='zero']
    if head_rows:
        layer_ids = sorted({r['layer'] for r in head_rows})
        heads = max(r['head'] for r in head_rows)+1
        loss = np.full((len(layer_ids), heads), np.nan)
        for row in head_rows:
            loss[layer_ids.index(row['layer']), row['head']] = -row['accuracy_delta']
        limit = max(.01, np.nanmax(np.abs(loss)))
        fig, axis = plt.subplots(figsize=(10, max(3, len(layer_ids)*.23)), layout='constrained')
        im = axis.imshow(loss, cmap='coolwarm', vmin=-limit, vmax=limit, aspect='auto', origin='lower')
        axis.set(xlabel='Ablated head', ylabel='Encoder layer', title='Full-B accuracy loss after head ablation')
        axis.set_yticks(range(len(layer_ids)), layer_ids)
        axis.set_xticks(range(heads))
        fig.colorbar(im, ax=axis, label='Baseline accuracy minus ablated accuracy')
        save(fig, 'head_loss')
        attributes = [name for name in summary['task_strata'] if not name.startswith('task_group:')]
        for layer in layer_ids:
            data = np.array([[-summary['task_strata'][name]['delta'][stats.index(row)] for name in attributes]
                             for row in head_rows if row['layer']==layer])
            height = max(6, 3.8 + max(map(len, attributes))*.045)
            fig, axis = plt.subplots(figsize=(max(8, len(attributes)*.28), height), layout='constrained')
            extent = max(.01, np.nanmax(np.abs(data)))
            im = axis.imshow(data, aspect='auto', cmap='coolwarm', vmin=-extent, vmax=extent, origin='lower')
            axis.set_xticks(range(len(attributes)), attributes, rotation=75, ha='right', fontsize=7)
            axis.set_yticks(range(heads))
            axis.set(xlabel='MMSU task stratum (not an acoustic probe)', ylabel='Head', title=f'Layer {layer}: task-specific loss')
            fig.colorbar(im, ax=axis, label='Accuracy loss')
            save(fig, f'head_attributes_l{layer}')
        # Six panels retain all layer/head cells and share one color scale.
        attributes = [f'task_group:{major}/{domain}' for major in ('Perception', 'Reasoning')
                      for domain in ('Semantic', 'Phonology', 'Paralinguistic')]
        if all(name in summary['task_strata'] for name in attributes):
            matrices = []
            for name in attributes:
                data = np.full_like(loss, np.nan)
                for row in head_rows:
                    data[layer_ids.index(row['layer']), row['head']] = -summary['task_strata'][name]['delta'][stats.index(row)]
                matrices.append(data)
            limit = max(.01, float(np.nanmax(np.abs(matrices))))
            fig, axes = plt.subplots(2, 3, figsize=(15, max(9, .65*len(layer_ids))), layout='constrained')
            for axis, name, data in zip(axes.flat, attributes, matrices):
                im = axis.imshow(data, aspect='auto', origin='lower', cmap='coolwarm', vmin=-limit, vmax=limit)
                axis.set(title=f'{name.split(":", 1)[1]} (n={summary["task_strata"][name]["n"]})',
                         xlabel='Ablated head', ylabel='Encoder layer')
                axis.set_xticks(range(heads))
                axis.set_yticks(range(len(layer_ids)), layer_ids)
                axis.tick_params(labelsize=8)
            fig.colorbar(im, ax=list(axes.flat), label='Baseline accuracy minus ablated accuracy', shrink=.6)
            save(fig, 'head_loss_six_groups')
    range_rows = [r for r in stats if r['family']=='range']
    if range_rows:
        chosen = sorted({r['layer'] for r in range_rows})
        chosen = sorted(set([chosen[0], chosen[len(chosen)//2], chosen[-1]]))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout='constrained')
        for axis, placement in zip(axes, ('pre', 'post')):
            for layer in chosen:
                for selector, linestyle in [('window', '-'), ('random', '--')]:
                    rows = sorted([r for r in range_rows if r['layer']==layer and r['placement']==placement
                                   and r['selector']==selector], key=lambda r:r['window_ms'])
                    x = [r['window_ms'] for r in rows]
                    curve, = axis.plot(x, [r['accuracy_delta'] for r in rows],
                                       linestyle, marker='o', label=f'L{layer} {selector}')
                    axis.fill_between(x, [r['ci95_low'] for r in rows], [r['ci95_high'] for r in rows],
                                      color=curve.get_color(), alpha=.1)
            axis.axhline(0, color='0.5', lw=.7)
            axis.set(xlabel='Window radius (ms)', ylabel='Accuracy change on full B', title=f'{placement}-softmax intervention')
            axis.legend(fontsize=7)
            axis.set_xticks(sorted({r['window_ms'] for r in range_rows}))
        save(fig, 'range_performance')
    evidence = [r for r in stats if r['family']=='evidence']
    if evidence:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), layout='constrained')
        for axis, operation in zip(axes, ('mean', 'norm', 'delete')):
            for selector in ('low', 'high', 'random'):
                rows = sorted([r for r in evidence if r['operation']==operation and r['selector']==selector],
                              key=lambda r:r['fraction'])
                x = [r['fraction'] for r in rows]
                curve, = axis.plot(x, [r['accuracy_delta'] for r in rows], '-o', label=selector)
                axis.fill_between(x, [r['ci95_low'] for r in rows], [r['ci95_high'] for r in rows],
                                  color=curve.get_color(), alpha=.1)
            axis.axhline(0, color='0.5', lw=.7)
            axis.set(xlabel='Fraction of eligible blocks', ylabel='Full-B accuracy change', title=operation)
            axis.legend()
        save(fig, 'low_attention_evidence')
    sink = [r for r in stats if r['family']=='sink' and r['selected_mean_r'] is not None]
    if sink:
        fig, axis = plt.subplots(figsize=(7, 5), layout='constrained')
        for operation in ('v_zero', 'v_norm', 'key_mask'):
            rows = [r for r in sink if r['operation']==operation]
            axis.scatter([r['selected_mean_r'] for r in rows], [-r['accuracy_delta'] for r in rows], label=operation, alpha=.6)
        axis.set(xlabel='Selected incoming r (audio mean)', ylabel='Full-B accuracy loss',
                 title='Each point is a frozen layer/head/selection condition')
        axis.axhline(0, color='0.5', lw=.7)
        axis.axvline(1, color='0.5', lw=.7, ls='--')
        axis.set_xlim(0, max(2, max(r['selected_mean_r'] for r in sink)*1.05))
        axis.legend()
        save(fig, 'sink_function')
    if summary['following']:
        fig, axis = plt.subplots(figsize=(8, 4), layout='constrained')
        for name, value in summary['following'].items():
            data = np.asarray(value, dtype=float)
            if np.isfinite(data).any():
                counts = np.isfinite(data).sum(1)
                mean = np.divide(np.nansum(data, 1), counts, out=np.full(len(data), np.nan), where=counts>0)
                axis.plot(layers, mean, label=name)
        axis.set(xlabel='Layer', ylabel='Mean correlation over heads', ylim=(-1, 1),
                 title='Position/content following after feature-domain input changes')
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(.5, .5, 'Insufficient valid positions after boundary exclusion', ha='center', transform=axis.transAxes)
        save(fig, 'position_content_following')
    refill_rows = [r for r in stats if r['family']=='refill']
    if refill_rows:
        fig, axis = plt.subplots(figsize=(7, 4), layout='constrained')
        for operation in ('refill_low', 'refill_random'):
            rows = sorted([r for r in refill_rows if r['operation']==operation], key=lambda r:r['keep_fraction'])
            contrasts = [summary['refill_comparison'][r['id']] for r in rows]
            x = [r['keep_fraction'] for r in rows]
            curve, = axis.plot(x, [r['accuracy_delta'] for r in contrasts], '-o', label=operation)
            axis.fill_between(x, [r['ci95_low'] for r in contrasts], [r['ci95_high'] for r in contrasts],
                              color=curve.get_color(), alpha=.15)
        axis.axhline(0, color='0.5', lw=.7)
        axis.set(xlabel='Kept token fraction (same for all policies)', ylabel='Full-B accuracy change vs TopK',
                 title='Recovering discarded low-attention evidence at equal budget')
        axis.legend()
        save(fig, 'low_attention_refill')
    if head_rows and summary.get('fixed_probe'):
        probe = summary['fixed_probe']
        for layer in sorted({r['layer'] for r in head_rows}):
            chosen = [r for r in head_rows if r['layer']==layer]
            data = np.asarray([probe['mean_squared_error_change'][stats.index(r)] for r in chosen])
            limit = max(.01, float(np.nanmax(np.abs(data))))
            fig, axis = plt.subplots(figsize=(max(6, len(probe['attributes'])*.6), 5), layout='constrained')
            im = axis.imshow(data, aspect='auto', origin='lower', cmap='coolwarm', vmin=-limit, vmax=limit)
            axis.set_xticks(range(len(probe['attributes'])), probe['attributes'], rotation=45, ha='right')
            axis.set_yticks(range(len(chosen)), [r['head'] for r in chosen])
            axis.set(xlabel='Frozen probe attribute', ylabel='Ablated head', title=f'Layer {layer}: fixed-probe error change')
            fig.colorbar(im, ax=axis, label='MSE change vs native reference')
            save(fig, f'head_probe_l{layer}')
    measured = [item for item in summary['kernels'] if item['status']=='measured']
    if measured:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout='constrained')
        for backend in ('dense_masked', 'flash_attention_2_window'):
            grouped = defaultdict(list)
            for item in measured:
                for row in item['rows']:
                    if row['backend']==backend:
                        grouped[row['radius_ms']].append(row)
            windows = sorted(grouped)
            for axis, key, divisor, label in [
                (axes[0], 'median_ms', 1, 'Kernel median time (ms)'),
                (axes[1], 'peak_extra_allocated_bytes', 1024**2, 'Extra peak allocated memory (MiB)')]:
                axis.plot(windows, [np.mean([r[key] for r in grouped[w]])/divisor for w in windows],
                          '-o', label=backend)
                axis.set(xlabel='Window radius (ms)', ylabel=label, title='Actual QKV; isolated kernels only')
        for axis in axes:
            axis.legend(fontsize=8)
        full = [r for item in measured for r in item['rows'] if r['backend']=='flash_attention_2_full']
        if full:
            axes[0].axhline(np.mean([r['median_ms'] for r in full]), color='0.4', ls='--', label='FA2 full reference')
            axes[1].axhline(np.mean([r['peak_extra_allocated_bytes'] for r in full])/1024**2,
                           color='0.4', ls='--', label='FA2 full reference')
            for axis in axes:
                axis.legend(fontsize=8)
        save(fig, 'kernel_cost')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--bootstrap', type=int, default=1000)
    parser.add_argument('--allow-smoke', action='store_true')
    parser.add_argument('--group-field', default='audio_path')
    args = parser.parse_args()
    if args.bootstrap < 1:
        parser.error('--bootstrap must be positive')
    directory = Path(args.run_dir)
    summary = summarize(directory, repeats=args.bootstrap, allow_smoke=args.allow_smoke, group_field=args.group_field)
    destination = directory/'suite_figures'
    destination.mkdir(exist_ok=True)
    render(summary, destination)
    (directory/'suite_summary.json').write_text(json.dumps(jsonable(summary), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    columns = ['id', 'family', 'samples', 'full_B', 'active_samples', 'accuracy', 'accuracy_delta', 'ci95_low', 'ci95_high',
               'changed_predictions', 'pooled_representation_drift', 'diagnostic_seconds_mean']
    with (directory/'condition_results.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(jsonable(summary['conditions']))
    print(f'Wrote {directory/"suite_summary.json"}; full_B={summary["full_B"]}')


if __name__ == '__main__':
    main()
