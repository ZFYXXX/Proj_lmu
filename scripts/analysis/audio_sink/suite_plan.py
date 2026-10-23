"""Freeze a full-MMSU experiment plan from the checkpoint architecture."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
from .interventions import conditions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT_PATH'), required=not os.environ.get('CHECKPOINT_PATH'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--families', default='range,head,sink,evidence')
    parser.add_argument('--layers', default='all', help='Fixed layer IDs or all; never selected using B answers')
    parser.add_argument('--sink-heads', default='all', help='All heads by default; an a priori subset may reduce run size')
    parser.add_argument('--windows-ms', default='80,320,1280')
    parser.add_argument('--fractions', default='0.05,0.1,0.2')
    parser.add_argument('--block-ms', type=float, default=80)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--kernel-benchmark-samples', type=int, default=1, help='Per rank, actual reference QKV on CUDA')
    parser.add_argument('--probe-weights', help='Optional frozen NPZ linear regression readout')
    parser.add_argument('--probe-targets', help='JSON: sample id -> attribute name -> target')
    args = parser.parse_args()
    checkpoint = Path(args.checkpoint).expanduser().resolve()
    raw = json.loads((checkpoint / 'config.json').read_text())
    config = raw['thinker_config']['audio_config']
    num_layers, num_heads = config['encoder_layers'], config['encoder_attention_heads']
    layers = list(range(num_layers)) if args.layers == 'all' else sorted(set(map(int, args.layers.split(','))))
    sink_heads = list(range(num_heads)) if args.sink_heads == 'all' else sorted(set(map(int, args.sink_heads.split(','))))
    windows = [float(v) for v in args.windows_ms.split(',')]
    fractions = [float(v) for v in args.fractions.split(',')]
    families = args.families.split(',')
    if not set(families) <= {'range', 'head', 'sink', 'evidence'} or not families:
        parser.error('Unknown experiment family')
    if not layers or not set(layers) <= set(range(num_layers)) or not sink_heads or not set(sink_heads) <= set(range(num_heads)):
        parser.error('Layer/head index outside the actual checkpoint')
    if any(v < 0 for v in windows) or any(not 0 < v < 1 for v in fractions) or args.block_ms <= 0:
        parser.error('Invalid window/block/budget')
    if bool(args.probe_weights) != bool(args.probe_targets):
        parser.error('Provide both fixed probe weights and targets')
    if args.kernel_benchmark_samples < 0:
        parser.error('--kernel-benchmark-samples must be nonnegative')
    plan = dict(checkpoint=str(checkpoint), expected_samples=5000, seed=args.seed,
        layers=layers, encoder_layers=num_layers, encoder_heads=num_heads,
        observer_layers=sorted(set(layers + ([num_layers-1] if 'evidence' in families else []))),
        conditions=conditions(layers, num_heads, families=families, windows_ms=windows,
            sink_heads=sink_heads, fractions=fractions, block_ms=args.block_ms, seed=args.seed),
        kernel_benchmark_samples=args.kernel_benchmark_samples,
        probe_weights=str(Path(args.probe_weights).resolve()) if args.probe_weights else None,
        probe_targets=str(Path(args.probe_targets).resolve()) if args.probe_targets else None,
        head_selection='exhaustive within frozen layers; no calibration or test-answer selection',
        attribute_analysis='MMSU task strata; frozen regression probe only when supplied',
        input_transform_scores='original-label agreement only; labels not assumed preserved')
    count = len(plan['conditions'])
    plan['generation_calls'] = 5000 * (count+1)
    plan['conditions_by_family'] = {family: sum(c['family'] == family for c in plan['conditions'])
                                    for family in sorted({c['family'] for c in plan['conditions']})}
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)
    print(f'Frozen {count} conditions + native reference; {plan["generation_calls"]:,} generations on full MMSU.')
    print('All conditions use the same 5,000 samples. Head random controls reuse the exhaustive head runs.')


if __name__ == '__main__':
    main()
