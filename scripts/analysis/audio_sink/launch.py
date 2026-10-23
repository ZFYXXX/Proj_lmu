"""Launch native MMSU evaluation with observation or paired intervention workers."""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]


def worker_plan(plan, observation_path):
    argv = list(plan.argv)
    matches = [i for i in range(1, len(argv)) if argv[i-1] == '-m' and argv[i] == 'lmms_eval']
    if len(matches) != 1:
        raise ValueError('Expected exactly one lmms_eval worker module')
    argv[matches[0]] = 'scripts.analysis.audio_sink.worker'
    environment = dict(plan.environment)
    environment['AUDIO_SINK_OBSERVATION_CONFIG'] = str(observation_path)
    existing = environment.get('PYTHONPATH', os.environ.get('PYTHONPATH', ''))
    environment['PYTHONPATH'] = os.pathsep.join(filter(None, [str(ROOT),
        str(ROOT / 'third_party/frameworks/lmms-eval'), existing]))
    return replace(plan, argv=argv, environment=environment)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'scripts/analysis/audio_sink/eval_mmsufinal.yaml'))
    parser.add_argument('--checkpoint', default=os.environ.get('CHECKPOINT_PATH') or os.environ.get('QWEN25_OMNI_MODEL_PATH'))
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mode', choices=['observe', 'baseline', 'suite'], default='observe')
    parser.add_argument('--suite-plan', help='Frozen suite_plan JSON; required for --mode suite')
    parser.add_argument('--num-processes', type=int, default=4)
    parser.add_argument('--main-process-port', type=int, default=29649)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--layers', help='all or comma-separated zero-based audio encoder layers; default all')
    parser.add_argument('--example-doc-ids', default='0,1,2', help='Global MMSU doc indices; frozen before inspecting results')
    parser.add_argument('--example-chunk', type=int, default=0)
    parser.add_argument('--query-block', type=int, default=64)
    parser.add_argument('--check-av-samples', type=int, default=3, help='Shadow AV checks per worker')
    parser.add_argument('--virtual-keep-ratio', type=float, default=0.1)
    parser.add_argument('--segments', type=int, default=10)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error('Provide the exact full 7B checkpoint (base or merged LoRA)')
    if (args.mode == 'suite') != bool(args.suite_plan):
        parser.error('Use --mode suite and --suite-plan together')
    if args.mode == 'suite' and args.layers is not None:
        parser.error('Freeze suite layers in suite_plan, not in the launch command')
    if args.limit is not None and args.limit < 1:
        parser.error('--limit must be positive')
    if args.query_block < 1 or args.example_chunk < 0 or args.check_av_samples < 0:
        parser.error('Invalid query block, example chunk or check count')
    if not 0 < args.virtual_keep_ratio <= 1 or args.segments < 1:
        parser.error('Invalid virtual diagnostic budget')

    import yaml
    from mllm_comp.config.loader import load_config
    from mllm_comp.integrations.lmms_eval.adapter import LMMSEvalAdapter
    from mllm_comp.integrations.native_cli import run_native_cli
    from .metrics import jsonable

    raw = copy.deepcopy(load_config(args.config).raw)
    if raw['backbone']['name'] != 'qwen25_omni':
        parser.error('This observer targets qwen25_omni only')
    if raw.get('pipeline', {}).get('stages'):
        parser.error('Use a baseline config with pipeline.stages: []')
    if raw['eval']['tasks'] != ['mmsufinal']:
        parser.error('This analysis requires eval.tasks: [mmsufinal] in --config')
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = str(Path(args.checkpoint).expanduser().resolve())
    suite = json.loads(Path(args.suite_plan).read_text()) if args.suite_plan else None
    if suite and suite['checkpoint'] != checkpoint:
        parser.error('Suite checkpoint differs from launch checkpoint')
    if suite:
        raw['experiment']['seed'] = suite['seed']
    raw['backbone']['model_path'] = checkpoint
    raw['backbone'].pop('model_id', None)
    raw['backbone']['config']['enable_talker'] = False
    raw['pipeline'] = {'stages': []}
    raw['analysis'] = {}
    raw['eval'].update(compression_mode='none', target='baseline',
                       runtime_mode='default', checkpoint_path=checkpoint,
                       runtime_batch_size=1, num_processes=args.num_processes,
                       main_process_port=args.main_process_port, sample_limit=args.limit)
    eval_args = raw['eval'].setdefault('args', {})
    for field in ('use_cache', 'cache_requests', 'forward_only_probe'):
        eval_args.pop(field, None)
    if any('cache' in str(v) for v in eval_args.get('native_argv', [])):
        parser.error('Remove cache overrides from eval.args.native_argv for observation')
    eval_args['log_samples'] = True
    raw['output'] = {'run_dir': str(output), 'manifest_path': str(output / 'manifest.json')}
    config_path = output / 'eval.resolved.yaml'
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding='utf-8')
    settings = {
        'mode': args.mode, 'output_dir': str(output),
        'layers': None if args.layers in (None, 'all') else [int(v) for v in args.layers.split(',')],
        'example_doc_ids': [int(v) for v in args.example_doc_ids.split(',') if v.strip()],
        'example_chunk': args.example_chunk, 'query_block': args.query_block,
        'check_av_samples': args.check_av_samples, 'virtual_keep_ratio': args.virtual_keep_ratio,
        'segments': args.segments, 'seed': raw['experiment'].get('seed', 42),
        'expected_samples': min(args.limit, 5000) if args.limit is not None else 5000,
        'checkpoint': checkpoint, 'config': str(config_path),
    }
    if suite:
        settings['suite_plan'] = str(output / 'suite_plan.json')
        (output / 'suite_plan.json').write_text(json.dumps(suite, indent=2), encoding='utf-8')
    settings_path = output / 'observation.json'
    settings_path.write_text(json.dumps(settings, indent=2), encoding='utf-8')

    def command_runner(plan):
        observed = worker_plan(plan, settings_path)
        (output / 'launch.json').write_text(json.dumps({
            'argv': observed.argv, 'cwd': observed.cwd, 'mode': args.mode,
            'compression_mode': 'none', 'analysis_interventions': args.mode == 'suite',
        }, indent=2), encoding='utf-8')
        return run_native_cli(observed)

    adapter = LMMSEvalAdapter(command_runner=command_runner)
    result = adapter.evaluate(config=load_config(config_path), checkpoint_path=checkpoint)
    from .report import load_records, coverage_report
    records = load_records(output, metadata_only=True)
    coverage = coverage_report(records, settings['expected_samples'])
    if suite:
        from .suite_report import condition_coverage
        coverage['conditions'] = condition_coverage(output, records, suite)
        coverage['complete'] = coverage['complete'] and all(v['complete'] for v in coverage['conditions'].values())
    (output / 'coverage.json').write_text(json.dumps(coverage, indent=2), encoding='utf-8')
    (output / 'manifest.json').write_text(json.dumps(jsonable({
        'analysis': settings, 'coverage': coverage,
        'native_result_path': result.raw_output_path,
        'native_sample_paths': result.sample_output_paths,
        'interpretation': ('Paired full-MMSU interventions; input changes reported separately.'
            if suite else 'Observational similarity and concentration; no compression or causal sink claim.'),
    }), indent=2), encoding='utf-8')
    if not coverage['complete']:
        raise RuntimeError(f'Incomplete MMSU observation; see {output / "coverage.json"}')
    report = 'suite_report' if suite else 'report'
    print(f'Collected {len(records)} samples. Plot: {sys.executable} -m scripts.analysis.audio_sink.{report} --run-dir {output}')


if __name__ == '__main__':
    main()
