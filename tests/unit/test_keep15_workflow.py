"""Matched fractional budgets, old checkpoint compatibility and native decoding."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mllm_comp.workflow.catalog import ROOT, load_catalog
from mllm_comp.workflow import runner
from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
from mllm_comp.compressors.image.mmtok import MMTokSpec

CATALOG = ROOT / 'configs/workflow/methods_keep15.json'
EXPERIMENT = ROOT / 'configs/workflow/experiments/qwen25_keep15.json'


def test_exact_requested_matrix_and_separate_old_and_new_weights():
    config = runner.configuration(ROOT / 'configs/workflow/profiles/p6000_keep15.json',
                                  EXPERIMENT, catalog_path=CATALOG)
    assert len(config['methods']) == 8 and len(config['tasks']) == 16
    assert len(config['requested_cells']) == 128
    assert 'chartqa' not in config['tasks'] and not any(t.startswith('vqav2') for t in config['tasks'])
    assert config['checkpoints']['old_mlp_r15'] == config['checkpoints']['importance_fps_r15']
    assert config['checkpoints']['importance_fps_balanced_r15'] != config['checkpoints']['importance_fps_r15']
    assert Path(config['checkpoints']['importance_fps_balanced_r15']).is_dir()
    for method, spec in load_catalog(CATALOG).items():
        options = spec.get('options', {})
        assert 'tokens_per_crop' not in options and 'target_count' not in options
        if method != 'dense':
            assert options.get('retention_ratio', options.get('retain_ratio')) == .15


def test_all_selected_configs_validate_through_framework(tmp_path):
    import yaml
    from mllm_comp.config.loader import load_config
    from scripts.eval.lmms_eval.p6000_dataset_smoke import model_config
    for method in load_catalog(CATALOG):
        config = model_config(SimpleNamespace(model='/model', method=method, method_checkpoint='/head',
            scorer_checkpoint='/scorer', identity_path='/identity', method_catalog=CATALOG))
        path = tmp_path / f'{method}.yaml'
        path.write_text(yaml.safe_dump(config))
        assert load_config(path) is not None


@pytest.mark.parametrize('n', [1, 6, 16, 64, 128, 256, 512, 1024, 1952])
def test_matched_rounding_and_no_hidden_k128_cap(n):
    owner = SimpleNamespace(retention_ratio=.15, config={})
    expected = min(n, max(1, round(.15 * n)))
    assert QwenProjectorComparison.token_budget(owner, n) == expected
    assert MMTokSpec(retain_ratio=.15, budget_rounding='round').resolve_target_count(n) == expected
    if n == 1024:
        assert expected == 154
    # Existing upstream-style floor budgets stay unchanged.
    assert MMTokSpec(retain_ratio=.2).resolve_target_count(107) == 21


@pytest.mark.parametrize('method', ['eadp', 'prunesid'])
def test_native_reference_generate_at_15_percent(method):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    bundle, state, _, _ = tiny_state()
    session = QwenProjectorComparison(bundle['model'], method, {'retention_ratio': .15}, 'qwen25_vl').install()
    try:
        with torch.no_grad():
            bundle['model'].generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3,
                                     do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()['selection']
        assert trace['source_count'] == 6 and trace['result_count'] == 1
        assert trace['decode_position_verified'] and trace['requested_retention_ratio'] == .15
    finally:
        session.uninstall()


def test_native_protection_uses_runtime_budget_not_serialized_k128():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.protected_topk import QwenProtectedTopKComparison
    bundle, state, _, _ = tiny_state()
    session = QwenProtectedTopKComparison(bundle['model'], {'retention_ratio': .15}).install()
    try:
        assert session.selector.spec.target_count == 128
        with torch.no_grad():
            bundle['model'].generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3,
                                     do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()['selection']
        assert trace['result_count'] == 1 and trace['protected_count'] <= 1
        assert trace['decode_position_verified'] and session.selector.spec.target_count == 128
        assert set(trace['protected_indices']) <= set(trace['selected_indices'])
    finally:
        session.uninstall()


@pytest.mark.parametrize('importance', [False, True])
def test_native_aggregation_generates_at_15_percent(importance):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.analysis.common.compression_training_check import create_anchor_adapter, AnchorAdapterSpec
    from mllm_comp.compressors.comparison.anchor_adapter import QwenAnchorAdapterComparison
    from mllm_comp.compressors.comparison.importance_anchor_adapter import QwenImportanceAnchorAdapter
    from mllm_comp.compressors.image.context_visual_topk import ContextVisualTopK, ContextVisualTopKSpec
    bundle, state, _, _ = tiny_state()
    adapter = create_anchor_adapter(AnchorAdapterSpec(hidden_size=24, width=4)).eval()
    if importance:
        scorer = ContextVisualTopK(ContextVisualTopKSpec(24, width=4, context_mode='image'))
        session = QwenImportanceAnchorAdapter(bundle['model'],
            {'retention_ratio': .15, 'importance_gamma': 1., 'pooling': 'mean'},
            adapter_module=adapter, selector_module=scorer)
    else:
        session = QwenAnchorAdapterComparison(bundle['model'], {'retention_ratio': .15}, adapter_module=adapter)
    session.install()
    try:
        with torch.no_grad():
            bundle['model'].generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3,
                                     do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()['selection']
        assert trace['result_count'] == 1 and trace['decode_position_verified']
        assert sorted(i for group in trace['cluster_members'] for i in group) == list(range(6))
    finally:
        session.uninstall()


def test_old_and_new_real_weights_still_load_without_hash_exemptions():
    from mllm_comp.analysis.common.anchor7b_runtime import load_adapter
    from mllm_comp.compressors.image.context_visual_topk import load_selector as load_context
    from mllm_comp.compressors.image.word_visual_topk import load_selector as load_word
    config = runner.configuration(ROOT / 'configs/workflow/profiles/p6000_keep15.json', EXPERIMENT, catalog_path=CATALOG)
    if not Path(config['identity']).is_file():
        pytest.skip('P6000 checkpoint package is not present on this machine')
    identity = json.loads(Path(config['identity']).read_text())
    for method in ('old_mlp_r15', 'importance_fps_balanced_r15'):
        module = load_adapter(config['checkpoints'][method], identity, 'cpu')
        for k in (1, 38, 154):
            assert torch.isfinite(module({'tokens': torch.randn(k, 3584)})['tokens']).all()
    load_context(config['scorer_checkpoint'], base_identity=identity['identity_sha256'], device='cpu')
    load_word(config['checkpoints']['protected_selector_r15'], base_identity=identity['identity_sha256'], device='cpu')


def test_native_mmtok_binding_rounds_the_same_15_percent_budget():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from tests.integration.test_image_projector_positions_cpu import _stage
    from tests.fixtures.native_image import activation
    bundle, _, _, _ = tiny_state()
    model = bundle['model']
    stage = _stage('qwen25_vl', 'mmtok', None, phase='generate')
    stage.config = {'retain_ratio': .15, 'budget_rounding': 'round'}
    session = activation(model, stage)
    # Synthetic vocabulary deliberately uses the fixture's non-ChatML path.
    # The real 7B GPU smoke exercises the native tokenizer and user prompt.
    ids = torch.tensor([[1, 7, *([10] * 12), 8, 2]])
    session.install()
    try:
        with torch.no_grad():
            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                pixel_values=torch.randn(48, 12), image_grid_thw=torch.tensor([[1, 6, 8]]),
                max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
        assert output.shape[1] == ids.shape[1] + 3
        records = session.runtime_metadata()['layout_delta_records']
        assert records
        assert any(row.get('source_count') == 12 and row.get('result_count') == 2 for row in records)
    finally:
        session.uninstall()


def test_waiting_smoke_does_not_start_while_gpu_is_busy(tmp_path, monkeypatch):
    from mllm_comp.workflow import anchor_training
    (tmp_path / 'workflow.json').write_text(json.dumps({'config': {'profile': {'gpus': ['0']}}}))
    polls = iter([[], [], ['0']])
    monkeypatch.delenv('SLURM_JOB_ID', raising=False)
    monkeypatch.setattr(anchor_training, 'idle_devices', lambda selected: next(polls))
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: None)
    calls = []
    monkeypatch.setattr(runner, 'execute', lambda output: calls.append(output) or 0)
    assert runner.wait_execute(tmp_path) == 0
    assert calls == [tmp_path]
    assert json.loads((tmp_path / 'workflow_status.json').read_text())['status'] == 'waiting_for_idle_gpu'


def test_keep15_lmu_still_has_four_shared_workers_and_no_k128_seeding():
    config = runner.configuration(ROOT / 'configs/workflow/profiles/lmu_keep15.json', EXPERIMENT, catalog_path=CATALOG)
    assert config['profile']['gpus'] == ['0', '1', '2', '3']
    assert set(config['methods']) & load_catalog().keys() == {'dense'}
    script = runner.slurm_script({'config': config, 'source_root': '/frozen'}, '/workspace/run')
    assert '--gres=gpu:4' in script and '--job-name=28-q25-keep15' in script
    assert '--mail-type="NONE"' in script
    saved = {'config': config, 'source_root': '/frozen', 'limit': None}
    command = runner.queue_command(saved, '/workspace/run', 3, 4)
    assert command[command.index('--shared-workers') + 1] == '4'
    assert command[command.index('--worker-index') + 1] == '3'
    assert '--mmmu-single-image-only' in command


def test_lmu_smoke_entry_is_limited_debug_only_and_requests_one_gpu():
    import subprocess
    path = ROOT / 'scripts/eval/lmms_eval/lmu_qwen25_keep15_smoke.sbatch'
    subprocess.run(['bash', '-n', str(path)], check=True)
    script = path.read_text()
    for expected in ('--job-name=28-', '--partition=dev_accelerated', '--gres=gpu:1',
                     '--cpus-per-task=4',
                     '--time=00:50:00', '--mail-type="NONE"', 'scripts/workflow.py smoke',
                     '--limit 2', 'qwen25_keep15_lmu_smoke_${SLURM_JOB_ID}'):
        assert expected in script
    assert 'sbatch ' not in script and 'salloc ' not in script


def test_lmu_single_gpu_smoke_and_four_gpu_full_are_distinct(monkeypatch):
    monkeypatch.setenv('SLURM_JOB_ID', 'smoke-fixture')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-allocated-one')
    profile = {'gpus': ['0', '1', '2', '3']}
    assert runner.execution_devices(profile, limited=True) == ['GPU-allocated-one']
    with pytest.raises(ValueError, match='requires 4'):
        runner.execution_devices(profile, limited=False)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', 'GPU-a,GPU-b,GPU-c,GPU-d')
    assert len(runner.execution_devices(profile, limited=False)) == 4
    with pytest.raises(ValueError, match='requires 1'):
        runner.execution_devices(profile, limited=True)
    monkeypatch.delenv('CUDA_VISIBLE_DEVICES')
    with pytest.raises(ValueError, match='Slurm must expose'):
        runner.execution_devices(profile, limited=True)


@pytest.mark.parametrize('smoke_fails', [False, True])
def test_one_allocation_smoke_gates_four_gpu_full(tmp_path, smoke_fails):
    import os
    import subprocess
    import sys
    job = ROOT / 'scripts/eval/lmms_eval/lmu_qwen25_keep15_full.sbatch'
    subprocess.run(['bash', '-n', str(job)], check=True)
    # Fake interpreter records command/device mapping; no model or GPU is used.
    fake = tmp_path / 'python'
    fake.write_text(f'#!{sys.executable}\n' + '''import json, os, pathlib, sys
with open(os.environ['CALL_LOG'], 'a') as stream:
    stream.write(json.dumps({'args': sys.argv[1:], 'gpus': os.environ['CUDA_VISIBLE_DEVICES']}) + '\\n')
if sys.argv[2] == 'smoke' and os.environ['SMOKE_FAIL'] == '1':
    sys.exit(7)
''')
    fake.chmod(0o755)
    log = tmp_path / 'calls.jsonl'
    env = {**os.environ, 'SLURM_JOB_ID': 'fixture', 'SLURM_SUBMIT_DIR': str(tmp_path),
           'CUDA_VISIBLE_DEVICES': 'GPU-a,GPU-b,GPU-c,GPU-d', 'KEEP15_REPO': str(tmp_path),
           'KEEP15_PYTHON': str(fake), 'KEEP15_OUTPUT': str(tmp_path / 'full'),
           'CALL_LOG': str(log), 'SMOKE_FAIL': str(int(smoke_fails))}
    result = subprocess.run(['bash', str(job)], env=env, capture_output=True, text=True)
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert calls[0]['args'][1] == 'prepare' and calls[1]['args'][1] == 'smoke'
    assert calls[1]['gpus'] == 'GPU-a'
    assert calls[1]['args'][calls[1]['args'].index('--limit') + 1] == '1'
    if smoke_fails:
        assert result.returncode == 7 and len(calls) == 2
    else:
        assert result.returncode == 0 and calls[2]['args'][1] == 'execute'
        assert calls[2]['gpus'] == 'GPU-a,GPU-b,GPU-c,GPU-d'
        assert calls[2]['args'][0].endswith('/full/source/scripts/workflow.py')
