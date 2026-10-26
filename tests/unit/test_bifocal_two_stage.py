import ast
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
torch = pytest.importorskip('torch')
from mllm_comp.compressors.comparison.bifocal_two_stage import BifocalTwoStage, density_scores, stage1_fraction
from tests.fixtures import native_image


def test_density_counts_missing_neighbors_as_zero():
    representatives = torch.tensor([0, 1, 4, 15])
    actual = density_scores(torch.ones(4), representatives, 4, 4)
    torch.testing.assert_close(actual, torch.tensor([3., 3., 3., 1.]) / 9)


def test_published_qwen_schedule():
    f = stage1_fraction(28, 15)
    assert f == pytest.approx(.41851851851851857)
    assert (15 * f + 13 * (32/576)) / 28 == pytest.approx(.25)


@pytest.mark.parametrize('method', ['bifocal_stage1_k128', 'bifocal_stage12_r25'])
def test_native_v2_stage_contract(method, monkeypatch):
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    monkeypatch.setenv('MLLM_COMP_METHOD_PROFILE', 'full')
    registry = builtin_multimodal_compression_contracts()
    full = method.endswith('r25')
    algorithm = 'bifocal_two_stage' if full else 'bifocal'
    stage = CompressionStageConfig.from_dict({'name': method, 'plugin': algorithm, 'algorithm': algorithm,
        'variant': f'{algorithm}.image', 'binding': f'{algorithm}.image.qwen25_vl',
        'placement': 'llm_level' if full else 'projector_level', 'method': 'token_compression',
        'object_family': 'tensor_slot', 'execution_phase': 'generate', 'modalities': ['image'],
        'target_modalities': ['image'], 'conditioning_modalities': ['text'],
        'preserved_modalities': ['text', 'video', 'audio'], 'coupling_policy': 'independent',
        'media_alignment': 'spatial', 'config': {}})
    validate_v2_stage_contract(stage, registry=registry, backbone_name='qwen25_vl')


class Tokenizer:
    def convert_tokens_to_ids(self, value):
        return {'<|vision_end|>': 8, '<|im_end|>': 9}[value]
    def decode(self, ids, skip_special_tokens=True):
        return 'red cube'
    def __call__(self, text, **kwargs):
        return SimpleNamespace(input_ids=[2 if text.strip() == 'red' else 3])


def inputs():
    ids = torch.tensor([[1, 7, *([10] * 64), 8, 2, 3, 9, 6]])
    return {'input_ids': ids, 'attention_mask': torch.ones_like(ids),
        'image_grid_thw': torch.tensor([[1, 16, 16]]), 'pixel_values': torch.randn(256, 12)}


def activate(model, **extra):
    runtime = SimpleNamespace(metadata={}, model_path_or_id='/fixtures/native-tiny-image',
        processor=SimpleNamespace(tokenizer=Tokenizer()))
    return BifocalTwoStage(model, runtime, {'stage2_layer': 2, 'stage1_config': {'vit_block': 1}, **extra})


def test_native_generate_heterogeneous_cache_decode_and_restoration():
    torch.manual_seed(42)
    model = native_image.qwen_model('qwen25_vl').eval()
    original = model.model.language_model.forward.__func__
    activation = activate(model)
    activation.install()
    try:
        with torch.no_grad():
            output = model.generate(**inputs(), do_sample=False, max_new_tokens=4, min_new_tokens=4,
                eos_token_id=None, pad_token_id=0)
        state = activation.runtime_metadata()['stage2']
        assert output.shape[1] == 75
        assert state['source_count'] == 64
        assert state['stage2_count'] == 8
        assert state['stage1_count'] > state['stage2_count']
        assert state['decode_position_verified'] is True
        assert state['decode_steps'] == 3
        assert state['prefill_cache_lengths'][:2] == [state['prefill_stage1_length']] * 2
        assert state['prefill_cache_lengths'][2:] == [state['prefill_stage2_length']] * 2
    finally:
        activation.uninstall()
    assert model.model.language_model.forward.__func__ is original


def test_identity_stage2_matches_stage1_native_generation():
    torch.manual_seed(43)
    model = native_image.qwen_model('qwen25_vl').eval()
    reference = copy.deepcopy(model)
    data = inputs()
    full = activate(model, average_fraction=.5, stage2_fraction=.5)
    stage1 = activate(reference, average_fraction=.5, stage2_fraction=.5).inner
    full.install()
    stage1.install()
    try:
        with torch.no_grad():
            kwargs = dict(do_sample=False, max_new_tokens=4, min_new_tokens=4, eos_token_id=None, pad_token_id=0)
            actual = model.generate(**data, **kwargs)
            expected = reference.generate(**data, **kwargs)
        assert torch.equal(actual, expected)
    finally:
        full.uninstall()
        stage1.uninstall()


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_stage2_matches_readonly_qwen_reference(dtype):
    path = Path('/home/YinPeiran/vlm_model/qwen2_5_vl/modeling_qwen2_5_vl.py')
    if not path.is_file():
        pytest.skip('Optional read-only Qwen reference not available')
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_multimodal_rotary_pos_emb, repeat_kv
    tree = ast.parse(path.read_text())
    transform = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_hope_s2_transform')
    method = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) for n in cls.body
        if isinstance(n, ast.FunctionDef) and n.name == '_hope_stage2_prune')
    namespace = {'torch': torch, 'F': torch.nn.functional, 'apply_multimodal_rotary_pos_emb': apply_multimodal_rotary_pos_emb,
        'repeat_kv': repeat_kv, '_hope_diag_on': lambda: False, '_HOPE_DIAG': {'s2': None}, '_hope_log_once': lambda *a: None}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[transform, method], type_ignores=[])), str(path), 'exec'), namespace)
    torch.manual_seed(44)
    decoder = native_image.qwen_model('qwen25_vl').to(dtype).eval().model.language_model
    activation = BifocalTwoStage.__new__(BifocalTwoStage)
    activation.decoder, activation.layers, activation.boundary, activation.density_weight = decoder, decoder.layers, 2, 1.
    hidden = torch.randn(1, 20, 24).to(dtype)
    positions = torch.arange(20).view(1, 1, -1).expand(3, 1, -1)
    mask = torch.zeros(1, 20, dtype=torch.bool)
    mask[:, 2:18] = True
    representatives = torch.tensor([0, 1, 2, 3, 8, 9, 10, 11, 20, 21, 22, 23, 36, 37, 38, 39])
    activation.state = {'visual': mask, 'representatives': representatives, 'grid': [8, 8], 'stage2_count': 8}
    rows = torch.zeros(1, 20, dtype=torch.long); cols = rows.clone()
    rows[mask] = representatives // 8; cols[mask] = representatives % 8
    meta = {'mask': mask, 'img': torch.zeros(1, 20, dtype=torch.long), 'row': rows, 'col': cols,
        'grid': [(8, 8)], 'k_img': [8], 'lam': 1., 'denom': 'density'}
    with torch.no_grad():
        expected = namespace['_hope_stage2_prune'](decoder, meta, 2, hidden, positions,
            torch.ones(1, 20), decoder.rotary_emb(hidden, positions))
        actual = activation._select(hidden, positions)
    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
