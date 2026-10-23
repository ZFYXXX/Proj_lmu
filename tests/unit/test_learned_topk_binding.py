import pytest
import torch

from mllm_comp.compressors.comparison.learned_topk import QwenLearnedTopKComparison
from tests.unit.test_projector_comparison_evaluation import tiny_state


@pytest.mark.parametrize('random', [False, True])
def test_native_generate_compact_cache_and_repeat(random):
    bundle, state, _, _ = tiny_state()
    model = bundle['model']
    session = QwenLearnedTopKComparison(model, {'target_count': 3}, random=random).install()
    try:
        with torch.no_grad():
            a = model.generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
            first = session.runtime_metadata()['selection']
            b = model.generate(**state['prompt_inputs'], max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
        assert torch.equal(a, b)
        assert first['result_count'] == 3
        assert first['decode_position_verified']
        assert first['observed_visual_tokens_per_decoder_layer'] == [3] * 4
    finally:
        session.uninstall()


def test_identity_preserves_dense_generation_logits():
    bundle, state, _, _ = tiny_state()
    model = bundle['model']
    with torch.no_grad():
        expected = model(**state['prompt_inputs'], use_cache=False).logits
    session = QwenLearnedTopKComparison(model, {'target_count': 128}).install()
    try:
        with torch.no_grad():
            actual = model(**state['prompt_inputs'], use_cache=False).logits
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        session.uninstall()


def test_soft_binding_backprop_without_backbone_gradients():
    bundle, state, _, _ = tiny_state()
    model = bundle['model'].requires_grad_(False)
    session = QwenLearnedTopKComparison(model, {'target_count': 3, 'training_mode': 'soft'}, phase='forward').install()
    try:
        labels = torch.full_like(state['prompt_inputs']['input_ids'], -100)
        labels[:, -1] = 3
        result = model(**state['prompt_inputs'], labels=labels, use_cache=False)
        result.loss.backward()
        assert result.logits.shape[1] == state['prefix']
        assert session.last_selection['result_count'] == 6
        assert all(p.grad is None for p in model.parameters())
        assert sum(float(p.grad.abs().sum()) for p in session.selector.parameters()) > 0
    finally:
        session.uninstall()


def test_soft_generation_rejected():
    bundle, _, _, _ = tiny_state()
    with pytest.raises(ValueError, match='forward-only'):
        QwenLearnedTopKComparison(bundle['model'], {'training_mode': 'soft'})
