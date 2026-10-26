"""Fixed-count Stage-1 control; not the full paper's two-stage budget."""
import pytest

torch = pytest.importorskip('torch')
from mllm_comp.compressors.image.bifocal import BifocalSpec, bifocal_image_tokens


@pytest.mark.parametrize('h,w', [(16, 16), (18, 35), (32, 64)])
def test_fixed_budget_matches_equivalent_per_image_fraction(h, w):
    torch.manual_seed(42)
    tokens = torch.randn(h * w, 24).bfloat16()
    words = torch.randn(5, 24).bfloat16()
    maps = {'mass': torch.rand(h * w)}
    kwargs = dict(grid_height=h, grid_width=w, word_embeddings=words)
    actual = bifocal_image_tokens(tokens, maps, spec=BifocalSpec(target_count=128), **kwargs)
    expected = bifocal_image_tokens(tokens, maps, spec=BifocalSpec(s1_frac=128 / (h * w)), **kwargs)
    assert actual.tokens.shape == (128, 24)
    assert len(actual.indices.unique()) == 128
    assert torch.equal(actual.tokens, expected.tokens)
    assert torch.equal(actual.indices, expected.indices)
    assert actual.layout_delta.source_count == h * w
    assert actual.layout_delta.result_count == 128
    assert actual.metadata['budget_policy'] == 'fixed_count'


@pytest.mark.parametrize('n', [16, 64, 128])
def test_short_image_is_identity_not_token_duplication(n):
    tokens = torch.randn(n, 24)
    result = bifocal_image_tokens(tokens, None, grid_height=1, grid_width=n, spec=BifocalSpec(target_count=128))
    assert result.tokens is tokens
    assert result.metadata['passthrough'] is True
    assert result.layout_delta.is_identity


@pytest.mark.parametrize('value', [0, -1, True, 1.5])
def test_invalid_target_count(value):
    with pytest.raises(ValueError, match='target_count'):
        BifocalSpec(target_count=value)
