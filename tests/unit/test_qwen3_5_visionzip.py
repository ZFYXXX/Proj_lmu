from __future__ import annotations

from types import SimpleNamespace

import pytest


def test_qwen3_5_patch_mean_received_attention_matches_dense_reference() -> None:
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_5.vision_attention import _mean_received_attention

    torch.manual_seed(4)
    query = torch.randn(9, 3, 4)
    key = torch.randn(9, 3, 4)
    got = _mean_received_attention(query, key, scaling=0.5, chunk_size=2)
    q = query.transpose(0, 1).float()
    k = key.transpose(0, 1).float()
    expected = torch.softmax(q @ k.transpose(-1, -2) * 0.5, dim=-1).mean(dim=(0, 1))

    assert torch.allclose(got, expected, atol=1e-6)
    assert got.shape == (9,)
    assert torch.isfinite(got).all()


def test_qwen3_5_visionzip_statistics_pool_to_merger_tokens() -> None:
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_5.vision_attention import (
        pool_patch_statistics_to_merger_tokens,
    )

    stats = {
        "mean_received": torch.arange(16, dtype=torch.float32),
        "keys": torch.arange(16 * 2 * 3, dtype=torch.float32).reshape(16, 2, 3),
    }
    attention, keys = pool_patch_statistics_to_merger_tokens(
        stats, merged_token_count=4, spatial_merge_size=2
    )

    assert attention.tolist() == [1.5, 5.5, 9.5, 13.5]
    assert keys.shape == (2, 4, 3)
    assert torch.allclose(keys[:, 0], stats["keys"][:4].mean(dim=0).T)


def test_qwen3_5_visionzip_compresses_and_compacts_mrope() -> None:
    torch = pytest.importorskip("torch")
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    class FakeAttention(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = torch.nn.Linear(4, 12, bias=False)
            self.num_heads = 2
            self.head_dim = 2
            self.scaling = self.head_dim**-0.5

        def forward(self, hidden_states, cu_seqlens, position_embeddings):
            del cu_seqlens, position_embeddings
            return hidden_states

    class FakeCore:
        rope_deltas = None

        def get_rope_index(self, *, input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
            del mm_token_type_ids, image_grid_thw, attention_mask
            positions = torch.arange(input_ids.shape[1]).view(1, 1, -1)
            return positions.expand(3, input_ids.shape[0], -1), torch.zeros((input_ids.shape[0], 1))

    class FakeModel:
        def __init__(self) -> None:
            self.config = SimpleNamespace(
                image_token_id=99,
                vision_config=SimpleNamespace(spatial_merge_size=2),
            )
            self.model = FakeCore()
            self.model.visual = SimpleNamespace(
                spatial_merge_size=2,
                blocks=[SimpleNamespace(attn=FakeAttention())],
            )
            self.features = torch.randn(4, 6)

        def get_image_features(self, pixel_values, grid, *, return_dict):
            del grid, return_dict
            n = pixel_values.shape[0]
            hidden = torch.randn(n, 4)
            cos = torch.ones(n, 2)
            sin = torch.zeros(n, 2)
            self.model.visual.blocks[-1].attn(
                hidden,
                cu_seqlens=torch.tensor([0, n]),
                position_embeddings=(cos, sin),
            )
            return BaseModelOutputWithPooling(pooler_output=(self.features,))

        def generate(self, **kwargs):
            ids = kwargs["input_ids"]
            assert ids.shape == (1, 4)
            assert int((ids == 99).sum()) == 2
            assert kwargs["position_ids"].shape == (3, 1, 4)
            assert "pixel_values" not in kwargs
            compressed = kwargs["mm_encoder_outputs"]["image"].pooler_output[0]
            assert compressed.shape == (2, 6)
            assert torch.isfinite(compressed).all()
            return torch.cat((ids, torch.tensor([[7]])), dim=-1)

    model = FakeModel()
    registry = __import__(
        "mllm_comp.compressors.multimodal_registry", fromlist=["builtin_multimodal_compression_contracts"]
    ).builtin_multimodal_compression_contracts()
    binding = registry.binding("visionzip.image.qwen3_5")
    from mllm_comp.backbones.qwen3_5.compression import (
        Qwen3_5VisionZipActivation,
        create_qwen3_5_image_compression_executor,
    )

    activation = create_qwen3_5_image_compression_executor(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    assert isinstance(activation, Qwen3_5VisionZipActivation)
    input_ids = torch.tensor([[1, 99, 99, 99, 99, 2]])
    mm_types = torch.zeros_like(input_ids)
    mm_types[:, 1:5] = 1
    activation.install()
    try:
        output = model.generate(
            input_ids=input_ids,
            attention_mask=torch.ones_like(input_ids),
            mm_token_type_ids=mm_types,
            pixel_values=torch.ones(16, 4),
            image_grid_thw=torch.tensor([[1, 4, 4]]),
        )
    finally:
        activation.uninstall()

    record = activation.runtime_metadata()["requests"][0]
    assert torch.equal(output[:, : input_ids.shape[1]], input_ids)
    assert record["image_tokens_before"] == 4
    assert record["image_tokens_after"] == 2
    assert record["images"][0]["attention_mode"] == "mean_received"
    assert record["images"][0]["patches_per_merger_token"] == 4
    assert model.model.rope_deltas is None
