from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

import test_qwen25_vl_image_compression_provider_bridge as fake  # noqa: E402

from mllm_comp.backbones.qwen25_vl import provider_bridge as provider  # noqa: E402
from mllm_comp.backbones.qwen25_vl.image_compression import (  # noqa: E402
    QWEN25_VL_IMAGE_BINDING_CONTRACTS,
    create_qwen25_vl_image_compression_executor,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError  # noqa: E402
from mllm_comp.core.vocabulary import ExecutionPhase, Modality  # noqa: E402

_IMAGE = 9
_VISION_START = 2
_VISION_END = 3
_IM_END = 4
_GRID = (1, 12, 12)
_FEATURES = 36
_WORDS = ("red", "bus", "left")


@pytest.fixture(autouse=True)
def _pinned_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        provider,
        "_installed_transformers_version",
        lambda: provider.SUPPORTED_TRANSFORMERS_VERSION,
    )


class _Tokenizer:
    def __init__(self) -> None:
        self.vocab = {word: 20 + index for index, word in enumerate(_WORDS)}

    def __call__(self, text: str, add_special_tokens: bool = False) -> Any:
        return SimpleNamespace(input_ids=[self.vocab[text.strip()]])

    def decode(self, ids: Any, skip_special_tokens: bool = True) -> str:
        inverse = {value: key for key, value in self.vocab.items()}
        return " ".join(inverse[int(value)] for value in ids)

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|vision_end|>": _VISION_END, "<|im_end|>": _IM_END}[token]


def _prompt() -> Any:
    words = [20 + index for index in range(len(_WORDS))]
    return torch.tensor(
        [[1, _VISION_START, *([_IMAGE] * _FEATURES), _VISION_END, *words, _IM_END, 5]]
    )


def _pixels() -> Any:
    count = _GRID[0] * _GRID[1] * _GRID[2]
    gen = torch.Generator().manual_seed(0)
    return torch.randn(count, 4, generator=gen)


def _model() -> Any:
    model = fake.Qwen2_5_VLForConditionalGeneration()
    model.get_input_embeddings = model.model.get_input_embeddings
    return model


def _activation(model: Any, *, phase: ExecutionPhase, config: dict[str, Any]) -> Any:
    contract = QWEN25_VL_IMAGE_BINDING_CONTRACTS["bifocal.image.qwen25_vl"]
    binding = SimpleNamespace(
        name=contract.binding,
        backbone="qwen25_vl",
        variant=contract.variant,
        target_modalities=frozenset({Modality.IMAGE}),
        supported_phases=frozenset({ExecutionPhase.FORWARD, ExecutionPhase.GENERATE}),
        ordered_targets=lambda: (),
        position_resolver=None,
        strategy_factory=None,
    )
    stage = SimpleNamespace(
        name="provider-bifocal",
        algorithm="bifocal",
        variant="bifocal.image",
        target_modalities={Modality.IMAGE},
        operation={},
        criterion={},
        schedule={},
        config=dict(config),
    )
    return create_qwen25_vl_image_compression_executor(
        model=model,
        runtime=SimpleNamespace(
            metadata={},
            model_path_or_id="/fixtures/unit-test-model",
            processor=SimpleNamespace(tokenizer=_Tokenizer()),
        ),
        facade=None,
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(config={}),
        execution_phase=phase,
    )


_CONFIG = {"s1_frac": 0.5, "vit_block": 1}


def test_forward_compacts_prompt_with_bifocal_layout_and_query() -> None:
    model = _model()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, phase=ExecutionPhase.FORWARD, config=_CONFIG)
    activation.install()
    input_ids = _prompt()
    length = int(input_ids.shape[1])

    model.forward(
        input_ids=input_ids,
        attention_mask=torch.ones(1, length, dtype=torch.long),
        pixel_values=_pixels(),
        image_grid_thw=torch.tensor([_GRID]),
        use_cache=False,
    )

    record = activation.runtime_metadata()["layout_delta_summaries"][-1]
    assert record["bifocal_query"] == "red bus left"
    assert record["bifocal_chain"] == "salr+crn(3w)"
    assert record["bifocal_passthrough"] is False
    kept = 18
    assert record["result_count"] == kept
    forwarded = model.model.language_model.last_forward
    assert forwarded is not None
    assert forwarded["inputs_embeds"].shape[1] == length - _FEATURES + kept
    positions = forwarded["position_ids"][0, 0].tolist()
    image_start = 2
    reps = record["bifocal_representatives"]
    expected = (
        list(range(image_start))
        + [image_start + int(index) for index in reps]
        + list(range(image_start + _FEATURES, length))
    )
    assert positions == expected
    metadata = bridge.runtime_metadata()
    assert metadata["latest_rope_delta_repair"]["policy"] == "bifocal_no_correction"
    assert metadata["latest_rope_delta_repair"]["published_rope_delta"] == [0]

    activation.uninstall()
    bridge.uninstall()


def test_column_mass_matches_dense_softmax_reference() -> None:
    model = _model()
    block = model.model.visual.blocks[1]
    length = 16
    gen = torch.Generator().manual_seed(1)
    normalised = torch.randn(length, 4, generator=gen)
    embedding = torch.zeros(length, 2)
    cos, sin = embedding.cos(), embedding.sin()
    cu = torch.tensor([0, length], dtype=torch.int32)

    maps = provider._bifocal_vit_column_mass(block, normalised, cu, (cos, sin))

    qkv = block.attn.qkv(normalised).reshape(length, 3, block.attn.num_heads, -1)
    q = qkv[:, 0].transpose(0, 1).float()
    k = qkv[:, 1].transpose(0, 1).float()
    attention = torch.softmax(q @ k.transpose(1, 2) * block.attn.scaling, dim=-1)
    mass = attention.sum(dim=(0, 1))
    diag = attention.diagonal(dim1=1, dim2=2).sum(dim=0)
    vnorm = qkv[:, 2].norm(dim=-1).t().float()
    torch.testing.assert_close(maps["mass"], mass)
    torch.testing.assert_close(maps["nodiag"], mass - diag)
    torch.testing.assert_close(maps["vmass"], (attention.sum(dim=1) * vnorm).sum(dim=0))


class _Cache:
    def __init__(self, length: int) -> None:
        self.length = length

    def get_seq_length(self) -> int:
        return self.length


def _install_fake_generate(model: Any, *, decode_offset: int = 0) -> None:
    def generate(self: Any, input_ids: Any, **kwargs: Any) -> Any:
        length = int(input_ids.shape[1])
        attention_mask = kwargs["attention_mask"]
        self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=kwargs["pixel_values"],
            image_grid_thw=kwargs["image_grid_thw"],
            use_cache=True,
        )
        compact = int(self.model.language_model.last_forward["inputs_embeds"].shape[1])
        rope = int(self.model.rope_deltas.reshape(-1)[0])
        for step in range(2):
            cache_position = length + step
            position = cache_position + rope + decode_offset
            self.forward(
                input_ids=torch.tensor([[5]]),
                attention_mask=torch.ones(1, cache_position + 1, dtype=torch.long),
                position_ids=torch.full((3, 1, 1), position, dtype=torch.long),
                past_key_values=_Cache(compact + step),
                cache_position=torch.tensor([cache_position]),
                use_cache=True,
            )
        return "bifocal-generate"

    model.generate = MethodType(generate, model)


def _generate(model: Any) -> Any:
    input_ids = _prompt()
    return model.generate(
        input_ids,
        attention_mask=torch.ones(1, int(input_ids.shape[1]), dtype=torch.long),
        pixel_values=_pixels(),
        image_grid_thw=torch.tensor([_GRID]),
    )


def test_projector_only_generate_verifies_first_decode_position() -> None:
    model = _model()
    _install_fake_generate(model)
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, phase=ExecutionPhase.GENERATE, config=_CONFIG)
    activation.install()
    assert bridge.runtime_metadata()["active_generate_supported"] is True

    assert _generate(model) == "bifocal-generate"

    state = bridge.runtime_metadata()["latest_projector_generate"]
    length = int(_prompt().shape[1])
    assert state["prefill_calls"] == 1
    assert state["prompt_removed"] == _FEATURES - 18
    assert state["decode_steps"] == 2
    assert state["expected_decode_position"] == length
    assert state["observed_decode_position"] == [length]
    assert state["decode_position_verified"] is True
    summary = activation.runtime_metadata()["bifocal_generate"]
    assert summary["generate_calls"] == 1
    assert summary["decode_position_verified"] == 1
    assert bridge.runtime_metadata()["latest_rope_delta_repair"]["published_rope_delta"] == [0]

    activation.uninstall()
    bridge.uninstall()


def test_projector_only_generate_rejects_shifted_decode_position() -> None:
    model = _model()
    _install_fake_generate(model, decode_offset=-(_FEATURES - 18))
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, phase=ExecutionPhase.GENERATE, config=_CONFIG)
    activation.install()

    with pytest.raises(InvariantError, match="first decode position"):
        _generate(model)
    state = bridge.runtime_metadata()["latest_projector_generate"]
    assert state["decode_position_verified"] is False
    assert activation.runtime_metadata()["bifocal_generate"]["decode_position_failed"] == 1

    activation.uninstall()
    bridge.uninstall()


def test_bifocal_rejects_non_full_attention_vit_block_and_multi_image() -> None:
    model = _model()
    model.model.visual.fullatt_block_indexes = [0]
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, phase=ExecutionPhase.FORWARD, config=_CONFIG)
    activation.install()
    input_ids = _prompt()
    with pytest.raises(BackendIntegrationError, match="not a full-attention block"):
        model.forward(
            input_ids=input_ids,
            attention_mask=torch.ones(1, int(input_ids.shape[1]), dtype=torch.long),
            pixel_values=_pixels(),
            image_grid_thw=torch.tensor([_GRID]),
            use_cache=False,
        )
    activation.uninstall()
    bridge.uninstall()

    model = _model()
    bridge = provider.install_qwen25_vl_transformers_4573_bridge(model)
    activation = _activation(model, phase=ExecutionPhase.FORWARD, config=_CONFIG)
    activation.install()
    with pytest.raises(BackendIntegrationError, match="exactly one image"):
        model.forward(
            input_ids=torch.tensor([[1, 9, 9, 9, 9, 2, 9, 9, 9, 9, 3]]),
            attention_mask=torch.ones(1, 11, dtype=torch.long),
            pixel_values=torch.arange(128, dtype=torch.float32).reshape(32, 4),
            image_grid_thw=torch.tensor([[1, 4, 4], [1, 4, 4]]),
            use_cache=False,
        )
    activation.uninstall()
    bridge.uninstall()
