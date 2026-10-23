from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones._decoder_attention import (
    DecoderAttentionImplementationOverride,
)
from mllm_comp.core.errors import BackendIntegrationError


def test_qwen_decoder_attention_override_is_deduplicated_and_reversible() -> None:
    config = SimpleNamespace(_attn_implementation="flash_attention_2")
    language_model = SimpleNamespace(
        config=config,
        layers=(
            SimpleNamespace(self_attn=SimpleNamespace(config=config)),
            SimpleNamespace(self_attn=SimpleNamespace(config=config)),
        ),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(language_model=language_model),
    )
    override = DecoderAttentionImplementationOverride(model)

    override.install()

    assert config._attn_implementation == "sdpa"
    assert override.metadata()["decoder_config_count"] == 1
    override.restore()
    assert config._attn_implementation == "flash_attention_2"


def test_llava_decoder_attention_override_removes_injected_test_attribute() -> None:
    config = SimpleNamespace()
    core = SimpleNamespace(
        config=config,
        layers=(SimpleNamespace(self_attn=SimpleNamespace(config=config)),),
    )
    model = SimpleNamespace(get_model=lambda: core)
    override = DecoderAttentionImplementationOverride(model)

    override.install()

    assert config._attn_implementation == "sdpa"
    override.restore()
    assert not hasattr(config, "_attn_implementation")


def test_decoder_attention_override_fails_when_decoder_is_unresolvable() -> None:
    override = DecoderAttentionImplementationOverride(SimpleNamespace())

    with pytest.raises(BackendIntegrationError, match="could not resolve"):
        override.install()
