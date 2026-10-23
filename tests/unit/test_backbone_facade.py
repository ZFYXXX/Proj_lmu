"""Contract checks for composed backbone facades and their registry factories."""

from __future__ import annotations

import pytest

from mllm_comp.backbones.base import BackboneFacade, ProcessedBatch
from mllm_comp.backbones.profiles import QWEN25_VL_PROFILE
from mllm_comp.backbones.qwen25_omni import (
    Qwen25OmniLegalizer,
    Qwen25OmniModelAdapter,
    Qwen25OmniProcessorAdapter,
    Qwen25OmniRunner,
    Qwen25OmniStateBuilder,
    create_qwen25_omni_backbone,
)
from mllm_comp.registry.backbones import ACTIVE_BACKBONES, get_backbone, get_backbone_factory


def test_qwen25_omni_components_expose_native_runtime_contract() -> None:
    facade = create_qwen25_omni_backbone()

    batch = ProcessedBatch(
        model_inputs={"input_ids": [[1, 2]], "attention_mask": [[1, 1]]},
        sample_ids=["sample"],
        prompt_lengths=[2],
        metadata={
            "token_ids": {"image": 10, "video": 11, "audio": 12},
            "expected_token_counts_by_sample": [{"image": 0, "video": 0, "audio": 0}],
        },
    )
    state = facade.state_builder.build(batch, phase="eval")

    assert state.input_ids == [[1, 2]]
    assert state.native_model_inputs == batch.model_inputs
    assert facade.native_validator is not None


def test_facade_rejects_mismatched_component_profiles() -> None:
    with pytest.raises(ValueError, match="different capability profiles"):
        BackboneFacade(
            processor_adapter=Qwen25OmniProcessorAdapter(),
            model_adapter=Qwen25OmniModelAdapter(QWEN25_VL_PROFILE),
            state_builder=Qwen25OmniStateBuilder(),
            legalizer=Qwen25OmniLegalizer(),
            runner=Qwen25OmniRunner(),
        )


def test_backbone_registry_stores_factories_and_constructs_facades(monkeypatch) -> None:
    monkeypatch.setitem(ACTIVE_BACKBONES, "qwen25_omni_test", create_qwen25_omni_backbone)

    assert get_backbone_factory("qwen25_omni_test") is create_qwen25_omni_backbone
    assert isinstance(get_backbone("qwen25_omni_test"), BackboneFacade)
