"""Compound video Projector values preserve gradients across actual device hooks."""

import pytest

from tests.integration.test_video_projector_positions_cpu import (
    test_vqtoken_configured_input_with_vision_and_projector_on_different_gpus as _run_on_gpus,
)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dycoke_projector_position_across_gpus(backbone):
    from tests.integration.test_video_dycoke_positions_cpu import _model, _inputs

    _run_on_gpus(backbone, algorithm="dycoke", model_factory=lambda: _model(backbone),
        inputs_factory=lambda torch: _inputs(torch, backbone))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_dytok_projector_position_across_gpus(backbone):
    from tests.integration.test_video_dytok_positions_cpu import _model, _inputs, _executor, _FrameWeights

    _run_on_gpus(backbone, algorithm="dytok", model_factory=lambda: _model(backbone),
        inputs_factory=lambda torch: _inputs(torch, backbone),
        activation_factory=lambda model, stage: _executor(model, backbone, stage, _FrameWeights()),
        request_context_factory=lambda activation: activation._bridge.request_context(
            {"conditioning_text": "Describe the motion."} if backbone == "llava_onevision"
            else {"video_conditioning_texts": ["Describe the motion."]}))
