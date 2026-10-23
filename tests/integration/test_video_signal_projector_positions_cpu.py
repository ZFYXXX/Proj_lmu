"""Signal providers and video values may execute on different GPUs."""

import pytest


from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_projector_positions_cpu import (
    test_vqtoken_configured_input_with_vision_and_projector_on_different_gpus as _run_on_gpus,
)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_signal_provider_keeps_fixed_cues_with_separate_vision_and_projector_gpus(backbone, algorithm, tmp_path):
    if backbone == "llava_onevision":
        from tests.integration.test_video_signal_projector_llava_onevision_positions_cpu import _video_model, _inputs, _pooling_head
        factory = _video_model
        head = _pooling_head(tmp_path) if algorithm == "fastvid" else None
    elif backbone == "qwen3_vl":
        from tests.integration.test_video_signal_projector_qwen3_vl_positions_cpu import _inputs
        factory, head = lambda: image_fixture.qwen_model(backbone), None
    else:
        from tests.integration.test_video_signal_projector_qwen25_vl_positions_cpu import _model, _inputs
        factory, head = _model, None

    def activation(model, stage):
        result = image_fixture.activation(model, stage)
        if head is not None:
            result.signal_provider._pooling_head = head
        return result

    _run_on_gpus(backbone, algorithm=algorithm, model_factory=factory,
                 inputs_factory=_inputs, activation_factory=activation)
