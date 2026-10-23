"""SparseVLM uses fixed Attention sources and configurable select/merge boundaries."""

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


def _stage(backbone, boundaries=(1, 3), *, sources=(0, 2), mid=False, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "sparsevlm-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "sparsevlm", "algorithm": "sparsevlm", "variant": "sparsevlm.image",
        "binding": f"sparsevlm.image.{backbone}", "target_modalities": ["image"],
        "config": {"prune_layers": list(sources), "schedule_keep_tokens": [2, 1][:len(sources)],
            "scale_schedule_to_input": False, "recycle_ratio": 1.0, "cluster_divisor": 10},
        "operation": {"targets": {"decoder_merge_boundaries": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((0, 2), False), ((1, 3), False), ((3, 4), False), ((0, 2), True), ((2, 3), True)])
def test_sparsevlm_native_application_and_mean_replay(backbone, boundaries, mid):
    # The delayed cases capture source 2 before stage 0 runs and must carry its
    # Attention mass through that merge; same-entry cases capture after stage 0.
    shared._run(backbone, boundaries, (0, 2), mid, stage_factory=_stage,
        visual_counts="recorded", checkpoint_modes=(None, False, True) if mid else (None,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_sparsevlm_default_aliases(backbone):
    shared._default_aliases(backbone, _stage, "decoder_merge_boundaries", (0, 2))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_sparsevlm_two_device_source_and_application(backbone):
    shared._run(backbone, (2, 3), (0, 2), True, stage_factory=_stage,
        visual_counts="recorded", checkpoint_modes=(None,), two_devices=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("invalid", ["source", "mid_application"])
def test_sparsevlm_public_depth_error(backbone, invalid):
    from mllm_comp.core.errors import BackendIntegrationError

    model = image_fixture.model(backbone, num_decoder_layers=4)
    depth = len(image_fixture.decoder(model, backbone).layers)
    stage = _stage(backbone, (1, depth), sources=(0, depth) if invalid == "source" else (0, 2), mid=True)
    if invalid == "source":
        stage.operation = {}
    activation = None
    try:
        with pytest.raises(BackendIntegrationError):
            activation = image_fixture.activation(model, stage)
            activation.install()
    finally:
        if activation is not None:
            activation.uninstall()
