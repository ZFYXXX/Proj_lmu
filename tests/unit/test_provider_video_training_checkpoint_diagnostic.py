from argparse import Namespace
from pathlib import Path

import pytest

from scripts.diagnostics import validate_provider_video_training_checkpoint as target


@pytest.mark.parametrize("algorithm", target.ALGORITHMS)
@pytest.mark.parametrize("backbone", target.BACKBONES)
def test_fixed_fixture_keeps_native_backbone_profile(algorithm, backbone):
    args = Namespace(algorithm=algorithm, backbone=backbone)
    sample = target._sample(
        args, {"prompt": "What happens next?", "target": "D"}, Path("video.mp4")
    )
    assert sample.sample_id == "nextqa_mc_test:442:sft"
    assert target._scope(args)["binding"] == f"{algorithm}.video.{backbone}"
    assert (
        target._stage(args, target.ExecutionPhase.FORWARD).binding
        == f"{algorithm}.video.{backbone}"
    )
    if backbone != "llava_onevision":
        video = sample.conversation[1]["content"][0]
        assert video["fps"] == 1.0
        if algorithm == "longvu":
            assert video["min_pixels"] < video["max_pixels"]
    assert target.OPTIMIZER_STEPS == 3


def test_missing_compressor_stage_gradient_cannot_pass():
    # A final-norm update alone cannot activate a trainable compressor.
    assert not all(
        target.parameter_checks(
            "llamavid",
            [
                {
                    "name": "norm.weight",
                    "gradient_finite": True,
                    "gradient_nonzero": True,
                    "max_parameter_delta": 0.1,
                }
            ],
        ).values()
    )


def test_nonfinite_gradient_and_zero_update_are_rejected():
    names = (
        "vision_query",
        "raw_projector.weight",
        "dino_projector.weight",
        "layers.weight",
        "output_projector.weight",
    )
    records = [
        {
            "name": name,
            "gradient_finite": True,
            "gradient_nonzero": True,
            "max_parameter_delta": 0.1,
        }
        for name in names
    ]
    assert all(target.parameter_checks("longvu", records).values())
    records[0]["max_parameter_delta"] = 0.0
    assert not all(target.parameter_checks("longvu", records).values())
    records[0]["gradient_finite"] = False
    assert not target.parameter_checks("longvu", records)["all_gradients_finite"]
