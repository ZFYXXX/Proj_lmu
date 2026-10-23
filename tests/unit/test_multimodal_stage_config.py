from __future__ import annotations

import pytest

from mllm_comp.core.errors import ConfigError
from mllm_comp.core.vocabulary import (
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
)
from mllm_comp.pipeline.stage import CompressionStageConfig


def _base() -> dict[str, object]:
    return {
        "name": "visionzip_image",
        "placement": "projector_level",
        "execution_phase": "generate",
        "plugin": "visionzip",
    }


def test_legacy_modalities_project_to_v2_target_modalities() -> None:
    raw = _base()
    raw["modalities"] = ["audio"]

    stage = CompressionStageConfig.from_dict(raw)

    assert stage.modalities == {Modality.AUDIO}
    assert stage.target_modalities == {Modality.AUDIO}
    assert not stage.uses_v2_contract
    assert "modality_contract" not in stage.plugin_config()


def test_v2_image_stage_preserves_explicit_modality_contract() -> None:
    raw = _base()
    raw.update(
        {
            "algorithm": "visionzip",
            "variant": "visionzip.image",
            "binding": "visionzip.image.llava_onevision",
            "target_modalities": ["image"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "video"],
            "coupling_policy": "independent",
            "media_alignment": "spatial",
        }
    )

    stage = CompressionStageConfig.from_dict(raw)

    assert stage.modalities == {Modality.IMAGE}
    assert stage.target_modalities == {Modality.IMAGE}
    assert stage.conditioning_modalities == {Modality.TEXT}
    assert stage.coupling_policy is ModalityCouplingPolicy.INDEPENDENT
    assert stage.media_alignment is MediaAlignment.SPATIAL
    contract = stage.plugin_config()["modality_contract"]
    assert contract["preserved_modalities"] == ["text", "video"]


def test_compound_stage_orders_bindings_and_requires_coupling() -> None:
    raw = _base()
    raw.update(
        {
            "algorithm": "future_joint",
            "variant": "future_joint.av",
            "target_modalities": ["audio", "video"],
            "coupling_policy": "joint",
            "media_alignment": "temporal",
            "bindings": [
                {
                    "binding": "video_leg",
                    "target_modalities": ["video"],
                    "depends_on": ["audio_leg"],
                    "order": 1,
                },
                {
                    "binding": "audio_leg",
                    "target_modalities": ["audio"],
                    "order": 0,
                },
            ],
        }
    )

    stage = CompressionStageConfig.from_dict(raw)

    contract = stage.plugin_config()["modality_contract"]
    assert [item["binding"] for item in contract["bindings"]] == [
        "audio_leg",
        "video_leg",
    ]
    assert stage.coupling_policy is ModalityCouplingPolicy.JOINT


def test_stage_rejects_legacy_and_v2_modality_disagreement() -> None:
    raw = _base()
    raw.update({"modalities": ["audio"], "target_modalities": ["image"]})

    with pytest.raises(ConfigError, match="must match"):
        CompressionStageConfig.from_dict(raw)


def test_stage_rejects_implicit_independent_multi_modality_claim() -> None:
    raw = _base()
    raw.update(
        {
            "algorithm": "future_joint",
            "variant": "future_joint.av",
            "target_modalities": ["audio", "video"],
        }
    )

    with pytest.raises(ConfigError, match="coupling_policy"):
        CompressionStageConfig.from_dict(raw)


def test_stage_rejects_target_preserved_overlap() -> None:
    raw = _base()
    raw.update(
        {
            "target_modalities": ["image"],
            "preserved_modalities": ["image", "text"],
        }
    )

    with pytest.raises(ConfigError, match="overlap"):
        CompressionStageConfig.from_dict(raw)


def test_stage_rejects_partial_algorithm_variant_pair() -> None:
    raw = _base()
    raw["algorithm"] = "visionzip"

    with pytest.raises(ConfigError, match="provided together"):
        CompressionStageConfig.from_dict(raw)
