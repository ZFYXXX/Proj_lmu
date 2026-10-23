from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen3_5 import create_qwen3_5_backbone
from mllm_comp.backbones.qwen3_5.compression import (
    Qwen3_5DivPruneActivation,
    Qwen3_5ToMeActivation,
    Qwen3_5VisionZipActivation,
    Qwen3_5OTPruneActivation,
    Qwen3_5PruMergeActivation,
    Qwen3_5MMTokActivation,
    Qwen3_5GAPActivation,
    Qwen3_5VisPrunerActivation,
    Qwen3_5AIMActivation,
    Qwen3_5RestoreActivation,
    create_qwen3_5_image_compression_executor,
)
from mllm_comp.backbones.qwen3_5.video_compression import (
    Qwen3_5VideoVQTokenActivation,
    create_qwen3_5_video_compression_executor,
)
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.core.errors import BackendIntegrationError


def test_qwen3_5_is_a_separate_image_video_backbone() -> None:
    profile = create_qwen3_5_backbone().capabilities()

    assert profile.name == "qwen3_5"
    assert profile.supports_image
    assert profile.supports_video
    assert not profile.supports_audio
    assert profile.position_system == "m_rope"


def test_qwen3_5_vqtoken_video_binding_is_projector_level_generate_only() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "vqtoken.video.qwen3_5"
    )

    assert binding.backbone == "qwen3_5"
    assert binding.variant == "vqtoken.video"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {
            "video_features_pre_llm_scatter",
            "vision_encoder",
            "vision_projector",
            "mrope_owner",
            "model",
            "llm",
        }
    )


def test_qwen3_5_tome_binding_is_projector_level_generate_only() -> None:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("tome.image.qwen3_5")

    assert binding.backbone == "qwen3_5"
    assert binding.variant == "tome.image.projector_feature"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "llm"}
    )


def test_qwen3_5_divprune_binding_is_projector_level_generate_only() -> None:
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("divprune.image.qwen3_5")

    assert binding.backbone == "qwen3_5"
    assert binding.variant == "divprune.image.projector_feature"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "llm"}
    )


def test_qwen3_5_visionzip_binding_declares_attention_and_merger_mapping() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "visionzip.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "visionzip.image.projector_feature"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {
            "image_features_pre_llm_scatter",
            "vision_encoder",
            "llm",
        }
    )


def test_qwen3_5_otprune_binding_is_projector_level_generate_only() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "otprune.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "otprune.image.projector_feature"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "llm"}
    )


def test_qwen3_5_otprune_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "otprune.image.qwen3_5"
    )
    activation = Qwen3_5OTPruneActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5, "gamma": 0.01}),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "otprune"
    assert model.generate() == {}
    activation.uninstall()
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_prumerge_binding_is_projector_level_and_attention_aware() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "prumerge.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "prumerge.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "vision_encoder", "llm"}
    )


def test_qwen3_5_prumerge_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "prumerge.image.qwen3_5"
    )
    activation = Qwen3_5PruMergeActivation(
        model=model,
        stage=SimpleNamespace(
            config={"retain_ratio": 0.5, "variant": "plus", "cluster_size": 32}
        ),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "prumerge"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_mmtok_binding_is_projector_level_text_conditioned() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "mmtok.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "mmtok.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "vision_encoder", "llm"}
    )


def test_qwen3_5_mmtok_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "mmtok.image.qwen3_5"
    )
    activation = Qwen3_5MMTokActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5, "alpha": 0.5}),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "mmtok"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_gap_binding_is_projector_level_attention_guided() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "gap.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "gap.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "vision_encoder", "llm"}
    )


def test_qwen3_5_gap_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "gap.image.qwen3_5"
    )
    activation = Qwen3_5GAPActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "gap"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_vispruner_binding_is_projector_level_visual_cue_aware() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "vispruner.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "vispruner.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "vision_encoder", "llm"}
    )


def test_qwen3_5_vispruner_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "vispruner.image.qwen3_5"
    )
    activation = Qwen3_5VisPrunerActivation(
        model=model,
        stage=SimpleNamespace(
            config={"retain_ratio": 0.5, "important_ratio": 0.5, "removal_step": 8}
        ),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "vispruner"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_aim_binding_is_projector_level_merge() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "aim.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "aim.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "llm"}
    )


def test_qwen3_5_aim_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "aim.image.qwen3_5"
    )
    activation = Qwen3_5AIMActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "aim"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_restore_binding_is_projector_level_anchor_merge() -> None:
    binding = builtin_multimodal_compression_contracts().binding(
        "restore.image.qwen3_5"
    )
    assert binding.backbone == "qwen3_5"
    assert binding.variant == "restore.image"
    assert {phase.value for phase in binding.supported_phases} == {"generate"}
    assert binding.targets[0].placement.value == "projector_level"
    assert binding.targets[0].required_handles == frozenset(
        {"image_features_pre_llm_scatter", "vision_encoder", "llm"}
    )


def test_qwen3_5_restore_activation_installs_and_removes_hook() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding(
        "restore.image.qwen3_5"
    )
    activation = Qwen3_5RestoreActivation(
        model=model,
        stage=SimpleNamespace(
            config={"retain_ratio": 0.5, "pruning_ratio": 0.5, "important_ratio": 0.5}
        ),
        binding=binding,
    )
    original = model.generate
    activation.install()
    assert activation.runtime_metadata()["algorithm"] == "restore"
    assert model.generate() == {}
    activation.uninstall()
    assert not activation.runtime_metadata()["installed"]
    assert model.generate.__func__ is original.__func__


def test_qwen3_5_tome_activation_installs_and_removes_only_instance_hooks() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

        def forward(self, **kwargs):
            return kwargs

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding("tome.image.qwen3_5")
    activation = Qwen3_5ToMeActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    original_generate = model.generate
    original_forward = model.forward

    activation.install()
    assert activation.runtime_metadata()["installed"] is True
    assert model.forward.__func__ is original_forward.__func__
    assert model.generate() == {}
    assert model.forward() == {}
    activation.uninstall()
    assert activation.runtime_metadata()["installed"] is False
    assert model.generate.__func__ is original_generate.__func__
    assert model.forward.__func__ is original_forward.__func__


def test_qwen3_5_divprune_activation_uses_its_binding_and_cleans_up() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

        def forward(self, **kwargs):
            return kwargs

    model = FakeModel()
    registry = builtin_multimodal_compression_contracts()
    binding = registry.binding("divprune.image.qwen3_5")
    activation = create_qwen3_5_image_compression_executor(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    assert isinstance(activation, Qwen3_5DivPruneActivation)
    original_generate = model.generate
    activation.install()
    metadata = activation.runtime_metadata()
    assert metadata["algorithm"] == "divprune"
    assert metadata["binding"] == "divprune.image.qwen3_5"
    assert metadata["retain_ratio"] == 0.5
    activation.uninstall()
    assert model.generate.__func__ is original_generate.__func__


def test_qwen3_5_divprune_compacts_features_placeholders_and_mrope() -> None:
    torch = pytest.importorskip("torch")
    from transformers.modeling_outputs import BaseModelOutputWithPooling

    class FakeInner:
        rope_deltas = None

        def get_rope_index(self, *, input_ids, mm_token_type_ids, image_grid_thw, attention_mask):
            del mm_token_type_ids, image_grid_thw, attention_mask
            positions = torch.arange(input_ids.shape[1]).view(1, 1, -1)
            return positions.expand(3, input_ids.shape[0], -1), torch.zeros((input_ids.shape[0], 1))

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(image_token_id=99)
            self.model = FakeInner()
            self.features = torch.randn(8, 4)

        def get_image_features(self, pixel_values, grid, *, return_dict):
            del pixel_values, grid, return_dict
            return BaseModelOutputWithPooling(pooler_output=(self.features,))

        def generate(self, **kwargs):
            assert kwargs["input_ids"].shape == (1, 8)
            assert int((kwargs["input_ids"] == self.config.image_token_id).sum()) == 4
            assert kwargs["attention_mask"].shape == kwargs["input_ids"].shape
            assert kwargs["mm_token_type_ids"].shape == kwargs["input_ids"].shape
            assert kwargs["position_ids"].shape == (3, 1, 8)
            assert "pixel_values" not in kwargs
            compressed = kwargs["mm_encoder_outputs"]["image"].pooler_output[0]
            assert compressed.shape == (4, 4)
            assert torch.isfinite(compressed).all()
            return torch.cat((kwargs["input_ids"], torch.tensor([[7]])), dim=-1)

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding("divprune.image.qwen3_5")
    activation = Qwen3_5DivPruneActivation(
        model=model,
        stage=SimpleNamespace(config={"retain_ratio": 0.5}),
        binding=binding,
    )
    input_ids = torch.tensor([[1, 2, 99, 99, 99, 99, 99, 99, 99, 99, 3, 4]])
    mm_types = torch.zeros_like(input_ids)
    mm_types[:, 2:10] = 1
    original = input_ids.clone()
    activation.install()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_types,
        pixel_values=torch.ones(8, 3),
        image_grid_thw=torch.tensor([[1, 2, 4]]),
    )
    activation.uninstall()

    assert output.shape == (1, 13)
    assert torch.equal(output[:, : original.shape[1]], original)
    request = activation.runtime_metadata()["requests"][0]
    assert request["image_tokens_before"] == 8
    assert request["image_tokens_after"] == 4
    assert request["mrope_positions_synchronized"] is True
    assert model.model.rope_deltas is None


def test_qwen3_5_executor_rejects_unknown_image_binding() -> None:
    class FakeModel:
        def generate(self, **kwargs):
            return kwargs

    with pytest.raises(BackendIntegrationError, match="Unsupported Qwen3.5 image binding"):
        create_qwen3_5_image_compression_executor(
            model=FakeModel(),
            binding=SimpleNamespace(name="unknown.image.qwen3_5"),
        )


def test_qwen3_5_vqtoken_video_activation_compacts_placeholders_and_mrope() -> None:
    torch = pytest.importorskip("torch")

    class FakeCore:
        rope_deltas = torch.tensor([[17]])

        def get_rope_index(self, *, input_ids, mm_token_type_ids, video_grid_thw, attention_mask):
            del mm_token_type_ids, video_grid_thw, attention_mask
            positions = torch.arange(input_ids.shape[1]).view(1, 1, -1)
            return positions.expand(3, input_ids.shape[0], -1), torch.zeros((input_ids.shape[0], 1))

    class FakeModel:
        def __init__(self):
            self.config = SimpleNamespace(
                video_token_id=99,
                vision_config=SimpleNamespace(
                    spatial_merge_size=2,
                    deepstack_visual_indexes=(),
                ),
            )
            self.model = FakeCore()
            self.features = torch.randn(128, 8)

        def get_video_features(self, pixel_values_videos, video_grid_thw, *, return_dict):
            del pixel_values_videos, video_grid_thw, return_dict
            return SimpleNamespace(pooler_output=(self.features,))

        def generate(self, **kwargs):
            ids = kwargs["input_ids"]
            assert ids.shape[0] == 1
            assert int((ids == 99).sum()) == 32
            assert kwargs["attention_mask"].shape == ids.shape
            assert kwargs["mm_token_type_ids"].shape == ids.shape
            assert kwargs["position_ids"].shape == (3, 1, ids.shape[1])
            assert "pixel_values_videos" not in kwargs
            assert "video_grid_thw" not in kwargs
            compressed = kwargs["mm_encoder_outputs"]["video"].pooler_output[0]
            assert compressed.shape == (32, 8)
            assert torch.isfinite(compressed).all()
            return torch.cat((ids, torch.tensor([[7]])), dim=-1)

    model = FakeModel()
    binding = builtin_multimodal_compression_contracts().binding("vqtoken.video.qwen3_5")
    activation = create_qwen3_5_video_compression_executor(
        model=model,
        stage=SimpleNamespace(config={"num_clusters": 32}),
        binding=binding,
    )
    assert isinstance(activation, Qwen3_5VideoVQTokenActivation)
    input_ids = torch.cat(
        (torch.tensor([[1]]), torch.full((1, 128), 99), torch.tensor([[2]])), dim=1
    )
    mm_types = torch.zeros_like(input_ids)
    mm_types[:, 1:129] = 2
    original_ids = input_ids.clone()

    activation.install()
    output = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        mm_token_type_ids=mm_types,
        pixel_values_videos=torch.ones((1, 3)),
        video_grid_thw=torch.tensor([[2, 16, 16]]),
    )
    metadata = activation.runtime_metadata()
    activation.uninstall()

    assert output.shape == (1, original_ids.shape[1] + 1)
    assert torch.equal(output[:, : original_ids.shape[1]], original_ids)
    assert metadata["requests"][0]["video_tokens_before"] == 128
    assert metadata["requests"][0]["video_tokens_after"] == 32
    assert metadata["requests"][0]["mrope_positions_synchronized"] is True
    assert model.model.rope_deltas.equal(torch.tensor([[17]]))
