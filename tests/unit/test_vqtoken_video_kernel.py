from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen3_vl.video_compression import (
    Qwen3VLPackedVideoRequest,
    Qwen3VLVideoCompressionResponse,
    Qwen3VLVideoGrid,
)
from mllm_comp.backbones.vqtoken_video import VQTokenVideoExecutor
from mllm_comp.compressors.video import (
    VQTokenVideoKernel,
    VideoTokenPayload,
    merge_vqtoken_source_groups,
    vqtoken_video,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="VQToken tensor kernel needs PyTorch")


def _features(*, requires_grad: bool = False):
    torch = _torch()
    values = torch.eye(32, dtype=torch.float32).repeat_interleave(2, dim=0)
    values[1::2] *= 2.0
    return values.clone().requires_grad_(requires_grad)


def _coordinates() -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (frame, row, column)
        for frame in range(2)
        for row in range(4)
        for column in range(8)
    )


def _payload(*, requires_grad: bool = False) -> VideoTokenPayload:
    return VideoTokenPayload(
        features=_features(requires_grad=requires_grad),
        media_ids=("video-vqtoken",),
        video_offsets=(0, 64),
        coordinates=_coordinates(),
    )


def test_vqtoken_control_plane_freezes_only_fixed_global_k32() -> None:
    kernel = VQTokenVideoKernel()
    kernel.validate_config({"config": {}})
    kernel.validate_config(
        {
            "modality_contract": {
                "algorithm": "vqtoken",
                "variant": "vqtoken.video",
            },
            "config": {
                "num_clusters": 32,
                "adaptive": False,
                "use_cross_attention": False,
            },
        }
    )
    with pytest.raises(ConfigError, match="non-canonical"):
        kernel.validate_config({"config": {"adaptive": True}})


def test_vqtoken_global_kmeans_is_exact_deterministic_and_differentiable() -> None:
    torch = _torch()
    payload = _payload(requires_grad=True)

    first = vqtoken_video(payload)
    second = VQTokenVideoKernel()(payload)

    expected_groups = tuple((index, index + 1) for index in range(0, 64, 2))
    assert first.source_count == 64
    assert first.result_count == 32
    assert first.source_groups == (expected_groups,)
    assert first.representative_indices == (tuple(index for index in range(0, 64, 2)),)
    assert first.representative_indices[0] == tuple(
        sorted(first.representative_indices[0])
    )
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    torch.testing.assert_close(first.features, torch.eye(32))
    assert first.metadata["clustering_scope"] == "global_per_video"
    assert first.metadata["output_feature_rule"] == "mean_l2_normalized_members"
    assert first.metadata["initialization"] == (
        "deterministic_farthest_first_source_order"
    )

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_vqtoken_keeps_video_boundaries_and_replays_qwen3_deepstack_groups() -> None:
    torch = _torch()
    source = torch.cat((_features(), _features() * 3.0), dim=0)
    payload = VideoTokenPayload(
        features=source,
        media_ids=("video-a", "video-b"),
        video_offsets=(0, 64, 128),
        coordinates=(*_coordinates(), *_coordinates()),
    )

    result = vqtoken_video(payload)

    assert result.video_offsets == (0, 32, 64)
    assert result.source_video_offsets == (0, 64, 128)
    assert all(len(group) == 2 for groups in result.source_groups for group in groups)

    single = vqtoken_video(_payload())
    branch = _features() * 5.0
    merged = merge_vqtoken_source_groups(branch, single)
    torch.testing.assert_close(merged, torch.eye(32))


def test_vqtoken_qwen3_executor_synchronizes_all_deepstack_branches() -> None:
    torch = _torch()
    features = _features()
    grid = Qwen3VLVideoGrid.from_native(
        (2, 8, 16),
        spatial_merge_size=2,
        second_per_grid_seconds=0.5,
    )
    payload = grid.payload(features, media_id="sample-vq:video-0")
    request = Qwen3VLPackedVideoRequest(
        invocation_id="vqtoken-video-1",
        media_id="sample-vq:video-0",
        payload=payload,
        deepstack_features=tuple(features * value for value in (2.0, 3.0, 4.0)),
        grid=grid,
        source_placeholder_positions=tuple(range(1, 65)),
        prompt_input_ids=torch.tensor([[1, *([8] * 64), 2]]),
        source_attention_mask=torch.ones(1, 66, dtype=torch.long),
        source_video_grid_thw=torch.tensor([[2, 8, 16]], dtype=torch.long),
        source_second_per_grid_ts=torch.tensor([0.5]),
    )
    executor = VQTokenVideoExecutor(
        backbone="qwen3_vl",
        model=object(),
        runtime=SimpleNamespace(),
        facade=SimpleNamespace(),
    )

    response = executor._projector_callback(request)

    assert isinstance(response, Qwen3VLVideoCompressionResponse)
    assert response.compression_result.result_count == 32
    assert len(response.deepstack_features) == 3
    assert all(
        tuple(branch.shape) == (32, 32) for branch in response.deepstack_features
    )
    assert (
        response.synchronization_source_groups
        == response.compression_result.source_groups
    )
    assert executor.runtime_metadata()["projector_invocations"] == 1


def test_vqtoken_rejects_identity_budget_zero_tokens_and_noncanonical_config() -> None:
    torch = _torch()
    coordinates = tuple((0, row, column) for row in range(4) for column in range(8))
    identity_budget = VideoTokenPayload(
        features=torch.eye(32),
        media_ids=("identity-budget",),
        video_offsets=(0, 32),
        coordinates=coordinates,
    )
    with pytest.raises(BackendIntegrationError, match="more than 32"):
        vqtoken_video(identity_budget)

    zero = _features()
    zero[0].zero_()
    zero_payload = VideoTokenPayload(
        features=zero,
        media_ids=("zero-token",),
        video_offsets=(0, 64),
        coordinates=_coordinates(),
    )
    with pytest.raises(BackendIntegrationError, match="non-zero"):
        vqtoken_video(zero_payload)
    with pytest.raises(ConfigError, match="non-canonical"):
        vqtoken_video(_payload(), config={"num_clusters": 16})


def _preserve_video_values(request):
    return request["values"]


@pytest.mark.parametrize("custom_rule", [False, True])
def test_vqtoken_rule_metadata_preserves_exact_evidence_scope(custom_rule):
    from dataclasses import replace
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts

    binding = builtin_multimodal_compression_contracts().binding("vqtoken.video.qwen3_vl")
    binding = replace(binding, metadata={**binding.metadata,
        "verified_phases": ("generate",), "generate_evidence_artifact": "fixture-evidence"})
    stage = SimpleNamespace(algorithm="vqtoken", variant="vqtoken.video", config={},
        operation={"rules": {"transform": {"callable":
            "tests.unit.test_vqtoken_video_kernel:_preserve_video_values"}}} if custom_rule else {})
    executor = VQTokenVideoExecutor(backbone="qwen3_vl", model=object(), stage=stage, binding=binding)
    metadata = executor.runtime_metadata()
    assert metadata["canonical_evidence_applicable"] is (not custom_rule)
    assert metadata["real_checkpoint_verified"] is (not custom_rule)
    assert metadata["support_activation_claimed"] is (not custom_rule)
    assert metadata["evidence_artifact"] == (None if custom_rule else "fixture-evidence")
    assert metadata.get("rules_changed", False) is custom_rule
