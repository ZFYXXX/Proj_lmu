from __future__ import annotations

from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_vl.video_compression import (
    PinnedQwen25VLVideoBridge,
)
from mllm_comp.backbones.qwen3_vl.video_compression import PinnedQwen3VLVideoBridge
from mllm_comp.backbones.prunevid_video import PruneVidVideoExecutor
from mllm_comp.compressors.video import (
    PruneVidVideoKernel,
    VideoTokenPayload,
    merge_prunevid_source_groups,
    plan_prunevid_vtp,
    prunevid_video,
    prunevid_vtp_visual_scores,
    prunevid_windows,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="PruneVid tensor kernel needs PyTorch")


def _payload(*, requires_grad: bool = False):
    torch = _torch()
    frame_count = 4
    height = 4
    width = 8
    tokens_per_frame = height * width
    values = []
    coordinates = []
    for frame in range(frame_count):
        for spatial in range(tokens_per_frame):
            if spatial < 16:
                feature = [1.0, spatial / 32.0, 0.0, 0.0]
            else:
                feature = [0.0, 0.0, 0.0, 0.0]
                feature[frame] = 1.0 + spatial / 64.0
            values.append(feature)
            coordinates.append((frame, spatial // width, spatial % width))
    features = torch.tensor(values, dtype=torch.float32, requires_grad=requires_grad)
    return VideoTokenPayload(
        features=features,
        media_ids=("video-prunevid",),
        video_offsets=(0, frame_count * tokens_per_frame),
        coordinates=tuple(coordinates),
    )


def test_prunevid_projector_is_deterministic_complete_and_differentiable() -> None:
    payload = _payload(requires_grad=True)

    first = prunevid_video(payload)
    second = PruneVidVideoKernel()(payload)

    assert first.source_count == 128
    assert first.result_count == 40
    assert first.compression_applied
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    assert first.representative_indices[0] == tuple(
        sorted(first.representative_indices[0])
    )
    assert first.metadata["temporal_tie_policy"] == "stable_source_index"
    represented = sorted(index for group in first.source_groups[0] for index in group)
    assert represented == list(range(128))
    assert all(
        group[0] <= following[0]
        for group, following in zip(first.source_groups[0], first.source_groups[0][1:])
    )

    windows = prunevid_windows(first)
    assert len(windows) == 1
    assert windows[0].frame_count == 4
    assert len(windows[0].static_result_indices) == 8
    assert tuple(
        len(indices) for indices in windows[0].dynamic_result_indices_by_frame
    ) == (8, 8, 8, 8)

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_prunevid_vtp_uses_explicit_static_and_per_frame_dynamic_sets() -> None:
    torch = _torch()
    projector = prunevid_video(_payload())
    visual_positions = tuple(range(2, 2 + projector.result_count))
    query_positions = (42, 43, 44)
    attention = torch.zeros(2, 45, 45, dtype=torch.float32)
    attention[:, list(query_positions), 2:42] = torch.arange(
        projector.result_count, dtype=torch.float32
    )

    scores = prunevid_vtp_visual_scores(
        attention,
        visual_sequence_indices=visual_positions,
        query_sequence_indices=query_positions,
    )
    plan = plan_prunevid_vtp(
        sequence_length=45,
        visual_sequence_indices=visual_positions,
        visual_scores=scores,
        projector_result=projector,
    )

    window = prunevid_windows(projector)[0]
    expected = sorted(window.static_result_indices[-3:])
    for frame_indices in window.dynamic_result_indices_by_frame:
        expected.extend(frame_indices[-3:])
    assert plan.kept_visual_indices == tuple(sorted(expected))
    assert len(plan.kept_visual_indices) == 15
    assert plan.visual_layout_delta.source_count == 40
    assert plan.visual_layout_delta.result_count == 15
    assert plan.sequence_layout_delta.source_count == 45
    assert plan.sequence_layout_delta.result_count == 20
    assert plan.kept_sequence_indices[:2] == (0, 1)
    assert plan.kept_sequence_indices[-3:] == query_positions
    assert plan.metadata["cache_policy"] == (
        "compact_layers_zero_through_selected_inclusive"
    )


def test_prunevid_executor_freezes_projector_and_vtp_decision_records() -> None:
    torch = _torch()
    payload = _payload()
    executor = PruneVidVideoExecutor(
        backbone="llava_onevision",
        model=object(),
    )
    request = SimpleNamespace(
        payload=payload,
        invocation_id="prunevid-record-test",
        media_id="video-prunevid",
    )

    projector = executor._projector_callback(request)
    visual_positions = tuple(range(2, 2 + projector.result_count))
    vtp = plan_prunevid_vtp(
        sequence_length=projector.result_count + 5,
        visual_sequence_indices=visual_positions,
        visual_scores=torch.arange(projector.result_count, dtype=torch.float32),
        projector_result=projector,
    )
    cache = {
        "policy": "compact_prefix",
        "boundary_layer": 10,
        "boundary": "after_layer",
        "source_length": vtp.sequence_layout_delta.source_count,
        "result_length": vtp.sequence_layout_delta.result_count,
        "touched_layers": list(range(11)),
    }
    executor._record_vtp(vtp, cache)

    metadata = executor.runtime_metadata()
    algorithm = metadata["algorithm_records"][0]
    vtp_record = metadata["vtp_layout_deltas"][0]
    assert algorithm["source_count"] == 128
    assert algorithm["result_count"] == 40
    assert algorithm["frame_count"] == 4
    assert algorithm["grid_height"] == 4
    assert algorithm["grid_width"] == 8
    assert algorithm["window_frame_coverage_exact"] is True
    assert algorithm["result_window_partition_exact"] is True
    assert algorithm["source_partition_exact"] is True
    assert algorithm["static_group_frame_coverage_exact"] is True
    assert algorithm["dynamic_group_frame_ownership_exact"] is True
    assert algorithm["expected_vtp_keep_count"] == 15
    assert vtp_record["source_count"] == 40
    assert vtp_record["result_count"] == 15
    assert vtp_record["sequence_layout_delta"]["source_count"] == 45
    assert vtp_record["sequence_layout_delta"]["result_count"] == 20
    assert vtp_record["cache"]["touched_layers"] == list(range(11))


def test_prunevid_replays_identical_groups_on_qwen3_deepstack_branch() -> None:
    torch = _torch()
    payload = _payload()
    projector = prunevid_video(payload)
    branch = torch.arange(128 * 4, dtype=torch.float32).reshape(128, 4)

    merged = merge_prunevid_source_groups(branch, projector)

    assert tuple(merged.shape) == (projector.result_count, 4)
    for result_index, group in enumerate(projector.source_groups[0]):
        index = torch.tensor(group, dtype=torch.long)
        assert torch.equal(
            merged[result_index],
            branch.index_select(0, index).mean(dim=0),
        )

    executor = PruneVidVideoExecutor(backbone="qwen3_vl", model=object())
    response = executor._projector_callback(
        SimpleNamespace(
            payload=payload,
            deepstack_features=(branch, branch + 1, branch + 2),
            invocation_id="prunevid-qwen3-deepstack",
            media_id="video-prunevid",
        )
    )
    algorithm = executor.runtime_metadata()["algorithm_records"][0]
    assert len(response.deepstack_features) == 3
    assert algorithm["deepstack_branch_count"] == 3
    assert algorithm["deepstack_source_shapes"] == ((128, 4),) * 3
    assert algorithm["deepstack_result_shapes"] == ((projector.result_count, 4),) * 3
    for offset, actual in enumerate(response.deepstack_features):
        torch.testing.assert_close(actual, merged + offset)


@pytest.mark.parametrize(
    ("bridge_type", "splice"),
    (
        (
            PinnedQwen25VLVideoBridge,
            lambda projector, plan: SimpleNamespace(
                result=projector,
                plan=plan,
            ),
        ),
        (
            PinnedQwen3VLVideoBridge,
            lambda projector, plan: SimpleNamespace(
                response=SimpleNamespace(compression_result=projector),
                plan=plan,
            ),
        ),
    ),
)
def test_prunevid_vtp_recomputes_qwen_decode_rope_delta(
    bridge_type,
    splice,
) -> None:
    torch = _torch()
    projector = prunevid_video(_payload())
    visual_positions = tuple(range(2, 2 + projector.result_count))
    scores = torch.arange(projector.result_count, dtype=torch.float32)
    vtp = plan_prunevid_vtp(
        sequence_length=45,
        visual_sequence_indices=visual_positions,
        visual_scores=scores,
        projector_result=projector,
    )
    prompt = SimpleNamespace(
        inputs_embeds=torch.zeros(1, 45, 4),
        attention_mask=torch.ones(1, 45, dtype=torch.long),
        position_ids=torch.arange(45, dtype=torch.long)
        .view(1, 1, 45)
        .expand(4, 1, 45)
        .clone(),
        visual_pos_masks=torch.tensor(
            [[index in visual_positions for index in range(45)]],
            dtype=torch.bool,
        ),
        deepstack_visual_embeds=(torch.zeros(projector.result_count, 4),) * 3,
    )
    plan = SimpleNamespace(
        retained_placeholder_positions=visual_positions,
        result_placeholder_positions=visual_positions,
    )
    prompt.plan = plan
    model = SimpleNamespace(model=SimpleNamespace())
    bridge = bridge_type(model, _contract_verifier=lambda _model: None)
    splice_token = bridge._active_splice.set(splice(projector, plan))
    prompt_token = bridge._prompt_splice.set(prompt)
    rope_token = bridge._vtp_rope_deltas.set(None)
    try:
        bridge.apply_prunevid_vtp_layout(vtp)
        rope_deltas = bridge._vtp_rope_deltas.get()
        assert rope_deltas.reshape(-1).tolist() == [25]
        latest_vtp = bridge.runtime_metadata()["latest_vtp"]
        assert latest_vtp["rope_deltas"] == [25]
        source_mrope = tuple(
            (position, position, position) for position in visual_positions
        )
        expected_result = tuple(
            source_mrope[index] for index in vtp.kept_visual_indices
        )
        assert latest_vtp["position_policy"] == "gather_native_mrope"
        assert latest_vtp["source_visual_mrope"] == source_mrope
        assert latest_vtp["result_visual_mrope"] == expected_result
        assert latest_vtp["active_position_count"] == 20
        assert latest_vtp["max_active_rotary_position"] == 44
        assert latest_vtp["expected_rope_delta"] == 25
        assert latest_vtp["rope_delta_exact"] is True
        if bridge_type is PinnedQwen3VLVideoBridge:
            assert latest_vtp["visual_pos_masks_compacted_exact"] is True
            assert latest_vtp["deepstack_branch_count"] == 3
            assert latest_vtp["deepstack_consumed_branch_count"] == 3
            assert latest_vtp["deepstack_completed_before_vtp"] is True
    finally:
        bridge._vtp_rope_deltas.reset(rope_token)
        bridge._prompt_splice.reset(prompt_token)
        bridge._active_splice.reset(splice_token)


def test_prunevid_rejects_irregular_grid_and_noncanonical_config() -> None:
    torch = _torch()
    irregular = VideoTokenPayload(
        features=torch.ones(3, 2),
        media_ids=("irregular",),
        video_offsets=(0, 3),
        coordinates=((0, 0, 0), (0, 0, 1), (1, 0, 1)),
    )
    with pytest.raises(BackendIntegrationError, match="rectangular frame grid"):
        prunevid_video(irregular)
    with pytest.raises(ConfigError, match="non-canonical"):
        PruneVidVideoKernel().validate_config({"config": {"tau": 0.7}})
