from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

import mllm_comp.backbones.metok_video as metok_provider
from mllm_comp.backbones.llava_onevision.video_compression import (
    PinnedLlavaOneVisionVideoBridge,
)
from mllm_comp.backbones.metok_video import (
    METokVideoExecutor,
    METokVideoSignalProvider,
)
from mllm_comp.backbones.qwen25_vl.video_compression import (
    PinnedQwen25VLVideoBridge,
)
from mllm_comp.backbones.qwen3_vl.video_compression import PinnedQwen3VLVideoBridge
from mllm_comp.compressors.video import (
    METokVideoKernel,
    VideoTokenPayload,
    metok_attention_scores_from_qk,
    metok_projector_events,
    metok_video,
    plan_metok_pruning_stage,
    replay_metok_projector_layout,
)
from mllm_comp.core.errors import BackendIntegrationError, ConfigError


def _torch():
    return pytest.importorskip("torch", reason="METok tensor kernel needs PyTorch")


def _payload(*, requires_grad: bool = False) -> VideoTokenPayload:
    torch = _torch()
    frame_count = 8
    height = width = 8
    directions = (
        (1.0, 0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (-1.0, 0.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
        (0.0, -1.0, 0.0, 0.0),
    )
    values = []
    coordinates = []
    for frame, direction in enumerate(directions):
        for row in range(height):
            for column in range(width):
                variation = (row * width + column) / 10000.0
                values.append(
                    [
                        direction[0],
                        direction[1],
                        variation,
                        direction[3],
                    ]
                )
                coordinates.append((frame, row, column))
    features = torch.tensor(values, dtype=torch.float32, requires_grad=requires_grad)
    raw = torch.arange(frame_count * 4 * 4, dtype=torch.float32).reshape(-1, 4)
    raw_coordinates = tuple(
        (frame, row, column)
        for frame in range(frame_count)
        for row in range(2)
        for column in range(2)
    )
    return VideoTokenPayload(
        features=features,
        media_ids=("video-metok",),
        video_offsets=(0, frame_count * height * width),
        coordinates=tuple(coordinates),
        raw_features=raw,
        raw_video_offsets=(0, int(raw.shape[0])),
        raw_coordinates=raw_coordinates,
        signals={
            "text_context": "Which event matters?",
            "siglip_text_encoder": "frozen-local",
            "prefill_decoder_attention": "native-qk",
            "prefill_layout_state": "before-3-10-19",
            "metok.frame_text_similarity": torch.tensor(
                [0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.05, 0.04],
                dtype=torch.float32,
            ),
        },
        metadata={
            "metok_similarity_source": "native_siglip_visual_features",
            "metok_conditioning_excludes_assistant_answer": True,
            "metok_siglip_local_only": True,
        },
    )


def test_metok_projector_is_deterministic_complete_and_differentiable() -> None:
    payload = _payload(requires_grad=True)

    first = metok_video(payload)
    second = METokVideoKernel()(payload)

    assert first.source_count == 512
    assert first.result_count == 65
    assert first.compression_applied
    assert first.source_groups == second.source_groups
    assert first.representative_indices == second.representative_indices
    assert first.representative_indices[0] == tuple(
        sorted(first.representative_indices[0])
    )
    assert first.metadata["metok_segments"][0] == (
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 8),
    )
    assert first.metadata["metok_important_events"][0] == (0, 1, 2)
    assert first.metadata["metok_keyframes"][0] == ((0,), (2,), (4,), (6,))
    assert first.metadata["metok_event_lengths"][0] == (20, 20, 20, 5)
    assert first.metadata["value_operation"] == (
        "bilinear_interpolate_align_corners_false"
    )
    represented = sorted(index for group in first.source_groups[0] for index in group)
    assert represented == list(range(512))

    first.features.sum().backward()
    assert payload.features.grad is not None
    assert bool((payload.features.grad != 0).any().item())


def test_metok_executor_freezes_projector_and_pruning_decisions() -> None:
    payload = _payload()
    executor = METokVideoExecutor(
        backbone="llava_onevision",
        model=SimpleNamespace(),
        signal_provider=SimpleNamespace(runtime_metadata=lambda: {"fake": True}),
    )
    request = SimpleNamespace(
        payload=payload,
        invocation_id="metok-evidence",
        media_id="video-metok",
        deepstack_features=(),
    )
    projector = executor._projector_callback(request)
    executor._active.set(None)

    metadata = executor.runtime_metadata()
    assert metadata["projector_invocations"] == 1
    assert len(metadata["algorithm_records"]) == 1
    record = metadata["algorithm_records"][0]
    assert record["source_count"] == 512
    assert record["result_count"] == 65
    assert record["frame_count"] == 8
    assert (record["grid_height"], record["grid_width"]) == (8, 8)
    assert record["raw_tokens_per_frame"] == 4
    assert record["segments"] == ((0, 2), (2, 4), (4, 6), (6, 8))
    assert record["important_events"] == (0, 1, 2)
    assert record["keyframes_by_event"] == ((0,), (2,), (4,), (6,))
    assert record["event_lengths"] == (20, 20, 20, 5)
    assert record["frame_records_exact"] is True
    assert record["source_partition_exact"] is True
    assert record["representatives_source_ordered"] is True
    assert record["representatives_first_in_group"] is True
    assert record["non_singleton_source_group_count"] > 0
    assert record["frame_text_similarity_finite"] is True
    assert record["similarity_source"] == "native_siglip_visual_features"
    assert record["assistant_answer_excluded"] is True
    assert record["siglip_local_only"] is True

    lengths, important = metok_projector_events(projector)
    sequence_length = 70
    visual_positions = tuple(range(2, 67))
    retained = tuple(range(projector.result_count))
    cumulative = tuple(range(sequence_length))
    for stage_index, layer in enumerate((3, 10, 19)):
        plan = plan_metok_pruning_stage(
            sequence_length=sequence_length,
            visual_sequence_indices=visual_positions,
            visual_scores=_torch().arange(
                len(visual_positions), dtype=_torch().float32
            ),
            projector_result=projector,
            event_lengths=lengths,
            important_event_indices=important,
            retained_projector_indices=retained,
            cumulative_prompt_indices=cumulative,
            boundary_layer=layer,
            stage_index=stage_index,
        )
        cache = {
            "policy": "compact_prefix",
            "boundary": "before_layer",
            "boundary_layer": layer,
            "source_length": sequence_length,
            "result_length": plan.sequence_layout_delta.result_count,
            "touched_layers": list(range(layer)),
        }
        executor._record_pruning(plan, cache)
        visual_positions = tuple(
            int(plan.sequence_layout_delta.old_to_new[position])
            for position in visual_positions
            if plan.sequence_layout_delta.old_to_new[position] is not None
        )
        lengths = plan.event_lengths_after
        retained = plan.retained_projector_indices
        cumulative = plan.cumulative_prompt_keep_indices
        sequence_length = plan.sequence_layout_delta.result_count

    pruning = executor.runtime_metadata()["pruning_layout_deltas"]
    assert [item["boundary_layer"] for item in pruning] == [3, 10, 19]
    assert [item["result_count"] for item in pruning] == [50, 39, 0]
    assert pruning[-1]["final_visual_removal"] is True
    assert pruning[-1]["sequence_layout_delta"]["result_count"] == 5
    assert pruning[-1]["cache"]["touched_layers"] == list(range(19))


def test_metok_replays_exact_bilinear_projector_on_deepstack() -> None:
    torch = _torch()
    projector = metok_video(_payload())
    branch = torch.arange(512 * 3, dtype=torch.float32).reshape(512, 3)

    replayed = replay_metok_projector_layout(branch, projector)

    assert tuple(replayed.shape) == (65, 3)
    first_frame = branch[:64].reshape(8, 8, 3)
    expected = (
        torch.nn.functional.interpolate(
            first_frame.permute(2, 0, 1).unsqueeze(0),
            size=(4, 4),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .reshape(16, 3)
    )
    assert torch.equal(replayed[:16], expected)


def test_metok_progressive_plans_end_in_a_truthful_zero_visual_layout() -> None:
    torch = _torch()
    projector = metok_video(_payload())
    lengths, important = metok_projector_events(projector)
    sequence_length = 70
    visual_positions = tuple(range(2, 67))
    retained = tuple(range(projector.result_count))
    cumulative = tuple(range(sequence_length))
    plans = []

    for stage_index, layer in enumerate((3, 10, 19)):
        plan = plan_metok_pruning_stage(
            sequence_length=sequence_length,
            visual_sequence_indices=visual_positions,
            visual_scores=torch.arange(len(visual_positions), dtype=torch.float32),
            projector_result=projector,
            event_lengths=lengths,
            important_event_indices=important,
            retained_projector_indices=retained,
            cumulative_prompt_indices=cumulative,
            boundary_layer=layer,
            stage_index=stage_index,
        )
        plans.append(plan)
        visual_positions = tuple(
            int(plan.sequence_layout_delta.old_to_new[position])
            for position in visual_positions
            if plan.sequence_layout_delta.old_to_new[position] is not None
        )
        lengths = plan.event_lengths_after
        retained = plan.retained_projector_indices
        cumulative = plan.cumulative_prompt_keep_indices
        sequence_length = plan.sequence_layout_delta.result_count

    assert tuple(plan.event_lengths_after for plan in plans) == (
        (16, 16, 16, 2),
        (13, 13, 13, 0),
        (0, 0, 0, 0),
    )
    assert tuple(plan.visual_layout_delta.result_count for plan in plans) == (
        50,
        39,
        0,
    )
    assert tuple(plan.sequence_layout_delta.result_count for plan in plans) == (
        55,
        44,
        5,
    )
    assert plans[-1].kept_sequence_indices == (0, 1, 41, 42, 43)
    assert plans[-1].cumulative_prompt_keep_indices == (0, 1, 67, 68, 69)
    assert plans[-1].visual_layout_delta.result_count == 0
    assert plans[-1].metadata["final_visual_removal"] is True


def test_metok_attention_scores_average_full_key_probabilities() -> None:
    torch = _torch()
    queries = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    keys = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
            [[-1.0, 0.0], [0.0, -1.0]],
        ]
    )
    visibility = torch.tensor([[True, True, True, False], [True, True, True, True]])

    actual = metok_attention_scores_from_qk(
        queries,
        keys,
        visual_sequence_indices=(1, 2),
        attention_key_mask=visibility,
    )
    logits = torch.einsum("qhd,thd->tqh", queries, keys) * (2**-0.5)
    logits = logits.masked_fill(
        ~visibility.transpose(0, 1).unsqueeze(-1),
        float("-inf"),
    )
    expected = torch.softmax(logits, dim=0)[[1, 2]].mean(dim=1).mean(dim=1)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize(
    ("bridge_type", "splice"),
    (
        (
            PinnedQwen25VLVideoBridge,
            lambda projector, anchors: SimpleNamespace(
                result=projector,
                plan=SimpleNamespace(retained_placeholder_positions=anchors),
            ),
        ),
        (
            PinnedQwen3VLVideoBridge,
            lambda projector, anchors: SimpleNamespace(
                response=SimpleNamespace(
                    compression_result=projector,
                    deepstack_features=(projector.features,) * 3,
                ),
                plan=SimpleNamespace(retained_placeholder_positions=anchors),
            ),
        ),
    ),
)
def test_metok_qwen_bridge_publishes_final_cumulative_rope_delta(
    bridge_type,
    splice,
) -> None:
    torch = _torch()
    projector = metok_video(_payload())
    plans = _three_plans(projector)
    prompt = SimpleNamespace(
        inputs_embeds=torch.zeros(1, 70, 4),
        attention_mask=torch.ones(1, 70, dtype=torch.long),
        position_ids=torch.arange(70, dtype=torch.long)
        .view(1, 1, 70)
        .expand(4, 1, 70)
        .clone(),
    )
    anchors = tuple(range(2, 67))
    prompt.plan = SimpleNamespace(result_placeholder_positions=anchors)
    prompt.visual_pos_masks = torch.zeros(1, 70, dtype=torch.bool)
    prompt.visual_pos_masks[0, list(anchors)] = True
    model = SimpleNamespace(model=SimpleNamespace())
    bridge = bridge_type(model, _contract_verifier=lambda _model: None)
    splice_token = bridge._active_splice.set(splice(projector, anchors))
    prompt_token = bridge._prompt_splice.set(prompt)
    rope_token = bridge._vtp_rope_deltas.set(None)
    label_token = bridge._training_labels.set(
        torch.full((1, 70), -100, dtype=torch.long)
    )
    try:
        for plan in plans:
            bridge.apply_metok_pruning_layout(plan)
        rope_deltas = bridge._vtp_rope_deltas.get()
        assert rope_deltas.reshape(-1).tolist() == [65]
        assert bridge._training_labels.get().shape == (1, 5)
        latest = bridge.runtime_metadata()["latest_metok"]
        assert [record["boundary_layer"] for record in latest] == [3, 10, 19]
        assert latest[-1]["result_visual_count"] == 0
        assert latest[-1]["rope_deltas"] == [65]
        assert all(
            record["position_policy"] == "gather_native_mrope" for record in latest
        )
        assert all(record["rope_delta_exact"] for record in latest)
        assert all(
            record["result_visual_mrope"]
            == tuple(
                record["source_visual_mrope"][index]
                for index in plans[stage_index].kept_visual_indices
            )
            for stage_index, record in enumerate(latest)
        )
        assert latest[-1]["result_visual_sequence_indices"] == ()
        if bridge_type is PinnedQwen3VLVideoBridge:
            assert all(record["visual_pos_masks_compacted_exact"] for record in latest)
            assert all(
                record["source_visual_pos_mask_indices"]
                == record["source_visual_sequence_indices"]
                for record in latest
            )
            assert all(
                record["result_visual_pos_mask_indices"]
                == record["result_visual_sequence_indices"]
                for record in latest
            )
            assert all(
                record["deepstack_completed_before_pruning"] for record in latest
            )
            assert all(record["deepstack_branch_count"] == 3 for record in latest)
            assert all(
                record["deepstack_consumed_branch_count"] == 3 for record in latest
            )
    finally:
        bridge._training_labels.reset(label_token)
        bridge._vtp_rope_deltas.reset(rope_token)
        bridge._prompt_splice.reset(prompt_token)
        bridge._active_splice.reset(splice_token)


def test_metok_llava_bridge_compacts_only_ignored_labels_at_each_stage() -> None:
    torch = _torch()
    projector = metok_video(_payload())
    plans = _three_plans(projector)
    prepared = (None, None, None, None, torch.zeros(1, 70, 4), None)
    splice = SimpleNamespace(prepared=prepared, compression_result=projector)
    model = SimpleNamespace()
    bridge = PinnedLlavaOneVisionVideoBridge(
        model,
        _contract_verifier=lambda _model: None,
    )
    splice_token = bridge._active_splice.set(splice)
    label_token = bridge._training_labels.set(
        torch.full((1, 70), -100, dtype=torch.long)
    )
    try:
        for plan in plans:
            bridge.apply_metok_pruning_layout(plan)
        assert bridge._training_labels.get().shape == (1, 5)
        latest = bridge.runtime_metadata()["latest_metok"]
        assert latest[-1]["position_policy"] == ("gather_native_one_dimensional_rope")
    finally:
        bridge._training_labels.reset(label_token)
        bridge._active_splice.reset(splice_token)


def test_metok_rejects_missing_provider_signal_and_noncanonical_config() -> None:
    payload = _payload()
    missing = VideoTokenPayload(
        features=payload.features,
        media_ids=payload.media_ids,
        video_offsets=payload.video_offsets,
        coordinates=payload.coordinates,
        raw_features=payload.raw_features,
        raw_video_offsets=payload.raw_video_offsets,
        raw_coordinates=payload.raw_coordinates,
        signals={
            name: value
            for name, value in payload.signals.items()
            if name != "siglip_text_encoder"
        },
    )
    with pytest.raises(BackendIntegrationError, match="siglip_text_encoder"):
        metok_video(missing)
    with pytest.raises(ConfigError, match="non-canonical"):
        METokVideoKernel().validate_config({"config": {"num_segments": 5}})


@pytest.mark.parametrize("backbone", ("llava_onevision", "qwen25_vl", "qwen3_vl"))
def test_metok_executor_installs_provider_and_both_reversible_patches(
    backbone: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []

    class _Owner:
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return None

    class _Model:
        def forward(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            return None

    class _Provider:
        def install(self) -> None:
            events.append((backbone, "provider_install"))

        def uninstall(self) -> None:
            events.append((backbone, "provider_uninstall"))

        def capture_context(self, **kwargs: Any) -> Any:
            del kwargs
            raise AssertionError("fake lifecycle does not execute a video")

        def payload_signals(self, *args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            raise AssertionError("fake lifecycle does not execute a video")

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_provider": backbone}

    class _Handle:
        def remove(self) -> None:
            events.append((backbone, "remove"))

    class _Bridge:
        def register_callback(self, site, callback, *, signal_provider=None):
            assert isinstance(site, str)
            assert callable(callback)
            assert signal_provider is provider
            events.append((backbone, "register"))
            return _Handle()

        def detach(self) -> None:
            events.append((backbone, "detach"))

        def runtime_metadata(self) -> dict[str, Any]:
            return {"fake_bridge": backbone}

    provider = _Provider()
    monkeypatch.setattr(
        metok_provider,
        "_decoder_owner",
        lambda _model, selected: _Owner() if selected == backbone else None,
    )
    monkeypatch.setattr(
        metok_provider,
        "install_video_bridge",
        lambda _model, selected: (
            _Bridge(),
            selected == backbone,
            f'{selected}:metok',
        ),
    )
    executor = METokVideoExecutor(
        backbone=backbone,
        model=_Model(),
        signal_provider=provider,
    )
    executor.install()
    executor.install()
    assert executor.runtime_metadata()["signal_provider"] == {"fake_provider": backbone}
    executor.uninstall()

    assert events == [
        (backbone, "provider_install"),
        (backbone, "register"),
        (backbone, "remove"),
        (backbone, "detach"),
        (backbone, "provider_uninstall"),
    ]


def test_llava_metok_provider_reuses_native_siglip_visual_features() -> None:
    torch = _torch()

    class _Siglip:
        vision_enabled = False

        def native_frame_similarity(
            self,
            text: str,
            raw: Any,
            *,
            frame_count: int,
            tokens_per_frame: int,
        ) -> Any:
            assert text == "user question only"
            assert tuple(raw.shape) == (32, 4)
            assert frame_count == 2
            assert tokens_per_frame == 16
            return torch.tensor([0.8, 0.2], dtype=raw.dtype)

    class _Outer:
        def forward(self, frames: Any) -> Any:
            return torch.arange(
                int(frames.shape[0]) * 16 * 4,
                dtype=frames.dtype,
            ).reshape(int(frames.shape[0]), 16, 4)

    class _Model:
        def __init__(self) -> None:
            self.outer = _Outer()

        def get_vision_tower(self) -> Any:
            return self.outer

    model = _Model()
    provider = METokVideoSignalProvider(
        backbone="llava_onevision",
        model=model,
        siglip_encoder=_Siglip(),
    )
    provider.install()
    frames = torch.zeros(2, 3, 8, 8)
    with provider.capture_context(frames=frames) as capture:
        model.outer.forward(frames)
    provider.payload_signals(
        capture,
        token_count=32,
        coordinates=tuple(
            (frame, row, column)
            for frame in range(2)
            for row in range(4)
            for column in range(4)
        ),
    )
    base = VideoTokenPayload(
        features=torch.ones(32, 4),
        media_ids=("llava-metok",),
        video_offsets=(0, 32),
        coordinates=tuple(
            (frame, row, column)
            for frame in range(2)
            for row in range(4)
            for column in range(4)
        ),
        signals={"text_context": "user question only"},
    )
    enriched = provider.enrich_payload(base, capture)

    assert tuple(enriched.raw_features.shape) == (32, 4)
    assert enriched.signals["metok.frame_text_similarity"].tolist() == pytest.approx(
        [0.8, 0.2]
    )
    assert enriched.signals["prefill_layout_state"] == "before_layers_3_10_19"
    assert enriched.metadata["metok_similarity_source"] == (
        "native_siglip_visual_features"
    )
    provider.uninstall()


@pytest.mark.parametrize(
    ("backbone", "expected_first_raw", "expected_first_pixel"),
    (
        ("qwen25_vl", 49.0, (0.48145466 - 0.5) / 0.5),
        ("qwen3_vl", 1.0, 0.0),
    ),
)
def test_qwen_metok_provider_inverse_patchifies_into_siglip_space(
    backbone: str,
    expected_first_raw: float,
    expected_first_pixel: float,
) -> None:
    torch = _torch()

    class _Siglip:
        vision_enabled = True

        def pixel_frame_similarity(self, text: str, pixels: Any) -> Any:
            assert text == "user question only"
            assert tuple(pixels.shape) == (1, 3, 384, 384)
            assert float(pixels[0, 0, 0, 0]) == pytest.approx(expected_first_pixel)
            return torch.tensor([0.6], dtype=pixels.dtype)

    class _Block:
        def forward(self, hidden_states: Any, **kwargs: Any) -> Any:
            del kwargs
            return hidden_states + 1.0

    block = _Block()
    reverse = torch.arange(3, -1, -1)
    visual = SimpleNamespace(
        blocks=(block,),
        spatial_merge_size=2,
        spatial_merge_unit=4,
        patch_embed=SimpleNamespace(patch_size=2, temporal_patch_size=2),
        get_window_index=lambda _grid: (reverse, torch.tensor([0, 4])),
    )
    model = SimpleNamespace(model=SimpleNamespace(visual=visual))
    provider = METokVideoSignalProvider(
        backbone=backbone,
        model=model,
        siglip_encoder=_Siglip(),
    )
    provider.install()
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long)
    raw = torch.arange(64, dtype=torch.float32).reshape(16, 4)
    with provider.capture_context(grid_thw=grid) as capture:
        block.forward(raw)
    base = VideoTokenPayload(
        features=torch.ones(4, 4),
        media_ids=("qwen-metok",),
        video_offsets=(0, 4),
        coordinates=tuple((0, row, column) for row in range(2) for column in range(2)),
        signals={
            "text_context": "user question only",
            f"{backbone}.pixel_values_videos": torch.zeros(16, 24),
        },
    )
    enriched = provider.enrich_payload(base, capture)

    assert float(enriched.raw_features[0, 0]) == expected_first_raw
    assert enriched.signals["metok.frame_text_similarity"].tolist() == pytest.approx(
        [0.6]
    )
    assert enriched.metadata["metok_similarity_source"] == (
        "qwen_patchified_video_inverse_siglip_384"
    )
    provider.uninstall()


def test_metok_missing_local_siglip_fails_before_patching() -> None:
    provider = METokVideoSignalProvider(
        backbone="llava_onevision",
        model=object(),
    )
    with pytest.raises(BackendIntegrationError, match="metok_siglip_path"):
        provider.prepare()


def _three_plans(projector):
    torch = _torch()
    lengths, important = metok_projector_events(projector)
    sequence_length = 70
    visual_positions = tuple(range(2, 67))
    retained = tuple(range(projector.result_count))
    cumulative = tuple(range(sequence_length))
    plans = []
    for stage_index, layer in enumerate((3, 10, 19)):
        plan = plan_metok_pruning_stage(
            sequence_length=sequence_length,
            visual_sequence_indices=visual_positions,
            visual_scores=torch.arange(len(visual_positions), dtype=torch.float32),
            projector_result=projector,
            event_lengths=lengths,
            important_event_indices=important,
            retained_projector_indices=retained,
            cumulative_prompt_indices=cumulative,
            boundary_layer=layer,
            stage_index=stage_index,
        )
        plans.append(plan)
        visual_positions = tuple(
            int(plan.sequence_layout_delta.old_to_new[position])
            for position in visual_positions
            if plan.sequence_layout_delta.old_to_new[position] is not None
        )
        lengths = plan.event_lengths_after
        retained = plan.retained_projector_indices
        cumulative = plan.cumulative_prompt_keep_indices
        sequence_length = plan.sequence_layout_delta.result_count
    return tuple(plans)
