"""CPU-only contracts for Segmentwise Top-K audio selection."""

from __future__ import annotations

from collections import Counter
import sys
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.audio_compression import (
    Qwen25OmniAudioCompressionController,
    _ActiveAudioRequest,
    _materialize_report_value,
    _require_verified_transformers,
    audio_placeholder_keep_positions,
    legalize_qwen25_omni_audio_tokens,
)
from mllm_comp.compressors.projector_level.segmentwise_topk import (
    SegmentwiseTopKProjectorStrategy,
)
from mllm_comp.compressors.base import CompressionContext
from mllm_comp.compressors.projector_level import (
    SegmentwiseTopKAudio,
    SegmentwiseTopKSpec,
    segment_boundaries,
    segment_budgets,
    segmentwise_random_indices,
    segmentwise_topk_indices,
    segmentwise_uniform_indices,
    target_keep_count,
)
from mllm_comp.core.actions import LegalizationPolicy, TokenAction
from mllm_comp.core.errors import InvariantError
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase


def test_spec_normalizes_stable_stage_fields() -> None:
    spec = SegmentwiseTopKSpec.from_config(
        {
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {
                "score_source": "encoder_last_layer_attention",
                "num_segments": 10,
            },
            "schedule": {
                "allocation": "paper_floor",
                "targets": [{"keep_ratio": 0.25}],
            },
        }
    )

    assert spec.keep_ratio == 0.25
    assert spec.num_segments == 10
    assert spec.allocation == "paper_floor"
    assert spec.score_source == "encoder_last_layer_attention"


def test_deferred_report_values_materialize_recursively() -> None:
    class _Deferred:
        def detach(self) -> "_Deferred":
            return self

        def cpu(self) -> "_Deferred":
            return self

        def tolist(self) -> list[int]:
            return [1, 3]

    assert _materialize_report_value(
        {"items": [{"kept_indices": _Deferred()}]}
    ) == {"items": [{"kept_indices": [1, 3]}]}


def test_real_transformers_controller_rejects_unverified_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Model:
        pass

    _Model.__module__ = "transformers.fixture"
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(__version__="4.58.0"),
    )

    with pytest.raises(InvariantError, match="expected Transformers 4.57.3"):
        _require_verified_transformers(_Model())


@pytest.mark.parametrize(
    "ratio,expected",
    [(1.0, 17), (0.5, 8), (0.25, 4), (0.01, 1)],
)
def test_target_keep_count_uses_floor_and_protects_one_token(
    ratio: float,
    expected: int,
) -> None:
    assert target_keep_count(17, ratio) == expected


def test_balanced_boundaries_cover_sequence_without_overlap() -> None:
    boundaries = segment_boundaries(17, 5)

    assert boundaries == [(0, 4), (4, 8), (8, 11), (11, 14), (14, 17)]
    assert [
        index
        for start, end in boundaries
        for index in range(start, end)
    ] == list(range(17))


def test_balanced_exact_budget_keeps_global_target() -> None:
    boundaries, budgets = segment_budgets(
        length=37,
        keep_count=9,
        num_segments=10,
        allocation="balanced_exact",
    )

    assert len(boundaries) == 9
    assert budgets == [1] * 9
    assert sum(budgets) == 9


def test_paper_floor_budget_exposes_achieved_budget() -> None:
    boundaries, budgets = segment_budgets(
        length=101,
        keep_count=25,
        num_segments=10,
        allocation="paper_floor",
    )

    assert len(boundaries) == 10
    assert budgets == [2] * 10
    assert sum(budgets) == 20


def test_segmentwise_topk_restores_chronological_order() -> None:
    scores = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]

    indices = segmentwise_topk_indices(
        scores,
        keep_ratio=0.5,
        num_segments=2,
    )

    assert indices == [1, 3, 5, 7]
    assert indices == sorted(indices)


def test_global_topk_is_num_segments_one() -> None:
    scores = [0.1, 0.9, 0.2, 0.8, 0.7, 0.3]

    assert segmentwise_topk_indices(
        scores,
        keep_ratio=0.5,
        num_segments=1,
    ) == [1, 3, 4]


def test_uniform_and_random_baselines_preserve_exact_budget() -> None:
    uniform = segmentwise_uniform_indices(41, keep_ratio=0.25, num_segments=10)
    random_a = segmentwise_random_indices(
        41,
        keep_ratio=0.25,
        num_segments=10,
        seed=7,
    )
    random_b = segmentwise_random_indices(
        41,
        keep_ratio=0.25,
        num_segments=10,
        seed=7,
    )

    assert len(uniform) == 10
    assert len(random_a) == 10
    assert random_a == random_b
    assert uniform == sorted(set(uniform))
    assert random_a == sorted(set(random_a))


def test_plugin_plan_defers_scores_but_fixes_per_audio_budget() -> None:
    state = ForwardState(
        modality_token_spans=[
            TokenSpan(sample_index=0, start=2, end=18, modality="audio"),
            TokenSpan(sample_index=0, start=20, end=24, modality="text"),
            TokenSpan(sample_index=1, start=3, end=11, modality="audio"),
        ]
    )
    plugin = SegmentwiseTopKAudio()
    config = {
        "_stage_name": "audio_prune",
        "_execution_phase": "forward",
        "criterion": {"num_segments": 4},
        "schedule": {"keep_ratio": 0.25},
    }

    plan = plugin.plan(
        state=state,
        context=CompressionContext(phase="eval", model=None),
        config=config,
    )

    assert plan.stage_name == "audio_prune"
    assert plan.execution_phase is ExecutionPhase.FORWARD
    assert plan.object_subtype == "audio_encoder_output_pre_projector"
    assert [action.metadata["keep_count"] for action in plan.token_actions] == [4, 2]
    assert all(action.metadata["selection_deferred"] for action in plan.token_actions)
    assert plan.this_stage_metadata is not None
    assert plan.this_stage_metadata.stats["original_audio_tokens"] == 24
    assert plan.this_stage_metadata.stats["requested_audio_tokens"] == 6
    assert plan.this_stage_metadata.stats["planned_audio_tokens"] == 6


def test_paper_floor_plan_distinguishes_requested_and_planned_budget() -> None:
    state = ForwardState(
        modality_token_spans=[
            TokenSpan(sample_index=0, start=0, end=101, modality="audio")
        ]
    )
    plan = SegmentwiseTopKAudio().plan(
        state=state,
        context=CompressionContext(phase="eval", model=None),
        config={
            "criterion": {"num_segments": 10},
            "schedule": {"keep_ratio": 0.25, "allocation": "paper_floor"},
        },
    )

    assert plan.token_actions[0].metadata["requested_keep_count"] == 25
    assert plan.token_actions[0].metadata["keep_count"] == 20
    assert plan.this_stage_metadata is not None
    assert plan.this_stage_metadata.stats["requested_audio_tokens"] == 25
    assert plan.this_stage_metadata.stats["planned_audio_tokens"] == 20


def test_placeholder_compaction_keeps_prefix_budget_per_audio_span() -> None:
    actions = [
        TokenAction(
            op="keep",
            target_indices=None,
            metadata={
                "sample_index": 0,
                "span_start": 2,
                "span_end": 8,
                "keep_count": 2,
            },
        ),
        TokenAction(
            op="keep",
            target_indices=None,
            metadata={
                "sample_index": 1,
                "span_start": 5,
                "span_end": 9,
                "keep_count": 1,
            },
        ),
    ]

    positions = audio_placeholder_keep_positions(
        sequence_length=10,
        batch_size=2,
        token_actions=actions,
    )

    assert positions[0] == [0, 1, 2, 3, 8, 9]
    assert positions[1] == [0, 1, 2, 3, 4, 5, 9]


def _dynamic_action(
    *,
    sample_index: int,
    start: int,
    end: int,
    keep_count: int,
) -> TokenAction:
    return TokenAction(
        op="keep",
        target_indices=None,
        metadata={
            "selector": "segmentwise_topk_audio",
            "target_site": "audio_encoder_output_pre_projector",
            "sample_index": sample_index,
            "audio_unit_index": sample_index,
            "span_start": start,
            "span_end": end,
            "original_count": end - start,
            "keep_count": keep_count,
        },
    )


def _compact_policy() -> LegalizationPolicy:
    return LegalizationPolicy(
        position_policy="compact",
        mask_policy="rebuild_from_tokens",
        label_policy="rebuild_ignore_index",
        coordinate_policy="drop_removed",
    )


def test_torch_legalizer_compacts_right_padded_batch() -> None:
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor(
        [
            [1, 12, 12, 12, 12, 2, 0, 0],
            [1, 12, 12, 2, 0, 0, 0, 0],
        ]
    )
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0],
        ]
    )
    token_type_ids = torch.tensor(
        [
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
        ]
    )
    audio_values = torch.randn(2, 4, 8)
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_type_mask=token_type_ids,
        audio_values=audio_values,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "input_features": audio_values,
            "cu_seq_lens_q": torch.tensor([0, 6, 10]),
            "cu_seq_lens_k": torch.tensor([0, 6, 10]),
            "max_length_q": 6,
            "max_length_k": 6,
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=1, end=5, modality="audio"),
            TokenSpan(sample_index=1, start=1, end=3, modality="audio"),
        ],
        modality_metadata={
            "token_ids": {"audio": 12},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [
                {"audio": 4},
                {"audio": 2},
            ],
            "semantic_token_labels_by_sample": [
                [
                    "ins_special",
                    "audio",
                    "audio",
                    "audio",
                    "audio",
                    "resp_special",
                    "pad",
                    "pad",
                ],
                [
                    "ins_special",
                    "audio",
                    "audio",
                    "resp_special",
                    "pad",
                    "pad",
                    "pad",
                    "pad",
                ],
            ],
        },
    )

    compact = legalize_qwen25_omni_audio_tokens(
        state,
        token_actions=[
            _dynamic_action(sample_index=0, start=1, end=5, keep_count=2),
            _dynamic_action(sample_index=1, start=1, end=3, keep_count=1),
        ],
        policy=_compact_policy(),
        stage_name="audio",
    )

    assert compact.input_ids.tolist() == [
        [1, 12, 12, 2, 0, 0, 0],
        [1, 12, 2, 0, 0, 0, 0],
    ]
    assert compact.attention_mask.tolist() == [
        [1, 1, 1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
    ]
    assert compact.token_type_mask.tolist() == [
        [0, 1, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
    ]
    assert compact.native_model_inputs["token_type_ids"] is compact.token_type_mask
    assert compact.audio_values is audio_values
    assert compact.native_model_inputs["input_features"] is audio_values
    assert not {"cu_seq_lens_q", "cu_seq_lens_k", "max_length_q", "max_length_k"}.intersection(
        compact.native_model_inputs
    )
    assert [(span.start, span.end) for span in compact.modality_token_spans] == [
        (1, 3),
        (1, 2),
    ]
    assert compact.modality_metadata["expected_token_counts_by_sample"] == [
        {"audio": 2},
        {"audio": 1},
    ]
    assert compact.modality_metadata["semantic_token_labels_by_sample"] == [
        ["ins_special", "audio", "audio", "resp_special", "pad", "pad", "pad"],
        ["ins_special", "audio", "resp_special", "pad", "pad", "pad", "pad"],
    ]


def test_legalizer_preserves_audio_boundaries_while_pruning_250_to_62() -> None:
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor(
        [[1, 13, *([12] * 250), 14, 30, 2, 0]],
    )
    attention_mask = torch.tensor([[1] * 255 + [0]])
    labels = [
        "ins_special",
        "audio_special",
        *(["audio"] * 250),
        "audio_special",
        "instruction",
        "resp_special",
        "pad",
    ]
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=2, end=252, modality="audio")
        ],
        modality_metadata={
            "token_ids": {"audio": 12},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [{"audio": 250}],
            "semantic_token_labels_by_sample": [labels],
        },
    )

    compact = legalize_qwen25_omni_audio_tokens(
        state,
        token_actions=[
            _dynamic_action(
                sample_index=0,
                start=2,
                end=252,
                keep_count=62,
            )
        ],
        policy=_compact_policy(),
        stage_name="audio",
    )

    before = Counter(labels)
    after = Counter(
        compact.modality_metadata["semantic_token_labels_by_sample"][0]
    )
    assert (before["audio"], after["audio"]) == (250, 62)
    assert (before["audio_special"], after["audio_special"]) == (2, 2)
    assert compact.modality_metadata["expected_token_counts_by_sample"] == [
        {"audio": 62}
    ]


def test_torch_legalizer_shifts_spans_for_new_left_padding() -> None:
    torch = pytest.importorskip("torch")
    input_ids = torch.tensor(
        [
            [0, 0, 1, 12, 12, 12, 12, 2],
            [0, 0, 0, 0, 1, 12, 12, 2],
        ]
    )
    attention_mask = torch.tensor(
        [
            [0, 0, 1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 1, 1, 1],
        ]
    )
    state = ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        modality_token_spans=[
            TokenSpan(sample_index=0, start=3, end=7, modality="audio"),
            TokenSpan(sample_index=1, start=5, end=7, modality="audio"),
        ],
        modality_metadata={
            "token_ids": {"audio": 12},
            "pad_token_id": 0,
            "padding_side": "left",
            "expected_token_counts_by_sample": [
                {"audio": 4},
                {"audio": 2},
            ],
        },
    )

    compact = legalize_qwen25_omni_audio_tokens(
        state,
        token_actions=[
            _dynamic_action(sample_index=0, start=3, end=7, keep_count=2),
            _dynamic_action(sample_index=1, start=5, end=7, keep_count=1),
        ],
        policy=_compact_policy(),
        stage_name="audio",
    )

    assert compact.input_ids.tolist() == [
        [0, 0, 0, 1, 12, 12, 2],
        [0, 0, 0, 0, 1, 12, 2],
    ]
    assert [(span.start, span.end) for span in compact.modality_token_spans] == [
        (4, 6),
        (5, 6),
    ]


def _projector_controller(*, keep_counts: list[int]) -> tuple[object, object]:
    controller = object.__new__(Qwen25OmniAudioCompressionController)
    controller.encoder_references = {}
    controller.spec = SegmentwiseTopKSpec(
        keep_ratio=0.5,
        score_source="uniform",
    )
    controller.strategy = SegmentwiseTopKProjectorStrategy(controller.spec)
    active = _ActiveAudioRequest(
        original_counts=[4, 6],
        keep_counts=keep_counts,
        sample_indices=[0, 1],
        audio_unit_indices=[0, 1],
        preflight_details=[None, None],
    )
    controller._active_request = active
    return controller, active


def test_torch_projector_hook_splits_concatenated_audio_units() -> None:
    torch = pytest.importorskip("torch")
    controller, active = _projector_controller(keep_counts=[2, 3])
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(10, 1)

    output = controller._compress_projector_input(None, (hidden_states,))

    assert output is not None
    compressed = output[0]
    assert compressed.shape == (5, 1)
    assert int((compressed[:2] < 4).all()) == 1
    assert int((compressed[2:] >= 4).all()) == 1
    assert active.projector_call_index == 2
    assert [item["original_tokens"] for item in active.item_reports] == [4, 6]
    assert [item["kept_tokens"] for item in active.item_reports] == [2, 3]
    assert all(item["indices_chronological"] for item in active.item_reports)


def test_torch_projector_hook_preserves_identity_tensor_for_concatenated_units() -> None:
    torch = pytest.importorskip("torch")
    controller, active = _projector_controller(keep_counts=[4, 6])
    hidden_states = torch.arange(10, dtype=torch.float32).reshape(10, 1)

    output = controller._compress_projector_input(None, (hidden_states,))

    assert output is not None
    assert output[0] is hidden_states
    assert active.projector_call_index == 2
    assert len(active.item_reports) == 2


@pytest.mark.parametrize(
    "config,pattern",
    [
        ({"schedule": {"keep_ratio": 0}}, "keep_ratio"),
        ({"criterion": {"num_segments": 0}}, "num_segments"),
        ({"schedule": {"allocation": "unknown"}}, "allocation"),
        ({"criterion": {"score_source": "hidden_norm"}}, "score_source"),
        ({"operation": {"type": "drop"}}, "operation.type=keep"),
    ],
)
def test_invalid_config_is_rejected(config: dict, pattern: str) -> None:
    with pytest.raises(ValueError, match=pattern):
        SegmentwiseTopKSpec.from_config(config)


@pytest.mark.parametrize("score_source", ["uniform", "random"])
@pytest.mark.parametrize("allocation", ["balanced_exact", "paper_floor"])
def test_segmentwise_layer_adapters_replay_exact_selection(score_source, allocation):
    from mllm_comp.backbones.qwen25_omni.audio_method_adapters import TemporalResamplingEncoderStrategy, TemporalResamplingLLMStrategy

    torch = pytest.importorskip("torch")
    spec = SegmentwiseTopKSpec(keep_ratio=0.6, num_segments=3, score_source=score_source, allocation=allocation)
    rule = SegmentwiseTopKProjectorStrategy(spec)
    original = torch.arange(21., requires_grad=True).reshape(7, 3).detach().requires_grad_()
    for adapter in [TemporalResamplingLLMStrategy(rule, (0,)), TemporalResamplingEncoderStrategy(rule, (0,))]:
        if isinstance(adapter, TemporalResamplingLLMStrategy):
            length = adapter.output_length(7)
            output, representatives, details = adapter.merge_audio_tokens(hidden_states=original, scores=None,
                sample_index=0, slots=None, output_length=length, audio_unit_index=2)
            replay = adapter.replay_merge_audio_tokens(hidden_states=original, replay_plan=details["_checkpoint_replay_plan"])
            item_index = 2
        else:
            output, lengths, _, plan = adapter.compress(original, None, [7], layer_index=0, active_layers=(0,), collect_details=False)
            length = lengths[0]
            representatives = plan[1][0]
            replay = adapter.replay(original, plan)
            item_index = 0
        assert length == (4 if allocation == "balanced_exact" else 3)
        kwargs = dict(keep_ratio=length / 7, num_segments=3, minimum_keep_tokens=length)
        expected = (segmentwise_random_indices(7, seed=42 + item_index, **kwargs) if score_source == "random"
                    else segmentwise_uniform_indices(7, **kwargs))
        assert representatives.tolist() == expected
        torch.testing.assert_close(output, original[expected], rtol=0, atol=0)
        torch.testing.assert_close(replay, output, rtol=0, atol=0)
        gradient = torch.autograd.grad(output.square().sum(), original)[0]
        replay_gradient = torch.autograd.grad(replay.square().sum(), original)[0]
        torch.testing.assert_close(replay_gradient, gradient, rtol=0, atol=0)
