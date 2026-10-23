from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.compressors.image.sgl import (  # noqa: E402
    SGLGuideController,
    SGLSpec,
    aggregate_sgl_visual_attention,
    sgl_image_tokens,
)


def test_sgl_sums_prompt_and_generated_attention_across_layers_heads() -> None:
    input_ids = torch.tensor([[10, 99, 20, 21]])
    prefill_0 = torch.zeros(1, 2, 7, 7)
    prefill_1 = torch.zeros(1, 2, 7, 7)
    decode_0 = torch.zeros(1, 2, 1, 8)
    decode_1 = torch.zeros(1, 2, 1, 8)

    prefill_0[0, 0, 5, 1:5] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    prefill_0[0, 1, 6, 1:5] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    prefill_1[0, :, 5:7, 1:5] = 0.25
    decode_0[0, 0, 0, 1:5] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    decode_1[0, 1, 0, 1:5] = torch.tensor([0.1, 0.2, 0.3, 0.4])

    summary = aggregate_sgl_visual_attention(
        ((prefill_0, prefill_1), (decode_0, decode_1)),
        input_ids=input_ids,
        image_token_id=99,
    )

    assert (summary.visual_start, summary.visual_end) == (1, 5)
    assert summary.prompt_query_tokens == 2
    assert summary.generated_query_tokens == 1
    assert summary.guide_layers == 2
    assert summary.guide_layer_calls == 4
    assert torch.allclose(summary.visual_scores, torch.full((4,), 2.0))


def test_sgl_selection_is_stable_source_ordered_and_layout_exact() -> None:
    hidden = torch.arange(14, dtype=torch.float32).reshape(7, 2)
    spec = SGLSpec(
        guide_model_path="guide-checkpoint",
        keep_ratio=0.5,
    )

    result = sgl_image_tokens(
        hidden,
        image_indices=torch.tensor([1, 2, 3, 4]),
        image_scores=torch.tensor([0.1, 0.9, 0.9, 0.2]),
        spec=spec,
        source_layout={"stage": "dense"},
        result_layout={"stage": "sparse"},
    )

    assert result.kept_image_indices.tolist() == [2, 3]
    assert result.kept_indices.tolist() == [0, 2, 3, 5, 6]
    assert result.dropped_indices.tolist() == [1, 4]
    assert torch.equal(result.hidden_states, hidden[result.kept_indices])
    assert result.layout_delta.source_count == 7
    assert result.layout_delta.result_count == 5
    assert result.layout_delta.source_layout == {"stage": "dense"}
    assert result.layout_delta.result_layout == {"stage": "sparse"}
    assert result.metadata["selected_visual_indices"] == (1, 2)


def test_sgl_spec_and_guide_target_identity_fail_closed() -> None:
    spec = SGLSpec(guide_model_path="guide", keep_ratio=0.05)
    assert spec.prune_layer(28) == 1
    assert spec.keep_count(2052) == 102
    with pytest.raises(ValueError, match="mode='sgp'"):
        SGLSpec(guide_model_path="guide", mode="see")
    with pytest.raises(ValueError, match="guide/target"):
        sgl_image_tokens(
            torch.ones(5, 2),
            image_indices=torch.tensor([1, 2]),
            image_scores=torch.tensor([1.0]),
            spec=spec,
        )

    runtime = SimpleNamespace(metadata={})
    controller = SGLGuideController(
        backbone_name="qwen25_vl",
        target_runtime=runtime,
        spec=spec,
    )
    controller.attach()
    controller.stage_scores(torch.tensor([0.25, 0.75]))
    scores = controller.consume_scores(target_visual_tokens=2, device="cpu")
    assert scores.tolist() == [0.25, 0.75]
    assert controller.runtime_metadata()["guide_scores_pending"] is False
    controller.detach()
    assert runtime.metadata == {}


def test_sgl_guide_runs_native_backbone_and_stages_full_generation_scores(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_batch = SimpleNamespace(samples=(object(),))
    guide_processed = SimpleNamespace(
        source_batch=source_batch,
        model_inputs={"input_ids": torch.tensor([[10, 99, 20, 21]])},
        metadata={"image_count": 1, "token_ids": {"image": 99}},
    )
    prefill = torch.zeros(1, 1, 7, 7)
    decode = torch.zeros(1, 1, 1, 8)
    prefill[0, 0, 5:7, 1:5] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    decode[0, 0, 0, 1:5] = torch.tensor([0.4, 0.3, 0.2, 0.1])
    observed: dict[str, object] = {}

    class _GuideModel:
        dtype = torch.float32

        def eval(self) -> None:
            observed["eval"] = True

        def requires_grad_(self, value: bool) -> None:
            observed["requires_grad"] = value

    facade = SimpleNamespace(
        processor_adapter=SimpleNamespace(
            load_processor=lambda *args, **kwargs: object(),
            prepare_inputs=lambda processor, raw, phase: guide_processed,
        ),
        model_adapter=SimpleNamespace(
            load_model=lambda *args, **kwargs: _GuideModel(),
        ),
        state_builder=SimpleNamespace(
            build=lambda processed, raw_batch, phase: SimpleNamespace(),
        ),
        runner=SimpleNamespace(),
    )

    def generate(
        runtime: object,
        state: object,
        *,
        processed_batch: object,
        trace_request: object,
        **kwargs: object,
    ) -> object:
        del runtime, state
        observed["processed_batch"] = processed_batch
        observed["trace_text_attentions"] = trace_request.text_attentions
        observed["trace_full_generation_text_attentions"] = (
            trace_request._full_generation_text_attentions
        )
        observed["generation_kwargs"] = kwargs
        return SimpleNamespace(
            traces=SimpleNamespace(text_attentions=((prefill,), (decode,)))
        )

    facade.runner.generate = generate
    monkeypatch.setattr(
        "mllm_comp.registry.backbones.get_backbone",
        lambda name: facade,
    )
    target_runtime = SimpleNamespace(
        metadata={},
        model_path_or_id="target-checkpoint",
        revision="target-revision",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    controller = SGLGuideController(
        backbone_name="qwen3_vl",
        target_runtime=target_runtime,
        spec=SGLSpec(guide_model_path="guide-checkpoint"),
    )
    controller.attach()
    controller.prepare(
        processed_batch=SimpleNamespace(
            source_batch=source_batch,
            metadata={"image_count": 1},
        ),
        generation_kwargs={"max_new_tokens": 1, "do_sample": False},
    )

    scores = controller.consume_scores(target_visual_tokens=4, device="cpu")
    assert torch.allclose(scores, torch.tensor([0.6, 0.7, 0.8, 0.9]))
    assert observed["processed_batch"] is guide_processed
    assert observed["trace_text_attentions"] is False
    assert observed["trace_full_generation_text_attentions"] is True
    assert observed["generation_kwargs"] == {
        "max_new_tokens": 1,
        "do_sample": False,
        "use_cache": True,
    }
    assert observed["eval"] is True
    assert observed["requires_grad"] is False
    metadata = controller.runtime_metadata()
    assert metadata["last_guide_request"]["guide_layers"] == 1
    assert metadata["last_guide_request"]["prompt_query_tokens"] == 2
    assert metadata["last_guide_request"]["generated_query_tokens"] == 1
    controller.finish_target(success=True)
    controller.detach()
