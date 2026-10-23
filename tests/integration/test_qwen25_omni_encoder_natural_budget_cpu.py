"""Natural Encoder budgets through the native Omni forward and checkpoint loop."""

import copy

import pytest

from mllm_comp.core.errors import InvariantError
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures.qwen25_omni_audio import _create_controller, _tiny_model


def _stage(position):
    algorithm = "unsupervised_segmentation_audio"
    return CompressionStageConfig.from_dict({
        "name": "encoder-unseg", "placement": "encoder_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "merge",
            "target_site": "audio_encoder_layer_post_mhsa_pre_ffn" if position == "mid" else "audio_encoder_layer_output_post_block",
            "layers": [0, 2] if position == "multiple" else [0 if position == "initial" else 1],
            **({"layer_edge": "before"} if position in {"initial", "before"} else {})},
    })


def _state(torch, features, lengths):
    counts = [((length + 1) // 2) // 2 for length in lengths]
    ids, spans = [1], []
    for i, count in enumerate(counts):
        spans.append(TokenSpan(0, len(ids), len(ids) + count, modality="audio"))
        ids.extend([99] * count + [i + 2])
    ids = torch.tensor([ids])
    lengths = torch.tensor(lengths)
    mask = torch.arange(features.shape[-1])[None, :] < lengths[:, None] if features.ndim == 3 else None
    return ForwardState(
        input_ids=ids, attention_mask=torch.ones_like(ids),
        labels=torch.tensor([[-100] * (ids.shape[1] - 1) + [3]]),
        audio_values=features, audio_feature_lengths=lengths, audio_attention_mask=mask,
        native_model_inputs={"input_ids": ids, "input_features": features,
                             **({"feature_attention_mask": mask} if mask is not None else {})},
        modality_token_spans=spans,
        modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0, "padding_side": "right",
                           "expected_token_counts_by_sample": [{"audio": sum(counts)}]},
    )


@pytest.mark.parametrize("position", ["initial", "before", "full", "mid", "multiple"])
@pytest.mark.parametrize("packed", [False, True])
def test_unseg_encoder_natural_budget_and_checkpoint(position, packed):
    torch = pytest.importorskip("torch")
    torch.manual_seed(99)
    original = _tiny_model(torch, audio_encoder_layers=3).train()
    features = torch.randn(4, 138) if packed else torch.randn(2, 4, 77)
    results = []
    for checkpointed in (None, False, True):
        model = copy.deepcopy(original)
        tower = model.thinker.audio_tower
        if checkpointed is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointed})
        controller = _create_controller(model, _stage(position), profile_enabled=True)
        observed = []
        compress = controller.strategy.rule.compress

        def observe(values, **kwargs):
            assert kwargs["keep_count"] is None
            output, indices, details = compress(values, **kwargs)
            with torch.no_grad():
                normalized = torch.nn.functional.normalize(values.detach().float(), dim=-1)
                distance = 1 - (normalized[:-1] * normalized[1:]).sum(-1)
                cuts = [i + 1 for i in range(1, len(distance) - 1)
                        if distance[i] > distance[i - 1] and distance[i] > distance[i + 1]]
                endpoints = [0, *cuts, len(values)]
                expected = torch.stack([values.detach()[start:end].mean(0) for start, end in zip(endpoints, endpoints[1:])])
                torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
                assert details["segments"] == list(zip(endpoints, endpoints[1:]))
            observed.append((len(values), len(output)))
            return output, indices, details

        controller.strategy.rule.compress = observe
        values = features.clone().requires_grad_()
        raw = _state(torch, values, [61, 77])
        attention_widths, ffn_widths = [[] for _ in range(3)], [[] for _ in range(3)]
        handles = []
        for i, layer in enumerate(tower.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, i=i: attention_widths[i].append(kwargs["hidden_states"].shape[0]), with_kwargs=True))
            handles.append(layer.fc1.register_forward_pre_hook(lambda module, args, i=i: ffn_widths[i].append(args[0].shape[0])))
        controller.install()
        try:
            rng = torch.get_rng_state().clone()
            state, plan = controller.prepare_state(raw)
            torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
            preflight = list(observed)
            assert len(preflight) == (36 if position == "multiple" else 18)
            final_chunks = [n for _, n in preflight[-18:]]
            kept = [sum(final_chunks[:8]) // 2, sum(final_chunks[8:]) // 2]
            assert [a.metadata["keep_count"] for a in plan.token_actions] == kept
            assert [a.metadata["final_chunk_lengths"] for a in plan.token_actions] == [final_chunks[:8], final_chunks[8:]]
            assert plan.this_stage_metadata.stats["preflight_audio_encoder_pass"] is True
            assert plan.this_stage_metadata.stats["planned_audio_tokens"] == sum(kept)
            assert state.input_ids.shape[1] == sum(kept) + 3
            observed.clear()
            for row in attention_widths + ffn_widths:
                row.clear()
            with controller.request(plan):
                cnn, output_lengths = tower._get_feat_extract_output_lengths(raw.audio_feature_lengths)
                if packed:
                    output = tower(values, feature_lens=raw.audio_feature_lengths, aftercnn_lens=cnn).last_hidden_state
                    loss = output.square().mean()
                else:
                    native = model.thinker(**state.native_model_inputs, use_cache=False)
                    output, loss = native.logits, native.loss
            loss.backward()
            assert observed == preflight  # Backward replays segments without detecting peaks again.
            assert output_lengths.tolist() == kept
            assert controller.last_report["final_encoder_tokens"] == sum(final_chunks)
            first = sum(n for _, n in preflight[:18])
            expected_attention = [70, 70, first]
            expected_ffn = list(expected_attention)
            if position == "initial":
                expected_attention = expected_ffn = [first] * 3
            elif position == "before":
                expected_attention = expected_ffn = [70, first, first]
            elif position == "mid":
                expected_ffn[1] = first
            elif position == "multiple":
                expected_attention = [70, first, first]
                expected_ffn = [70, first, first]
            assert [row[0] for row in attention_widths] == expected_attention
            assert [row[0] for row in ffn_widths] == expected_ffn
            assert output.shape[0 if packed else 1] == (sum(kept) if packed else sum(kept) + 3)
            results.append((output.detach(), values.grad, *[layer.fc1.weight.grad for layer in tower.layers]))
            if not packed and checkpointed is None:
                model.eval()
                raw.labels = None
                state, plan = controller.prepare_state(raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2,
                                                       do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, state.input_ids.shape[1] + 2)
                assert controller.last_report["audio_forward_calls"] == 1
        finally:
            controller.uninstall()
            for handle in handles:
                handle.remove()
        assert controller._active_request is None
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            assert actual is not None and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected)


def test_unseg_encoder_rejects_empty_pooler_and_releases_preflight():
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch, audio_encoder_layers=3).eval()
    controller = _create_controller(model, _stage("initial"))
    controller.install()
    try:
        with pytest.raises(InvariantError, match="fewer than two tokens"):
            controller.prepare_state(_state(torch, torch.randn(1, 4, 6), [6]))
        assert controller._active_request is None
        state, plan = controller.prepare_state(_state(torch, torch.randn(1, 4, 32), [32]))
        assert state.input_ids.shape[1] >= 3
        assert plan.this_stage_metadata.stats["planned_audio_tokens"] >= 1
    finally:
        controller.uninstall()


@pytest.mark.parametrize("family", ["encoder", "projector"])
def test_unseg_preflight_with_string_offload_device(family):
    torch = pytest.importorskip("torch")
    from accelerate import cpu_offload

    torch.manual_seed(102)
    model = _tiny_model(torch, audio_encoder_layers=3).eval()
    stage = _stage("mid")
    if family == "projector":
        stage.placement = type(stage.placement)("projector_level")
        stage.operation = {"type": "merge", "target_site": "audio_encoder_output_pre_projector"}
    controller = _create_controller(model, stage)
    controller.install()
    cpu_offload(model.thinker, execution_device="cpu")
    raw = _state(torch, torch.randn(2, 4, 77), [61, 77])
    try:
        state, plan = controller.prepare_state(raw)
        with controller.request(plan), torch.no_grad():
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        assert output.logits.shape[1] == state.input_ids.shape[1]
        assert torch.isfinite(output.logits).all()
    finally:
        controller.uninstall()
