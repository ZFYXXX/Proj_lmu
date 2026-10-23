"""Native encoder-attention rules applied at configurable LLM boundaries."""

import copy
import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures.qwen25_omni_audio import _tiny_model, _create_controller, _hta_audio_state


def _stage(algorithm, position):
    multiple = position.startswith("multiple")
    return CompressionStageConfig.from_dict({
        "name": "encoder-rule-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "activation_scope": {"layers": [2, 0] if position == "multiple_reverse" else [0, 2] if multiple else [2]},
        "operation": {"type": "keep", "layers": [0, 2] if multiple else [0 if position == "initial" else 1],
            "target_site": "llm_decoder_layer_post_attention_audio_tokens" if position == "mid" else "llm_decoder_layer_output_audio_tokens",
            **({"layer_edge": "before"} if position in {"initial", "before"} else {})},
        "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1} if algorithm == "encoder_attention_topk_audio" else {
            "policy": "linear", "pruning_rate": 0.5, "start_pruning_rate": 0.25, "end_pruning_rate": 0.5, "minimum_keep_tokens": 1},
    })


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio"])
@pytest.mark.parametrize("position", ["initial", "mid", "full", "before", "multiple", "multiple_reverse"])
def test_native_encoder_sources_llm_application_and_checkpoint(algorithm, position):
    torch = pytest.importorskip("torch")
    torch.manual_seed(97)
    base = _tiny_model(torch, audio_encoder_layers=3).train()
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        torch.manual_seed(98)
        raw = _hta_audio_state(torch)
        controller = _create_controller(model, _stage(algorithm, position))
        controller.debug_enabled = True
        captured = {}
        handles = []
        for source in controller.stage_layers:
            def capture(module, args, kwargs, source=source):
                captured.setdefault(source, (kwargs["hidden_states"].detach().clone(), kwargs["cu_seqlens"].tolist()))
            handles.append(model.thinker.audio_tower.layers[source].self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        selected = torch.arange(8)
        compress = controller.strategy.rule.compress
        checks = []
        def checked(hidden, probe, lengths, **kwargs):
            nonlocal selected
            source = kwargs["layer_index"]
            normalized, cu = captured[source]
            attention = model.thinker.audio_tower.layers[source].self_attn
            expected = []
            with torch.no_grad():
                for start, end in zip(cu, cu[1:]):
                    chunk = normalized[start:end]
                    q = attention.q_proj(chunk).reshape(end-start, 2, -1).transpose(0, 1)
                    k = attention.k_proj(chunk).reshape(end-start, 2, -1).transpose(0, 1)
                    expected.append((q @ k.transpose(-2, -1) * attention.scaling).softmax(-1, dtype=torch.float32).sum(-2).mean(0))
                expected = torch.cat(expected).reshape(8, 2).mean(1)[selected]
            torch.testing.assert_close(probe, expected, rtol=0, atol=0)
            result = compress(hidden, probe, lengths, **kwargs)
            selected = selected[result[3][1][0]]
            checks.append(source)
            return result
        controller.strategy.rule.compress = checked
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            assert plan.object_subtype == controller.execution.target_site
            for metadata in [plan.this_stage_metadata.stats, *(action.metadata for action in plan.token_actions)]:
                assert metadata["target_site"] == controller.execution.target_site
                if algorithm == "progressive_attention_pruning_audio":
                    assert metadata["pruning_rates"] == [controller.spec.pruning_rate_for_layer(layer, controller.strategy.active_layers) for layer in controller.stage_layers]
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            assert checks == list(controller.stage_layers)
            assert output.logits.shape[1] == 5 + len(selected)
            assert all(r.source is None and r.sequence_scores is None for r in controller.encoder_references.values())
            results.append((output.logits.detach(), raw.audio_values.grad.detach(), model.thinker.audio_tower.layers[-1].fc1.weight.grad.detach(), model.thinker.model.layers[0].mlp.gate_proj.weight.grad.detach()))
            if checkpoint is None:
                from dataclasses import replace
                expected_cache = [13] * 4
                for event in controller.last_report["stages"]:
                    point = event["application_position"]
                    start = point["index"] + (point["boundary"] == "post_attention_pre_ffn")
                    for i in range(start, 4):
                        expected_cache[i] -= event["input_audio_tokens"] - event["output_audio_tokens"]
                controller.strategy.rule.compress = compress
                model.eval()
                eval_raw = replace(raw, labels=None, native_model_inputs={k: v for k, v in raw.native_model_inputs.items() if k != "labels"})
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    cached = model.thinker(**state.native_model_inputs, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == expected_cache
                    cached = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 14), dtype=torch.long), past_key_values=cached.past_key_values, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [n+1 for n in expected_cache]
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, 15)
                assert controller.last_report["decode_forward_calls"] == 1

        finally:
            for handle in handles:
                handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio"])
@pytest.mark.parametrize("boundary", ["mid", "full"])
@pytest.mark.parametrize("family", ["llm", "projector_pre", "projector_post"])
def test_multiple_encoder_sources_across_upstream_layout_change(algorithm, boundary, family):
    from contextlib import ExitStack

    torch = pytest.importorskip("torch")
    torch.manual_seed(101)
    base = _tiny_model(torch, audio_encoder_layers=3).train()
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = CompressionStageConfig.from_dict({
            "name": "encoder", "placement": "encoder_level", "execution_phase": "forward",
            "plugin": "uniform_sampling_audio", "algorithm": "uniform_sampling_audio",
            "variant": "uniform_sampling_audio.audio", "binding": "uniform_sampling_audio.audio.qwen25_omni",
            "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
            "operation": {"type": "keep", "layers": [1], "target_site": "audio_encoder_layer_post_mhsa_pre_ffn" if boundary == "mid" else "audio_encoder_layer_output_post_block"},
            "schedule": {"compression_factor": 2},
        })
        encoder = _create_controller(model, stage)
        target_stage = _stage(algorithm, "multiple")
        if family != "llm":
            from mllm_comp.core.vocabulary import Placement
            target_stage.placement = Placement.PROJECTOR_LEVEL
            target_stage.operation = {"type": "keep", "target_site": "audio_encoder_output_pre_projector" if family == "projector_pre" else "audio_projector_output_post_projector"}
        target = _create_controller(model, target_stage)
        captured, checks = {}, []
        selected = torch.arange(4)
        handles = []
        for source in target.strategy.layers:
            def capture(module, args, kwargs, source=source):
                captured.setdefault(source, (kwargs["hidden_states"].detach().clone(), kwargs["cu_seqlens"].diff().tolist()))
            handles.append(model.thinker.audio_tower.layers[source].self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        compress = target.strategy.rule.compress
        def checked(hidden, scores, lengths, **kwargs):
            nonlocal selected
            source = kwargs["layer_index"]
            # Source 0 precedes the upstream keep; source 2 already sees its result.
            normalized, chunks = captured[source]
            if source == 0:
                normalized = normalized[::2]
                chunks = [length // 2 for length in chunks]
            assert normalized.shape[0] == 8
            attention = model.thinker.audio_tower.layers[source].self_attn
            with torch.no_grad():
                expected = []
                for chunk in normalized.split(chunks):
                    q = attention.q_proj(chunk).reshape(len(chunk), 2, -1).transpose(0, 1)
                    k = attention.k_proj(chunk).reshape(len(chunk), 2, -1).transpose(0, 1)
                    expected.append((q @ k.transpose(-2, -1) * attention.scaling).softmax(-1, dtype=torch.float32).sum(-2).mean(0))
                expected = torch.cat(expected).reshape(4, 2).mean(1)[selected]
            torch.testing.assert_close(scores, expected, rtol=0, atol=0)
            result = compress(hidden, scores, lengths, **kwargs)
            selected = selected[result[3][1][0]]
            checks.append(source)
            return result
        target.strategy.rule.compress = checked
        encoder.install()
        original_observer = encoder._layout_observer
        target.install()
        try:
            torch.manual_seed(102)
            raw = _hta_audio_state(torch)
            state, encoder_plan = encoder.prepare_state(raw)
            state, target_plan = target.prepare_state(state)
            with ExitStack() as requests:
                requests.enter_context(encoder.request(encoder_plan))
                requests.enter_context(target.request(target_plan))
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            assert checks == [0, 2]
            assert captured[0][0].shape[0] == 16 and captured[2][0].shape[0] == 8
            assert all(reference.source is None and reference.sequence_scores is None for reference in target.encoder_references.values())
            results.append((output.logits.detach(), raw.audio_values.grad.detach(), model.thinker.audio_tower.layers[0].fc1.weight.grad.detach()))
            # Request failure must clear every observer's state as well.
            _, target_plan = target.prepare_state(state)
            with pytest.raises(RuntimeError, match="test request failure"):
                with target.request(target_plan):
                    for reference in target.encoder_references.values():
                        reference.source = torch.ones(1)
                    raise RuntimeError("test request failure")
            assert all(reference.source is None and not reference.active for reference in target.encoder_references.values())
        finally:
            for handle in handles:
                handle.remove()
            target.uninstall()
            assert encoder._layout_observer is original_observer
            encoder.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)
