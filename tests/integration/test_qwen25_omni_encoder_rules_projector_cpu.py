"""Original encoder selection rules at both native projector boundaries."""

import copy
from dataclasses import replace

import pytest

from mllm_comp.core.vocabulary import Placement
from tests.integration.test_qwen25_omni_encoder_rules_llm_cpu import _stage
from tests.fixtures.qwen25_omni_audio import _tiny_model, _create_controller, _hta_audio_state


def _projector_stage(algorithm, post, sources):
    stage = _stage(algorithm, sources)
    stage.placement = Placement.PROJECTOR_LEVEL
    stage.operation = {"type": "keep", "target_site": "audio_projector_output_post_projector" if post else "audio_encoder_output_pre_projector"}
    if algorithm == "adjacent_token_merging_audio":
        stage.operation["type"] = "merge"
        stage.schedule = {"merge_ratio": 0.25}
    return stage


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio"])
@pytest.mark.parametrize("sources", ["single", "multiple", "multiple_reverse"])
@pytest.mark.parametrize("post", [False, True])
def test_encoder_attention_projector_positions_sources_gradients_and_generate(algorithm, sources, post):
    torch = pytest.importorskip("torch")
    torch.manual_seed(104)
    base = _tiny_model(torch, audio_encoder_layers=3).train()
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = _create_controller(model, _projector_stage(algorithm, post, sources))
        captured, checks, shapes = {}, [], []
        handles = []
        for source in controller.strategy.layers:
            def capture(module, args, kwargs, source=source):
                captured.setdefault(source, (kwargs["hidden_states"].detach().clone(), kwargs["cu_seqlens"].diff().tolist()))
            handles.append(model.thinker.audio_tower.layers[source].self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        selected = torch.arange(8)
        compress = controller.strategy.rule.compress
        def checked(hidden, probe, lengths, **kwargs):
            nonlocal selected
            source = kwargs["layer_index"]
            normalized, chunks = captured[source]
            attention = model.thinker.audio_tower.layers[source].self_attn
            expected = []
            with torch.no_grad():
                for chunk in normalized.split(chunks):
                    q = attention.q_proj(chunk).reshape(len(chunk), 2, -1).transpose(0, 1)
                    k = attention.k_proj(chunk).reshape(len(chunk), 2, -1).transpose(0, 1)
                    expected.append((q @ k.transpose(-2, -1) * attention.scaling).softmax(-1, dtype=torch.float32).sum(-2).mean(0))
                expected = torch.cat(expected).reshape(8, 2).mean(1)[selected]
            torch.testing.assert_close(probe, expected, rtol=0, atol=0)
            result = compress(hidden, probe, lengths, **kwargs)
            selected = selected[result[3][1][0]]
            checks.append(source)
            return result
        controller.strategy.rule.compress = checked
        controller.install()
        handles.append(model.thinker.audio_tower.proj.register_forward_pre_hook(lambda module, args: shapes.append(("input", len(args[0])))))
        handles.append(model.thinker.audio_tower.proj.register_forward_hook(lambda module, args, output: shapes.append(("output", len(output)))))
        try:
            torch.manual_seed(105)
            raw = _hta_audio_state(torch)
            state, plan = controller.prepare_state(raw)
            final_count = plan.token_actions[0].metadata["keep_count"]
            assert plan.placement is Placement.PROJECTOR_LEVEL
            assert plan.object_subtype == controller.execution.target_site
            assert plan.legalization_policy.position_policy == "compact"
            assert plan.this_stage_metadata.stats["lifecycle"] == "native"
            assert state.input_ids.shape[1] == 5 + final_count
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            assert checks == list(controller.strategy.layers)
            assert shapes == [("input", 8 if post else final_count), ("output", final_count)]
            assert output.logits.shape[1] == 5 + final_count
            assert controller.last_report["target_site"] == controller.execution.target_site
            assert controller.last_report["items"][0]["kept_indices"] == selected.tolist()
            assert all(reference.source is None and not reference.audio_scores for reference in controller.encoder_references.values())
            results.append((output.logits.detach(), raw.audio_values.grad.detach(), model.thinker.audio_tower.layers[0].fc1.weight.grad.detach(), model.thinker.audio_tower.proj.weight.grad.detach()))
            if checkpoint is None:
                controller.strategy.rule.compress = compress
                model.eval()
                raw = replace(raw, labels=None, native_model_inputs={k: v for k, v in raw.native_model_inputs.items() if k != "labels"})
                state, plan = controller.prepare_state(raw)
                with controller.request(plan), torch.no_grad():
                    output = model.thinker(**state.native_model_inputs, use_cache=True)
                    assert [output.past_key_values.get_seq_length(i) for i in range(4)] == [5 + final_count] * 4
                    output = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 6 + final_count), dtype=torch.long), past_key_values=output.past_key_values, cache_position=torch.tensor([5 + final_count]), use_cache=True)
                    assert [output.past_key_values.get_seq_length(i) for i in range(4)] == [6 + final_count] * 4
                state, plan = controller.prepare_state(raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, 7 + final_count)
        finally:
            for handle in handles:
                handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("algorithm", ["encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"])
@pytest.mark.parametrize("post", [False, True])
def test_encoder_attention_projector_identity_preserves_native_output(algorithm, post):
    torch = pytest.importorskip("torch")
    torch.manual_seed(106)
    original = _tiny_model(torch, audio_encoder_layers=3).train()
    model = copy.deepcopy(original)
    raw = _hta_audio_state(torch)
    native = original.thinker(**raw.native_model_inputs, use_cache=False)
    native.loss.backward()
    expected_gradient = raw.audio_values.grad.clone()
    raw.audio_values.grad = None
    stage = _projector_stage(algorithm, post, "multiple")
    stage.schedule = ({"merge_ratio": 0.0} if algorithm == "adjacent_token_merging_audio" else {"keep_ratio": 1.0} if algorithm == "encoder_attention_topk_audio" else {"policy": "fixed", "pruning_rate": 0.0})
    controller = _create_controller(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(raw)
        with controller.request(plan):
            assert all(not reference.active for reference in controller.encoder_references.values())
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        output.loss.backward()
        torch.testing.assert_close(output.logits, native.logits, rtol=0, atol=0)
        torch.testing.assert_close(raw.audio_values.grad, expected_gradient, rtol=0, atol=0)
    finally:
        controller.uninstall()
