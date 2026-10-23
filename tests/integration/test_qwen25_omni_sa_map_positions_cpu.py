"""SA-MAP position changes preserve distinct attention and weighted values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.integration.test_qwen25_omni_encoder_natural_budget_cpu import _state
from tests.fixtures.qwen25_omni_audio import _create_controller, _tiny_model


def _stage(family, position):
    algorithm = "similarity_attention_merge_prune_audio"
    sites = ("audio_encoder_layer_post_mhsa_pre_ffn", "audio_encoder_layer_output_post_block") if family == "encoder" else (
        "llm_decoder_layer_post_attention_audio_tokens", "llm_decoder_layer_output_audio_tokens")
    return CompressionStageConfig.from_dict({
        "name": f"sa-map-{family}", "placement": f"{family}_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "merge_prune", "target_site": sites[position != "mid"],
            "layers": [1, 2] if position == "multiple" else [0 if position == "initial" else 1],
            **({"layer_edge": "before"} if position in {"initial", "before"} else {})},
        "schedule": {"keep_ratio": 0.5}, "criterion": {"similarity_threshold": 1.0},
    })


@pytest.mark.parametrize("family", ["encoder", "llm"])
@pytest.mark.parametrize("position", ["initial", "before", "mid", "full", "multiple"])
def test_sa_map_actual_positions_and_checkpoint(family, position):
    torch = pytest.importorskip("torch")
    torch.manual_seed(104)
    layers = 2 if family == "encoder" and position == "initial" else 3
    original = _tiny_model(torch, audio_encoder_layers=layers).train()
    features = torch.randn(2, 4, 77)
    results = []
    for checkpointed in (None, False, True):
        model = copy.deepcopy(original)
        if checkpointed is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointed})
        controller = _create_controller(model, _stage(family, position), profile_enabled=True)
        observed = []
        compress = controller.strategy.rule.compress

        def observe(values, **kwargs):
            scores = kwargs["scores"]
            assert scores.shape == (len(values), 2)
            assert scores.requires_grad is False
            output, indices, details = compress(values, **kwargs)
            with torch.no_grad():
                expected = []
                for index in details["selected_group_indices"].tolist():
                    group = details["groups"][index]
                    weights = scores[group, 0] / scores[group, 0].sum()
                    expected.append((values.detach()[group] * weights.unsqueeze(-1)).sum(0))
                torch.testing.assert_close(output.detach(), torch.stack(expected), rtol=0, atol=0)
            observed.append((len(values), len(output)))
            return output, indices, details

        controller.strategy.rule.compress = observe
        raw = _state(torch, features.clone().requires_grad_(), [61, 77])
        actual_layers = model.thinker.audio_tower.layers if family == "encoder" else model.thinker.model.layers
        attn, ffn = [[] for _ in actual_layers], [[] for _ in actual_layers]
        handles = []
        axis = 0 if family == "encoder" else 1
        for i, layer in enumerate(actual_layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, i=i: attn[i].append(kwargs["hidden_states"].shape[axis]), with_kwargs=True))
            module = layer.fc1 if family == "encoder" else layer.mlp.gate_proj
            handles.append(module.register_forward_pre_hook(lambda module, args, i=i: ffn[i].append(args[0].shape[axis])))
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            per_stage = 18 if family == "encoder" else 2
            assert len(observed) == per_stage * (2 if position == "multiple" else 1)
            base = 70 if family == "encoder" else 37
            text = 0 if family == "encoder" else 3
            first = text + sum(count for _, count in observed[:per_stage])
            final = text + sum(count for _, count in observed[-per_stage:])
            expected_attn = [base, base, first] + ([first] if family == "llm" else [])
            expected_ffn = list(expected_attn)
            if position == "initial":
                expected_attn = expected_ffn = [first] * len(actual_layers)
            elif position == "before":
                expected_attn = expected_ffn = [base] + [first] * (len(actual_layers) - 1)
            elif position == "mid":
                expected_ffn[1] = first
            elif position == "multiple" and family == "llm":
                expected_attn[-1] = expected_ffn[-1] = final
            assert [row[0] for row in attn] == expected_attn
            assert [row[0] for row in ffn] == expected_ffn
            expected_width = state.input_ids.shape[1] if family == "encoder" else final
            assert output.logits.shape == (1, expected_width, 128)
            assert controller.runtime_metadata()["score_layers_zero_based"] == ([] if family == "llm" else [layers - 2])
            results.append((output.logits.detach(), raw.audio_values.grad, *[layer.fc1.weight.grad for layer in model.thinker.audio_tower.layers],
                            *[layer.mlp.gate_proj.weight.grad for layer in model.thinker.model.layers]))
            if checkpointed is None:
                model.eval()
                raw.labels = None
                state, plan = controller.prepare_state(raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2,
                                                       do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, state.input_ids.shape[1] + 2)
            if family == "llm":
                assert controller.encoder_references[None].source is None
                assert controller.encoder_references[None].sequence_scores is None
        finally:
            controller.uninstall()
            for handle in handles:
                handle.remove()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            assert actual is not None and torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("pooled", [False, True])
def test_sa_map_attention_reductions_match_full_block_matrix(pooled):
    from mllm_comp.backbones.qwen25_omni.audio_compression import sa_map_audio_importance

    torch = pytest.importorskip("torch")
    torch.manual_seed(105)
    attention = _tiny_model(torch, audio_encoder_layers=3).thinker.audio_tower.layers[1].self_attn
    chunks, lengths = [4, 3, 4, 1], [7, 5]
    hidden = torch.randn(12, 12)
    with torch.no_grad():
        full = torch.zeros(2, 12, 12)
        start = 0
        for n, values in zip(chunks, hidden.split(chunks)):
            q = attention.q_proj(values).reshape(n, 2, -1).transpose(0, 1)
            k = attention.k_proj(values).reshape(n, 2, -1).transpose(0, 1)
            full[:, start:start + n, start:start + n] = (q @ k.transpose(-1, -2) * attention.scaling).softmax(-1)
            start += n
        expected_merge, expected_prune = [], []
        start = 0
        for n in lengths:
            used = n // 2 * 2 if pooled else n
            values = full[:, start:start + used, start:start + used]
            if pooled:
                merge = values.max(0).values.sum(0).reshape(-1, 2).sum(-1) / used
                prune = values.mean(0).reshape(used // 2, 2, used // 2, 2).mean(1).sum(-1).diagonal()
            else:
                merge = values.max(0).values.mean(0)
                prune = values.mean(0).diagonal()
            expected_merge.append(merge)
            expected_prune.append(prune)
            start += n
        scores = sa_map_audio_importance(attention, hidden, chunks, lengths, pooled=pooled)
        torch.testing.assert_close(scores["merge"], torch.cat(expected_merge))
        torch.testing.assert_close(scores["prune"], torch.cat(expected_prune))
        assert not torch.allclose(scores["merge"], scores["prune"])


@pytest.mark.parametrize("family,layers,position,message", [
    ("encoder", 1, "initial", "no layer"), ("llm", 1, "initial", "no layer"),
    ("encoder", 3, "initial", "precedes scoring"),
])
def test_sa_map_rejects_missing_or_future_fixed_source(family, layers, position, message):
    from mllm_comp.core.errors import InvariantError

    torch = pytest.importorskip("torch")
    model = _tiny_model(torch, audio_encoder_layers=layers)
    with pytest.raises(InvariantError, match=message):
        _create_controller(model, _stage(family, position))
