"""LLaVA AnyRes structure and fixed sources survive delayed SparseVLM merges."""

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_sparsevlm_positions_cpu import _stage


def test_sparsevlm_anyres_newlines_and_pending_attention_mass(monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image import decoder_selection as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider

    torch.manual_seed(352)
    model = image_fixture.llava_model(num_decoder_layers=4).eval()
    model.config.mm_patch_merge_type = "spatial_unpad"
    model.config.image_aspect_ratio = "anyres"
    model.config.image_grid_pinpoints = [[54, 54]]
    model.model.image_newline = torch.nn.Parameter(torch.randn(16))
    activation = image_fixture.activation(model, _stage("llava_onevision", (2, 3), mid=True))
    activation.install()
    decoder = image_fixture.decoder(model, "llava_onevision")
    project, kernel = provider._decoder_query_keys, adapter.sparsevlm_image_tokens
    select_raters = adapter.select_sparsevlm_text_raters
    pending, sampled, native_inputs, stages, rater_calls = {}, {}, {}, [], []
    inside_score = False

    def raters(*args, **kwargs):
        result = select_raters(*args, **kwargs)
        rater_calls.append(result.rater_indices.clone())
        return result

    def observe(index):
        def hook(module, args):
            if not inside_score:
                native_inputs[index] = args[0].detach().clone()
        return hook

    handles = [decoder.layers[index].self_attn.q_proj.register_forward_pre_hook(observe(index)) for index in (0, 2)]

    def capture(layer, hidden, **kwargs):
        nonlocal inside_score
        index = list(decoder.layers).index(layer)
        assert index in (0, 2) and index not in sampled
        sampled[index] = layer.input_layernorm(hidden).detach().clone()
        inside_score = True
        try:
            query, image_keys, keys = project(layer, hidden, **kwargs)
        finally:
            inside_score = False
        # Both sources precede the first application; the original prompt raters
        # and the full native-key softmax denominator must remain unchanged.
        torch.testing.assert_close(kwargs["instruction_indices"], rater_calls[0])
        logits = torch.einsum("qhd,thd->qht", query, keys).float() / query.shape[-1] ** 0.5
        visible = torch.arange(keys.shape[0], device=keys.device)[None, :] <= kwargs["instruction_indices"][:, None]
        probabilities = logits.masked_fill(~visible[:, None, :], float("-inf")).softmax(-1)
        pending[index] = probabilities.mean((0, 1))[kwargs["image_indices"]]
        return query, image_keys, keys

    def apply(hidden, **kwargs):
        stage = len(stages)
        source = (0, 2)[stage]
        assert kwargs["layer_index"] == source
        torch.testing.assert_close(kwargs["image_attention_scores"], pending.pop(source))
        result = kernel(hidden, **kwargs)
        delta = result.layout_delta
        newlines = tuple(kwargs["protected_indices"])
        assert len(newlines) == 27
        for index in newlines:
            target = delta.old_to_new[index]
            assert target is not None
            assert [i for i, mapped in delta.old_to_new.items() if mapped == target] == [index]
            torch.testing.assert_close(result.hidden_states[target], hidden[index], atol=0, rtol=0)
        images = kwargs["image_indices"].tolist()
        for future, scores in pending.items():
            target_ids = sorted({delta.old_to_new[index] for index in images if delta.old_to_new[index] is not None})
            pending[future] = torch.stack([
                scores[[i for i, index in enumerate(images) if delta.old_to_new[index] == target]].sum()
                for target in target_ids
            ])
        assert delta.result_layout["application_position"] == {"boundary": "post_attention_pre_ffn", "index": (2, 3)[stage]}
        assert delta.result_layout["position_policy"] == "contiguous"
        stages.append(delta)
        return result

    monkeypatch.setattr(adapter, "select_sparsevlm_text_raters", raters)
    monkeypatch.setattr(provider, "_decoder_query_keys", capture)
    monkeypatch.setattr(adapter, "sparsevlm_image_tokens", apply)
    ids = torch.tensor([[1, -200, 2, 3]])
    try:
        with torch.no_grad():
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                images=[torch.randn(2, 3, 54, 54)], image_sizes=[(54, 54)], modalities=["image"], use_cache=False)
        assert len(rater_calls) == 1 and len(stages) == 2 and not pending
        assert stages[0].source_count == 1488  # base crop + spatial crop + 27 row newlines + text
        assert any(len(group) > 1 for group in stages[0].merge_groups)
        assert result.logits.shape[1] == stages[-1].result_count
        for source in (0, 2):
            torch.testing.assert_close(sampled[source], native_inputs[source], atol=0, rtol=0)
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
