"""Qwen2.5 source-entry scoring keeps native M-RoPE and full-key normalization."""

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_sparsevlm_positions_cpu import _stage


@pytest.mark.parametrize("first_boundary", [pytest.param(0, id="initial"), 1, 2])
def test_sparsevlm_qwen25_fixed_sources_and_pending_attention_mass(monkeypatch, first_boundary):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image import decoder_selection as adapter
    from mllm_comp.backbones.qwen25_vl import provider_bridge as provider

    torch.manual_seed(274)
    model = image_fixture.model("qwen25_vl", num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, "qwen25_vl")
    stage = _stage("qwen25_vl", (first_boundary, 3), sources=(0, 1))
    stage.config["recycle_ratio"] = 1.0  # Two recycled tokens form a nontrivial mean.
    events, sources, applications, raters = [], {}, [], []
    original_features = provider._decoder_query_keys
    original_kernel = adapter.sparsevlm_image_tokens
    original_raters = adapter.select_sparsevlm_text_raters

    def select_raters(*args, **kwargs):
        result = original_raters(*args, **kwargs)
        raters.append(result.rater_indices.clone())
        return result

    def capture_source(layer, hidden, **kwargs):
        index = next(i for i, candidate in enumerate(decoder.layers) if candidate is layer)
        events.append(("source", index))
        query, images, keys = original_features(layer, hidden, **kwargs)
        # Independent dense Attention oracle: each rater has a causal prefix,
        # including every text key in the softmax denominator.
        rows = []
        for local_query, sequence_query in enumerate(kwargs["instruction_indices"].tolist()):
            heads = []
            for head in range(query.shape[1]):
                logits = keys[:, head] @ query[local_query, head] / query.shape[-1] ** 0.5
                logits = logits.masked_fill(torch.arange(keys.shape[0], device=keys.device) > sequence_query, float("-inf"))
                heads.append(logits.float().softmax(dim=0))
            rows.append(torch.stack(heads).mean(dim=0))
        expected = torch.stack(rows).mean(dim=0).index_select(0, kwargs["image_indices"])
        sources[index] = (int(hidden.shape[1]), kwargs["instruction_indices"].clone(), expected)
        return query, images, keys

    def apply_kernel(hidden, **kwargs):
        index = kwargs["layer_index"]
        events.append(("apply", index))
        expected = sources[index][2]
        if index == 1 and first_boundary == 2:
            prior_images, prior = applications[0]
            delta = prior.layout_delta
            destinations = sorted({delta.old_to_new[int(source)] for source in prior_images
                if delta.old_to_new[int(source)] is not None})
            expected = torch.stack([sum((expected[local] for local, source in enumerate(prior_images)
                if delta.old_to_new[int(source)] == destination), expected.new_zeros(()))
                for destination in destinations])
        torch.testing.assert_close(kwargs["image_attention_scores"], expected)
        result = original_kernel(hidden, **kwargs)
        applications.append((kwargs["image_indices"].clone(), result))
        return result

    monkeypatch.setattr(provider, "_decoder_query_keys", capture_source)
    monkeypatch.setattr(adapter, "select_sparsevlm_text_raters", select_raters)
    monkeypatch.setattr(adapter, "sparsevlm_image_tokens", apply_kernel)
    handles = [layer.self_attn.o_proj.register_forward_pre_hook(
        lambda module, args, index=index: events.append(("attention", index)))
        for index, layer in enumerate(decoder.layers)]
    activation = image_fixture.activation(model, stage)
    try:
        activation.install()
        with torch.no_grad():
            model(**image_fixture.inputs(torch, "qwen25_vl"), use_cache=False)
        assert len(raters) == 1
        assert events.index(("source", 0)) < events.index(("attention", 0))
        assert events.index(("source", 1)) < events.index(("attention", 1))
        prior = applications[0][1]
        assert any(len(group) > 1 for group in prior.layout_delta.merge_groups)
        if first_boundary <= 1:
            assert events.index(("apply", 0)) < events.index(("source", 1))
            assert sources[1][0] == prior.layout_delta.result_count
            assert sources[1][1].tolist() == [prior.layout_delta.old_to_new[index] for index in raters[0].tolist()]
        else:
            assert events.index(("source", 1)) < events.index(("apply", 0))
            assert sources[1][0] == sources[0][0]
        records = activation.runtime_metadata()["layout_delta_records"]
        assert [record["boundary_layer"] for record in records] == [first_boundary, 3]
        assert [record["application_position"]["index"] for record in records] == [first_boundary, 3]
        assert all(record["result_layout"]["position_policy"] == "contiguous" for record in records)
        if first_boundary == 0:
            assert records[0]["result_layout"]["application_position"] == {
                "boundary": "inter_layer", "index": 0}
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
