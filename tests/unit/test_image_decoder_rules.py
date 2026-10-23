"""Source rules receive original mathematical inputs, with shape contracts."""

import pytest


def test_original_qk_sources_can_replace_softmax_scoring():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.pyramid_drop import pyramid_attention_scores
    from mllm_comp.compressors.image.halfv import halfv_last_text_attention_scores
    query = torch.tensor([[[1., 2.], [3., 4.]]])
    keys = torch.arange(16, dtype=torch.float32).reshape(4, 2, 2)
    observed = []

    def scorer(request):
        assert request.get("instruction_query", request.get("query")) is query
        assert request["image_keys"] is keys
        observed.append(True)
        return (keys * query).sum(dim=(-1, -2))

    expected = (keys * query).sum(dim=(-1, -2))
    torch.testing.assert_close(pyramid_attention_scores(instruction_query=query, image_keys=keys,
        rules={"source_scorer": scorer}), expected)
    torch.testing.assert_close(halfv_last_text_attention_scores(query, keys,
        rules={"source_scorer": scorer}), expected)
    assert len(observed) == 2


def test_transition_rule_uses_actual_module_pairs_and_preserves_alignment():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.transprune import transprune_layer_transition_variation, TransPruneSpec
    before = torch.arange(30, dtype=torch.float32).reshape(1, 5, 6)
    after = before + 2
    images = torch.tensor([1, 2, 3])
    selected = torch.tensor([0, 2])
    keep = torch.tensor([0, 1, 3, 4])
    spec = TransPruneSpec()
    calls = []

    def scorer(request):
        calls.append(request["module"])
        return (request["after"] - request["before"]).abs().sum(-1)[0][request["image_indices"]]

    result = transprune_layer_transition_variation(before, after, before[:, keep], after[:, keep] + 1,
        image_indices=torch.tensor([1, 2]), attention_image_indices=images,
        selected_visual_indices=selected, direction_rule=spec.direction_rule, softmax_scope=spec.ttv_softmax_scope, rules={"transition_scorer": scorer})
    torch.testing.assert_close(result, torch.full((2,), 30.))
    assert calls == ["attention", "ffn"]


def test_iac_custom_buckets_and_representative_validation():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.finepruner import FinePrunerSpec, finepruner_iac_image_tokens
    hidden = torch.arange(30, dtype=torch.float32).reshape(10, 3).requires_grad_()
    images = torch.arange(1, 9)
    spec = FinePrunerSpec(iac_after_layers=(1, 2), iac_hash_bits=(2, 1), target_visual_tokens=1)
    rules = {"grouper": lambda request: torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
             "iac_selector": lambda request: torch.tensor([1, 3, 5, 7])}
    result = finepruner_iac_image_tokens(hidden, image_indices=images,
        completed_layers=1, initial_visual_tokens=8, spec=spec, rules=rules)
    torch.testing.assert_close(result.hidden_states, hidden[torch.tensor([0, 2, 4, 6, 8, 9])])
    with pytest.raises(ValueError, match="one representative"):
        finepruner_iac_image_tokens(hidden, image_indices=images, completed_layers=1,
            initial_visual_tokens=8, spec=spec, rules={**rules, "iac_selector": lambda request: torch.tensor([0, 1, 2, 3])})
    with pytest.raises(ValueError, match="integer labels"):
        finepruner_iac_image_tokens(hidden, image_indices=images, completed_layers=1,
            initial_visual_tokens=8, spec=spec, rules={"grouper": lambda request: torch.zeros(8)})


def test_sgl_scorer_receives_full_guide_attention_steps():
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image.sgl import aggregate_sgl_visual_attention
    ids = torch.tensor([[1, 99, 2, 3]])
    prefill = torch.ones(1, 2, 7, 7)
    decode = torch.ones(1, 2, 1, 8)
    steps = ((prefill,), (decode,))

    def scorer(request):
        assert request["attentions"] is steps
        # Replace the original layer/head/query sum with just decode-head zero.
        return request["attentions"][-1][0][0, 0, 0, request["visual_start"]:request["visual_end"]] * 3

    result = aggregate_sgl_visual_attention(steps, input_ids=ids, image_token_id=99, rules={"scorer": scorer})
    torch.testing.assert_close(result.visual_scores, torch.full((4,), 3.))
