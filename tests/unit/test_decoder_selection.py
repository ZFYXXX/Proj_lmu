"""Shared method timing and request ownership, independent of native layouts."""

from types import SimpleNamespace

import pytest

from mllm_comp.compressors.image import decoder_selection as execution
from mllm_comp.compressors.layer_positions import LayerPosition
from mllm_comp.pipeline.stage import CompressionStageConfig


def _stage(algorithm, backbone, layers, *, mid=False):
    target = {"fastv": "decoder_prune_boundary", "pyramid_drop": "decoder_stage_boundary",
        "btp": "decoder_balanced_pruning_boundaries", "halfv": "decoder_ivr_ssr_boundaries", "transprune": "decoder_ttv_iga_boundaries"}[algorithm]
    return CompressionStageConfig.from_dict({
        "name": "selection", "algorithm": algorithm, "plugin": algorithm,
        "variant": f"{algorithm}.image", "binding": f"{algorithm}.image.{backbone}",
        "placement": "llm_level", "execution_phase": "forward", "target_modalities": ["image"],
        "operation": {"targets": {target: {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": layers, **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_fastv_keeps_fixed_scoring_input_when_application_moves(backbone, monkeypatch):
    method = execution.FastVDecoderMethod(stage=_stage("fastv", backbone, [4]),
        config={"prune_layer": 2, "prune_ratio": 0.5}, rules={}, backbone=backbone)
    method.resolve_depth(6)
    seen = []
    monkeypatch.setattr(execution, "fastv_scorer_query_indices",
        lambda mask, **kwargs: seen.append((mask, kwargs)) or "queries")
    call = method.new_call()
    assert call.capture_input(0, "mask", padding_mask="padding", labels="labels") is None
    capture = call.capture_input(1, "mask", padding_mask="padding", labels="labels")
    assert (capture.source_layer, capture.projection_layer, capture.query_indices) == (1, 1, "queries")
    assert method.positions == (LayerPosition("inter_layer", 4),)
    assert seen[0][1]["labels"] == "labels"
    assert call.capture_output(1, "mask", padding_mask=None, labels=None) is None


def test_fastv_calls_own_attention_independently_and_consume_it_once(monkeypatch):
    method = execution.FastVDecoderMethod(stage=_stage("fastv", "qwen25_vl", [2]),
        config={"prune_layer": 2}, rules={}, backbone="qwen25_vl")
    method.resolve_depth(4)
    first, second = method.new_call(), method.new_call()
    for call, query in ((first, "first"), (second, "second")):
        call.record_attention(None, query=query, image_keys=None, attention_keys="keys",
            visibility="mask", image_indices=None)
    monkeypatch.setattr(execution, "fastv_image_tokens", lambda hidden, **kwargs: (hidden, kwargs["scorer_query"]))
    assert first.select(method.positions[0], "a", image_indices=()) == ("a", "first")
    assert second.select(method.positions[0], "b", image_indices=()) == ("b", "second")
    with pytest.raises(execution.InvariantError, match="not captured"):
        first.select(method.positions[0], "a", image_indices=())


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_pyramid_observes_completed_sources_before_delayed_application(backbone, monkeypatch):
    method = execution.PyramidDecoderMethod(stage=_stage("pyramid_drop", backbone, [2, 3, 4]),
        config={"num_stages": 4, "layer_indices": [0, 1, 2]}, rules={}, backbone=backbone)
    method.resolve_depth(5)
    assert method.position_policy == ("contiguous" if backbone == "llava_onevision" else "sparse_source")
    mask = SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: True))
    monkeypatch.setattr(execution, "pyramid_instruction_indices", lambda *args, **kwargs: "query")
    call = method.new_call()
    assert call.capture_input(0, mask, padding_mask=None, labels=None) is None
    assert call.capture_output(3, mask, padding_mask=None, labels=None) is None
    source = call.capture_output(1, mask, padding_mask=None, labels=None)
    assert (source.source_layer, source.projection_layer) == (1, 2)
    assert method.positions[1] == LayerPosition("inter_layer", 3)
    assert method.new_call().scores == {}


def test_native_attention_is_read_only_when_a_method_requests_it():
    def unexpected(*args):
        pytest.fail("unscheduled native attention must not be evaluated")
    execution.capture_decoder_attention(None, None, None, unexpected)


def test_pure_selection_derives_layout_values_and_gradient_without_dropping_structure():
    torch = pytest.importorskip("torch")
    hidden = torch.arange(24, dtype=torch.float32).reshape(8, 3).requires_grad_()
    decision = execution.ImageTokenSelection([4, 1], metadata={"algorithm": "independent"})
    call = SimpleNamespace(select=lambda *args, **kwargs: decision)
    result = execution.execute_decoder_selection(call, LayerPosition("inter_layer", 2), hidden,
        image_indices=torch.tensor([1, 2, 3, 4, 5]), protected_indices=(3,),
        source_layout={"source": "image"}, result_layout={"position_policy": "sparse_source"})
    assert result.kept_indices.tolist() == [0, 1, 3, 4, 6, 7]
    assert result.kept_image_indices.tolist() == [1, 3, 4]
    assert result.dropped_indices.tolist() == [2, 5]
    assert tuple(result.layout_delta.old_to_new[index] for index in range(8)) == (0, 1, None, 2, 3, None, 4, 5)
    assert result.layout_delta.protected_indices == (0, 3, 6, 7)
    assert result.metadata["algorithm"] == "independent"
    torch.testing.assert_close(result.hidden_states, hidden[[0, 1, 3, 4, 6, 7]])
    result.hidden_states.sum().backward()
    torch.testing.assert_close(hidden.grad,
        torch.tensor([[1., 1., 1.], [1., 1., 1.], [0., 0., 0.], [1., 1., 1.],
                      [1., 1., 1.], [0., 0., 0.], [1., 1., 1.], [1., 1., 1.]]))


def test_pure_selection_cannot_select_text_as_an_image_token():
    torch = pytest.importorskip("torch")
    call = SimpleNamespace(select=lambda *args, **kwargs: execution.ImageTokenSelection([0, 2]))
    with pytest.raises(ValueError, match="only image positions"):
        execution.execute_decoder_selection(call, LayerPosition("inter_layer", 1),
            torch.zeros(4, 2), image_indices=torch.tensor([1, 2]))


def test_pyramid_remaps_pending_scores_without_scoring_them_again(monkeypatch):
    torch = pytest.importorskip("torch")
    method = execution.PyramidDecoderMethod(stage=_stage("pyramid_drop", "qwen3_vl", [2, 3]),
        config={"num_stages": 3, "layer_indices": [0, 1]}, rules={}, backbone="qwen3_vl")
    method.resolve_depth(4)
    call = method.new_call()
    call.scores = {0: torch.tensor([1., 2., 3.]), 1: torch.tensor([4., 5., 6.])}
    observed = []
    def select(hidden, **kwargs):
        observed.append(kwargs["image_attention_scores"].clone())
        return SimpleNamespace(kept_indices=torch.tensor([0, 2, 3, 4]))
    monkeypatch.setattr(execution, "pyramid_drop", select)
    call.select(method.positions[0], torch.zeros(5, 2), image_indices=torch.tensor([1, 2, 3]))
    torch.testing.assert_close(observed[0], torch.tensor([1., 2., 3.]))
    torch.testing.assert_close(call.scores[1], torch.tensor([5., 6.]))
    assert 0 not in call.scores


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_fixed_input_methods_only_apply_earlier_stages_before_capture(backbone, mid):
    btp = execution.BTPDecoderMethod(stage=_stage("btp", backbone, [1, 2, 3], mid=mid),
        config={"prune_before_layers": (2, 3, 4), "retain_ratios": (0.5, 0.5, 0.),
            "attention_fractions": (0.4, 0.4, 0.), "attention_oversample": (1.2, 1.2, 1.2),
            "rebalance_stages": (False, True, False)}, rules={}, backbone=backbone)
    halfv = execution.HalfVDecoderMethod(stage=_stage("halfv", backbone, [1, 2], mid=mid),
        config={"ivr_layer_index": 1, "ssr_layer_index": 2}, rules={}, backbone=backbone)
    for method in (btp, halfv):
        method.resolve_depth(5)
        call = method.new_call()
        assert call.select_before_input(0) is False
        assert call.begin_transition(0, None) is False
        # Only a preceding inter-layer stage runs before source 1. A mid-layer
        # application still follows source 1's attention in the native layer.
        assert call.select_before_input(1) is (not mid)
        assert call.select_before_input(4) is False
    fastv = execution.FastVDecoderMethod(stage=_stage("fastv", backbone, [1]),
        config={"prune_layer": 2}, rules={}, backbone=backbone)
    assert fastv.new_call().select_before_input(1) is False


def test_halfv_requests_native_norm_once_and_keeps_invocations_independent(monkeypatch):
    method = execution.HalfVDecoderMethod(stage=_stage("halfv", "qwen3_vl", [1, 3]),
        config={"ivr_layer_index": 1, "ssr_layer_index": 2}, rules={}, backbone="qwen3_vl")
    first, second = method.new_call(), method.new_call()
    mask = SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: True),
        sum=lambda: SimpleNamespace(item=lambda: 8))
    seen = []
    monkeypatch.setattr(execution, "fastv_scorer_query_indices",
        lambda *args, **kwargs: seen.append(kwargs) or "last-text")
    ivr = first.capture_input(0, mask, padding_mask="padding", labels="labels")
    ssr = first.capture_input(1, mask, padding_mask="padding", labels="labels")
    assert not ivr.use_rope and ssr.use_rope
    assert all(kwargs["labels"] is None for kwargs in seen)
    assert first.initial_visual_tokens == 8
    assert first.output_hidden_layer == second.output_hidden_layer == 0
    first.record_output_hidden(0, "normalized")
    assert first.output_hidden_layer is None and first.selection_hidden == "normalized"
    assert second.output_hidden_layer == 0 and second.selection_hidden is None
    with pytest.raises(execution.InvariantError, match="wrong layer"):
        first.record_output_hidden(1, "wrong")


def test_btp_remaps_delayed_input_scores_and_consumes_final_wipe_without_capture(monkeypatch):
    torch = pytest.importorskip("torch")
    method = execution.BTPDecoderMethod(stage=_stage("btp", "qwen25_vl", [2, 3, 4]),
        config={"prune_before_layers": (2, 3, 4), "retain_ratios": (0.5, 0.5, 0.),
            "attention_fractions": (0.4, 0.4, 0.), "attention_oversample": (1.2, 1.2, 1.2),
            "rebalance_stages": (False, True, False)}, rules={}, backbone="qwen25_vl")
    call = method.new_call()
    call.scores = {0: torch.tensor([1., 2., 3.]), 1: torch.tensor([4., 5., 6.])}
    observed = []
    def select(hidden, **kwargs):
        observed.append(kwargs["image_attention_scores"])
        return SimpleNamespace(kept_indices=torch.tensor([0, 2, 3, 4]))
    monkeypatch.setattr(execution, "balanced_token_pruning_image_tokens", select)
    call.select(method.positions[0], torch.zeros(5, 2), image_indices=torch.tensor([1, 2, 3]))
    torch.testing.assert_close(call.scores[1], torch.tensor([5., 6.]))
    assert method.score_layers == (0, 1)
    assert call.capture_input(2, None, padding_mask=None, labels=None) is None
    call.select(method.positions[-1], torch.zeros(5, 2), image_indices=torch.tensor([2, 3]))
    assert observed[-1] is None


@pytest.mark.parametrize("algorithm", ["btp", "halfv"])
def test_input_score_kernels_can_prune_newlines_inside_the_serialized_image_span(algorithm):
    torch = pytest.importorskip("torch")
    hidden = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    images = torch.tensor([1, 2, 3])
    rules = {"selector": lambda request: torch.tensor([0])}
    common = dict(image_indices=images, image_attention_scores=torch.tensor([1., 2., 3.]),
        boundary_after_layer=0, protected_indices=(3,))
    if algorithm == "btp":
        method = execution.BTPDecoderMethod(stage=_stage("btp", "llava_onevision", [1, 2, 3]),
            config={"prune_before_layers": (2, 3, 4), "retain_ratios": (0.5, 0.5, 0.),
                "attention_fractions": (0.4, 0.4, 0.), "attention_oversample": (1.2, 1.2, 1.2),
                "rebalance_stages": (False, True, False)}, rules=rules, backbone="llava_onevision")
    else:
        method = execution.HalfVDecoderMethod(stage=_stage("halfv", "llava_onevision", [1, 2]),
            config={"ivr_layer_index": 1, "ssr_layer_index": 2, "ivr_retention_ratio": 0.5},
            rules=rules, backbone="llava_onevision")
        common.update(initial_visual_tokens=3, selection_hidden_states=hidden)
    result = method.select(hidden, **common)
    assert result.kept_indices.tolist() == [0, 1, 4, 5]
    assert result.layout_delta.old_to_new[3] is None
    assert result.kept_image_indices.numel() == 1
    torch.testing.assert_close(result.hidden_states, hidden.index_select(0, result.kept_indices))


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
def test_transprune_requests_only_consumed_transition_and_iga_sources(backbone, monkeypatch):
    method = execution.TransPruneDecoderMethod(stage=_stage("transprune", backbone, [2, 3, 4]),
        config={"accumulating_layers": (1, 2, 3, 4), "prune_after_layers": (1, 2, 3),
            "retain_ratios": (0.75, 0.5, 0.25)}, rules={}, backbone=backbone)
    method.resolve_depth(5)
    call = method.new_call()
    images = SimpleNamespace(numel=lambda: 3)
    empty = SimpleNamespace(numel=lambda: 0)
    mask = SimpleNamespace(any=lambda: SimpleNamespace(item=lambda: True))
    observed = []
    monkeypatch.setattr(execution, "transprune_layer_transition_variation",
        lambda *args, **kwargs: observed.append((args, kwargs)) or "ttv")
    monkeypatch.setattr(execution, "dart_post_image_text_indices", lambda *args, **kwargs: images)
    assert method.score_layers == (0, 1, 2)
    assert method.position_policy == "contiguous"
    assert method.cache_policy is execution.CacheRepairPolicy.LAYER_LOCAL
    assert call.begin_transition(0, images)
    observation = SimpleNamespace(attention_before="a", attention_after="b", ffn_before="c", ffn_after="d")
    call.record_transition(observation, image_indices=images)
    assert observed[0][0] == ("a", "b", "c", "d")
    assert observed[0][1]["attention_image_indices"] is images
    capture = call.capture_output(0, mask, padding_mask=None, labels=None)
    assert (capture.source_layer, capture.projection_layer) == (0, 1)
    # Accumulation after the last IGA source and an empty image span are unused.
    assert call.begin_transition(3, images) is False
    assert call.begin_transition(1, empty) is False
    assert call.capture_output(3, mask, padding_mask=None, labels=None) is None
    assert method.new_call().history == []


def test_transprune_remaps_pending_fixed_history_and_mid_layer_survivors(monkeypatch):
    torch = pytest.importorskip("torch")
    method = execution.TransPruneDecoderMethod(stage=_stage("transprune", "qwen3_vl", [2, 3, 4], mid=True),
        config={"accumulating_layers": (1, 2, 3), "prune_after_layers": (1, 2, 3),
            "retain_ratios": (0.5, 0.25, 0.125)}, rules={}, backbone="qwen3_vl")
    method.resolve_depth(5)
    call = method.new_call()
    original = torch.tensor([1, 2, 3, 4])
    call.initial_visual_tokens = 4
    first, second = torch.tensor([1., 2., 3., 4.]), torch.tensor([5., 6., 7., 8.])
    call.history = [first, second]
    call.payloads = {0: ((first,), first * 2), 1: ((first, second), second * 2)}
    assert call.begin_transition(2, original)
    seen = []
    def select(hidden, **kwargs):
        seen.append(kwargs)
        return SimpleNamespace(selected_visual_indices=torch.tensor([0, 2]))
    monkeypatch.setattr(execution, "transprune_image_tokens", select)
    call.select(method.positions[0], "hidden", image_indices=original)
    assert seen[0]["initial_visual_tokens"] == 4
    torch.testing.assert_close(seen[0]["ttv_history"], (first,))
    torch.testing.assert_close(call.history, [torch.tensor([1., 3.]), torch.tensor([5., 7.])])
    pending_history, pending_scores = call.payloads[1]
    torch.testing.assert_close(pending_history, tuple(call.history))
    torch.testing.assert_close(pending_scores, torch.tensor([10., 14.]))
    observed = []
    monkeypatch.setattr(execution, "transprune_layer_transition_variation",
        lambda *args, **kwargs: observed.append(kwargs) or torch.tensor([9., 10.]))
    call.record_transition(SimpleNamespace(attention_before=None, attention_after=None, ffn_before=None, ffn_after=None),
        image_indices=torch.tensor([1, 2]))
    torch.testing.assert_close(observed[0]["attention_image_indices"], original)
    torch.testing.assert_close(observed[0]["selected_visual_indices"], torch.tensor([0, 2]))
    assert len(call.history) == 3
    assert len(call.payloads[1][0]) == 2  # Later observations cannot alter a fixed source snapshot.
    assert method.new_call().payloads == {}
