"""TransPrune preserves source-stage statistics across application boundaries."""

import importlib

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


def _stage(backbone, boundaries=(1, 2, 3), *, sources=None, mid=False, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "transprune-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "transprune", "algorithm": "transprune", "variant": "transprune.image",
        "binding": f"transprune.image.{backbone}", "target_modalities": ["image"],
        "config": {"accumulating_layers": [1, 2, 3], "prune_after_layers": [1, 2, 3],
            "retain_ratios": [0.75, 0.5, 0.25]},
        "operation": {"targets": {"decoder_ttv_iga_boundaries": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 3), False), ((2, 3, 4), False), ((1, 2, 3), True)])
def test_transprune_native_boundaries_checkpoint_gradients_and_cache(backbone, boundaries, mid):
    initial = 729 if backbone == "llava_onevision" else 4
    shared._run(backbone, boundaries, None, mid, stage_factory=_stage,
        visual_counts=[int(initial * ratio) for ratio in (0.75, 0.5, 0.25)], layer_local=True,
        checkpoint_modes=(None, False, True) if mid else (None,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_transprune_two_device_sources(backbone):
    initial = 729 if backbone == "llava_onevision" else 4
    shared._run(backbone, (1, 2, 3), None, True, two_devices=True, stage_factory=_stage,
        visual_counts=[int(initial * ratio) for ratio in (0.75, 0.5, 0.25)], layer_local=True, checkpoint_modes=(None,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_transprune_final_empty_span_does_not_collect_unused_transitions(backbone):
    def stage_factory(*args, **kwargs):
        stage = _stage(*args, **kwargs)
        stage.config.update(accumulating_layers=[1, 2, 3, 4], retain_ratios=[0.75, 0.5, 0.0])
        return stage
    initial = 729 if backbone == "llava_onevision" else 4
    shared._run(backbone, (1, 2, 3), None, True, stage_factory=stage_factory,
        visual_counts=[int(initial * 0.75), int(initial * 0.5), 0], layer_local=True, checkpoint_modes=(True,))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 3), False), ((2, 3, 4), False), ((1, 2, 3), True)])
def test_transprune_native_ttv_scopes_fixed_history_and_iga(backbone, boundaries, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.compressors.image import decoder_selection as methods
    torch.manual_seed(572)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    transition, iga, kernel, project = methods.transprune_layer_transition_variation, methods.transprune_instruction_guided_attention, methods.transprune_image_tokens, provider._decoder_query_keys
    history, pending, source_indices, stages, shapes = [], {}, [], [], []
    completed = []
    handles = [layer.register_forward_hook(lambda module, args, output, i=i: completed.append(i)) for i, layer in enumerate(decoder.layers)]
    def module_ttv(before, after, images):
        before, after = before.float(), after.float()
        magnitude = after.norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-8)
        direction = 1 - torch.nn.functional.cosine_similarity(before, after, dim=-1, eps=1e-8)
        return (direction.softmax(-1) * magnitude)[0, images]
    def observe_transition(attention_before, attention_after, ffn_before, ffn_after, **kwargs):
        images = kwargs["image_indices"]
        original = kwargs["attention_image_indices"]
        keep = kwargs["selected_visual_indices"]
        attention = module_ttv(attention_before, attention_after, original)
        if keep is not None:
            attention = attention[keep]
        expected = attention + module_ttv(ffn_before, ffn_after, images)
        actual = transition(attention_before, attention_after, ffn_before, ffn_after, **kwargs)
        torch.testing.assert_close(actual, expected)
        history.append(expected)
        shapes.append((attention_before.shape[1], ffn_before.shape[1]))
        return actual
    def observe_project(layer, hidden, **kwargs):
        index = list(decoder.layers).index(layer)
        assert index == len(source_indices) + 1
        assert completed[-1] == index - 1  # Full source layer, next-layer projection.
        source_indices.append(index - 1)
        return project(layer, hidden, **kwargs)
    def observe_iga(**kwargs):
        queries, keys = kwargs["instruction_queries"].float(), kwargs["attention_keys"].float()
        logits = torch.einsum("qhd,thd->qht", queries, keys) / queries.shape[-1] ** 0.5
        visibility = kwargs["attention_key_mask"]
        if visibility is not None:
            logits = logits.masked_fill(~visibility.unsqueeze(1), float("-inf"))
        expected = logits.softmax(-1).mean((0, 1))[kwargs["image_indices"]]
        actual = iga(**kwargs)
        torch.testing.assert_close(actual, expected)
        source = source_indices[-1]
        assert len(history) == source + 1
        pending[source] = (tuple(history), expected)
        return actual
    def select(hidden, **kwargs):
        source = kwargs["boundary_after_layer"]
        assert source == len(stages)
        vectors, scores = pending.pop(source)
        assert len(kwargs["ttv_history"]) == source + 1
        torch.testing.assert_close(tuple(kwargs["ttv_history"]), vectors)
        torch.testing.assert_close(kwargs["image_attention_scores"], scores)
        result = kernel(hidden, **kwargs)
        accumulated = torch.stack(vectors).sum(0)
        combined = 0.5 * (accumulated / accumulated.sum().clamp_min(1e-8)) + 0.5 * (scores / scores.sum().clamp_min(1e-8))
        torch.testing.assert_close(result.combined_scores, combined)
        expected_keep = combined.topk(result.selected_visual_indices.numel()).indices.sort().values
        torch.testing.assert_close(result.selected_visual_indices, expected_keep)
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        for i, value in enumerate(history):
            history[i] = value[expected_keep]
        for key, (old_history, old_scores) in pending.items():
            pending[key] = (tuple(value[expected_keep] for value in old_history), old_scores[expected_keep])
        stages.append(source)
        return result
    monkeypatch.setattr(methods, "transprune_layer_transition_variation", observe_transition)
    monkeypatch.setattr(provider, "_decoder_query_keys", observe_project)
    monkeypatch.setattr(methods, "transprune_instruction_guided_attention", observe_iga)
    monkeypatch.setattr(methods, "transprune_image_tokens", select)
    activation = image_fixture.activation(model, _stage(backbone, boundaries, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**image_fixture.inputs(torch, backbone), use_cache=False)
        assert source_indices == stages == [0, 1, 2] and not pending
        assert any(attention != ffn for attention, ffn in shapes) is mid
        assert [record["application_position"] for record in activation.runtime_metadata()["layout_delta_records"]] == [
            {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": i} for i in boundaries]
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
