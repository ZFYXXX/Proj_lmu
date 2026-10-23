"""FinePruner IAC application features and fixed ARP sources share native boundaries."""

import importlib

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("algorithm", ["finepruner", "transprune"])
def test_native_layer_local_decode_uses_actual_cache_and_rotary_positions(backbone, algorithm):
    """Keep the old provider monolith's position check on real native layers."""
    torch = pytest.importorskip("torch")
    from tests.integration.test_image_transprune_positions_cpu import _stage as transprune_stage

    torch.manual_seed(481)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    language = image_fixture.decoder(model, backbone)
    stage = (_stage if algorithm == "finepruner" else transprune_stage)(backbone, phase="generate")
    activation = image_fixture.activation(model, stage)
    activation.install()
    if backbone == "qwen3_vl":
        from mllm_comp.backbones.qwen3_vl.provider_bridge import install_qwen3_vl_transformers_4573_bridge
        bridge = install_qwen3_vl_transformers_4573_bridge(model)
    else:
        bridge = activation._bridge
    rotary_positions = []
    handle = language.rotary_emb.register_forward_pre_hook(
        lambda module, args: rotary_positions.append(args[1].detach().clone()))
    inputs = image_fixture.inputs(torch, backbone)
    if backbone == "llava_onevision":
        inputs["inputs"] = inputs.pop("input_ids")
    try:
        with torch.no_grad():
            generated = model.generate(**inputs, min_new_tokens=3, max_new_tokens=3,
                do_sample=False, use_cache=True, eos_token_id=None, pad_token_id=0,
                return_dict_in_generate=True)
        report = bridge.runtime_metadata()["latest_decoder_lifecycle"]
        all_positions = report["layer_local_decode_positions"]
        assert len(all_positions) == 2 * len(language.layers)
        positions = all_positions[-len(language.layers):]
        # The last decode wrote one new entry into each layer's physical cache.
        physical = [layer.get_seq_length() - 1 for layer in generated.past_key_values.layers]
        assert [item["cache_positions"][0] for item in all_positions] == [p - 1 for p in physical] + physical
        assert [item["text_position_values"][0] for item in positions] == physical
        # The Qwen fixture's 2x2 image grid occupies four tokens but two RoPE
        # positions. Before pruning its native decode offset is therefore -2.
        offset = 0 if backbone == "llava_onevision" else -2
        expected = [physical[0] + offset, *physical[1:]]
        assert [item["rotary_position_values"][0] for item in positions] == expected
        assert all(item["rotary_matches_expected"] for item in positions)
        actual = rotary_positions[-len(language.layers):]
        assert len(actual) == len(expected)
        for coordinates, value in zip(actual, expected):
            assert coordinates.reshape(-1).tolist() == [value] * (1 if backbone == "llava_onevision" else 3)
        assert report["selection_repeated_during_decode"] is False
    finally:
        handle.remove()
        activation.uninstall()


def _stage(backbone, boundaries=(1, 2, 4), *, sources=None, mid=False, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "finepruner-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "finepruner", "algorithm": "finepruner", "variant": "finepruner.image",
        "binding": f"finepruner.image.{backbone}", "target_modalities": ["image"],
        "config": {"iac_after_layers": [1, 2], "iac_hash_bits": [2, 1],
            "arp_center_after_layer": 3, "arp_layer_offsets": [0, 1], "target_visual_tokens": 1},
        "operation": {"targets": {"decoder_iac_arp_boundaries": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 4), False), ((0, 1, 3), False),
    ((2, 3, 4), False), ((0, 1, 3), True), ((1, 2, 3), True)])
def test_finepruner_native_boundaries_checkpoint_gradients_and_cache(backbone, boundaries, mid):
    shared._run(backbone, boundaries, None, mid, stage_factory=_stage, visual_counts="recorded", layer_local=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_finepruner_default_aliases(backbone):
    shared._default_aliases(backbone, _stage, "decoder_iac_arp_boundaries", (0, 1, 3))


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_finepruner_two_device_sources(backbone):
    shared._run(backbone, (1, 2, 3), None, True, two_devices=True,
        stage_factory=_stage, visual_counts="recorded", layer_local=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 4), False), ((2, 3, 4), False), ((1, 2, 3), True)])
def test_finepruner_native_sources_hash_representatives_and_arp_ranking(backbone, boundaries, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(472)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    adapter = importlib.import_module("mllm_comp.compressors.image.decoder_selection")
    project, iac, arp = provider._decoder_query_keys, adapter.finepruner_iac_image_tokens, adapter.finepruner_arp_image_tokens
    pending, sampled, actual, stages = [], {}, {}, []
    inside_score = False
    def observe(index):
        def hook(module, args):
            if not inside_score:
                actual[index] = args[0].detach().clone()
        return hook
    handles = [decoder.layers[index].self_attn.q_proj.register_forward_pre_hook(observe(index)) for index in (2, 3)]
    def capture(layer, hidden, **kwargs):
        nonlocal inside_score
        index = list(decoder.layers).index(layer)
        assert index in (2, 3) and index not in sampled
        sampled[index] = layer.input_layernorm(hidden).detach().clone()
        inside_score = True
        try:
            query, image_keys, keys = project(layer, hidden, **kwargs)
        finally:
            inside_score = False
        assert kwargs["instruction_indices"].tolist() == [hidden.shape[1] - 1]
        logits = torch.einsum("qhd,thd->qht", query, keys) / query.shape[-1] ** 0.5
        pending.append(logits.float().softmax(-1)[0, :, kwargs["image_indices"]])
        return query, image_keys, keys
    def select_iac(hidden, **kwargs):
        stage_index = len(stages)
        assert kwargs["completed_layers"] == stage_index + 1
        images, spec = kwargs["image_indices"], kwargs["spec"]
        bits = spec.iac_hash_bits[stage_index]
        generator = torch.Generator().manual_seed(spec.random_seed + stage_index)
        planes = torch.randn(bits, hidden.shape[-1], generator=generator)
        signs = (hidden[images].float() @ planes.T >= 0).tolist()
        representatives = {}
        for i, row in enumerate(signs):
            representatives.setdefault(tuple(row), i)
        expected = sorted(representatives.values())
        protected = set(kwargs.get("protected_indices", ()))
        expected = sorted(set(expected) | {i for i, value in enumerate(images.tolist()) if value in protected})
        result = iac(hidden, **kwargs)
        assert result.kept_image_indices.tolist() == images[expected].tolist()
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        for i, value in enumerate(pending):
            pending[i] = value[:, expected]
        stages.append("iac")
        return result
    def select_arp(hidden, attentions, **kwargs):
        assert len(attentions) == len(pending) == 2
        torch.testing.assert_close(list(attentions), pending)
        average = torch.stack(pending).mean(0)
        distribution = average.softmax(-1)
        kl = (distribution * (distribution.clamp_min(1e-12).log() + torch.log(torch.tensor(float(average.shape[-1]))))).sum(-1)
        heads = torch.argsort(kl, stable=True)[:max(1, round(average.shape[0] * kwargs["spec"].head_ratio))]
        scores = average[heads].mean(0)
        result = arp(hidden, attentions, **kwargs)
        torch.testing.assert_close(result.token_scores, scores)
        assert result.selected_heads.tolist() == heads.tolist()
        assert result.kept_image_indices.tolist() == [kwargs["image_indices"][torch.argsort(scores, descending=True, stable=True)[0]].item()]
        torch.testing.assert_close(result.hidden_states, hidden[result.kept_indices], atol=0, rtol=0)
        stages.append("arp")
        return result
    monkeypatch.setattr(provider, "_decoder_query_keys", capture)
    monkeypatch.setattr(adapter, "finepruner_iac_image_tokens", select_iac)
    monkeypatch.setattr(adapter, "finepruner_arp_image_tokens", select_arp)
    activation = image_fixture.activation(model, _stage(backbone, boundaries, mid=mid))
    activation.install()
    try:
        with torch.no_grad():
            model(**image_fixture.inputs(torch, backbone), use_cache=False)
        assert stages == ["iac", "iac", "arp"]
        torch.testing.assert_close(sampled, actual, atol=2e-6, rtol=2e-5)
        assert [record["application_position"] for record in activation.runtime_metadata()["layout_delta_records"]] == [
            {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": i} for i in boundaries]
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
