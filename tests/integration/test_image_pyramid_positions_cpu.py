"""PyramidDrop reuses the shared stage boundaries and retains layer-local caches."""

import importlib

from mllm_comp.compressors.image import decoder_selection

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture
from tests.integration import test_image_btp_positions_cpu as shared


def _stage(backbone, boundaries=(1, 2, 3), *, sources=(0, 1, 2), mid=False, phase="forward"):
    config = {"num_stages": 4, "keep_ratio": 0.5}
    if backbone != "llava_onevision":
        config["layer_indices"] = list(sources)
    return CompressionStageConfig.from_dict({
        "name": "pyramid-positions", "placement": "llm_level", "execution_phase": phase,
        "plugin": "pyramid_drop", "algorithm": "pyramid_drop", "variant": "pyramid_drop.image",
        "binding": f"pyramid_drop.image.{backbone}", "target_modalities": ["image"], "config": config,
        "operation": {"targets": {"decoder_stage_boundary": {
            "target_site": "llm_decoder_layer_post_attention_image_tokens" if mid else "llm_decoder_layer_output_image_tokens",
            "layers": list(boundaries), **({} if mid else {"layer_edge": "before"})}}},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("boundaries,mid", [((1, 2, 3), False), ((2, 3, 4), False), ((1, 2, 3), True)])
def test_pyramid_native_stages_checkpoint_gradients_and_layer_local_cache(backbone, boundaries, mid):
    count = 729 if backbone == "llava_onevision" else 4
    counts = []
    for _ in boundaries:
        count = max(1, int(count * 0.5))
        counts.append(count)
    shared._run(backbone, boundaries, (0, 1, 2), mid, stage_factory=_stage, visual_counts=counts, layer_local=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pyramid_default_aliases(backbone):
    shared._default_aliases(backbone, _stage, "decoder_stage_boundary", (0, 1, 2))


def test_native_llava_pyramid_preserves_anyres_newlines_at_each_boundary(monkeypatch):
    torch = pytest.importorskip("torch")
    from tests.integration.test_projector_llava_onevision_positions_cpu import _model
    from tests.integration.test_projector_aim_vispruner_llava_onevision_cpu import _inputs

    model = _model().eval()
    stage = _stage("llava_onevision", (1, 2))
    stage.config["num_stages"] = 3
    activation = image_fixture.activation(model, stage)
    original = activation.apply_decoder_selection
    observed = []

    def check(selection, point, hidden_states, *, sequence, **kwargs):
        newlines = tuple(sequence.newline_indices)
        response = original(selection, point, hidden_states, sequence=sequence, **kwargs)
        assert len(newlines) == len(response.newline_indices) == 27
        torch.testing.assert_close(response.hidden_states[list(response.newline_indices)],
            hidden_states[list(newlines)], atol=0, rtol=0)
        observed.append((hidden_states.shape[0], response.hidden_states.shape[0]))
        return response

    monkeypatch.setattr(activation, "apply_decoder_selection", check)
    activation.install()
    try:
        with torch.no_grad():
            output = model(**_inputs(torch))
        assert observed == [(1488, 745), (745, 374)]
        assert output.logits.shape[1] == 374
    finally:
        activation.uninstall()


@pytest.mark.parametrize("mid", [False, True])
def test_pyramid_deepstack_record_uses_actual_application(mid):
    torch = pytest.importorskip("torch")
    model = image_fixture.model("qwen3_vl", num_decoder_layers=4).eval()
    stage = _stage("qwen3_vl", (3,), mid=mid)
    stage.config.update(num_stages=2, layer_indices=[0])
    activation = image_fixture.activation(model, stage)
    activation.install()
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        model(**args, labels=labels, use_cache=False)
        metadata = activation.runtime_metadata()
        assert [record["application_position"] for record in metadata["layout_delta_records"]] == [
            {"boundary": "post_attention_pre_ffn" if mid else "inter_layer", "index": 3}]
        assert metadata["first_drop_after_deepstack"] is True
        assert activation.first_drop_layer == 0  # The old property retains its rule-stage identity.
    finally:
        activation.uninstall()


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
def test_pyramid_two_device_sources(backbone):
    count = 729 if backbone == "llava_onevision" else 4
    counts = []
    for _ in range(3):
        count = max(1, int(count * 0.5))
        counts.append(count)
    shared._run(backbone, (1, 2, 3), (0, 1, 2), True, two_devices=True,
        stage_factory=_stage, visual_counts=counts, layer_local=True)


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl", "llava_onevision"])
@pytest.mark.parametrize("mid", [False, True])
def test_pyramid_completed_sources_and_score_mapping(backbone, mid, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(340)
    model = image_fixture.model(backbone, num_decoder_layers=4).eval()
    decoder = image_fixture.decoder(model, backbone)
    provider = importlib.import_module(f"mllm_comp.backbones.{backbone}.provider_bridge")
    kernel, project = decoder_selection.pyramid_drop, provider._decoder_query_keys
    completed, pending, stages = [], {}, []
    handles = [layer.register_forward_hook(lambda module, args, output, i=i: completed.append(i)) for i, layer in enumerate(decoder.layers)]

    def capture(layer, hidden, **kwargs):
        index = list(decoder.layers).index(layer)
        assert completed[-1] == index - 1  # Complete source k, projected using layer k+1.
        query, keys, attention_keys = project(layer, hidden, **kwargs)
        # The decoder call below has one supervised answer boundary and a causal mask.
        queries = kwargs["instruction_indices"].to(attention_keys.device)
        logits = torch.einsum("qhd,khd->hqk", query.float(), attention_keys.float()) / (query.shape[-1] ** 0.5)
        visibility = torch.arange(attention_keys.shape[0], device=attention_keys.device)[None, :] <= queries[:, None]
        scores = logits.masked_fill(~visibility[None], float("-inf")).softmax(-1).mean((0, 1))
        pending[index - 1] = scores.index_select(0, kwargs["image_indices"].to(scores.device))
        return query, keys, attention_keys

    def apply(hidden, **kwargs):
        stage = len(stages)
        torch.testing.assert_close(kwargs["image_attention_scores"], pending.pop(stage))
        result = kernel(hidden, **kwargs)
        local = torch.isin(kwargs["image_indices"], result.kept_indices).nonzero(as_tuple=False).flatten()
        for source, scores in pending.items():
            pending[source] = scores.index_select(0, local.to(scores.device))
        torch.testing.assert_close(result.hidden_states, hidden.index_select(0, result.kept_indices))
        stages.append(stage)
        return result

    monkeypatch.setattr(provider, "_decoder_query_keys", capture)
    monkeypatch.setattr(decoder_selection, "pyramid_drop", apply)
    activation = image_fixture.activation(model, _stage(backbone, (1, 2, 3) if mid else (2, 3, 4), mid=mid))
    activation.install()
    if backbone == "llava_onevision":
        activation._bridge.prepare_training_forward()
    try:
        args = image_fixture.inputs(torch, backbone)
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        model(**args, labels=labels, use_cache=False)
        assert stages == [0, 1, 2] and not pending
    finally:
        activation.uninstall()
        for handle in handles:
            handle.remove()
