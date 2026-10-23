"""Fixed video cues survive relocation of the values being compressed."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig

from tests.fixtures import native_image as image_fixture
from tests.integration.test_video_vqtoken_llava_onevision_positions_cpu import _video_model, _inputs, _pool


def _stage(algorithm, position=None, phase="forward"):
    return CompressionStageConfig.from_dict({
        "name": "video-signal-position", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.video",
        "binding": f"{algorithm}.video.llava_onevision", "target_modalities": ["video"],
        "operation": {} if position is None else {"target_site": position},
    })


def _pooling_head(tmp_path):
    # Construct the original probe/attention/MLP head from local random weights,
    # using the production loader. No external checkpoint or replacement rule.
    from transformers import SiglipVisionConfig, SiglipVisionModel
    from safetensors.torch import save_file
    from mllm_comp.backbones.fastvid_video import _load_frozen_siglip_pooling_head

    config = SiglipVisionConfig(hidden_size=16, intermediate_size=24, num_attention_heads=2,
                               num_hidden_layers=1, image_size=54, patch_size=2)
    model = SiglipVisionModel(config)
    config.to_json_file(tmp_path / "config.json")
    save_file({name: value for name, value in model.state_dict().items()
               if ".head." in name or ".post_layernorm." in name}, str(tmp_path / "model.safetensors"))
    return _load_frozen_siglip_pooling_head(tmp_path)


@pytest.mark.parametrize("algorithm", ["visionzip", "fastvid"])
def test_native_fixed_sources_actual_values_default_and_generation(algorithm, tmp_path):
    torch = pytest.importorskip("torch")
    torch.manual_seed(452)
    base = _video_model().train()
    head = _pooling_head(tmp_path) if algorithm == "fastvid" else None
    baseline = None
    for position in (None, "projector_output", "projector_input"):
        model = copy.deepcopy(base)
        projector = model.model.mm_projector
        activation = image_fixture.activation(model, _stage(algorithm, position))
        if head is not None:
            activation.signal_provider._pooling_head = copy.deepcopy(head)
        results, requests, raw_inputs, decoder_inputs = [], [], [], []
        original = activation._projector_callback

        def callback(request):
            requests.append(request)
            result = original(request)
            results.append(result)
            return result

        activation._projector_callback = callback
        activation.install()
        hooks = [projector.register_forward_pre_hook(lambda module, args: raw_inputs.append(args[0])),
                 model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs.get("inputs_embeds")), with_kwargs=True)]
        torch.manual_seed(453)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert len(requests) == 1
            request, result = requests[0], results[0]
            assert request.payload.token_count == 392
            before = position == "projector_input"
            raw = raw_inputs[0]
            values = _pool(torch, raw, "bilinear") if before else _pool(torch, projector(raw), "bilinear")
            torch.testing.assert_close(request.payload.features, values)
            if algorithm == "visionzip":
                cues = (request.payload.signals["vision_encoder_attention"][-2],
                        request.payload.signals["vision_encoder_hidden_states"][-2])
                targets = result.metadata["merge_target_indices_by_video"][0]
                expected = []
                for group, center in zip(result.source_groups[0], targets):
                    others = [i for i in group if i != center]
                    expected.append(values[center] + (values[others].mean(0) if others else 0))
                assert result.result_count == 294
            else:
                cues = (request.payload.signals["frame_global_features"], request.payload.signals["token_saliency_attention"])
                expected = []
                for item in result.metadata["value_records_by_video"][0]:
                    center, weight, assigned = item["center_index"], item["center_weight"], list(item["assigned_indices"])
                    expected.append(weight * values[center] + (1 - weight) * values[assigned].mean(0) if assigned else values[center])
            compressed = torch.stack(expected)
            torch.testing.assert_close(result.features, compressed)
            content = projector(compressed) if before else compressed
            entries = [(request.packing.patch_positions[source], content[i]) for i, source in enumerate(result.representative_indices[0])]
            entries += [(i, model.model.image_newline) for i in request.packing.structural_positions]
            entries.sort(key=lambda item: item[0])
            expected_splice = torch.stack([item[1] for item in entries])
            actual = decoder_inputs[0][0, 1:-2]
            torch.testing.assert_close(actual, expected_splice)
            parameters = (raw, *projector.parameters(), model.model.image_newline)
            upstream = torch.randn_like(actual)
            torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                                       torch.autograd.grad(expected_splice, parameters, upstream, retain_graph=True), atol=2e-6, rtol=2e-5)
            output.loss.backward()
            pixels_grad = args["images"][0].grad
            assert pixels_grad is not None and torch.isfinite(pixels_grad).all() and pixels_grad.abs().sum() > 0
            if baseline is None:
                baseline = (output.logits.detach().clone(), pixels_grad.clone(), tuple(value.detach().clone() for value in cues), result.source_groups)
            else:
                torch.testing.assert_close(cues, baseline[2], atol=0, rtol=0)
                if not before:
                    torch.testing.assert_close(output.logits, baseline[0], atol=0, rtol=0)
                    torch.testing.assert_close(pixels_grad, baseline[1], atol=0, rtol=0)
                if algorithm == "visionzip":
                    assert result.source_groups == baseline[3]
            if before:
                model.eval()
                generation = dict(inputs=args["input_ids"], images=[args["images"][0].detach()], image_sizes=args["image_sizes"],
                                  modalities=["video"], attention_mask=args["attention_mask"], max_new_tokens=2, min_new_tokens=2,
                                  do_sample=False, use_cache=True, return_dict_in_generate=True, eos_token_id=None, pad_token_id=0)
                first, second = model.generate(**generation), model.generate(**generation)
                torch.testing.assert_close(first.sequences, second.sequences, atol=0, rtol=0)
                assert activation.runtime_metadata()["projector_invocations"] == 3
                assert first.past_key_values.get_seq_length() == output.logits.shape[1] + 1
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()
