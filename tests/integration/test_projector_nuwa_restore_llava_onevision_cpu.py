"""Compound LLaVA reducers preserve legacy raw aggregation and expose real post values."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.core.vocabulary import ExecutionPhase
from tests.fixtures import native_image as image_fixture
from tests.integration.test_projector_aim_vispruner_llava_onevision_cpu import _inputs
from tests.integration.test_projector_llava_onevision_positions_cpu import _model


def _stage(algorithm, position):
    target = "stage1_regional_visual_aggregation" if algorithm == "nuwa" else "vispruner_pristine_plus_distinctive_anchor_merge"
    config = {"stage1_total_tokens": 12, "stage2_after_layer": 1, "stage2_retain_ratio": 1.0,
        "semantic_hidden_start": 0, "semantic_hidden_end": 3} if algorithm == "nuwa" else {"target_count": 6}
    return CompressionStageConfig.from_dict({
        "name": algorithm + "-values", "placement": "projector_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": algorithm + ".image",
        "binding": algorithm + ".image.llava_onevision", "target_modalities": ["image"],
        "operation": {} if position is None else {"targets": {target: {"target_site": position}}},
        "config": config,
    })


@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_native_anyres_decode_preserves_newlines_and_physical_cache(algorithm):
    torch = pytest.importorskip("torch")
    torch.manual_seed(529)
    model = _model().eval()
    stage = _stage(algorithm, None)
    stage.execution_phase = ExecutionPhase.GENERATE
    if algorithm == "nuwa":
        stage.config["stage2_retain_ratio"] = 0.5
    activation = image_fixture.activation(model, stage)
    activation.install()
    args = _inputs(torch)
    args.pop("labels")
    args.pop("use_cache")
    args["inputs"] = args.pop("input_ids")
    try:
        with torch.no_grad():
            result = model.generate(**args, min_new_tokens=2, max_new_tokens=2,
                do_sample=False, use_cache=True, return_dict_in_generate=True)
        # Two 729-patch crops plus 27 newline tokens become 12 content tokens
        # plus those same newlines. Nuwa's second stage keeps six content tokens.
        final = 36 if algorithm == "nuwa" else 42
        assert [layer.get_seq_length() for layer in result.past_key_values.layers] == [final + 1] * 3
        records = activation.runtime_metadata()["layout_delta_records"]
        first = records[0]
        assert (first["source_count"], first["result_count"]) == (1485, 39)
        assert first["protected_indices"] == [729 + row * 28 + 27 for row in range(27)]
        bridge = activation._bridge
        metadata = bridge.runtime_metadata()
        lifecycle = metadata["latest_decoder_lifecycle"]
        assert lifecycle["decode_forward_calls"] == 1
        assert lifecycle["selection_repeated_during_decode"] is False
        if algorithm == "nuwa":
            assert len(records) == 2
            assert (records[1]["source_count"], records[1]["result_count"]) == (42, 36)
            assert first["result_layout"]["position_policy"] == "rpme"
            assert lifecycle["selection_runs"] == 1
            boundary = lifecycle["boundaries"][0]
            assert boundary["cache"]["policy"] == "compact_prefix"
            assert boundary["cache"]["touched_layers"] == [0]
            assert set(index + 1 for index in first["result_layout"]["newline_indices"]).issubset(boundary["kept_indices"])
        else:
            assert len(records) == 1
            assert first["operation"] == "merge"
            assert sum(first["result_layout"]["token_sizes"]) == 1485
            assert set(first["protected_indices"]).issubset(first["result_layout"]["representative_indices"])
            assert metadata["latest_restore_calibration"] == {
                "phase": "decode", "layers_calibrated": 3, "query_tokens": 1,
                "key_tokens": 43, "visual_source_tokens": 1485,
                "first_layer_size": "uniform", "position_policy": "real_packed_source_representative",
                "cache_transform": "none",
            }
        assert bridge._vision_source_capture.get() is None
    finally:
        activation.uninstall()


@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
def test_anyres_legacy_raw_and_explicit_projected_aggregation_sources_and_gradients(algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    torch.manual_seed(420)
    base = _model().train()
    kernel_name = "nuwa_stage1_image_tokens" if algorithm == "nuwa" else "restore_image_tokens"
    kernel = getattr(adapter, kernel_name)
    monkeypatch.setattr(provider, "_packed_qk_metrics", lambda *args: pytest.fail("The original SigLIP source must be retained."))
    legacy = None
    for position in (None, "projector_input", "projector_output"):
        model = copy.deepcopy(base)
        projector = model.model.mm_projector
        inner = model.model.vision_tower.vision_tower
        activation = image_fixture.activation(model, _stage(algorithm, position))
        activation.install()
        activation._bridge.prepare_training_forward()
        source_outputs, projection_inputs, decoder_inputs, results = [], [], [], []
        handles = [inner.register_forward_hook(lambda module, args, output: source_outputs.append(output)),
            projector.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0])),
            model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True)]

        def capture(values, selection, attention, *args, **kwargs):
            crop = len(results)
            source = source_outputs[0]
            layer = -1 if algorithm == "nuwa" else -2
            raw = source.hidden_states[layer][crop]
            assert values.shape == (729, 24 if position == "projector_output" else 16)
            torch.testing.assert_close(projection_inputs[0][crop], raw)
            torch.testing.assert_close(attention, source.attentions[layer][crop])
            if algorithm == "nuwa":
                torch.testing.assert_close(selection, torch.stack(source.hidden_states[:3]).float().mean(0)[crop])
                final_layer = inner.vision_model.encoder.layers[-1]
                torch.testing.assert_close(args[0], final_layer.self_attn.k_proj(final_layer.layer_norm1(source.hidden_states[-2]))[crop])
            else:
                torch.testing.assert_close(selection, raw)
            result = kernel(values, selection, attention, *args, **kwargs)
            if algorithm == "nuwa":
                weights = result.aggregation_weights.to(values.dtype)
            else:
                weights = result.membership.to(values.dtype) / result.token_sizes[:, None]
            torch.testing.assert_close(result.tokens, weights @ values)
            results.append(result)
            return result

        monkeypatch.setattr(adapter, kernel_name, capture)
        torch.manual_seed(421)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert len(source_outputs) == 1 and len(results) == 2
            projected_call_shapes = [tuple(value.shape) for value in projection_inputs]
            assert projected_call_shapes[0] == (2, 729, 16)
            if position == "projector_output":
                assert len(projected_call_shapes) == 1
            elif position == "projector_input":
                assert len(projected_call_shapes) == 2 and projected_call_shapes[1][1] == 16
            else:
                assert len(projected_call_shapes) == 3
            record = activation.runtime_metadata()["layout_delta_records"][0]
            expected_site = {None: adapter.VISIONZIP_SITE, "projector_input": "pre_vision_projector", "projector_output": "post_vision_projector"}[position]
            assert record["site"] == record["result_layout"]["compression_site"] == expected_site
            assert record["callback_site"] == adapter.VISIONZIP_SITE
            raw = projection_inputs[0]
            entries, group_metadata = [], []
            for crop, result in enumerate(results):
                if algorithm == "nuwa":
                    weights = result.aggregation_weights.to(raw.dtype)
                    representatives = result.selected_indices.tolist()
                    group_metadata.append((representatives, weights.detach().clone()))
                else:
                    weights = result.membership.to(raw.dtype) / result.token_sizes[:, None]
                    representatives = result.representative_indices.tolist()
                    group_metadata.append((representatives, result.membership.clone(), result.token_sizes.clone()))
                # Independent linear combination followed by the actual nonlinear
                # projector, or the same combination of dense projected values.
                values = weights @ projector(raw[crop]) if position == "projector_output" else projector(weights @ raw[crop])
                for index, representative in enumerate(representatives):
                    packed_position = representative if crop == 0 else 729 + representative + representative // 27
                    entries.append((packed_position, values[index]))
            for row in range(27):
                entries.append((729 + row * 28 + 27, model.model.image_newline))
            entries.sort(key=lambda item: item[0])
            expected = torch.stack([value for _, value in entries])
            actual = decoder_inputs[0][0, 1:-2]
            torch.testing.assert_close(actual, expected)
            newline_targets = [i for i, (source, _) in enumerate(entries) if source in record["protected_indices"]]
            assert len(newline_targets) == 27
            torch.testing.assert_close(actual[newline_targets], model.model.image_newline.expand(27, -1), atol=0, rtol=0)
            parameters = (raw, *tuple(projector.parameters()), model.model.image_newline)
            upstream = torch.randn_like(actual)
            torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                torch.autograd.grad(expected, parameters, upstream, retain_graph=True), atol=2e-6, rtol=2e-5)
            output.loss.backward()
            gradient = args["images"][0].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            if legacy is None:
                legacy = group_metadata, actual.detach().clone(), output.logits.detach().clone(), gradient.clone()
            else:
                torch.testing.assert_close(group_metadata, legacy[0], atol=0, rtol=0)
                if position == "projector_input":
                    torch.testing.assert_close(actual, legacy[1])
                    torch.testing.assert_close(output.logits, legacy[2], atol=2e-6, rtol=2e-5)
                    torch.testing.assert_close(gradient, legacy[3], atol=2e-6, rtol=2e-5)
                else:
                    assert not torch.allclose(actual, legacy[1], atol=1e-5, rtol=1e-5)
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
