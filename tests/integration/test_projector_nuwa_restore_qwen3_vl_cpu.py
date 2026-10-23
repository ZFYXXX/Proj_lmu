"""Compound Qwen3 projector positions preserve fixed sources and branch weights."""

import copy

import pytest

from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _stage(algorithm, main_pre, *, phase="forward"):
    names = (("stage1_regional_visual_aggregation", "stage1_deepstack_aggregation") if algorithm == "nuwa" else
             ("vispruner_pristine_plus_distinctive_anchor_merge", "shared_deepstack_distinctive_anchor_merge"))
    config = ({"stage1_total_tokens": 3, "stage2_after_layer": 2, "stage2_retain_ratio": 0.5,
               "semantic_hidden_start": 0, "semantic_hidden_end": 3, "distance_squared_ratio": 1.0}
              if algorithm == "nuwa" else {"target_count": 2})
    targets = {names[0]: {"target_site": "projector_input" if main_pre else "projector_output"},
               names[1]: {"target_site": "projector_output" if main_pre else "projector_input"}}
    if algorithm == "nuwa":
        targets["stage2_last_prompt_attention_selection"] = {
            "target_site": "llm_decoder_layer_output_image_tokens", "layers": [2], "layer_edge": "before"}
    return CompressionStageConfig.from_dict({
        "name": f"{algorithm}-mixed-compound", "placement": "projector_level", "execution_phase": phase,
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.image",
        "binding": f"{algorithm}.image.qwen3_vl", "target_modalities": ["image"], "config": config,
        "operation": {"targets": targets},
    })


@pytest.mark.parametrize("backbone", ["qwen3_vl", "qwen25_vl"])
def test_restore_native_decode_keeps_calibration_and_physical_cache(backbone):
    torch = pytest.importorskip("torch")
    from tests.integration.test_image_compound_projector_positions_cpu import _stage as compound_stage

    torch.manual_seed(540)
    model = image_fixture.qwen_model(backbone).eval()
    if backbone == "qwen25_vl":
        model.model.visual.fullatt_block_indexes = [1, 2]
    stage = _stage("restore", False, phase="generate") if backbone == "qwen3_vl" else compound_stage(backbone, "restore", phase="generate")
    activation = image_fixture.activation(model, stage)
    activation.install()
    bridge = model._mllm_comp_qwen3_vl_provider_bridge if backbone == "qwen3_vl" else activation._bridge
    try:
        with torch.no_grad():
            output = model.generate(
                **image_fixture.inputs(torch, backbone), use_cache=True,
                min_new_tokens=3, max_new_tokens=3, do_sample=False,
                return_dict_in_generate=True,
            )
        assert output.sequences.shape[-1] == 11
        assert [layer.get_seq_length() for layer in output.past_key_values.layers] == [8] * 4
        calibration = bridge.runtime_metadata()["latest_restore_calibration"]
        assert calibration["phase"] == "decode"
        assert calibration["layers_calibrated"] == 4
        assert calibration["query_tokens"] == 1 and calibration["key_tokens"] == 8
        assert calibration["visual_source_tokens"] == 4
        assert calibration["first_layer_size"] == "uniform"
        assert calibration["position_policy"] == "native_sparse_mrope_representatives"
        assert calibration["cache_transform"] == "none"
        record = activation.runtime_metadata()["layout_delta_records"][0]
        assert (record["source_count"], record["result_count"]) == (4, 2)
        assert record["operation"] == "merge"
        assert sum(record["result_layout"]["token_sizes"]) == 4
        assert len(record["result_layout"]["representative_indices"]) == 2
    finally:
        activation.uninstall()


@pytest.mark.parametrize("algorithm", ["nuwa", "restore"])
@pytest.mark.parametrize("main_pre", [True, False])
def test_compound_mixed_values_keep_original_sources_weights_and_branch_gradients(algorithm, main_pre, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen3_vl import provider_bridge as provider

    torch.manual_seed(539)
    model = image_fixture.qwen_model("qwen3_vl").train()
    visual = model.model.visual
    mergers = [visual.merger, *visual.deepstack_merger_list]
    references = [copy.deepcopy(merger) for merger in mergers]
    activation = image_fixture.activation(model, _stage(algorithm, main_pre))
    activation.install()
    bridge = provider.install_qwen3_vl_transformers_4573_bridge(model)
    restore_dense, metrics = provider._restore_tome_steps, provider._visionzip_metrics_from_capture
    callback = activation.compress_after_visual_mergers
    raw_taps, sources, applications, outputs = [], {}, [], []
    branch_values, branch_outputs = [], []
    counts = [[] for _ in mergers]

    def capture_dense(*args, **kwargs):
        value = restore_dense(*args, **kwargs)
        if not raw_taps or value is not raw_taps[-1]:
            raw_taps.append(value)
        return value

    def capture_metrics(*args, **kwargs):
        attentions, keys = metrics(*args, **kwargs)
        sources.update(attention=attentions[0], keys=keys[0])
        return attentions, keys

    def check_application(main, branches, **kwargs):
        assert len(raw_taps) == 3
        raw = [raw_taps[-1], *raw_taps]
        for index, value in enumerate((main, *branches)):
            before = main_pre if index == 0 else not main_pre
            expected = raw[index].reshape(4, 64) if before else references[index](raw[index])
            torch.testing.assert_close(value, expected)
        torch.testing.assert_close(kwargs["attention"], sources["attention"])
        if algorithm == "nuwa":
            semantic = (sources["initial"] + raw_taps[0] + raw_taps[1]) / 3
            torch.testing.assert_close(kwargs["semantic_reference"], semantic.reshape(4, 64))
            torch.testing.assert_close(kwargs["keys"], sources["keys"])
            assert kwargs["feature_grid"] == (2, 2)
        else:
            torch.testing.assert_close(kwargs["vision_reference"], raw_taps[-1].reshape(4, 64))
        result = callback(main, branches, **kwargs)
        if algorithm == "nuwa":
            weights = result.kernel_result.aggregation_weights
        else:
            kernel = result.kernel_result
            weights = kernel.membership.to(kernel.token_sizes.dtype) / kernel.token_sizes[:, None]
            assert kernel.token_sizes.sum() == 4
            torch.testing.assert_close(kernel.membership.sum(0), torch.ones(4, dtype=torch.long))
        torch.testing.assert_close(result.main_image_features, weights.to(main) @ main)
        for value, compact in zip(branches, result.deepstack_image_features):
            torch.testing.assert_close(compact, weights.to(value) @ value)
            value.retain_grad()
            compact.retain_grad()
            branch_values.append(value)
            branch_outputs.append(compact)
        applications.append((result, weights))
        return result

    monkeypatch.setattr(provider, "_restore_tome_steps", capture_dense)
    monkeypatch.setattr(provider, "_visionzip_metrics_from_capture", capture_metrics)
    monkeypatch.setattr(activation, "compress_after_visual_mergers", check_application)
    handles = [merger.register_forward_pre_hook(lambda module, args, i=i: counts[i].append(args[0].shape[0]))
               for i, merger in enumerate(mergers)]
    handles.append(visual.blocks[0].norm1.register_forward_pre_hook(
        lambda module, args: sources.__setitem__("initial", args[0])))
    handles.append(visual.register_forward_hook(lambda module, args, output: outputs.append(output)))
    try:
        args = image_fixture.inputs(torch, "qwen3_vl")
        labels = torch.full_like(args["input_ids"], -100)
        labels[0, -1] = args["input_ids"][0, -1]
        result = model(**args, labels=labels, use_cache=False)
        assert len(applications) == len(outputs) == 1
        stage1, weights = applications[0]
        compact_patches = stage1.layout_delta.result_count * 4
        assert counts == ([[compact_patches], [16], [16], [16]] if main_pre else
                          [[16], [compact_patches], [compact_patches], [compact_patches]])
        for index, compact in enumerate((stage1.main_image_features, *stage1.deepstack_image_features)):
            before = main_pre if index == 0 else not main_pre
            expected = references[index](compact.reshape(-1, 16)) if before else compact
            actual = outputs[0][0] if index == 0 else outputs[0][1][index - 1]
            torch.testing.assert_close(actual, expected)
        result.loss.backward()
        for value, compact in zip(branch_values, branch_outputs):
            torch.testing.assert_close(value.grad, weights.to(value).T @ compact.grad)
        assert args["pixel_values"].grad.abs().sum() > 0
        assert all(merger.linear_fc2.weight.grad.abs().sum() > 0 for merger in mergers)
        records = activation.runtime_metadata()["layout_delta_records"]
        assert records[0]["site"] == activation.projector_target_sites[0]
        assert records[0]["projector_target_sites"] == list(activation.projector_target_sites)
        if algorithm == "nuwa":
            assert len(records) == 2 and records[0]["stage"] == "stage1"
            assert records[1]["source_layout"]["stage"] == "stage2"
            assert activation.runtime_metadata()["application_position"] == {"boundary": "inter_layer", "index": 2}
        else:
            calibration = bridge.runtime_metadata()["latest_restore_calibration"]
            assert calibration["layers_calibrated"] == 4 and calibration["visual_source_tokens"] == 4
            assert calibration["cache_transform"] == "none"
    finally:
        for handle in handles:
            handle.remove()
        activation.uninstall()
