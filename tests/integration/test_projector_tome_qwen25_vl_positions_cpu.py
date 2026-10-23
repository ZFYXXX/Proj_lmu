"""Independent ToMe permanent output uses either side of the native Q25 merger."""

import copy

import pytest

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures import native_image as image_fixture


def _model():
    return image_fixture.qwen_model("qwen25_vl")


def _inputs(torch):
    ids = torch.tensor([[1, 7, *([10] * 16), 8, 2]])
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids),
        pixel_values=torch.randn(64, 12, requires_grad=True), image_grid_thw=torch.tensor([[1, 8, 8]]))


def _stage(position="projector_input", *, encoder="mid", phase="forward"):
    targets = {"encoder_merge": {"target_site": "visual_block_post_attention_residual_pre_mlp"
        if encoder == "mid" else "visual_block_output", "layers": [1]}}
    if position is not None:
        targets["permanent_output_merge"] = {"target_site": position}
    return CompressionStageConfig.from_dict({
        "name": "tome-output-position", "placement": "encoder_level", "execution_phase": phase,
        "plugin": "tome", "algorithm": "tome", "variant": "tome.image.encoder_dense_boundary",
        "binding": "tome.image.qwen25_vl", "target_modalities": ["image"],
        "operation": {"targets": targets}, "criterion": {"score_layer": 0},
        "config": {"r": 4, "output_r": 3, "token_sizes": list(range(1, 65)), "protected_indices": [63]},
    })


@pytest.mark.parametrize("encoder", ["mid", "output"])
def test_encoder_merge_and_output_position_preserve_source_math_dense_restore_and_gradients(encoder, monkeypatch):
    torch = pytest.importorskip("torch")
    torch.manual_seed(451)
    base = _model().train()
    raw = _inputs(torch)
    default = None
    for position in (None, "projector_output", "projector_input"):
        model = copy.deepcopy(base)
        visual = model.model.visual
        native_merger = copy.deepcopy(visual.merger)
        activation = image_fixture.activation(model, _stage(position, encoder=encoder))
        before = position == "projector_input"
        args = {name: value.detach().clone() for name, value in raw.items()}
        args["pixel_values"].requires_grad_(True)
        window, _ = visual.get_window_index(args["image_grid_thw"])
        reverse = window.argsort()
        assert not torch.equal(window, torch.arange(16))
        observed, mlp_counts, merger_inputs, merger_outputs, encoder_results = {}, [[], [], []], [], [], []
        hooks = [block.mlp.register_forward_pre_hook(lambda module, args, i=i: mlp_counts[i].append(args[0].shape[0]))
            for i, block in enumerate(visual.blocks)]
        hooks += [visual.merger.register_forward_pre_hook(lambda module, args: merger_inputs.append(args[0])),
            visual.merger.register_forward_hook(lambda module, args, output: merger_outputs.append(output))]
        original_encoder = activation.encoder_dense_boundary

        def encoder_merge(*values, **kwargs):
            result = original_encoder(*values, **kwargs)
            encoder_results.append(result.kernel_result)
            return result

        activation.encoder_dense_boundary = encoder_merge
        original_output = activation.compact_visual_output

        def compact(values, keys, **kwargs):
            dense = observed["final_visual"]
            grouped_raw = dense.reshape(16, 64)[reverse]
            expected_values = grouped_raw if before else native_merger(dense)[reverse]
            torch.testing.assert_close(values, expected_values)
            block, normalized, embeddings = observed["final_key_source"]
            with torch.no_grad():
                qkv = block.attn.qkv(normalized).reshape(64, 3, block.attn.num_heads, -1)
                raw_keys = qkv[:, 1]
                cos, sin = (value.unsqueeze(-2).to(raw_keys.dtype) for value in embeddings)
                left, right = raw_keys.chunk(2, dim=-1)
                rotary_keys = raw_keys * cos + torch.cat((-right, left), -1) * sin
                expected_keys = rotary_keys.float().reshape(16, 4, 2, 8).mean(1)[reverse]
            torch.testing.assert_close(keys, expected_keys)
            # Permanent output starts with unit masses, independently of the
            # encoder's nonuniform 1..64 masses and its r=4 merge.
            metric = keys.mean(1)
            metric = metric / metric.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            a, b = torch.arange(0, 16, 2), torch.arange(1, 16, 2)
            similarities = metric[a] @ metric[b].T
            scores, destinations = similarities.max(-1)
            chosen = scores.argsort(descending=True, stable=True)[:3]
            edges = {int(a[i]): int(b[destinations[i]]) for i in chosen}
            groups = [(int(index),) for index in a if int(index) not in edges]
            groups += [tuple(sorted([int(index), *[source for source, target in edges.items() if target == int(index)]])) for index in b]
            groups = tuple(sorted(groups, key=lambda group: group[0]))
            expected = torch.stack([values[list(group)].mean(0) for group in groups])
            result = original_output(values, keys, **kwargs)
            assert result.sequence_source_groups == groups
            torch.testing.assert_close(result.hidden_states, expected)
            torch.testing.assert_close(result.token_sizes.flatten(), torch.tensor([len(group) for group in groups]).to(values))
            assert result.token_sizes.sum().item() == 16
            observed["compact"] = expected
            return result

        activation.compact_visual_output = compact
        try:
            activation.install()
            original_block = activation._bridge._execute_visual_block

            def block(*values, **kwargs):
                result = original_block(*values, **kwargs)
                if kwargs["layer_index"] == 2:
                    observed["final_visual"] = result[0]
                    observed["final_key_source"] = values[0], values[2], values[4]
                return result

            monkeypatch.setattr(activation._bridge, "_execute_visual_block", block)
            labels = args["input_ids"].masked_fill(args["input_ids"] == 10, -100)
            output = model(**args, labels=labels, use_cache=False)
            assert output.logits.shape == (1, 17, 64)
            assert [counts[0] for counts in mlp_counts] == [64, 60 if encoder == "mid" else 64, 64]
            assert len(encoder_results) == 1
            assert encoder_results[0].token_sizes.sum().item() == sum(range(1, 65))
            assert [value.shape[0] for value in merger_inputs] == [52 if before else 64]
            if before:
                torch.testing.assert_close(merger_inputs[0], observed["compact"].reshape(52, 16))
                torch.testing.assert_close(merger_outputs[0], native_merger(merger_inputs[0]))
            output.loss.backward()
            gradient = args["pixel_values"].grad
            projector_gradients = {name: parameter.grad for name, parameter in visual.merger.named_parameters()}
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            assert all(value is not None and torch.isfinite(value).all() for value in projector_gradients.values())
            assert any(value.abs().sum() > 0 for value in projector_gradients.values())
            if position is None:
                default = output.logits.detach(), gradient, projector_gradients
            elif position == "projector_output":
                torch.testing.assert_close((output.logits.detach(), gradient, projector_gradients), default, rtol=0, atol=0)
            metadata = activation.runtime_metadata()
            site = "pre_vision_merger" if before else "post_vision_merger_pre_llm_scatter_tome"
            assert metadata["target_sites"][-2:] == ["visual_output_pre_spatial_merger", site]
            assert metadata["permanent_output_merge_site"] == site
            record = metadata["layout_delta_records"][-1]
            assert record["site"] == "permanent_visual_output"
            assert record["result_layout"]["target_site"] == site
            assert record["result_layout"]["projector_input_group_count"] == (13 if before else 16)
            assert record["source_layout"]["key_source"] == "final_visual_group_keys"
            if before:
                assert metadata["canonical_evidence_applicable"] is False
        finally:
            for hook in hooks:
                hook.remove()
            activation.uninstall()


@pytest.mark.parametrize("position", [None, "projector_input"])
def test_repeated_forward_reinstall_and_original_cache_generate_boundary(position):
    torch = pytest.importorskip("torch")
    torch.manual_seed(452)
    model = _model().eval()
    inputs = _inputs(torch)
    activation = image_fixture.activation(model, _stage(position))
    native_forward = model.forward
    native_visual_forward = model.model.visual.forward
    native_merger_forward = model.model.visual.merger.forward
    first = None
    for _ in range(2):
        activation.install()
        try:
            with torch.no_grad():
                for _ in range(2):
                    output = model(**inputs, use_cache=False)
                    assert output.logits.shape == (1, 17, 64)
                    if first is None:
                        first = output.logits
                    else:
                        torch.testing.assert_close(output.logits, first, rtol=0, atol=0)
                with pytest.raises(BackendIntegrationError, match="use_cache=False"):
                    model(**inputs, use_cache=True)
                with pytest.raises(BackendIntegrationError, match="does not support generate for this activation"):
                    model.generate(**inputs, max_new_tokens=2, use_cache=True)
                # A rejected request leaves the next supported request usable.
                torch.testing.assert_close(model(**inputs, use_cache=False).logits, first, rtol=0, atol=0)
        finally:
            activation.uninstall()
        assert model.forward == native_forward
        assert model.model.visual.forward == native_visual_forward
        assert model.model.visual.merger.forward == native_merger_forward
