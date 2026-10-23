"""MMTok/ZOO retain per-crop dense scoring and select raw AnyRes values."""

import copy

import pytest

from tests.fixtures import native_image as image_fixture
from tests.integration.test_image_projector_positions_cpu import _stage
from tests.integration.test_projector_aim_vispruner_llava_onevision_cpu import _inputs
from tests.integration.test_projector_llava_onevision_positions_cpu import _model, _packed


@pytest.mark.parametrize("algorithm", ["mmtok", "zoo_prune"])
def test_pre_anyres_fixed_crop_sources_projection_and_gradients(algorithm, monkeypatch):
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.llava_onevision import image_compression as adapter
    from mllm_comp.backbones.llava_onevision import provider_bridge as provider
    from mllm_comp.compressors.image.mmtok import greedy_mmtok_coverage_indices
    from mllm_comp.compressors.image.zoo_prune import sample_zoo_directions, sensitivity_aware_diverse_indices
    torch.manual_seed(409)
    base = _model().train()
    expected_run = None
    kernel_name = "mmtok_image_tokens" if algorithm == "mmtok" else "zoo_prune_image_tokens"
    kernel = getattr(adapter, kernel_name)
    monkeypatch.setattr(provider, "_packed_qk_metrics", lambda *args: pytest.fail("These methods retain their original crop sources."))
    for position in ("projector_output", "projector_input"):
        model = copy.deepcopy(base)
        projector = model.model.mm_projector
        reference = copy.deepcopy(projector)
        stage = _stage("llava_onevision", algorithm, position)
        stage.config = {"tokens_per_crop": 4} if algorithm == "mmtok" else {
            "target_count": 4, "num_directions": 4, "low_rank_rank": 2, "low_rank_solver": "randomized"}
        activation = image_fixture.activation(model, stage)
        activation.install()
        activation._bridge.prepare_training_forward()
        projection_inputs, decoder_inputs, selections, vision_calls = [], [], [], []
        handles = [projector.register_forward_pre_hook(lambda module, args: projection_inputs.append(args[0])),
            model.model.register_forward_pre_hook(lambda module, args, kwargs: decoder_inputs.append(kwargs["inputs_embeds"]), with_kwargs=True),
            model.model.vision_tower.register_forward_hook(lambda module, args, output: vision_calls.append(1))]

        def select(projected, vision, third, **kwargs):
            crop = len(selections)
            raw = projection_inputs[0][crop]
            assert projected.shape == (729, 24) and vision.shape == (729, 16)
            torch.testing.assert_close(vision, raw)
            with torch.no_grad():
                torch.testing.assert_close(projected, reference(raw))
            result = kernel(projected, vision, third, **kwargs)
            spec = kwargs["spec"]
            if algorithm == "mmtok":
                torch.testing.assert_close(third, model.model.embed_tokens(torch.tensor([1, 2, 3])))
                assert kwargs["target_count"] == 4
                norm = lambda values: torch.nn.functional.normalize(values.float(), dim=-1, eps=1e-8)
                text_scores = (norm(third) @ norm(projected).T / spec.text_temperature).softmax(1) / third.shape[0]
                visual_scores = (norm(vision) @ norm(vision).T / spec.vision_temperature).softmax(1) / vision.shape[0]
                indices = greedy_mmtok_coverage_indices(torch.cat((text_scores, spec.alpha * visual_scores)), 4).sort().values
            else:
                assert tuple(third) == tuple(projector.children())
                assert spec.projector_proxy == "low_rank" and spec.low_rank_rank == 2 and spec.num_directions == 4
                proxy = kwargs["proxy_cache"].resolve(third, spec)
                directions = sample_zoo_directions(4, 16, device=vision.device, dtype=vision.dtype, seed=spec.seed)
                with torch.no_grad():
                    sensitivities = torch.stack([
                        (proxy(vision + spec.step_size * direction) - proxy(vision - spec.step_size * direction)).float().norm(dim=-1) / (2 * spec.step_size)
                        for direction in directions
                    ]).mean(0)
                torch.testing.assert_close(result.sensitivity, sensitivities)
                indices, _ = sensitivity_aware_diverse_indices(projected, sensitivities, 4)
                assert result.metadata["low_rank_solver_used"] == "randomized+randomized"
            torch.testing.assert_close(result.indices, indices, atol=0, rtol=0)
            selections.append(result.indices.tolist())
            return result

        monkeypatch.setattr(adapter, kernel_name, select)
        torch.manual_seed(410)
        args = _inputs(torch)
        try:
            output = model(**args)
            assert len(vision_calls) == 1 and len(selections) == 2 and all(len(ids) == 4 for ids in selections)
            assert output.logits.shape[1] == 38
            metadata = activation.runtime_metadata()
            record = metadata["layout_delta_records"][0]
            mapping = record["old_to_new"]
            selected = [i for i, target in enumerate(mapping) if target is not None]
            source_newlines = record["protected_indices"]
            newline_targets = [mapping[i] for i in source_newlines]
            assert len(source_newlines) == 27
            actual = decoder_inputs[0][0, 1:-2]
            torch.testing.assert_close(actual[newline_targets], model.model.image_newline.expand(27, -1), atol=0, rtol=0)
            if position == "projector_input":
                assert [tuple(x.shape) for x in projection_inputs] == [(2, 729, 16), (8, 16)]
                raw = _packed(projection_inputs[0], projection_inputs[0].new_zeros(16))[selected]
                content_indices = [i for i in range(35) if i not in newline_targets]
                content = projector(raw[content_indices])
                expected = content.new_zeros((35, 24)).index_copy(0, torch.tensor(content_indices), content)
                expected = expected.index_copy(0, torch.tensor(newline_targets), model.model.image_newline.expand(27, -1))
                torch.testing.assert_close(actual, expected)
                parameters = (projection_inputs[0], *tuple(projector.parameters()), model.model.image_newline)
                upstream = torch.randn_like(actual)
                torch.testing.assert_close(torch.autograd.grad(actual, parameters, upstream, retain_graph=True),
                    torch.autograd.grad(expected, parameters, upstream, retain_graph=True))
                assert record["site"] == "pre_vision_projector" and metadata["dense_projector_metric_source"] is True
            else:
                assert len(projection_inputs) == 1
            output.loss.backward()
            gradient = args["images"][0].grad
            assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
            if expected_run is None:
                expected_run = selections, output.logits.detach(), gradient.clone()
            else:
                assert selections == expected_run[0]
                torch.testing.assert_close(output.logits, expected_run[1], atol=2e-6, rtol=2e-5)
                torch.testing.assert_close(gradient, expected_run[2], atol=2e-6, rtol=2e-5)
        finally:
            for handle in handles:
                handle.remove()
            activation.uninstall()
