"""DPP numerics and native OneVision selection, layout and training integration."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from mllm_comp.compressors.comparison.cdpruner import (
    normalize_dppruner_config, select_cdpruner_tokens, official_fast_map_dpp_indices,
)


def test_dpp_matches_independent_greedy_loop_and_preserves_gradients():
    torch.manual_seed(321)
    features = torch.randn(2, 12, 8, requires_grad=True)
    kernel = features.detach() @ features.detach().transpose(1, 2)
    residual = torch.diagonal(kernel, dim1=1, dim2=2).clone()
    rows = torch.arange(2)
    factors = torch.zeros(4, 2, 12)
    expected = []
    for step in range(4):
        chosen = residual.argmax(-1)
        expected.append(chosen)
        numerator = kernel[rows, chosen] - (factors[:step] * factors[:step, rows, chosen][:, :, None]).sum(0)
        factors[step] = numerator / residual[rows, chosen].clamp_min(1e-12).sqrt()[:, None]
        residual -= factors[step].square()
        residual[rows, chosen] = -torch.inf
    reference = torch.stack(expected, dim=1).sort(1).values
    assert torch.equal(official_fast_map_dpp_indices(kernel, 4), reference)
    output, selected, _ = select_cdpruner_tokens(features, None,
        cfg=normalize_dppruner_config({"tokens_per_crop": 4}))
    output.square().sum().backward()
    assert output.shape == (2, 4, 8)
    assert torch.isfinite(features.grad).all() and features.grad.abs().sum() > 0
    for row in range(2):
        assert features.grad[row].abs().sum(-1).nonzero().flatten().tolist() == selected[row].tolist()


@pytest.mark.parametrize("method", ["cdpruner", "dppruner", "eadp", "hiprune"])
def test_native_ov_forward_backward_and_uninstall(method):
    pytest.importorskip("llava")
    from tests.fixtures import native_image
    from mllm_comp.pipeline.stage import CompressionStageConfig
    model = native_image.llava_model().train()
    model.resize_token_embeddings(151646, mean_resizing=False)
    args = native_image.inputs(torch, "llava_onevision")
    ids = torch.tensor([[151644, 872, 198, -200, 2, 151645, 3]])
    args.update(input_ids=ids, attention_mask=torch.ones_like(ids))
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 3
    stage = CompressionStageConfig.from_dict({
        "name": method, "algorithm": method, "plugin": method, "variant": f"{method}.image",
        "binding": f"{method}.image.llava_onevision", "placement": "projector_level",
        "execution_phase": "forward", "target_modalities": ["image"],
        "config": {"tokens_per_crop": 8, **({"object_layer": 1} if method == "hiprune" else {})},
    })
    original = model.forward.__func__
    activation = native_image.activation(model, stage)
    activation.install()
    try:
        output = model(**args, labels=labels, use_cache=False)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        assert model.model.mm_projector.weight.grad.abs().sum() > 0
        assert output.logits.shape[1] == ids.shape[1] - 1 + 8
        record = activation.runtime_metadata()["layout_delta_records"][-1]
        assert record["source_count"] == 729 and record["result_count"] == 8
        activation.uninstall()
        from dataclasses import replace
        from mllm_comp.core.vocabulary import ExecutionPhase
        activation = native_image.activation(model, replace(stage, execution_phase=ExecutionPhase.GENERATE))
        activation.install()
        model.eval()
        with torch.no_grad():
            args["inputs"] = args.pop("input_ids")
            result = model.generate(**args, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert result.shape[-1] >= 3
    finally:
        activation.uninstall()
    assert model.forward.__func__ is original
    assert not hasattr(model, "mllm_compression_bridge")


def test_instruction_guidance_excludes_system_and_answer_tokens():
    from mllm_comp.compressors.comparison.llava import instruction_ids
    ids = torch.tensor([[151644, 10, 198, 5, 151645, 151644, 872, 198, -200, 6, 7, 151645, 8]])
    labels = torch.full_like(ids, -100)
    labels[0, -1] = 8
    assert instruction_ids(ids, torch.ones_like(ids), labels).tolist() == [6, 7]
