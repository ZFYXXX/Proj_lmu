import pytest

torch = pytest.importorskip("torch")
from tests.fixtures import native_image
from mllm_comp.compressors.comparison.operations import OperationComparison

POLICIES = {
    "dop": {"mha_depth": 1, "mlp_depth": 1, "token_reducer": "none"},
    "shortv": {"replaced_layers": [1]},
    "redundancy_lens": {"attention_reduction_layers": [1], "ffn_reduction_layers": [1],
        "attention_range": 4, "operation_backend": "indexed_structural"},
    "yopo": {"head_mask_source": "none", "layer_drop_start_layer": 1, "head_prune_start_layer": 0,
        "ffn_keep_ratio": .5},
    "structural_pruning": {"pruned_layers": [0, 2], "original_layer_count": 4,
        "target_flops_retain_ratio": .5},
    "adallava_l": {"num_prefix_layers": 1, "active_layers": 3, "target_flops_retain_ratio": .75},
    "adallava_h": {"num_prefix_layers": 1, "active_units": 3, "target_flops_retain_ratio": .75},
    "skip_vision": {"attention_visual_retain_ratio": .75, "ffn_visual_retain_ratio": .25},
}


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("method", list(POLICIES))
def test_qwen_operation_native_train_and_cache(method, backbone):
    torch.manual_seed(83)
    model = native_image.qwen_model(backbone).train()
    session = OperationComparison(model, method, POLICIES[method], backbone).install()
    inputs = native_image.inputs(torch, backbone)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -1] = 2
    try:
        result = model(**inputs, labels=labels, use_cache=False)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert inputs["pixel_values"].grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            result = model(**inputs, use_cache=True)
            for token in (4, 5):
                result = model(input_ids=torch.tensor([[token]]), past_key_values=result.past_key_values, use_cache=True)
                assert torch.isfinite(result.logits).all()
            generated = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert generated.shape[-1] == 11
    finally:
        session.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("method", ["cdpruner", "dppruner", "eadp", "hiprune"])
def test_qwen_projector_native_selection(method, backbone):
    from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
    model = native_image.qwen_model(backbone).train()
    model.resize_token_embeddings(151646, mean_resizing=False)
    inputs = native_image.inputs(torch, backbone)
    # A rectangular post-merger grid; preserve native image start/end markers.
    inputs["pixel_values"] = torch.randn(24, 12, requires_grad=True)
    inputs["image_grid_thw"] = torch.tensor([[1, 4, 6]])
    inputs["input_ids"] = torch.tensor([[151644, 872, 198, 7, *([10] * 6), 8, 2, 151645, 3]])
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -1] = 3
    session = QwenProjectorComparison(model, method, {"tokens_per_crop": 2}, backbone).install()
    try:
        result = model(**inputs, labels=labels, use_cache=False)
        assert result.logits.shape[1] == inputs["input_ids"].shape[1] - 4
        assert torch.isfinite(result.loss)
        result.loss.backward()
        assert inputs["pixel_values"].grad.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            result = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert result.shape[-1] == inputs["input_ids"].shape[-1] + 3
        assert session.runtime_metadata()["selection"]["native_grid"] == [2, 3]
    finally:
        session.uninstall()


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl"])
@pytest.mark.parametrize("method,config", [
    ("vlm_pruner", {"prune_after_layer": 2, "retain_ratio": .5}),
    ("core_matching", {"prune_after_layer": 2, "maximum_visual_retain_ratio": .5}),
    ("twigvlm", {"twig_start_layer": 1, "twig_layer_count": 1, "average_retention_ratio": .5}),
])
def test_qwen_decoder_comparisons(method, config, backbone):
    from dataclasses import replace
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from mllm_comp.core.vocabulary import ExecutionPhase
    model = native_image.qwen_model(backbone).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    stage = CompressionStageConfig.from_dict({"name": method, "algorithm": method, "plugin": method,
        "variant": f"{method}.image", "binding": f"{method}.image.{backbone}", "placement": "llm_level",
        "execution_phase": "forward", "target_modalities": ["image"], "config": config})
    session = native_image.activation(model, stage)
    inputs = native_image.inputs(torch, backbone)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -1] = 2
    session.install()
    try:
        result = model(**inputs, labels=labels, use_cache=False)
        assert torch.isfinite(result.loss)
        assert result.logits.shape[1] <= 6
        result.loss.backward()
        assert inputs["pixel_values"].grad.abs().sum() > 0
        session.uninstall()
        session = native_image.activation(model, replace(stage, execution_phase=ExecutionPhase.GENERATE))
        session.install()
        model.eval()
        with torch.no_grad():
            result = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert result.shape[-1] == 11
    finally:
        session.uninstall()
