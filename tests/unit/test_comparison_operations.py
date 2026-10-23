"""Native operator semantics, gradients and unequal per-layer caches."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("llava")

from mllm_comp.compressors.comparison.operations import OperationComparison
from tests.fixtures import native_image

POLICIES = {
    "dop": {"mha_depth": 1, "mlp_depth": 1, "token_reducer": "none"},
    "shortv": {"replaced_layers": [1]},
    "redundancy_lens": {"attention_reduction_layers": [1], "ffn_reduction_layers": [1],
        "attention_range": 4, "operation_backend": "indexed_structural"},
    "yopo": {"head_mask_source": "none", "layer_drop_start_layer": 1, "head_prune_start_layer": 0,
        "ffn_keep_ratio": .5},
}


@pytest.mark.parametrize("method", list(POLICIES))
def test_native_operator_forward_backward_and_cached_decode(method):
    torch.manual_seed(84)
    model = native_image.llava_model().train()
    inputs = native_image.inputs(torch, "llava_onevision")
    labels = inputs["input_ids"].clone()
    labels[:, :2] = -100
    session = OperationComparison(model, method, POLICIES[method], "llava_onevision")
    original = model.forward.__func__
    session.install()
    try:
        result = model(**inputs, labels=labels, use_cache=False)
        assert torch.isfinite(result.loss)
        result.loss.backward()
        gradient = model.model.mm_projector.weight.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
        model.eval()
        with torch.no_grad():
            result = model(**inputs, use_cache=True)
            past = result.past_key_values
            assert past.get_seq_length(0) == 732
            assert past.get_seq_length(1) == (3 if method == "dop" else 732)
            result = model(input_ids=torch.tensor([[4]]), past_key_values=past, use_cache=True)
            assert torch.isfinite(result.logits).all()
            assert past.get_seq_length(1) == (4 if method == "dop" else 733)
            result = model(input_ids=torch.tensor([[5]]), past_key_values=past, use_cache=True)
            assert torch.isfinite(result.logits).all()
        assert session.runtime_metadata()["forward_calls"] == 4
    finally:
        session.uninstall()
    assert model.forward.__func__ is original
    assert not hasattr(model, "_mllm_comparison_operations")


def test_shortv_is_full_kv_with_visual_residuals_preserved():
    model = native_image.llava_model().eval()
    session = OperationComparison(model, "shortv", {"replaced_layers": [1]}, "llava_onevision")
    captured = {}
    def capture(_module, args, output):
        captured["attn"] = output[0]
    session.install()
    hook = model.model.layers[1].self_attn.register_forward_hook(capture)
    try:
        with torch.no_grad():
            model(**native_image.inputs(torch, "llava_onevision"), use_cache=False)
        assert torch.count_nonzero(captured["attn"][:, 1:730]) == 0
        assert captured["attn"][:, 730:].abs().sum() > 0
    finally:
        hook.remove()
        session.uninstall()


@pytest.mark.parametrize("backbone", ["llava_onevision", "qwen25_vl", "qwen3_vl"])
def test_dop_v_combines_visual_selection_and_depth_policy(backbone):
    from mllm_comp.pipeline.stage import CompressionStageConfig
    model = native_image.model(backbone).train()
    stage = CompressionStageConfig.from_dict({"name": "dop", "algorithm": "dop", "plugin": "dop",
        "variant": "dop.image", "binding": f"dop.image.{backbone}", "placement": "llm_level",
        "method": "operation_compression", "object_family": "module_unit", "execution_phase": "prepare",
        "modalities": ["image"], "target_modalities": ["image"],
        "config": {"mha_depth": 1, "mlp_depth": 1, "token_reducer": "vispruner", "tokens_per_crop": 2}})
    session = native_image.activation(model, stage)
    session.install()
    inputs = native_image.inputs(torch, backbone)
    labels = torch.full_like(inputs["input_ids"], -100)
    labels[:, -1] = 3
    try:
        output = model(**inputs, labels=labels, use_cache=False)
        assert output.logits.shape[1] == (5 if backbone == "llava_onevision" else 6)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        pixels = inputs["images" if backbone == "llava_onevision" else "pixel_values"]
        assert pixels.grad.abs().sum() > 0
        metadata = session.runtime_metadata()
        assert metadata["runtime"]["visual_tokens"] == 2
        assert metadata["operation_records"] and metadata["layout_delta_records"]
        model.eval()
        generation_inputs = dict(inputs)
        if backbone == "llava_onevision":
            generation_inputs["inputs"] = generation_inputs.pop("input_ids")
        with torch.no_grad():
            generated = model.generate(**generation_inputs, max_new_tokens=3, min_new_tokens=3, do_sample=False)
            assert generated.shape[-1] == (3 if backbone == "llava_onevision" else 11)
        state = session.request
        assert len(state["cache_positions"][1]) == (5 if backbone == "llava_onevision" else 6)
    finally:
        session.uninstall()
