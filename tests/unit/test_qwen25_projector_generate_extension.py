"""Real Transformers CPU gates, not paper-quality verification."""
import pytest
import torch

from mllm_comp.core.vocabulary import ExecutionPhase
from tests.unit.test_projector_comparison_evaluation import tiny_state
from tests.unit.test_qwen25_vl_image_compression_provider_bridge import _activation


METHODS = ("divprune", "visionzip", "aim", "gap", "mmtok")


@pytest.mark.parametrize("method", METHODS)
def test_public_registry_exposes_generate_only_after_lifecycle_is_implemented(method):
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    binding = builtin_multimodal_compression_contracts().binding(f"{method}.image.qwen25_vl")
    assert ExecutionPhase.GENERATE in binding.supported_phases


def activation(model, method, count):
    cfg = {"budget" if method == "visionzip" else "target_count": count}
    if method == "visionzip":
        cfg["attention_mode"] = "mean_received"
    executor = _activation(model, f"{method}.image.qwen25_vl", config=cfg, phase=ExecutionPhase.GENERATE)
    if method == "mmtok":
        from types import SimpleNamespace
        class Tokenizer:
            bos_token_id = None
            def decode(self, ids, **kwargs):
                assert ids == [9, 2]  # No image, system or assistant tokens.
                return "color chart"
            def __call__(self, words, **kwargs):
                assert words == ["Question:", "color", "chart"]
                return {"input_ids": torch.tensor([[9, 2]])}
        executor.runtime.processor = SimpleNamespace(tokenizer=Tokenizer())
    return executor


@pytest.mark.parametrize("method", METHODS)
def test_projector_generate_exact_budget_and_verified_native_decode(method):
    bundle, state, _, _ = tiny_state()
    model = bundle["model"]
    model.model.visual.fullatt_block_indexes = [1, 2]
    executor = activation(model, method, 3)
    executor.install()
    try:
        with torch.inference_mode():
            result = model.generate(**state["prompt_inputs"], max_new_tokens=3, min_new_tokens=3,
                do_sample=False, eos_token_id=151643, pad_token_id=0)
        assert result.shape[-1] == state["prefix"] + 3
        metadata = executor.runtime_metadata()
        delta = metadata["latest_layout_delta"]
        assert delta["source_count"] == 6
        assert delta["result_count"] == 3
        generated = metadata["projector_generate"]["last"]
        assert generated["prompt_removed"] == 3
        assert generated["prefill_calls"] == 1
        assert generated["decode_position_verified"] is True
        assert generated["expected_decode_position"] == int(state["positions"].max()) + 1
    finally:
        executor.uninstall()


@pytest.mark.parametrize("method", METHODS)
def test_projector_identity_budget_preserves_dense_generation_and_logits(method):
    bundle, state, _, _ = tiny_state()
    model = bundle["model"]
    model.model.visual.fullatt_block_indexes = [1, 2]
    kwargs = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False,
                  eos_token_id=151643, pad_token_id=0, return_dict_in_generate=True, output_scores=True)
    with torch.inference_mode():
        dense = model.generate(**state["prompt_inputs"], **kwargs)
    executor = activation(model, method, 6)
    executor.install()
    try:
        with torch.inference_mode():
            actual = model.generate(**state["prompt_inputs"], **kwargs)
        assert torch.equal(dense.sequences, actual.sequences)
        for expected_score, actual_score in zip(dense.scores, actual.scores):
            torch.testing.assert_close(expected_score, actual_score, rtol=2e-5, atol=2e-5)
        assert executor.runtime_metadata()["projector_generate"]["last"]["decode_position_verified"] is True
    finally:
        executor.uninstall()
