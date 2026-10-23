"""Generate parity and gold-free selection gates for the matched evaluation."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from mllm_comp.analysis.common.projector_comparison import (
    METHODS, capture_hiprune, select_indices, compact_prompt, decode_prompt,
)
from mllm_comp.compressors.comparison.qwen_projector import QwenProjectorComparison
from tests.fixtures import native_image


def tiny_state():
    torch.manual_seed(501)
    model = native_image.qwen_model("qwen25_vl").eval()
    model.resize_token_embeddings(151646, mean_resizing=False)
    model.generation_config.eos_token_id = 151643
    inputs = native_image.inputs(torch, "qwen25_vl")
    inputs["pixel_values"] = torch.randn(24, 12)
    inputs["image_grid_thw"] = torch.tensor([[1, 4, 6]])
    inputs["input_ids"] = torch.tensor([[151644, 872, 198, 9, 7, *([10] * 6), 8, 2, 151645, 3]])
    inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])
    bundle = {"model": model, "llm": model.model.language_model,
        "device": torch.device("cpu"),
        "processor": SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids, **kw: str(ids)))}
    with torch.no_grad(), capture_hiprune(bundle, inputs["image_grid_thw"], budget=2) as (source, config):
        features, = model.model.get_image_features(inputs["pixel_values"], inputs["image_grid_thw"])
        hidden = model.get_input_embeddings()(inputs["input_ids"])
        visual = (inputs["input_ids"][0] == 10).nonzero(as_tuple=True)[0].tolist()
        hidden[0, visual] = features
        positions, _ = model.model.get_rope_index(inputs["input_ids"], inputs["image_grid_thw"],
                                                attention_mask=inputs["attention_mask"])
    state = {"features": features, "hidden": hidden, "positions": positions[-3:],
        "prompt_inputs": inputs, "visual": visual, "prefix": hidden.shape[1], "height": 2, "width": 3}
    return bundle, state, source["attention"], config


@pytest.mark.parametrize("method", METHODS)
def test_direct_kernel_decode_equals_native_framework_binding(method):
    bundle, state, attention, config = tiny_state()
    model = bundle["model"]
    with torch.inference_mode():
        selected, trace = select_indices(bundle, state, method, attention=attention,
                                          capture_config=config, budget=2)
        direct = decode_prompt(bundle, state, max_new=3, selected=selected)
        session = QwenProjectorComparison(model, method, {"tokens_per_crop": 2}, "qwen25_vl").install()
        try:
            native = model.generate(**state["prompt_inputs"], max_new_tokens=3, do_sample=False,
                eos_token_id=151643, pad_token_id=0, repetition_penalty=1.0)
            assert native[0, state["prefix"]:].tolist() == direct["generated_token_ids"]
            assert session.runtime_metadata()["selection"]["selected_indices"] == trace["selected_indices"]
        finally:
            session.uninstall()
        assert direct["retained_visual_tokens"] == 2
        assert direct["text_positions_preserved"]
        assert direct["first_appended_token_position"] == int(state["positions"].max()) + 1


def test_compaction_preserves_original_text_ids_positions_and_rejects_duplicate():
    _, state, _, _ = tiny_state()
    compact = compact_prompt(state, torch.tensor([1, 4]))
    before = state["prompt_inputs"]["input_ids"][0] != 10
    after = compact["input_ids"][0] != 10
    assert torch.equal(compact["input_ids"][:, after], state["prompt_inputs"]["input_ids"][:, before])
    assert torch.equal(compact["positions"][:, :, after], state["positions"][:, :, before])
    assert compact["prefix"] == state["prefix"] - 4
    assert (compact["input_ids"] == 10).sum() == 2
    with pytest.raises(ValueError, match="unique"):
        compact_prompt(state, torch.tensor([1, 1]))


@pytest.mark.parametrize("method", ("cdpruner", "eadp"))
def test_selection_does_not_read_teacher_answer(method):
    bundle, state, _, _ = tiny_state()
    with torch.inference_mode():
        selected, _ = select_indices(bundle, state, method, budget=2)
        contaminated = dict(state, teacher_inputs={"labels": torch.tensor([[12345]])},
                            target="THIS ANSWER MUST NEVER BE USED")
        changed, _ = select_indices(bundle, contaminated, method, budget=2)
    assert torch.equal(selected, changed)


@pytest.mark.parametrize("method", METHODS)
def test_native_generate_decode_position_uses_compacted_cache_offset(method):
    bundle, state, _, _ = tiny_state()
    model = bundle["model"]
    expected = int(state["positions"].max()) + 1
    observed = []
    def capture_decode(module, args, kwargs):
        past = kwargs.get("past_key_values")
        if past is not None and past.get_seq_length() > 0:
            observed.append(int(kwargs["position_ids"][-3:].max()))
    hook = bundle["llm"].register_forward_pre_hook(capture_decode, with_kwargs=True)
    session = QwenProjectorComparison(model, method, {"tokens_per_crop": 2}, "qwen25_vl").install()
    try:
        with torch.no_grad():
            model.generate(**state["prompt_inputs"], max_new_tokens=3, min_new_tokens=3,
                           do_sample=False, eos_token_id=151643, pad_token_id=0)
        assert observed == [expected, expected + 1]
    finally:
        session.uninstall()
        hook.remove()
