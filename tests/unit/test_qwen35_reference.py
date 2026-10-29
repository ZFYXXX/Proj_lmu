"""Released text recipes and the native hybrid-cache generation contract."""
import copy
from types import SimpleNamespace

import pytest
import torch

from mllm_comp.compressors.comparison.qwen35_reference import (
    METHODS, Qwen35ReferenceProjector, reference_question_embeddings,
    reference_selection, user_instruction_text,
)


class Tokenizer:
    bos_token_id = 1

    def __init__(self, text="How many?\nAnswer the question with a single word."):
        self.text, self.calls = text, []

    def encode(self, text, **kwargs):
        assert text == "<|im_start|>user\n"
        return [100, 12, 13]

    def convert_tokens_to_ids(self, text):
        assert text == "<|im_end|>"
        return 101

    def decode(self, ids, **kwargs):
        assert 101 not in ids and 10 not in ids and 7 not in ids and 8 not in ids
        return "\n" + self.text

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        if kwargs.get("return_tensors") == "pt":
            return {"input_ids": torch.tensor([[1, 21, 23, 24]])}
        return SimpleNamespace(input_ids=[25, 26])


def fixture():
    model = SimpleNamespace(config=SimpleNamespace(image_token_id=10, video_token_id=11,
                            vision_start_token_id=7, vision_end_token_id=8), device=torch.device("cpu"))
    embed = torch.nn.Embedding(128, 8)
    model.get_input_embeddings = lambda: embed
    ids = torch.tensor([[100, 12, 13, 7, 10, 10, 8, 23, 24, 101, 100, 14, 13]])
    return model, {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def test_mmtok_qwen_prefix_retokenization_and_bos_removal():
    model, inputs = fixture()
    tokenizer = Tokenizer()
    q, audit = reference_question_embeddings(model, tokenizer, inputs, "mmtok")
    assert tokenizer.calls[0][0] == ["Question:", "How", "many?", "Answer", "the", "question", "with", "a", "single", "word."]
    assert tokenizer.calls[0][1]["is_split_into_words"] is True
    assert torch.equal(q, model.get_input_embeddings()(torch.tensor([21, 23, 24])))
    assert audit["question_vectors"] == 3


def test_eadp_retokenizes_only_stripped_instruction():
    model, inputs = fixture()
    tokenizer = Tokenizer()
    reference_question_embeddings(model, tokenizer, inputs, "eadp")
    assert tokenizer.calls[0][0] == tokenizer.text
    assert tokenizer.calls[0][1]["add_special_tokens"] is True


def test_bifocal_strips_task_suffix_and_averages_subwords_per_word():
    model, inputs = fixture()
    tokenizer = Tokenizer()
    q, audit = reference_question_embeddings(model, tokenizer, inputs, "bifocal_stage1")
    assert [r[0] for r in tokenizer.calls] == ["how", "many"]
    expected = model.get_input_embeddings()(torch.tensor([25, 26])).mean(0)
    assert torch.equal(q, expected.expand(2, -1))
    assert audit["question_vectors"] == 2


def test_last_user_extraction_excludes_answer_and_validates_termination():
    model, inputs = fixture()
    tokenizer = Tokenizer()
    question = user_instruction_text(model, tokenizer, inputs)
    other = {**inputs, "input_ids": torch.cat((inputs["input_ids"], torch.tensor([[29, 30]])), 1)}
    other["attention_mask"] = torch.ones_like(other["input_ids"])
    other["labels"] = torch.full_like(other["input_ids"], -100)
    other["labels"][:, -2:] = other["input_ids"][:, -2:]
    assert user_instruction_text(model, tokenizer, other) == question
    with pytest.raises(ValueError, match="Unterminated"):
        user_instruction_text(model, tokenizer, {"input_ids": inputs["input_ids"][:, :8]})


def test_prunesid_does_not_access_question():
    q, audit = reference_question_embeddings(None, None, {}, "prunesid")
    assert q is None and not audit["reads_question"]


def test_mmtok_qwen_temperature_is_explicit_and_selected_features_are_original():
    torch.manual_seed(7)
    x, raw, q = torch.randn(24, 16), torch.randn(24, 8), torch.randn(7, 16)
    out, stats = reference_selection(x, method="mmtok", question=q, grid=(4, 6), raw_features=raw, target_count=8)
    assert stats["text_temperature"] == .01 and stats["vision_temperature"] == .2
    assert torch.equal(out["tokens"], x[out["indices"]])
    assert len(out["indices"].unique()) == 8
    with pytest.raises(ValueError, match="rectangular grid"):
        reference_selection(x, method="mmtok", question=q, grid=(5, 5), raw_features=raw)
    with pytest.raises(ValueError, match="pre-merger"):
        reference_selection(x, method="mmtok", question=q, grid=(4, 6))


@pytest.mark.parametrize("method", METHODS)
def test_native_generation_keeps_positions_cache_and_restores_prefix(method):
    from tests.unit.test_qwen35_transfer import tiny
    model, inputs = tiny()
    before = copy.deepcopy(inputs)
    session = Qwen35ReferenceProjector(model, {"method": method, "target_count": 3}, tokenizer=Tokenizer()).install()
    try:
        with torch.no_grad():
            result = model.generate(**inputs, max_new_tokens=3, min_new_tokens=3,
                                    do_sample=False, pad_token_id=0, eos_token_id=2)
        record = session.runtime_metadata()["selection"]
        assert result.shape[1] == inputs["input_ids"].shape[1] + 3
        assert torch.equal(result[:, :inputs["input_ids"].shape[1]], inputs["input_ids"])
        assert record["result_count"] == 3 and record["decode_position_verified"]
        assert record["observed_visual_tokens_per_decoder_layer"] == [3] * 4
        assert record["text_mrope_preserved"]
    finally:
        session.uninstall()
    assert "generate" not in model.__dict__
    assert all(torch.equal(v, inputs[k]) for k, v in before.items())


@pytest.mark.parametrize("method", METHODS)
def test_identity_equals_dense_and_runs_vision_once(method):
    from tests.unit.test_qwen35_transfer import tiny
    model, inputs = tiny()
    call = dict(max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=2)
    with torch.no_grad():
        expected = model.generate(**inputs, **call)
    calls = []
    hook = model.model.visual.register_forward_hook(lambda *args: calls.append(1))
    session = Qwen35ReferenceProjector(model, {"method": method}, tokenizer=Tokenizer()).install()
    try:
        with torch.no_grad():
            actual = model.generate(**inputs, **call)
        assert torch.equal(actual, expected) and calls == [1]
    finally:
        session.uninstall()
        hook.remove()


def test_generate_exception_restores_rope_state_and_hooks():
    from tests.unit.test_qwen35_transfer import tiny
    model, inputs = tiny()
    delta = model.model.rope_deltas
    layers = model.model.language_model.layers
    hook_counts = [len(layer._forward_pre_hooks) for layer in layers]
    session = Qwen35ReferenceProjector(model, {"method": "mmtok"}, tokenizer=Tokenizer()).install()
    try:
        with pytest.raises(ValueError, match="answer-free"):
            model.generate(**inputs, labels=torch.ones_like(inputs["input_ids"]))
        assert model.model.rope_deltas is delta
        assert hook_counts == [len(layer._forward_pre_hooks) for layer in layers]
    finally:
        session.uninstall()
