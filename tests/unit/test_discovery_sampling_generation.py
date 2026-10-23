import pytest

from mllm_comp.analysis.common.discovery_sampling import sample_unique_image_docs
from mllm_comp.analysis.common.visual_token_phenomena import media_fingerprint
from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings, primary_task_score


def test_image_sampling_is_disjoint_order_independent_and_ignores_answers() -> None:
    from PIL import Image
    docs = [(2*i+j, {"image": Image.new("RGB", (2, 2), (i, 0, 0)), "answer": str(j)})
            for i in range(9) for j in range(2)]
    excluded = {media_fingerprint([docs[0][1]["image"]])}
    select = lambda values: sample_unique_image_docs(values, doc_to_visual=lambda d: [d["image"]], count=4, seed=42, excluded=excluded)
    a, metadata = select(docs)
    b, _ = select(list(reversed(docs)))
    changed = [(i, {**d, "answer": "changed"}) for i, d in docs]
    c, _ = select(changed)
    assert [i for i, _ in a] == [i for i, _ in b] == [i for i, _ in c]
    fingerprints = [media_fingerprint([d["image"]]) for _, d in a]
    assert len(set(fingerprints)) == 4
    assert not set(fingerprints) & excluded
    assert metadata["selection_reads_answers"] is False
    with pytest.raises(ValueError, match="independent eligible"):
        sample_unique_image_docs(docs, doc_to_visual=lambda d: [d["image"]], count=10, seed=42, excluded=excluded)


def test_gold_free_decode_uses_original_mrope_max_and_stops_on_eos() -> None:
    torch = pytest.importorskip("torch")
    prompt = torch.zeros(1, 2, 1)
    positions = torch.tensor([[[0, 4]], [[0, 6]], [[0, 8]]])
    seen = []
    def forward(hidden, ids):
        seen.append((hidden.clone(), ids.clone()))
        logits = torch.zeros(1, 1, 4)
        logits[0, 0, 2 if hidden.shape[1] == 2 else 3] = 10
        return logits
    result = greedy_decode_embeddings(
        prompt_hidden=prompt, prompt_position_ids=positions,
        token_embedding=lambda token: token.float().unsqueeze(-1), forward_logits=forward,
        max_new_tokens=5, eos_token_ids=[3],
    )
    assert result["generated_token_ids"] == [2, 3]
    assert result["first_appended_token_position"] == 9
    assert torch.equal(seen[1][1][:, :, -1], torch.tensor([[9], [9], [9]]))
    assert seen[1][0][0, -1, 0].item() == 2
    assert prompt.shape[1] == 2
    assert result["stopped_on_eos"] is True
    assert result["gold_answer_in_decode_context"] is False


def test_native_score_extraction_handles_soft_vqa_and_pope() -> None:
    assert primary_task_score("textvqa_val", {"exact_match": .6}) == .6
    assert primary_task_score("pope_random", {"pope_accuracy": {"score": 1}}) == 1
