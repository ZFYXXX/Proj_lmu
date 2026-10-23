from copy import deepcopy

import pytest

from scripts.eval.lmms_eval.run_anchor7b_full import prompt_target, score_prediction, clustered_delta


def test_native_prompts_and_targets():
    prompt, target = prompt_target("chartqa", {"question": "How many?", "answer": "42"})
    assert prompt == "How many?\nAnswer the question with a single word."
    assert target == "42"
    doc = {"question": "WHAT word is visible?", "answers": ["Hello"]*10, "question_id": 12}
    prompt, target = prompt_target("textvqa", doc)
    assert prompt == "What word is visible?\nAnswer the question using a single word or phrase."
    assert target == "Hello"
    assert "Hello" not in prompt
    prompt, target = prompt_target("ai2d", {"question": "Which one?", "options": ["circle", "triangle"], "answer": 1})
    assert prompt == "Which one?\nA. circle\nB. triangle\nAnswer with the option's letter from the given choices directly."
    assert target == "B"


def test_native_textvqa_scoring_keeps_all_answers_unchanged():
    doc = {"question": "Which?", "question_id": 12, "answers": ["The CAT"]*3+["dog"]*7}
    before = deepcopy(doc)
    score, _, strict, _ = score_prediction("textvqa", "cat", doc)
    assert score == pytest.approx(.9)
    assert strict is None
    assert doc == before
    assert score_prediction("textvqa", "cat", doc)[0] == pytest.approx(.9)


def test_chartqa_relaxed_and_strict_are_separate():
    score, _, strict, _ = score_prediction("chartqa", "41", {"answer": "42", "type": "human_test"})
    assert score == 1.
    assert strict == 0.


def test_paired_cluster_bootstrap_retains_question_weighted_estimand():
    def question(a, b):
        return {"adapter": {"primary_task_score": a}, "dense": {"primary_task_score": b}}
    images = [[question(1., 0.) for _ in range(10)], [question(0., 1.)]]
    result = clustered_delta(images, "adapter", repeats=1000)
    assert result["mean"] == pytest.approx(9/11)
    assert result["n_images"] == 2
    assert result["n_questions"] == 11
    assert result["ci95"] == [-1., 1.]


def test_revised_full_scope_excludes_textvqa_and_prumerge():
    from scripts.eval.lmms_eval.run_anchor7b_full import DATASETS, METHODS
    assert set(DATASETS) == {"chartqa", "ai2d"}
    assert METHODS == ["dense", "feature_anchor_adapter"]


def test_compatible_prediction_reuse_copies_only_two_retained_arms_with_provenance(tmp_path):
    import json
    from scripts.eval.lmms_eval.run_anchor7b_full import reuse_conditions, compatible_reuse, METHODS
    from mllm_comp.analysis.common.evidence_interaction import digest
    old = {key: "fixed" for key in ["checkpoint", "revision", "base_identity_sha256", "adapter_sha256",
        "adapter_config_sha256", "training_manifest_sha256", "dtype", "attention"]}
    old.update(budget=128, seed=42, processor={"min_pixels": 200704},
        generation={"do_sample": False, "repetition_penalty": 1., "chartqa_max_new": 16, "textvqa_max_new": 32})
    new = deepcopy(old)
    new["generation"].pop("textvqa_max_new")
    reuse = tmp_path/"old";reuse.mkdir()
    (reuse/"protocol.json").write_text(json.dumps(old))
    new["reuse_run"] = str(reuse)
    compatible_reuse(new, old)
    entry = {"sample_id": "chartqa:0", "image_fingerprint": "image0"}
    source = reuse/"chartqa/samples/full/chartqa_0/conditions.jsonl"
    source.parent.mkdir(parents=True)
    original_hash = digest(json.dumps(old, sort_keys=True)+"chartqa"+"full")
    rows = [{**entry, "variant": name, "config_hash": original_hash, "status": "completed",
        "gold_answer_in_decode_context": False, "base_identity_sha256": "fixed", "adapter_sha256": "fixed",
        "repetition_penalty": 1., "retained_visual_tokens": 400 if name == "dense" else 128,
        "prediction": "42", "primary_task_score": 1.}
        for name in METHODS+["feature_anchor_mean_k128", "prumerge_plus_k128"]]
    source.write_text("".join(json.dumps(r)+"\n" for r in rows)+"{unfinished")
    before = source.read_bytes()
    saved = {}; ledger = tmp_path/"conditions.jsonl"
    assert reuse_conditions(tmp_path, new, "chartqa", "full", entry, "newhash", ledger, saved) == 2
    assert {key[1] for key in saved} == set(METHODS)
    assert all(r["prediction"] == "42" and r["primary_task_score"] == 1. and r["config_hash"] == "newhash" for r in saved.values())
    assert all(r["reuse_provenance"]["original_config_hash"] == original_hash for r in saved.values())
    assert source.read_bytes() == before
    assert reuse_conditions(tmp_path, new, "chartqa", "full", entry, "newhash", ledger, saved) == 0
    with pytest.raises(ValueError, match="changed generation"):
        compatible_reuse({**new, "generation": {"repetition_penalty": 1.05}}, old)
