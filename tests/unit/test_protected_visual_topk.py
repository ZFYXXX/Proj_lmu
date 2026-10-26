import copy

import pytest
import torch

from mllm_comp.compressors.image.protected_visual_topk import (
    ProtectedTopKSpec, ProtectedWordSelector, protected_context_indices,
)
from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec


@pytest.mark.parametrize("n,k", [(300, 128), (128, 128), (37, 128), (5, 3), (1, 1)])
def test_budget_identity_determinism_and_protection(n, k):
    generator = torch.Generator().manual_seed(123)
    x = torch.randn(n, 12, generator=generator)
    r, v = torch.randn(2, n, generator=generator)
    spec = ProtectedTopKSpec(minimum_anchor=1 if k < 20 else 20)
    a, anchor = protected_context_indices(x, r, v, k, spec)
    b, other_anchor = protected_context_indices(x, r, v, k, spec)
    assert len(a) == min(n, k) and len(a.unique()) == len(a)
    assert torch.equal(a, a.sort().values) and torch.equal(a, b)
    assert torch.equal(anchor, other_anchor) and torch.isin(anchor, a).all()
    if n <= k:
        assert torch.equal(a, torch.arange(n))


def test_cumulative_novelty_against_fixed_initial_anchor():
    x = torch.tensor([[1., 0.], [1., .01], [0., 1.], [1., 0.], [-1., 0.],
                      [0., -1.], [1., 0.], [1., 0.], [0., 1.], [1., 1.]])
    spec = ProtectedTopKSpec(minimum_anchor=2, maximum_anchor_ratio=.75, novelty_patience=2)
    selected, anchor = protected_context_indices(x, -torch.arange(10).float(), torch.zeros(10), 8, spec)
    assert anchor.tolist() == [0, 1, 2, 3, 4]
    assert len(selected) == 8


def test_relevant_similar_tokens_cannot_be_displaced_by_visual_prior():
    x = torch.tensor([[1., 0.], [1., 0.], [0., 1.], [-1., 0.], [0., -1.], [1., 1.]])
    r = torch.tensor([10., 9., 1., 0., -1., -2.])
    v = torch.tensor([-100., -100., 10., 9., 8., 7.])
    selected, anchor = protected_context_indices(x, r, v, 4, ProtectedTopKSpec(minimum_anchor=2))
    assert anchor.tolist() == [0, 1] and torch.isin(anchor, selected).all()


def test_all_ties_use_source_order_and_still_select_unique_tokens():
    x = torch.ones(12, 4)
    selected, _ = protected_context_indices(x, torch.zeros(12), torch.zeros(12), 5,
                                           ProtectedTopKSpec(minimum_anchor=2))
    assert selected.tolist() == list(range(5))


@pytest.mark.parametrize("bad", ["nan", "shape", "k", "device"])
def test_invalid_inputs_fail_explicitly(bad):
    x, r, v, k = torch.ones(8, 4), torch.zeros(8), torch.zeros(8), 4
    if bad == "nan":
        x[0, 0] = float("nan")
    elif bad == "shape":
        r = r[:4]
    elif bad == "k":
        k = True
    else:
        # Configuration validation is also fail-closed; no CUDA needed.
        with pytest.raises(ValueError):
            ProtectedTopKSpec(relevance="unknown")
        return
    with pytest.raises(ValueError):
        protected_context_indices(x, r, v, k, ProtectedTopKSpec())


@pytest.mark.parametrize("mode", ["learned_word", "raw_cosine"])
def test_wrapper_changes_only_policy_and_never_checkpoint_parameters(mode):
    from mllm_comp.analysis.common.topk_training_modes import parameter_digest
    base = WordVisualTopK(WordVisualTopKSpec(12, width=8, target_count=4, match_width=4))
    wrapper = ProtectedWordSelector(base, ProtectedTopKSpec(relevance=mode, minimum_anchor=1))
    before = parameter_digest(base)
    x, q = torch.randn(11, 12), torch.randn(3, 12)
    out = wrapper(x, question=q)
    assert torch.equal(out["tokens"], x[out["indices"]])
    assert len(out["indices"]) == 4 and not out["tokens"].requires_grad
    assert parameter_digest(base) == before and not any(p.requires_grad for p in base.parameters())
    assert out["word_diagnostics"]["protected_count"] + out["word_diagnostics"]["context_count"] == 4
    with pytest.raises(ValueError):
        wrapper(x, question=q, soft=True)


@pytest.mark.parametrize("mode", ["learned_word", "raw_cosine"])
def test_native_qwen_generate_labels_positions_and_identity(mode):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.protected_topk import QwenProtectedTopKComparison
    bundle, state, _, _ = tiny_state()
    base, prompt = bundle["model"], state["prompt_inputs"]
    before = copy.deepcopy(prompt)
    with torch.no_grad():
        dense = base(**prompt, use_cache=False).logits
    identity = QwenProtectedTopKComparison(base, {"target_count":128,
        "protection":{"relevance":mode}}).install()
    try:
        with torch.no_grad():
            torch.testing.assert_close(base(**prompt, use_cache=False).logits, dense, rtol=1e-5, atol=1e-6)
    finally:
        identity.uninstall()
    session = QwenProtectedTopKComparison(base, {"target_count":3,
        "protection":{"relevance":mode, "minimum_anchor":1}}).install()
    try:
        with torch.no_grad():
            base.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()["selection"]
        assert trace["result_count"] == 3 and trace["decode_position_verified"]
        assert trace["observed_visual_tokens_per_decoder_layer"] == [3]*4
        assert set(trace["protected_indices"]) <= set(trace["selected_indices"])
        assert torch.equal(before["input_ids"], prompt["input_ids"])
        labels = torch.full_like(prompt["input_ids"], -100)
        labels[:, -1] = 3
        with torch.no_grad():
            output = base(**prompt, labels=labels, use_cache=False)
        assert output.loss.isfinite()
    finally:
        session.uninstall()


def test_binding_contract_is_registered():
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    registry = builtin_multimodal_compression_contracts()
    binding = next(b for b in registry.bindings() if b.name == "protected_visual_topk.image.qwen25_vl")
    assert binding.legalizer == "qwen25_vl.original_sparse_mrope_compact_cache"


def test_condition_audits_never_change_llm_question(tmp_path):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from tests.unit.test_selector_condition_audit import manifest, SAMPLE
    from mllm_comp.analysis.common.anchor7b_runtime import file_sha
    from mllm_comp.core.serialization import atomic_write_json
    from mllm_comp.integrations.lmms_eval.request_context import sample_request
    from mllm_comp.compressors.comparison.protected_topk import QwenProtectedTopKComparison
    path = tmp_path/"assignment.json"
    atomic_write_json(path, manifest())
    bundle, state, _, _ = tiny_state()
    base, prompt = bundle["model"], state["prompt_inputs"]
    session = QwenProtectedTopKComparison(base, {"target_count":3,
        "protection":{"minimum_anchor":1}, "audit_conditions":str(path),
        "audit_conditions_sha256":file_sha(path)}).install()
    try:
        with torch.no_grad(), sample_request(SAMPLE):
            base.generate(**prompt, max_new_tokens=3, min_new_tokens=3, do_sample=False, pad_token_id=0)
        trace = session.runtime_metadata()["selection"]
        assert set(trace["question_selection_audit"]) == {"shuffled", "generic"}
        assert all(not a["answers_generated_for_other_question"] for a in trace["question_selection_audit"].values())
    finally:
        session.uninstall()


def test_runner_completes_all_smokes_before_full_and_resumes(tmp_path, monkeypatch):
    from scripts.train.qwen25_vl import run_protected_visual_topk as runner
    from mllm_comp.core.serialization import atomic_write_json
    protocol = {"source":{"test":"snapshot"}, "arms":{}, "limits":[]}
    atomic_write_json(tmp_path/"manifest.json", {"completed_cells":[]})
    monkeypatch.setattr(runner, "validate", lambda root: protocol)
    monkeypatch.setattr(runner, "available_gpus", lambda requested: requested)
    calls = []
    monkeypatch.setattr(runner, "config", lambda root, p, arm, phase, task, run:
        {"arm":arm, "phase":phase, "task":task, "run":str(run)})
    def execute(command, log, gpus, env):
        import yaml
        cfg = yaml.safe_load(__import__('pathlib').Path(command[-1]).read_text())
        if cfg["phase"] == "full":
            assert (tmp_path/"smoke_complete.json").exists()
            assert len([c for c in calls if c[1] == "smoke"]) == 9
        calls.append((cfg["arm"], cfg["phase"], cfg["task"]))
    monkeypatch.setattr(runner.word.context.native, "execute", execute)
    monkeypatch.setattr(runner, "check", lambda root,p,run,arm,phase,task:
        {"requests":8 if phase == "smoke" else runner.word.context.COUNTS[task]})
    monkeypatch.setattr(runner, "report", lambda root,p: None)
    runner.pipeline(tmp_path, 1)
    assert len(calls) == 15
    assert runner.read(tmp_path/"manifest.json")["status"] == "complete"
    runner.pipeline(tmp_path, 1)
    assert len(calls) == 15  # completed cells are checked, not regenerated


def test_runner_rejects_edited_completed_config(tmp_path, monkeypatch):
    from scripts.train.qwen25_vl import run_protected_visual_topk as runner
    from mllm_comp.core.serialization import atomic_write_json, atomic_write_text
    atomic_write_json(tmp_path/"manifest.json", {"completed_cells":[]})
    monkeypatch.setattr(runner, "validate", lambda root: {"source":{"test":"snapshot"}})
    config = tmp_path/"old.yaml"
    atomic_write_text(config, "changed: true")
    atomic_write_json(tmp_path/"eval/smoke/unified_parity/chartqa/complete.json",
        {"source":{"test":"snapshot"}, "config":str(config), "config_sha256":"wrong"})
    with pytest.raises(ValueError, match="provenance"):
        runner.pipeline(tmp_path, 1)
    assert runner.read(tmp_path/"manifest.json")["status"] == "failed"
