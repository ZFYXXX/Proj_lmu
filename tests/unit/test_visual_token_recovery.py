import copy

import pytest
import torch

from mllm_comp.compressors.image.visual_token_recovery import (
    RecoverySpec, RecoveredProtectedSelector, recover_visual_tokens,
)
from mllm_comp.compressors.image.protected_visual_topk import ProtectedTopKSpec, ProtectedWordSelector
from mllm_comp.compressors.image.word_visual_topk import WordVisualTopK, WordVisualTopKSpec


@pytest.mark.parametrize("mode", ["neighbor_mean", "query_gated_residual"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_exact_population_budget_protection_and_determinism(mode, dtype):
    gen = torch.Generator().manual_seed(77)
    x = torch.randn(31, 12, generator=gen).to(dtype)
    s, p = torch.tensor([0, 3, 8, 11, 20]), torch.tensor([3, 11])
    r, v = torch.randn(2, 31, generator=gen)
    before = x.clone()
    out, trace = recover_visual_tokens(x, s, p, r, v, RecoverySpec(mode))
    again, other = recover_visual_tokens(x, s, p, r, v, RecoverySpec(mode))
    assert torch.equal(out, again) and trace == other
    assert torch.equal(x, before) and out.shape == (5, 12) and out.dtype == dtype
    assert torch.equal(out[torch.isin(s, p)], x[p])
    assert sorted(s.tolist()+trace["recovery_dropped_indices"]) == list(range(31))
    assert sum(trace["recovery_assignment_counts"]) == 26
    assert not set(trace["recovery_target_indices"]) & set(p.tolist())
    assert not out.requires_grad and trace["recovery_extra_tokens"] == 0
    if mode == "query_gated_residual":
        assert all(0 <= b <= .25+1e-6 for b in trace["recovery_blend"])


@pytest.mark.parametrize("mode", ["neighbor_mean", "query_gated_residual"])
def test_formula_against_independent_loop(mode):
    x = torch.tensor([[2.,0.], [1.,1.], [3.,1.], [-2.,1.], [0.,3.], [2.,2.]])
    selected, protected = torch.tensor([0, 1, 4]), torch.tensor([0])
    r = torch.tensor([0., .1, .3, -.2, -.1, .2])
    v = torch.tensor([1.,2.,3.,4.,5.,6.])
    out, trace = recover_visual_tokens(x, selected, protected, r, v, RecoverySpec(mode))
    expected = x[selected].clone()
    for j in (1, 2):
        sources = [i for i,t in zip(trace["recovery_dropped_indices"], trace["recovery_target_indices"]) if t == int(selected[j])]
        if not sources:
            continue
        if mode == "neighbor_mean":
            expected[j] = torch.stack([x[selected[j]], *[x[i] for i in sources]]).mean(0)
        else:
            residual = torch.zeros(2)
            for i in sources:
                cos = torch.nn.functional.cosine_similarity(x[i][None], x[selected[j]][None])[0]
                gate = torch.sigmoid((r[i]-r.mean())/v.std(unbiased=False))
                residual += cos.clamp(0,1).square()*gate*(x[i]-x[selected[j]])
            expected[j] += .25*residual/len(sources)
    torch.testing.assert_close(out, expected)


def test_nonpositive_similarity_rejected_and_zero_blend_exact_identity():
    x = torch.tensor([[1.,0.], [0.,1.], [0.,-1.]])
    s, p, r = torch.tensor([0,1]), torch.tensor([0]), torch.zeros(3)
    out, trace = recover_visual_tokens(x, s, p, r, r, RecoverySpec())
    assert torch.equal(out, x[s]) and trace["recovery_effective_sources"] == 0
    assert trace["recovery_nonpositive_affinity_sources"] == 1
    x = torch.randn(20,8)
    out, trace = recover_visual_tokens(x, s, p, torch.randn(20), torch.randn(20), RecoverySpec(maximum_blend=0))
    assert torch.equal(out, x[s]) and trace["recovery_changed_tokens"] == 0


@pytest.mark.parametrize("all_selected", [False, True])
def test_no_context_or_no_dropped_is_identity(all_selected):
    x = torch.randn(7,8)
    s = torch.arange(7 if all_selected else 3)
    out, trace = recover_visual_tokens(x, s, s, torch.zeros(7), torch.zeros(7), RecoverySpec())
    assert torch.equal(out, x[s]) and not trace["recovery_applied"]


def test_zero_features_ties_and_query_changes_only_residual_weights():
    x = torch.tensor([[1.,0.], [1.,0.], [1.,1.], [1.,2.], [1.,3.]])
    s, p = torch.tensor([0,1]), torch.tensor([],dtype=torch.long)
    a, ta = recover_visual_tokens(x, s, p, torch.arange(5).float(), torch.arange(5).float(), RecoverySpec())
    b, tb = recover_visual_tokens(x, s, p, -torch.arange(5).float(), torch.arange(5).float(), RecoverySpec())
    assert ta["recovery_target_indices"] == tb["recovery_target_indices"] == [0,0,0]
    assert not torch.equal(a,b)
    out, _ = recover_visual_tokens(torch.zeros(5,2),s,p,torch.zeros(5),torch.zeros(5),RecoverySpec())
    assert out.isfinite().all()


@pytest.mark.parametrize("bad", ["duplicate", "order", "outside", "protected", "score_shape", "nan", "integer"])
def test_invalid_inputs_fail_explicitly(bad):
    x, s, p, r, v = torch.randn(9,4), torch.tensor([1,3,5]), torch.tensor([1]), torch.zeros(9), torch.zeros(9)
    if bad == "duplicate": s = torch.tensor([1,1,3])
    if bad == "order": s = torch.tensor([3,1,5])
    if bad == "outside": s = torch.tensor([1,3,9])
    if bad == "protected": p = torch.tensor([2])
    if bad == "score_shape": r = r[:-1]
    if bad == "nan": x[0,0] = float("nan")
    if bad == "integer": x = x.long()
    with pytest.raises(ValueError):
        recover_visual_tokens(x,s,p,r,v,RecoverySpec())


@pytest.mark.parametrize("kwargs", [{"mode":"unknown"}, {"maximum_blend":True}, {"maximum_blend":-1}, {"maximum_blend":float("nan")}])
def test_bad_configuration(kwargs):
    with pytest.raises(ValueError): RecoverySpec(**kwargs)


@pytest.mark.parametrize("mode", ["neighbor_mean", "query_gated_residual"])
def test_wrapper_preserves_selector_indices_parameters_and_changes_context(mode):
    from mllm_comp.analysis.common.topk_training_modes import parameter_digest
    base = WordVisualTopK(WordVisualTopKSpec(12,width=8,target_count=4,match_width=4))
    spec = ProtectedTopKSpec(minimum_anchor=1)
    control = ProtectedWordSelector(base,spec)
    recover = RecoveredProtectedSelector(base,spec,RecoverySpec(mode))
    before = parameter_digest(base)
    x,q = torch.randn(21,12),torch.randn(3,12)
    old,new = control(x,question=q),recover(x,question=q)
    assert torch.equal(old["indices"],new["indices"])
    assert old["word_diagnostics"]["protected_indices"] == new["word_diagnostics"]["protected_indices"]
    assert new["word_diagnostics"]["recovery_changed_tokens"] > 0
    assert before == parameter_digest(base) and not any(p.requires_grad for p in base.parameters())


@pytest.mark.parametrize("mode", ["neighbor_mean", "query_gated_residual"])
def test_native_forward_generate_injection_labels_and_position_integrity(mode):
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.recovered_topk import QwenRecoveredTopKComparison
    bundle,state,_,_ = tiny_state()
    model,prompt = bundle["model"],state["prompt_inputs"]
    before = copy.deepcopy(prompt)
    with torch.no_grad(): dense = model(**prompt,use_cache=False).logits
    identity = QwenRecoveredTopKComparison(model,{"target_count":128,"recovery":{"mode":mode}}).install()
    try:
        with torch.no_grad(): torch.testing.assert_close(model(**prompt,use_cache=False).logits,dense,rtol=1e-5,atol=1e-6)
    finally: identity.uninstall()
    session = QwenRecoveredTopKComparison(model,{"target_count":3,"protection":{"minimum_anchor":1},
        "recovery":{"mode":mode}}).install()
    captured = []
    compact = session.compact_selection
    def capture(call,hidden,local,stats):
        visual = session.incoming.get().vision[0].nonzero(as_tuple=True)[0]
        value = compact(call,hidden,local,stats)
        kept_visual = session.incoming.get().vision[0]
        assert torch.equal(value[0,kept_visual],hidden[0,visual[local]])
        captured.append(value.detach().clone())
        return value
    session.compact_selection = capture
    try:
        with torch.no_grad():
            model.generate(**prompt,max_new_tokens=3,min_new_tokens=3,do_sample=False,pad_token_id=0)
        trace = session.runtime_metadata()["selection"]
        assert captured and trace["recovered_features_injected"] and trace["decode_position_verified"]
        assert trace["observed_visual_tokens_per_decoder_layer"] == [3]*4
        assert trace["recovery_protected_unchanged"] and trace["recovery_changed_tokens"] > 0
        labels = torch.full_like(prompt["input_ids"],-100); labels[:,-1] = 3
        with torch.no_grad(): assert model(**prompt,labels=labels,use_cache=False).loss.isfinite()
    finally: session.uninstall()
    assert all(torch.equal(before[k],prompt[k]) for k in before)


def test_zero_recovery_native_logits_equal_protected_logits():
    from tests.unit.test_projector_comparison_evaluation import tiny_state
    from mllm_comp.compressors.comparison.recovered_topk import QwenRecoveredTopKComparison
    from mllm_comp.compressors.comparison.protected_topk import QwenProtectedTopKComparison
    bundle,state,_,_ = tiny_state()
    model,prompt = bundle["model"],state["prompt_inputs"]
    cfg = {"target_count":3,"protection":{"minimum_anchor":1}}
    control = QwenProtectedTopKComparison(model,cfg).install()
    try:
        with torch.no_grad(): expected = model(**prompt,use_cache=False).logits
    finally: control.uninstall()
    session = QwenRecoveredTopKComparison(model,{**cfg,"recovery":{"maximum_blend":0}}).install()
    try:
        with torch.no_grad(): actual = model(**prompt,use_cache=False).logits
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    finally: session.uninstall()


def test_recovery_binding_registered_with_merge_contract():
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    registry = builtin_multimodal_compression_contracts()
    binding = next(b for b in registry.bindings() if b.name == "recovered_visual_topk.image.qwen25_vl")
    assert binding.legalizer == "qwen25_vl.original_sparse_mrope_compact_cache"


def test_runner_smoke_prerequisite_resume_and_no_historical_full_reruns(tmp_path,monkeypatch):
    from scripts.train.qwen25_vl import run_visual_token_recovery as runner
    from mllm_comp.core.serialization import atomic_write_json
    protocol = {"source":{"test":"snapshot"}}
    atomic_write_json(tmp_path/"manifest.json",{"completed_cells":[]})
    monkeypatch.setattr(runner,"validate",lambda root:protocol)
    monkeypatch.setattr(runner,"available_gpus",lambda requested:requested)
    monkeypatch.setattr(runner,"config",lambda root,p,arm,phase,task,run:
        {"arm":arm,"phase":phase,"task":task})
    calls = []
    def execute(command,log,gpus,env):
        import yaml
        cfg = yaml.safe_load(__import__("pathlib").Path(command[-1]).read_text())
        if cfg["phase"] == "full":
            assert (tmp_path/"smoke_complete.json").exists()
            assert cfg["arm"] in runner.ARMS
        calls.append((cfg["arm"],cfg["phase"],cfg["task"]))
    monkeypatch.setattr(runner.word.context.native,"execute",execute)
    monkeypatch.setattr(runner,"check",lambda root,p,run,arm,phase,task:
        {"requests":8 if phase == "smoke" else runner.word.context.COUNTS[task]})
    monkeypatch.setattr(runner,"report",lambda root,p:None)
    runner.pipeline(tmp_path,1)
    assert len(calls) == 18
    assert runner.read(tmp_path/"manifest.json")["status"] == "complete"
    runner.pipeline(tmp_path,1)
    assert len(calls) == 18
