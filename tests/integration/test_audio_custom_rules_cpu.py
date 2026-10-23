"""Formal audio rules on real Omni encoder/projector/decoder and prepare paths."""
import copy
from dataclasses import replace
import pytest
from tests.fixtures.researcher_audio_rules import NAMES, stage_for, specifications

TEMPORAL = [name for name in NAMES if name not in {
    "audio_pyramid_attention_pruning", "headrouter_dynamic_audio_pruning",
    "fastadasp_weighted_token_merging", "structured_llm_layer_pruning"}]
LLM_ONLY = ["audio_pyramid_attention_pruning", "headrouter_dynamic_audio_pruning", "fastadasp_weighted_token_merging"]


def _fixture(algorithm, family):
    torch = pytest.importorskip("torch")
    from tests.fixtures.qwen25_omni_audio import _tiny_model, _hta_audio_state, _create_controller
    torch.manual_seed(531)
    model = _tiny_model(torch, audio_encoder_layers=4).train()
    stage = stage_for(algorithm, family)
    raw = _hta_audio_state(torch)
    # Scored LLM methods use the actual instruction/response role map.
    raw.modality_metadata["semantic_token_labels_by_sample"] = [
        ["ins_special", "ins_special"] + ["audio"] * 8 + ["ins_special", "instruction", "response"]]
    return torch, model, stage, raw, _create_controller


def _observe(monkeypatch, algorithm):
    import tests.fixtures.researcher_audio_rules as rules
    import mllm_comp.compressors.rules as loading
    calls = {name: [] for name in NAMES[algorithm]}
    for name, function_name in NAMES[algorithm].items():
        original = getattr(rules, function_name)
        def invoke(request, _original=original, _name=name, **kwargs):
            calls[_name].append((tuple(request["values"].shape) if "values" in request else None,
                                  bool(request.get("values") is not None and request["values"].requires_grad)))
            return _original(request, **kwargs)
        monkeypatch.setattr(rules, function_name, invoke)
    loads = []
    original = loading.load_rule_specs

    def load(specs):
        loads.append(specs)
        return original(specs)

    monkeypatch.setattr(loading, "load_rule_specs", load)
    return calls, loads


def _finite(torch, gradient):
    assert gradient is not None and torch.isfinite(gradient).all() and gradient.count_nonzero()


def _metadata(controller, algorithm):
    metadata = controller.runtime_metadata()
    assert metadata["rules"] == specifications(algorithm)
    assert metadata["rules_changed"] is True
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["development_runtime"] is True


@pytest.mark.parametrize("algorithm,family", [(a, f) for a in TEMPORAL for f in ("encoder", "projector", "llm")]
    + [(a, "llm") for a in LLM_ONLY])
def test_all_audio_rules_execute_through_existing_native_families(algorithm, family, monkeypatch):
    torch, model, stage, raw, create = _fixture(algorithm, family)
    calls, loads = _observe(monkeypatch, algorithm)
    controller = create(model, stage)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    if algorithm == "fastslm_hta_audio":
        optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
        assert all(id(p) in optimized for p in controller.strategy_runtime_module.parameters())
    controller.install()
    try:
        state, plan = controller.prepare_state(raw)
        with controller.request(plan):
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        assert output.logits.shape == (1, 9, 128)
        counts = {name: len(rows) for name, rows in calls.items()}
        assert all(counts.values()), counts
        output.loss.backward()
        assert {name: len(rows) for name, rows in calls.items()} == counts
        assert loads == [specifications(algorithm)]
        _finite(torch, raw.audio_values.grad)
        _finite(torch, model.thinker.audio_tower.proj.weight.grad)
        _finite(torch, model.thinker.model.layers[3].mlp.gate_proj.weight.grad)
        if algorithm == "fastslm_hta_audio":
            assert any(p.grad is not None and p.grad.count_nonzero() for p in controller.strategy_runtime_module.parameters())
        _metadata(controller, algorithm)
    finally:
        controller.uninstall()


@pytest.mark.parametrize("algorithm,family", [
    ("adjacent_token_merging_audio", "encoder"),
    ("uniform_sampling_audio", "encoder"),
    ("uniform_average_pooling_audio", "llm"),
    ("unsupervised_segmentation_audio", "encoder"),
    ("similarity_attention_merge_prune_audio", "llm"),
    ("fastadasp_weighted_token_merging", "llm"),
    ("fastslm_hta_audio", "encoder"),
])
def test_audio_custom_mid_checkpoint_replays_fixed_plan_after_request(algorithm, family, monkeypatch):
    torch, original, stage, raw, create = _fixture(algorithm, family)
    calls, loads = _observe(monkeypatch, algorithm)
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        torch.manual_seed(872)  # Same owned HTA initialization for each checkpoint mode.
        controller = create(model, stage)
        controller.install()
        features = raw.audio_values.detach().clone().requires_grad_()
        state_input = replace(raw, audio_values=features,
            native_model_inputs={**raw.native_model_inputs, "input_features": features})
        try:
            state, plan = controller.prepare_state(state_input)
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            decisions = {name: len(rows) for name, rows in calls.items() if name not in {"aggregator", "transform"}}
            output.loss.backward()
            assert {name: len(rows) for name, rows in calls.items() if name not in {"aggregator", "transform"}} == decisions
            _finite(torch, features.grad)
            results.append((output.logits.detach(), features.grad, model.thinker.audio_tower.proj.weight.grad))
        finally:
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)
    assert len(loads) == 3


@pytest.mark.parametrize("algorithm", ["audio_pyramid_attention_pruning", "headrouter_dynamic_audio_pruning", "fastadasp_weighted_token_merging"])
def test_custom_llm_rules_repeat_prefill_decode_without_reselection(algorithm, monkeypatch):
    from mllm_comp.core.vocabulary import ExecutionPhase
    torch, model, stage, raw, create = _fixture(algorithm, "llm")
    model.eval()
    stage.execution_phase = ExecutionPhase.GENERATE
    raw = replace(raw, labels=None, native_model_inputs={key: value for key, value in raw.native_model_inputs.items() if key != "labels"})
    calls, loads = _observe(monkeypatch, algorithm)
    controller = create(model, stage)
    controller.install()
    try:
        for _ in range(2):
            state, plan = controller.prepare_state(raw)
            with controller.request(plan), torch.no_grad():
                output = model.thinker(**state.native_model_inputs, use_cache=True)
                assert [output.past_key_values.get_seq_length(i) for i in range(4)] == [13, 13, 9, 9]
                counts = {name: len(rows) for name, rows in calls.items()}
                decoded = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones(1, 14, dtype=torch.long),
                    past_key_values=output.past_key_values, use_cache=True)
                assert [decoded.past_key_values.get_seq_length(i) for i in range(4)] == [14, 14, 10, 10]
                assert {name: len(rows) for name, rows in calls.items()} == counts
        assert len(loads) == 1
    finally:
        controller.uninstall()


def test_spade_configured_prepare_decision_removes_chosen_native_layers_and_trains():
    torch = pytest.importorskip("torch")
    from tests.fixtures.qwen25_omni_audio import _tiny_model, _state
    from mllm_comp.backbones.qwen25_omni.v2_compression import apply_qwen25_omni_operation_compression
    model = _tiny_model(torch).train()
    original = list(model.thinker.model.layers)
    stage = stage_for("structured_llm_layer_pruning")
    plan, result = apply_qwen25_omni_operation_compression(model, stage)
    assert [a.target_indices for a in plan.operation_actions] == [[0], [1]]
    assert list(model.thinker.model.layers) == original[2:]
    assert plan.this_stage_metadata.stats["rules"] == specifications(stage.algorithm)
    assert result.metadata["rules"] == specifications(stage.algorithm)
    assert model._mllm_comp_operation_manifests[-1]["rules"] == specifications(stage.algorithm)
    raw = _state(torch, training=True)
    output = model.thinker(input_ids=raw.input_ids, attention_mask=raw.attention_mask, labels=raw.labels, use_cache=False)
    output.loss.backward()
    _finite(torch, model.thinker.model.layers[0].mlp.gate_proj.weight.grad)


def test_unseg_projector_replays_first_custom_natural_plan_without_regrouping(monkeypatch):
    import tests.fixtures.researcher_audio_rules as functions
    torch, model, stage, raw, create = _fixture("unsupervised_segmentation_audio", "projector")
    calls = []
    def actual_groups(request):
        calls.append(request["values"].detach().clone())
        # A natural three-group plan differs from the default peaks and fixed-rate budget.
        length = len(request["values"])
        return [tuple(range(length - 2)), (length - 2,), (length - 1,)]
    monkeypatch.setattr(functions, "chronological_groups", actual_groups)
    controller = create(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(raw)
        assert len(calls) == 1
        assert plan.token_actions[0].metadata["keep_count"] == 3
        # RNG changes after planning may change values, but never the saved grouping/budget.
        torch.randn(17)
        with controller.request(plan):
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        assert output.logits.shape == (1, 8, 128)
        output.loss.backward()
        assert len(calls) == 1
        _finite(torch, raw.audio_values.grad)
    finally:
        controller.uninstall()


def test_segmentwise_saturated_budget_mid_checkpoint_retains_identity_graph():
    torch, original, stage, raw, create = _fixture("segmentwise_topk_audio", "llm")
    stage.operation.pop("rules")
    stage.schedule["keep_ratio"] = 1.0
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = create(model, stage)
        controller.install()
        features = raw.audio_values.detach().clone().requires_grad_()
        state_input = replace(raw, audio_values=features,
            native_model_inputs={**raw.native_model_inputs, "input_features": features})
        try:
            state, plan = controller.prepare_state(state_input)
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            assert output.logits.shape == (1, 13, 128)
            output.loss.backward()
            _finite(torch, features.grad)
            _finite(torch, model.thinker.model.layers[1].mlp.gate_proj.weight.grad)
            results.append((output.logits.detach(), features.grad,
                model.thinker.model.layers[1].mlp.gate_proj.weight.grad))
        finally:
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)
