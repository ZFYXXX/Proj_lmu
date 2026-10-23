"""Rule tests reuse native kernel payloads and check values, lineage and gradients."""

import importlib

import pytest

torch = pytest.importorskip("torch")


def _payload(name):
    return importlib.import_module(f"tests.unit.test_{name}_video_kernel")._payload(requires_grad=True)


def _group_mean(request):
    return request["values"][list(request["group"])].mean(0)


def _vision_rules(calls):
    def scorer(r):
        calls.add("scorer")
        return -r["attention_scores"]
    def dominant(r):
        calls.add("dominant_selector")
        return torch.argsort(r["scores"], descending=True, stable=True)[:r["keep_count"]]
    def context(r):
        calls.add("context_selector")
        return r["candidates"][:r["keep_count"]]
    def grouper(r):
        calls.add("grouper")
        targets = r["targets"]
        remaining = [i for i in r["candidates"] if i not in targets]
        return [[target, *remaining[j::len(targets)]] for j, target in enumerate(targets)]
    def aggregator(r):
        calls.add("aggregator")
        return _group_mean(r)
    return dict(scorer=scorer, dominant_selector=dominant, context_selector=context,
                grouper=grouper, aggregator=aggregator)


@pytest.mark.parametrize("name", ("visionzip", "dytok"))
def test_visionzip_and_dytok_rules_change_actual_groups_values_and_branch_replay(name):
    from mllm_comp.compressors.video import merge_visionzip_source_groups
    module = importlib.import_module(f"mllm_comp.compressors.video.{name}")
    payload = _payload(name)
    calls = set()
    rules = _vision_rules(calls)
    if name == "dytok":
        def frame_scorer(r):
            calls.add("frame_scorer")
            return r["default_scores"].new_tensor([3., 1.])
        def allocate(r):
            calls.add("budget_allocator")
            return tuple(reversed(r["default_budgets"]))
        rules.update(frame_scorer=frame_scorer, budget_allocator=allocate)
    result = getattr(module, f"{name}_video")(payload, rules=rules)
    expected = torch.stack([payload.features[list(group)].mean(0) for group in result.source_groups[0]])
    torch.testing.assert_close(result.features, expected)
    branch = payload.features.detach().clone().mul(2).requires_grad_()
    replay = merge_visionzip_source_groups(branch, result, rules=rules)
    torch.testing.assert_close(replay, expected.detach() * 2)
    (result.features.sum() + replay.sum()).backward()
    assert payload.features.grad is not None and branch.grad is not None
    assert calls == set(rules)
    if name == "dytok":
        assert result.metadata["total_budgets_by_video"] == ((98, 196),)


@pytest.mark.parametrize("name", ("visionzip", "dytok"))
def test_video_visionzip_replay_preserves_frame_context_and_local_targets(name):
    from mllm_comp.compressors.video import merge_visionzip_source_groups
    module = importlib.import_module(f"mllm_comp.compressors.video.{name}")
    payload = _payload(name)
    calls = []
    def aggregate(request):
        values = request["values"]
        calls.append((len(values), request["target"]))
        return (values[list(request["group"])].square().mean(0)
                + values.mean(0) * .5 + request["target"] / len(values))
    rules = {"aggregator": aggregate}
    result = getattr(module, f"{name}_video")(payload, rules=rules)
    first_calls = tuple(calls)
    branch = (payload.features.detach() + .3).requires_grad_()
    calls.clear()
    replay = merge_visionzip_source_groups(branch, result, rules=rules)
    assert tuple(calls) == first_calls
    def expected(source):
        rows, source_start, result_start = [], 0, 0
        targets = result.metadata["merge_target_indices_by_video"][0]
        for record in result.metadata["frame_records_by_video"][0]:
            source_stop = source_start + record["source_count"]
            frame = source[source_start:source_stop]
            result_stop = result_start + len(record["representative_indices"])
            for index in range(result_start, result_stop):
                group = result.source_groups[0][index]
                rows.append(source[list(group)].square().mean(0) + frame.mean(0) * .5
                            + (targets[index] - source_start) / len(frame))
            source_start, result_start = source_stop, result_stop
        return torch.stack(rows)
    torch.testing.assert_close(result.features, expected(payload.features))
    reference = branch.detach().clone().requires_grad_()
    wanted = expected(reference)
    torch.testing.assert_close(replay, wanted)
    replay.square().sum().backward()
    wanted.square().sum().backward()
    torch.testing.assert_close(branch.grad, reference.grad)


def test_vqtoken_transform_groups_and_aggregation_replay_actual_values():
    from mllm_comp.compressors.video import vqtoken_video, merge_vqtoken_source_groups
    payload = _payload("vqtoken")
    rules = {"transform": lambda r: r["values"] * 3,
             "grouper": lambda r: [(i, i + 32) for i in range(32)],
             "aggregator": _group_mean}
    result = vqtoken_video(payload, rules=rules)
    expected = (payload.features[:32] + payload.features[32:]) * 1.5
    torch.testing.assert_close(result.features, expected)
    branch = payload.features.detach().clone().add(1).requires_grad_()
    torch.testing.assert_close(merge_vqtoken_source_groups(branch, result, rules=rules), expected.detach() + 3)
    result.features.sum().backward()
    torch.testing.assert_close(payload.features.grad, torch.full_like(payload.features, 1.5))


def test_fastvid_rules_keep_overlapping_lineage_separate_from_layout():
    from mllm_comp.compressors.video import fastvid_video, replay_fastvid_value_lineage
    from tests.unit.test_fastvid_video_kernel import _payload
    payload = _payload(frame_count=2, requires_grad=True)
    calls = set()
    def record(name, function):
        def call(r):
            calls.add(name)
            return function(r)
        return call
    rules = {
        "frame_grouper": record("frame_grouper", lambda r: [(0, 1)]),
        "saliency_scorer": record("saliency_scorer", lambda r: -r["default_scores"]),
        "saliency_selector": record("saliency_selector", lambda r: torch.argsort(r["scores"], descending=True)[:r["keep_count"]]),
        "context_allocator": record("context_allocator", lambda r: r["default_allocations"]),
        "center_scorer": record("center_scorer", lambda r: torch.arange(r["values"].shape[0], device=r["values"].device).float()),
        "center_selector": record("center_selector", lambda r: torch.argsort(r["scores"], descending=True)[:r["keep_count"]]),
        "grouper": record("grouper", lambda r: [r["candidates"][i::len(r["targets"])] for i in range(len(r["targets"]))]),
        "aggregator": record("aggregator", lambda r: r["values"][r["target"]] * 2 + r["values"][list(r["assigned_indices"])].mean(0)),
    }
    result = fastvid_video(payload, rules=rules)
    expected = torch.stack([payload.features[r["center_index"]] if r["kind"] == "salient" else
        payload.features[r["center_index"]] * 2 + payload.features[list(r["assigned_indices"])].mean(0)
        for r in result.metadata["value_records_by_video"][0]])
    torch.testing.assert_close(result.features, expected)
    torch.testing.assert_close(replay_fastvid_value_lineage(payload.features, result, rules=rules), expected)
    result.features.sum().backward()
    assert payload.features.grad.abs().sum() > 0
    assert calls == set(rules)


def test_metok_custom_events_pool_and_three_pruning_decisions():
    from mllm_comp.compressors.video import metok_video, replay_metok_projector_layout, plan_metok_pruning_stage
    payload = _payload("metok")
    stages = []
    def choose(r):
        stages.append(r["stage_index"])
        return tuple(range(len(r["scores"]) - r["keep_count"], len(r["scores"])))
    rules = {"frame_scorer": lambda r: -r["default_scores"],
        "frame_grouper": lambda r: [tuple(range(8))],
        "event_selector": lambda r: tuple(range(r["keep_count"])),
        "keyframe_selector": lambda r: tuple(range(r["keep_count"])),
        "pool_transform": lambda r: r["values"].mean((0, 1)).expand(r["output_shape"][0] * r["output_shape"][1], -1) * 2,
        "prune_selector": choose}
    result = metok_video(payload, rules=rules)
    assert result.metadata["metok_segments"] == (((0, 8),),)
    for record in result.metadata["metok_frames"][0]:
        start, stop = record["result_span"]
        expected = payload.features.reshape(8, 64, -1)[record["frame_index"]].mean(0) * 2
        torch.testing.assert_close(result.features[start:stop], expected.expand(stop-start, -1))
    torch.testing.assert_close(replay_metok_projector_layout(payload.features, result, rules=rules), result.features)
    count = result.result_count
    for stage in range(3):
        decision = plan_metok_pruning_stage(sequence_length=count + 2,
            visual_sequence_indices=range(1, count+1), visual_scores=torch.arange(count).float(),
            projector_result=result, event_lengths=(count,), important_event_indices=(0,),
            boundary_layer=(3,10,19)[stage], stage_index=stage, rules=rules)
        expected_count = 0 if stage == 2 else __import__('math').ceil(count * .76)
        assert decision.kept_visual_indices == tuple(range(count-expected_count, count))
    assert stages == [0,1,2]
    result.features.sum().backward()
    assert payload.features.grad.abs().sum() > 0


def test_prunevid_custom_windows_static_groups_pool_and_vtp():
    from mllm_comp.compressors.video import prunevid_video, merge_prunevid_source_groups, prunevid_vtp_select, prunevid_windows
    payload = _payload("prunevid")
    kinds = set()
    def group(r):
        kinds.add(r["kind"])
        return [tuple(range(i, min(i+2, len(r["values"])))) for i in range(0, len(r["values"]), 2)]
    rules = {"frame_grouper": lambda r: [(0,1,2,3)],
        "static_selector": lambda r: tuple(range(16)), "token_grouper": group,
        "aggregator": lambda r: _group_mean(r) * 3,
        "vtp_selector": lambda r: r["candidates"][-r["keep_count"]:] if r["keep_count"] else ()}
    result = prunevid_video(payload, rules=rules)
    expected = torch.stack([payload.features[list(g)].mean(0) * 3 for g in result.source_groups[0]])
    torch.testing.assert_close(result.features, expected)
    torch.testing.assert_close(merge_prunevid_source_groups(payload.features, result, rules=rules), expected)
    window, = prunevid_windows(result)
    candidates = (window.static_result_indices, *window.dynamic_result_indices_by_frame)
    selected = prunevid_vtp_select(torch.arange(result.result_count).float(), result, rules=rules)
    assert selected == tuple(sorted(i for c in candidates for i in c[-int(len(c)*.4):] if int(len(c)*.4)))
    assert kinds == {"static", "dynamic"}
    result.features.sum().backward()
    assert payload.features.grad.abs().sum() > 0


def test_dycoke_custom_ttm_and_decode_decision_preserve_nonvisual_view():
    from mllm_comp.compressors.video import dycoke_video, plan_dycoke_kv_view
    payload = _payload("dycoke")
    result = dycoke_video(payload, rules={
        "ttm_scorer": lambda r: torch.arange(len(r["values"])).float(),
        "ttm_selector": lambda r: tuple(range(r["keep_count"]))})
    expected_indices = (0,1,2,3,4,8,12)
    torch.testing.assert_close(result.features, payload.features[list(expected_indices)])
    rules = {"decode_decision": lambda r: r["previous_scores"] is not None,
             "decode_selector": lambda r: r["visual_positions"][-r["keep_count"]:]}
    first = plan_dycoke_kv_view(torch.arange(4).float(), visual_positions=(1,2,4,5),
        sequence_length=7, drop_ratio=.5, rules=rules)
    second = plan_dycoke_kv_view(torch.arange(4).float().flip(0), visual_positions=(1,2,4,5),
        sequence_length=8, drop_ratio=.5, previous_attention_scores=torch.arange(4).float(),
        previous_kept_visual_positions=first.kept_visual_positions, rules=rules)
    assert first.kept_visual_positions == second.kept_visual_positions == (4,5)
    assert second.reused_visual_selection
    assert second.view_indices == (0,3,4,5,6,7)
    result.features.sum().backward()
    assert payload.features.grad.abs().sum() > 0


def test_longvu_frame_rules_and_real_fusion_transform_keep_module_gradients():
    from mllm_comp.compressors.video import longvu_video
    from tests.integration.test_video_longvu_positions_cpu import _fusion
    payload = _payload("longvu")
    module = _fusion(raw_hidden_size=3, output_hidden_size=4)
    calls = set()
    def frame_score(r):
        calls.add("frame_scorer")
        return r["default_scores"] * 0
    def frame_select(r):
        calls.add("frame_selector")
        return tuple(range(len(r["values"])))
    def transform(r):
        calls.add("fusion_transform")
        assert r["module"] is module
        return r["default_transform"]() * 2
    baseline = longvu_video(payload, fusion_module=module,
        rules={"frame_scorer": frame_score, "frame_selector": frame_select})
    actual = longvu_video(payload, fusion_module=module,
        rules={"frame_scorer": frame_score, "frame_selector": frame_select, "fusion_transform": transform})
    torch.testing.assert_close(actual.features, baseline.features * 2)
    actual.features.square().mean().backward()
    assert payload.raw_features.grad is not None
    assert module.vision_query.grad is not None
    assert calls == {"frame_scorer", "frame_selector", "fusion_transform"}


@pytest.mark.parametrize("position", ("before", "after"))
def test_llamavid_real_query_scores_two_pools_and_branch_replay(position):
    from dataclasses import replace
    from mllm_comp.compressors.video import llamavid_video, replay_llamavid_value_lineage
    from tests.integration.test_video_llamavid_positions_cpu import _qformer
    payload = _payload("llamavid")
    payload = replace(payload, signals={**payload.signals,
        "llamavid.bert_tokens_by_video": ({"input_ids": torch.tensor([[1,2]]),
                                         "attention_mask": torch.ones(1,2,dtype=torch.long)},)})
    module = _qformer(raw_hidden=3, output_hidden=4)
    calls, captured = set(), []
    def query(r):
        calls.add("query_transform")
        return r["values"] * 2
    def scorer(r):
        calls.add("context_scorer")
        return r["default_scores"] * .5
    def context(r):
        calls.add("context_aggregator")
        return (r["values"] * r["weights"][:, None]).sum(0) * 2
    def content(r):
        calls.add("content_aggregator")
        return r["values"].mean(0) * 3
    rules = dict(query_transform=query, context_scorer=scorer,
                 context_aggregator=context, content_aggregator=content)
    handle = module.vlm_att_projector.register_forward_pre_hook(lambda _m,args: captured.append(args[0]))
    try:
        result = llamavid_video(payload, qformer_module=module, pool_position=position, rules=rules)
    finally:
        handle.remove()
    raw = payload.raw_features.reshape(2,16,3)
    q = module.vlm_att_projector(captured[0])
    keys = module.vlm_att_key_projector(raw)
    weights = (q @ keys.transpose(-1,-2) / (3**.5) * .5).softmax(-1).mean(1)
    if position == "before":
        expected_context = module.vlm_att_val_projector(((raw * weights[...,None]).sum(1) * 2)[:,None,:])
        expected_content = module.content_projector((raw.mean(1) * 3)[:,None,:])
    else:
        expected_context = ((module.vlm_att_val_projector(raw) * weights[...,None]).sum(1) * 2)[:,None,:]
        expected_content = (module.content_projector(raw).mean(1) * 3)[:,None,:]
    torch.testing.assert_close(result.features, torch.cat((expected_context,expected_content),1).reshape(-1,4))
    branch = payload.features.detach().clone().requires_grad_()
    replay = replay_llamavid_value_lineage(branch, result, rules=rules)
    expected = []
    for record in result.metadata["value_records_by_video"][0]:
        values = branch[list(record["source_indices"])]
        expected.append((values * values.new_tensor(record["weights"])[:,None]).sum(0) * 2
            if record["kind"] == "context" else values.mean(0) * 3)
    torch.testing.assert_close(replay, torch.stack(expected))
    (result.features.square().mean() + replay.square().mean()).backward()
    assert payload.raw_features.grad is not None and branch.grad is not None
    assert module.query_tokens.grad is not None
    assert calls == set(rules)


@pytest.mark.parametrize("bad", ([0,0], [0,99], [0,1.5], [[0],1]))
def test_rule_selector_rejects_invalid_indices(bad):
    from mllm_comp.compressors.video._rules import select
    from mllm_comp.core.errors import BackendIntegrationError
    with pytest.raises(BackendIntegrationError, match="indices"):
        select(lambda _:bad, {}, count=2, candidates=range(4), name="selector")


def test_rule_groups_and_values_reject_wrong_partition_shape_and_detach():
    from mllm_comp.compressors.video import _rules
    from mllm_comp.core.errors import BackendIntegrationError
    with pytest.raises(BackendIntegrationError):
        _rules.groups(lambda _:[(0,1),(1,2)], {}, count=2, candidates=range(4), name="grouper")
    source = torch.randn(4,3,requires_grad=True)
    for bad in (lambda _:source, lambda _:source.mean(0).detach()):
        with pytest.raises(BackendIntegrationError):
            _rules.value(bad, {}, reference=source, shape=(3,), name="aggregator")


def test_prunevid_custom_token_groups_keep_budget_and_identity_threshold():
    from mllm_comp.compressors.video.prunevid import _spatial_cluster_groups
    from mllm_comp.compressors.video.config import resolve_canonical_video_config
    from mllm_comp.core.errors import BackendIntegrationError
    config = resolve_canonical_video_config("prunevid", None)
    calls = []
    def group(request):
        calls.append(request["group_count"])
        return [tuple(range(len(request["values"])))]
    identity = _spatial_cluster_groups(torch.randn(14,3), config=config, torch=torch,
        rules={"token_grouper":group})
    assert identity == tuple((i,) for i in range(14)) and not calls
    with pytest.raises(BackendIntegrationError, match="8 nonempty groups"):
        _spatial_cluster_groups(torch.randn(16,3), config=config, torch=torch,
            rules={"token_grouper":group})
    assert calls == [8]
