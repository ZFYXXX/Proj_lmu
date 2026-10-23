"""Small real-Transformers integration for Qwen LLM-level executors."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.v2_compression import (
    apply_qwen25_omni_operation_compression,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.export.operation_export import export_operation_actions
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures.qwen25_omni_audio import (
    llm_stage, _operation_stage, _create_controller, _state, _tiny_model,
    _hta_layer_stage, _hta_audio_state,
)


@pytest.mark.parametrize(
    "target_site",
    [
        "llm_decoder_layer_output_audio_tokens",
        "llm_decoder_layer_post_attention_audio_tokens",
    ],
)
def test_real_transformers_prefill_decode_and_checkpoint_backward(target_site: str) -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch)
    thinker = model.thinker
    stage = llm_stage(phase="generate")
    stage.operation["target_site"] = target_site
    ffn_widths = []
    handle = thinker.model.layers[0].mlp.gate_proj.register_forward_pre_hook(
        lambda module, args: ffn_widths.append(int(args[0].shape[1]))
    )
    norm_widths = []
    norm_handle = thinker.model.layers[0].post_attention_layernorm.register_forward_pre_hook(
        lambda module, args: norm_widths.append(int(args[0].shape[1]))
    )
    controller = _create_controller(model, stage)
    controller.install()

    eval_state, eval_plan = controller.prepare_state(_state(torch, training=False))
    with controller.request(eval_plan):
        sequences = thinker.generate(
            input_ids=eval_state.input_ids,
            attention_mask=eval_state.attention_mask,
            min_new_tokens=3,
            max_new_tokens=3,
            do_sample=False,
            use_cache=True,
            pad_token_id=0,
        )
    assert sequences.shape == (1, 10)
    eval_report = controller.last_report
    assert eval_report["prefill_forward_calls"] == 1
    assert eval_report["decode_forward_calls"] == 2
    assert eval_report["decode_bypassed"] is True
    assert eval_report["selection_repeated_during_decode"] is False
    assert eval_report["original_audio_tokens"] == 4
    assert eval_report["kept_audio_tokens"] == 1
    assert ffn_widths[0] == (
        5 if target_site == "llm_decoder_layer_post_attention_audio_tokens" else 7
    )
    assert norm_widths == ffn_widths
    norm_handle.remove()
    handle.remove()

    controller.uninstall()
    thinker.rope_deltas = None
    thinker.gradient_checkpointing_enable()
    thinker.train()
    train_stage = llm_stage(phase="forward")
    train_stage.operation["target_site"] = target_site
    train_controller = _create_controller(model, train_stage)
    train_controller.install()
    train_state, train_plan = train_controller.prepare_state(
        _state(torch, training=True)
    )
    try:
        with train_controller.request(train_plan):
            output = thinker(
                input_ids=train_state.input_ids,
                attention_mask=train_state.attention_mask,
                labels=train_state.labels,
                use_cache=False,
            )
        assert output.loss is not None and output.loss.requires_grad
        assert torch.isfinite(output.loss.detach()).all()
        output.loss.backward()
    finally:
        train_controller.uninstall()
    embedding_gradient = thinker.model.embed_tokens.weight.grad
    assert embedding_gradient is not None
    assert torch.isfinite(embedding_gradient).all()
    assert int(torch.count_nonzero(embedding_gradient).item()) > 0
    train_report = train_controller.last_report
    assert train_report["prefill_forward_calls"] == 1
    assert train_report["decode_forward_calls"] == 0
    assert train_report["kept_audio_tokens"] == 1


@pytest.mark.parametrize("target_site, application, algorithm", [
    (site, application, algorithm)
    for site, application in [
        ("llm_decoder_layer_output_audio_tokens", "default"),
        ("llm_decoder_layer_output_audio_tokens", "initial"),
        ("llm_decoder_layer_output_audio_tokens", "later"),
        ("llm_decoder_layer_post_attention_audio_tokens", "default"),
        ("llm_decoder_layer_post_attention_audio_tokens", "later"),
    ]
    for algorithm in ("audio_pyramid_attention_pruning", "fastadasp_weighted_token_merging")
] + [("llm_decoder_layer_output_audio_tokens", "overlap", "audio_pyramid_attention_pruning")])
def test_position_preserves_checkpoint_gradients(target_site: str, algorithm: str, application: str) -> None:
    import copy

    torch = pytest.importorskip("torch")
    torch.manual_seed(17)
    original = _tiny_model(torch)
    stage = llm_stage(phase="forward")
    stage.plugin = stage.algorithm = algorithm
    stage.variant = f"{stage.algorithm}.audio"
    stage.binding = f"{stage.variant}.qwen25_omni"
    stage.operation["target_site"] = target_site
    if algorithm == "fastadasp_weighted_token_merging":
        stage.operation["type"] = "merge"
        stage.criterion = {}
        stage.schedule = {"merge_ratio": 0.5}
    stage.activation_scope = {"layers": [1], "lifecycle": "prefill_only"}
    if application == "initial":
        stage.activation_scope["layers"] = [0]
        stage.operation.update(layers=[0], layer_edge="before")
    elif application == "later":
        stage.operation["layers"] = [2]
    elif application == "overlap":
        stage.activation_scope["layers"] = [0, 1]
        stage.operation["layers"] = [1, 2]
    results = []
    for checkpointed in (False, True):
        model = copy.deepcopy(original)
        model.train()
        if checkpointed:
            model.thinker.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        controller = _create_controller(model, stage)
        score_inputs, source_norms = [], []
        original_score = controller.strategy.score_audio_tokens
        def score(**kwargs):
            score_inputs.append(kwargs["normalized_hidden_states"].detach().clone())
            return original_score(**kwargs)
        controller.strategy.score_audio_tokens = score
        handle = model.thinker.model.layers[1].input_layernorm.register_forward_hook(
            lambda module, args, output: source_norms.append(output.detach().clone())
        )
        controller.install()
        try:
            state, plan = controller.prepare_state(_state(torch, training=True))
            with controller.request(plan) as request:
                output = model.thinker(
                    input_ids=state.input_ids,
                    attention_mask=state.attention_mask,
                    labels=state.labels,
                    use_cache=False,
                )
                if application == "overlap":
                    indices = request.compression_positions[0][0]
                    expected = source_norms[0].index_select(1, indices[indices.ge(0)])
                    torch.testing.assert_close(score_inputs[1], expected, rtol=0, atol=0)
            output.loss.backward()
            assert controller.last_report["kept_audio_tokens"] == (1 if application == "overlap" else 2)
            results.append((
                output.loss.detach(),
                model.thinker.model.embed_tokens.weight.grad.detach().clone(),
                model.thinker.model.layers[1].mlp.gate_proj.weight.grad.detach().clone(),
            ))
        finally:
            handle.remove()
            controller.uninstall()
    for actual, expected in zip(results[1], results[0]):
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, expected)


def test_real_transformers_operation_rewrite_export_and_reload(
    tmp_path: Path,
) -> None:
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch)
    thinker = model.thinker
    source_layers = len(thinker.model.layers)

    _, apply_result = apply_qwen25_omni_operation_compression(
        model, _operation_stage()
    )
    assert not apply_result.violations
    assert len(thinker.model.layers) == source_layers - 2
    assert thinker.config.text_config.num_hidden_layers == source_layers - 2
    assert [
        int(layer.self_attn.layer_idx) for layer in thinker.model.layers
    ] == list(range(source_layers - 2))

    input_ids = torch.tensor([[1, 2, 3, 4]])
    attention_mask = torch.ones_like(input_ids)
    output = thinker(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=input_ids,
        use_cache=False,
    )
    assert output.loss is not None and output.loss.requires_grad
    output.loss.backward()
    assert thinker.model.embed_tokens.weight.grad is not None

    export_dir = tmp_path / "spade-tiny"
    manifest = export_operation_actions(
        model=thinker,
        output_dir=export_dir,
        apply_result=apply_result,
    )
    assert manifest["operation_compression"]["result_num_hidden_layers"] == 2

    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import (
        Qwen2_5OmniThinkerForConditionalGeneration,
    )

    reloaded = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        export_dir
    )
    assert len(reloaded.model.layers) == 2
    assert reloaded.config.text_config.num_hidden_layers == 2
    assert [int(layer.self_attn.layer_idx) for layer in reloaded.model.layers] == [
        0,
        1,
    ]
    generated = reloaded.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        min_new_tokens=2,
        max_new_tokens=2,
        do_sample=False,
        use_cache=True,
        pad_token_id=0,
    )
    assert generated.shape == (1, 6)


def test_operation_prepare_and_token_request_share_model_lifecycle(tmp_path):
    """A pruned, token-compressed model resumes the same optimizer update."""
    from mllm_comp.backbones.base import BackboneRuntime
    from mllm_comp.backbones.qwen25_omni.implementation import Qwen25OmniModelAdapter
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.core.vocabulary import ExecutionPhase
    from mllm_comp.config import config_to_dict
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.core.serialization import atomic_write_json, to_jsonable
    from mllm_comp.restoration import restore_config
    from mllm_comp.pipeline.operation_apply import Qwen25OmniOperationApplier
    from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher

    torch = pytest.importorskip("torch")
    torch.manual_seed(84)
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=Qwen25OmniModelAdapter(),
        operation_applier=Qwen25OmniOperationApplier(),
    )
    dispatcher = RuntimeCompressionDispatcher(
        contract_registry=builtin_multimodal_compression_contracts()
    )
    token = llm_stage(phase="forward")
    token.activation_scope["layers"] = [0]
    operation = _operation_stage()

    def activate(model, stages, phase):
        return dispatcher.activate(
            config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages)),
            facade=facade,
            runtime=BackboneRuntime(model=model, processor=None),
            compression_mode="compressed",
            execution_phase=phase,
        )

    def train_step(model, controller, optimizer):
        thinker = getattr(model, "thinker", model)
        optimizer.zero_grad(set_to_none=True)
        state, plan = controller.prepare_state(_state(torch, training=True))
        with controller.request(plan):
            output = thinker(
                input_ids=state.input_ids,
                attention_mask=state.attention_mask,
                labels=state.labels,
                use_cache=False,
            )
        assert output.logits.shape == (1, 5, 128)
        assert torch.isfinite(output.loss)
        output.loss.backward()
        weight = thinker.model.layers[1].mlp.gate_proj.weight
        assert torch.isfinite(weight.grad).all() and weight.grad.count_nonzero() > 0
        before = weight.detach().clone()
        optimizer.step()
        assert not torch.equal(weight, before)
        assert controller.last_report["kept_audio_tokens"] == 2
        return output.loss.detach().clone()

    model = _tiny_model(torch).train()
    original_layers = tuple(model.thinker.model.layers)
    native_forward = model.thinker.model.forward
    # Deliberately reverse configuration order: token hooks must see final layers.
    session = activate(model, [token, operation], "forward")
    try:
        assert session.install_order == (operation.name, token.name)
        assert tuple(model.thinker.model.layers) == (original_layers[0], original_layers[2])
        assert [layer.self_attn.layer_idx for layer in model.thinker.model.layers] == [0, 1]
        controller = session.active_records[1].activation
        model.thinker.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        train_step(model, controller, optimizer)
        checkpoint = tmp_path / "combined"
        export_operation_actions(
            model=model.thinker,
            output_dir=checkpoint,
            apply_result=session.active_records[0].activation.apply_result,
        )
        torch.save(optimizer.state_dict(), checkpoint / "optimizer.pt")
        expected_loss = train_step(model, controller, optimizer)
        expected_weights = {
            name: value.detach().clone() for name, value in model.state_dict().items()
        }
    finally:
        session.uninstall()
    assert model.thinker.model.forward == native_forward
    assert len(model.thinker.model.layers) == 2

    recorded_run = tmp_path / "recorded_run"
    recorded_config = experiment_config_from_dict({
        "experiment": {"name": "operation_token"},
        "backbone": {"name": "qwen25_omni", "model_path": "original_model"},
        "pipeline": {"stages": [to_jsonable(token), to_jsonable(operation)]},
        "train": {"compression_mode": "compressed"},
    })
    atomic_write_json(recorded_run / "effective_config.json", config_to_dict(recorded_config))
    atomic_write_json(recorded_run / "result.json", {
        "task": "train", "status": "succeeded", "output": {"checkpoint_path": str(checkpoint)},
    })
    restored_config = restore_config(recorded_run, task="eval")
    assert not restored_config.pipeline.stages[1].enabled
    assert restored_config.pipeline.stages[0].execution_phase is ExecutionPhase.GENERATE
    restored = Qwen25OmniModelAdapter().load_model(
        restored_config.backbone.model_path, train=True,
        **restored_config.backbone.config,
    )
    assert restored.config.mllm_comp_operations[0]["stage_name"] == operation.name
    restored.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    # The exported config already describes the final structure; do not prune twice.
    restored_session = activate(restored, [token], "forward")
    try:
        optimizer = torch.optim.SGD(restored.parameters(), lr=0.05, momentum=0.9)
        optimizer.load_state_dict(torch.load(checkpoint / "optimizer.pt", weights_only=True))
        actual_loss = train_step(restored, restored_session.active_records[0].activation, optimizer)
        torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected_weights[f"thinker.{name}"], rtol=0, atol=0)
    finally:
        restored_session.uninstall()

    restored.eval()
    inference = activate(restored, restored_config.pipeline.stages, "generate")
    try:
        controller = inference.active_records[0].activation
        state, plan = controller.prepare_state(_state(torch, training=False))
        with controller.request(plan), torch.no_grad():
            generated = restored.generate(
                input_ids=state.input_ids, attention_mask=state.attention_mask,
                min_new_tokens=2, max_new_tokens=2, do_sample=False,
                use_cache=True, pad_token_id=0,
            )
        assert generated.shape == (1, 9)
        assert controller.last_report["kept_audio_tokens"] == 2
        assert controller.last_report["decode_forward_calls"] == 1
    finally:
        inference.uninstall()


@pytest.mark.parametrize("edge, layer, source, expected_widths", [
    ("before", 0, 0, [5, 5, 5, 5]),
    ("before", 2, 2, [7, 7, 5, 5]),
    ("after", 1, 2, [7, 7, 5, 5]),
    ("after", 2, 0, [7, 7, 7, 5]),
])
def test_application_boundary_preserves_scoring_and_cache(edge, layer, source, expected_widths):
    torch = pytest.importorskip("torch")
    torch.manual_seed(21)
    model = _tiny_model(torch).eval()
    stage = llm_stage(phase="generate")
    stage.activation_scope["layers"] = [source]
    stage.operation.update(layers=[layer], layer_edge=edge)
    controller = _create_controller(model, stage)
    observed = []
    original_score = controller.strategy.score_audio_tokens
    def score(**kwargs):
        observed.append((kwargs["attention_module"], kwargs["normalized_hidden_states"].detach().clone()))
        return original_score(**kwargs)
    controller.strategy.score_audio_tokens = score
    widths = [[] for _ in range(4)]
    handles = [block.self_attn.register_forward_pre_hook(
        lambda module, args, kwargs, i=i: widths[i].append(int(kwargs["hidden_states"].shape[1])),
        with_kwargs=True,
    ) for i, block in enumerate(model.thinker.model.layers)]
    norms = []
    handles.append(model.thinker.model.layers[source].input_layernorm.register_forward_hook(
        lambda module, args, output: norms.append(output.detach().clone())
    ))
    controller.install()
    try:
        state, plan = controller.prepare_state(_state(torch, training=False))
        with controller.request(plan):
            output = model.thinker(
                input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=True,
            )
            assert [int(output.past_key_values.get_seq_length(i)) for i in range(4)] == expected_widths
            for step in range(2):
                output = model.thinker(
                    input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 8 + step), dtype=torch.long),
                    past_key_values=output.past_key_values, use_cache=True,
                    cache_position=torch.tensor([7 + step]),
                )
            assert [int(output.past_key_values.get_seq_length(i)) for i in range(4)] == [n + 2 for n in expected_widths]
        assert [row[0] for row in widths] == expected_widths
        assert len(observed) == 1
        torch.testing.assert_close(observed[0][1], norms[0], rtol=0, atol=0)
        assert observed[0][0] is model.thinker.model.layers[source].self_attn
        assert controller.runtime_metadata()["score_layers_zero_based"] == [source]
        assert controller.runtime_metadata()["application_positions"] == [{
            "boundary": "inter_layer", "index": layer + (edge == "after"),
        }]
    finally:
        controller.uninstall()
        for handle in handles:
            handle.remove()


def test_before_and_after_aliases_produce_identical_outputs():
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(22)
    original = _tiny_model(torch).eval()
    outputs = []
    for edge, layer in (("before", 2), ("after", 1)):
        model = copy.deepcopy(original)
        stage = llm_stage(phase="forward")
        stage.activation_scope["layers"] = [2]
        stage.operation.update(layers=[layer], layer_edge=edge)
        controller = _create_controller(model, stage)
        controller.install()
        try:
            state, plan = controller.prepare_state(_state(torch, training=False))
            with controller.request(plan):
                output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
            outputs.append(output.logits.detach())
            assert controller.last_report["kept_audio_tokens"] == 2
        finally:
            controller.uninstall()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_reject_future_hidden_scoring_at_initial_boundary():
    torch = pytest.importorskip("torch")
    stage = llm_stage(phase="forward")
    stage.activation_scope["layers"] = [2]
    stage.operation.update(layers=[0], layer_edge="before")
    with pytest.raises(Exception, match="precedes scoring layer 2"):
        _create_controller(_tiny_model(torch), stage)


@pytest.mark.parametrize("site, layers, edge", [
    ("llm_decoder_layer_output_audio_tokens", [0], "before"),
    ("llm_decoder_layer_output_audio_tokens", [1], "after"),
    ("llm_decoder_layer_output_audio_tokens", [2], "after"),
    ("llm_decoder_layer_post_attention_audio_tokens", [2], None),
])
@pytest.mark.parametrize("probe_layer", [1, 3])
def test_headrouter_application_keeps_calibrated_probe_and_initial_embeddings(site, layers, edge, probe_layer):
    from mllm_comp.compressors.llm_level.headrouter import PROFILE_NAMES
    torch = pytest.importorskip("torch")
    torch.manual_seed(23)
    model = _tiny_model(torch).eval()
    stage = llm_stage(phase="forward")
    stage.plugin = stage.algorithm = "headrouter_dynamic_audio_pruning"
    stage.variant = f"{stage.algorithm}.audio"
    stage.binding = f"{stage.variant}.qwen25_omni"
    stage.criterion = {}
    stage.activation_scope["layers"] = [probe_layer]
    stage.schedule = {"pruning_ratio": 0.5}
    stage.config = {"profiles": {name: [0.25, 0.75] for name in PROFILE_NAMES}}
    stage.operation.update(target_site=site, layers=layers)
    if edge is not None:
        stage.operation["layer_edge"] = edge
    controller = _create_controller(model, stage)
    scored = []
    original_score = controller.strategy.score_audio_tokens
    def score(**kwargs):
        scored.append((kwargs["attention_module"], kwargs["initial_hidden_states"].clone()))
        return original_score(**kwargs)
    controller.strategy.score_audio_tokens = score
    controller.install()
    try:
        state, plan = controller.prepare_state(_state(torch, training=False))
        embeddings = model.thinker.model.embed_tokens(state.input_ids).detach()
        with controller.request(plan):
            model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
        assert len(scored) == 1
        assert scored[0][0] is model.thinker.model.layers[probe_layer].self_attn
        torch.testing.assert_close(scored[0][1], embeddings, rtol=0, atol=0)
        assert controller.runtime_metadata()["qk_probe_layer_zero_based"] == probe_layer
        assert controller.last_report["kept_audio_tokens"] == 2
        if edge == "before" or layers != [probe_layer]:
            assert "pruning_layer_zero_based" not in controller.runtime_metadata()
    finally:
        controller.uninstall()


@pytest.mark.parametrize("application", ["initial", "later"])
def test_native_generate_with_application_override(application):
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch).eval()
    stage = llm_stage(phase="generate")
    stage.activation_scope["layers"] = [0]
    stage.operation.update(layers=[0] if application == "initial" else [2],
                           layer_edge="before" if application == "initial" else "after")
    controller = _create_controller(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(_state(torch, training=False))
        with controller.request(plan):
            output = model.thinker.generate(
                input_ids=state.input_ids, attention_mask=state.attention_mask,
                min_new_tokens=3, max_new_tokens=3, do_sample=False, use_cache=True, pad_token_id=0,
            )
        assert output.shape == (1, 10)
        assert controller.last_report["decode_forward_calls"] == 2
    finally:
        controller.uninstall()


def test_initial_boundary_infers_absolute_position_for_cached_forward():
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(24)
    original = _tiny_model(torch).eval()
    outputs = []
    for explicit in (True, False):
        model = copy.deepcopy(original)
        stage = llm_stage(phase="generate")
        stage.activation_scope["layers"] = [0]
        stage.operation.update(layers=[0], layer_edge="before")
        controller = _create_controller(model, stage)
        controller.install()
        try:
            state, plan = controller.prepare_state(_state(torch, training=False))
            with controller.request(plan):
                output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=True)
                for step in range(2):
                    output = model.thinker(
                        input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 8 + step), dtype=torch.long),
                        past_key_values=output.past_key_values, use_cache=True,
                        **({"cache_position": torch.tensor([7 + step])} if explicit else {}),
                    )
                outputs.append(output.logits.detach())
        finally:
            controller.uninstall()
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


@pytest.mark.parametrize("has_audio, fail_after_capture", [(False, False), (True, True)])
def test_score_inputs_are_not_retained_by_replay(has_audio, fail_after_capture):
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch).train()
    stage = llm_stage(phase="forward")
    stage.activation_scope["layers"] = [0]
    stage.operation["layers"] = [2]
    controller = _create_controller(model, stage)
    state = _state(torch, training=False)
    if not has_audio:
        state.modality_token_spans = []
        state.input_ids = torch.tensor([[1, 4, 4, 4, 4, 2, 3]])
    state, plan = controller.prepare_state(state)
    controller.install()
    seen_sizes = []
    handle = model.thinker.model.layers[0].self_attn.register_forward_pre_hook(
        lambda module, args, kwargs: seen_sizes.append(len(controller._active_request.score_inputs)),
        with_kwargs=True,
    )
    try:
        if fail_after_capture:
            def fail(module, args, kwargs):
                raise RuntimeError("test abort after score capture")
            failure = model.thinker.model.layers[0].self_attn.register_forward_pre_hook(fail, with_kwargs=True)
            try:
                with pytest.raises(RuntimeError, match="test abort"):
                    with controller.request(plan):
                        model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
            finally:
                failure.remove()
            assert seen_sizes == [1]
        else:
            with controller.request(plan):
                model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
            assert seen_sizes == [0]
        assert controller._replay_request.score_inputs == {}
    finally:
        handle.remove()
        controller.uninstall()


@pytest.mark.parametrize("algorithm, application", [
    ("audio_pyramid_attention_pruning", "default"),
    ("audio_pyramid_attention_pruning", "later"),
    ("fastadasp_weighted_token_merging", "later"),
    ("audio_pyramid_attention_pruning", "mid"),
    ("fastadasp_weighted_token_merging", "mid"),
    ("headrouter_dynamic_audio_pruning", "initial"),
])
def test_two_device_application_uses_source_device(algorithm, application):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device smoke; CPU tests do not validate device_map.")
    from accelerate import dispatch_model
    model = _tiny_model(torch).eval()
    device_map = {name: "cuda:0" for name, _ in model.thinker.named_children() if name != "model"}
    device_map.update({f"model.{name}": "cuda:0" for name, _ in model.thinker.model.named_children() if name != "layers"})
    device_map.update({f"model.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(4)})
    model.thinker = dispatch_model(model.thinker, device_map=device_map)
    raw = _state(torch, training=False)
    baseline = model.thinker.generate(
        input_ids=raw.input_ids.to("cuda:0"), attention_mask=raw.attention_mask.to("cuda:0"),
        min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0,
    )
    assert baseline.shape == (1, 9)
    model.thinker.rope_deltas = None
    stage = llm_stage(phase="generate")
    stage.plugin = stage.algorithm = algorithm
    stage.variant = f"{algorithm}.audio"
    stage.binding = f"{stage.variant}.qwen25_omni"
    stage.activation_scope["layers"] = [0]
    if algorithm == "fastadasp_weighted_token_merging":
        stage.operation["type"] = "merge"
        stage.criterion = {}
        stage.schedule = {"merge_ratio": 0.5}
    elif algorithm == "headrouter_dynamic_audio_pruning":
        from mllm_comp.compressors.llm_level.headrouter import PROFILE_NAMES
        stage.activation_scope["layers"] = [2]
        stage.criterion = {}
        stage.schedule = {"pruning_ratio": 0.5}
        stage.config = {"profiles": {name: [0.25, 0.75] for name in PROFILE_NAMES}}
    if application in {"later", "mid"}:
        stage.operation["layers"] = [2]
        if application == "mid":
            stage.operation["target_site"] = "llm_decoder_layer_post_attention_audio_tokens"
    elif application == "initial":
        stage.operation.update(layers=[0], layer_edge="before")
    controller = _create_controller(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(_state(torch, training=False))
        with controller.request(plan), torch.no_grad():
            output = model.thinker.generate(
                input_ids=state.input_ids.to("cuda:0"), attention_mask=state.attention_mask.to("cuda:0"),
                min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0,
            )
        assert output.shape == (1, 9)
        assert controller.last_report["kept_audio_tokens"] == 2
    finally:
        controller.uninstall()

    from mllm_comp.core.vocabulary import ExecutionPhase
    stage.execution_phase = ExecutionPhase.FORWARD
    model.train()
    model.thinker.rope_deltas = None
    model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    controller = _create_controller(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(_state(torch, training=True))
        with controller.request(plan):
            output = model.thinker(
                input_ids=state.input_ids.to("cuda:0"), attention_mask=state.attention_mask.to("cuda:0"),
                labels=state.labels.to("cuda:0"), use_cache=False,
            )
        output.loss.backward()
        assert torch.isfinite(model.thinker.model.embed_tokens.weight.grad).all()
        assert model.thinker.model.embed_tokens.weight.grad.count_nonzero() > 0
    finally:
        controller.uninstall()


@pytest.mark.parametrize("application", ["default", "initial", "later", "mid"])
def test_offloaded_weights_use_hook_execution_device(application):
    torch = pytest.importorskip("torch")
    from accelerate import cpu_offload
    model = _tiny_model(torch).eval()
    cpu_offload(model.thinker, execution_device=torch.device("cpu"))
    raw = _state(torch, training=False)
    with torch.no_grad():
        baseline = model.thinker(input_ids=raw.input_ids, attention_mask=raw.attention_mask, use_cache=False)
    assert baseline.logits.shape == (1, 7, 128)
    stage = llm_stage(phase="forward")
    stage.activation_scope["layers"] = [0]
    if application == "initial":
        stage.operation.update(layers=[0], layer_edge="before")
    elif application in {"later", "mid"}:
        stage.operation["layers"] = [2]
        if application == "mid":
            stage.operation["target_site"] = "llm_decoder_layer_post_attention_audio_tokens"
    controller = _create_controller(model, stage)
    controller.install()
    try:
        state, plan = controller.prepare_state(raw)
        with controller.request(plan), torch.no_grad():
            output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
        assert output.logits.shape == (1, 5, 128)
        assert controller.last_report["kept_audio_tokens"] == 2
    finally:
        controller.uninstall()

@pytest.mark.parametrize("algorithm,score_source,factor", [
    ("uniform_sampling_audio", None, 2), ("uniform_average_pooling_audio", None, 2),
    ("local_temporal_bipartite_merging_audio", None, 2),
    ("segmentwise_topk_audio", "uniform", 2), ("segmentwise_topk_audio", "random", 2),
    ("uniform_sampling_audio", None, 1), ("fastslm_hta_audio", None, 2),
])
@pytest.mark.parametrize("position", ["initial", "before", "full", "mid", "multiple"])
def test_temporal_resampling_llm_positions_and_checkpoint(algorithm, score_source, factor, position):
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(45)
    original = _tiny_model(torch)
    operation = {"type": "keep" if algorithm in {"uniform_sampling_audio", "segmentwise_topk_audio"} else "merge"}
    operation["target_site"] = (
        "llm_decoder_layer_post_attention_audio_tokens" if position == "mid"
        else "llm_decoder_layer_output_audio_tokens"
    )
    operation["layers"] = [0, 2] if position == "multiple" else [0 if position == "initial" else 1]
    if position in {"initial", "before"}:
        operation["layer_edge"] = "before"
    stage = CompressionStageConfig.from_dict({
        "name": "temporal-resampling-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
        "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": operation, "criterion": {"score_source": score_source, "num_segments": 3} if score_source else {}, "schedule": (
            {"keep_ratio": 0.5 if position == "multiple" else 0.25}
            if algorithm == "local_temporal_bipartite_merging_audio" else (
                {"keep_ratio": 0.5} if algorithm in {"segmentwise_topk_audio", "fastslm_hta_audio"} else {"compression_factor": factor}
            )
        ),
    })
    if algorithm == "fastslm_hta_audio":
        stage.config = {"window_tokens": 8, "output_query_tokens": 4, "stage_query_tokens": 4, "detail_recovery_depth": 1}
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original).train()
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        if algorithm == "fastslm_hta_audio":
            torch.manual_seed(81)
        controller = _create_controller(model, stage)
        widths = [[] for _ in range(4)]
        attention_widths = [[] for _ in range(4)]
        handles = []
        for i, layer in enumerate(model.thinker.model.layers):
            handles.append(layer.mlp.gate_proj.register_forward_pre_hook(
                lambda module, args, i=i: widths[i].append(args[0].shape[1]),
            ))
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, i=i: attention_widths[i].append(kwargs["hidden_states"].shape[1]),
                with_kwargs=True,
            ))
        raw = ForwardState(
            input_ids=torch.tensor([[1, 99, 99, 99, 99, 99, 2, 3]]),
            attention_mask=torch.ones((1, 8), dtype=torch.long),
            labels=torch.tensor([[-100] * 7 + [3]]),
            modality_token_spans=[TokenSpan(0, 1, 6, modality="audio", role="observation")],
            modality_metadata={"semantic_token_labels_by_sample": [["ins_special"] + ["audio"] * 5 + ["instruction", "response"]]},
        )
        first = 3 if algorithm == "uniform_sampling_audio" else 2
        second = 2 if algorithm == "uniform_sampling_audio" else 1
        if algorithm == "local_temporal_bipartite_merging_audio" and position != "multiple":
            first = 1
        if factor == 1:
            first = second = 5
        expected_attention = [8, 8, 3 + first, 3 + first]
        expected_ffn = list(expected_attention)
        if position == "initial":
            expected_attention = expected_ffn = [3 + first] * 4
        elif position == "before":
            expected_attention = expected_ffn = [8, 3 + first, 3 + first, 3 + first]
        elif position == "mid":
            expected_ffn[1] = 3 + first
        elif position == "multiple":
            expected_attention = expected_ffn = [8, 3 + first, 3 + first, 3 + second]
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            with controller.request(plan) as request:
                output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask,
                                       labels=state.labels, use_cache=False)
                assert request.score_inputs == {}
            output.loss.backward()
            assert [row[0] for row in attention_widths] == expected_attention
            assert [row[0] for row in widths] == expected_ffn
            final_count = second if position == "multiple" else first
            assert output.logits.shape == (1, 3 + final_count, 128)
            assert controller.runtime_metadata()["score_layers_zero_based"] == []
            assert controller.last_report["kept_audio_tokens"] == final_count
            module_gradients = []
            if algorithm == "fastslm_hta_audio":
                module_gradients = [p.grad for p in controller.strategy_runtime_module.parameters() if p.requires_grad]
                assert all(g is not None and torch.isfinite(g).all() for g in module_gradients)
                assert any(torch.count_nonzero(g) for g in module_gradients)
                assert controller.strategy_runtime_module_name.endswith("_llm_token")
                assert controller.strategy_runtime_module.compressor.query.shape[-1] == model.thinker.model.config.hidden_size
            results.append((output.logits.detach(), model.thinker.model.embed_tokens.weight.grad,
                            *[layer.mlp.gate_proj.weight.grad for layer in model.thinker.model.layers], *module_gradients))
            if checkpoint is None:
                from mllm_comp.core.vocabulary import ExecutionPhase
                from dataclasses import replace
                eval_raw = replace(raw, labels=None)
                controller.uninstall()
                model.eval()
                stage.execution_phase = ExecutionPhase.GENERATE
                if algorithm != "fastslm_hta_audio":
                    controller = _create_controller(model, stage)
                controller.install()
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    cached = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == expected_attention
                    cached = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 9), dtype=torch.long),
                                           past_key_values=cached.past_key_values, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [n + 1 for n in expected_attention]
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(input_ids=state.input_ids, attention_mask=state.attention_mask,
                        min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, 10)
                assert controller.last_report["prefill_forward_calls"] == 1
                assert controller.last_report["decode_forward_calls"] == 1
                stage.execution_phase = ExecutionPhase.FORWARD
        finally:
            controller.uninstall()
            for handle in handles:
                handle.remove()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            assert torch.isfinite(actual).all()
            torch.testing.assert_close(actual, expected)


# Each projector and downstream rule is wired once below, followed by the
# interactions that change reference mapping or preflight counts. Sorting itself
# is covered for all permutations by the lightweight contract/session test.
# Checkpoint replay is shared: exercise it for mean, weighted and dynamic layouts
# rather than multiplying every rule, audio count and stage by every mode.
@pytest.mark.parametrize("encoder_algorithm,projector_algorithm,llm_algorithm,audio_units,checkpointed", [
    pytest.param(None, "uniform_average_pooling_audio", "uniform_sampling_audio", 1, False, id="mean-then-sample"),
    pytest.param(None, "unsupervised_segmentation_audio", "unsupervised_segmentation_audio", 2, False, id="dynamic-then-dynamic"),
    pytest.param(None, "uniform_sampling_audio", "segmentwise_topk_audio", 2, False, id="sample-then-score"),
    pytest.param(None, "segmentwise_topk_audio", "similarity_attention_merge_prune_audio", 1, False, id="score-then-weighted-merge"),
    pytest.param(None, "local_temporal_bipartite_merging_audio", "encoder_attention_topk_audio", 2, False, id="mean-map-then-encoder-score"),
    pytest.param(None, "similarity_attention_merge_prune_audio", "progressive_attention_pruning_audio", 2, False, id="weighted-map-then-progressive"),
    pytest.param(None, "encoder_attention_topk_audio", "adjacent_token_merging_audio", 2, False, id="encoder-score-then-key-merge"),
    pytest.param(None, "progressive_attention_pruning_audio", "uniform_sampling_audio", 1, False, id="progressive-then-sample"),
    pytest.param(None, "adjacent_token_merging_audio", "similarity_attention_merge_prune_audio", 2, False, id="key-merge-then-weighted-merge"),
    pytest.param("uniform_sampling_audio", "uniform_average_pooling_audio", "uniform_sampling_audio", 2, False, id="encoder-select-mean-select"),
    pytest.param("uniform_average_pooling_audio", "local_temporal_bipartite_merging_audio", "encoder_attention_topk_audio", 2, False, id="encoder-mean-mean-score"),
    pytest.param("unsupervised_segmentation_audio", "segmentwise_topk_audio", "progressive_attention_pruning_audio", 2, False, id="encoder-dynamic-score-progressive"),
    pytest.param("uniform_sampling_audio", "encoder_attention_topk_audio", "similarity_attention_merge_prune_audio", 2, False, id="encoder-select-score-weighted"),
    pytest.param("uniform_average_pooling_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio", 2, False, id="encoder-mean-progressive-key-merge"),
    pytest.param("unsupervised_segmentation_audio", "adjacent_token_merging_audio", "encoder_attention_topk_audio", 2, False, id="encoder-dynamic-key-merge-score"),
    pytest.param("uniform_sampling_audio", "unsupervised_segmentation_audio", "unsupervised_segmentation_audio", 2, False, id="encoder-select-dynamic-dynamic"),
    pytest.param("uniform_average_pooling_audio", "similarity_attention_merge_prune_audio", "segmentwise_topk_audio", 2, False, id="encoder-mean-weighted-score"),
    pytest.param("unsupervised_segmentation_audio", "uniform_sampling_audio", "uniform_sampling_audio", 2, False, id="encoder-dynamic-select-select"),
    pytest.param("uniform_average_pooling_audio", "local_temporal_bipartite_merging_audio", "encoder_attention_topk_audio", 2, True, id="checkpoint-mean-reference"),
    pytest.param("unsupervised_segmentation_audio", "segmentwise_topk_audio", "progressive_attention_pruning_audio", 2, True, id="checkpoint-dynamic-reference"),
    pytest.param("uniform_sampling_audio", "unsupervised_segmentation_audio", "unsupervised_segmentation_audio", 2, True, id="checkpoint-dynamic-llm"),
    pytest.param("uniform_average_pooling_audio", "similarity_attention_merge_prune_audio", "adjacent_token_merging_audio", 2, True, id="checkpoint-weighted-reference"),
])
def test_audio_stages_prepare_in_physical_order(checkpointed, encoder_algorithm, projector_algorithm, audio_units, llm_algorithm):
    import copy
    from contextlib import ExitStack
    from dataclasses import replace
    from mllm_comp.backbones.base import BackboneRuntime
    from mllm_comp.backbones.qwen25_omni.implementation import _prepare_runtime_compression
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.pipeline.runtime_session import RuntimeCompressionDispatcher

    torch = pytest.importorskip("torch")
    torch.manual_seed(48)
    encoder_rule = llm_algorithm in {"encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"}
    projector_encoder_rule = projector_algorithm in {"encoder_attention_topk_audio", "progressive_attention_pruning_audio", "adjacent_token_merging_audio"}
    original = _tiny_model(torch, audio_encoder_layers=3 if encoder_rule or projector_encoder_rule or "similarity_attention_merge_prune_audio" in {projector_algorithm, llm_algorithm} else 1).train()
    feature_length = 32 if encoder_algorithm else 16
    features = torch.randn(audio_units, 4, feature_length)
    projector = CompressionStageConfig.from_dict({
        "name": "projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": projector_algorithm, "algorithm": projector_algorithm,
        "variant": f"{projector_algorithm}.audio", "binding": f"{projector_algorithm}.audio.qwen25_omni",
        "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": ("merge" if projector_algorithm == "adjacent_token_merging_audio" else "keep" if projector_encoder_rule or projector_algorithm in {"uniform_sampling_audio", "segmentwise_topk_audio"}
                               else "merge_prune" if projector_algorithm == "similarity_attention_merge_prune_audio" else "merge"),
                      "target_site": "audio_encoder_output_pre_projector"},
        "criterion": {"similarity_threshold": 1.0} if projector_algorithm == "similarity_attention_merge_prune_audio" else {},
        "schedule": ({} if projector_algorithm == "unsupervised_segmentation_audio" else
                     {"compression_factor": 2} if projector_algorithm.startswith("uniform_") else {"keep_ratio": 0.5}),
    })
    if projector_encoder_rule:
        projector.activation_scope["layers"] = [2]
        projector.schedule = {"merge_ratio": 0.5} if projector_algorithm == "adjacent_token_merging_audio" else {"minimum_keep_tokens": 1, **({"keep_ratio": 0.5} if projector_algorithm == "encoder_attention_topk_audio" else {"pruning_rate": 0.5})}
    llm = CompressionStageConfig.from_dict({
        "name": "llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": llm_algorithm, "algorithm": llm_algorithm,
        "variant": f"{llm_algorithm}.audio", "binding": f"{llm_algorithm}.audio.qwen25_omni",
        "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "merge" if llm_algorithm in {"unsupervised_segmentation_audio", "adjacent_token_merging_audio"} else "merge_prune" if llm_algorithm == "similarity_attention_merge_prune_audio" else "keep", "target_site": "llm_decoder_layer_output_audio_tokens", "layers": [0], "layer_edge": "before"},
        "criterion": {"similarity_threshold": 1.0} if llm_algorithm == "similarity_attention_merge_prune_audio" else {},
        "schedule": ({} if llm_algorithm == "unsupervised_segmentation_audio" else
                     {"compression_factor": 2} if llm_algorithm == "uniform_sampling_audio" else {"keep_ratio": 0.5}),
    })
    if encoder_rule:
        llm.activation_scope["layers"] = [2]
        llm.schedule = {"merge_ratio": 0.5} if llm_algorithm == "adjacent_token_merging_audio" else {"minimum_keep_tokens": 1, **({"keep_ratio": 0.5} if llm_algorithm == "encoder_attention_topk_audio" else {"pruning_rate": 0.5})}
    ordered = [projector, llm]
    if encoder_algorithm:
        ordered.insert(0, CompressionStageConfig.from_dict({
            "name": "encoder", "placement": "encoder_level", "execution_phase": "forward",
            "plugin": encoder_algorithm, "algorithm": encoder_algorithm,
            "variant": f"{encoder_algorithm}.audio", "binding": f"{encoder_algorithm}.audio.qwen25_omni",
            "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
            "operation": {"type": "keep" if encoder_algorithm in {"uniform_sampling_audio", "segmentwise_topk_audio"} else "merge",
                          "target_site": "audio_encoder_layer_output_post_block", "layers": [0], "layer_edge": "before"},
            "schedule": {} if encoder_algorithm == "unsupervised_segmentation_audio" else {"compression_factor": 2},
        }))
    expected_order = tuple(stage.name for stage in ordered)
    results = []
    for stages in (list(reversed(ordered)), ordered):
        model = copy.deepcopy(original)
        if checkpointed:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        runtime = BackboneRuntime(model=model, processor=None)
        native_rope = model.thinker.get_rope_index
        native_lengths = model.thinker.audio_tower._get_feat_extract_output_lengths
        native_encoder = model.thinker.audio_tower.forward
        session = RuntimeCompressionDispatcher(contract_registry=builtin_multimodal_compression_contracts()).activate(
            config=SimpleNamespace(pipeline=SimpleNamespace(stages=stages)),
            facade=SimpleNamespace(capabilities=lambda: SimpleNamespace(name="qwen25_omni")),
            runtime=runtime, compression_mode="compressed", execution_phase="forward",
        )
        assert session.install_order == expected_order
        input_features = features.clone().requires_grad_()
        audio_tokens = feature_length // 4
        token_ids = [1]
        spans = []
        for _ in range(audio_units):
            token_ids.append(102)
            spans.append(TokenSpan(0, len(token_ids), len(token_ids) + audio_tokens, modality="audio", role="observation"))
            token_ids.extend([*([99] * audio_tokens), 103])
        token_ids.extend([2, 3])
        ids = torch.tensor([token_ids])
        text_tokens = 3 + 2 * audio_units
        mask = torch.ones_like(ids)
        labels = torch.tensor([[-100] * (len(token_ids) - 1) + [3]])
        raw = ForwardState(
            input_ids=ids, attention_mask=mask, labels=labels, audio_values=input_features,
            audio_feature_lengths=torch.tensor([feature_length] * audio_units),
            modality_token_spans=spans,
            modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0, "padding_side": "right",
                               "expected_token_counts_by_sample": [{"audio": audio_tokens * audio_units}]},
            native_model_inputs={"input_ids": ids, "attention_mask": mask, "labels": labels,
                                 "input_features": input_features, "feature_attention_mask": torch.ones((audio_units, feature_length), dtype=torch.long)},
        )
        try:
            state, prepared = _prepare_runtime_compression(runtime, raw)
            assert tuple(item.name for item in prepared) == expected_order
            projector_counts = [action.metadata for action in prepared[-2].plan.token_actions]
            encoder_counts = [a.metadata["keep_count"] for a in prepared[0].plan.token_actions] if encoder_algorithm else [4] * audio_units
            if encoder_algorithm != "unsupervised_segmentation_audio":
                assert encoder_counts == [4] * audio_units
            assert [item["original_count"] for item in projector_counts] == encoder_counts
            projected_counts = [item["keep_count"] for item in projector_counts]
            projected_tokens = sum(projected_counts)
            final_tokens = (None if llm_algorithm == "unsupervised_segmentation_audio" else
                            sum((count + 1) // 2 if llm_algorithm in {"uniform_sampling_audio", "adjacent_token_merging_audio"} else max(1, count // 2) for count in projected_counts))
            if final_tokens is None:
                assert all(a.metadata["keep_count"] is None for a in prepared[-1].plan.token_actions)
                assert prepared[-1].plan.this_stage_metadata.stats["preflight_audio_encoder_pass"] is False
            assert state.input_ids.shape == (1, projected_tokens + text_tokens)
            with ExitStack() as stack:
                for item in prepared:
                    stack.enter_context(item.activation.request(item.plan))
                rope_kwargs = dict(input_ids=state.input_ids, attention_mask=state.attention_mask)
                actual_rope = model.thinker.get_rope_index(**rope_kwargs, audio_seqlens=torch.tensor([feature_length] * audio_units))
                expected_rope = native_rope(**rope_kwargs, audio_seqlens=torch.tensor([4 * count for count in projected_counts]))
                for actual, expected in zip(actual_rope, expected_rope):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            if final_tokens is None:
                final_tokens = prepared[-1].activation.last_report["kept_audio_tokens"]
            assert output.logits.shape == (1, final_tokens + text_tokens, 128)
            assert [item.activation.last_report["kept_audio_tokens"] for item in prepared] == (
                ([sum(encoder_counts)] if encoder_algorithm else []) + [projected_tokens, final_tokens]
            )
            assert input_features.grad.count_nonzero() > 0
            results.append((output.logits.detach(), input_features.grad,
                            model.thinker.model.layers[0].mlp.gate_proj.weight.grad))
            if not checkpointed:
                model.eval()
                eval_raw = replace(raw, labels=None, native_model_inputs={
                    key: value for key, value in raw.native_model_inputs.items() if key != "labels"
                })
                state, prepared = _prepare_runtime_compression(runtime, eval_raw)
                with ExitStack() as stack, torch.no_grad():
                    for item in prepared:
                        stack.enter_context(item.activation.request(item.plan))
                    generated = model.thinker.generate(**state.native_model_inputs,
                        min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, projected_tokens + text_tokens + 2)
                assert prepared[-1].activation.last_report["decode_forward_calls"] == 1
        finally:
            session.uninstall()
            assert model.thinker.get_rope_index == native_rope
            assert model.thinker.audio_tower._get_feat_extract_output_lengths == native_lengths
            assert model.thinker.audio_tower.forward == native_encoder
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("encoder_algorithm,score_source,boundary", [
    (algorithm, score, boundary)
    for algorithm, score in [
        ("uniform_sampling_audio", None), ("uniform_average_pooling_audio", None),
        ("local_temporal_bipartite_merging_audio", None),
        ("segmentwise_topk_audio", "uniform"), ("segmentwise_topk_audio", "random"),
    ]
    for boundary in ["initial", "mid", "full"]
] + [("segmentwise_topk_audio", "encoder_last_layer_attention", boundary) for boundary in ["mid", "full"]]
  + [("unsupervised_segmentation_audio", None, boundary) for boundary in ["initial", "mid", "full"]]
  + [("similarity_attention_merge_prune_audio", None, boundary) for boundary in ["mid", "full"]])
@pytest.mark.parametrize("projector_algorithm", ["segmentwise_topk_audio", "similarity_attention_merge_prune_audio"])
def test_encoder_layout_updates_attention_projector_source(encoder_algorithm, score_source, projector_algorithm, boundary):
    import copy
    from contextlib import ExitStack

    torch = pytest.importorskip("torch")
    torch.manual_seed(51)
    original = _tiny_model(torch, audio_encoder_layers=3).train()
    features = torch.randn(4, 138)
    ids = torch.tensor([[1, *([99] * 15), 2, *([99] * 19), 3]])
    lengths = torch.tensor([61, 77])
    stages = []
    for name, algorithm, placement, operation, schedule in [
        ("encoder", encoder_algorithm, "encoder_level", {
            "type": "keep" if encoder_algorithm in {"uniform_sampling_audio", "segmentwise_topk_audio"} else "merge_prune" if encoder_algorithm == "similarity_attention_merge_prune_audio" else "merge",
            "target_site": "audio_encoder_layer_post_mhsa_pre_ffn" if boundary == "mid" else "audio_encoder_layer_output_post_block",
            "layers": [0] if boundary == "initial" else [2],
            **({"layer_edge": "before"} if boundary == "initial" else {}),
        }, {} if encoder_algorithm == "unsupervised_segmentation_audio" else
           {"keep_ratio": 0.5} if encoder_algorithm in {"local_temporal_bipartite_merging_audio", "segmentwise_topk_audio", "similarity_attention_merge_prune_audio"} else {"compression_factor": 2}),
        ("projector", projector_algorithm, "projector_level", {
            "type": "keep" if projector_algorithm == "segmentwise_topk_audio" else "merge_prune",
            "target_site": "audio_encoder_output_pre_projector",
        }, {"keep_ratio": 0.5}),
    ]:
        stages.append(CompressionStageConfig.from_dict({
            "name": name, "placement": placement, "execution_phase": "forward",
            "plugin": algorithm, "algorithm": algorithm, "variant": f"{algorithm}.audio",
            "binding": f"{algorithm}.audio.qwen25_omni", "target_modalities": ["audio"],
            "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
            "operation": operation, "schedule": schedule,
            "criterion": ({"similarity_threshold": 1.0} if algorithm == "similarity_attention_merge_prune_audio"
                          else {"score_source": score_source} if name == "encoder" and score_source else {}),
        }))
    results = []
    for reentrant in (None, False, True):
        model = copy.deepcopy(original)
        tower = model.thinker.audio_tower
        if reentrant is not None:
            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": reentrant})
        encoder, projector = [_create_controller(model, stage) for stage in stages]
        captured = []
        original_capture = projector._capture_attention_scores
        def capture(module, args, kwargs, *, retain_source=True):
            if projector._active_request is not None:
                captured.append((retain_source, kwargs["hidden_states"].detach().clone()))
            return original_capture(module, args, kwargs, retain_source=retain_source)
        projector._capture_attention_scores = capture
        encoder.install()
        projector.install()
        layout_updates = []
        original_observer = encoder._layout_observer
        def observe_layout(strategy, plan, chunks, chunk_counts):
            layout_updates.append(plan)
            original_observer(strategy, plan, chunks, chunk_counts)
        encoder._layout_observer = observe_layout
        values = features.clone().requires_grad_()
        state = ForwardState(
            input_ids=ids, attention_mask=torch.ones_like(ids), audio_values=values,
            audio_feature_lengths=lengths,
            native_model_inputs={"input_ids": ids, "input_features": values},
            modality_token_spans=[TokenSpan(0, 1, 16, modality="audio"), TokenSpan(0, 17, 36, modality="audio")],
            modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0, "padding_side": "right",
                               "expected_token_counts_by_sample": [{"audio": 34}]},
        )
        try:
            compact, encoder_plan = encoder.prepare_state(state)
            compact, projector_plan = projector.prepare_state(compact)
            layout_updates.clear()  # Natural-budget Encoder preparation also executes the real loop.
            with ExitStack() as requests:
                requests.enter_context(encoder.request(encoder_plan))
                active = requests.enter_context(projector.request(projector_plan))
                aftercnn, output_lengths = tower._get_feat_extract_output_lengths(lengths)
                output = tower(values, feature_lens=lengths, aftercnn_lens=aftercnn).last_hidden_state
                expected_cnn = [a.metadata["final_encoder_count"] for a in encoder_plan.token_actions]
                assert active.aftercnn_lengths == expected_cnn
                assert active.attention_source is None
            output.square().mean().backward()
            assert len(layout_updates) == 1
            expected_counts = [a.metadata["keep_count"] for a in projector_plan.token_actions]
            if encoder_algorithm != "unsupervised_segmentation_audio":
                assert expected_counts == ([4, 5] if encoder_algorithm == "uniform_sampling_audio" else [3, 4])
            assert output_lengths.tolist() == expected_counts
            assert output.shape[0] == sum(expected_counts)
            assert [retained for retained, _ in captured] == ([True] if boundary == "initial" else [True, False])
            if boundary != "initial":
                chunks = [n for action in encoder_plan.token_actions for n in action.metadata["initial_chunk_lengths"]]
                expected = []
                for chunk_index, chunk in enumerate(captured[0][1].split(chunks)):
                    if encoder_algorithm == "uniform_sampling_audio":
                        expected.append(chunk[::2])
                    elif encoder_algorithm == "segmentwise_topk_audio":
                        expected.append(chunk[layout_updates[0][1][chunk_index]])
                    elif encoder_algorithm == "unsupervised_segmentation_audio":
                        expected.append(torch.stack([chunk[start:end].mean(0) for start, end in layout_updates[0][1][chunk_index]]))
                    elif encoder_algorithm == "similarity_attention_merge_prune_audio":
                        groups, importance, selected = layout_updates[0][1][chunk_index]
                        mapped = []
                        for group_index in selected.tolist():
                            group = list(groups[group_index])
                            weights = importance[group] / importance[group].sum()
                            mapped.append((chunk[group] * weights.unsqueeze(-1)).sum(0))
                        expected.append(torch.stack(mapped))
                    elif encoder_algorithm == "local_temporal_bipartite_merging_audio":
                        for pairs in layout_updates[0][1][chunk_index]:
                            sources = {source for source, _ in pairs}
                            chunk = torch.stack([
                                chunk[[i] + [source for source, destination in pairs if destination == i]].mean(0)
                                for i in range(chunk.shape[0]) if i not in sources
                            ])
                        expected.append(chunk)
                    else:
                        expected.append(chunk[:chunk.shape[0] // 2 * 2].reshape(-1, 2, chunk.shape[-1]).mean(1))
                torch.testing.assert_close(captured[1][1], torch.cat(expected), rtol=0, atol=0)
            results.append((output.detach(), values.grad, *[layer.fc1.weight.grad for layer in tower.layers]))
        finally:
            projector.uninstall()
            assert encoder._layout_observer is None
            encoder.uninstall()
            assert not hasattr(tower, "_mllm_encoder_compression")
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("score_source", ["uniform", "random"])
@pytest.mark.parametrize("position", ["initial", "mid"])
def test_segmentwise_llm_preserves_per_audio_seed_and_absolute_positions(score_source, position):
    import copy
    from mllm_comp.compressors.projector_level.segmentwise_topk import segmentwise_random_indices, segmentwise_uniform_indices

    torch = pytest.importorskip("torch")
    torch.manual_seed(56)
    original = _tiny_model(torch).train()
    stage = CompressionStageConfig.from_dict({
        "name": "segmentwise-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": "segmentwise_topk_audio",
        "algorithm": "segmentwise_topk_audio", "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "keep", "layers": [0], "target_site": (
            "llm_decoder_layer_output_audio_tokens" if position == "initial" else "llm_decoder_layer_post_attention_audio_tokens"
        ), **({"layer_edge": "before"} if position == "initial" else {})},
        "criterion": {"score_source": score_source, "num_segments": 2, "random_seed": 47},
        "schedule": {"keep_ratio": 0.4},
    })
    ids = torch.tensor([[1, *([99] * 5), 2, *([99] * 5), 3], [1, *([99] * 5), 2, 3, *([0] * 5)]])
    raw = ForwardState(
        input_ids=ids, attention_mask=(ids != 0).long(),
        labels=torch.tensor([[-100] * 12 + [3], [-100] * 7 + [3] + [-100] * 5]),
        modality_token_spans=[TokenSpan(0, 1, 6, modality="audio"), TokenSpan(0, 7, 12, modality="audio"), TokenSpan(1, 1, 6, modality="audio")],
        modality_metadata={"semantic_token_labels_by_sample": [
            ["ins_special"] + ["audio"] * 5 + ["instruction"] + ["audio"] * 5 + ["response"],
            ["ins_special"] + ["audio"] * 5 + ["instruction", "response"] + ["pad"] * 5,
        ]},
    )
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = _create_controller(model, stage)
        controller.debug_enabled = True
        calls = []
        compress = controller.strategy.rule.compress
        def observe(values, **kwargs):
            calls.append(kwargs["item_index"])
            return compress(values, **kwargs)
        controller.strategy.rule.compress = observe
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            with controller.request(plan):
                output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, labels=state.labels, use_cache=False)
            output.loss.backward()
            assert calls == [0, 1, 2]
            assert output.logits.shape == (2, 7, 128)
            for index, (unit, start) in enumerate(zip(controller.last_report["stages"][0]["units"], [1, 7, 1])):
                kwargs = dict(keep_ratio=0.4, num_segments=2, minimum_keep_tokens=2)
                selected = (segmentwise_random_indices(5, seed=47 + index, **kwargs) if score_source == "random"
                            else segmentwise_uniform_indices(5, **kwargs))
                assert unit["kept_absolute_sequence_positions"] == [start + i for i in selected]
            results.append((output.logits.detach(), model.thinker.model.embed_tokens.weight.grad,
                            *[layer.mlp.gate_proj.weight.grad for layer in model.thinker.model.layers]))
        finally:
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("position", ["initial", "before", "full", "mid", "multiple"])
def test_encoder_attention_reference_llm_positions_and_gradients(position):
    import copy
    from dataclasses import replace
    from mllm_comp.core.errors import InvariantError

    torch = pytest.importorskip("torch")
    torch.manual_seed(75)
    original = _tiny_model(torch, audio_encoder_layers=3).train()
    features = torch.randn(1, 4, 32)
    ids = torch.tensor([[1, 102, *([99] * 8), 103, 2, 3]])
    stage = CompressionStageConfig.from_dict({
        "name": "encoder-reference-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": "segmentwise_topk_audio", "algorithm": "segmentwise_topk_audio", "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
        "operation": {"type": "keep", "layers": [0, 2] if position == "multiple" else [0 if position == "initial" else 1],
            "target_site": "llm_decoder_layer_post_attention_audio_tokens" if position == "mid" else "llm_decoder_layer_output_audio_tokens",
            **({"layer_edge": "before"} if position in {"initial", "before"} else {})},
        "criterion": {"num_segments": 3}, "schedule": {"keep_ratio": 0.5},
    })
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = _create_controller(model, stage, profile_enabled=True)
        values = features.clone().requires_grad_()
        mask = torch.ones_like(ids)
        labels = torch.tensor([[-100] * 12 + [3]])
        raw = ForwardState(input_ids=ids, attention_mask=mask, labels=labels, audio_values=values,
            audio_feature_lengths=torch.tensor([32]), modality_token_spans=[TokenSpan(0, 2, 10, modality="audio")],
            modality_metadata={"token_ids": {"audio": 99}, "pad_token_id": 0, "expected_token_counts_by_sample": [{"audio": 8}]},
            native_model_inputs={"input_ids": ids, "attention_mask": mask, "labels": labels,
                                 "input_features": values, "feature_attention_mask": torch.ones((1, 32), dtype=torch.long)})
        normalized = []
        attention = model.thinker.audio_tower.layers[-1].self_attn
        capture = attention.register_forward_pre_hook(
            lambda module, args, kwargs: normalized.append((kwargs["hidden_states"].detach().clone(), kwargs["cu_seqlens"].tolist())), with_kwargs=True)
        widths = [[] for _ in range(4)]
        handles = [layer.mlp.gate_proj.register_forward_pre_hook(lambda module, args, i=i: widths[i].append(args[0].shape[1]))
                   for i, layer in enumerate(model.thinker.model.layers)]
        controller.install()
        try:
            if checkpoint is None and position == "initial":
                state, plan = controller.prepare_state(raw)
                with pytest.raises(InvariantError, match="requires an encoder forward"):
                    with controller.request(plan):
                        model.thinker(input_ids=ids, attention_mask=mask, use_cache=False)
                assert controller.encoder_references[None].source is None
                assert controller.encoder_references[None].sequence_scores is None
            state, plan = controller.prepare_state(raw)
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
                assert controller.encoder_references[None].source is None
                assert controller.encoder_references[None].audio_scores == []
                with torch.no_grad():
                    source, cu = normalized[0]
                    scores = []
                    for start, end in zip(cu, cu[1:]):
                        chunk = source[start:end]
                        query = attention.q_proj(chunk).reshape(end - start, attention.num_heads, -1).transpose(0, 1)
                        key = attention.k_proj(chunk).reshape(end - start, attention.num_heads, -1).transpose(0, 1)
                        received = (query @ key.transpose(-2, -1) * attention.scaling).softmax(-1, dtype=torch.float32).sum(-2).mean(0)
                        scores.append(received)
                    expected = torch.cat(scores).reshape(8, 2).mean(1)
                    torch.testing.assert_close(controller.encoder_references[None].sequence_scores[0, 2:10], expected, rtol=0, atol=0)
            assert controller.encoder_references[None].sequence_scores is None
            output.loss.backward()
            expected_attention = [13, 13, 9, 9]
            expected_ffn = list(expected_attention)
            if position == "initial":
                expected_attention = expected_ffn = [9] * 4
            elif position == "before":
                expected_attention = expected_ffn = [13, 9, 9, 9]
            elif position == "mid":
                expected_ffn[1] = 9
            elif position == "multiple":
                expected_attention = expected_ffn = [13, 9, 9, 7]
            assert [row[0] for row in widths] == expected_ffn
            assert controller.runtime_metadata()["encoder_score_layer_zero_based"] == 2
            assert controller.last_report["score_seconds"] > 0
            assert controller.runtime_metadata()["score_layers_zero_based"] == []
            assert controller.last_report["kept_audio_tokens"] == (2 if position == "multiple" else 4)
            results.append((output.logits.detach(), values.grad, model.thinker.model.embed_tokens.weight.grad,
                            *[layer.mlp.gate_proj.weight.grad for layer in model.thinker.model.layers]))
            if checkpoint is None:
                model.eval()
                eval_raw = replace(raw, labels=None, native_model_inputs={k: v for k, v in raw.native_model_inputs.items() if k != "labels"})
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    output = model.thinker(**state.native_model_inputs, use_cache=True)
                    assert [output.past_key_values.get_seq_length(i) for i in range(4)] == expected_attention
                    output = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 14), dtype=torch.long),
                        past_key_values=output.past_key_values, use_cache=True)
                    assert [output.past_key_values.get_seq_length(i) for i in range(4)] == [n + 1 for n in expected_attention]
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2,
                        do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, 15)
                assert controller.last_report["decode_forward_calls"] == 1
        finally:
            capture.remove()
            for handle in handles:
                handle.remove()
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("checkpoint_format", ["pytorch", "pytorch-folder", "safetensors-shards", "safetensors-index"])
def test_hta_three_positions_register_train_and_reload_distinct_modules(checkpoint_format, tmp_path):
    from contextlib import ExitStack
    import json
    from safetensors.torch import save_file
    from mllm_comp.compressors.projector_level.fastslm_hta import load_fastslm_hta_checkpoint

    torch = pytest.importorskip("torch")
    torch.manual_seed(88)
    model = _tiny_model(torch, audio_encoder_layers=2).train()
    stages = [_hta_layer_stage(family, initial=family == "encoder") for family in ("encoder", "projector", "llm")]
    controllers = [_create_controller(model, stage) for stage in stages]
    names = [c.strategy_runtime_module_name for c in controllers]
    assert len(set(names)) == 3
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    for c in controllers:
        assert all(id(p) in optimized for p in c.strategy_runtime_module.parameters())
        c.install()
    raw = _hta_audio_state(torch)
    try:
        state = raw
        plans = []
        for c in controllers:
            state, plan = c.prepare_state(state)
            plans.append(plan)
        with ExitStack() as stack:
            for c, plan in zip(controllers, plans):
                stack.enter_context(c.request(plan))
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        assert output.logits.shape == (1, 6, 128)
        output.loss.backward()
        assert raw.audio_values.grad.count_nonzero()
        for c in controllers:
            assert any(p.grad is not None and p.grad.count_nonzero() for p in c.strategy_runtime_module.parameters())
        optimizer.step()
        checkpoint = tmp_path / checkpoint_format
        if checkpoint_format.startswith("pytorch"):
            if checkpoint_format == "pytorch-folder":
                checkpoint.mkdir()
                torch.save(model.state_dict(), checkpoint / "pytorch_model.bin")
            else:
                checkpoint = checkpoint.with_suffix(".bin")
                torch.save(model.state_dict(), checkpoint)
        else:
            checkpoint.mkdir()
            weight_map = {}
            for index, c in enumerate(controllers):
                shard_name = f"model-{index}.safetensors"
                values = {f"thinker.{c.strategy_runtime_module_name}.{key}": value.detach().contiguous()
                          for key, value in c.strategy_runtime_module.state_dict().items()}
                save_file(values, checkpoint / shard_name)
                weight_map.update({key: shard_name for key in values})
            if checkpoint_format == "safetensors-index":
                (checkpoint / "model.safetensors.index.json").write_text(json.dumps({"weight_map": weight_map}))
        replacement = _tiny_model(torch, audio_encoder_layers=2).train()
        for stage, c in zip(stages, controllers):
            stage.config["checkpoint_path"] = str(checkpoint)
            restored = _create_controller(replacement, stage)
            for key, expected in c.strategy_runtime_module.state_dict().items():
                torch.testing.assert_close(restored.strategy_runtime_module.state_dict()[key], expected, rtol=0, atol=0)
        # A multi-slot checkpoint must not silently substitute the projector.
        from mllm_comp.core.errors import InvariantError
        partial = tmp_path / "missing-llm-hta.bin"
        torch.save({key: value for key, value in model.state_dict().items() if names[2] not in key}, partial)
        with pytest.raises(InvariantError, match="no HTA weights"):
            load_fastslm_hta_checkpoint(controllers[2].strategy_runtime_module, partial, module_name=names[2])
        # A standard checkpoint remains reusable at another same-width position.
        standard = tmp_path / "standard-hta.bin"
        torch.save({f"thinker.{names[1]}.{key}": value for key, value in controllers[1].strategy_runtime_module.state_dict().items()}, standard)
        load_fastslm_hta_checkpoint(controllers[0].strategy_runtime_module, standard, module_name=names[0])
        for key, expected in controllers[1].strategy_runtime_module.state_dict().items():
            torch.testing.assert_close(controllers[0].strategy_runtime_module.state_dict()[key], expected, rtol=0, atol=0)
    finally:
        for c in reversed(controllers):
            c.uninstall()


@pytest.mark.parametrize("family,initial,identity,allowed", [
    ("encoder", True, False, True), ("encoder", False, False, False),
    ("projector", False, False, False), ("encoder", False, True, True), ("projector", False, True, True),
])
def test_hta_fixed_encoder_reference_requires_a_defined_token_transfer(family, initial, identity, allowed):
    from contextlib import ExitStack
    from mllm_comp.core.errors import InvariantError

    torch = pytest.importorskip("torch")
    torch.manual_seed(89)
    model = _tiny_model(torch, audio_encoder_layers=2).train()
    upstream = _create_controller(model, _hta_layer_stage(family, initial=initial, identity=identity))
    stage = CompressionStageConfig.from_dict({
        "name": "reference", "placement": "llm_level", "execution_phase": "forward",
        "plugin": "segmentwise_topk_audio", "algorithm": "segmentwise_topk_audio", "variant": "segmentwise_topk_audio.audio",
        "binding": "segmentwise_topk_audio.audio.qwen25_omni", "target_modalities": ["audio"],
        "operation": {"type": "keep", "target_site": "llm_decoder_layer_output_audio_tokens", "layers": [0], "layer_edge": "before"},
        "schedule": {"keep_ratio": 0.5},
    })
    upstream.install()
    llm = _create_controller(model, stage)
    llm.install()
    raw = _hta_audio_state(torch)
    try:
        state, upstream_plan = upstream.prepare_state(raw)
        state, llm_plan = llm.prepare_state(state)
        def run():
            with upstream.request(upstream_plan), llm.request(llm_plan):
                return model.thinker(**state.native_model_inputs, use_cache=False)
        if allowed:
            result = run()
            result.loss.backward()
            assert raw.audio_values.grad.count_nonzero()
        else:
            with pytest.raises(InvariantError, match="no defined transfer"):
                run()
        assert llm.encoder_references[None].source is None
        assert llm.encoder_references[None].sequence_scores is None
    finally:
        llm.uninstall()
        upstream.uninstall()


@pytest.mark.parametrize("family", ["encoder", "projector", "llm"])
@pytest.mark.parametrize("offload_before_controller", [False, True])
def test_hta_positions_follow_offload_execution_device(family, offload_before_controller):
    from accelerate import cpu_offload
    torch = pytest.importorskip("torch")
    torch.manual_seed(90)
    model = _tiny_model(torch, audio_encoder_layers=2).eval()
    if offload_before_controller:
        cpu_offload(model.thinker, execution_device=torch.device("cpu"))
    controller = _create_controller(model, _hta_layer_stage(family))
    if not offload_before_controller:
        cpu_offload(model.thinker, execution_device=torch.device("cpu"))
    controller.install()
    try:
        state, plan = controller.prepare_state(_hta_audio_state(torch))
        with controller.request(plan), torch.no_grad():
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        assert torch.isfinite(output.logits).all()
        assert output.logits.shape == (1, 9, 128)
    finally:
        controller.uninstall()


@pytest.mark.parametrize("family", ["encoder", "projector", "llm"])
def test_hta_positions_follow_trainer_wrapper_and_frozen_rule(family):
    torch = pytest.importorskip("torch")
    torch.manual_seed(91)
    model = _tiny_model(torch, audio_encoder_layers=2).train()
    stage = _hta_layer_stage(family)
    stage.config["trainable"] = False
    controller = _create_controller(model, stage)
    class Wrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
            self.calls = 0
        def forward(self, *args, **kwargs):
            self.calls += 1
            return self.inner(*args, **kwargs)
    wrapper = Wrapper(controller.strategy_runtime_module)
    setattr(model.thinker, controller.strategy_runtime_module_name, wrapper)
    controller.install()
    try:
        raw = _hta_audio_state(torch)
        state, plan = controller.prepare_state(raw)
        with controller.request(plan):
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        output.loss.backward()
        assert wrapper.calls > 0
        assert controller.strategy_runtime_module is wrapper
        assert all(not p.requires_grad and p.grad is None for p in wrapper.parameters())
        assert raw.audio_values.grad.count_nonzero()
    finally:
        controller.uninstall()


@pytest.mark.parametrize("boundary", ["full", "mid"])
@pytest.mark.parametrize("family", ["encoder", "llm"])
def test_hta_trains_with_frozen_encoder_and_untracked_features(boundary, family):
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(92)
    original = _tiny_model(torch, audio_encoder_layers=2).train()
    original.thinker.requires_grad_(False)
    raw = _hta_audio_state(torch)
    raw.audio_values.requires_grad_(False)
    stage = _hta_layer_stage(family)
    stage.operation["target_site"] = (
        ("audio_encoder_layer_post_mhsa_pre_ffn" if boundary == "mid" else "audio_encoder_layer_output_post_block")
        if family == "encoder" else
        ("llm_decoder_layer_post_attention_audio_tokens" if boundary == "mid" else "llm_decoder_layer_output_audio_tokens")
    )
    results = []
    for checkpointed in (None, False, True):
        model = copy.deepcopy(original)
        if checkpointed is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointed})
        torch.manual_seed(93)
        controller = _create_controller(model, stage)
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            with controller.request(plan):
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            gradients = [p.grad for p in controller.strategy_runtime_module.parameters()]
            assert all(g is not None and torch.isfinite(g).all() for g in gradients)
            assert any(g.count_nonzero() for g in gradients)
            results.append((output.logits.detach(), *gradients))
            assert all(p.grad is None for p in model.thinker.audio_tower.parameters())
        finally:
            controller.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


def test_hta_input_gradient_hook_preserves_no_audio_and_uninstalls():
    torch = pytest.importorskip("torch")
    model = _tiny_model(torch).train()
    model.thinker.requires_grad_(False)
    model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    previous_hooks = len(model.thinker.model._forward_pre_hooks)
    controller = _create_controller(model, _hta_layer_stage("llm"))
    controller.install()
    observed = []
    capture = model.thinker.model.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(kwargs["inputs_embeds"].requires_grad), with_kwargs=True)
    ids = torch.tensor([[1, 2, 3]])
    raw = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), modality_token_spans=[])
    try:
        state, plan = controller.prepare_state(raw)
        with controller.request(plan):
            output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=False)
        assert observed == [False]
        assert not output.logits.requires_grad
    finally:
        capture.remove()
        controller.uninstall()
    assert len(model.thinker.model._forward_pre_hooks) == previous_hooks


@pytest.mark.parametrize("position", ["initial", "before", "full", "mid", "multiple"])
def test_unseg_llm_uses_natural_peaks_at_actual_boundary(position):
    import copy
    torch = pytest.importorskip("torch")
    torch.manual_seed(96)
    original = _tiny_model(torch).train()
    ids = torch.tensor([[1, *([99] * 8), 2, *([99] * 6), 3, 4]])
    original_width = ids.shape[1]
    features = torch.randn(1, original_width, original.thinker.model.config.hidden_size)
    stage = CompressionStageConfig.from_dict({
        "name": "unseg-llm", "placement": "llm_level", "execution_phase": "forward",
        "plugin": "unsupervised_segmentation_audio", "algorithm": "unsupervised_segmentation_audio",
        "variant": "unsupervised_segmentation_audio.audio", "binding": "unsupervised_segmentation_audio.audio.qwen25_omni",
        "target_modalities": ["audio"], "operation": {"type": "merge",
            "target_site": "llm_decoder_layer_post_attention_audio_tokens" if position == "mid" else "llm_decoder_layer_output_audio_tokens",
            "layers": [0, 2] if position == "multiple" else [0 if position == "initial" else 1],
            **({"layer_edge": "before"} if position in {"initial", "before"} else {})},
    })
    results = []
    for checkpointed in (None, False, True):
        model = copy.deepcopy(original)
        if checkpointed is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpointed})
        controller = _create_controller(model, stage)
        observed = []
        compress = controller.strategy.rule.compress
        def observe(values, **kwargs):
            assert kwargs["keep_count"] is None
            output, indices, details = compress(values, **kwargs)
            with torch.no_grad():
                normalized = torch.nn.functional.normalize(values.detach().float(), dim=-1)
                distance = 1 - (normalized[:-1] * normalized[1:]).sum(-1)
                cuts = [i + 1 for i in range(1, len(distance) - 1) if distance[i] > distance[i - 1] and distance[i] > distance[i + 1]]
                endpoints = [0, *cuts, len(values)]
                expected = torch.stack([values.detach()[start:end].mean(0) for start, end in zip(endpoints, endpoints[1:])])
                torch.testing.assert_close(output.detach(), expected, rtol=0, atol=0)
                assert details["segments"] == list(zip(endpoints, endpoints[1:]))
            observed.append((len(values), len(output)))
            return output, indices, details
        controller.strategy.rule.compress = observe
        embeddings = features.clone().requires_grad_()
        raw = ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids), labels=torch.tensor([[-100] * (original_width - 1) + [4]]),
            modality_token_spans=[TokenSpan(0, 1, 9, modality="audio"), TokenSpan(0, 10, 16, modality="audio")])
        attention_widths, ffn_widths = [[] for _ in range(4)], [[] for _ in range(4)]
        handles = []
        for i, layer in enumerate(model.thinker.model.layers):
            handles.append(layer.self_attn.register_forward_pre_hook(
                lambda module, args, kwargs, i=i: attention_widths[i].append(kwargs["hidden_states"].shape[1]), with_kwargs=True))
            handles.append(layer.mlp.gate_proj.register_forward_pre_hook(lambda module, args, i=i: ffn_widths[i].append(args[0].shape[1])))
        controller.install()
        try:
            state, plan = controller.prepare_state(raw)
            assert all(a.metadata["keep_count"] is None and a.metadata["natural_peak_count"] is None for a in plan.token_actions)
            assert plan.this_stage_metadata.stats["planned_audio_tokens"] is None
            assert plan.this_stage_metadata.stats["preflight_audio_encoder_pass"] is False
            with controller.request(plan):
                output = model.thinker(input_ids=state.input_ids, inputs_embeds=embeddings, attention_mask=state.attention_mask,
                                       labels=state.labels, use_cache=False)
            output.loss.backward()
            assert len(observed) == (4 if position == "multiple" else 2)
            first = 4 + sum(count for _, count in observed[:2])
            final = 4 + sum(count for _, count in observed[-2:])
            expected_attention = [original_width, original_width, first, first]
            expected_ffn = list(expected_attention)
            if position == "initial":
                expected_attention = expected_ffn = [first] * 4
            elif position == "before":
                expected_attention = expected_ffn = [original_width, first, first, first]
            elif position == "mid":
                expected_ffn[1] = first
            elif position == "multiple":
                expected_attention = expected_ffn = [original_width, first, first, final]
            assert [row[0] for row in attention_widths] == expected_attention
            assert [row[0] for row in ffn_widths] == expected_ffn
            assert output.logits.shape == (1, final, 128)
            assert controller.last_report["kept_audio_tokens"] == final - 4
            results.append((output.logits.detach(), embeddings.grad, *[layer.mlp.gate_proj.weight.grad for layer in model.thinker.model.layers]))
            if checkpointed is None:
                from dataclasses import replace
                model.eval()
                eval_raw = replace(raw, labels=None)
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    cached = model.thinker(input_ids=state.input_ids, inputs_embeds=embeddings.detach(), attention_mask=state.attention_mask, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == expected_attention
                    cached = model.thinker(input_ids=torch.tensor([[7]]), attention_mask=torch.ones((1, original_width + 1), dtype=torch.long),
                                           past_key_values=cached.past_key_values, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [n + 1 for n in expected_attention]
                state, plan = controller.prepare_state(eval_raw)
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(input_ids=state.input_ids, inputs_embeds=embeddings.detach(), attention_mask=state.attention_mask,
                        min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, original_width + 2)
                assert controller.last_report["prefill_forward_calls"] == 1
                assert controller.last_report["decode_forward_calls"] == 1
        finally:
            controller.uninstall()
            for handle in handles:
                handle.remove()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)
