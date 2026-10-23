"""Native Omni consumers execute configured LTBM rules at existing positions."""
import copy
from types import SimpleNamespace

import pytest

from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import ExecutionPhase
from tests.unit.test_ltbm_custom_rules import _specs


_ALGORITHM = "local_temporal_bipartite_merging_audio"


def _torch_and_encoder():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if transformers.__version__ != "4.57.3":
        pytest.skip("Pinned native integration requires Transformers 4.57.3")
    from transformers.models.qwen2_5_omni.configuration_qwen2_5_omni import Qwen2_5OmniAudioEncoderConfig
    from transformers.models.qwen2_5_omni.modeling_qwen2_5_omni import Qwen2_5OmniAudioEncoder
    config = Qwen2_5OmniAudioEncoderConfig(num_mel_bins=8, encoder_layers=4,
        encoder_attention_heads=4, encoder_ffn_dim=32, d_model=16,
        max_source_positions=64, n_window=16, output_dim=12)
    config._attn_implementation = "eager"
    return torch, Qwen2_5OmniAudioEncoder(config).train()


def _audio_state(torch, features):
    ids = torch.tensor([[1, *([12] * 16), 2]])
    lengths = torch.tensor([64])
    return ForwardState(input_ids=ids, attention_mask=torch.ones_like(ids),
        audio_values=features, audio_feature_lengths=lengths,
        native_model_inputs={"input_ids": ids, "input_features": features},
        modality_token_spans=[TokenSpan(0, 1, 17, modality="audio")],
        modality_metadata={"token_ids": {"audio": 12}, "pad_token_id": 0,
            "padding_side": "right", "expected_token_counts_by_sample": [{"audio": 16}]})


def _observe_rules(monkeypatch):
    import mllm_comp.compressors.rules as loading
    import tests.unit.test_ltbm_custom_rules as functions
    group_calls, aggregate_calls, loads = [], [], []
    group, aggregate, load = (
        functions.reverse_sources,
        functions.destination_weighted,
        loading.load_rule_specs,
    )
    def grouping(request):
        assert not request["values"].requires_grad
        group_calls.append((tuple(request["values"].shape), request["round_index"]))
        return group(request)
    def aggregation(request, *, destination_weight=3.0):
        aggregate_calls.append((tuple(request["values"].shape), request["round_index"]))
        return aggregate(request, destination_weight=destination_weight)

    def loading_once(specs):
        loads.append(specs)
        return load(specs)

    monkeypatch.setattr(functions, "reverse_sources", grouping)
    monkeypatch.setattr(functions, "destination_weighted", aggregation)
    monkeypatch.setattr(loading, "load_rule_specs", loading_once)
    return group_calls, aggregate_calls, loads


def _assert_metadata(controller):
    metadata = controller.runtime_metadata()
    assert metadata["rules"] == _specs()
    assert metadata["rules_changed"] is True
    assert metadata["canonical_evidence_applicable"] is False
    assert metadata["development_runtime"] is True
    assert controller.execution.binding.metadata["rules_changed"] is True


def _finite_nonzero(torch, tensor):
    assert tensor is not None and torch.isfinite(tensor).all() and torch.count_nonzero(tensor) > 0


@pytest.mark.parametrize("site,width,projector_length", [
    ("audio_encoder_output_pre_projector", 16, 8),
    ("audio_projector_output_post_projector", 12, 16),
])
def test_configured_rules_execute_before_and_after_native_audio_projector(site, width, projector_length, monkeypatch):
    from mllm_comp.backbones.qwen25_omni.audio_compression import Qwen25OmniAudioCompressionController
    from tests.fixtures.qwen25_omni_audio import _MiniThinker, projector_stage
    torch, tower = _torch_and_encoder()
    groups, aggregates, loads = _observe_rules(monkeypatch)
    stage = projector_stage(_ALGORITHM)
    stage.execution_phase = ExecutionPhase.FORWARD
    stage.operation.update(target_site=site, **_specs())
    controller = Qwen25OmniAudioCompressionController(SimpleNamespace(thinker=_MiniThinker(tower)), stage)
    controller.install()
    widths = []
    hook = tower.proj.register_forward_pre_hook(lambda module, args: widths.append(args[0].shape[0]))
    features = torch.randn(8, 64, requires_grad=True)
    raw = _audio_state(torch, features)
    try:
        state, plan = controller.prepare_state(raw)
        assert state.input_ids.shape[1] == 10
        with controller.request(plan):
            aftercnn, final = tower._get_feat_extract_output_lengths(raw.audio_feature_lengths)
            output = tower(features, feature_lens=raw.audio_feature_lengths, aftercnn_lens=aftercnn).last_hidden_state
        group_count = len(groups)
        output.square().mean().backward()
        assert len(groups) == group_count == 1
        assert groups[0][0] == (16, width)
        assert all(shape[1] == width for shape, _ in aggregates)
        assert widths == [projector_length]
        assert output.shape == (8, 12) and final.tolist() == [8]
        assert len(loads) == 1
        _finite_nonzero(torch, features.grad)
        _finite_nonzero(torch, tower.proj.weight.grad)
        _assert_metadata(controller)
    finally:
        hook.remove()
        controller.uninstall()


def test_configured_encoder_mid_rules_replay_after_request_for_both_checkpoints(monkeypatch):
    from mllm_comp.backbones.qwen25_omni.encoder_compression import Qwen25OmniEncoderCompressionController
    from tests.fixtures.qwen25_omni_audio import _MiniThinker, encoder_extension_stage
    torch, original = _torch_and_encoder()
    torch.manual_seed(305)
    features = torch.randn(8, 64)
    groups, _, loads = _observe_rules(monkeypatch)
    stage = encoder_extension_stage(_ALGORITHM)
    stage.operation.update(target_site="audio_encoder_layer_post_mhsa_pre_ffn", layers=[1], **_specs())
    stage.schedule = {"keep_ratio": 0.25}
    stage.execution_phase = ExecutionPhase.FORWARD
    results = []
    for checkpoint in (None, False, True):
        tower = copy.deepcopy(original)
        if checkpoint is not None:
            tower.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = Qwen25OmniEncoderCompressionController(SimpleNamespace(thinker=_MiniThinker(tower)), stage)
        controller.install()
        actual = features.clone().requires_grad_()
        raw = _audio_state(torch, actual)
        ffn_lengths = []
        handle = tower.layers[1].fc1.register_forward_pre_hook(lambda module, args: ffn_lengths.append(args[0].shape[0]))
        try:
            state, plan = controller.prepare_state(raw)
            assert state.input_ids.shape[1] == 6
            group_start = len(groups)
            with controller.request(plan):
                aftercnn, final = tower._get_feat_extract_output_lengths(raw.audio_feature_lengths)
                output = tower(actual, feature_lens=raw.audio_feature_lengths, aftercnn_lens=aftercnn).last_hidden_state
            decisions = len(groups)
            assert decisions > group_start
            output.square().mean().backward()
            assert len(groups) == decisions
            assert ffn_lengths[0] == 8 and output.shape == (4, 12) and final.tolist() == [4]
            _finite_nonzero(torch, actual.grad)
            _finite_nonzero(torch, tower.layers[1].fc1.weight.grad)
            _assert_metadata(controller)
            results.append((output.detach(), actual.grad, tower.layers[1].fc1.weight.grad))
        finally:
            handle.remove()
            controller.uninstall()
    assert len(loads) == 3
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


def test_configured_llm_mid_rules_replay_positions_gradients_and_decode(monkeypatch):
    from mllm_comp.pipeline.stage import CompressionStageConfig
    from tests.fixtures.qwen25_omni_audio import _tiny_model, _create_controller
    torch = pytest.importorskip("torch")
    torch.manual_seed(306)
    original = _tiny_model(torch)
    groups, _, loads = _observe_rules(monkeypatch)
    stage = CompressionStageConfig.from_dict({"name": "custom-ltbm-llm", "placement": "llm_level",
        "execution_phase": "forward", "plugin": _ALGORITHM, "algorithm": _ALGORITHM,
        "variant": f"{_ALGORITHM}.audio", "binding": f"{_ALGORITHM}.audio.qwen25_omni",
        "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"],
        "media_alignment": "temporal", "schedule": {"keep_ratio": 0.4},
        "operation": {"type": "merge", "target_site": "llm_decoder_layer_post_attention_audio_tokens",
            "layers": [1], **_specs()}})
    raw = ForwardState(input_ids=torch.tensor([[1, 99, 99, 99, 99, 99, 2, 3]]),
        attention_mask=torch.ones(1, 8, dtype=torch.long), labels=torch.tensor([[-100] * 7 + [3]]),
        modality_token_spans=[TokenSpan(0, 1, 6, modality="audio", role="observation")],
        modality_metadata={"semantic_token_labels_by_sample": [["ins_special"] + ["audio"] * 5 + ["instruction", "response"]]})
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(original).train()
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        controller = _create_controller(model, stage)
        controller.install()
        ffn_lengths = []
        handle = model.thinker.model.layers[1].mlp.gate_proj.register_forward_pre_hook(
            lambda module, args: ffn_lengths.append(args[0].shape[1]))
        try:
            state, plan = controller.prepare_state(raw)
            group_start = len(groups)
            with controller.request(plan):
                output = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask,
                    labels=state.labels, use_cache=False)
            decisions = len(groups)
            assert decisions > group_start
            output.loss.backward()
            assert len(groups) == decisions
            assert output.logits.shape == (1, 5, 128) and ffn_lengths[0] == 5
            _finite_nonzero(torch, model.thinker.model.embed_tokens.weight.grad)
            _finite_nonzero(torch, model.thinker.model.layers[1].mlp.gate_proj.weight.grad)
            _assert_metadata(controller)
            assert controller.last_report["kept_audio_tokens"] == 2
            results.append((output.logits.detach(), model.thinker.model.embed_tokens.weight.grad,
                model.thinker.model.layers[1].mlp.gate_proj.weight.grad))
            if checkpoint is None:
                from dataclasses import replace
                controller.uninstall()
                model.eval()
                stage.execution_phase = ExecutionPhase.GENERATE
                controller = _create_controller(model, stage)
                controller.install()
                for _ in range(2):
                    state, plan = controller.prepare_state(replace(raw, labels=None))
                    with controller.request(plan), torch.no_grad():
                        cached = model.thinker(input_ids=state.input_ids, attention_mask=state.attention_mask, use_cache=True)
                        assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [8, 8, 5, 5]
                        cached = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, 9), dtype=torch.long),
                            past_key_values=cached.past_key_values, use_cache=True)
                        assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [9, 9, 6, 6]
                stage.execution_phase = ExecutionPhase.FORWARD
        finally:
            handle.remove()
            controller.uninstall()
    assert len(loads) == 4
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)
