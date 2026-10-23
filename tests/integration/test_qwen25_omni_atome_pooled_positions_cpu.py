"""Native A-ToMe across pooled audio boundaries with independent Key replay."""

import copy
import json
from contextlib import ExitStack
from dataclasses import dataclass, replace

import pytest

from mllm_comp.core.vocabulary import Placement
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.integration.test_qwen25_omni_encoder_rules_llm_cpu import _stage
from tests.fixtures.qwen25_omni_audio import _tiny_model, _create_controller, _hta_audio_state


def _merge_pairs(torch, values, starts):
    starts = set(starts.tolist())
    return torch.stack([values[i:i+2].mean(0) if i in starts else values[i]
                        for i in range(len(values)) if i - 1 not in starts])


@pytest.mark.parametrize("position", ["initial", "mid", "full", "before", "multiple", "projector_pre", "projector_post"])
@pytest.mark.parametrize("upstream", [False, True])
def test_atome_pooled_key_mapping_actual_values_gradients_and_generate(position, upstream):
    torch = pytest.importorskip("torch")
    torch.manual_seed(109)
    base = _tiny_model(torch, audio_encoder_layers=3).train()
    stage = _stage("adjacent_token_merging_audio", "multiple" if position.startswith("projector") else position)
    stage.operation["type"] = "merge"
    stage.schedule = {"merge_ratio": 0.25}
    if position.startswith("projector"):
        stage.placement = Placement.PROJECTOR_LEVEL
        stage.operation = {"type": "merge", "target_site": "audio_encoder_output_pre_projector" if position == "projector_pre" else "audio_projector_output_post_projector"}
    results = []
    for checkpoint in (None, False, True):
        model = copy.deepcopy(base)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        encoder = None
        if upstream:
            encoder = _create_controller(model, CompressionStageConfig.from_dict({
                "name": "encoder", "placement": "encoder_level", "execution_phase": "forward",
                "plugin": "uniform_sampling_audio", "algorithm": "uniform_sampling_audio", "variant": "uniform_sampling_audio.audio", "binding": "uniform_sampling_audio.audio.qwen25_omni",
                "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"], "media_alignment": "temporal",
                "operation": {"type": "keep", "layers": [1], "target_site": "audio_encoder_layer_post_mhsa_pre_ffn"}, "schedule": {"compression_factor": 2},
            }))
            encoder.install()
        controller = _create_controller(model, stage)
        controller.debug_enabled = True
        captured, history, checks = {}, [], []
        handles = []
        for source in controller.strategy.layers:
            def capture(module, args, kwargs, source=source):
                captured.setdefault(source, kwargs["hidden_states"].detach().clone())
            handles.append(model.thinker.audio_tower.layers[source].self_attn.register_forward_pre_hook(capture, with_kwargs=True))
        compress = controller.strategy.rule.compress
        def checked(hidden, keys, lengths, **kwargs):
            source = kwargs["layer_index"]
            normalized = captured[source][::2] if upstream and source == 0 else captured[source]
            with torch.no_grad():
                projected = model.thinker.audio_tower.layers[source].self_attn.k_proj(normalized)
                expected_keys = projected.reshape(-1, 2, projected.shape[-1]).mean(1)
                for starts in history:
                    expected_keys = _merge_pairs(torch, expected_keys, starts)
                torch.testing.assert_close(keys, expected_keys, rtol=0, atol=0)
                paired = keys[:len(keys)//2*2].reshape(len(keys)//2, 2, -1)
                similarities = torch.nn.functional.cosine_similarity(paired[:, 0].float(), paired[:, 1].float(), dim=-1, eps=1e-8)
                starts = (similarities.argsort(descending=True, stable=True)[:int(len(keys)*0.25)] * 2).sort().values
            result = compress(hidden, keys, lengths, **kwargs)
            torch.testing.assert_close(result[3][1][0], starts, rtol=0, atol=0)
            torch.testing.assert_close(result[0], _merge_pairs(torch, hidden, starts), rtol=0, atol=0)
            history.append(starts)
            checks.append(source)
            return result
        controller.strategy.rule.compress = checked
        controller.install()
        try:
            torch.manual_seed(110)
            raw = _hta_audio_state(torch)
            state = raw
            if encoder is not None:
                state, encoder_plan = encoder.prepare_state(state)
            state, plan = controller.prepare_state(state)
            final_count = plan.token_actions[0].metadata["keep_count"]
            with ExitStack() as requests:
                if encoder is not None:
                    requests.enter_context(encoder.request(encoder_plan))
                requests.enter_context(controller.request(plan))
                output = model.thinker(**state.native_model_inputs, use_cache=False)
            output.loss.backward()
            assert checks == list(controller.strategy.layers)
            assert output.logits.shape[1] == 5 + final_count
            assert "_reference_replay_plans" not in json.dumps(controller.last_report)
            assert all(reference.source is None and reference.sequence_scores is None for reference in controller.encoder_references.values())
            results.append((output.logits.detach(), raw.audio_values.grad.detach(), model.thinker.audio_tower.layers[0].fc1.weight.grad.detach(), model.thinker.model.layers[0].mlp.gate_proj.weight.grad.detach()))
            if checkpoint is None:
                controller.strategy.rule.compress = compress
                model.eval()
                raw = replace(raw, labels=None, native_model_inputs={k: v for k, v in raw.native_model_inputs.items() if k != "labels"})
                state = raw
                if encoder is not None:
                    state, encoder_plan = encoder.prepare_state(state)
                state, plan = controller.prepare_state(state)
                expected_cache = [int(state.input_ids.shape[1])] * 4
                if stage.placement is Placement.LLM_LEVEL:
                    counts = plan.token_actions[0].metadata["stage_lengths"]
                    for point, before, after in zip(controller.application_positions, counts, counts[1:]):
                        start = point.index + (point.boundary == "post_attention_pre_ffn")
                        for layer in range(start, 4):
                            expected_cache[layer] -= before - after
                with ExitStack() as requests, torch.no_grad():
                    if encoder is not None:
                        requests.enter_context(encoder.request(encoder_plan))
                    requests.enter_context(controller.request(plan))
                    cached = model.thinker(**state.native_model_inputs, use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == expected_cache
                    cached = model.thinker(input_ids=torch.tensor([[3]]), attention_mask=torch.ones((1, state.input_ids.shape[1]+1), dtype=torch.long),
                                           past_key_values=cached.past_key_values, cache_position=torch.tensor([state.input_ids.shape[1]]), use_cache=True)
                    assert [cached.past_key_values.get_seq_length(i) for i in range(4)] == [n+1 for n in expected_cache]
                state = raw
                if encoder is not None:
                    state, encoder_plan = encoder.prepare_state(state)
                state, plan = controller.prepare_state(state)
                with ExitStack() as requests, torch.no_grad():
                    if encoder is not None:
                        requests.enter_context(encoder.request(encoder_plan))
                    requests.enter_context(controller.request(plan))
                    generated = model.thinker.generate(**state.native_model_inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, state.input_ids.shape[1] + 2)
        finally:
            for handle in handles:
                handle.remove()
            controller.uninstall()
            if encoder is not None:
                assert encoder._layout_observer is None
                encoder.uninstall()
    for result in results[1:]:
        for actual, expected in zip(result, results[0]):
            torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("mid", [False, True])
def test_atome_key_replay_across_two_devices(mid):
    torch = pytest.importorskip("torch")
    if torch.cuda.device_count() < 2:
        pytest.skip("Explicit two-device Key mapping test.")
    from accelerate import dispatch_model

    for checkpoint in (None, False, True):
        torch.manual_seed(111)
        model = _tiny_model(torch, audio_encoder_layers=3).train()
        device_map = {name: "cuda:0" for name, _ in model.thinker.named_children() if name != "model"}
        device_map.update({f"model.{name}": "cuda:0" for name, _ in model.thinker.model.named_children() if name != "layers"})
        device_map.update({f"model.layers.{i}": "cuda:0" if i < 2 else "cuda:1" for i in range(4)})
        model.thinker = dispatch_model(model.thinker, device_map=device_map)
        if checkpoint is not None:
            model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": checkpoint})
        stage = _stage("adjacent_token_merging_audio", "multiple")
        stage.schedule = {"merge_ratio": 0.25}
        stage.operation["type"] = "merge"
        if mid:
            stage.operation["target_site"] = "llm_decoder_layer_post_attention_audio_tokens"
        controller = _create_controller(model, stage)
        controller.install()
        try:
            raw = _hta_audio_state(torch)
            state, plan = controller.prepare_state(raw)
            inputs = {name: value.to("cuda:0") if torch.is_tensor(value) else value for name, value in state.native_model_inputs.items()}
            with controller.request(plan):
                output = model.thinker(**inputs, use_cache=False)
            output.loss.backward()
            assert output.logits.shape[1] == 10
            assert torch.isfinite(raw.audio_values.grad).all() and raw.audio_values.grad.count_nonzero() > 0
            assert torch.isfinite(model.thinker.model.layers[2].mlp.gate_proj.weight.grad).all()
            assert all(reference.source is None and reference.sequence_scores is None for reference in controller.encoder_references.values())
            if checkpoint is None:
                model.eval()
                state, plan = controller.prepare_state(replace(raw, labels=None, native_model_inputs={k: v for k, v in raw.native_model_inputs.items() if k != "labels"}))
                inputs = {name: value.to("cuda:0") if torch.is_tensor(value) else value for name, value in state.native_model_inputs.items()}
                with controller.request(plan), torch.no_grad():
                    generated = model.thinker.generate(**inputs, min_new_tokens=2, max_new_tokens=2, do_sample=False, use_cache=True, pad_token_id=0)
                assert generated.shape == (1, 15)
        finally:
            controller.uninstall()


def test_independent_encoder_rule_reuses_projector_and_llm_with_opaque_reference_plans():
    torch = pytest.importorskip("torch")
    from mllm_comp.backbones.qwen25_omni.audio_compression import Qwen25OmniAudioCompressionController
    from mllm_comp.backbones.qwen25_omni.llm_token_compression import Qwen25OmniLLMTokenCompressionController
    from mllm_comp.backbones.qwen25_omni.v2_compression import resolve_qwen25_omni_compression_execution
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts

    @dataclass(frozen=True)
    class EndpointPlan:
        groups: tuple[tuple[int, ...], ...]

    class EndpointEncoderRule:
        """Reuse the existing budget, but independently merge distant endpoints."""
        token_operation = "merge"
        encoder_reference_kind = "key_vectors"

        def __init__(self, spec):
            self.spec = spec

        def active_layer_indices(self, num_layers):
            return self.spec.layer_indices

        def output_lengths(self, layer_index, lengths, active_layers):
            return [length - int(length * self.spec.merge_ratio) for length in lengths]

        def runtime_metadata(self, active_layers):
            return {"grouping_rule": "weighted_endpoints"}

        def compress(self, hidden, probe, lengths, *, layer_index, active_layers, collect_details):
            assert lengths == [len(hidden)]
            merged = int(len(hidden) * self.spec.merge_ratio)
            groups = tuple((i, len(hidden) - 1 - i) if i < merged else (i,)
                           for i in range(len(hidden) - merged))
            plan = EndpointPlan(groups)
            # No selected_pair_starts or other A-ToMe diagnostic is provided.
            return self.replay(hidden, plan), [len(groups)], [{"group_count": len(groups)}], plan

        def replay(self, hidden, plan):
            return torch.stack([hidden[group[0]] if len(group) == 1 else
                                hidden[group[0]] * 0.25 + hidden[group[1]] * 0.75
                                for group in plan.groups])

        def representative_indices(self, length, plan, device):
            return torch.tensor([group[0] for group in plan.groups], device=device)

    torch.manual_seed(113)
    model = _tiny_model(torch, audio_encoder_layers=3).train()
    registry = builtin_multimodal_compression_contracts()
    controllers = []
    for placement, controller_type in ((Placement.PROJECTOR_LEVEL, Qwen25OmniAudioCompressionController),
                                       (Placement.LLM_LEVEL, Qwen25OmniLLMTokenCompressionController)):
        stage = _stage("adjacent_token_merging_audio", "multiple")
        stage.name = f"endpoints-{placement.value}"
        stage.placement = placement
        stage.schedule = {"merge_ratio": 0.25}
        stage.operation["type"] = "merge"
        if placement is Placement.PROJECTOR_LEVEL:
            stage.operation = {"type": "merge", "target_site": "audio_encoder_output_pre_projector"}
        binding = replace(registry.binding(stage.binding), strategy_factory=EndpointEncoderRule)
        execution = resolve_qwen25_omni_compression_execution(stage=stage, binding=binding)
        controller = controller_type(model, stage, execution=execution)
        assert type(controller.strategy.rule) is EndpointEncoderRule
        controller.debug_enabled = True
        controllers.append(controller)

    captured, matrices, checks, handles = {}, [], [], []
    for source in (0, 2):
        def capture(module, args, kwargs, source=source):
            captured[source] = kwargs["hidden_states"].detach().clone()
        handles.append(model.thinker.audio_tower.layers[source].self_attn.register_forward_pre_hook(capture, with_kwargs=True))
    for controller in controllers:
        compress = controller.strategy.rule.compress
        def checked(hidden, keys, lengths, compress=compress, **kwargs):
            source = kwargs["layer_index"]
            with torch.no_grad():
                projected = model.thinker.audio_tower.layers[source].self_attn.k_proj(captured[source])
                expected_keys = projected.reshape(-1, 2, projected.shape[-1]).mean(1)
                for matrix in matrices:
                    expected_keys = matrix @ expected_keys
                torch.testing.assert_close(keys, expected_keys)
            result = compress(hidden, keys, lengths, **kwargs)
            merged = int(len(hidden) * 0.25)
            matrix = hidden.new_zeros((len(hidden) - merged, len(hidden)))
            for i in range(len(matrix)):
                matrix[i, i] = 0.25 if i < merged else 1.0
                if i < merged:
                    matrix[i, len(hidden) - 1 - i] = 0.75
            torch.testing.assert_close(result[0], matrix @ hidden)
            gradient, = torch.autograd.grad(result[0].sum(), hidden, retain_graph=True)
            torch.testing.assert_close(gradient, matrix.sum(0)[:, None].expand_as(hidden))
            matrices.append(matrix)
            checks.append(source)
            return result
        controller.strategy.rule.compress = checked

    try:
        for controller in controllers:
            controller.install()
        state = raw = _hta_audio_state(torch)
        plans = []
        for controller in controllers:
            state, plan = controller.prepare_state(state)
            plans.append(plan)
        with ExitStack() as requests:
            for controller, plan in zip(controllers, plans):
                requests.enter_context(controller.request(plan))
            output = model.thinker(**state.native_model_inputs, use_cache=False)
        output.loss.backward()
        assert checks == [0, 2, 0, 2]
        assert output.logits.shape[1] == 8  # Eight audio tokens become 6, 5, 4, then 3.
        for gradient in (raw.audio_values.grad, model.thinker.audio_tower.layers[0].fc1.weight.grad,
                         model.thinker.model.layers[0].mlp.gate_proj.weight.grad):
            assert torch.isfinite(gradient).all() and gradient.count_nonzero() > 0
        for controller in controllers:
            report = json.dumps(controller.last_report)
            assert "_reference_replay_plans" not in report and "_checkpoint_replay_plan" not in report
            assert all(reference.source is None and reference.sequence_scores is None and not reference.audio_scores
                       for reference in controller.encoder_references.values())
    finally:
        for controller in reversed(controllers):
            controller.uninstall()
        for handle in handles:
            handle.remove()
