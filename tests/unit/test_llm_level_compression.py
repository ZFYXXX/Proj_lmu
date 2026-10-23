from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.qwen25_omni.v2_compression import (
    create_qwen25_omni_dynamic_controller,
    resolve_qwen25_omni_stage_execution,
)
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.pipeline.stage import CompressionStageConfig


torch = pytest.importorskip("torch")
ModelOutput = pytest.importorskip("transformers.utils").ModelOutput


class _Attention(torch.nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        self.num_heads = 2
        self.num_key_value_heads = 2
        self.head_dim = hidden_size // 2
        self.scaling = self.head_dim**-0.5
        self.rope_scaling = {"mrope_section": [1, 1, 1]}
        self.layer_idx = layer_idx
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)


class _Layer(torch.nn.Module):
    def __init__(self, hidden_size: int, layer_idx: int) -> None:
        super().__init__()
        self.input_layernorm = torch.nn.LayerNorm(hidden_size)
        self.self_attn = _Attention(hidden_size, layer_idx)
        self.proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, hidden_states, **kwargs):
        del kwargs
        return (hidden_states + self.proj(hidden_states),)


class _Thinker(torch.nn.Module):
    def __init__(self, hidden_size: int = 12, num_layers: int = 4) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList(
            [_Layer(hidden_size, index) for index in range(num_layers)]
        )

    def forward(self, *args, **kwargs):
        raise AssertionError("The tensor-level unit test calls decoder layers directly")


class _TrainingThinker(_Thinker):
    def __init__(self, hidden_size: int = 12, num_layers: int = 4) -> None:
        super().__init__(hidden_size=hidden_size, num_layers=num_layers)
        self.lm_head = torch.nn.Linear(hidden_size, 16, bias=False)
        self.config = SimpleNamespace(
            get_text_config=lambda: SimpleNamespace(vocab_size=16)
        )

    def loss_function(self, *, logits, labels, vocab_size):
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab_size),
            labels.reshape(-1),
            ignore_index=-100,
        )

    def forward(
        self,
        *,
        inputs_embeds,
        attention_mask=None,
        position_embeddings=None,
        cache_position=None,
        labels=None,
        **kwargs,
    ):
        del kwargs
        hidden_states = inputs_embeds
        for layer in self.model.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
                cache_position=cache_position,
            )[0]
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.get_text_config().vocab_size,
            )
        return SimpleNamespace(logits=logits, loss=loss)


class _SwiftOuterModel(torch.nn.Module):
    def __init__(self, thinker: _TrainingThinker) -> None:
        super().__init__()
        self.thinker = thinker


@dataclass
class _TransformersTrainingOutput(ModelOutput):
    loss: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    rope_deltas: torch.Tensor | None = None


def _stage(phase: str = "forward") -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "pyramid",
            "placement": "llm_level",
            "execution_phase": phase,
            "plugin": "audio_pyramid_attention_pruning",
            "algorithm": "audio_pyramid_attention_pruning",
            "variant": "audio_pyramid_attention_pruning.audio",
            "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "llm_decoder_layer_output_audio_tokens",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "token",
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "llm_decoder_layer_output_audio_tokens",
            },
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
            "activation_scope": {"layers": [0, 1, 2], "lifecycle": "prefill_only"},
        }
    )


def _create_controller(
    model: object,
    stage: CompressionStageConfig,
    *,
    profile_enabled: bool = False,
    debug_enabled: bool = False,
) -> object:
    execution = resolve_qwen25_omni_stage_execution(stage)
    runtime_mode = (
        RuntimeMode.DEBUG
        if debug_enabled
        else (RuntimeMode.PROFILE if profile_enabled else RuntimeMode.DEFAULT)
    )
    return create_qwen25_omni_dynamic_controller(
        model=model,
        stage=stage,
        binding=execution.binding,
        runtime_mode=runtime_mode,
    )


def _state() -> ForwardState:
    return ForwardState(
        input_ids=torch.tensor([[10, 99, 99, 99, 99, 11, 12]]),
        attention_mask=torch.ones((1, 7), dtype=torch.long),
        labels=torch.tensor([[-100, -100, -100, -100, -100, -100, 12]]),
        modality_token_spans=[
            TokenSpan(0, 1, 5, modality="audio", role="observation")
        ],
        modality_metadata={
            "semantic_token_labels_by_sample": [
                [
                    "ins_special",
                    "audio",
                    "audio",
                    "audio",
                    "audio",
                    "instruction",
                    "response",
                ]
            ]
        },
        native_model_inputs={},
    )


def _causal_mask(length: int) -> torch.Tensor:
    value = torch.full((length, length), torch.finfo(torch.float32).min)
    value = torch.triu(value, diagonal=1)
    return value[None, None]


@pytest.mark.parametrize("merge", [False, True])
def test_ragged_audio_repacking_preserves_text_positions_and_gradients(merge):
    from mllm_comp.backbones.qwen25_omni.llm_token_compression import (
        _compact_hidden_and_positions, _replace_audio_units,
    )

    hidden = torch.arange(16, dtype=torch.float32).reshape(2, 8, 1).requires_grad_()
    positions = torch.arange(8).expand(2, -1)
    valid = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1], [1, 1, 1, 0, 0, 0, 0, 0]], dtype=torch.bool)
    if merge:
        # Two audio units arrive in reverse order; the short row contains text only.
        replacements = {0: [
            {"slots": torch.tensor([start, start + 1]),
             "hidden_states": hidden[0, start:start + 2].mean(0, keepdim=True),
             "positions": positions[0, start:start + 1]}
            for start in (5, 2)
        ]}
        output, actual_positions = _replace_audio_units(hidden, positions, valid, replacements)
    else:
        keep = valid.clone()
        keep[0, [3, 6]] = False
        output, actual_positions = _compact_hidden_and_positions(hidden, positions, keep)

    torch.testing.assert_close(output.squeeze(-1), torch.tensor([
        [1, 2.5 if merge else 2, 4, 5.5 if merge else 5, 7], [8, 9, 10, 0, 0],
    ], dtype=hidden.dtype))
    assert actual_positions.tolist() == [[1, 2, 4, 5, 7], [0, 1, 2, -1, -1]]
    output.sum().backward()
    audio_gradient = [0.5, 0.5] if merge else [1, 0]
    torch.testing.assert_close(hidden.grad.squeeze(-1), torch.tensor([
        [0, 1, *audio_gradient, 1, *audio_gradient, 1], [1, 1, 1, 0, 0, 0, 0, 0],
    ], dtype=hidden.dtype))


@pytest.mark.parametrize(
    "layout", ["sequence", "shared_batch", "batch", "head", "mrope"]
)
def test_audio_rotary_layouts_preserve_values_and_gradients(layout: str) -> None:
    from mllm_comp.compressors.llm_level._rotary import apply_rotary_position_embeddings

    torch.manual_seed(23)
    batch, length, width = 2, 5, 8
    query = torch.randn(batch, 4, length, width, requires_grad=True)
    key = torch.randn(batch, 2, length, width, requires_grad=True)
    angles = torch.randn(3, batch, length, width, requires_grad=True)
    cos, sin = angles.cos(), angles.sin()
    if layout == "mrope":
        # Sections [1, 2, 1] select temporal, height and width axes twice.
        axes = [0, 1, 1, 2, 0, 1, 1, 2]
        expected_cos, expected_sin = (
            torch.stack(
                [value[axis, :, :, column] for column, axis in enumerate(axes)], dim=-1
            )
            for value in (cos, sin)
        )
    else:
        cos, sin = cos[0], sin[0]
        if layout in {"sequence", "shared_batch"}:
            cos, sin = cos[:1], sin[:1]
        expected_cos, expected_sin = cos, sin
        if layout == "sequence":
            cos, sin = cos[0], sin[0]
        elif layout == "head":
            cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
    attention = SimpleNamespace(
        config=SimpleNamespace(rope_scaling={"mrope_section": [1, 2, 1]})
    )
    actual = apply_rotary_position_embeddings(
        query, key, position_embeddings=(cos, sin), attention_module=attention,
        batch_size=batch, sequence_length=length,
    )
    expected = []
    for value in (query, key):
        left, right = value[..., :width // 2], value[..., width // 2:]
        rotated = torch.cat((-right, left), dim=-1)
        expected.append(
            value * expected_cos.unsqueeze(1) + rotated * expected_sin.unsqueeze(1)
        )
    for output, reference in zip(actual, expected):
        torch.testing.assert_close(output, reference)
    gradients = torch.autograd.grad(
        sum(item.square().sum() for item in actual), (query, key, angles),
        retain_graph=True,
    )
    references = torch.autograd.grad(
        sum(item.square().sum() for item in expected), (query, key, angles)
    )
    for gradient, reference in zip(gradients, references):
        torch.testing.assert_close(gradient, reference)


def test_audio_pyramid_prefill_prunes_and_decode_bypasses_with_gradients() -> None:
    thinker = _Thinker()
    model = SimpleNamespace(thinker=thinker)
    controller = _create_controller(model, _stage())
    controller.install()
    state, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)
    original_mask = _causal_mask(7)
    original_rope = (
        torch.ones(3, 1, 7, 6),
        torch.zeros(3, 1, 7, 6),
    )

    with controller.request(plan):
        active = hidden
        for layer in thinker.model.layers:
            active = layer(
                active,
                attention_mask=original_mask,
                position_embeddings=original_rope,
                cache_position=torch.arange(7),
            )[0]
        assert active.shape[1] == 4  # 3 text/special + one retained audio token
        active.square().mean().backward()
        assert hidden.grad is not None
        assert torch.isfinite(hidden.grad).all()
        assert hidden.grad.abs().sum() > 0

        decode = torch.randn(1, 1, 12)
        decode_mask = torch.zeros((1, 1, 1, 8))
        decode_rope = (torch.ones(3, 1, 1, 6), torch.zeros(3, 1, 1, 6))
        for layer in thinker.model.layers:
            decode = layer(
                decode,
                attention_mask=decode_mask,
                position_embeddings=decode_rope,
                cache_position=torch.tensor([7]),
            )[0]
        assert decode.shape == (1, 1, 12)

    report = controller.last_report
    assert report["original_audio_tokens"] == 4
    assert report["kept_audio_tokens"] == 1
    assert report["prefill_forward_calls"] == 1
    assert report["decode_forward_calls"] == 1
    assert report["decode_bypassed"] is True
    assert report["selection_repeated_during_decode"] is False
    assert [item["boundary_one_based"] for item in report["stages"]] == [1, 2, 3]
    assert len(controller.last_token_layouts[0][0]) == 7
    assert controller.last_token_layouts[1][0].count("audio") == 2
    assert controller.last_token_layouts[2][0].count("audio") == 1
    controller.uninstall()


def test_audio_pyramid_prefill_is_safe_under_bfloat16_autocast() -> None:
    thinker = _Thinker()
    controller = _create_controller(
        SimpleNamespace(thinker=thinker), _stage()
    )
    controller.install()
    _, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12)

    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        with controller.request(plan):
            active = hidden
            for layer in thinker.model.layers:
                active = layer(
                    active,
                    attention_mask=_causal_mask(7),
                    position_embeddings=(
                        torch.ones(3, 1, 7, 6),
                        torch.zeros(3, 1, 7, 6),
                    ),
                    cache_position=torch.arange(7),
                )[0]

    assert active.shape[1] == 4
    assert torch.isfinite(active).all()
    controller.uninstall()


def test_swift_forward_refresh_compacts_labels_after_dynamic_llm_pruning() -> None:
    swift_utils = pytest.importorskip("swift.model.utils")
    thinker = _TrainingThinker()
    model = _SwiftOuterModel(thinker)

    # Reproduce ms-swift loading order: it first captures the provider forward,
    # then the compression controller patches the Thinker instance.
    swift_utils.use_submodel_func(model, "thinker", func_list=["forward"])
    controller = _create_controller(model, _stage())
    controller.install()

    # This is the integration fix: the outer forward must capture the patched
    # Thinker method, which removes original labels and gathers them only after
    # the final token layout is known.
    swift_utils.use_submodel_func(model, "thinker", func_list=["forward"])
    _, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)

    with controller.request(plan):
        output = model(
            inputs_embeds=hidden,
            attention_mask=_causal_mask(7),
            position_embeddings=(
                torch.ones(3, 1, 7, 6),
                torch.zeros(3, 1, 7, 6),
            ),
            cache_position=torch.arange(7),
            labels=_state().labels,
        )

    assert output.logits.shape[:2] == (1, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    controller.uninstall()


def test_swift_forward_uses_request_labels_when_outer_wrapper_consumes_them() -> None:
    swift_utils = pytest.importorskip("swift.model.utils")
    thinker = _TrainingThinker()
    model = _SwiftOuterModel(thinker)
    controller = _create_controller(model, _stage())
    controller.install()
    swift_utils.use_submodel_func(model, "thinker", func_list=["forward"])
    _, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)

    # The template/PEFT layer can keep labels for Trainer bookkeeping while
    # omitting them from the inner Thinker call.  Loss must still use the labels
    # retained by the active compression request and its final token layout.
    with controller.request(plan):
        output = model(
            inputs_embeds=hidden,
            attention_mask=_causal_mask(7),
            position_embeddings=(
                torch.ones(3, 1, 7, 6),
                torch.zeros(3, 1, 7, 6),
            ),
            cache_position=torch.arange(7),
        )

    assert output.logits.shape[:2] == (1, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    controller.uninstall()


def test_bridge_can_attach_loss_when_provider_bypasses_thinker_wrapper() -> None:
    thinker = _TrainingThinker()
    model = _SwiftOuterModel(thinker)
    provider_forward = thinker.forward
    controller = _create_controller(model, _stage())
    controller.install()
    state, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)

    # Reproduce the real Swift/PEFT path: decoder layer hooks still compact the
    # sequence, but a previously captured provider forward bypasses the patched
    # Thinker wrapper and therefore returns no loss.
    with controller.request(plan):
        output = provider_forward(
            inputs_embeds=hidden,
            attention_mask=_causal_mask(7),
            position_embeddings=(
                torch.ones(3, 1, 7, 6),
                torch.zeros(3, 1, 7, 6),
            ),
            cache_position=torch.arange(7),
        )
        assert output.loss is None
        output = controller.attach_compact_training_loss(
            output,
            labels=state.labels,
        )

    assert output.logits.shape[:2] == (1, 4)
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    controller.uninstall()


def test_bridge_can_restore_loss_after_request_completion() -> None:
    thinker = _TrainingThinker()
    model = _SwiftOuterModel(thinker)
    provider_forward = thinker.forward
    controller = _create_controller(model, _stage())
    controller.install()
    state, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)

    with controller.request(plan):
        output = provider_forward(
            inputs_embeds=hidden,
            attention_mask=_causal_mask(7),
            position_embeddings=(
                torch.ones(3, 1, 7, 6),
                torch.zeros(3, 1, 7, 6),
            ),
            cache_position=torch.arange(7),
        )

    assert output.loss is None
    output = controller.attach_compact_training_loss(
        output,
        labels=state.labels,
    )
    assert output.loss is not None and torch.isfinite(output.loss)
    output.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    controller.uninstall()


def test_bridge_inserts_loss_into_transformers_model_output_mapping() -> None:
    thinker = _TrainingThinker()
    model = _SwiftOuterModel(thinker)
    provider_forward = thinker.forward
    controller = _create_controller(model, _stage())
    controller.install()
    state, plan = controller.prepare_state(_state())
    hidden = torch.randn(1, 7, 12, requires_grad=True)

    with controller.request(plan):
        provider_output = provider_forward(
            inputs_embeds=hidden,
            attention_mask=_causal_mask(7),
            position_embeddings=(
                torch.ones(3, 1, 7, 6),
                torch.zeros(3, 1, 7, 6),
            ),
            cache_position=torch.arange(7),
        )

    output = _TransformersTrainingOutput(
        logits=provider_output.logits,
        rope_deltas=torch.zeros(1),
    )
    assert output.loss is None
    assert "loss" not in output

    output = controller.attach_compact_training_loss(
        output,
        labels=state.labels,
    )

    assert output.loss is not None and torch.isfinite(output.loss)
    assert "loss" in output
    assert output["loss"] is output.loss
    output.loss.backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    controller.uninstall()


def test_swift_template_completes_a_pending_compressed_forward(tmp_path) -> None:
    script = textwrap.dedent(
        """
        import base64
        import json
        import os
        from dataclasses import dataclass
        from types import SimpleNamespace

        import torch
        from transformers.utils import ModelOutput

        stage = {
            "name": "pyramid",
            "placement": "llm_level",
            "execution_phase": "forward",
            "plugin": "audio_pyramid_attention_pruning",
            "algorithm": "audio_pyramid_attention_pruning",
            "variant": "audio_pyramid_attention_pruning.audio",
            "binding": "audio_pyramid_attention_pruning.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "llm_decoder_layer_output_audio_tokens",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "token",
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "llm_decoder_layer_output_audio_tokens",
            },
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
            "activation_scope": {
                "layers": [0, 1, 2],
                "lifecycle": "prefill_only",
            },
        }
        payload = {
            "schema_version": 2,
            "runtime_mode": "default",
            "executor_family": "llm_token",
            "stage": stage,
        }
        raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        os.environ["MLLM_COMP_TRAIN_COMPRESSION_SPEC"] = (
            base64.urlsafe_b64encode(raw).decode("ascii")
        )
        os.environ["MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR"] = os.environ[
            "TEST_EVIDENCE_DIR"
        ]

        from mllm_comp.core.vocabulary import RuntimeMode
        from mllm_comp.integrations.ms_swift import training_plugin as plugin
        from swift.template.templates.qwen import Qwen2_5OmniTemplate

        class Stack:
            closed = False

            def close(self):
                self.closed = True

        class Controller:
            execution = SimpleNamespace(executor_family="llm_token")
            last_report = {}

            def compact_training_labels(self):
                return labels

            def attach_compact_training_loss(self, output, *, labels=None):
                # The controller owns labels already compacted by upstream stages.
                assert labels is None
                output.loss = output.logits.float().sum()
                return output

            def prepare_state(self, state):
                return state, object()

        @dataclass
        class Output(ModelOutput):
            loss: torch.Tensor | None = None
            logits: torch.Tensor | None = None
            rope_deltas: torch.Tensor | None = None

        class Model:
            training = True

            def __init__(self):
                self._mllm_comp_training_session = SimpleNamespace(
                    active_records=[SimpleNamespace(activation=Controller())]
                )
                self._mllm_comp_training_runtime_mode = RuntimeMode.DEFAULT
                self._mllm_comp_training_debug_collector = None
                self._mllm_comp_training_recorder = object()

            def __call__(self, **inputs):
                del inputs
                return Output(
                    logits=torch.ones(1, 2, 4, requires_grad=True),
                    rope_deltas=torch.zeros(1),
                )

        model = Model()
        labels = torch.tensor([[-100, 1]])
        active_state = SimpleNamespace(
            labels=labels,
            modality_metadata={"audio_compression": {}},
        )
        stack = Stack()
        pending = {
            "stack": stack,
            "active_state": active_state,
            "model_training": False,
            "timing": None,
        }
        setattr(model, plugin._PENDING_FORWARD, pending)
        template = SimpleNamespace(get_base_model=lambda value: value)

        outputs = Qwen2_5OmniTemplate.compute_sft_loss(
            template,
            model,
            {"labels": labels},
        )
        assert outputs.loss is not None
        assert "loss" in outputs
        assert stack.closed is True
        assert not hasattr(model, plugin._PENDING_FORWARD)
        outputs.loss.backward()

        # Reproduce the observed real path: an inner bridge already consumed the
        # pending request, then an outer wrapper returned a ModelOutput whose loss
        # field was empty.  The template's final seam must restore it directly.
        completed_model = Model()
        completed_outputs = Qwen2_5OmniTemplate.compute_sft_loss(
            template,
            completed_model,
            {"labels": labels},
        )
        assert completed_outputs.loss is not None
        assert "loss" in completed_outputs
        completed_outputs.loss.backward()
        """
    )
    environment = dict(os.environ)
    environment["TEST_EVIDENCE_DIR"] = str(tmp_path)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
