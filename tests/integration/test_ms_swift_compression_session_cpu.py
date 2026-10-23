"""Real TinyOmni forwards through the Swift bridge; no dataset/Trainer simulation.

The native backend still owns checkpoint serialization and optimizer restoration.
These tests exercise its model boundary with real tensors, autograd and checkpoint
replay; they do not claim to run a complete Swift CLI training job.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.pipeline.stage import CompressionStageConfig
from tests.fixtures.qwen25_omni_audio import _operation_stage, llm_stage, _tiny_model


def _plugin(monkeypatch):
    pytest.importorskip("swift")
    from swift.model.models.qwen import Qwen2_5OmniLoader

    monkeypatch.setattr(Qwen2_5OmniLoader, "_mllm_comp_training_loader_patch", True, raising=False)
    path = Path(__file__).parents[2] / "mllm_comp/integrations/ms_swift/training_plugin.py"
    spec = importlib.util.spec_from_file_location("_swift_session_cpu", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    from swift.template.templates.qwen import Qwen2_5OmniTemplate
    from swift.trainers import Seq2SeqTrainer
    monkeypatch.setattr(Seq2SeqTrainer, "prediction_step", Seq2SeqTrainer.prediction_step)
    # Restore the real class after each test, including inherited methods.
    for name in ("pre_forward_hook", "compute_sft_loss"):
        monkeypatch.setattr(Qwen2_5OmniTemplate, name, getattr(Qwen2_5OmniTemplate, name))
    monkeypatch.setattr(Qwen2_5OmniTemplate, plugin._TEMPLATE_PATCH_MARKER, False, raising=False)
    plugin._patch_qwen_template()
    return plugin


def _projector_stage():
    return CompressionStageConfig.from_dict({
        "name": "projector", "placement": "projector_level", "execution_phase": "forward",
        "plugin": "uniform_sampling_audio", "algorithm": "uniform_sampling_audio",
        "variant": "uniform_sampling_audio.audio", "binding": "uniform_sampling_audio.audio.qwen25_omni",
        "target_modalities": ["audio"], "preserved_modalities": ["text", "image", "video"],
        "media_alignment": "temporal",
        "operation": {"type": "keep", "target_site": "audio_encoder_output_pre_projector"},
        "schedule": {"compression_factor": 2},
    })


@pytest.mark.parametrize("projector,runtime_mode", [
    (False, "default"), (True, "default"), (True, "profile"), (True, "debug"),
])
def test_prepare_and_dynamic_stages_train_resume_and_release_same_model(
    tmp_path, monkeypatch, projector, runtime_mode,
):
    torch = pytest.importorskip("torch")
    plugin = _plugin(monkeypatch)
    torch.manual_seed(184)
    original = _tiny_model(torch).eval()
    operation = _operation_stage()
    token = llm_stage(phase="forward")
    token.activation_scope["layers"] = [0]
    stages = [token, operation]
    if projector:
        stages.insert(0, _projector_stage())
    ids = torch.tensor([[1, 104, 102, 99, 99, 99, 99, 103, 2, 127, 3]])
    labels = torch.full_like(ids, -100)
    labels[:, -1] = 3
    features = torch.randn(1, 4, 16)

    def install(folder):
        monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR", str(folder))
        model = copy.deepcopy(original)
        # Swift's native loader binds the outer call to its Thinker.
        model.forward = model.thinker.forward
        model.device = torch.device("cpu")
        native_forward = model.forward
        native_decoder = model.thinker.model.forward
        source_layers = tuple(model.thinker.model.layers)
        recorder = plugin._EvidenceRecorder(
            spec_hash="session-cpu", stage=stages[0], runtime_mode=RuntimeMode(runtime_mode),
        )
        plugin._install_model_bridge(model, stages=stages, runtime_mode=RuntimeMode(runtime_mode), recorder=recorder)
        assert tuple(model.thinker.model.layers) == (source_layers[0], source_layers[2])
        assert model._mllm_comp_training_session.install_order == (
            (operation.name, "projector", token.name) if projector else (operation.name, token.name)
        )
        assert model._mllm_comp_training_session.active_records[0].activation.phase == "train"
        model.train()
        model.thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Native tuner/optimizer construction happens after the loader returns.
        optimizer = torch.optim.SGD(model.parameters(), lr=0.03, momentum=0.9)
        removed_ids = {id(p) for layer in (source_layers[1], source_layers[3]) for p in layer.parameters()}
        assert not removed_ids.intersection(id(p) for group in optimizer.param_groups for p in group["params"])
        from swift.template.templates.qwen import Qwen2_5OmniTemplate
        from transformers import Qwen2VLImageProcessor
        template = Qwen2_5OmniTemplate.__new__(Qwen2_5OmniTemplate)
        template.mode = "train"
        template.processor = SimpleNamespace(image_processor=Qwen2VLImageProcessor(
            patch_size=2, temporal_patch_size=1, merge_size=1, min_pixels=16, max_pixels=16,
        ))
        template.use_audio_in_video = False
        template.get_base_model = lambda value: value
        template._get_get_rope_index = lambda: model.thinker.get_rope_index
        handle = model.register_forward_pre_hook(template.pre_forward_hook, with_kwargs=True)
        return model, optimizer, recorder, native_forward, native_decoder, template, handle

    def step(model, optimizer, template):
        optimizer.zero_grad(set_to_none=True)
        audio = features.clone().requires_grad_()
        inputs = {
            "input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels,
            "input_features": audio, "feature_attention_mask": torch.ones((1, 16), dtype=torch.long),
            "use_cache": False,
        }
        output = template.compute_sft_loss(model, inputs)
        assert output.logits.shape == (1, 8 if projector else 9, 128)
        assert inputs["labels"].shape == output.logits.shape[:2]
        assert inputs["labels"][0, -1] == 3
        assert not hasattr(model, plugin._LABEL_CAPTURE)
        # Independent causal-label check, including upstream projector compaction.
        expected_loss = torch.nn.functional.cross_entropy(output.logits[:, -2].float(), labels[:, -1])
        torch.testing.assert_close(output.loss, expected_loss)
        output.loss.backward()
        assert torch.isfinite(audio.grad).all() and audio.grad.count_nonzero() > 0
        optimizer.step()
        return output.loss.detach().clone()

    model, optimizer, recorder, native_forward, native_decoder, template, handle = install(tmp_path / "first")
    try:
        step(model, optimizer, template)
        # The backend saves its own model/optimizer state before bridge cleanup.
        checkpoint = tmp_path / "checkpoint.pt"
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint)
        expected_loss = step(model, optimizer, template)
        expected_weights = {name: value.detach().clone() for name, value in model.state_dict().items()}
        records = [json.loads(line) for line in recorder.path.read_text(encoding="utf-8").splitlines()]
        for stage in stages:
            events = [record for record in records if record["stage_name"] == stage.name]
            assert {record["event"] for record in events} == {"install", "forward", "backward"}
            forwards = [record for record in events if record["event"] == "forward"]
            assert len(forwards) == (1 if runtime_mode == "default" else 2)
            backwards = {record["forward_index"] for record in events if record["event"] == "backward"}
            assert all(record["forward_index"] in backwards for record in forwards)
        token_report = next(record for record in records if record["event"] == "forward" and record["stage_name"] == token.name)
        assert token_report["compression_state"]["original_audio_tokens"] == (2 if projector else 4)
        assert token_report["compression"]["kept_audio_tokens"] == (1 if projector else 2)
        if runtime_mode == "debug":
            assert token_report["token_debug"]["layers"]
        if runtime_mode == "profile":
            assert token_report["timing"]["forward_seconds"] >= 0
    finally:
        handle.remove()
        plugin._close_model_bridge(model)
    assert model.forward == native_forward
    assert model.thinker.model.forward == native_decoder
    assert not hasattr(model, "_mllm_comp_training_session")
    assert len(model.thinker.model.layers) == 2

    restored, optimizer, _, _, _, template, handle = install(tmp_path / "restored")
    try:
        # Full resume rebuilds the original pipeline, then the native backend
        # loads checkpoint values into those modules and restores its optimizer.
        saved = torch.load(checkpoint, weights_only=True)
        restored.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        torch.testing.assert_close(step(restored, optimizer, template), expected_loss, rtol=0, atol=0)
        for name, value in restored.state_dict().items():
            torch.testing.assert_close(value, expected_weights[name], rtol=0, atol=0)
    finally:
        handle.remove()
        plugin._close_model_bridge(restored)


def test_failed_native_forward_closes_requests_and_allows_next_batch(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    plugin = _plugin(monkeypatch)
    monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR", str(tmp_path))
    model = _tiny_model(torch).train()
    model.forward = model.thinker.forward
    token = llm_stage(phase="forward")
    token.activation_scope["layers"] = [0]
    plugin._install_model_bridge(
        model, stages=[_operation_stage(), token], runtime_mode=RuntimeMode.DEFAULT,
        recorder=plugin._EvidenceRecorder(spec_hash="failure", stage=token, runtime_mode=RuntimeMode.DEFAULT),
    )
    ids = torch.tensor([[1, 104, 99, 99, 2, 127, 3]])
    inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids),
              "labels": torch.tensor([[-100] * 6 + [3]]), "use_cache": False}

    def fail(module, args):
        raise RuntimeError("native layer failed")

    hook = model.thinker.model.layers[0].register_forward_pre_hook(fail)
    try:
        with pytest.raises(RuntimeError, match="native layer failed"):
            model(**inputs)
        assert not hasattr(model, plugin._PENDING_FORWARD)
        hook.remove()
        output = model(**inputs)
        assert torch.isfinite(output.loss)
        output.loss.backward()
    finally:
        hook.remove()
        plugin._close_model_bridge(model)
