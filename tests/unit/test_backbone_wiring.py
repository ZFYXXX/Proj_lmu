"""Shared wiring contracts; no model weights or tensor runtime required."""

from importlib import import_module
import inspect
from types import MethodType, SimpleNamespace

import pytest

from mllm_comp.backbones._provider_patching import MethodPatch
from mllm_comp.backbones.fastvid_video import FastVIDVideoExecutor
from mllm_comp.backbones.vqtoken_video import VQTokenVideoExecutor
from mllm_comp.backbones.qwen25_omni.audio_compression import Qwen25OmniAudioCompressionController
from mllm_comp.backbones.qwen25_omni.encoder_compression import Qwen25OmniEncoderCompressionController
from mllm_comp.core.errors import BackendIntegrationError


class _Model:
    def forward(self, input_ids, *, position_ids=None):
        return input_ids, position_ids


@pytest.mark.parametrize("instance_override", [False, True])
def test_method_patch_preserves_signature_binding_and_original_attribute(instance_override):
    model = _Model()
    if instance_override:
        model.forward = MethodType(lambda self, input_ids, *, position_ids=None: (position_ids, input_ids), model)
    original = model.forward

    def replacement(self, *args, **kwargs):
        assert self is model
        return original(*args, **kwargs)

    patch = MethodPatch.apply(model, "forward", replacement, original, label="Fixture")
    assert inspect.signature(model.forward) == inspect.signature(original)
    assert model.forward([1], position_ids=[2]) == original([1], position_ids=[2])
    # Providers can rebind the same wrapper; restoration still owns that patch.
    model.forward = MethodType(replacement, model)
    patch.restore()
    assert ("forward" in vars(model)) is instance_override
    assert model.forward == original


def test_method_patch_does_not_overwrite_a_replacement_owned_by_someone_else():
    model = _Model()
    patch = MethodPatch.apply(model, "forward", lambda self, **kwargs: None, model.forward, label="Fixture")
    other = MethodType(lambda self, *args, **kwargs: "other", model)
    model.forward = other
    with pytest.raises(BackendIntegrationError, match="Fixture patched method 'forward' changed while active"):
        patch.restore()
    assert model.forward is other


_BRIDGES = [
    ("llava_onevision", "LLAVA_ONEVISION", ("mllm_video_compression_bridge", "_mllm_video_compression_bridge")),
    ("qwen25_vl", "QWEN25_VL", ("_mllm_comp_qwen25_vl_video_bridge", "mllm_video_compression_bridge")),
    ("qwen3_vl", "QWEN3_VL", ("_mllm_comp_qwen3_vl_video_bridge", "mllm_video_compression_bridge")),
]


@pytest.mark.parametrize("executor_type", [VQTokenVideoExecutor, FastVIDVideoExecutor])
@pytest.mark.parametrize("backbone,prefix,aliases", _BRIDGES)
@pytest.mark.parametrize("existing_alias", [None, 0, 1])
@pytest.mark.parametrize("registration_fails", [False, True])
def test_native_bridge_selection_ownership_and_executor_rollback(
    monkeypatch, executor_type, backbone, prefix, aliases, existing_alias, registration_fails,
):
    native = import_module(f"mllm_comp.backbones.{backbone}.video_compression")
    expected_site = getattr(native, f"{prefix}_VIDEO_SPLICE_SITE")
    events = []
    model = _Model()
    original_forward = model.forward
    executor = executor_type(backbone=backbone, model=model)

    class Provider:
        def install(self):
            events.append("provider_install")
            self.patch = MethodPatch.apply(
                model, "forward", lambda self, *args, **kwargs: original_forward(*args, **kwargs),
                original_forward, label="Fixture",
            )

        def uninstall(self):
            self.patch.restore()
            events.append("provider_uninstall")

    if executor_type is FastVIDVideoExecutor:
        executor.signal_provider = Provider()

    class Bridge:
        def register_callback(self, site, callback, **kwargs):
            assert site == expected_site
            assert callback == executor._projector_callback
            if executor_type is FastVIDVideoExecutor:
                assert kwargs["signal_provider"] is executor.signal_provider
            events.append("register")
            if registration_fails:
                raise RuntimeError("callback registration failed")
            return SimpleNamespace(remove=lambda: events.append("remove"))

        def detach(self):
            events.append("detach")

    bridge = Bridge()
    if existing_alias is not None:
        setattr(model, aliases[existing_alias], bridge)

    def installer(received):
        assert received is model
        events.append("install")
        setattr(model, aliases[0], bridge)
        return bridge

    # Exercise the real shared router, stubbing only the native model installer.
    monkeypatch.setattr(native, f"install_{backbone}_video_bridge", installer)
    if registration_fails:
        with pytest.raises(RuntimeError, match="callback registration failed"):
            executor.install()
    else:
        executor.install()
        executor.install()
        assert executor.installed
        executor.uninstall()
    executor.uninstall()
    expected = ["install", "register"]
    if not registration_fails:
        expected.append("remove")
    if existing_alias is None:
        expected.append("detach")
    if executor_type is FastVIDVideoExecutor:
        expected = ["provider_install", *expected, "provider_uninstall"]
    assert events == expected
    assert not executor.installed
    assert model.forward == original_forward and "forward" not in vars(model)
    if existing_alias is not None:
        assert getattr(model, aliases[existing_alias]) is bridge


@pytest.mark.parametrize("controller_type", [Qwen25OmniAudioCompressionController, Qwen25OmniEncoderCompressionController])
@pytest.mark.parametrize("request_kind", ["compressed", "identity", "empty", "inactive"])
@pytest.mark.parametrize("argument_form", ["positional", "keyword", "none", "absent"])
def test_omni_rope_conversion_preserves_arguments_and_request_identity(
    controller_type, request_kind, argument_form,
):
    controller = controller_type.__new__(controller_type)
    request = SimpleNamespace(identity=request_kind == "identity", keep_counts=[] if request_kind == "empty" else [3, 5])
    controller._active_request = None if request_kind == "inactive" else request
    calls = []
    result = object()

    def original(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    controller._original_get_rope_index = original
    torch = pytest.importorskip("torch")
    lengths = torch.tensor([80, 100])
    mask = object()
    args = tuple(object() for _ in range(5))
    kwargs = {"attention_mask": mask, "use_audio_in_video": True}
    if argument_form == "positional":
        args = (*args, lengths, "trailing argument")
    elif argument_form != "absent":
        kwargs["audio_seqlens"] = None if argument_form == "none" else lengths
    assert controller._compressed_rope_index(object(), *args, **kwargs) is result
    assert len(calls) == 1
    received_args, received_kwargs = calls[0]
    assert received_args[:5] == args[:5]
    assert received_kwargs["attention_mask"] is mask
    assert received_kwargs["use_audio_in_video"] is True
    if argument_form == "positional":
        converted = received_args[5]
        assert received_args[6:] == args[6:]
    else:
        converted = received_kwargs.get("audio_seqlens")
        assert ("audio_seqlens" in received_kwargs) == (argument_form != "absent")
    if argument_form in {"positional", "keyword"}:
        if request_kind == "compressed":
            assert converted.tolist() == [12, 20]
            assert converted is not lengths
        else:
            assert converted is lengths
    else:
        assert converted is None
    assert lengths.tolist() == [80, 100]
    if "audio_seqlens" in kwargs:
        assert kwargs["audio_seqlens"] is (None if argument_form == "none" else lengths)
    assert controller._active_request is (None if request_kind == "inactive" else request)
