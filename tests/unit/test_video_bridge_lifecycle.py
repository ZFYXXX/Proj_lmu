"""Shared projector callback lifetime, ownership, and partial-failure cleanup."""

from types import SimpleNamespace

import pytest

from mllm_comp.backbones import vqtoken_video, video_projector
from mllm_comp.core.errors import BackendIntegrationError


@pytest.mark.parametrize("owned", (False, True))
@pytest.mark.parametrize("with_provider", (False, True))
def test_projector_lifecycle_orders_provider_callback_and_owned_bridge(
    monkeypatch, owned, with_provider
):
    events = []
    provider = SimpleNamespace(
        install=lambda: events.append("provider_install"),
        uninstall=lambda: events.append("provider_uninstall"),
        runtime_metadata=lambda: {"source": "visual"},
    ) if with_provider else None

    def register(site, callback, *, signal_provider=None):
        assert site == "projector"
        assert callback == executor._projector_callback
        assert signal_provider is provider
        events.append("register")
        return SimpleNamespace(remove=lambda: events.append("remove"))

    bridge = SimpleNamespace(
        register_callback=register,
        detach=lambda: events.append("detach"),
        runtime_metadata=lambda: {"site": "projector"},
    )

    def install_bridge(model, backbone):
        assert model is executor.model
        assert backbone == "qwen3_vl"
        events.append("bridge_install")
        return bridge, owned, "projector"

    monkeypatch.setattr(video_projector, "install_video_bridge", install_bridge)
    projection = video_projector.builtin_video_projector_projection(
        algorithm_name="vqtoken", backbone="qwen3_vl", stage=None,
        binding=None, config={},
    )
    executor = video_projector.VideoProjectorExecutor(
        backbone="qwen3_vl", model=object(), contract_projection=projection,
        binding=projection.bindings[0], signal_provider=provider,
    )
    executor.install()
    executor.install()
    assert executor.installed
    assert executor.runtime_metadata()["bridge"] == {"site": "projector"}
    if with_provider:
        assert executor.runtime_metadata()["signal_provider"] == {"source": "visual"}
    executor.uninstall()
    executor.uninstall()
    assert not executor.installed
    assert events == (
        (["provider_install"] if with_provider else [])
        + ["bridge_install", "register", "remove"]
        + (["detach"] if owned else [])
        + (["provider_uninstall"] if with_provider else [])
    )


@pytest.mark.parametrize("owned", (False, True))
def test_uninstall_attempts_owned_bridge_after_callback_removal_fails(
    monkeypatch, owned
):
    events = []

    def remove():
        events.append("remove")
        raise RuntimeError("callback cleanup failed")

    def detach():
        events.append("detach")
        raise RuntimeError("bridge cleanup failed")

    bridge = SimpleNamespace(
        register_callback=lambda *args, **kwargs: SimpleNamespace(remove=remove),
        detach=detach,
    )
    monkeypatch.setattr(
        video_projector, "install_video_bridge", lambda *args: (bridge, owned, "site")
    )
    executor = vqtoken_video.VQTokenVideoExecutor(backbone="qwen3_vl", model=object())
    executor.install()
    with pytest.raises(
        BackendIntegrationError, match="callback cleanup failed"
    ) as error:
        executor.uninstall()
    assert events == (["remove", "detach"] if owned else ["remove"])
    assert ("bridge cleanup failed" in str(error.value)) is owned
    assert not executor.installed
    executor.uninstall()
    assert len(events) == (2 if owned else 1)


@pytest.mark.parametrize("owned", (False, True))
@pytest.mark.parametrize("invalid_handle", (False, True))
def test_registration_failure_releases_only_owned_bridge(
    monkeypatch, owned, invalid_handle
):
    events = []

    def register(*args, **kwargs):
        if invalid_handle:
            return object()
        raise RuntimeError("registration failed")

    bridge = SimpleNamespace(
        register_callback=register, detach=lambda: events.append("detach")
    )
    monkeypatch.setattr(
        video_projector, "install_video_bridge", lambda *args: (bridge, owned, "site")
    )
    executor = vqtoken_video.VQTokenVideoExecutor(backbone="qwen3_vl", model=object())
    message = "non-removable handle" if invalid_handle else "registration failed"
    with pytest.raises((BackendIntegrationError, RuntimeError), match=message):
        executor.install()
    assert events == (["detach"] if owned else [])
    assert not executor.installed
