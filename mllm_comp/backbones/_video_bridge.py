"""Select the native video bridge and splice site shared by video executors."""

from typing import Any

from mllm_comp.core.errors import BackendIntegrationError


def register_video_callback(
    bridge: Any, site: str, callback: Any, *, label: str, **kwargs: Any
) -> Any:
    """Register the executor's callback with the bridge's removable-handle contract."""
    handle = bridge.register_callback(site, callback, **kwargs)
    if not callable(getattr(handle, "remove", None)):
        raise BackendIntegrationError(
            f"{label} video bridge returned a non-removable handle"
        )
    return handle


def release_video_callback(bridge: Any, handle: Any, owns_bridge: bool) -> list[str]:
    """Remove a callback before detaching an owned bridge; attempt both cleanups."""
    errors = []
    if handle is not None:
        try:
            handle.remove()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    if owns_bridge and bridge is not None:
        try:
            bridge.detach()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
    return errors


def install_video_bridge(model: Any, backbone: str) -> tuple[Any, bool, str]:
    """Return the bridge, whether this caller created it, and its callback site."""

    if backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision.video_compression import (
            LLAVA_ONEVISION_VIDEO_SPLICE_SITE,
            install_llava_onevision_video_bridge,
        )

        names = ("mllm_video_compression_bridge", "_mllm_video_compression_bridge")
        installer = install_llava_onevision_video_bridge
        site = LLAVA_ONEVISION_VIDEO_SPLICE_SITE
    elif backbone == "qwen25_vl":
        from mllm_comp.backbones.qwen25_vl.video_compression import (
            QWEN25_VL_VIDEO_SPLICE_SITE,
            install_qwen25_vl_video_bridge,
        )

        names = (
            "_mllm_comp_qwen25_vl_video_bridge",
            "mllm_video_compression_bridge",
        )
        installer = install_qwen25_vl_video_bridge
        site = QWEN25_VL_VIDEO_SPLICE_SITE
    else:
        from mllm_comp.backbones.qwen3_vl.video_compression import (
            QWEN3_VL_VIDEO_SPLICE_SITE,
            install_qwen3_vl_video_bridge,
        )

        names = (
            "_mllm_comp_qwen3_vl_video_bridge",
            "mllm_video_compression_bridge",
        )
        installer = install_qwen3_vl_video_bridge
        site = QWEN3_VL_VIDEO_SPLICE_SITE
    existing = next(
        (
            getattr(model, name, None)
            for name in names
            if getattr(model, name, None) is not None
        ),
        None,
    )
    return installer(model), existing is None, site
