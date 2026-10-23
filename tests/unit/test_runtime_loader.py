"""Runtime loader contracts shared by speech and text-only backbones."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.vocabulary import ExecutionPhase
from mllm_comp.pipeline import runtime_compression
from mllm_comp.runtime import loader


class _ProcessorAdapter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def load_processor(self, path: str, **kwargs: Any) -> object:
        self.kwargs = {"path": path, **kwargs}
        return object()


class _ModelAdapter:
    def __init__(self) -> None:
        self.kwargs: dict[str, Any] | None = None

    def load_model(self, path: str, **kwargs: Any) -> object:
        self.kwargs = {"path": path, **kwargs}
        return SimpleNamespace(dtype="bfloat16")


def _qwen3_config() -> Any:
    return experiment_config_from_dict(
        {
            "experiment": {"name": "runtime-loader"},
            "backbone": {"name": "qwen3_vl", "model_id": "Qwen/fixture"},
            "pipeline": {"stages": []},
            "analysis": {},
            "train": {},
            "eval": {},
            "data": {"dataset_id": "fixture"},
            "output": {
                "run_dir": "runs/runtime-loader",
                "manifest_path": "runs/runtime-loader/manifest.json",
            },
        }
    )


def test_text_only_runtime_does_not_forward_omni_talker_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_adapter = _ProcessorAdapter()
    model_adapter = _ModelAdapter()
    facade = SimpleNamespace(
        processor_adapter=processor_adapter,
        model_adapter=model_adapter,
        capabilities=lambda: SimpleNamespace(
            name="qwen3_vl", output_modalities=("text",)
        ),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(loader, "get_backbone", lambda name: facade)
    activation_phases: list[Any] = []

    def activate(**kwargs: Any) -> None:
        activation_phases.append(kwargs["execution_phase"])

    monkeypatch.setattr(runtime_compression, "activate_runtime_compression", activate)

    loaded_facade, runtime = loader.load_backbone_runtime(
        _qwen3_config(), device="cpu", enable_talker=False
    )

    assert loaded_facade is facade
    assert model_adapter.kwargs is not None
    assert "enable_talker" not in model_adapter.kwargs
    assert runtime.metadata["enable_talker"] is False
    assert runtime.metadata["speech_sample_rate"] is None
    assert runtime.metadata["runtime_mode"] == "default"
    assert activation_phases[-1] is ExecutionPhase.GENERATE

    loader.load_backbone_runtime(
        _qwen3_config(),
        device="cpu",
        enable_talker=False,
        execution_phase="forward",
    )
    assert activation_phases[-1] is ExecutionPhase.FORWARD

    with pytest.raises(BackendIntegrationError, match="does not expose"):
        loader.load_backbone_runtime(
            _qwen3_config(), device="cpu", enable_talker=True
        )

    with pytest.raises(BackendIntegrationError, match="only for qwen25_omni"):
        loader.load_backbone_runtime(
            _qwen3_config(), device="cpu", runtime_mode="profile"
        )


def test_runtime_loader_forwards_llava_vision_tower_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor_adapter = _ProcessorAdapter()
    model_adapter = _ModelAdapter()
    facade = SimpleNamespace(
        processor_adapter=processor_adapter,
        model_adapter=model_adapter,
        capabilities=lambda: SimpleNamespace(
            name="llava_onevision", output_modalities=("text",)
        ),
    )
    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        device=lambda value: value,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setattr(loader, "get_backbone", lambda name: facade)
    monkeypatch.setattr(
        runtime_compression,
        "activate_runtime_compression",
        lambda **kwargs: None,
    )
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "llava-runtime-loader"},
            "backbone": {
                "name": "llava_onevision",
                "model_id": "lmms-lab/fixture",
                "model_revision": "checkpoint-sha",
                "config": {
                    "model_load_kwargs": {
                        "vision_tower_revision": "vision-sha"
                    }
                },
            },
            "pipeline": {"stages": []},
            "analysis": {},
            "train": {},
            "eval": {},
            "data": {"dataset_id": "fixture"},
            "output": {
                "run_dir": "runs/llava-runtime-loader",
                "manifest_path": "runs/llava-runtime-loader/manifest.json",
            },
        }
    )

    loader.load_backbone_runtime(config, device="cpu")

    assert processor_adapter.kwargs == {
        "path": "lmms-lab/fixture",
        "revision": "checkpoint-sha",
    }
    assert model_adapter.kwargs is not None
    assert model_adapter.kwargs["revision"] == "checkpoint-sha"
    assert model_adapter.kwargs["vision_tower_revision"] == "vision-sha"
