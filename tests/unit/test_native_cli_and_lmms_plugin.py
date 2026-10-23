"""Contracts for native process launching and lmms model entry points."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import signal
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.config import config_to_dict, load_config
from mllm_comp.core.vocabulary import ExecutionPhase, RuntimeMode
from mllm_comp.integrations.native_cli import NativeCLIPlan, run_native_cli


def test_native_cli_inherits_streams_and_forwards_sigterm(monkeypatch: Any) -> None:
    seen: dict[str, Any] = {}

    class _Process:
        def __init__(self) -> None:
            self.signals: list[int] = []

        def poll(self) -> None:
            return None

        def send_signal(self, signum: int) -> None:
            self.signals.append(signum)

        def wait(self) -> int:
            handler = signal.getsignal(signal.SIGTERM)
            assert callable(handler)
            handler(signal.SIGTERM, None)
            return -signal.SIGTERM

    process = _Process()

    def popen_factory(argv: list[str], **kwargs: Any) -> _Process:
        seen.update({"argv": argv, **kwargs})
        return process

    monkeypatch.setenv("NATIVE_PARENT", "kept")
    monkeypatch.setenv("NATIVE_REMOVED", "remove-me")
    result = run_native_cli(
        NativeCLIPlan(
            argv=["native", "--flag"],
            environment={"NATIVE_CHILD": "set"},
            unset_environment=["NATIVE_REMOVED"],
            cwd=".",
            expected_output_paths=[],
        ),
        popen_factory=popen_factory,
    )

    assert result.returncode == -signal.SIGTERM
    assert result.shell_returncode == 128 + signal.SIGTERM
    assert process.signals == [signal.SIGTERM]
    assert seen["env"]["NATIVE_PARENT"] == "kept"
    assert seen["env"]["NATIVE_CHILD"] == "set"
    assert "NATIVE_REMOVED" not in seen["env"]
    assert "stdout" not in seen and "stderr" not in seen


def test_lmms_entrypoint_exposes_chat_and_simple_classes_with_one_loader(
    monkeypatch: Any,
    tmp_path: Any,
) -> None:
    lmms_eval = ModuleType("lmms_eval")
    lmms_eval.__path__ = []  # type: ignore[attr-defined]
    api = ModuleType("lmms_eval.api")
    api.__path__ = []  # type: ignore[attr-defined]
    api_model = ModuleType("lmms_eval.api.model")
    api_task = ModuleType("lmms_eval.api.task")
    api_task.ConfigurableTask = type("ConfigurableTask", (), {
        "process_results": lambda self, doc, results: {"native": results},
    })
    models = ModuleType("lmms_eval.models")
    models.__path__ = []  # type: ignore[attr-defined]
    registry = ModuleType("lmms_eval.models.registry_v2")

    class _CacheHook:
        def add_partial(self, *args: Any) -> None:
            del args

    class _LMM:
        def __init__(self) -> None:
            self.task_dict: dict[str, Any] = {}
            self.cache_hook = _CacheHook()

    @dataclass
    class _Manifest:
        model_id: str
        simple_class_path: str | None = None
        chat_class_path: str | None = None
        aliases: tuple[str, ...] = ()

    api_model.lmms = _LMM  # type: ignore[attr-defined]
    registry.ModelManifest = _Manifest  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "lmms_eval", lmms_eval)
    monkeypatch.setitem(sys.modules, "lmms_eval.api", api)
    monkeypatch.setitem(sys.modules, "lmms_eval.api.model", api_model)
    monkeypatch.setitem(sys.modules, "lmms_eval.api.task", api_task)
    monkeypatch.setitem(sys.modules, "lmms_eval.models", models)
    monkeypatch.setitem(sys.modules, "lmms_eval.models.registry_v2", registry)

    accelerate = ModuleType("accelerate")

    class _Accelerator:
        device = "cuda:3"
        process_index = 0
        num_processes = 1

    accelerate.Accelerator = _Accelerator  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "accelerate", accelerate)

    module_name = "mllm_comp.integrations.lmms_eval.plugin"
    sys.modules.pop(module_name, None)
    plugin = importlib.import_module(module_name)
    read_spec = plugin._load_plugin_spec
    manifests = plugin.get_model_manifests()
    assert [manifest.model_id for manifest in manifests] == [
        "qwen2_5_omni",
        "mllm_comp_qwen25_vl",
        "mllm_comp_qwen3_vl",
        "mllm_comp_llava_onevision",
    ]
    assert len({manifest.simple_class_path for manifest in manifests}) == 1
    assert len({manifest.chat_class_path for manifest in manifests}) == 1
    assert manifests[0].simple_class_path.endswith(".MLLMCompLMM")
    assert manifests[0].chat_class_path.endswith(".MLLMCompChatLMM")
    assert plugin.MLLMCompLMM.is_simple is True
    assert plugin.MLLMCompChatLMM.is_simple is False
    assert plugin._ranked_trace_path(
        "trace.jsonl", rank=1, world_size=2
    ).endswith("trace.rank1.jsonl")

    config = SimpleNamespace(
        backbone=SimpleNamespace(name="qwen25_omni", config={}),
        pipeline=SimpleNamespace(stages=[]),
        eval=SimpleNamespace(runtime_mode=RuntimeMode.DEFAULT),
    )
    runtime = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace()),
        processor=SimpleNamespace(tokenizer=object()),
        device="cuda:3",
    )

    class _Facade:
        def capabilities(self) -> Any:
            return SimpleNamespace(name="qwen25_omni")

    loader_calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        plugin,
        "_load_plugin_spec",
        lambda path: {
            "schema_version": 1,
            "config_path": "config.yaml",
            "checkpoint_path": "Qwen/fixture",
            "backbone_name": "qwen25_omni",
            "compression_mode": "none",
            "generation": {},
        },
    )
    monkeypatch.setattr(plugin, "load_config", lambda path: config)

    def load_runtime(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        loader_calls.append({"args": args, **kwargs})
        return _Facade(), runtime

    monkeypatch.setattr(plugin, "load_backbone_runtime", load_runtime)
    model = plugin.MLLMCompLMM(
        pretrained="Qwen/fixture",
        mllm_comp_spec="plugin.json",
        batch_size=1,
    )

    assert model.model is runtime.model
    assert loader_calls[0]["checkpoint_path"] == "Qwen/fixture"
    assert loader_calls[0]["device"] == "cuda:3"
    assert loader_calls[0]["train"] is False
    assert loader_calls[0]["enable_talker"] is False
    assert loader_calls[0]["runtime_mode"].value == "default"
    assert loader_calls[0]["execution_phase"] is ExecutionPhase.GENERATE

    for runtime_mode in (RuntimeMode.DEBUG, RuntimeMode.PROFILE):
        monkeypatch.setattr(
            plugin,
            "_load_plugin_spec",
            lambda path, mode=runtime_mode: {
                "schema_version": 1,
                "config_path": "config.yaml",
                "checkpoint_path": "Qwen/fixture",
                "backbone_name": "qwen25_omni",
                "compression_mode": "none",
                "runtime_mode": mode.value,
                "generation": {},
            },
        )
        plugin.MLLMCompLMM(
            pretrained="Qwen/fixture",
            mllm_comp_spec="plugin.json",
            batch_size=1,
        )
        assert loader_calls[-1]["runtime_mode"] is runtime_mode

    monkeypatch.setattr(
        plugin,
        "_load_plugin_spec",
        lambda path: {
            "schema_version": 1,
            "config_path": "config.yaml",
            "checkpoint_path": "Qwen/fixture",
            "backbone_name": "qwen25_omni",
            "compression_mode": "none",
            "generation": {},
            "expected_world_size": 2,
        },
    )
    with pytest.raises(BackendIntegrationError, match="world size"):
        plugin.MLLMCompLMM(
            pretrained="Qwen/fixture",
            mllm_comp_spec="plugin.json",
            batch_size=1,
        )

    # New launches carry the exact typed config, even without a source YAML.
    effective = load_config("tests/fixtures/qwen25_omni_compressed_train.yaml")
    effective.path = None
    effective.backbone.config["device"] = "cpu"
    effective.eval.compression_mode = "compressed"
    effective.backbone.config["python_override"] = {"enabled": True}
    effective.pipeline.stages[0].schedule["keep_ratio"] = 0.25
    effective.pipeline.extensions = ["research.example:register"]
    new_spec = {
        "schema_version": 2,
        "config_path": None,
        "effective_config": config_to_dict(effective),
        "checkpoint_path": "Qwen/fixture",
        "backbone_name": "qwen25_omni",
        "compression_mode": "compressed",
        "generation": {},
    }
    spec_path = tmp_path / "plugin.json"
    spec_path.write_text(json.dumps(new_spec), encoding="utf-8")
    monkeypatch.setattr(plugin, "_load_plugin_spec", read_spec)
    monkeypatch.setattr(plugin, "load_config", lambda path: pytest.fail("New specs must not reload YAML"))
    plugin.MLLMCompLMM(pretrained="Qwen/fixture", mllm_comp_spec=str(spec_path))
    restored = loader_calls[-1]["args"][0]
    assert restored.path is None
    assert config_to_dict(restored) == config_to_dict(effective)
    assert loader_calls[-1]["compression_mode"] == "compressed"
    assert loader_calls[-1]["device"] == "cpu"

    plugin.MLLMCompLMM(
        pretrained="Qwen/fixture", mllm_comp_spec=str(spec_path), device="cuda:1"
    )
    assert loader_calls[-1]["device"] == "cuda:1"

    del new_spec["effective_config"]
    spec_path.write_text(json.dumps(new_spec), encoding="utf-8")
    with pytest.raises(BackendIntegrationError, match="effective_config"):
        plugin.MLLMCompLMM(pretrained="Qwen/fixture", mllm_comp_spec=str(spec_path))
    sys.modules.pop(module_name, None)
