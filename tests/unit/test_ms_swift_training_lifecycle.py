"""Swift integration ownership independent of model dependencies."""

import base64
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest


def _plugin(monkeypatch):
    loader = type("Loader", (), {"_mllm_comp_training_loader_patch": True})
    monkeypatch.setitem(sys.modules, "swift.model.models.qwen", SimpleNamespace(Qwen2_5OmniLoader=loader))
    path = Path(__file__).parents[2] / "mllm_comp/integrations/ms_swift/training_plugin.py"
    spec = importlib.util.spec_from_file_location("_swift_lifecycle_test", path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    return plugin


@pytest.mark.parametrize("fails", [False, True])
def test_native_save_and_failure_cleanup_keep_original_model_owner(monkeypatch, fails):
    plugin = _plugin(monkeypatch)
    events = []
    native_forward = lambda: None
    model = SimpleNamespace(
        forward=lambda: None,
        _mllm_comp_original_forward=native_forward,
        _mllm_comp_training_runtime=SimpleNamespace(close=lambda: events.append("close")),
        _mllm_comp_training_session=object(),
    )

    class NativePipeline:
        def __init__(self):
            self.model = model

        def run(self):
            assert hasattr(model, "_mllm_comp_training_session")
            events.append("native_train")
            # A native tuner can replace pipeline.model with a PEFT wrapper.
            self.model = SimpleNamespace(base_model=model)
            try:
                if fails:
                    raise RuntimeError("native failed")
                return "trained"
            finally:
                assert hasattr(model, "_mllm_comp_training_session")
                events.append("native_save")

    monkeypatch.setitem(sys.modules, "swift.pipelines.train.sft", SimpleNamespace(SwiftSft=NativePipeline))
    plugin._patch_swift_training_lifecycle()
    if fails:
        with pytest.raises(RuntimeError, match="native failed"):
            NativePipeline().run()
    else:
        assert NativePipeline().run() == "trained"
    assert events == ["native_train", "native_save", "close"]
    assert model.forward is native_forward
    assert not hasattr(model, "_mllm_comp_training_session")
    plugin._close_model_bridge(model)
    assert events.count("close") == 1


@pytest.mark.parametrize("schema_version", [1, 2, 3])
def test_current_and_persisted_worker_specs_remain_readable(monkeypatch, schema_version):
    plugin = _plugin(monkeypatch)
    payload = {"schema_version": schema_version, **(
        {"stages": [{"name": "stage"}]} if schema_version == 3 else {"stage": {"name": "stage"}}
    )}
    raw = json.dumps(payload).encode("utf-8")
    monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_SPEC", base64.urlsafe_b64encode(raw).decode("ascii"))
    loaded, _ = plugin._load_spec()
    assert loaded == payload


def test_multistage_debug_summary_keeps_each_input_budget_separate():
    from mllm_comp.integrations.ms_swift.diagnostics import _pretty_training_diagnostic_records

    records = [{
        "event": "forward", "stage_name": name, "forward_index": 0,
        "compression": {"original_audio_tokens": before, "kept_audio_tokens": after},
    } for name, before, after in (("projector", 8, 4), ("llm", 4, 1))]
    summary = _pretty_training_diagnostic_records(records, source_path=Path("debug_diagnostic.jsonl"))[0]
    assert summary["multi_stage"]
    assert "audio_tokens" not in summary
    assert [(stage["stage_name"], stage["audio_tokens"]["original"], stage["audio_tokens"]["kept"])
            for stage in summary["stages"]] == [("projector", 8, 4), ("llm", 4, 1)]
