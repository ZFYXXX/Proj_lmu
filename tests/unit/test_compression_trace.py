"""Request retention and append-only compression evidence without model weights."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from mllm_comp.backbones.vqtoken_video import VQTokenVideoExecutor
from mllm_comp.core.layout import LayoutDelta
from mllm_comp.pipeline.trace import CompressionTraceWriter
from mllm_comp.pipeline.runtime_session import CompressionActivationRecord, CompressionSession


def test_trace_appends_each_request_once_and_manifest_stays_bounded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mllm_comp.pipeline.trace as trace_module

    manifests = []
    original_write = trace_module.atomic_write_json

    def capture(path, payload):
        manifests.append(json.dumps(payload))
        original_write(path, payload)

    monkeypatch.setattr(trace_module, "atomic_write_json", capture)
    writer = CompressionTraceWriter(
        str(tmp_path / "runtime.json"), model_class="fixture",
    )
    writer.write({"enabled": True}, last_request=None)
    assert writer.trace_path.read_text(encoding="utf-8") == ""
    for index in range(60):
        runtime = {
            "enabled": True,
            "activations": [{
                "name": "compress_video", "binding": "fixture.video",
                "projector_invocations": index + 1,
                "projector_layout_deltas": [{
                    "invocation_id": index,
                    "kept_indices": list(range(500)),
                }],
            }],
        }
        writer.write(runtime, last_request={"sample_id": str(index)})
    records = [json.loads(line) for line in writer.trace_path.read_text(encoding="utf-8").splitlines()]
    assert [r["request"]["sample_id"] for r in records] == list(map(str, range(60)))
    assert [r["request_index"] for r in records] == list(range(60))
    assert all(len(r["runtime"]["activations"][0]["projector_layout_deltas"]) == 1 for r in records)
    sizes = [len(json.dumps(r)) for r in records]
    assert sum(sizes[30:]) < 1.05 * sum(sizes[:30])
    assert max(map(len, manifests[1:])) < 1.05 * min(map(len, manifests[1:]))
    assert all("kept_indices" not in value for value in manifests)
    summary = json.loads(manifests[-1])
    assert summary["runtime"]["activations"][0]["projector_invocations"] == 60
    assert summary["runtime"]["activations"][0]["projector_layout_deltas_count"] == 1
    assert summary["request_count"] == summary["trace"]["record_count"] == 60
    assert Path(summary["trace"]["path"]) == writer.trace_path


def test_failed_trace_serialization_preserves_completed_requests(
    tmp_path: Path,
) -> None:
    writer = CompressionTraceWriter(str(tmp_path / "runtime.json"), model_class="fixture")
    writer.write({"enabled": True}, last_request={"sample_id": "completed"})
    manifest = writer.manifest_path.read_bytes()
    trace = writer.trace_path.read_bytes()
    with pytest.raises(TypeError):
        writer.write({"not_serializable": object()}, last_request={"sample_id": "failed"})
    assert writer.manifest_path.read_bytes() == manifest
    assert writer.trace_path.read_bytes() == trace
    assert writer.request_count == 1


def test_new_writer_resets_previous_run_without_path_collision(tmp_path: Path) -> None:
    path = str(tmp_path / "runtime.jsonl")
    first = CompressionTraceWriter(path, model_class="fixture")
    first.write({"enabled": True}, last_request={"sample_id": "old"})
    second = CompressionTraceWriter(path, model_class="fixture")
    second.write({"enabled": True}, last_request=None)
    assert second.trace_path != second.manifest_path
    assert second.trace_path.read_text(encoding="utf-8") == ""
    assert json.loads(second.manifest_path.read_text(encoding="utf-8"))["request_count"] == 0


def test_video_executor_retains_current_request_and_lifetime_counter() -> None:
    executor = VQTokenVideoExecutor(backbone="qwen25_vl", model=object())
    delta = LayoutDelta.identity(4)
    result = SimpleNamespace(layout_deltas=(delta,))
    executor.kernel = lambda payload, config, rules: result
    for index in range(60):
        returned = executor._projector_callback(SimpleNamespace(
            payload=object(), invocation_id=f"request-{index}", media_id="video",
        ))
        assert returned is result
        metadata = executor.runtime_metadata()
        assert metadata["projector_invocations"] == index + 1
        assert len(metadata["projector_layout_deltas"]) == 1
        assert metadata["projector_layout_deltas"][0]["invocation_id"] == f"request-{index}"


class _DiagnosticActivation:
    def __init__(self):
        self.records = []
        self.replay = object()
        self.invocations = 0

    def install(self):
        pass

    def uninstall(self):
        pass

    def clear_diagnostic_records(self):
        self.records.clear()

    def record_forward(self):
        self.invocations += 1
        self.records.append({"invocation_index": self.invocations, "output_length": 2})

    def runtime_metadata(self):
        return {"layout_delta_records": list(self.records)}


def _diagnostic_session():
    activation = _DiagnosticActivation()
    session = CompressionSession([
        CompressionActivationRecord("fixture", activation, binding="fixture.image"),
    ]).install()
    return session, activation


@pytest.mark.parametrize("backbone", ["qwen25_vl", "qwen3_vl", "llava_onevision"])
@pytest.mark.parametrize("phase", ["forward_train", "generate"])
def test_image_runners_bound_records_per_request_without_resetting_decode_or_replay(
    backbone: str, phase: str,
) -> None:
    from importlib import import_module

    implementation = import_module(f"mllm_comp.backbones.{backbone}.implementation")
    runner_type = {
        "qwen25_vl": "Qwen25VLRunner", "qwen3_vl": "Qwen3VLRunner",
        "llava_onevision": "LlavaOneVisionRunner",
    }[backbone]
    runner = getattr(implementation, runner_type)(
        processor_adapter=SimpleNamespace(detokenize=lambda *args: ["answer"]),
        native_validator=SimpleNamespace(
            validate_state=lambda *args: [], validate_generation=lambda *args: [],
        ),
    )
    session, activation = _diagnostic_session()
    replay = activation.replay

    class Model:
        def __call__(self, **kwargs):
            activation.record_forward()
            return "loss"

        def generate(self, **kwargs):
            # A single generation owns several forwards, not several requests.
            for _ in range(3):
                self()
            return [[1, 2, 3]]

    runtime = SimpleNamespace(
        model=Model(), processor=None, device=None, dtype=None,
        metadata={"_compression_session": session},
    )
    state = SimpleNamespace(
        input_ids=[[1, 2]], native_model_inputs={"input_ids": [[1, 2]]},
        modality_metadata={},
    )
    batch = SimpleNamespace(sample_ids=["fixture"], metadata={})
    per_request = 3 if phase == "generate" else 1
    for index in range(60):
        if phase == "generate":
            output = runner.generate(runtime, state, processed_batch=batch)
            assert output.text.token_ids == [[3]]
        else:
            assert runner.forward_train(runtime, state) == "loss"
        assert len(activation.records) == per_request
        assert activation.invocations == (index + 1) * per_request
        assert activation.replay is replay


def test_swift_image_training_keeps_current_layout_and_backward_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import importlib.util
    import sys
    from mllm_comp.core.vocabulary import ExecutionPhase, RuntimeMode

    # The external plugin patches its loader on import. Use an already-patched
    # fixture loader so this test exercises the real forward wrapper without
    # importing or installing the training backend.
    loader = type("Loader", (), {"_mllm_comp_training_loader_patch": True})
    monkeypatch.setitem(sys.modules, "swift.model.models.qwen", SimpleNamespace(Qwen2_5OmniLoader=loader))
    plugin_path = Path(__file__).parents[2] / "mllm_comp/integrations/ms_swift/training_plugin.py"
    spec = importlib.util.spec_from_file_location("_swift_retention_fixture", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    session, activation = _diagnostic_session()
    replay = activation.replay
    events = []
    forward_indices = iter(range(60))
    recorder = SimpleNamespace(
        record=lambda event, **fields: events.append((event, fields)),
        next_forward_index=lambda: next(forward_indices),
    )
    monkeypatch.setattr(plugin, "get_backbone", lambda name: object())
    monkeypatch.setattr(plugin, "RuntimeCompressionDispatcher", lambda **kwargs: SimpleNamespace(activate=lambda **kwargs: session))

    class Loss:
        requires_grad = True

        def register_hook(self, callback):
            self.backward_callback = callback

    class Model:
        training = True

        def parameters(self):
            yield SimpleNamespace(device="cpu", dtype="fixture")

        def forward(self, **kwargs):
            assert kwargs["use_cache"] is False
            activation.record_forward()
            if kwargs.get("fail"):
                raise RuntimeError("forward failed")
            return SimpleNamespace(loss=Loss())

    model = plugin._install_image_model_bridge(
        Model(), backbone_name="qwen25_vl",
        stage=SimpleNamespace(execution_phase=ExecutionPhase.FORWARD, binding="fixture", variant="fixture"),
        runtime_mode=RuntimeMode.DEBUG, recorder=recorder,
    )
    for index in range(60):
        output = model.forward(use_cache=True)
        gradient = object()
        assert output.loss.backward_callback(gradient) is gradient
        assert len(activation.records) == 1
        assert activation.invocations == index + 1
        assert activation.replay is replay
    forwards = [fields for event, fields in events if event == "forward"]
    backwards = [fields for event, fields in events if event == "backward"]
    assert [item["layout_records"][0]["invocation_index"] for item in forwards] == list(range(1, 61))
    assert all(len(item["layout_records"]) == 1 for item in forwards)
    assert [item["forward_index"] for item in forwards] == [item["forward_index"] for item in backwards]
    with pytest.raises(RuntimeError, match="forward failed"):
        model.forward(fail=True)
    assert len(activation.records) == 1 and activation.invocations == 61
    assert len(events) == 121  # install + the 60 completed forward/backward pairs
