from __future__ import annotations

import json
from pathlib import Path
import sys
from types import SimpleNamespace
from types import ModuleType
from typing import Any, Mapping

import pytest

from mllm_comp.analysis.common.representation import (
    aggregate_representation_samples,
    cosine_similarity_rows,
)
from mllm_comp.analysis.common.residual import ResidualNodeCapture
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.analysis.probes import RepresentationDynamics
from mllm_comp.analysis.probes.representation_dynamics import (
    _comparisons_for_collect_points,
    _sample_layer_metrics,
)
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.definition import AnalysisOverrides, resolve_analysis_definition
from mllm_comp.analysis.runtime import (
    _analysis_compression_execution_phase,
    execute_analysis,
)
from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.config.loader import load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord
from mllm_comp.core.vocabulary import ExecutionPhase
from mllm_comp.pipeline.runtime_session import (
    CompressionActivationRecord,
    CompressionSession,
)
from mllm_comp.registry.catalog import CapabilityKind, CapabilityStatus


class _Handle:
    def __init__(self, name: str, removed: list[str]) -> None:
        self.name = name
        self.removed = removed

    def remove(self) -> None:
        self.removed.append(self.name)


class _Module:
    def __init__(self, removed: list[str]) -> None:
        self.removed = removed
        self.index = 0

    def register_forward_pre_hook(self, _callback: Any) -> _Handle:
        self.index += 1
        return _Handle(f"pre-{self.index}", self.removed)

    def register_forward_hook(self, _callback: Any) -> _Handle:
        self.index += 1
        return _Handle(f"forward-{self.index}", self.removed)


def test_hook_session_removes_in_reverse_order_on_failure() -> None:
    removed: list[str] = []
    module = _Module(removed)

    with pytest.raises(RuntimeError, match="forward failed"):
        with HookSession() as hooks:
            hooks.register_forward_pre_hook(module, lambda *_args: None)
            hooks.register_forward_hook(module, lambda *_args: None)
            raise RuntimeError("forward failed")

    assert removed == ["forward-2", "pre-1"]


def test_representation_formulas_and_absent_group_aggregation() -> None:
    values = cosine_similarity_rows(
        [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]],
    )
    assert values == pytest.approx([1.0, 0.0, -1.0])

    aggregate = aggregate_representation_samples(
        [
            {
                "layers": {
                    "0": {
                        "full_layer_res": {
                            "vision": {"cosine": 0.5},
                            "system": {"cosine": None},
                        }
                    }
                }
            },
            {
                "layers": {
                    "0": {
                        "full_layer_res": {
                            "vision": {"cosine": 1.0},
                            "system": {"cosine": None},
                        }
                    }
                }
            },
        ]
    )
    assert aggregate["layers"]["0"]["full_layer_res"]["vision"] == {
        "cosine": 0.75,
        "finite_sample_count": 2,
        "absent_or_nonfinite_sample_count": 0,
    }
    assert aggregate["layers"]["0"]["full_layer_res"]["system"]["cosine"] is None


def test_residual_capture_snapshots_before_provider_in_place_updates() -> None:
    class _MutableTensor:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self) -> "_MutableTensor":
            return self

        def clone(self) -> "_MutableTensor":
            return _MutableTensor(self.value)

    capture = ResidualNodeCapture(())
    provider_output = _MutableTensor(1.0)

    capture._store(0, "after_ffn", provider_output)
    provider_output.value = 7.375

    assert capture.values[0]["after_ffn"].value == 1.0


def test_representation_collect_points_select_actual_comparisons() -> None:
    selected = _comparisons_for_collect_points(("layer_input", "layer_output"))

    assert selected == {"full_layer_res": ("before_attn", "after_ffn")}
    with pytest.raises(ValueError, match="both ends"):
        RepresentationDynamics().validate_config(
            {"collect_points": ["layer_input"], "metrics": ["cosine"]}
        )


def test_representation_probe_accepts_audio_token_group() -> None:
    RepresentationDynamics().validate_config(
        {"token_groups": ["text", "audio"], "layers": [0]}
    )


@pytest.mark.parametrize(
    ("path", "expected_phase"),
    [
        (
            "examples/analysis/compression/"
            "qwen25_omni_audio_segmentwise_representation.yaml",
            ExecutionPhase.FORWARD,
        ),
        (
            "examples/analysis/compression/"
            "qwen25_omni_audio_atome_representation.yaml",
            ExecutionPhase.FORWARD,
        ),
        (
            "examples/analysis/compression/"
            "qwen25_omni_audio_pyramid_representation.yaml",
            ExecutionPhase.FORWARD,
        ),
        (
            "examples/analysis/compression/"
            "qwen25_omni_audio_spade_representation.yaml",
            ExecutionPhase.FORWARD,
        ),
    ],
)
def test_audio_post_compression_analysis_selects_stage_lifecycle(
    path: str,
    expected_phase: ExecutionPhase,
) -> None:
    config = load_config(path)
    definition = resolve_analysis_definition(config, AnalysisOverrides())

    assert definition.phase == "post_compression"
    assert _analysis_compression_execution_phase(config, definition) is expected_phase


@pytest.mark.parametrize("request_phase", [ExecutionPhase.FORWARD, ExecutionPhase.GENERATE])
def test_analysis_combines_prepare_with_forward_requests(request_phase) -> None:
    config = load_config(
        "examples/analysis/compression/qwen25_omni_audio_segmentwise_representation.yaml"
    )
    operation = load_config(
        "examples/analysis/compression/qwen25_omni_audio_spade_representation.yaml"
    ).pipeline.stages[0]
    definition = resolve_analysis_definition(config, AnalysisOverrides())
    config.pipeline.stages[0].execution_phase = request_phase
    config.pipeline.stages.append(operation)
    if request_phase is ExecutionPhase.GENERATE:
        with pytest.raises(BackendIntegrationError, match="probes execute model forward"):
            _analysis_compression_execution_phase(config, definition)
    else:
        assert _analysis_compression_execution_phase(config, definition) is ExecutionPhase.FORWARD


def test_representation_sample_metrics_emit_requested_comparisons(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Node:
        def float(self) -> "_Node":
            return self

    class _Scalar:
        def __init__(self, value: float) -> None:
            self.value = value

        def detach(self) -> "_Scalar":
            return self

        def cpu(self) -> "_Scalar":
            return self

        def item(self) -> float:
            return self.value

    class _Selected:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def mean(self) -> _Scalar:
            return _Scalar(sum(self.values) / len(self.values))

    class _Row:
        def __init__(self, values: list[float]) -> None:
            self.values = values

        def index_select(self, dim: int, indices: tuple[int, ...]) -> _Selected:
            assert dim == 0
            return _Selected([self.values[index] for index in indices])

    class _Cosine:
        ndim = 2
        shape = (1, 2)
        device = "cpu"

        def __getitem__(self, index: int) -> _Row:
            assert index == 0
            return _Row([0.25, 0.75])

    torch_module = ModuleType("torch")
    torch_module.__path__ = []  # type: ignore[attr-defined]
    torch_module.long = object()  # type: ignore[attr-defined]
    torch_module.tensor = (  # type: ignore[attr-defined]
        lambda positions, **_kwargs: tuple(positions)
    )
    nn_module = ModuleType("torch.nn")
    nn_module.__path__ = []  # type: ignore[attr-defined]
    functional_module = ModuleType("torch.nn.functional")
    functional_module.cosine_similarity = (  # type: ignore[attr-defined]
        lambda _left, _right, **_kwargs: _Cosine()
    )
    torch_module.nn = nn_module  # type: ignore[attr-defined]
    nn_module.functional = functional_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.nn", nn_module)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", functional_module)

    result = _sample_layer_metrics(
        {0: {"before_attn": _Node(), "after_ffn": _Node()}},
        layout=TokenGroupLayout(
            labels_by_sample=(("image", "text"),),
            evidence={},
        ),
        token_groups=["vision", "all"],
        comparisons={"full_layer_res": ("before_attn", "after_ffn")},
    )

    assert result == {
        "0": {
            "full_layer_res": {
                "vision": {
                    "cosine": 0.25,
                    "token_count": 1,
                    "status": "finite",
                },
                "all": {
                    "cosine": 0.5,
                    "token_count": 2,
                    "status": "finite",
                },
            }
        }
    }


@pytest.mark.parametrize("supervised", [False, True])
def test_execute_analysis_shares_one_dataset_loop_across_probe_sessions(supervised) -> None:
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "runtime"},
            "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {
                "backend": "lmms_eval",
                "tasks": ["task"],
                "probes": [],
            },
            "train": {},
            "eval": {},
            "data": {},
            "output": {"run_dir": "runs/runtime", "manifest_path": "runs/runtime/m.json"},
        }
    )
    definition = SimpleNamespace(
        compression_mode="none",
        phase="pre_compression",
        sample_limit=2,
        probes=[
            {"name": "first", "method": "fixture"},
            {"name": "second", "method": "fixture"},
        ],
    )
    sessions: list[Any] = []

    class _Session:
        def __init__(self, name: str) -> None:
            self.name = name
            self.samples: list[Any] = []
            self.closed = False

        def process(self, sample: Any) -> None:
            self.samples.append(sample)

        def finish(self) -> AnalysisRecord:
            return AnalysisRecord(
                name=self.name,
                phase="pre_compression",
                method="fixture",
                summary={},
                artifact_paths={"aggregate": f"{self.name}.json"},
                sample_count=len(self.samples),
            )

        def close(self) -> None:
            self.closed = True

    class _Plugin:
        def open(self, **kwargs: Any) -> _Session:
            session = _Session(kwargs["config"]["name"])
            sessions.append(session)
            return session

    class _Adapter:
        calls = 0

        def iter_analysis_samples(self, **_kwargs: Any) -> Any:
            assert _kwargs["requires_teacher_forcing"] is supervised
            self.calls += 1
            yield {"sample_id": "a"}
            yield {"sample_id": "b"}

        def dependency_report(self) -> dict[str, Any]:
            return {"versions": {"lmms-eval": "fixture-version"}}

    _Plugin.requires_teacher_forcing = supervised
    adapter = _Adapter()
    runtime = BackboneRuntime(model=object(), processor=object())

    def runtime_loader(_config: Any, **kwargs: Any) -> tuple[Any, Any]:
        assert kwargs["execution_phase"] is ExecutionPhase.FORWARD
        return object(), runtime

    result = execute_analysis(
        config,
        definition,
        runtime_loader=runtime_loader,
        adapter=adapter,
        provider_resolver=lambda _name: _Plugin,
    )

    assert adapter.calls == 1
    assert [len(session.samples) for session in sessions] == [2, 2]
    assert all(session.closed for session in sessions)
    assert runtime.model is None
    assert runtime.processor is None
    assert result.sample_count == 2
    assert result.backend.version == "fixture-version"
    assert result.backend.args["probe_capabilities"][0][
        "capability_status"
    ] == "external"
    assert "candidate_validation" not in result.backend.args
    assert [record.name for record in result.records] == ["first", "second"]


@pytest.mark.parametrize(
    ("failure", "cleanup_failure"),
    [
        (None, None),
        ("open", None),
        ("process", None),
        ("finish", None),
        (None, "probe"),
        ("process", "compression"),
    ],
)
def test_execute_analysis_owns_compression_and_runtime_lifecycle(
    failure: str | None,
    cleanup_failure: str | None,
) -> None:
    config = experiment_config_from_dict(
        {
            "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {"tasks": ["task"], "probes": []},
            "output": {
                "run_dir": "runs/runtime-lifecycle",
                "manifest_path": "runs/runtime-lifecycle/manifest.json",
            },
        }
    )
    definition = SimpleNamespace(
        compression_mode="compressed",
        phase="post_compression",
        sample_limit=1,
        probes=[{"name": name, "method": "fixture"} for name in ("first", "second")],
    )
    events: list[str] = []
    removed: list[str] = []
    runtimes: list[BackboneRuntime] = []
    compressions: list[CompressionSession] = []
    models: list[Any] = []
    run_error = RuntimeError(f"{failure} failed")

    class _Activation:
        def __init__(self, model: Any) -> None:
            self.model = model
            self.history = [{"tokens": 2}]

        def install(self) -> None:
            assert not self.model.compressed
            self.model.compressed = True

        def uninstall(self) -> None:
            events.append("compression")
            self.model.compressed = False
            self.history.clear()
            if cleanup_failure == "compression":
                raise RuntimeError("compression cleanup failed")

        def runtime_metadata(self) -> Mapping[str, Any]:
            return {"history": self.history}

    def runtime_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, BackboneRuntime]:
        model = _Module(removed)
        model.compressed = False
        compression = CompressionSession(
            [CompressionActivationRecord("fixture", _Activation(model), "fixture")]
        ).install()
        runtime = BackboneRuntime(
            model=model,
            processor=object(),
            metadata={
                "_compression_session": compression,
                "_token_debug_collector": SimpleNamespace(
                    uninstall=lambda: events.append("debug")
                ),
                "structural_handles": {"model": model},
            },
        )
        runtimes.append(runtime)
        compressions.append(compression)
        models.append(model)
        return object(), runtime

    class _Probe:
        def open(self, *, runtime: BackboneRuntime, config: Any, **_kwargs: Any) -> Any:
            name = config["name"]
            if failure == "open" and name == "second":
                raise run_error
            hooks = HookSession()
            hooks.register_forward_hook(runtime.model, lambda *_args: None)

            def process(_sample: Any) -> None:
                assert runtime.model.compressed
                if failure == "process":
                    raise run_error

            def finish() -> AnalysisRecord:
                if failure == "finish":
                    raise run_error
                return AnalysisRecord(
                    name=name,
                    phase="post_compression",
                    method="fixture",
                    summary={},
                    artifact_paths={},
                )

            def close() -> None:
                assert runtime.model.compressed
                hooks.close()
                events.append(name)
                if cleanup_failure == "probe":
                    raise RuntimeError("probe cleanup failed")

            return SimpleNamespace(process=process, finish=finish, close=close)

    adapter = SimpleNamespace(iter_analysis_samples=lambda **_kwargs: iter([{}]))

    def run() -> Any:
        return execute_analysis(
            config,
            definition,
            runtime_loader=runtime_loader,
            adapter=adapter,
            provider_resolver=lambda _name: _Probe,
        )

    if failure is not None:
        with pytest.raises(RuntimeError) as caught:
            run()
        assert caught.value is run_error
    elif cleanup_failure is not None:
        with pytest.raises(BackendIntegrationError, match="probe cleanup failed"):
            run()
    else:
        result = run()
        assert result.compression_runtime_metadata["status"] == "active"
        assert result.compression_runtime_metadata["activations"][0]["history"] == [
            {"tokens": 2}
        ]

    assert events == (["first"] if failure == "open" else ["second", "first"]) + [
        "debug", "compression"
    ]
    assert removed == (["forward-1"] if failure == "open" else ["forward-2", "forward-1"])
    assert compressions[0].status == (
        "uninstall_failed" if cleanup_failure == "compression" else "uninstalled"
    )
    assert not models[0].compressed
    assert runtimes[0].model is None
    assert runtimes[0].processor is None
    assert runtimes[0].metadata == {}
    events_before_close = list(events)
    runtimes[0].close()
    assert events == events_before_close

    # A failed experiment must not prevent an independent experiment from running.
    if failure is not None or cleanup_failure is not None:
        failure = cleanup_failure = None
        assert run().sample_count == 1
        assert compressions[1].status == "uninstalled"
        assert not models[1].compressed


def test_execute_analysis_rejects_duplicate_probe_names_before_model_load() -> None:
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "runtime-duplicate"},
            "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {"tasks": ["task"], "probes": []},
            "train": {},
            "eval": {},
            "data": {},
            "output": {"run_dir": "runs/runtime", "manifest_path": "runs/m.json"},
        }
    )
    called = False

    def runtime_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        nonlocal called
        called = True
        return object(), object()

    with pytest.raises(BackendIntegrationError, match="duplicated"):
        execute_analysis(
            config,
            SimpleNamespace(
                probes=[
                    {"name": "same", "method": "fixture"},
                    {"name": "same", "method": "fixture"},
                ]
            ),
            runtime_loader=runtime_loader,
        )

    assert called is False


def test_execute_analysis_rejects_target_probe_before_model_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "runtime-target"},
            "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {"tasks": ["task"], "probes": []},
            "train": {},
            "eval": {},
            "data": {},
            "output": {
                "run_dir": "runs/runtime-target",
                "manifest_path": "runs/runtime-target/manifest.json",
            },
        }
    )
    called = False

    def runtime_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        nonlocal called
        called = True
        return object(), object()

    from mllm_comp.registry import catalog

    original_get_capability_spec = catalog.get_capability_spec

    def fixture_get_capability_spec(kind: Any, name: str) -> Any:
        if kind is CapabilityKind.ANALYSIS and name == "fixture_target":
            return SimpleNamespace(
                status=CapabilityStatus.TARGET,
                metadata={"backbones": ["qwen25_vl"]},
            )
        return original_get_capability_spec(kind, name)

    monkeypatch.setattr(catalog, "get_capability_spec", fixture_get_capability_spec)

    with pytest.raises(BackendIntegrationError, match="not active"):
        execute_analysis(
            config,
            SimpleNamespace(
                probes=[
                    {
                        "name": "fixture_target",
                        "method": "fixture_target",
                    }
                ]
            ),
            runtime_loader=runtime_loader,
        )

    assert called is False


def test_execute_analysis_rejects_stale_outputs_before_model_load(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "stale-run"
    analysis_dir = run_dir / "analysis" / "old-probe"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "old.json").write_text("{}", encoding="utf-8")
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    config = experiment_config_from_dict(
        {
            "experiment": {"name": "runtime-stale"},
            "backbone": {"name": "qwen25_vl", "model_id": "fixture"},
            "pipeline": {"stages": []},
            "analysis": {"tasks": ["task"], "probes": []},
            "train": {},
            "eval": {},
            "data": {},
            "output": {
                "run_dir": str(run_dir),
                "manifest_path": str(manifest),
            },
        }
    )
    called = False

    def runtime_loader(*_args: Any, **_kwargs: Any) -> tuple[Any, Any]:
        nonlocal called
        called = True
        return object(), object()

    with pytest.raises(BackendIntegrationError, match="fresh output paths"):
        execute_analysis(
            config,
            SimpleNamespace(probes=[{"name": "fixture", "method": "fixture"}]),
            runtime_loader=runtime_loader,
        )

    assert called is False


def test_representation_session_writes_finite_minimal_artifacts(
    tmp_path: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attention = SimpleNamespace(
        num_heads=2,
        num_key_value_heads=1,
        head_dim=2,
        config=SimpleNamespace(
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=2,
            hidden_size=4,
        ),
    )
    layer = SimpleNamespace(self_attn=attention, mlp=object())
    handles = {
        "llm": SimpleNamespace(),
        "llm.layer[0]": layer,
        "llm.layer[0].attn": attention,
        "llm.layer[0].ffn": layer.mlp,
        "llm.final_norm": object(),
        "lm_head": object(),
    }
    backbone = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_vl"),
        model_adapter=SimpleNamespace(
            structural_handles=lambda _model: handles
        ),
    )
    runtime = SimpleNamespace(
        model=object(),
        model_path_or_id="Qwen/Qwen2.5-VL-3B-Instruct",
        revision="revision",
    )
    session = RepresentationDynamics().open(
        backbone=backbone,
        runtime=runtime,
        config={
            "name": "representation_dynamics",
            "layers": "all",
            "token_groups": ["vision"],
            "metrics": ["cosine"],
            "save_raw": True,
        },
        phase="pre_compression",
        output_dir=str(tmp_path),
    )
    report = {
        "sample_id": "sample",
        "task_name": "task",
        "layers": {
            "0": {
                comparison: {
                    "vision": {
                        "cosine": 0.5,
                        "token_count": 2,
                        "status": "finite",
                    }
                }
                for comparison in (
                    "attn_res",
                    "attn_pure",
                    "ffn_res",
                    "ffn_pure",
                    "full_layer_res",
                )
            }
        },
    }
    original_add_success = session.records.add_success

    def fail_commit(_report: Mapping[str, Any]) -> None:
        raise ValueError("commit failed")

    monkeypatch.setattr(session, "_process_sample", lambda _sample: report)
    monkeypatch.setattr(session.records, "add_success", fail_commit)
    with pytest.raises(ValueError, match="commit failed"):
        session.process({"sample_id": "source", "task_name": "task"})
    assert session.records.skipped_count == 0

    monkeypatch.setattr(session.records, "add_success", original_add_success)
    session.records.add_success(report)

    record = session.finish()
    session.close()

    assert record.sample_count == 1
    assert record.summary["finite_aggregate_values"] == 5
    assert set(record.artifact_paths) == {"aggregate", "per_sample"}
    assert all(Path(path).is_file() for path in record.artifact_paths.values())
    aggregate = json.loads(
        Path(record.artifact_paths["aggregate"]).read_text(encoding="utf-8")
    )
    assert aggregate["payload"]["task_aggregates"]["task"]["sample_count"] == 1
    raw = json.loads(
        Path(record.artifact_paths["per_sample"]).read_text(encoding="utf-8")
    )
    assert raw["payload"]["samples"][0]["sample_id"] == "sample"
