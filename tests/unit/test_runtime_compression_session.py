"""CPU-only lifecycle tests for generic runtime compression sessions."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.support import SupportStatus
from mllm_comp.core.vocabulary import (
    CompressionMode,
    ExecutionPhase,
    MediaAlignment,
    Modality,
    ModalityCouplingPolicy,
    Placement,
)
from mllm_comp.pipeline.runtime_session import (
    CompressionActivationRecord,
    CompressionSession,
    RuntimeCompressionDispatcher,
    refresh_runtime_compression_metadata,
)


class _Activation:
    def __init__(
        self,
        name: str,
        events: list[str],
        *,
        fail_install: bool = False,
        fail_uninstall: bool = False,
    ) -> None:
        self.name = name
        self.events = events
        self.fail_install = fail_install
        self.fail_uninstall = fail_uninstall

    def install(self) -> None:
        self.events.append(f"install:{self.name}")
        if self.fail_install:
            raise RuntimeError(f"cannot install {self.name}")

    def uninstall(self) -> None:
        self.events.append(f"uninstall:{self.name}")
        if self.fail_uninstall:
            raise RuntimeError(f"cannot uninstall {self.name}")

    def runtime_metadata(self) -> dict[str, Any]:
        return {"executor": self.name}


class _LayoutActivation(_Activation):
    def __init__(self, name: str, events: list[str]) -> None:
        super().__init__(name, events)
        self.records: list[dict[str, Any]] = []

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            **super().runtime_metadata(),
            "layout_delta_records": list(self.records),
        }


def _record(
    name: str,
    events: list[str],
    *,
    order: int,
    dependencies: tuple[str, ...] = (),
    fail_install: bool = False,
) -> CompressionActivationRecord:
    return CompressionActivationRecord(
        name=name,
        activation=_Activation(
            name,
            events,
            fail_install=fail_install,
        ),
        binding=f"fixture:{name}",
        dependencies=dependencies,
        config_order=order,
    )


def _runtime_v2_case(
    *,
    support_binding: str = "qwen3_vl.visionzip.image",
    conditioning_modalities: set[Modality] | None = None,
    required_handles: frozenset[str] = frozenset(),
    available_handles: dict[str, Any] | None = None,
) -> tuple[RuntimeCompressionDispatcher, Any, Any, Any]:
    binding_name = "qwen3_vl.visionzip.image"
    binding = SimpleNamespace(
        name=binding_name,
        backbone="qwen3_vl",
        variant="visionzip.image",
        supported_phases=frozenset({ExecutionPhase.GENERATE}),
        executor_factory=lambda **_: _Activation("visionzip", []),
        metadata={},
        prepare_after_slots=(),
        exclusive_runtime_slot=None,
        dependencies=(),
        conflicts=frozenset(),
        target_modalities=frozenset({Modality.IMAGE}),
        targets=(
            SimpleNamespace(
                placement=Placement.PROJECTOR_LEVEL,
                required_handles=required_handles,
            ),
        ),
        required_handles=required_handles,
    )
    algorithm = SimpleNamespace(
        name="visionzip",
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
    )
    variant = SimpleNamespace(
        name="visionzip.image",
        algorithm="visionzip",
        target_modalities=frozenset({Modality.IMAGE}),
        conditioning_modalities=frozenset(),
        preserved_modalities=frozenset(),
        coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
        media_alignment=MediaAlignment.NONE,
        supported_rules=frozenset(),
        resolve_config=lambda config: config,
    )
    registry = SimpleNamespace(
        binding=lambda name: binding,
        algorithm=lambda name: algorithm,
        variant=lambda name: variant,
        support=SimpleNamespace(
            get=lambda identity: SimpleNamespace(
                status=SupportStatus.ACTIVE,
                binding=support_binding,
            )
        ),
    )
    stage = SimpleNamespace(
        name="compress_image",
        enabled=True,
        uses_v2_contract=True,
        algorithm="visionzip",
        plugin="visionzip",
        variant="visionzip.image",
        binding=binding_name,
        bindings=[],
        placement=Placement.PROJECTOR_LEVEL,
        target_modalities={Modality.IMAGE},
        conditioning_modalities=(
            set() if conditioning_modalities is None else conditioning_modalities
        ),
        preserved_modalities=set(),
        coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
        media_alignment=MediaAlignment.NONE,
        execution_phase=ExecutionPhase.GENERATE,
        depends_on=[],
        config={},
    )
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))
    model_adapter = None
    if available_handles is not None:
        model_adapter = SimpleNamespace(
            structural_handles=lambda model: available_handles
        )
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen3_vl"),
        model_adapter=model_adapter,
    )
    runtime = SimpleNamespace(model=object(), metadata={})
    return RuntimeCompressionDispatcher(contract_registry=registry), config, facade, runtime


def test_session_uses_dependency_then_config_order_and_uninstalls_in_reverse() -> None:
    events: list[str] = []
    session = CompressionSession(
        [
            _record("alpha", events, order=20),
            _record("beta", events, order=10),
            _record("gamma", events, order=0, dependencies=("alpha",)),
        ]
    )

    session.install()

    assert session.install_order == ("beta", "alpha", "gamma")
    assert events == ["install:beta", "install:alpha", "install:gamma"]
    assert session.runtime_metadata()["active_activations"] == [
        "beta",
        "alpha",
        "gamma",
    ]

    session.uninstall()

    assert events[-3:] == [
        "uninstall:gamma",
        "uninstall:alpha",
        "uninstall:beta",
    ]
    assert session.status == "uninstalled"


def test_session_rolls_back_prior_and_partially_installed_activations() -> None:
    events: list[str] = []
    session = CompressionSession(
        [
            _record("first", events, order=0),
            _record("broken", events, order=1, fail_install=True),
            _record("never", events, order=2),
        ]
    )

    with pytest.raises(BackendIntegrationError, match="broken.*failed"):
        session.install()

    assert events == [
        "install:first",
        "install:broken",
        "uninstall:broken",
        "uninstall:first",
    ]
    assert session.status == "rolled_back"
    assert session.runtime_metadata()["active_activations"] == []


def test_session_rejects_missing_dependencies_before_mutating_any_executor() -> None:
    events: list[str] = []

    with pytest.raises(BackendIntegrationError, match="dependencies are missing"):
        CompressionSession(
            [_record("only", events, order=0, dependencies=("absent",))]
        )

    assert events == []


def test_dispatcher_resolves_v2_binding_without_installing_it() -> None:
    events: list[str] = []
    factory_calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> _Activation:
        factory_calls.append(kwargs)
        return _Activation("visionzip", events)

    binding = SimpleNamespace(
        name="qwen3_vl.visionzip.image",
        backbone="qwen3_vl",
        variant="visionzip.image",
        supported_phases=frozenset({ExecutionPhase.GENERATE}),
        executor_factory=factory,
        metadata={},
        prepare_after_slots=(),
        exclusive_runtime_slot=None,
        dependencies=(),
        conflicts=frozenset(),
        target_modalities=frozenset({Modality.IMAGE}),
        targets=(
            SimpleNamespace(
                placement=Placement.PROJECTOR_LEVEL,
                required_handles=frozenset(),
            ),
        ),
        required_handles=frozenset(),
    )
    algorithm = SimpleNamespace(
        name="visionzip",
        supported_placements=frozenset({Placement.PROJECTOR_LEVEL}),
    )
    variant = SimpleNamespace(
        name="visionzip.image",
        algorithm="visionzip",
        target_modalities=frozenset({Modality.IMAGE}),
        conditioning_modalities=frozenset(),
        preserved_modalities=frozenset(),
        coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
        media_alignment=MediaAlignment.NONE,
        supported_rules=frozenset(),
        resolve_config=lambda config: config,
    )
    registry = SimpleNamespace(
        binding=lambda name: binding,
        algorithm=lambda name: algorithm,
        variant=lambda name: variant,
        support=SimpleNamespace(
            get=lambda identity: SimpleNamespace(
                status=SupportStatus.ACTIVE,
                binding=binding.name,
            )
        ),
    )
    stage = SimpleNamespace(
        name="compress_image",
        enabled=True,
        uses_v2_contract=True,
        algorithm="visionzip",
        plugin="visionzip",
        variant="visionzip.image",
        binding=binding.name,
        bindings=[],
        placement=Placement.PROJECTOR_LEVEL,
        target_modalities={Modality.IMAGE},
        conditioning_modalities=set(),
        preserved_modalities=set(),
        coupling_policy=ModalityCouplingPolicy.INDEPENDENT,
        media_alignment=MediaAlignment.NONE,
        execution_phase=ExecutionPhase.GENERATE,
        depends_on=[],
        config={},
    )
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen3_vl")
    )
    runtime = SimpleNamespace(model=object(), metadata={})
    dispatcher = RuntimeCompressionDispatcher(contract_registry=registry)

    session = dispatcher.resolve(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )

    assert session is not None
    assert events == []
    assert factory_calls == []
    session.install()
    assert factory_calls[0]["binding"] is binding
    assert events == ["install:visionzip"]
    activation_metadata = session.runtime_metadata()["activations"][0]
    assert activation_metadata["algorithm"] == "visionzip"
    assert activation_metadata["variant"] == "visionzip.image"


@pytest.mark.parametrize("evidence", ("alternate_binding", "declared", "missing"))
def test_research_runtime_keeps_exact_evidence_without_requiring_it(evidence) -> None:
    dispatcher, config, facade, runtime = _runtime_v2_case(
        support_binding="unverified.alternate.binding"
    )
    if evidence == "declared":
        dispatcher.contract_registry.support.get = lambda identity: SimpleNamespace(
            status=SupportStatus.DECLARED, binding=config.pipeline.stages[0].binding
        )
    elif evidence == "missing":
        from mllm_comp.core.errors import RegistryError

        def missing(identity):
            raise RegistryError("No historical verification")

        dispatcher.contract_registry.support.get = missing

    session = dispatcher.activate(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    assert session.installed
    metadata = session.runtime_metadata()["activations"][0]
    assert metadata["support_status"] == ("declared" if evidence == "declared" else "undeclared")
    assert metadata["development_runtime"] is True
    session.uninstall()


@pytest.mark.parametrize("failure", (None, "missing_handle", "install", "cycle"))
def test_prepare_steps_finish_before_request_targets_are_resolved(failure) -> None:
    dispatcher, config, facade, runtime = _runtime_v2_case(
        required_handles=frozenset({"slot"}), available_handles={"slot": object()}
    )
    events = []
    original, replacement = object(), object()
    runtime.model = SimpleNamespace(slot=original)
    runtime.metadata["structural_handles"] = {"slot": original}
    facade.model_adapter.structural_handles = lambda model: {"slot": model.slot}
    request = config.pipeline.stages[0]
    request_binding = dispatcher.contract_registry.binding(request.binding)
    prepare = SimpleNamespace(**vars(request))
    prepare.name, prepare.binding = "prepare_model", "prepare_slot"
    prepare.execution_phase = ExecutionPhase.PREPARE
    prepare_binding = SimpleNamespace(**vars(request_binding))
    prepare_binding.name = prepare.binding
    prepare_binding.supported_phases = frozenset({ExecutionPhase.PREPARE})

    class Rewrite(_Activation):
        rollback_safe = False

        def install(self):
            super().install()
            runtime.model.slot = None if failure == "missing_handle" else replacement

    def prepare_factory(**kwargs):
        assert kwargs["model"].slot is original
        events.append("resolve:prepare")
        result = Rewrite("prepare", events)
        result.rollback_safe = False
        return result

    def request_factory(**kwargs):
        assert kwargs["model"].slot is replacement
        events.append("resolve:request")
        return _Activation("request", events, fail_install=failure == "install")

    prepare_binding.executor_factory = prepare_factory
    request_binding.executor_factory = request_factory
    dispatcher.contract_registry.binding = lambda name: (
        prepare_binding if name == prepare.binding else request_binding
    )
    config.pipeline.stages = [request, prepare]
    kwargs = dict(config=config, facade=facade, runtime=runtime, compression_mode="compressed")
    if failure == "cycle":
        prepare.depends_on = (request.name,)
        with pytest.raises(BackendIntegrationError, match="dependency cycle"):
            dispatcher.resolve(**kwargs)
        assert events == [] and runtime.model.slot is original
        return

    session = dispatcher.resolve(**kwargs)
    assert events == []
    assert session.install_order == (prepare.name, request.name)
    if failure:
        message = "missing structural handles" if failure == "missing_handle" else "request"
        with pytest.raises(BackendIntegrationError, match=message):
            session.install()
        assert session.status == "rollback_incomplete"
        assert events[-1] == "uninstall:prepare"
        assert ("resolve:request" in events) is (failure == "install")
    else:
        session.install()
        assert events == ["resolve:prepare", "install:prepare", "resolve:request", "install:request"]
        session.uninstall()
        assert events[-2:] == ["uninstall:request", "uninstall:prepare"]


def test_runtime_uses_shared_v2_modality_semantics() -> None:
    dispatcher, config, facade, runtime = _runtime_v2_case(
        conditioning_modalities={Modality.TEXT}
    )

    with pytest.raises(BackendIntegrationError, match="conditioning modalities"):
        dispatcher.resolve(
            config=config,
            facade=facade,
            runtime=runtime,
            compression_mode=CompressionMode.COMPRESSED,
        )


def test_runtime_enforces_required_handles_when_facade_exposes_them() -> None:
    dispatcher, config, facade, runtime = _runtime_v2_case(
        required_handles=frozenset({"vision_projector"}),
        available_handles={},
    )

    with pytest.raises(BackendIntegrationError, match="vision_projector"):
        dispatcher.resolve(
            config=config,
            facade=facade,
            runtime=runtime,
            compression_mode=CompressionMode.COMPRESSED,
        )


@pytest.mark.parametrize("backbone", ["qwen25_omni", "qwen3_vl"])
def test_runtime_has_no_implicit_qwen_legacy_binding(backbone) -> None:
    stage = SimpleNamespace(
        name="audio",
        algorithm="audio", plugin="audio", modalities={Modality.AUDIO},
        enabled=True,
        uses_v2_contract=False,
        execution_phase=ExecutionPhase.GENERATE,
    )
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage]))
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name=backbone)
    )

    with pytest.raises(BackendIntegrationError, match="requires a selected compression binding"):
        RuntimeCompressionDispatcher().resolve(
            config=config,
            facade=facade,
            runtime=SimpleNamespace(model=object(), metadata={}),
            compression_mode=CompressionMode.COMPRESSED,
        )


def test_dispatcher_applies_and_restores_qwen3_prepare_operation() -> None:
    from mllm_comp.config.schema import experiment_config_from_dict
    from mllm_comp.pipeline.runtime_compression import activate_runtime_compression

    layer = SimpleNamespace(self_attn=object(), mlp=object())
    original = layer.self_attn
    handles = {
        "llm": SimpleNamespace(layers=[layer]),
        "llm.layer[0]": layer,
        "llm.layer[0].attn": original,
    }
    replacement = object()

    class _Applier:
        def apply(self, *, structural_handles: Any, actions: Any, **kwargs: Any) -> Any:
            del kwargs
            assert actions[0].strategy_name == "attn_identity"
            layer.self_attn = replacement
            structural_handles[actions[0].module_path] = replacement
            return SimpleNamespace(
                installed_handles=[actions[0].module_path],
                metadata={"operation_count": 1},
            )

    config = experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl"},
        "pipeline": {"stages": [{
            "name": "replace-attention", "algorithm": "qwen3_vl_attention_replacement",
            "operation": {"type": "replace"},
            "schedule": {"targets": [{"site": "llm.layer[0].attn"}]},
            "config": {"strategy": "attn_identity"},
        }]},
    })
    runtime = SimpleNamespace(model=SimpleNamespace(training=False), metadata={})
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen3_vl"),
        model_adapter=SimpleNamespace(structural_handles=lambda model: handles),
        operation_applier=_Applier(),
    )
    session = activate_runtime_compression(
        config=config, facade=facade, runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.GENERATE,
    )

    assert session is not None
    assert config.pipeline.stages[0].uses_v2_contract
    assert layer.self_attn is replacement
    metadata = runtime.metadata["compression"]["activations"][0]
    assert metadata["installed_handles"] == ["llm.layer[0].attn"]
    assert metadata["binding"] == "qwen3_vl_attention_replacement.shared_llm.qwen3_vl"
    assert metadata["execution_phase"] == "prepare"
    session.uninstall()
    assert layer.self_attn is original
    assert handles["llm.layer[0].attn"] is original



def test_dispatcher_publishes_session_metadata_on_runtime() -> None:
    events: list[str] = []
    stage = SimpleNamespace(
        name="legacy",
        algorithm="legacy", plugin="legacy", modalities=set(),
        enabled=True,
        uses_v2_contract=False,
        execution_phase=ExecutionPhase.GENERATE,
        binding=None,
        bindings=[],
        depends_on=[],
    )
    activation = _Activation("legacy", events)

    class _LegacyBinding:
        def resolve(self, **kwargs: Any) -> CompressionActivationRecord:
            return CompressionActivationRecord(
                name=kwargs["stage"].name,
                activation=activation,
                binding="legacy:fixture",
                config_order=kwargs["config_order"],
            )

    runtime = SimpleNamespace(model=object(), metadata={})
    session = RuntimeCompressionDispatcher(
        legacy_bindings={"fixture": _LegacyBinding()}
    ).activate(
        config=SimpleNamespace(pipeline=SimpleNamespace(stages=[stage])),
        facade=SimpleNamespace(
            capabilities=lambda: SimpleNamespace(name="fixture")
        ),
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )

    assert session is runtime.metadata["_compression_session"]
    assert runtime.metadata["compression"]["status"] == "active"
    assert runtime.metadata["compression"]["backbone"] == "fixture"
    assert events == ["install:legacy"]


def test_runtime_refresh_publishes_post_callback_layout_evidence() -> None:
    events: list[str] = []
    activation = _LayoutActivation("visionzip", events)
    record = CompressionActivationRecord(
        name="visionzip-stage",
        activation=activation,
        binding="visionzip.image.qwen3_vl",
        metadata={"stage_name": "visionzip-stage"},
    )
    session = CompressionSession([record])
    runtime = SimpleNamespace(
        metadata={"_compression_session": session},
    )
    session.install()
    activation.records.append(
        {
            "source_count": 64,
            "result_count": 32,
            "operation": "merge",
            "media_unit": "image_feature",
            "mapping_sha256": "abc",
            "old_to_new": list(range(64)),
        }
    )

    refreshed = refresh_runtime_compression_metadata(runtime)

    assert refreshed is runtime.metadata["compression"]
    assert refreshed["layout_delta_summary"]["record_count"] == 1
    summary = refreshed["layout_delta_summary"]["records"][0]
    assert summary["stage_name"] == "visionzip-stage"
    assert summary["mapping_sha256"] == "abc"
    assert "old_to_new" not in summary
    assert refreshed["activations"][0]["layout_delta_records"][0][
        "old_to_new"
    ] == list(range(64))


@pytest.mark.parametrize("runtime_error", (None, "handles", "phase", "executor"))
def test_resolved_contract_reuse_keeps_runtime_gates(monkeypatch, runtime_error):
    from mllm_comp.pipeline.definition import CompressionDefinition
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract
    import mllm_comp.pipeline.runtime_session as module

    dispatcher, config, facade, runtime = _runtime_v2_case(
        required_handles=frozenset({"vision_tower"}),
        available_handles={"vision_tower": object()},
    )
    stage = config.pipeline.stages[0]
    projection = validate_v2_stage_contract(
        stage, registry=dispatcher.contract_registry, backbone_name="qwen3_vl"
    )
    definition = CompressionDefinition(
        compression_mode=CompressionMode.COMPRESSED,
        stages=(stage,),
        projections={stage.name: projection},
    )

    def unexpected_resolution(*args, **kwargs):
        pytest.fail("Runtime rebuilt the already resolved static contract")

    monkeypatch.setattr(module, "validate_v2_stage_contract", unexpected_resolution)
    phase = ExecutionPhase.GENERATE
    if runtime_error == "handles":
        facade.model_adapter.structural_handles = lambda model: {}
    elif runtime_error == "phase":
        phase = ExecutionPhase.FORWARD
    elif runtime_error == "executor":
        projection.bindings[0].executor_factory = None

    kwargs = dict(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode="compressed",
        execution_phase=phase,
        definition=definition,
    )
    if runtime_error:
        message = {
            "handles": "missing structural handles",
            "phase": "runtime requires",
            "executor": "no executor factory",
        }[runtime_error]
        with pytest.raises(BackendIntegrationError, match=message):
            dispatcher.activate(**kwargs)
    else:
        session = dispatcher.activate(**kwargs)
        assert session.installed
        assert session.activations[0].binding == projection.bindings[0].name
        session.uninstall()
