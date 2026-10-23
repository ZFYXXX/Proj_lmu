"""Acceptance test for adding a method without editing shared execution code."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.compressors.extensions import compression_contracts, register_method
from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
from mllm_comp.config import config_to_dict, load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.experiments import run_experiment
from examples.compression_extensions.first_k_audio import (
    ALGORITHM as _ALGORITHM,
    FirstKStrategy as _ResearcherFirstKStrategy,
)
from mllm_comp.core.state import ForwardState, TokenSpan
from mllm_comp.core.vocabulary import (
    CompressionMode,
    Placement,
)
from mllm_comp.pipeline.definition import resolve_compression_definition
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.run_manifest import build_run_manifest
from mllm_comp.restoration import restore_config


_EXTENSION = "examples.compression_extensions.first_k_audio:register"


class _Handle:
    def remove(self) -> None:
        return None


class _Projector:
    def register_forward_pre_hook(self, hook: Any) -> _Handle:
        self.pre_hook = hook
        return _Handle()


class _AudioTower:
    def __init__(self) -> None:
        self.layers: list[Any] = []
        self.proj = _Projector()

    def _get_feat_extract_output_lengths(self, value: Any) -> Any:
        return value, value


class _Thinker:
    def __init__(self) -> None:
        self.audio_tower = _AudioTower()

    def get_rope_index(self, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs


def _registry() -> Any:
    return compression_contracts([_EXTENSION])


def _config(*, phase: str = "generate") -> Any:
    stage = to_jsonable(_stage())
    stage["execution_phase"] = phase
    return experiment_config_from_dict({
        "experiment": {"name": "research-first-k", "seed": 42},
        "backbone": {"name": "qwen25_omni", "model_id": "Qwen/fixture"},
        "pipeline": {"extensions": [_EXTENSION], "stages": [stage]},
        "eval": {"compression_mode": "compressed", "tasks": ["fixture"]},
        "train": {"compression_mode": "compressed", "args": {"datasets": ["fixture"]}},
    })


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "researcher_first_k",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": _ALGORITHM,
            "algorithm": _ALGORITHM,
            "variant": f"{_ALGORITHM}.audio",
            "binding": f"{_ALGORITHM}.audio.qwen25_omni",
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "audio_encoder_output_pre_projector",
            "modalities": ["audio"],
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {"score_source": "uniform", "num_segments": 1},
            "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 1},
        }
    )


def _state(torch: Any) -> ForwardState:
    input_ids = torch.tensor([[1, 12, 12, 12, 12, 2]])
    attention_mask = torch.ones_like(input_ids)
    return ForwardState(
        input_ids=input_ids,
        attention_mask=attention_mask,
        native_model_inputs={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        },
        modality_token_spans=[
            TokenSpan(0, 1, 5, modality="audio", role="observation")
        ],
        modality_metadata={
            "token_ids": {"audio": 12},
            "pad_token_id": 0,
            "padding_side": "right",
            "expected_token_counts_by_sample": [{"audio": 4}],
            "semantic_token_labels_by_sample": [
                ["ins_special", "audio", "audio", "audio", "audio", "instruction"]
            ],
        },
    )


def _activate(config: Any = None) -> tuple[Any, Any]:
    config = _config() if config is None else config
    model = SimpleNamespace(thinker=_Thinker())
    runtime = SimpleNamespace(model=model, metadata={})
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(
            structural_handles=lambda active_model: {
                "thinker": active_model.thinker,
                "audio_encoder": active_model.thinker.audio_tower,
                "audio_projector": active_model.thinker.audio_tower.proj,
            }
        ),
    )
    session = activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=config.pipeline.stages[0].execution_phase,
    )
    assert session is not None
    return session, session.activations[0].activation


def test_registered_researcher_strategy_resolves_through_v2_contract() -> None:
    session, controller = _activate()
    try:
        assert isinstance(controller.strategy, _ResearcherFirstKStrategy)
        assert session.runtime_metadata()["activations"][0]["algorithm"] == (
            _ALGORITHM
        )
    finally:
        session.uninstall()


def test_registered_researcher_strategy_runs_through_unchanged_runtime() -> None:
    torch = pytest.importorskip("torch")
    session, controller = _activate()

    active_state, plan = controller.prepare_state(_state(torch))
    active_span = active_state.modality_token_spans[0]
    assert active_span.end - active_span.start == 2
    hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2).requires_grad_()
    try:
        with controller.request(plan):
            output = controller._compress_projector_tensor(hidden)
    finally:
        session.uninstall()

    assert output.tolist() == hidden[:2].tolist()
    output.sum().backward()
    torch.testing.assert_close(hidden.grad, torch.tensor([[1., 1.], [1., 1.], [0., 0.], [0., 0.]]))
    assert controller.last_report["plugin"] == _ALGORITHM
    assert controller.last_report["selection"] == "chronological_first_k"


def test_extensions_are_explicit_and_do_not_leak_into_builtin_registry() -> None:
    builtin = builtin_multimodal_compression_contracts()
    assert compression_contracts() is builtin
    first, second = _registry(), _registry()
    assert first is not second and first is not builtin
    assert first.algorithm(_ALGORITHM).kernel().name == _ALGORITHM
    assert builtin.algorithm("segmentwise_topk_audio").kernel().name == "segmentwise_topk_audio"
    assert set(first.algorithm(_ALGORITHM).metadata) == {"spec_factory"}
    assert first.algorithm(_ALGORITHM).stage_validator == builtin.algorithm("segmentwise_topk_audio").stage_validator
    assert not first.support.find(algorithm=_ALGORITHM)
    with pytest.raises(RegistryError, match="Unknown algorithm"):
        builtin.algorithm(_ALGORITHM)
    first.register_algorithm(replace(first.algorithm(_ALGORITHM), name="local-only"))
    with pytest.raises(RegistryError, match="Unknown algorithm"):
        second.algorithm("local-only")


@pytest.mark.parametrize("entry, message", [
    ("not-an-entry", "expected package.module:register"),
    ("missing_research_module:register", "Cannot load compression extension"),
    ("examples.compression_extensions.first_k_audio:missing", "Cannot load compression extension"),
])
def test_unknown_extension_is_a_configuration_error(entry: str, message: str) -> None:
    with pytest.raises(ConfigError, match=message):
        compression_contracts([entry])


def test_duplicate_extension_reports_the_registration_and_algorithm() -> None:
    with pytest.raises(ConfigError, match="first_k_audio:register.*Duplicate algorithm"):
        compression_contracts([_EXTENSION, _EXTENSION])


def test_method_registration_requires_an_explicit_execution_adapter() -> None:
    registry = compression_contracts().clone()
    algorithm = replace(registry.algorithm("fastv"), name="unadapted_kernel")
    with pytest.raises(RegistryError, match="strategy_factory or executor_factory"):
        register_method(registry, algorithm=algorithm, variant="unadapted_kernel.image",
                        reuse_bindings=["fastv.image.qwen25_vl"])
    with pytest.raises(RegistryError, match="Unknown algorithm"):
        registry.algorithm(algorithm.name)


def test_plan_reuse_rejects_an_execution_interface_without_a_spec() -> None:
    registry = compression_contracts().clone()
    with pytest.raises(RegistryError, match="new_selection.*fastv.*AlgorithmDescriptor"):
        register_method(
            registry, algorithm="new_selection", reuse_bindings=["fastv.image.qwen25_vl"],
            strategy_factory=_ResearcherFirstKStrategy,
        )
    with pytest.raises(RegistryError, match="Unknown algorithm"):
        registry.algorithm("new_selection")


def test_reusing_a_research_plan_retains_identity_and_conflicts() -> None:
    registry = _registry()
    source = f"{_ALGORITHM}.audio.qwen25_omni"
    register_method(
        registry, algorithm="second_first_k", reuse_bindings=[source],
        strategy_factory=_ResearcherFirstKStrategy,
    )
    assert registry.algorithm("second_first_k").kernel().name == "second_first_k"
    assert registry.algorithm(_ALGORITHM).kernel().name == _ALGORITHM
    binding = registry.binding("second_first_k.audio.qwen25_omni")
    assert {source, "segmentwise_topk_audio.audio.qwen25_omni"} <= binding.conflicts
    assert not registry.support.find(algorithm="second_first_k")


@pytest.mark.parametrize("other", ["fastv.image.qwen25_vl", "researcher_alternating_image.image.qwen25_vl"])
def test_derived_methods_keep_conflicts_with_source_and_other_research_methods(other) -> None:
    from examples.compression_extensions import image_selection as research
    from mllm_comp.pipeline.v2_validation import V2ContractValidationError, validate_v2_stage_contract

    registry = compression_contracts(["examples.compression_extensions.image_selection:register"])
    algorithm = replace(registry.algorithm(research.ALGORITHM), name="second_selection")
    register_method(
        registry, algorithm=algorithm, variant="second_selection.image",
        reuse_bindings=["fastv.image.qwen25_vl"],
        strategy_factory=research.AlternatingImageMethod, config_validator=research.resolve_config,
    )
    stage = replace(research.experiment_config("qwen25_vl").pipeline.stages[0],
                    algorithm=algorithm.name, plugin=algorithm.name, variant="second_selection.image",
                    binding="second_selection.image.qwen25_vl")
    with pytest.raises(V2ContractValidationError, match="conflicts with selected bindings"):
        validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_vl",
                                  selected_binding_names=(stage.binding, other))


@pytest.mark.parametrize("source", ["llamavid", "longvu"])
def test_reused_video_mechanism_does_not_claim_source_method_positions(source, monkeypatch) -> None:
    from functools import partial
    from examples.compression_extensions import stride_video as research
    from mllm_comp.backbones.video_projector import VideoProjectorExecutor
    from mllm_comp.compressors import extensions
    from mllm_comp import run_manifest
    from mllm_comp.core.vocabulary import TokenOperation

    registry = compression_contracts().clone()
    algorithm = replace(registry.algorithm("vqtoken"), name="new_video_selection",
                        kernel=research.stride_video_tokens, operations=frozenset({TokenOperation.DROP}),
                        stage_validator=None, metadata={})
    original = registry.binding(f"{source}.video.qwen25_vl")
    assert original.targets[0].metadata["application_positions"]
    register_method(
        registry, algorithm=algorithm, variant="new_video_selection.video",
        reuse_bindings=[original.name],
        executor_factory=partial(VideoProjectorExecutor, backbone="qwen25_vl",
                                 replay=research.replay_stride_video_features),
        config_validator=research.resolve_config, target_name="ordinal_sample",
    )
    monkeypatch.setattr(extensions, "compression_contracts", lambda entries: registry)
    monkeypatch.setattr(run_manifest, "compression_contracts", lambda entries: registry)
    config = experiment_config_from_dict({
        "backbone": {"name": "qwen25_vl"},
        "pipeline": {"stages": [{"name": "selection", "algorithm": algorithm.name,
                                  "execution_phase": "forward"}]},
    })
    record = build_run_manifest(config).compression_stages[0]
    assert "application_positions" not in record.metadata
    assert record.support_status == "undeclared"
    assert record.evidence == []
    assert original.targets[0].metadata["application_positions"]


def test_external_definition_validates_its_registered_provider() -> None:
    config = _config()
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="eval")
    assert definition.registry.algorithm(_ALGORITHM) is definition.projections["researcher_first_k"].algorithm
    config.pipeline.stages[0].schedule["keep_ratio"] = 2.0
    with pytest.raises(ConfigError, match="keep_ratio"):
        resolve_compression_definition(config, compression_mode="compressed", interface="eval")


@pytest.mark.parametrize("task", ["train", "eval"])
def test_first_k_example_plans_and_restores_with_its_method_defaults(tmp_path, monkeypatch, task) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", str(tmp_path / "audio.jsonl"))
    config = load_config(Path(__file__).parents[2] / "examples/compression_extensions/first_k_audio_train.yaml")
    assert config.pipeline.stages[0].config == {"keep_ratio": 0.5}
    result = run_experiment(config, task="train", plan_only=True)
    assert result.status == "planned", result.to_dict()

    # These artifact fixtures exercise registration/recipe restoration, not model weights.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text('{"model_type": "qwen2_5_omni_thinker"}', encoding="utf-8")
    (checkpoint / "model.safetensors").write_bytes(b"fixture")
    (checkpoint / "trainer_state.json").write_text('{"global_step": 2}', encoding="utf-8")
    (tmp_path / "effective_config.json").write_text(json.dumps(config_to_dict(config)), encoding="utf-8")
    (tmp_path / "result.json").write_text(json.dumps({
        "task": "train", "status": "succeeded", "output": {"checkpoint_path": str(checkpoint)},
    }), encoding="utf-8")
    restored = restore_config(tmp_path, task=task)
    assert restored.pipeline.extensions == [_EXTENSION]
    assert restored.pipeline.stages[0].execution_phase.value == ("forward" if task == "train" else "generate")
    session, controller = _activate(restored)
    try:
        assert controller.execution.name == _ALGORITHM
        assert controller.spec.keep_ratio == 0.5
        assert controller.spec.num_segments == 1
        assert controller.spec.score_source == "uniform"
    finally:
        session.uninstall()

    config.pipeline.stages[0].config["keep_ratio"] = 2
    failed = run_experiment(config, task="train", plan_only=True)
    assert failed.status == "failed"
    assert _ALGORITHM in failed.error.message and "keep_ratio" in failed.error.message


@pytest.mark.parametrize("has_stage_validator", [False, True])
def test_definition_does_not_construct_parameterized_research_kernel(
    has_stage_validator: bool, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from mllm_comp.pipeline import definition as definition_module

    constructed = []
    validated = []

    class ParameterizedKernel:
        def __init__(self, scale: float) -> None:
            constructed.append(scale)
            self.scale = scale

        def __call__(self, features: Any) -> Any:
            return features * self.scale

    class KernelWithInstanceValidator(ParameterizedKernel):
        def validate_config(self, config: Any) -> None:
            pytest.fail("Planning must use the explicitly registered stage validator.")

    def validate_stage(config: Any) -> None:
        validated.append(config)
        if config["schedule"]["keep_ratio"] > 1:
            raise ValueError("keep_ratio must not exceed 1")

    registry = _registry()
    name = "parameterized_research"
    algorithm = replace(
        registry.algorithm(_ALGORITHM), name=name,
        kernel=KernelWithInstanceValidator if has_stage_validator else ParameterizedKernel,
        stage_validator=validate_stage if has_stage_validator else None,
    )
    variant = replace(
        registry.variant(f"{_ALGORITHM}.audio"),
        name=f"{name}.audio", algorithm=name,
        config_validator=lambda config: {"scale": config.get("scale", 2.0)},
    )
    registry.register_algorithm(algorithm)
    registry.register_variant(variant)
    binding = replace(
        registry.binding(f"{_ALGORITHM}.audio.qwen25_omni"),
        name=f"{name}.audio.qwen25_omni", variant=variant.name,
    )
    registry.register_binding(binding)
    monkeypatch.setattr(definition_module, "compression_contracts", lambda entries: registry)
    config = _config()
    config.pipeline.stages[0] = replace(
        config.pipeline.stages[0], plugin=name, algorithm=name,
        variant=variant.name, binding=binding.name,
    )
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="eval")
    assert definition.projections["researcher_first_k"].algorithm_config["scale"] == 2.0
    assert not constructed
    assert bool(validated) is has_stage_validator
    if has_stage_validator:
        config.pipeline.stages[0].schedule["keep_ratio"] = 2.0
        with pytest.raises(ConfigError, match="keep_ratio must not exceed 1"):
            resolve_compression_definition(config, compression_mode="compressed", interface="eval")
        assert not constructed


def test_research_identity_can_reuse_the_existing_layer_position_resolver() -> None:
    from mllm_comp.backbones.qwen25_omni.v2_compression import (
        resolve_qwen25_omni_compression_execution,
        resolve_qwen25_omni_stage_execution,
    )
    from mllm_comp.pipeline.v2_validation import validate_v2_stage_contract

    registry = compression_contracts().clone()
    algorithm = replace(registry.algorithm("uniform_sampling_audio"), name="research_uniform")
    original = registry.binding("uniform_sampling_audio.audio.qwen25_omni")
    register_method(
        registry, algorithm=algorithm, variant="research_uniform.audio", reuse_bindings=[original.name],
        strategy_factory=original.strategy_factory, position_resolver=original.position_resolver,
    )
    variant = registry.variant("research_uniform.audio")
    binding = registry.binding("research_uniform.audio.qwen25_omni")
    stage = replace(
        _stage(), plugin=algorithm.name, algorithm=algorithm.name,
        variant=variant.name, binding=binding.name, placement=Placement.LLM_LEVEL,
        criterion={}, operation={"type": "keep", "target_site": "llm_decoder_layer_output_audio_tokens", "layers": [2]},
    )
    projection = validate_v2_stage_contract(stage, registry=registry, backbone_name="qwen25_omni")
    execution = resolve_qwen25_omni_stage_execution(stage, registry=registry)
    assert projection.algorithm is algorithm
    assert execution.algorithm is algorithm
    assert execution.executor_family == "llm_token"
    assert execution.binding.metadata["position_changed"] is True
    resolved_again = resolve_qwen25_omni_compression_execution(
        stage=stage, binding=projection.bindings[0], algorithm=algorithm, variant=variant,
    )
    assert resolved_again.binding.metadata["position_changed"] is True


@pytest.mark.parametrize("method", ["first_k", "ltbm"])
def test_research_identity_survives_cli_python_plan_manifest_and_swift_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: Any, method: str,
) -> None:
    from mllm_comp.cli.train import main

    config = _config(phase="forward") if method == "first_k" else _ltbm_rule_config(phase="forward")
    algorithm_name = config.pipeline.stages[0].algorithm
    config.output.run_dir = str(tmp_path / "run")
    config.experiment.run_id = "fixture-research"
    source = tmp_path / "config.json"
    source.write_text(json.dumps(config_to_dict(config)), encoding="utf-8")
    planned = run_experiment(config, task="train", plan_only=True)
    assert planned.status == "planned", planned.to_dict()
    assert main(["--config", str(source), "--plan_only"]) == 0
    cli = json.loads(capsys.readouterr().out)
    encoded = planned.output["plan"].environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"]
    assert cli["output"]["plan"]["environment"]["MLLM_COMP_TRAIN_COMPRESSION_SPEC"] == encoded
    assert not Path(config.output.run_dir).exists()
    manifest = build_run_manifest(config)
    record = manifest.compression_stages[0]
    assert record.algorithm == algorithm_name
    assert record.support_status == "undeclared"
    assert manifest.effective_config["pipeline"]["extensions"] == config.pipeline.extensions

    # Import the real plugin, then exercise its loader patch with a fake native
    # loader. This verifies registration transport, not backend training.
    loader = type("Loader", (), {"_mllm_comp_training_loader_patch": True})
    monkeypatch.setitem(sys.modules, "swift.model.models.qwen", SimpleNamespace(Qwen2_5OmniLoader=loader))
    plugin_path = Path(__file__).parents[2] / "mllm_comp/integrations/ms_swift/training_plugin.py"
    spec = importlib.util.spec_from_file_location("_research_swift_plugin", plugin_path)
    plugin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(plugin)
    monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_SPEC", encoded)
    monkeypatch.setenv("MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR", str(tmp_path / "evidence"))
    payload, _ = plugin._load_spec()
    assert payload["extensions"] == config.pipeline.extensions
    model = SimpleNamespace(thinker=_Thinker(), forward=lambda **kwargs: None)
    delattr(loader, "_mllm_comp_training_loader_patch")
    loader.get_model = lambda self, model_dir: model
    monkeypatch.setattr(plugin, "_patch_qwen_template", lambda: None)
    monkeypatch.setattr(plugin, "_patch_swift_output_dir", lambda recorder: None)
    monkeypatch.setattr(plugin, "_patch_swift_training_lifecycle", lambda: None)
    plugin._patch_qwen_loader()
    assert loader().get_model("fixture") is model
    controller = model._mllm_comp_training_session.active_records[0].activation
    try:
        assert controller.execution.algorithm.name == algorithm_name
        if method == "first_k":
            assert isinstance(controller.strategy, _ResearcherFirstKStrategy)
        else:
            assert set(controller.strategy.rules) == {"grouper", "aggregator"}
            assert controller.strategy.rule_specs == record.metadata["rules"]
    finally:
        plugin._close_model_bridge(model)


def _ltbm_rule_config(*, phase: str = "generate") -> Any:
    return experiment_config_from_dict({
        "experiment": {"name": "research-ltbm-rules"},
        "backbone": {"name": "qwen25_omni", "model_id": "Qwen/fixture"},
        "pipeline": {
            "extensions": ["examples.compression_extensions.ltbm_audio:register"],
            "stages": [{
                "name": "research-merge", "algorithm": "researcher_ltbm_audio",
                "execution_phase": phase, "config": {"keep_ratio": 0.5, "window_size": 1},
                "operation": {"rules": {
                    "grouper": {"callable": "mllm_comp.compressors.rule_examples:ltbm_nearest_pairs"},
                    "aggregator": {"callable": "mllm_comp.compressors.rule_examples:ltbm_destination_weighted_mean",
                                   "params": {"destination_weight": 3.0}},
                }},
            }],
        },
        "eval": {"compression_mode": "compressed", "tasks": ["fixture"]},
        "train": {"compression_mode": "compressed", "args": {"datasets": ["fixture"]}},
    })


def test_cloned_variant_preserves_consumable_rules_in_plan_manifest_and_activation() -> None:
    config = _ltbm_rule_config()
    definition = resolve_compression_definition(config, compression_mode="compressed", interface="eval")
    projection = definition.projections["research-merge"]
    assert projection.variant.supported_rules == frozenset({"grouper", "aggregator"})
    assert not definition.registry.support.find(algorithm="researcher_ltbm_audio")
    assert run_experiment(config, task="eval", plan_only=True).status == "planned"
    record = build_run_manifest(config).compression_stages[0]
    assert record.metadata["rules_changed"] is True
    assert record.metadata["rules"]["aggregator"]["params"] == {"destination_weight": 3.0}
    assert record.metadata["support_claim_is_exact"] is False
    session, controller = _activate(config)
    try:
        assert set(controller.strategy.rules) == {"grouper", "aggregator"}
        assert controller.strategy.rule_specs == record.metadata["rules"]
    finally:
        session.uninstall()
    config.pipeline.stages[0].operation["rules"]["selector"] = {
        "callable": "mllm_comp.compressors.rule_examples:fastv_uniform_select",
    }
    with pytest.raises(ConfigError, match="does not support.*selector"):
        resolve_compression_definition(config, compression_mode="compressed", interface="eval")


def test_cloned_variant_rules_change_runtime_values_and_preserve_replay_gradients() -> None:
    torch = pytest.importorskip("torch")
    session, controller = _activate(_ltbm_rule_config())
    values = torch.tensor([[0.], [10.], [20.], [30.]], requires_grad=True)
    try:
        _, plan = controller.prepare_state(_state(torch))
        with controller.request(plan):
            output = controller._compress_projector_tensor(values)
        torch.testing.assert_close(output, torch.tensor([[7.5], [27.5]]))
        output.sum().backward()
        torch.testing.assert_close(values.grad, torch.tensor([[.25], [.75], [.25], [.75]]))
        strategy = controller.strategy
        _, _, details = strategy.compress(values, scores=None, keep_count=2, item_index=0)
        replay_values = values.detach().clone().requires_grad_()
        replay = strategy.replay(replay_values, strategy.replay_plan(2, details))
        torch.testing.assert_close(replay, output)
        replay.sum().backward()
        torch.testing.assert_close(replay_values.grad, values.grad)
    finally:
        session.uninstall()


def test_replacement_strategy_declares_only_the_hooks_it_consumes() -> None:
    config = _config()
    config.pipeline.stages[0].operation["rules"] = {
        "selector": {"callable": "mllm_comp.compressors.rule_examples:fastv_uniform_select"},
    }
    with pytest.raises(ConfigError, match="does not support.*selector"):
        resolve_compression_definition(config, compression_mode="compressed", interface="eval")
