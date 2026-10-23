"""Reference token-plan contracts and production runtime compression reporting."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from mllm_comp.backbones.qwen25_omni.implementation import (
    _capture_thinker_generation_timing,
    _runtime_compression_report,
)
from mllm_comp.core.errors import BackendIntegrationError, InvariantError
from mllm_comp.core.actions import OperationApplyResult
from mllm_comp.core.metadata import StageMetadata
from mllm_comp.core.plan import CompressionPlan
from mllm_comp.core.state import ForwardState
from mllm_comp.core.vocabulary import (
    CompressionMethod,
    CompressionMode,
    ExecutionPhase,
    Modality,
    ObjectFamily,
    Placement,
    TokenRole,
)
from mllm_comp.pipeline.runner import CompressionPipeline
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.pipeline.stage import CompressionStageConfig


class _Plugin:
    def validate_config(self, config: dict[str, Any]) -> None:
        assert config["_stage_name"] == "first"

    def plan(self, *, state: ForwardState, context: Any, config: dict[str, Any]) -> CompressionPlan:
        assert context.phase == "eval"
        assert context.structural_handles == {"audio_projector": "projector"}
        return CompressionPlan(
            stage_name=config["_stage_name"],
            placement=Placement.PROJECTOR_LEVEL,
            execution_phase=ExecutionPhase.GENERATE,
            modalities={Modality.AUDIO},
            token_roles={TokenRole.OBSERVATION},
            method=CompressionMethod.TOKEN_COMPRESSION,
            object_family=ObjectFamily.TENSOR_SLOT,
            object_subtype="audio_encoder_output_pre_projector",
            this_stage_metadata=StageMetadata(
                stage_name=config["_stage_name"],
                stats={"planned": True},
            ),
        )


class _ModelAdapter:
    def structural_handles(self, model: Any) -> dict[str, Any]:
        assert model == "model"
        return {"audio_projector": "projector"}


def _stage(*, name: str = "first", depends_on: list[str] | None = None) -> CompressionStageConfig:
    return CompressionStageConfig(
        name=name,
        placement=Placement.PROJECTOR_LEVEL,
        execution_phase=ExecutionPhase.GENERATE,
        plugin="fixture",
        modalities={Modality.AUDIO},
        token_roles={TokenRole.OBSERVATION},
        depends_on=list(depends_on or []),
    )


def test_pipeline_plans_records_and_returns_valid_state() -> None:
    state = ForwardState()
    backbone = SimpleNamespace(
        model_adapter=_ModelAdapter(),
        legalizer=SimpleNamespace(),
    )

    output, records = CompressionPipeline().apply(
        state=state,
        model="model",
        backbone=backbone,
        stages=[_stage()],
        registry={"fixture": _Plugin},
        phase="eval",
    )

    assert output is state
    assert records[0].stats == {"planned": True}
    assert output.stage_history["first"] is records[0]


def test_pipeline_rejects_unmet_stage_dependency() -> None:
    with pytest.raises(InvariantError, match="unmet dependencies"):
        CompressionPipeline().apply(
            state=ForwardState(),
            model=None,
            backbone=SimpleNamespace(model_adapter=_ModelAdapter()),
            stages=[_stage(depends_on=["missing"])],
            registry={"fixture": _Plugin},
            phase="eval",
        )


class _Handle:
    def remove(self) -> None:
        return None


class _Hookable:
    def register_forward_pre_hook(self, hook: Any, **kwargs: Any) -> _Handle:
        self.pre_hook = hook
        self.pre_hook_kwargs = kwargs
        return _Handle()


class _AudioTower:
    def __init__(self) -> None:
        self.layers = [SimpleNamespace(self_attn=_Hookable())]
        self.proj = _Hookable()

    def _get_feat_extract_output_lengths(self, value: Any) -> Any:
        return value


class _Thinker:
    def __init__(self) -> None:
        self.audio_tower = _AudioTower()

    def get_rope_index(self, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs


def test_runtime_activation_installs_qwen_audio_controller(monkeypatch) -> None:
    from mllm_comp.pipeline import definition, runtime_session

    projections = []
    original = definition.validate_v2_stage_contract

    def resolve(*args, **kwargs):
        projection = original(*args, **kwargs)
        projections.append(projection)
        return projection

    monkeypatch.setattr(definition, "validate_v2_stage_contract", resolve)
    monkeypatch.setattr(runtime_session, "validate_v2_stage_contract", resolve)
    stage = CompressionStageConfig.from_dict(
        {
            "name": "audio",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "segmentwise_topk_audio",
            "algorithm": "segmentwise_topk_audio",
            "variant": "segmentwise_topk_audio.audio",
            "binding": "segmentwise_topk_audio.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "audio_encoder_output_pre_projector",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "schedule": {"keep_ratio": 0.25},
        }
    )
    model = SimpleNamespace(thinker=_Thinker())
    runtime = SimpleNamespace(model=model, metadata={})
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage], extensions=[]),
                             backbone=SimpleNamespace(name="qwen25_omni", config={}))
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(
            structural_handles=lambda _model: {
                "thinker": model.thinker,
                "audio_encoder": model.thinker.audio_tower,
                "audio_projector": model.thinker.audio_tower.proj,
            }
        ),
    )

    session = activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )

    assert runtime.metadata["_compression_session"] is session
    assert len(projections) == 1
    controller = session.activations[0].activation
    assert runtime.metadata["compression"]["activations"][0]["target_site"] == (
        "audio_encoder_output_pre_projector"
    )
    assert controller.profile_enabled is False
    assert model.thinker.audio_tower.proj.pre_hook is not None
    session.uninstall()

    profile_model = SimpleNamespace(thinker=_Thinker())
    profile_runtime = SimpleNamespace(
        model=profile_model,
        metadata={"runtime_mode": "profile"},
    )
    profile_facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(
            structural_handles=lambda _model: {
                "thinker": profile_model.thinker,
                "audio_encoder": profile_model.thinker.audio_tower,
                "audio_projector": profile_model.thinker.audio_tower.proj,
            }
        ),
    )
    profile_session = activate_runtime_compression(
        config=config,
        facade=profile_facade,
        runtime=profile_runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )
    profile_controller = profile_session.activations[0].activation
    assert len(projections) == 2
    assert profile_controller.profile_enabled is True
    profile_session.uninstall()


def test_runtime_activation_rejects_compression_without_an_enabled_stage() -> None:
    with pytest.raises(
        BackendIntegrationError,
        match="at least one enabled pipeline stage",
    ):
        activate_runtime_compression(
            config=SimpleNamespace(pipeline=SimpleNamespace(stages=[]),
                                   backbone=SimpleNamespace(name="qwen25_omni", config={})),
            facade=SimpleNamespace(
                capabilities=lambda: SimpleNamespace(name="qwen25_omni")
            ),
            runtime=SimpleNamespace(model=object(), metadata={}),
            compression_mode=CompressionMode.COMPRESSED,
        )


class _EncoderLayer:
    def forward(self, *args: Any, **kwargs: Any) -> tuple[Any, ...]:
        return args


class _EncoderAudioTower:
    def __init__(self) -> None:
        self.n_window = 16
        self.layers = [_EncoderLayer() for _ in range(6)]

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return args, kwargs

    def _get_feat_extract_output_lengths(self, value: Any) -> Any:
        return value


def test_runtime_activation_installs_encoder_level_controller() -> None:
    stage = CompressionStageConfig.from_dict(
        {
            "name": "atome",
            "placement": "encoder_level",
            "execution_phase": "generate",
            "plugin": "adjacent_token_merging_audio",
            "algorithm": "adjacent_token_merging_audio",
            "variant": "adjacent_token_merging_audio.audio",
            "binding": "adjacent_token_merging_audio.audio.qwen25_omni",
            "method": "token_compression",
            "object_family": "tensor_slot",
            "object_subtype": "audio_encoder_layer_post_mhsa_pre_ffn",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "temporal",
            "token_roles": ["observation"],
            "operation": {
                "type": "merge",
                "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
            },
        }
    )
    thinker = _Thinker()
    thinker.audio_tower = _EncoderAudioTower()
    model = SimpleNamespace(thinker=thinker)
    runtime = SimpleNamespace(model=model, metadata={"runtime_mode": "debug"})
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage], extensions=[]),
                             backbone=SimpleNamespace(name="qwen25_omni", config={}))
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=SimpleNamespace(
            structural_handles=lambda _model: {
                "thinker": thinker,
                "audio_encoder": thinker.audio_tower,
                "audio_projector": object(),
            }
        ),
    )

    session = activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
    )

    controller = session.activations[0].activation
    assert controller.active_layers == (2, 5)
    assert controller.debug_enabled is True
    activation_metadata = runtime.metadata["compression"]["activations"][0]
    assert activation_metadata["merge_ratio_per_layer"] == 0.10
    session.uninstall()


class _OperationModelAdapter:
    def structural_handles(self, model: Any) -> dict[str, Any]:
        return {"llm": model.thinker.model}


class _OperationApplier:
    def apply(self, **kwargs: Any) -> OperationApplyResult:
        actions = kwargs["actions"]
        assert [action.module_path for action in actions] == [
            "llm.layer[1]",
            "llm.layer[3]",
        ]
        return OperationApplyResult(
            applied_actions=actions,
            reversible=False,
            metadata={
                "source_num_hidden_layers": 5,
                "result_num_hidden_layers": 3,
                "pruned_source_layers": [1, 3],
                "requires_checkpoint_export": True,
            },
        )


def test_runtime_activation_applies_llm_operation_before_runner() -> None:
    stage = CompressionStageConfig.from_dict(
        {
            "name": "operation",
            "placement": "llm_level",
            "execution_phase": "prepare",
            "plugin": "structured_llm_layer_pruning",
            "algorithm": "structured_llm_layer_pruning",
            "variant": "structured_llm_layer_pruning.audio",
            "binding": "structured_llm_layer_pruning.audio.qwen25_omni",
            "method": "operation_compression",
            "object_family": "module_unit",
            "object_subtype": "transformer_layer",
            "modalities": ["audio"],
            "target_modalities": ["audio"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "video"],
            "coupling_policy": "independent",
            "media_alignment": "none",
            "operation": {
                "type": "prune",
                "target_site": "llm.layers",
                "target_scope": "llm",
                "target_kind": "layer",
                "apply_mode": "structural_rewrite",
                "shape_effect": "remove_module",
            },
            "criterion": {"selection_source": "explicit"},
            "activation_scope": {"layers": [1, 3]},
        }
    )
    model = SimpleNamespace(
        training=False,
        thinker=SimpleNamespace(model=SimpleNamespace(layers=[object()] * 5)),
    )
    runtime = SimpleNamespace(model=model, metadata={})
    config = SimpleNamespace(pipeline=SimpleNamespace(stages=[stage], extensions=[]),
                             backbone=SimpleNamespace(name="qwen25_omni", config={}))
    facade = SimpleNamespace(
        capabilities=lambda: SimpleNamespace(name="qwen25_omni"),
        model_adapter=_OperationModelAdapter(),
        operation_applier=_OperationApplier(),
    )

    session = activate_runtime_compression(
        config=config,
        facade=facade,
        runtime=runtime,
        compression_mode=CompressionMode.COMPRESSED,
        execution_phase=ExecutionPhase.PREPARE,
    )

    assert runtime.metadata["_compression_session"] is session
    assert "_compression_controller" not in runtime.metadata
    assert runtime.metadata["compression"]["execution_phase"] == "prepare"
    operation_metadata = runtime.metadata["compression"]["activations"][0]
    assert operation_metadata["result_num_hidden_layers"] == 3
    assert operation_metadata["irreversible"] is True
    assert _runtime_compression_report(runtime, None) == runtime.metadata["compression"]


class _TimingLLM:
    def register_forward_pre_hook(self, hook: Any, **kwargs: Any) -> _Handle:
        assert kwargs == {"with_kwargs": True}
        self.pre_hook = hook
        return _Handle()

    def register_forward_hook(self, hook: Any, **kwargs: Any) -> _Handle:
        assert kwargs == {"with_kwargs": True}
        self.post_hook = hook
        return _Handle()

    def run(self, shape: tuple[int, ...]) -> None:
        carrier = SimpleNamespace(shape=shape)
        self.pre_hook(self, (carrier,), {})
        self.post_hook(self, (carrier,), {}, None)


def test_generation_timing_records_prefill_and_decode_contract() -> None:
    llm = _TimingLLM()
    model = SimpleNamespace(thinker=SimpleNamespace(model=llm))

    with _capture_thinker_generation_timing(model, device=None) as timing:
        llm.run((1, 7, 16))
        llm.run((1, 1, 16))

    assert timing["llm_forward_calls"] == 2
    assert timing["llm_input_tokens"] == 7
    assert timing["prefill_seconds"] is not None
    assert len(timing["decode_forward_seconds"]) == 1
    assert timing["decode_seconds_per_token"] == timing["decode_forward_seconds"][0]
    assert timing["ttft_seconds"] is not None
    assert timing["cuda_peak_memory_delta_bytes"] is None
