"""Regression checks for the single capability/configuration sources of truth."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mllm_comp.config.loader import load_config
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.config.validators import validate_runnable_config
from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.core.vocabulary import RuntimeMode
from mllm_comp.pipeline.stage import CompressionStageConfig
from mllm_comp.registry.analyses import ACTIVE_ANALYSES, get_analysis
from mllm_comp.registry.backbones import ACTIVE_BACKBONES
from mllm_comp.registry.catalog import (
    BUILTIN_CAPABILITIES,
    CapabilityKind,
    CapabilityStatus,
    capability_names,
    get_capability_spec,
)
from mllm_comp.registry.compressors import ACTIVE_COMPRESSORS
from mllm_comp.registry.integrations import ACTIVE_INTEGRATIONS


def _minimal_config() -> dict[str, object]:
    return {
        "experiment": {"name": "catalog-test"},
        "backbone": {"name": "qwen25_omni", "model_path": "/models/qwen25-omni"},
        "pipeline": {"stages": []},
        "analysis": {},
        "train": {},
        "eval": {},
        "data": {"dataset_id": "fixture"},
        "output": {"run_dir": "runs/test", "manifest_path": "runs/test/manifest.json"},
    }


@pytest.mark.parametrize(
    "path",
    [
        Path("configs/eval/lmms_eval/qwen25_omni_aud_progressive_pruning.yaml"),
        Path(
            "configs/train/qwen2_5_omni/"
            "train_aud_sft_progressive_pruning.yaml"
        ),
    ],
)
def test_qwen_progressive_configs_use_the_validated_three_stage_budget(
    path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    monkeypatch.setenv("MLLM_COMP_TRAIN_DATASET", "/data/train.jsonl")
    monkeypatch.setenv("MLLM_COMP_TRAIN_RUN_DIR", "/runs/train")

    config = load_config(path)

    assert config.pipeline.stages[0].activation_scope == {
        "layers": [6, 12, 18]
    }


def test_fastslm_eval_separates_backbone_and_hta_checkpoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QWEN25_OMNI_MODEL_PATH", "/models/qwen25-omni")
    monkeypatch.setenv("FASTSLM_HTA_CHECKPOINT_PATH", "/models/fastslm-hta.pt")

    config = load_config(
        Path("configs/eval/lmms_eval/qwen25_omni_aud_fastslm_hta.yaml")
    )
    assert config.backbone.model_path == "/models/qwen25-omni"
    assert config.eval.checkpoint_path == "/models/qwen25-omni"
    assert config.pipeline.stages[0].config["checkpoint_path"] == (
        "/models/fastslm-hta.pt"
    )


def test_builtin_capability_identity_is_unique() -> None:
    identities = [(spec.kind, spec.name) for spec in BUILTIN_CAPABILITIES]

    assert len(identities) == len(set(identities))


def test_catalog_separates_active_and_target_compression_methods() -> None:
    backbone = get_capability_spec(CapabilityKind.BACKBONE, "qwen25_omni")
    qwen3 = get_capability_spec(CapabilityKind.BACKBONE, "qwen3_vl")
    qwen35 = get_capability_spec(CapabilityKind.BACKBONE, "qwen3_5")
    llava = get_capability_spec(CapabilityKind.BACKBONE, "llava_onevision")
    segmentwise = get_capability_spec(CapabilityKind.COMPRESSOR, "segmentwise_topk_audio")
    atome = get_capability_spec(
        CapabilityKind.COMPRESSOR,
        "adjacent_token_merging_audio",
    )
    active_extension_names = {
        "local_temporal_bipartite_merging_audio",
        "similarity_attention_merge_prune_audio",
        "encoder_attention_topk_audio",
        "progressive_attention_pruning_audio",
        "audio_pyramid_attention_pruning",
        "structured_llm_layer_pruning",
    }
    target_candidate_names = {
        "fastslm_hta_audio",
        "uniform_average_pooling_audio",
        "uniform_sampling_audio",
        "unsupervised_segmentation_audio",
        "fastadasp_weighted_token_merging",
        "headrouter_dynamic_audio_pruning",
    }
    active_extensions = {
        name: get_capability_spec(CapabilityKind.COMPRESSOR, name)
        for name in active_extension_names
    }
    target_candidates = {
        name: get_capability_spec(CapabilityKind.COMPRESSOR, name)
        for name in target_candidate_names
    }
    qwen3_replacement = get_capability_spec(
        CapabilityKind.COMPRESSOR,
        "qwen3_vl_attention_replacement",
    )
    qwen3_ffn_replacement = get_capability_spec(
        CapabilityKind.COMPRESSOR,
        "qwen3_vl_ffn_replacement",
    )
    analyses = {
        name: get_capability_spec(CapabilityKind.ANALYSIS, name)
        for name in {
            "representation_dynamics",
            "layer_importance",
            "late_injection_early_exit",
            "residual_stream_decomposition",
            "prediction_dynamics",
            "attention_flow",
            "hierarchical_taylor",
            "visual_merge_attribution",
            "visual_token_phenomena",
            "visual_evidence_interaction",
            "visual_reconstruction_utility",
            "learned_hybrid_evaluation",
            "anchor_training_evaluation",
            "anchor7b_evaluation",
        }
    }

    assert backbone.status is CapabilityStatus.ACTIVE
    assert backbone.profile.name == "qwen25_omni"
    assert qwen3.status is CapabilityStatus.ACTIVE
    assert qwen3.profile.name == "qwen3_vl"
    assert qwen35.status is CapabilityStatus.ACTIVE
    assert qwen35.profile.name == "qwen3_5"
    assert llava.status is CapabilityStatus.ACTIVE
    assert llava.profile.name == "llava_onevision"
    assert segmentwise.status is CapabilityStatus.ACTIVE
    assert atome.status is CapabilityStatus.ACTIVE
    assert all(
        spec.status is CapabilityStatus.ACTIVE
        for spec in active_extensions.values()
    )
    assert all(
        spec.status is CapabilityStatus.TARGET
        for spec in target_candidates.values()
    )
    qwen25_vl = get_capability_spec(CapabilityKind.BACKBONE, "qwen25_vl")
    assert qwen25_vl.status is CapabilityStatus.ACTIVE
    assert ACTIVE_BACKBONES == {
        "qwen25_omni": backbone.provider,
        "qwen3_vl": qwen3.provider,
        "qwen3_5": qwen35.provider,
        "qwen25_vl": qwen25_vl.provider,
        "llava_onevision": llava.provider,
    }
    assert ACTIVE_COMPRESSORS == {
        "segmentwise_topk_audio": segmentwise.provider,
        "adjacent_token_merging_audio": atome.provider,
        **{name: spec.provider for name, spec in active_extensions.items()},
        "qwen3_vl_attention_replacement": qwen3_replacement.provider,
        "qwen3_vl_ffn_replacement": qwen3_ffn_replacement.provider,
    }
    assert target_candidate_names.isdisjoint(ACTIVE_COMPRESSORS)
    assert qwen3_replacement.status is CapabilityStatus.ACTIVE
    assert qwen3_ffn_replacement.status is CapabilityStatus.ACTIVE
    assert ACTIVE_ANALYSES == {
        name: spec.provider for name, spec in analyses.items()
    }
    assert set(ACTIVE_INTEGRATIONS) == {"lmms_eval", "ms_swift"}


def test_active_analyses_include_omni_and_exclude_gate_placeholders() -> None:
    analyses = [
        spec for spec in BUILTIN_CAPABILITIES if spec.kind is CapabilityKind.ANALYSIS
    ]

    assert {spec.name for spec in analyses} == {
        "representation_dynamics",
        "layer_importance",
        "late_injection_early_exit",
        "residual_stream_decomposition",
        "prediction_dynamics",
        "attention_flow",
        "hierarchical_taylor",
        "visual_merge_attribution",
        "visual_token_phenomena",
        "visual_evidence_interaction",
        "visual_reconstruction_utility",
        "learned_hybrid_evaluation",
        "anchor_training_evaluation",
        "anchor7b_evaluation",
    }
    assert set(ACTIVE_ANALYSES) == {spec.name for spec in analyses}
    for spec in analyses:
        assert spec.status is CapabilityStatus.ACTIVE
        if spec.name in {
            "visual_merge_attribution", "visual_token_phenomena",
            "visual_evidence_interaction", "visual_reconstruction_utility",
            "learned_hybrid_evaluation", "anchor_training_evaluation", "anchor7b_evaluation",
        }:
            assert set(spec.metadata["backbones"]) == {"qwen25_vl"}
        else:
            assert set(spec.metadata["backbones"]) == {
                "llava_onevision",
                "qwen25_omni",
                "qwen25_vl",
                "qwen3_vl",
                "qwen3_5",
            }
        assert "gate" not in spec.name.lower()
        assert "hardconcrete" not in spec.name.lower()


def test_integration_cli_names_and_roles_come_from_catalog() -> None:
    assert capability_names(CapabilityKind.INTEGRATION, interface="analysis") == ("lmms_eval",)
    assert capability_names(CapabilityKind.INTEGRATION, interface="eval") == ("lmms_eval",)
    assert capability_names(CapabilityKind.INTEGRATION, interface="train") == (
        "native_source",
        "ms_swift",
    )


def test_ms_swift_catalog_declares_compressed_training_scope() -> None:
    from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter

    spec = get_capability_spec(CapabilityKind.INTEGRATION, "ms_swift")
    assert set(spec.metadata["backbones"]) == {
        "qwen25_omni", "qwen25_vl", "qwen3_vl", "llava_onevision",
    }
    for backbone in spec.metadata["backbones"]:
        config = experiment_config_from_dict({
            "backbone": {"name": backbone, "model_id": "fixture"},
            "data": {"dataset_id": "fixture"}, "output": {"run_dir": "runs/fixture"},
        })
        plan = MSSwiftTrainAdapter().build_launch_plan(config)
        assert plan.backbone_name == backbone
        assert plan.trainable_branch in spec.metadata["trainable_branches"][backbone]
    scope = spec.metadata["compressed_training"]
    assert scope["single_stage_only"] is False
    assert scope["stage_limit_by_backbone"]["qwen3_vl"] == 1


def test_catalog_runs_active_compressor_schema_validator() -> None:
    spec = get_capability_spec(CapabilityKind.COMPRESSOR, "segmentwise_topk_audio")

    spec.validate_config(
        {
            "operation": {
                "type": "keep",
                "target_site": "audio_encoder_output_pre_projector",
            },
            "criterion": {"score_source": "uniform", "num_segments": 1},
            "schedule": {
                "granularity": "per_audio",
                "keep_ratio": 0.5,
            },
        }
    )
    with pytest.raises(ConfigError, match="keep_ratio"):
        spec.validate_config({"schedule": {"keep_ratio": 0}})


def test_unknown_capability_has_one_consistent_error_boundary() -> None:
    with pytest.raises(RegistryError, match="missing"):
        get_capability_spec(CapabilityKind.ANALYSIS, "missing")


def test_active_analysis_plugins_resolve_from_active_registry() -> None:
    names = {
        "representation_dynamics",
        "layer_importance",
        "late_injection_early_exit",
        "residual_stream_decomposition",
        "prediction_dynamics",
        "attention_flow",
        "hierarchical_taylor",
    }

    for name in names:
        spec = get_capability_spec(CapabilityKind.ANALYSIS, name)
        assert get_analysis(name) is spec.provider


def test_loader_normalizes_pipeline_stage_once() -> None:
    raw = _minimal_config()
    raw["pipeline"] = {
        "stages": [
            {
                "name": "drop",
                "placement": "projector_level",
                "execution_phase": "forward",
                "plugin": "segmentwise_topk_audio",
                "modalities": ["audio"],
            }
        ]
    }

    config = experiment_config_from_dict(raw)

    assert isinstance(config.pipeline.stages[0], CompressionStageConfig)
    assert config.pipeline.stages[0].placement.value == "projector_level"


def test_eval_runtime_mode_defaults_and_validates() -> None:
    default_config = experiment_config_from_dict(_minimal_config())
    assert default_config.eval.runtime_mode is RuntimeMode.DEFAULT

    profile = _minimal_config()
    profile["eval"] = {"runtime_mode": "profile"}
    assert experiment_config_from_dict(profile).eval.runtime_mode is RuntimeMode.PROFILE

    invalid = _minimal_config()
    invalid["eval"] = {"runtime_mode": "normal"}
    with pytest.raises(ConfigError, match="eval.runtime_mode"):
        experiment_config_from_dict(invalid)


def test_train_runtime_mode_defaults_and_validates() -> None:
    default_config = experiment_config_from_dict(_minimal_config())
    assert default_config.train.runtime_mode is RuntimeMode.DEFAULT

    debug = _minimal_config()
    debug["train"] = {"runtime_mode": "debug"}
    assert experiment_config_from_dict(debug).train.runtime_mode is RuntimeMode.DEBUG

    invalid = _minimal_config()
    invalid["train"] = {"runtime_mode": "normal"}
    with pytest.raises(ConfigError, match="train.runtime_mode"):
        experiment_config_from_dict(invalid)


def test_config_rejects_section_and_stage_typos() -> None:
    bad_section = _minimal_config()
    bad_section["backbone"] = {"name": "qwen25_omni", "adaptor": "typo"}
    with pytest.raises(ConfigError, match="adaptor"):
        experiment_config_from_dict(bad_section)

    bad_stage = _minimal_config()
    bad_stage["pipeline"] = {
        "stages": [
            {
                "name": "drop",
                "placement": "projector_level",
                "execution_phase": "forward",
                "plugin": "segmentwise_topk_audio",
                "modalties": ["image"],
            }
        ]
    }
    with pytest.raises(ConfigError, match="modalties"):
        experiment_config_from_dict(bad_stage)

    bad_tasks = _minimal_config()
    bad_tasks["analysis"] = {"tasks": "not-a-list"}
    with pytest.raises(ConfigError, match="analysis.tasks"):
        experiment_config_from_dict(bad_tasks)

    legacy_template_flag = _minimal_config()
    legacy_template_flag["template_only"] = False
    with pytest.raises(ConfigError, match="template_only"):
        experiment_config_from_dict(legacy_template_flag)


def test_analysis_runtime_batch_size_rejects_boolean_before_normalization() -> None:
    invalid = _minimal_config()
    invalid["analysis"] = {"runtime_batch_size": True}

    with pytest.raises(ConfigError, match="analysis.runtime_batch_size"):
        experiment_config_from_dict(invalid)


def test_runnable_config_requires_materialized_runtime_values() -> None:
    runnable = experiment_config_from_dict(_minimal_config())
    validate_runnable_config(runnable)

    missing_model = _minimal_config()
    missing_model["backbone"] = {"name": "qwen25_omni"}
    with pytest.raises(ConfigError, match="backbone.model_path"):
        validate_runnable_config(experiment_config_from_dict(missing_model))

    missing_data = _minimal_config()
    missing_data["data"] = {}
    with pytest.raises(ConfigError, match="data source"):
        validate_runnable_config(experiment_config_from_dict(missing_data))

    unresolved_model = _minimal_config()
    unresolved_model["backbone"] = {
        "name": "qwen25_omni",
        "model_path": "${MISSING_MODEL_PATH}",
    }
    with pytest.raises(ConfigError, match="MISSING_MODEL_PATH"):
        validate_runnable_config(experiment_config_from_dict(unresolved_model))


def test_runnable_config_checks_current_nested_values_instead_of_original_raw() -> None:
    raw = _minimal_config()
    raw["backbone"]["model_path"] = "${OLD_MODEL_PATH}"
    config = experiment_config_from_dict(raw)
    config.backbone.model_path = "/models/override"

    validate_runnable_config(config)

    config.backbone.config["nested"] = [{"checkpoint": "${MISSING_RULE_CHECKPOINT}"}]
    with pytest.raises(ConfigError, match="MISSING_RULE_CHECKPOINT"):
        validate_runnable_config(config)


def test_config_loader_applies_field_overrides_before_type_parsing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("WORKERS", "MODE", "TASKS"):
        monkeypatch.delenv(name, raising=False)
    raw = _minimal_config()
    raw["eval"] = {
        "num_processes": "${WORKERS}",
        "runtime_mode": "${MODE}",
        "tasks": "${TASKS}",
        "generation": {"temperature": 0.5, "use_cache": True},
    }
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")
    overrides = {
        "eval": {
            "num_processes": 1,
            "runtime_mode": "default",
            "tasks": ["mme"],
            "generation": {"use_cache": False},
        }
    }

    config = load_config(path, overrides=overrides)

    assert config.eval.num_processes == 1
    assert config.eval.runtime_mode is RuntimeMode.DEFAULT
    assert config.eval.tasks == ["mme"]
    assert config.eval.generation == {"temperature": 0.5, "use_cache": False}
    assert config.raw["eval"]["num_processes"] == "${WORKERS}"
    validate_runnable_config(config)

    raw["eval"]["num_proceses"] = 2
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ConfigError, match="num_proceses"):
        load_config(path, overrides=overrides)


@pytest.mark.parametrize("uncovered_field", ["num_processes", "runtime_mode"])
def test_config_loader_still_rejects_invalid_values_without_an_override(
    tmp_path: Path,
    uncovered_field: str,
) -> None:
    raw = _minimal_config()
    raw["eval"] = {uncovered_field: "not-a-valid-value"}
    path = tmp_path / "config.yaml"
    path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises((ConfigError, ValueError), match="not-a-valid-value"):
        load_config(path, overrides={"eval": {"tasks": ["mme"]}})


def test_pipeline_extensions_requires_explicit_function_names() -> None:
    raw = _minimal_config()
    raw["pipeline"] = {"extensions": ["research.audio:register"]}
    config = experiment_config_from_dict(raw)
    assert config.pipeline.extensions == ["research.audio:register"]
    raw["pipeline"]["extensions"] = "research.audio:register"
    with pytest.raises(ConfigError, match="pipeline.extensions"):
        experiment_config_from_dict(raw)
