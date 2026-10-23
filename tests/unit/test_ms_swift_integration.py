"""Contract tests for the ms-swift training adapter."""

from __future__ import annotations

import base64
import json
from pathlib import Path
import sys
from typing import Any

import pytest

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter
from mllm_comp.integrations.ms_swift.adapter import (
    _collect_training_compression_evidence,
)
from mllm_comp.integrations.native_cli import NativeCLIResult
from mllm_comp.run_manifest import build_run_manifest


def _config(
    *,
    backbone_name: str = "qwen25_omni",
    backbone_config: dict[str, Any] | None = None,
    compression_mode: str = "none",
    checkpoint_in: str | None = None,
    train_args: dict[str, Any] | None = None,
    output_dir: str = "tests/fixtures/ms_swift_run",
    compression_plugin: str = "segmentwise_topk_audio",
    runtime_mode: str = "default",
) -> Any:
    pipeline = {"stages": []}
    if compression_mode == "compressed":
        projector_stages = {
            "segmentwise_topk_audio": {
                "name": "audio_segmentwise_topk",
                "operation": {
                    "type": "keep",
                    "target_site": "audio_encoder_output_pre_projector",
                },
                "criterion": {
                    "score_source": "encoder_last_layer_attention",
                },
                "schedule": {"keep_ratio": 0.5},
            },
            "local_temporal_bipartite_merging_audio": {
                "name": "audio_ltbm",
                "operation": {
                    "type": "merge",
                    "target_site": "audio_encoder_output_pre_projector",
                    "merge_value": "mean",
                },
                "criterion": {
                    "score_source": "projector_token_cosine",
                    "pairing": "alternating_bipartite_local",
                },
                "schedule": {"keep_ratio": 0.5},
            },
            "similarity_attention_merge_prune_audio": {
                "name": "audio_sa_map",
                "operation": {
                    "type": "merge_prune",
                    "target_site": "audio_projector_output_post_projector",
                    "merge_value": "attention_weighted_mean",
                },
                "criterion": {
                    "score_source": "encoder_penultimate_layer_attention",
                    "pruning": "greedy_attention_dpp",
                },
                "schedule": {"keep_ratio": 0.5},
            },
        }
        encoder_stages = {
            "adjacent_token_merging_audio": {
                "name": "audio_atome",
                "operation": {
                    "type": "merge",
                    "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
                    "merge_value": "mean",
                },
                "criterion": {
                    "policy": "fixed_ratio",
                    "score_source": "encoder_key_cosine",
                    "pairing": "disjoint_adjacent",
                },
                "schedule": {"merge_ratio": 0.1},
                "activation_scope": {"first_layer": 2, "layer_interval": 3},
            },
            "encoder_attention_topk_audio": {
                "name": "audio_encoder_topk",
                "operation": {
                    "type": "keep",
                    "target_site": "audio_encoder_layer_post_mhsa_pre_ffn",
                },
                "criterion": {"score_source": "mean_received_self_attention"},
                "schedule": {"keep_ratio": 0.9},
                "activation_scope": {"layers": [3, 6, 9]},
            },
            "progressive_attention_pruning_audio": {
                "name": "audio_progressive_pruning",
                "operation": {
                    "type": "keep",
                    "target_site": "audio_encoder_layer_output_post_block",
                },
                "criterion": {"score_source": "mean_received_self_attention"},
                "schedule": {"policy": "fixed", "pruning_rate": 0.1},
                "activation_scope": {"first_layer": 6, "layer_interval": 1},
            },
        }
        llm_token_stages = {
            "audio_pyramid_attention_pruning": {
                "name": "audio_pyramid",
                "operation": {
                    "type": "keep",
                    "target_site": "llm_decoder_layer_output_audio_tokens",
                },
                "criterion": {
                    "score_source": "instruction_to_audio_received_attention"
                },
                "schedule": {"keep_ratio": 0.5, "minimum_keep_tokens": 2},
                "activation_scope": {
                    "layers": [1, 3, 5],
                    "lifecycle": "prefill_only",
                },
            }
        }
        operation_stages = {
            "structured_llm_layer_pruning": {
                "name": "spade_operation_only",
                "operation": {
                    "type": "prune",
                    "target_site": "llm.layers",
                    "target_scope": "llm",
                    "target_kind": "layer",
                    "apply_mode": "structural_rewrite",
                    "shape_effect": "remove_module",
                },
                "criterion": {"selection_source": "explicit"},
                "activation_scope": {"layers": [1, 3, 5]},
            }
        }
        if compression_plugin in projector_stages:
            method = projector_stages[compression_plugin]
            stage = {
                **method,
                "placement": "projector_level",
                "execution_phase": "forward",
                "plugin": compression_plugin,
                "method": "token_compression",
                "object_family": "tensor_slot",
                "modalities": ["audio"],
                "token_roles": ["observation"],
            }
        elif compression_plugin in encoder_stages:
            method = encoder_stages[compression_plugin]
            stage = {
                **method,
                "placement": "encoder_level",
                "execution_phase": "forward",
                "plugin": compression_plugin,
                "method": "token_compression",
                "object_family": "tensor_slot",
                "modalities": ["audio"],
                "token_roles": ["observation"],
            }
        elif compression_plugin in llm_token_stages:
            method = llm_token_stages[compression_plugin]
            stage = {
                **method,
                "placement": "llm_level",
                "execution_phase": "forward",
                "plugin": compression_plugin,
                "method": "token_compression",
                "object_family": "tensor_slot",
                "object_subtype": "llm_decoder_layer_output_audio_tokens",
                "modalities": ["audio"],
                "token_roles": ["observation"],
            }
        else:
            method = operation_stages[compression_plugin]
            stage = {
                **method,
                "placement": "llm_level",
                "execution_phase": "prepare",
                "plugin": compression_plugin,
                "method": "operation_compression",
                "object_family": "module_unit",
                "object_subtype": "transformer_layer",
            }
        stage.update(
            {
                "algorithm": compression_plugin,
                "variant": f"{compression_plugin}.audio",
                "binding": f"{compression_plugin}.audio.qwen25_omni",
                "modalities": ["audio"],
                "target_modalities": ["audio"],
                "conditioning_modalities": (
                    ["text"] if compression_plugin in llm_token_stages else []
                ),
                "preserved_modalities": ["text", "image", "video"],
                "coupling_policy": "independent",
                "media_alignment": (
                    "token"
                    if compression_plugin in llm_token_stages
                    else (
                        "none"
                        if compression_plugin in operation_stages
                        else "temporal"
                    )
                ),
            }
        )
        pipeline = {"stages": [stage]}
    return experiment_config_from_dict(
        {
            "experiment": {"name": "train", "seed": 11},
            "backbone": {
                "name": backbone_name,
                "model_id": "Qwen/fixture",
                "config": backbone_config or {},
            },
            "pipeline": pipeline,
            "analysis": {},
            "train": {
                "backend": "ms_swift",
                "compression_mode": compression_mode,
                "runtime_mode": runtime_mode,
                "checkpoint_in": checkpoint_in,
                "max_steps": 3,
                "args": train_args or {
                    "torch_dtype": "bfloat16",
                    "tuner_type": "lora",
                    "gradient_accumulation_steps": 2,
                    "target_modules": ["all-linear"],
                },
            },
            "eval": {},
            "data": {"dataset_id": "org/dataset#8"},
            "output": {
                "run_dir": output_dir,
                "manifest_path": "tests/fixtures/ms_swift_manifest.test.json",
            },
        },
        path="train.yaml",
    )


def _value_after(argv: list[str], name: str) -> str:
    return argv[argv.index(f"--{name}") + 1]


def test_launch_plan_maps_platform_config_without_owning_trainer() -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(_config())

    assert plan.argv[:2] == ["swift", "sft"]
    assert _value_after(plan.argv, "model") == "Qwen/fixture"
    assert _value_after(plan.argv, "model_type") == "qwen2_5_omni"
    assert _value_after(plan.argv, "template") == "qwen2_5_omni"
    assert _value_after(plan.argv, "dataset") == "org/dataset#8"
    assert _value_after(plan.argv, "max_steps") == "3"
    assert _value_after(plan.argv, "seed") == "11"
    assert plan.environment == {"ENABLE_AUDIO_OUTPUT": "0"}
    assert plan.unset_environment == [
        "MAX_PIXELS",
        "USE_AUDIO_IN_VIDEO",
        "SAMPLING_RATE",
        "VIDEO_MIN_PIXELS",
        "VIDEO_MAX_PIXELS",
        "FPS",
        "FPS_MIN_FRAMES",
        "FPS_MAX_FRAMES",
        "FRAME_FACTOR",
    ]
    assert plan.trainable_branch == "thinker"
    assert plan.talker_training is False
    assert plan.argv.index("--gradient_accumulation_steps") < plan.argv.index("--target_modules")


@pytest.mark.parametrize("train_args,expected", [
    ({}, ["--seed", "93"]),
    ({"seed": 17}, ["--seed", "17"]),
    ({"native_argv": ["--seed", "31"]}, ["--seed", "31"]),
    ({"seed": 17, "native_argv": ["--seed=31"]}, ["--seed=31"]),
])
def test_training_seed_uses_experiment_default_and_native_overrides(train_args, expected):
    config = _config(train_args=train_args)
    config.experiment.seed = 93
    plan = MSSwiftTrainAdapter().build_launch_plan(config)
    seed_args = []
    for index, arg in enumerate(plan.argv):
        if arg == "--seed":
            seed_args.extend(plan.argv[index:index + 2])
        elif arg.startswith("--seed="):
            seed_args.append(arg)
    assert seed_args == expected


def test_vscode_debug_launch_uses_python_module_for_subprocess_attach(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MLLM_COMP_VSCODE_DEBUG_SUBPROCESSES", "1")

    plan = MSSwiftTrainAdapter().build_launch_plan(_config())

    assert plan.argv[:4] == [sys.executable, "-m", "swift.cli.main", "sft"]
    assert _value_after(plan.argv, "model") == "Qwen/fixture"


def test_train_invokes_native_cli_and_collects_checkpoint_and_logs(
    tmp_path: Path,
) -> None:
    invocations: list[Any] = []

    def command_runner(plan: Any) -> NativeCLIResult:
        invocations.append(plan)
        checkpoint = Path(plan.output_dir) / "v0" / "checkpoint-3"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"best_metric": 0.25, "global_step": 3}),
            encoding="utf-8",
        )
        (checkpoint.parent / "logging.jsonl").write_text(
            json.dumps({"loss": 1.5, "step": 3}) + "\n",
            encoding="utf-8",
        )
        return NativeCLIResult(returncode=0)

    config = _config(output_dir=str(tmp_path / "ms_swift_run"))
    result = MSSwiftTrainAdapter(command_runner=command_runner).train(config=config)

    assert invocations and _value_after(invocations[0].argv, "model") == "Qwen/fixture"
    assert invocations[0].argv[:2] == ["swift", "sft"]
    assert invocations[0].environment == {"ENABLE_AUDIO_OUTPUT": "0"}
    assert result.checkpoint_path.endswith("checkpoint-3")
    assert result.train_log_path is not None
    assert Path(result.train_log_path).name == "logging.jsonl"
    assert result.metrics == {
        "best_metric": 0.25,
        "global_step": 3.0,
        "last/loss": 1.5,
        "last/step": 3.0,
    }
    assert result.backend is not None and result.backend.name == "ms_swift"
    assert Path(result.backend_args["checkpoint_path"]).parent.as_posix().endswith(
        "ms_swift_run/v0"
    )
    assert result.backend_args["argv"] == invocations[0].argv
    assert result.backend_args["unset_environment"] == invocations[0].unset_environment
    manifest = build_run_manifest(
        config,
        train_backend=result.backend,
        checkpoint_path=result.checkpoint_path,
    )
    assert manifest.model_path_or_id == "Qwen/fixture"
    assert manifest.checkpoint_path == result.checkpoint_path


def test_qwen3_vl_launch_plan_uses_native_swift_contract_without_omni_controls() -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(
        _config(
            backbone_name="qwen3_vl",
            train_args={
                "torch_dtype": "bfloat16",
                "tuner_type": "lora",
                "target_modules": ["all-linear"],
            },
        )
    )

    assert _value_after(plan.argv, "model_type") == "qwen3_vl"
    assert _value_after(plan.argv, "template") == "qwen3_vl"
    assert plan.environment == {}
    assert plan.unset_environment == []
    assert plan.backbone_name == "qwen3_vl"
    assert plan.trainable_branch == "language_model"
    assert plan.talker_training is False


@pytest.mark.parametrize(
    ("backbone_name", "model_type", "template"),
    [
        ("qwen25_vl", "qwen2_5_vl", "qwen2_5_vl"),
        ("llava_onevision", "llava_onevision_qwen", "llava_onevision_qwen"),
    ],
)
def test_image_backbone_launch_plan_uses_native_swift_contract(
    backbone_name: str,
    model_type: str,
    template: str,
) -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(
        _config(
            backbone_name=backbone_name,
            train_args={"torch_dtype": "bfloat16", "tuner_type": "full"},
        )
    )

    assert _value_after(plan.argv, "model_type") == model_type
    assert _value_after(plan.argv, "template") == template
    assert plan.backbone_name == backbone_name
    assert plan.trainable_branch == "language_model"
    assert plan.environment == {}


def test_qwen3_vl_policy_rejects_omni_only_controls_and_wrong_branch() -> None:
    adapter = MSSwiftTrainAdapter()

    with pytest.raises(BackendIntegrationError, match="language_model"):
        adapter.build_launch_plan(
            _config(
                backbone_name="qwen3_vl",
                train_args={"trainable_branch": "thinker"},
            )
        )
    with pytest.raises(BackendIntegrationError, match="Qwen2.5-Omni-only"):
        adapter.build_launch_plan(
            _config(
                backbone_name="qwen3_vl",
                train_args={"enable_audio_output": False},
            )
        )
    with pytest.raises(BackendIntegrationError, match="media controls"):
        adapter.build_launch_plan(
            _config(
                backbone_name="qwen3_vl",
                backbone_config={"max_pixels": 1003520},
            )
        )


def test_qwen25_omni_maps_explicit_backbone_media_config_only() -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(
        _config(
            backbone_config={
                "max_pixels": 1003520,
                "use_audio_in_video": True,
                "sampling_rate": 16000,
                "video_min_pixels": 50176,
                "video_max_pixels": 100352,
                "fps": 2.0,
                "fps_min_frames": 4,
                "fps_max_frames": 12,
                "frame_factor": 2,
            }
        )
    )

    assert "--max_pixels" not in plan.argv
    assert plan.environment == {
        "ENABLE_AUDIO_OUTPUT": "0",
        "MAX_PIXELS": "1003520",
        "USE_AUDIO_IN_VIDEO": "1",
        "SAMPLING_RATE": "16000",
        "VIDEO_MIN_PIXELS": "50176",
        "VIDEO_MAX_PIXELS": "100352",
        "FPS": "2.0",
        "FPS_MIN_FRAMES": "4",
        "FPS_MAX_FRAMES": "12",
        "FRAME_FACTOR": "2",
    }


@pytest.mark.parametrize(
    ("backbone_config", "message"),
    [
        ({"use_audio_in_video": 1}, "use_audio_in_video"),
        ({"use_audio_in_video": None}, "use_audio_in_video"),
        ({"max_pixels": 0}, "max_pixels"),
        ({"sampling_rate": 16000.0}, "sampling_rate"),
        ({"fps": False}, "fps"),
        (
            {"video_min_pixels": 100352, "video_max_pixels": 50176},
            "video_min_pixels",
        ),
        ({"fps_min_frames": 12, "fps_max_frames": 4}, "fps_min_frames"),
    ],
)
def test_qwen25_omni_rejects_invalid_media_config(
    backbone_config: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(BackendIntegrationError, match=message):
        MSSwiftTrainAdapter().build_launch_plan(
            _config(backbone_config=backbone_config)
        )


@pytest.mark.parametrize(
    "train_args",
    [
        {"max_pixels": 1003520},
        {"use_audio_in_video": False},
        {"fps": 2.0},
    ],
)
def test_qwen25_omni_rejects_media_controls_in_train_args(
    train_args: dict[str, Any],
) -> None:
    with pytest.raises(BackendIntegrationError, match="backbone.config"):
        MSSwiftTrainAdapter().build_launch_plan(
            _config(
                train_args=train_args,
            )
        )


@pytest.mark.parametrize(
    "compression_plugin",
    [
        "segmentwise_topk_audio",
        "adjacent_token_merging_audio",
        "local_temporal_bipartite_merging_audio",
        "similarity_attention_merge_prune_audio",
        "encoder_attention_topk_audio",
        "progressive_attention_pruning_audio",
        "audio_pyramid_attention_pruning",
    ],
)
def test_compressed_training_injects_platform_plugin_into_real_forward(
    compression_plugin: str,
) -> None:
    adapter = MSSwiftTrainAdapter()
    plan = adapter.build_launch_plan(
        _config(
            compression_mode="compressed",
            compression_plugin=compression_plugin,
            train_args={"tuner_type": "full"},
        )
    )
    plugin_path = _value_after(plan.argv, "external_plugins")
    assert plugin_path.replace("\\", "/").endswith(
        "mllm_comp/integrations/ms_swift/training_plugin.py"
    )
    assert plan.compression_training["enabled"] is True
    assert plan.compression_training["plugin"] == compression_plugin
    assert plan.compression_training["execution_phase"] == "forward"
    assert plan.environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"]
    assert (
        plan.environment["MLLM_COMP_TRAIN_COMPRESSION_EVIDENCE_DIR"]
        == plan.output_dir
    )
    assert plan.compression_training["evidence_dir"] == plan.output_dir
    assert "launch_id" not in plan.compression_training


def test_operation_compressed_training_installs_before_optimizer() -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(
        _config(
            compression_mode="compressed",
            compression_plugin="structured_llm_layer_pruning",
            train_args={
                "tuner_type": "full",
                "artifact_format": "transformers",
                "packing": True,
            },
        )
    )

    assert plan.compression_training["executor_family"] == "llm_operation"
    assert plan.compression_training["execution_phase"] == "prepare"
    assert plan.artifact_format == "transformers"
    assert plan.environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"]
    assert _value_after(plan.argv, "packing") == "true"


def _combined_config(**kwargs: Any) -> Any:
    config = _config(compression_mode="compressed", **kwargs)
    operation = _config(
        compression_mode="compressed", compression_plugin="structured_llm_layer_pruning",
    ).pipeline.stages[0]
    config.pipeline.stages.append(operation)
    return config


def test_omni_training_transports_all_stages_and_native_resume_source() -> None:
    from mllm_comp.registry.catalog import CapabilityKind, get_capability_spec

    capabilities = get_capability_spec(CapabilityKind.INTEGRATION, "ms_swift").metadata
    assert capabilities["compressed_training"]["stage_limit_by_backbone"]["qwen25_omni"] is None
    config = _combined_config()
    config.train.resume = "/native/checkpoint-12"
    plan = MSSwiftTrainAdapter().build_launch_plan(config)
    payload = json.loads(base64.urlsafe_b64decode(plan.environment["MLLM_COMP_TRAIN_COMPRESSION_SPEC"]))
    assert payload["schema_version"] == 3
    assert [stage["name"] for stage in payload["stages"]] == [stage.name for stage in config.pipeline.stages]
    assert [stage["executor_family"] for stage in plan.compression_training["stages"]] == [
        "projector_token", "llm_operation",
    ]
    assert _value_after(plan.argv, "resume_from_checkpoint") == config.train.resume
    assert _value_after(plan.argv, "model") == config.backbone.model_id


@pytest.mark.parametrize("name", ["packing", "padding_free"])
def test_prepare_stage_does_not_relax_dynamic_training_layout_contract(name: str) -> None:
    with pytest.raises(BackendIntegrationError, match=name):
        MSSwiftTrainAdapter().build_launch_plan(_combined_config(train_args={name: True}))


def test_compressed_resume_rejects_early_materialized_model_loading() -> None:
    config = _combined_config(train_args={"resume_only_model": True})
    config.train.resume = "/native/checkpoint-12"
    with pytest.raises(BackendIntegrationError, match="resume_only_model"):
        MSSwiftTrainAdapter().build_launch_plan(config)


@pytest.mark.parametrize("missing_stage", [None, "audio_segmentwise_topk", "spade_operation_only"])
def test_combined_training_requires_each_stage_forward_and_backward(tmp_path: Path, missing_stage: str | None) -> None:
    plan = MSSwiftTrainAdapter().build_launch_plan(_combined_config())
    records = []
    for stage in plan.compression_training["stages"]:
        family = stage["executor_family"]
        common = {**stage, "spec_hash": plan.compression_training["spec_hash"]}
        records.append({**common, "event": "install", "operation": {"result_num_hidden_layers": 2}})
        records.append({
            **common, "event": "forward", "forward_index": 0, "runtime_mode": "default",
            "loss_requires_grad": True, "timing": None, "token_debug": None,
            "bridge": ("swift_loader_pre_optimizer_forward" if family == "llm_operation"
                       else "swift_template_pre_forward_hook"),
            "compression": {"original_audio_tokens": 4, "kept_audio_tokens": 2},
        })
        if stage["stage_name"] != missing_stage:
            records.append({**common, "event": "backward", "forward_index": 0, "gradient_present": True})
    (tmp_path / "compression_training_rank_0.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records), encoding="utf-8",
    )
    kwargs = dict(
        expected_spec_hash=plan.compression_training["spec_hash"], expected_runtime_mode="default",
        expected_executor_family="omni_multistage", expected_stages=plan.compression_training["stages"],
    )
    if missing_stage is not None:
        with pytest.raises(BackendIntegrationError, match="backward"):
            _collect_training_compression_evidence(tmp_path, **kwargs)
    else:
        evidence = _collect_training_compression_evidence(tmp_path, **kwargs)
        assert evidence["verified"]
        assert all(stage["verified"] for stage in evidence["stages"])


def test_audio_research_training_does_not_require_historical_support(monkeypatch) -> None:
    from mllm_comp.compressors.multimodal_registry import builtin_multimodal_compression_contracts
    from mllm_comp.core.support import SupportRegistry
    from mllm_comp.pipeline.definition import resolve_compression_definition

    registry = builtin_multimodal_compression_contracts()
    monkeypatch.setattr(registry, "support", SupportRegistry())
    config = _config(compression_mode="compressed")
    definition = resolve_compression_definition(
        config, compression_mode="compressed", interface="train"
    )
    assert definition.stages[0].plugin == "segmentwise_topk_audio"
    plan = MSSwiftTrainAdapter().build_launch_plan(config)
    assert plan.compression_training["enabled"] is True
    record = build_run_manifest(config).compression_stages[0]
    assert record.support_status == "undeclared"
    assert record.evidence == []


def test_training_rejects_an_unknown_artifact_format() -> None:
    config = _config(train_args={"artifact_format": "custom_checkpoint"})

    with pytest.raises(BackendIntegrationError, match="artifact_format"):
        MSSwiftTrainAdapter().build_launch_plan(config)


def test_compressed_training_requires_forward_stage() -> None:
    config = _config(compression_mode="compressed")
    config.pipeline.stages[0].execution_phase = "generate"

    with pytest.raises(BackendIntegrationError, match="execution_phase=forward"):
        MSSwiftTrainAdapter().build_launch_plan(config)


@pytest.mark.parametrize("name", ["packing", "padding_free"])
def test_compressed_training_rejects_unverified_sequence_layouts(name: str) -> None:
    config = _config(
        compression_mode="compressed",
        train_args={"tuner_type": "lora", name: True},
    )

    with pytest.raises(BackendIntegrationError, match=name):
        MSSwiftTrainAdapter().build_launch_plan(config)


def test_compressed_training_collects_forward_and_backward_evidence(
    tmp_path: Path,
) -> None:

    def command_runner(plan: Any) -> NativeCLIResult:
        output = Path(plan.output_dir)
        native_output = output / "v0"
        checkpoint = native_output / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        (checkpoint / "trainer_state.json").write_text(
            json.dumps({"global_step": 1}),
            encoding="utf-8",
        )
        (native_output / "args.json").write_text("{}\n", encoding="utf-8")
        (native_output / "logging.jsonl").write_text(
            json.dumps({"loss": 1.0}) + "\n",
            encoding="utf-8",
        )
        common = {
            "schema_version": 1,
            "spec_hash": plan.compression_training["spec_hash"],
            "stage_name": plan.compression_training["stage_name"],
            "plugin": plan.compression_training["plugin"],
            "execution_phase": "forward",
        }
        records = [
            {**common, "event": "install"},
            {
                **common,
                "event": "forward",
                "forward_index": 0,
                "runtime_mode": plan.compression_training["runtime_mode"],
                "bridge": "swift_template_pre_forward_hook",
                "loss_requires_grad": True,
                "timing": None,
                "token_debug": {"mode": "debug", "layers": [], "modules": []},
                "compression": {
                    "original_audio_tokens": 20,
                    "kept_audio_tokens": 10,
                    "layers": [
                        {
                            "layer_index": 2,
                            "input_tokens": 20,
                            "output_tokens": 10,
                            "merged_tokens": 10,
                            "input_chunk_lengths": [20],
                            "output_chunk_lengths": [10],
                            "chunks": [
                                {
                                    "chunk_index": 0,
                                    "input_tokens": 20,
                                    "output_tokens": 10,
                                    "merged_pairs": 10,
                                    "selected_pair_starts": list(range(0, 20, 2)),
                                    "similarity_min": 0.1,
                                    "similarity_max": 0.9,
                                }
                            ],
                        }
                    ],
                },
                "compression_state": {
                    "removed_sequence_positions_by_sample": [
                        list(range(25, 35)),
                        [41, 43],
                    ]
                },
            },
            {
                **common,
                "event": "backward",
                "forward_index": 0,
                "gradient_present": True,
            },
        ]
        (native_output / "debug_diagnostic.jsonl").write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return NativeCLIResult(returncode=0)

    config = _config(
        compression_mode="compressed",
        runtime_mode="debug",
        output_dir=str(tmp_path / "compressed_train"),
    )
    result = MSSwiftTrainAdapter(command_runner=command_runner).train(config=config)

    evidence = result.backend_args["compression_training"]
    assert evidence["verified"] is True
    assert evidence["audio_backward_verified"] is True
    assert evidence["training_forward_only_verified"] is True
    assert evidence["template_bridge_verified"] is True
    assert evidence["profile_evidence_verified"] is True
    assert evidence["debug_evidence_verified"] is True
    assert evidence["observed_token_reduction"] is True
    evidence_path = Path(evidence["evidence_paths"][0])
    assert evidence_path.name == "debug_diagnostic.jsonl"
    assert evidence_path.parent == Path(result.train_log_path).parent
    assert (evidence_path.parent / "args.json").is_file()
    pretty_path = Path(evidence["diagnostic_pretty_paths"][0])
    assert pretty_path.name == "debug_diagnostic.pretty.json"
    assert pretty_path.parent == evidence_path.parent
    pretty_records = json.loads(pretty_path.read_text(encoding="utf-8"))
    assert pretty_records[0]["event"] == "summary"
    assert pretty_records[0]["all_forwards_have_backward"] is True
    assert pretty_records[0]["audio_tokens"] == {
        "original": 20,
        "kept": 10,
        "removed": 10,
        "keep_ratio": 0.5,
    }
    pretty_forward = next(
        record for record in pretty_records if record.get("event") == "forward"
    )
    pretty_layer = pretty_forward["compression"]["layers"][0]
    assert "chunks" not in pretty_layer
    assert pretty_layer["chunk_summary"] == {
        "count": 1,
        "merged_pairs_by_chunk": [10],
        "similarity_range": [0.1, 0.9],
    }
    assert pretty_forward["compression_state"][
        "removed_sequence_positions_by_sample"
    ] == [
        {"sample": 0, "count": 10, "half_open_spans": [[25, 35]]},
        {
            "sample": 1,
            "count": 2,
            "half_open_spans": [[41, 42], [43, 44]],
        },
    ]
    raw_records = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    raw_forward = next(
        record for record in raw_records if record.get("event") == "forward"
    )
    assert "chunks" in raw_forward["compression"]["layers"][0]
    manifest = build_run_manifest(config, train_backend=result.backend)
    assert manifest.trace_paths["compression_training_evidence_0"] == str(
        evidence_path
    )
    assert manifest.trace_paths["compression_training_pretty_0"] == str(
        pretty_path
    )


def test_compressed_training_rejects_validation_forward_evidence(
    tmp_path: Path,
) -> None:
    common = {
        "schema_version": 1,
        "spec_hash": "expected-spec",
        "stage_name": "audio_segmentwise_topk",
        "plugin": "segmentwise_topk_audio",
        "execution_phase": "forward",
    }
    records = [
        {**common, "event": "install"},
        {
            **common,
            "event": "forward",
            "forward_index": 0,
            "runtime_mode": "default",
            "bridge": "swift_template_pre_forward_hook",
            "loss_requires_grad": True,
            "compression": {"original_audio_tokens": 20, "kept_audio_tokens": 10},
        },
        {
            **common,
            "event": "backward",
            "forward_index": 0,
            "gradient_present": True,
        },
        {
            **common,
            "event": "forward",
            "forward_index": 1,
            "runtime_mode": "default",
            "bridge": "swift_template_pre_forward_hook",
            "loss_requires_grad": False,
            "compression": {"original_audio_tokens": 20, "kept_audio_tokens": 10},
        },
    ]
    evidence_path = tmp_path / "compression_training_rank_0.jsonl"
    evidence_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(BackendIntegrationError, match="validation/no-grad"):
        _collect_training_compression_evidence(
            tmp_path,
            expected_spec_hash="expected-spec",
            expected_runtime_mode="default",
        )


def test_operation_training_evidence_verifies_rewrite_and_backward(
    tmp_path: Path,
) -> None:
    common = {
        "schema_version": 1,
        "spec_hash": "operation-spec",
        "stage_name": "spade_operation_only",
        "plugin": "structured_llm_layer_pruning",
        "execution_phase": "prepare",
    }
    records = [
        {
            **common,
            "event": "install",
            "executor_family": "llm_operation",
            "lifecycle": "post_load_pre_optimizer",
            "operation": {
                "source_num_hidden_layers": 36,
                "result_num_hidden_layers": 33,
                "pruned_source_layers": [8, 17, 26],
                "parameter_reduction": 123,
                "requires_checkpoint_export": True,
            },
        },
        {
            **common,
            "event": "forward",
            "forward_index": 0,
            "runtime_mode": "default",
            "executor_family": "llm_operation",
            "loss_requires_grad": True,
        },
        {
            **common,
            "event": "backward",
            "forward_index": 0,
            "executor_family": "llm_operation",
            "gradient_present": True,
        },
    ]
    evidence_path = tmp_path / "compression_training_rank_0.jsonl"
    evidence_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = _collect_training_compression_evidence(
        tmp_path,
        expected_spec_hash="operation-spec",
        expected_runtime_mode="default",
        expected_executor_family="llm_operation",
    )

    assert result["verified"] is True
    assert result["pre_optimizer_install_verified"] is True
    assert result["operation_backward_verified"] is True
    assert result["result_num_hidden_layers"] == 33
    assert result["pruned_source_layers"] == [8, 17, 26]


def test_image_training_evidence_requires_token_reduction_and_backward(
    tmp_path: Path,
) -> None:
    common = {
        "schema_version": 1,
        "spec_hash": "image-spec",
        "stage_name": "tome-image-train",
        "plugin": "tome",
        "execution_phase": "forward",
        "executor_family": "image_multimodal",
    }
    records = [
        {**common, "event": "install"},
        {
            **common,
            "event": "forward",
            "forward_index": 0,
            "runtime_mode": "default",
            "loss_requires_grad": True,
            "use_cache": False,
            "layout_records": [{"source_count": 16, "result_count": 8}],
        },
        {
            **common,
            "event": "backward",
            "forward_index": 0,
            "gradient_present": True,
        },
    ]
    evidence_path = tmp_path / "compression_training_rank_0.jsonl"
    evidence_path.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = _collect_training_compression_evidence(
        tmp_path,
        expected_spec_hash="image-spec",
        expected_runtime_mode="default",
        expected_executor_family="image_multimodal",
    )

    assert result["verified"] is True
    assert result["image_backward_verified"] is True
    assert result["token_reduction_verified"] is True
    assert result["teacher_forcing_verified"] is True


def test_talker_training_is_rejected_instead_of_overclaimed() -> None:
    config = _config(train_args={"trainable_branch": "talker"})

    with pytest.raises(BackendIntegrationError, match="trainable_branch"):
        MSSwiftTrainAdapter().build_launch_plan(config)
