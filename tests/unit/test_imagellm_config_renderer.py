from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.integrations.ms_swift import MSSwiftTrainAdapter
from mllm_comp.pipeline.definition import resolve_compression_definition
from scripts.train.render_imagellm_config import BACKBONES, build_config, main
from scripts.train.verify_imagellm_train_step import ALGORITHMS


@pytest.mark.parametrize("algorithm", ALGORITHMS)
@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("mode", ("eval", "train"))
def test_renderer_matches_current_execution_contracts(
    algorithm: str,
    backbone: str,
    mode: str,
) -> None:
    config_args = dict(
        algorithm=algorithm,
        backbone=backbone,
        mode=mode,
        model_path="/models/checkpoint",
        dataset="/datasets/data.jsonl",
        output_dir="/runs/output",
    )
    if algorithm == "sgl" and mode == "train":
        with pytest.raises(ValueError, match="no forward binding for ms-swift training"):
            build_config(**config_args)
        return
    config = experiment_config_from_dict(build_config(**config_args))
    section = getattr(config, mode)

    definition = resolve_compression_definition(
        config,
        compression_mode=section.compression_mode,
        interface=mode,
        runtime_batch_size=1 if mode == "eval" else None,
    )

    assert len(definition.stages) == 1
    assert definition.stages[0].algorithm == algorithm
    if mode == "train":
        MSSwiftTrainAdapter().build_launch_plan(config)


@pytest.mark.parametrize("backbone", BACKBONES)
@pytest.mark.parametrize("mode", ("eval", "train"))
def test_renderer_refuses_d2pruner_without_prior(backbone: str, mode: str) -> None:
    with pytest.raises(ValueError, match="calibrated prior"):
        build_config(
            algorithm="d2pruner",
            backbone=backbone,
            mode=mode,
            model_path="/models/checkpoint",
            dataset="/datasets/data.jsonl",
            output_dir="/runs/output",
        )


def test_renderer_cli_writes_reloadable_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    written: dict[str, str] = {}

    monkeypatch.setattr(Path, "mkdir", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        Path,
        "write_text",
        lambda _path, content, **_kwargs: written.setdefault("content", content),
    )

    assert (
        main(
            [
                "--algorithm",
                "tome",
                "--backbone",
                "qwen3_vl",
                "--mode",
                "train",
                "--model_path",
                "/models/checkpoint",
                "--dataset",
                "/datasets/data.jsonl",
                "--output_dir",
                "/runs/output",
                "--output",
                "/virtual/qwen3-vl-tome.yaml",
            ]
        )
        == 0
    )
    reloaded = experiment_config_from_dict(
        yaml.safe_load(written["content"])
    )

    assert reloaded.pipeline.stages[0].placement.value == "encoder_level"


def test_llava_training_config_uses_vendored_source_route() -> None:
    config = build_config(
        algorithm="tome",
        backbone="llava_onevision",
        mode="train",
        model_path="/models/checkpoint",
        dataset="/datasets/data.jsonl",
        output_dir="/runs/output",
    )

    args = config["train"]["args"]
    assert args["local_repo_path"] == (
        "third_party/backbones/LLaVA-NeXT"
    )
    assert args["gradient_checkpointing"] is True
    assert args["vit_gradient_checkpointing"] is False

    parsed = experiment_config_from_dict(config)
    plan = MSSwiftTrainAdapter().build_launch_plan(parsed)
    option = plan.argv.index("--vit_gradient_checkpointing")
    assert plan.argv[option + 1] == "false"

    parsed.train.args.pop("vit_gradient_checkpointing")
    with pytest.raises(BackendIntegrationError, match="cannot be replayed safely"):
        MSSwiftTrainAdapter().build_launch_plan(parsed)
