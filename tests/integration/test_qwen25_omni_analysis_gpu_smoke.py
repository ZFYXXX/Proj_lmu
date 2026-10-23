"""Opt-in synthetic-audio GPU smoke for Qwen2.5-Omni analysis."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from tests.fixtures.qwen25_omni_audio import create_audio as _create_audio

from mllm_comp.analysis.definition import AnalysisOverrides, resolve_analysis_definition
from mllm_comp.analysis.records import AnalysisSample
from mllm_comp.analysis.runtime import execute_analysis
from mllm_comp.backbones.qwen25_omni import (
    DEFAULT_AUDIO_SYSTEM_PROMPT,
    Qwen25OmniSample,
)
from mllm_comp.experiments import analysis_invariants_passed
from mllm_comp.config.loader import load_config
from mllm_comp.core.serialization import atomic_write_json, to_jsonable
from mllm_comp.run_manifest import build_run_manifest


_SCENARIO_CONFIGS = {
    "baseline": "examples/analysis/qwen25_omni_audio_analysis_matrix.yaml",
    "segmentwise": (
        "examples/analysis/compression/"
        "qwen25_omni_audio_segmentwise_representation.yaml"
    ),
    "atome": (
        "examples/analysis/compression/"
        "qwen25_omni_audio_atome_representation.yaml"
    ),
    "pyramid": (
        "examples/analysis/compression/"
        "qwen25_omni_audio_pyramid_representation.yaml"
    ),
    "spade": (
        "examples/analysis/compression/"
        "qwen25_omni_audio_spade_representation.yaml"
    ),
}


class _SyntheticAudioAdapter:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path

    def iter_analysis_samples(
        self,
        *,
        config: object,
        backbone: object,
        runtime: object,
        sample_limit: int | None,
        requires_teacher_forcing: bool = False,
    ):
        del config
        assert sample_limit == 1
        sample = Qwen25OmniSample(
            sample_id="synthetic-audio",
            conversation=[
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": DEFAULT_AUDIO_SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "audio", "audio": str(self.audio_path)},
                        {"type": "text", "text": "Briefly describe the sound."},
                    ],
                },
            ],
        )
        processed = backbone.processor_adapter.prepare_inputs(
            runtime.processor,
            sample,
            phase="eval",
        )
        target_text = "A steady tone is audible."
        record = AnalysisSample(
            sample_id=sample.sample_id,
            task_name="synthetic_audio_smoke",
            processor_outputs=dict(processed.model_inputs),
            target_text=target_text,
            target_source="synthetic_fixture",
            media=[str(self.audio_path)],
            metadata={
                "prompt": "Briefly describe the sound.",
                "prompt_lengths": list(processed.prompt_lengths),
                **dict(processed.metadata),
            },
        )
        if requires_teacher_forcing:
            teacher_forced = backbone.processor_adapter.prepare_teacher_forced_inputs(
                runtime.processor,
                sample,
                target_text=target_text,
                prompt_batch=processed,
                phase="eval",
            )
            objective = teacher_forced.metadata.get("teacher_forcing")
            assert isinstance(objective, dict)
            record["teacher_forced_processor_outputs"] = dict(teacher_forced.model_inputs)
            record["objective"] = dict(objective)
        yield record

    def dependency_report(self) -> dict[str, object]:
        return {"versions": {"lmms-eval": "synthetic-audio-adapter"}}


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("MLLM_COMP_RUN_QWEN25_OMNI_ANALYSIS_GPU_SMOKE") != "1",
    reason="Set MLLM_COMP_RUN_QWEN25_OMNI_ANALYSIS_GPU_SMOKE=1 to run.",
)
def test_qwen25_omni_audio_analysis_gpu_smoke() -> None:
    torch = pytest.importorskip("torch")
    assert torch.cuda.is_available()
    assert torch.cuda.device_count() == 1
    scenario = os.environ.get("QWEN25_OMNI_ANALYSIS_SCENARIO", "baseline")
    assert scenario in _SCENARIO_CONFIGS
    model_path = Path(os.environ["QWEN25_OMNI_MODEL_PATH"])
    output_root = Path(os.environ["QWEN25_OMNI_ANALYSIS_SMOKE_OUTPUT"])
    assert model_path.is_dir()
    assert not output_root.exists()
    output_root.mkdir(parents=True)
    audio_path = output_root / "synthetic-1s-16khz.wav"
    _create_audio(audio_path)

    config = load_config(_SCENARIO_CONFIGS[scenario])
    config.experiment.name = f"qwen25_omni_audio_analysis_{scenario}_gpu_smoke"
    config.backbone.model_path = str(model_path)
    config.backbone.model_id = None
    config.backbone.config["model_load_kwargs"] = {"local_files_only": True}
    config.backbone.config["processor_load_kwargs"] = {"local_files_only": True}
    config.analysis.tasks = ["synthetic_audio_smoke"]
    config.output.run_dir = str(output_root)
    config.output.manifest_path = str(output_root / "manifest.json")
    definition = resolve_analysis_definition(config, AnalysisOverrides())
    config.analysis.probes = [dict(value) for value in definition.probes]

    torch.cuda.reset_peak_memory_stats()
    result = execute_analysis(
        config,
        definition,
        device="cuda:0",
        adapter=_SyntheticAudioAdapter(audio_path),
    )
    assert result.sample_count == 1
    assert analysis_invariants_passed(result)
    assert len(result.records) == (7 if scenario == "baseline" else 1)
    for record in result.records:
        assert record.sample_count == 1
        assert record.summary["skipped_samples"] == 0
        assert record.artifact_paths
        assert all(Path(path).is_file() for path in record.artifact_paths.values())

    if scenario != "baseline":
        assert result.compression_runtime_metadata is not None
        assert result.compression_runtime_metadata["activation_count"] == 1
        activation = result.compression_runtime_metadata["activations"][0]
        assert activation["algorithm"] == config.pipeline.stages[0].algorithm

    manifest = build_run_manifest(
        config,
        eval_backend=result.backend,
        analysis_records=result.records,
        checkpoint_path=str(model_path),
        invariants_passed=True,
        invariants_run=[
            "AnalysisSemanticBinding",
            "AnalysisTokenGroupLayout",
            "AnalysisHookCleanup",
        ],
        compression_runtime_metadata=result.compression_runtime_metadata,
    )
    assert manifest.invariants_passed is True
    assert len(manifest.analysis_records) == len(result.records)
    atomic_write_json(config.output.manifest_path, manifest)
    atomic_write_json(
        output_root / "smoke_summary.json",
        {
            "scenario": scenario,
            "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "analysis_records": to_jsonable(result.records),
            "compression_runtime": to_jsonable(
                result.compression_runtime_metadata
            ),
            "manifest_path": config.output.manifest_path,
        },
    )
    print(
        json.dumps(
            {
                "scenario": scenario,
                "record_count": len(result.records),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "manifest_path": config.output.manifest_path,
            },
            ensure_ascii=False,
        )
    )
