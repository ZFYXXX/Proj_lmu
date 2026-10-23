"""Validate the exact METok/LLaVA-OneVision five-step training cell.

This offline, batch-one evidence harness trains only the facade-owned final
normalization weight on the shared doc-442 test-derived fixture. Every forward
must execute event-aware projector pooling and progressive visual pruning
before decoder layers 3, 10, and 19. Training deliberately uses
``use_cache=False``; its three boundary records must therefore prove that no
DynamicCache was created or compacted. The local SigLIP text capability stays
frozen and outside both optimizer and lightweight-checkpoint ownership.
"""

from __future__ import annotations

import argparse
import gc
import time
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.llava_onevision import (
    LlavaOneVisionSample,
    create_llava_onevision_backbone,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics import validate_llava_metok_video_compression_checkpoint as inference
from scripts.diagnostics.video_checkpoint_common import (
    prepare_checkpoint_environment,
    checkpoint_gpu_report,
    _checkpoint_record,
    _gpu_snapshot,
)
from scripts.diagnostics.video_training_checkpoint_common import (
    training_report,
    FIXTURE_DOC_ID,
    OPTIMIZER_STEPS,
    append_jsonl,
    canonical,
    checkpoint_saved,
    final_norm_optimizer,
    inference_forward,
    inference_state,
    llava_training_fixture as _fixture,
    load_lightweight_checkpoint,
    load_training_runtime,
    prepare_training_device,
    print_training_step,
    record_post_reload,
    record_training_step,
    save_training_checkpoint,
    target_parameter_checks,
    training_activation,
    training_cache_exact as _training_cache_exact,
    training_document,
    training_environment_report,
    training_forward_step,
    training_main,
    training_parameter_report,
    training_parser,
    training_state as _training_state,
)

CANONICAL_METOK_CONFIG = inference.CANONICAL_METOK_CONFIG
METOK_BOUNDARY_LAYERS = inference.METOK_BOUNDARY_LAYERS
FIXED_NEXTQA_DOC_IDS = inference.FIXED_NEXTQA_DOC_IDS
SOURCE_TOKENS_PER_FRAME = 196
EXPECTED_FRAME_COUNT = 32
SCOPE = {
    "algorithm": "metok",
    "variant": "metok.video",
    "backbone": "llava_onevision",
    "binding": "metok.video.llava_onevision",
}


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "metok-video-llava-onevision-five-step-train",
            "placement": "projector_level",
            "execution_phase": "forward",
            "plugin": "metok",
            "algorithm": SCOPE["algorithm"],
            "variant": SCOPE["variant"],
            "binding": SCOPE["binding"],
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "config": {},
        }
    )


def _load_runtime(
    checkpoint: Path,
    siglip_checkpoint: Path,
    *,
    torch: Any,
    train: bool,
) -> tuple[Any, Any, Any, BackboneRuntime]:
    facade, processor, model, runtime = load_training_runtime(
        checkpoint,
        facade=create_llava_onevision_backbone(),
        torch=torch,
        train=train,
        device_map="cuda:0",
    )
    runtime.metadata["metok_siglip_path"] = str(siglip_checkpoint)
    return facade, processor, model, runtime


_activation = partial(training_activation, canonical_config=CANONICAL_METOK_CONFIG)


def _training_pruning_chain_exact(
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    latest_metok: Sequence[Mapping[str, Any]],
    *,
    projector_count: int,
) -> bool:
    if not (
        len(pruning_records)
        == len(cache_records)
        == len(latest_metok)
        == len(METOK_BOUNDARY_LAYERS)
    ):
        return False
    prior_visual = projector_count

    prior_sequence_result: int | None = None
    for stage_index, boundary_layer in enumerate(METOK_BOUNDARY_LAYERS):
        record = pruning_records[stage_index]
        cache = cache_records[stage_index]
        bridge = latest_metok[stage_index]
        sequence = record.get("sequence_layout_delta", {})
        source_visual = int(record.get("source_count", -1))
        result_visual = int(record.get("result_count", -1))
        source_sequence = int(sequence.get("source_count", -1))
        result_sequence = int(sequence.get("result_count", -1))
        kept = tuple(int(item) for item in record.get("kept_visual_indices", ()))
        dropped = tuple(int(item) for item in record.get("dropped_visual_indices", ()))
        retained = tuple(
            int(item)
            for item in record.get("cumulative_retained_projector_indices", ())
        )
        exact = (
            int(record.get("boundary_layer", -1)) == boundary_layer
            and int(record.get("stage_index", -1)) == stage_index
            and source_visual == prior_visual
            and result_visual == len(kept)
            and tuple(sorted((*kept, *dropped))) == tuple(range(source_visual))
            and len(retained) == result_visual
            and retained == tuple(sorted(set(retained)))
            and source_sequence - result_sequence == source_visual - result_visual
            and len(tuple(record.get("kept_sequence_indices", ()))) == result_sequence
            and len(tuple(record.get("cumulative_prompt_keep_indices", ())))
            == result_sequence
            and bool(record.get("final_visual_removal"))
            is (stage_index == len(METOK_BOUNDARY_LAYERS) - 1)
            and dict(record.get("cache", {})) == dict(cache)
            and _training_cache_exact(
                cache,
                boundary_layer=boundary_layer,
                source_length=source_sequence,
                result_length=result_sequence,
            )
            and int(bridge.get("boundary_layer", -1)) == boundary_layer
            and int(bridge.get("stage_index", -1)) == stage_index
            and int(bridge.get("source_visual_count", -1)) == source_visual
            and int(bridge.get("result_visual_count", -1)) == result_visual
            and int(bridge.get("source_sequence_length", -1)) == source_sequence
            and int(bridge.get("result_sequence_length", -1)) == result_sequence
            and bool(bridge.get("structural_tokens_preserved"))

        )
        if not exact:
            return False
        if (
            prior_sequence_result is not None
            and source_sequence != prior_sequence_result
        ):
            return False
        prior_visual = result_visual

        prior_sequence_result = result_sequence
    return prior_visual == 0


def _metok_algorithm_checks(
    *,
    projector: Mapping[str, Any] | None,
    algorithm: Mapping[str, Any] | None,
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    latest_metok: Sequence[Mapping[str, Any]],
    provider: Mapping[str, Any] | None,
    provider_capture_delta: int,
    frame_count: int,
) -> dict[str, bool]:
    projector = {} if projector is None else projector
    algorithm = {} if algorithm is None else algorithm
    splice = {} if splice is None else splice
    provider = {} if provider is None else provider
    source_count = int(projector.get("source_count", 0))
    projector_count = int(projector.get("result_count", 0))
    structural_source = tuple(splice.get("structural_source_positions", ()))
    structural_result = tuple(splice.get("structural_result_positions", ()))
    source_span = tuple(int(item) for item in splice.get("source_visual_span", ()))
    result_span = tuple(int(item) for item in splice.get("result_visual_span", ()))
    pruning_exact = _training_pruning_chain_exact(
        pruning_records,
        cache_records,
        latest_metok,
        projector_count=projector_count,
    )
    return {
        "canonical_source_raster_exact": (
            frame_count == EXPECTED_FRAME_COUNT
            and source_count == frame_count * SOURCE_TOKENS_PER_FRAME
            and int(algorithm.get("frame_count", 0)) == frame_count
            and int(algorithm.get("grid_height", 0)) == 14
            and int(algorithm.get("grid_width", 0)) == 14
            and int(algorithm.get("tokens_per_frame", 0)) == SOURCE_TOKENS_PER_FRAME
        ),
        "projector_non_identity_reduction": source_count > projector_count > 0,
        "projector_layout_and_splice_counts_agree": (
            int(algorithm.get("source_count", 0)) == source_count
            and int(algorithm.get("result_count", 0)) == projector_count
            and int(splice.get("source_patch_count", 0)) == source_count
            and int(splice.get("result_patch_count", 0)) == projector_count
        ),
        "canonical_config_exact": inference._canonical_config_exact(
            algorithm.get("canonical_config", {})
        ),
        "projected_and_raw_signals_exact": (
            tuple(algorithm.get("projected_features_shape", ()))[:1] == (source_count,)
            and tuple(algorithm.get("result_features_shape", ()))[:1]
            == (projector_count,)
            and tuple(algorithm.get("raw_features_shape", ()))[:1]
            == (frame_count * 729,)
            and int(algorithm.get("raw_tokens_per_frame", 0)) == 729
            and bool(algorithm.get("projected_features_finite"))
            and bool(algorithm.get("result_features_finite"))
            and bool(algorithm.get("raw_features_finite"))
            and bool(algorithm.get("frame_text_similarity_finite"))
            and len(tuple(algorithm.get("frame_text_similarities", ()))) == frame_count
        ),
        "local_frozen_siglip_text_provider_exact": (
            algorithm.get("similarity_source") == "native_siglip_visual_features"
            and bool(algorithm.get("assistant_answer_excluded"))
            and bool(algorithm.get("siglip_local_only"))
            and provider_capture_delta == 1
            and bool(provider.get("installed"))
            and bool(provider.get("siglip_loaded"))
            and provider.get("siglip_vision_enabled") is False
            and bool(provider.get("siglip_frozen"))
            and bool(provider.get("siglip_local_only"))
            and bool(provider.get("assistant_answer_excluded"))
            and bool(provider.get("siglip_path"))
        ),
        "three_stage_no_cache_pruning_exact": pruning_exact,
        "final_visual_removal_exact": bool(
            pruning_exact
            and int(pruning_records[-1].get("result_count", -1)) == 0
            and int(latest_metok[-1].get("result_visual_count", -1)) == 0
        ),
        "native_structural_tokens_preserved": (
            len(source_span) == len(result_span) == 2
            and len(structural_source) == len(structural_result)
            and source_span[1] - source_span[0] == source_count + len(structural_source)
            and result_span[1] - result_span[0]
            == projector_count + len(structural_result)
        ),
    }


def _step_checks(
    *,
    loss: Any,
    logits: Any,
    projector_records: Sequence[Mapping[str, Any]],
    algorithm_records: Sequence[Mapping[str, Any]],
    pruning_records: Sequence[Mapping[str, Any]],
    cache_records: Sequence[Mapping[str, Any]],
    splice: Mapping[str, Any] | None,
    latest_metok: Sequence[Mapping[str, Any]],
    provider: Mapping[str, Any] | None,
    provider_capture_delta: int,
    parameters: Sequence[Mapping[str, Any]],
    stable_layout: bool,
    frame_count: int,
    torch: Any,
) -> dict[str, bool]:
    exact_record_counts = len(projector_records) == len(algorithm_records) == 1 and len(
        pruning_records
    ) == len(cache_records) == len(latest_metok) == len(METOK_BOUNDARY_LAYERS)
    final_sequence = (
        int(latest_metok[-1].get("result_sequence_length", 0)) if latest_metok else 0
    )
    return {
        "one_projector_and_three_pruning_invocations": exact_record_counts,
        "compression_applied": bool(
            splice is not None and splice.get("compression_applied")
        ),
        "finite_loss": bool(loss is not None and torch.isfinite(loss).all().item()),
        "finite_logits": bool(
            logits is not None and torch.isfinite(logits).all().item()
        ),
        "post_layer19_label_logit_layout_aligned": bool(
            logits is not None
            and final_sequence > 0
            and int(logits.shape[1]) == final_sequence
        ),
        "layout_stable_across_steps": stable_layout,
        **_metok_algorithm_checks(
            projector=(projector_records[0] if len(projector_records) == 1 else None),
            algorithm=(algorithm_records[0] if len(algorithm_records) == 1 else None),
            pruning_records=pruning_records,
            cache_records=cache_records,
            splice=splice,
            latest_metok=latest_metok,
            provider=provider,
            provider_capture_delta=provider_capture_delta,
            frame_count=frame_count,
        ),
        **target_parameter_checks(parameters),
    }


def _run(args: argparse.Namespace) -> dict[str, Any]:
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    siglip_checkpoint = args.siglip_checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()
    output_dir = args.output_dir.resolve()
    progress_path = output_dir / "progress.jsonl"
    if not parquet_path.is_file():
        raise RuntimeError(f"NExT-QA Parquet is missing: {parquet_path}")
    if not video_root.is_dir():
        raise RuntimeError(f"NExT-QA video root is missing: {video_root}")
    checkpoint_info = _checkpoint_record(checkpoint)
    siglip_checkpoint_info = _checkpoint_record(siglip_checkpoint)
    document, video = training_document(parquet_path, video_root, FIXED_NEXTQA_DOC_IDS)
    fixture, fixture_path = _fixture(
        document=document,
        video=video,
        output_dir=output_dir,
    )
    append_jsonl(
        progress_path,
        {
            "event": "start",
            "scope": SCOPE,
            "code_revision": args.code_revision,
            "fixture_doc_id": FIXTURE_DOC_ID,
            "intended_trainable_handle": "llm.final_norm.weight",
            "defining_stages": [
                "event_segment_keyframe_adaptive_pool",
                "decoder_prune_layers_3_10_19",
            ],
        },
    )

    torch, transformers, gpu_before, admission = prepare_training_device(args, framework_label='LLaVA')
    torch.use_deterministic_algorithms(True, warn_only=True)
    started = time.time()
    stage = _stage()
    sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:sft",
        prompt=str(document["prompt"]),
        video=str(video),
    )

    facade, processor, model, runtime = _load_runtime(
        checkpoint,
        siglip_checkpoint,
        torch=torch,
        train=True,
    )
    batch, state = _training_state(
        facade=facade,
        processor=processor,
        sample=sample,
        target=str(document["target"]),
    )
    frame_count = int(batch.metadata["video"]["frame_count"])
    selected, optimizer, optimizer_config = final_norm_optimizer(
        facade, model, learning_rate=args.learning_rate, torch=torch,
    )
    selected_names = [name for name, _ in selected]
    activation = _activation(
        facade=facade,
        model=model,
        runtime=runtime,
        stage=stage,
    )
    activation.install()
    step_records: list[dict[str, Any]] = []
    first_layout: Any | None = None
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            metadata_before = activation.runtime_metadata()
            projector_before = int(metadata_before["projector_invocations"])
            pruning_before = int(metadata_before["pruning_invocations"])
            provider_before = metadata_before.get("signal_provider") or {}
            provider_capture_before = int(provider_before.get("capture_count", 0))

            loss, logits, parameter_evidence = training_forward_step(
                facade=facade, runtime=runtime, state=state, optimizer=optimizer,
                selected=selected, torch=torch,
            )

            metadata = activation.runtime_metadata()
            projector_records = metadata["projector_layout_deltas"]
            algorithm_records = metadata["algorithm_records"]
            pruning_records = metadata["pruning_layout_deltas"]
            cache_records = metadata["cache_repairs"]
            provider = metadata.get("signal_provider") or {}
            provider_capture_delta = (
                int(provider.get("capture_count", 0)) - provider_capture_before
            )
            bridge = metadata.get("bridge") or {}
            splice = bridge.get("latest_splice")
            latest_metok = tuple(bridge.get("latest_metok") or ())
            current_layout = canonical(
                {
                    "projector": (
                        projector_records[0] if len(projector_records) == 1 else None
                    ),
                    "algorithm": (
                        algorithm_records[0] if len(algorithm_records) == 1 else None
                    ),
                    "pruning": pruning_records,
                    "cache": cache_records,
                    "splice": splice,
                    "latest_metok": latest_metok,
                    "provider": {
                        key: value
                        for key, value in provider.items()
                        if key != "capture_count"
                    },
                }
            )
            if first_layout is None:
                first_layout = current_layout
            checks = _step_checks(
                loss=loss,
                logits=logits,
                projector_records=projector_records,
                algorithm_records=algorithm_records,
                pruning_records=pruning_records,
                cache_records=cache_records,
                splice=splice,
                latest_metok=latest_metok,
                provider=provider,
                provider_capture_delta=provider_capture_delta,
                parameters=parameter_evidence,
                stable_layout=current_layout == first_layout,
                frame_count=frame_count,
                torch=torch,
            )
            checks["one_projector_and_three_pruning_invocations"] = bool(
                checks["one_projector_and_three_pruning_invocations"]
                and int(metadata["projector_invocations"]) == projector_before + 1
                and int(metadata["pruning_invocations"]) == pruning_before + 3
            )
            record = {
                "event": "optimizer_step",
                "step": step,
                "loss": float(loss.detach().float().item()),
                "logits_shape": [int(item) for item in logits.shape],
                "trainable_parameters": parameter_evidence,
                "checks": checks,
            }
            if step == 1:
                record["native_layout"] = current_layout
            record_training_step(progress_path, step_records, record)
            print_training_step(
                record,
                source_visual_tokens=projector_records[0]['source_count'],
                projector_visual_tokens=projector_records[0]['result_count'],
                layer3_visual_tokens=pruning_records[0]['result_count'],
                layer10_visual_tokens=pruning_records[1]['result_count'],
                layer19_visual_tokens=pruning_records[2]['result_count'],
                gradient_norm=parameter_evidence[0]['gradient_norm'],
                parameter_delta=parameter_evidence[0]['max_parameter_delta'],
            )
        lightweight_checkpoint = save_training_checkpoint(
            args, records=step_records, selected=selected,
            scope=SCOPE, canonical_config=CANONICAL_METOK_CONFIG,
            optimizer=optimizer_config, optimizer_steps=OPTIMIZER_STEPS, torch=torch,
        )
        checkpoint_saved(progress_path, lightweight_checkpoint, optimizer_steps=OPTIMIZER_STEPS)
    finally:
        if activation.installed:
            activation.uninstall()

    del optimizer, selected, state, batch, runtime, model, processor, facade, activation
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    reload_facade, reload_processor, reload_model, reload_runtime = _load_runtime(
        checkpoint,
        siglip_checkpoint,
        torch=torch,
        train=False,
    )
    reload_result = load_lightweight_checkpoint(
        reload_model,
        lightweight_checkpoint,
        torch=torch,
    )
    reload_sample = LlavaOneVisionSample(
        sample_id=f"nextqa_mc_test:{FIXTURE_DOC_ID}:reload",
        prompt=str(document["prompt"]),
        video=str(video),
    )
    reload_batch, reload_state = inference_state(
        facade=reload_facade, processor=reload_processor, sample=reload_sample,
    )
    reload_frame_count = int(reload_batch.metadata["video"]["frame_count"])
    reload_activation = _activation(
        facade=reload_facade,
        model=reload_model,
        runtime=reload_runtime,
        stage=stage,
    )
    reload_activation.install()
    try:
        reload_logits = inference_forward(
            facade=reload_facade, runtime=reload_runtime, state=reload_state, torch=torch,
        )
        reload_metadata = reload_activation.runtime_metadata()
        reload_projectors = reload_metadata["projector_layout_deltas"]
        reload_algorithms = reload_metadata["algorithm_records"]
        reload_pruning = reload_metadata["pruning_layout_deltas"]
        reload_caches = reload_metadata["cache_repairs"]
        reload_provider = reload_metadata.get("signal_provider") or {}
        reload_bridge = reload_metadata.get("bridge") or {}
        reload_splice = reload_bridge.get("latest_splice")
        reload_latest_metok = tuple(reload_bridge.get("latest_metok") or ())
        reload_algorithm_checks = _metok_algorithm_checks(
            projector=(reload_projectors[0] if len(reload_projectors) == 1 else None),
            algorithm=(reload_algorithms[0] if len(reload_algorithms) == 1 else None),
            pruning_records=reload_pruning,
            cache_records=reload_caches,
            splice=reload_splice,
            latest_metok=reload_latest_metok,
            provider=reload_provider,
            provider_capture_delta=int(reload_provider.get("capture_count", 0)),
            frame_count=reload_frame_count,
        )
        final_sequence_length = (
            int(reload_latest_metok[-1].get("result_sequence_length", 0))
            if reload_latest_metok
            else 0
        )
        post_reload_checks = {
            "checkpoint_tensors_exact": bool(reload_result["all_exact"]),
            "one_projector_and_three_pruning_invocations": bool(
                int(reload_metadata["projector_invocations"]) == 1
                and int(reload_metadata["pruning_invocations"]) == 3
                and len(reload_projectors) == len(reload_algorithms) == 1
                and len(reload_pruning)
                == len(reload_caches)
                == len(reload_latest_metok)
                == len(METOK_BOUNDARY_LAYERS)
            ),
            "compression_applied": bool(
                reload_splice is not None and reload_splice.get("compression_applied")
            ),
            "labels_absent": reload_state.labels is None,
            "finite_logits": bool(
                reload_logits is not None and torch.isfinite(reload_logits).all().item()
            ),
            "post_layer19_layout_aligned": bool(
                reload_logits is not None
                and final_sequence_length > 0
                and int(reload_logits.shape[1]) == final_sequence_length
            ),
            **reload_algorithm_checks,
        }
        post_reload = {
            "checkpoint_reload": reload_result,
            "logits_shape": (
                [int(item) for item in reload_logits.shape]
                if reload_logits is not None
                else None
            ),
            "projector_compression": (
                canonical(reload_projectors[0]) if len(reload_projectors) == 1 else None
            ),
            "algorithm": (
                canonical(reload_algorithms[0]) if len(reload_algorithms) == 1 else None
            ),
            "pruning_compression": canonical(reload_pruning),
            "cache": canonical(reload_caches),
            "splice": canonical(reload_splice),
            "bridge_latest_metok": canonical(reload_latest_metok),
            "provider": canonical(reload_provider),
            "checks": post_reload_checks,
        }
        record_post_reload(progress_path, post_reload)
    finally:
        if reload_activation.installed:
            reload_activation.uninstall()

    torch.cuda.synchronize()
    gpu_after = _gpu_snapshot(args.physical_gpu)
    return training_report(
        args,
        scope=SCOPE,
        canonical_config=CANONICAL_METOK_CONFIG,
        checkpoint=checkpoint_info,
        fixture_path=fixture_path,
        started=started,
        provenance={
            "siglip_checkpoint": siglip_checkpoint_info,
        },
        boundaries={
            "single_sample_multiple_videos": False,
        },
        environment={
            **training_environment_report(visible=visible, torch=torch, transformers=transformers),
            "deterministic_algorithms": True,
        },
        fixture=fixture,
        training={
            **training_parameter_report(
                optimizer_config,
                selected_names,
                step_records,
                optimizer_steps=OPTIMIZER_STEPS,
            ),
            "siglip_trainable_parameters": 0,
            "frame_count": frame_count,
            "defining_stages": ["event_segment_keyframe_adaptive_pool", "decoder_prune_layers_3_10_19"],
            "use_cache": False,
        },
        lightweight_checkpoint=lightweight_checkpoint,
        post_reload_compressed_inference=post_reload,
        gpu=checkpoint_gpu_report(
            torch=torch,
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            estimate_basis="10.080 GiB evidenced METok generation reservation plus "
            "final-norm-only autograd; the explicit launch bound and "
            "safety margin are frozen in admission",
        ),
    )


def _parser() -> argparse.ArgumentParser:
    parser = training_parser(estimated_peak_mib=32768)
    parser.add_argument("--siglip_checkpoint", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    return training_main(argv, parser=_parser(), run=_run, scope=SCOPE, gpu_snapshot=_gpu_snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
