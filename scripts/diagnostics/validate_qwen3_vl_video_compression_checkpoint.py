"""Validate one canonical Qwen3-VL video-compression inference cell.

The diagnostic is offline, batch-one, and single-GPU.  It runs the same four
fixed ``nextqa_mc_test`` documents through the native baseline and one exact
video binding, then freezes Qwen3-VL's interleaved video placeholders, sparse
M-RoPE, synchronized DeepStack streams, ``visual_pos_masks``, prompt/cache
repair, finite generation, and GPU evidence needed by that support unit.

The shared harness admits an algorithm only through an explicit profile whose
sample validator checks that algorithm's actual execution and native layout.
The default profile remains canonical VQToken.
"""

from __future__ import annotations

import argparse
import shlex
import sys
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen3VLSample,
    create_qwen3_vl_backbone,
)
from mllm_comp.pipeline.stage import CompressionStageConfig
from scripts.diagnostics.video_checkpoint_common import (
    add_checkpoint_arguments,
    admit_generation_device,
    compare_generation,
    generation_activation,
    generation_case,
    generation_documents,
    generation_environment,
    checkpoint_gpu_report,
    generation_sample,
    inference_main,
    load_generation_runtime,
    prepare_checkpoint_environment,
    prepare_generation_state,
)

if __package__:
    from scripts.diagnostics import qwen_video_checkpoint_common as _qwen25
    from scripts.diagnostics import video_checkpoint_common as _common
else:
    import qwen_video_checkpoint_common as _qwen25
    import video_checkpoint_common as _common


CANONICAL_VQTOKEN_CONFIG = _common.CANONICAL_VQTOKEN_CONFIG
FIXED_NEXTQA_DOC_IDS = _common.FIXED_NEXTQA_DOC_IDS
_DecoderInputAudit = _common._DecoderInputAudit
_checkpoint_record = _common._checkpoint_record
_gpu_snapshot = _common._gpu_snapshot
_shape = _common._shape
_tensor_rows = _common._tensor_rows
_float_values = _qwen25.float_values
_generation_report = _qwen25.generation_report


QWEN3_VIDEO_FPS = 1.0
QWEN3_VIDEO_MAX_PIXELS = 128 * 32 * 32


@dataclass(frozen=True)
class Qwen3VideoDiagnosticProfile:
    """Algorithm-specific assertions plugged into the shared Qwen3 harness."""

    algorithm: str
    variant: str
    binding: str
    canonical_config: Mapping[str, Any]
    stage_factory: Callable[[], CompressionStageConfig]
    conversation_factory: Callable[..., list[dict[str, Any]]]
    sample_validator: Callable[..., dict[str, bool]]
    preprocessing_record: Callable[[argparse.Namespace], Mapping[str, Any]]
    projector_label: str
    reference_revision: str | None = None
    post_boundary_layer: int | None = None
    post_boundary_layers: tuple[int, ...] = ()
    auxiliary_checkpoints: tuple[tuple[str, str], ...] = ()
    secondary_layout_key: str = "vtp_layout_deltas"
    runtime_metadata_factory: Callable[[Any], Mapping[str, Any]] | None = None
    min_new_tokens: int = 2


def _conversation(
    *,
    prompt: str,
    video: Path,
    fps: float = QWEN3_VIDEO_FPS,
    max_pixels: int = QWEN3_VIDEO_MAX_PIXELS,
) -> list[dict[str, Any]]:
    if fps <= 0:
        raise ValueError("Qwen3-VL video fps must be positive")
    if max_pixels <= 0:
        raise ValueError("Qwen3-VL video max_pixels must be positive")
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(video),
                    "fps": float(fps),
                    "max_pixels": int(max_pixels),
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]


def _stage() -> CompressionStageConfig:
    return CompressionStageConfig.from_dict(
        {
            "name": "vqtoken-video-qwen3-vl-real-checkpoint",
            "placement": "projector_level",
            "execution_phase": "generate",
            "plugin": "vqtoken",
            "algorithm": "vqtoken",
            "variant": "vqtoken.video",
            "target_modalities": ["video"],
            "conditioning_modalities": [],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "binding": "vqtoken.video.qwen3_vl",
            "config": {},
        }
    )


def _spans(value: Any) -> tuple[tuple[int, int], ...]:
    return tuple(tuple(int(item) for item in span) for span in value)


def _span_count(spans: Sequence[Sequence[int]]) -> int:
    return sum(int(end) - int(start) for start, end in spans)


def _detector_calls(metadata: Mapping[str, Any]) -> int:
    provider = metadata.get("signal_provider")
    detector = provider.get("detector") if isinstance(provider, Mapping) else None
    return int(detector.get("calls", 0)) if isinstance(detector, Mapping) else 0


def _validate_vqtoken_sample(
    *,
    baseline: Mapping[str, Any],
    compressed: Mapping[str, Any],
    activation_records: Sequence[Mapping[str, Any]],
    latest_splice: Mapping[str, Any],
    latest_generation: Mapping[str, Any],
    source_grid_thw: Sequence[int],
    source_grid_token_count: int,
    labels_absent: bool,
    source_placeholder_spans: Sequence[Sequence[int]],
    native_video_count: int,
    expected_deepstack_branches: int,
    **_unused: Any,
) -> dict[str, bool]:
    if len(activation_records) != 1:
        raise RuntimeError(
            "VQToken must publish one projector LayoutDelta per sample, got "
            f"{len(activation_records)}"
        )
    layout = activation_records[0]
    source_count = int(layout["source_count"])
    result_count = int(layout["result_count"])
    baseline_length = int(baseline["decoder"]["prefill"]["query_length"])
    compressed_length = int(compressed["decoder"]["prefill"]["query_length"])
    representatives = tuple(
        int(item) for item in latest_splice["representative_indices"]
    )
    source_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["source_visual_mrope"]
    )
    result_mrope = tuple(
        tuple(int(value) for value in row)
        for row in latest_splice["result_visual_mrope"]
    )
    expected_result_mrope = tuple(source_mrope[index] for index in representatives)
    splice_layout = latest_splice["layout_delta"]
    decode_record = latest_generation.get("last_decode_record")
    native_spans = _spans(source_placeholder_spans)
    recorded_source_spans = _spans(latest_splice["source_placeholder_spans"])
    result_spans = _spans(latest_splice["result_placeholder_spans"])
    deepstack_shapes = tuple(
        tuple(int(item) for item in shape)
        for shape in latest_splice["deepstack_result_shapes"]
    )
    checks = {
        "compression_applied": bool(latest_splice["compression_applied"]),
        "non_identity_token_reduction": source_count > result_count,
        "canonical_fixed_k_result": result_count == 32,
        "one_native_video": native_video_count == 1,
        "native_placeholder_spans_preserved": (
            recorded_source_spans == native_spans
            and _span_count(recorded_source_spans) == source_count
        ),
        "result_placeholder_spans_agree": _span_count(result_spans) == result_count,
        "grid_and_source_counts_agree": (
            source_grid_token_count == source_count
            and int(latest_splice["source_placeholder_count"]) == source_count
        ),
        "result_placeholder_count_agrees": (
            int(latest_splice["result_placeholder_count"]) == result_count
        ),
        "layout_records_agree": (
            int(splice_layout["source_count"]) == source_count
            and int(splice_layout["result_count"]) == result_count
        ),
        "decoder_length_reduction_exact": (
            baseline_length - compressed_length == source_count - result_count
        ),
        "source_video_grid_preserved": (
            bool(latest_splice["source_video_grid_preserved"])
            and tuple(latest_splice["source_video_grid_thw"])
            == tuple(int(item) for item in source_grid_thw)
        ),
        "sparse_grid_policy": (
            latest_splice["result_grid_policy"] == "irregular_sparse_source_anchors"
        ),
        "representatives_source_ordered": (
            len(representatives) == result_count
            and representatives == tuple(sorted(representatives))
            and len(set(representatives)) == result_count
        ),
        "mrope_source_count_aligned": len(source_mrope) == source_count,
        "mrope_result_count_aligned": len(result_mrope) == result_count,
        "mrope_sparse_anchor_gather_exact": result_mrope == expected_result_mrope,
        "visual_pos_masks_compacted": bool(latest_splice["visual_pos_masks_compacted"]),
        "deepstack_branch_count_exact": (
            int(latest_splice["deepstack_branch_count"])
            == expected_deepstack_branches
            == len(deepstack_shapes)
        ),
        "deepstack_result_counts_exact": all(
            len(shape) == 2 and shape[0] == result_count for shape in deepstack_shapes
        ),
        "main_deepstack_layout_shared": bool(
            latest_splice["shared_main_deepstack_layout"]
        ),
        "generation_prefill_complete": bool(latest_generation["prefill_complete"]),
        "generation_source_length_aligned": (
            int(latest_generation["source_sequence_length"]) == baseline_length
        ),
        "generation_result_length_aligned": (
            int(latest_generation["result_sequence_length"]) == compressed_length
        ),
        "generation_removed_count_aligned": (
            int(latest_generation["removed_count"]) == source_count - result_count
        ),
        "decode_repair_exercised": (
            int(latest_generation["decode_repairs"]) >= 1
            and isinstance(decode_record, Mapping)
        ),
        "inference_labels_absent": labels_absent,
        "baseline_logits_finite": bool(baseline["logits_finite"]),
        "compressed_logits_finite": bool(compressed["logits_finite"]),
        "baseline_decoder_alignment": all(baseline["decoder"]["invariants"].values()),
        "compressed_decoder_alignment": all(
            compressed["decoder"]["invariants"].values()
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"VQToken Qwen3-VL sample evidence failed: {checks!r}")
    return checks


def _conversation_for_args(
    *,
    prompt: str,
    video: Path,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    return _conversation(
        prompt=prompt,
        video=video,
        fps=args.video_fps,
        max_pixels=args.video_max_pixels,
    )


def _preprocessing_record(
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    return {
        "fps": float(args.video_fps),
        "max_pixels": int(args.video_max_pixels),
        "image_patch_size": 16,
        "enable_thinking": False,
        "provider": "qwen_vl_utils.process_vision_info",
    }


def _vqtoken_profile() -> Qwen3VideoDiagnosticProfile:
    return Qwen3VideoDiagnosticProfile(
        algorithm="vqtoken",
        variant="vqtoken.video",
        binding="vqtoken.video.qwen3_vl",
        canonical_config=CANONICAL_VQTOKEN_CONFIG,
        stage_factory=_stage,
        conversation_factory=_conversation_for_args,
        sample_validator=_validate_vqtoken_sample,
        preprocessing_record=_preprocessing_record,
        projector_label="VQToken",
    )


def _profile_generation_report(
    *,
    facade: Any,
    runtime: BackboneRuntime,
    state: Any,
    batch: Any,
    first_layer_audit: Any,
    post_boundary_audits: Mapping[int, Any],
    torch: Any,
    max_new_tokens: int,
    min_new_tokens: int = 2,
) -> dict[str, Any]:
    post_boundary_marks = {
        layer: audit.mark() for layer, audit in post_boundary_audits.items()
    }
    report = _generation_report(
        facade=facade,
        runtime=runtime,
        state=state,
        batch=batch,
        audit=first_layer_audit,
        torch=torch,
        max_new_tokens=max_new_tokens,
        min_new_tokens=min_new_tokens,
    )
    decoder_boundaries = {
        str(layer): _common._decoder_summary(audit.since(post_boundary_marks[layer]))
        for layer, audit in post_boundary_audits.items()
    }
    if decoder_boundaries:
        report["decoder_boundaries"] = decoder_boundaries
    if len(decoder_boundaries) == 1:
        report["decoder_after_vtp_boundary"] = next(iter(decoder_boundaries.values()))
    return report


def _run(
    args: argparse.Namespace,
    *,
    profile: Qwen3VideoDiagnosticProfile | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    profile = _vqtoken_profile() if profile is None else profile
    if not 1 <= profile.min_new_tokens <= args.max_new_tokens:
        raise RuntimeError(
            "The profile minimum generation length exceeds max_new_tokens"
        )
    visible = prepare_checkpoint_environment(args.physical_gpu)

    checkpoint = args.checkpoint.resolve()
    parquet_path = args.nextqa_parquet.resolve()
    video_root = args.video_root.resolve()

    checkpoint_info = _checkpoint_record(checkpoint)
    auxiliary_checkpoint_info: dict[str, Any] = {}
    auxiliary_checkpoint_paths: dict[str, Path] = {}
    runtime_metadata = {"enable_talker": False, "speech_sample_rate": None}
    for argument_name, runtime_metadata_key in profile.auxiliary_checkpoints:
        value = getattr(args, argument_name, None)
        if value is None:
            raise RuntimeError(
                f"Qwen3-VL diagnostic requires --{argument_name.replace('_', '-')}"
            )
        auxiliary_path = value.resolve()
        auxiliary_checkpoint_info[argument_name] = _checkpoint_record(auxiliary_path)
        auxiliary_checkpoint_paths[argument_name] = auxiliary_path
        runtime_metadata[runtime_metadata_key] = str(auxiliary_path)
    documents, videos = generation_documents(parquet_path, video_root)

    torch, transformers, gpu_before, admission = admit_generation_device(
        args,
        bridge="Qwen3-VL",
    )
    if profile.runtime_metadata_factory is not None:
        runtime_metadata.update(profile.runtime_metadata_factory(torch))

    started = time.time()
    facade = create_qwen3_vl_backbone()
    processor, model, runtime = load_generation_runtime(
        checkpoint,
        facade=facade,
        torch=torch,
        dtype=torch.bfloat16,
        device_map={"": 0},
        metadata=runtime_metadata,
    )
    handles = facade.model_adapter.structural_handles(model)
    first_layer = handles.get("llm.layer[0]")
    if first_layer is None:
        raise RuntimeError("Qwen3-VL structural handle llm.layer[0] is missing")
    requested_boundary_layers = tuple(
        dict.fromkeys(
            (
                *(
                    ()
                    if profile.post_boundary_layer is None
                    else (profile.post_boundary_layer,)
                ),
                *profile.post_boundary_layers,
            )
        )
    )
    post_boundary_handles = {
        layer: handles.get(f"llm.layer[{layer}]") for layer in requested_boundary_layers
    }
    missing_boundary_layers = tuple(
        layer for layer, handle in post_boundary_handles.items() if handle is None
    )
    if missing_boundary_layers:
        raise RuntimeError(
            "Qwen3-VL structural handles do not cover requested post-boundary "
            f"layers {missing_boundary_layers!r}"
        )
    deepstack_handles = sorted(
        name for name in handles if name.startswith("vision_deepstack_projector[")
    )
    if not deepstack_handles:
        raise RuntimeError("Qwen3-VL DeepStack structural handles are missing")
    first_layer_audit = _DecoderInputAudit(first_layer)
    post_boundary_audits = {
        layer: _DecoderInputAudit(handle)
        for layer, handle in post_boundary_handles.items()
    }
    activation = generation_activation(
        facade=facade,
        runtime=runtime,
        binding_name=profile.binding,
        stage=profile.stage_factory(),
    )
    if dict(activation.runtime_metadata()["canonical_config"]) != dict(
        profile.canonical_config
    ):
        raise RuntimeError(
            f"Resolved {profile.projector_label} configuration differs from the "
            "canonical profile"
        )

    cases: dict[str, Any] = {}
    sample_rows: list[dict[str, Any]] = []
    try:
        for document in documents:
            doc_id = int(document["doc_id"])
            video = videos[document["video"]]
            conversation = profile.conversation_factory(
                prompt=document["prompt"],
                video=video,
                args=args,
            )
            sample = Qwen3VLSample(
                sample_id=f"nextqa_mc_test:{doc_id}",
                conversation=conversation,
                enable_thinking=False,
            )
            batch, state = prepare_generation_state(
                facade=facade,
                runtime=runtime,
                processor=processor,
                sample=sample,
                bridge="Qwen3-VL",
            )
            video_spans = [
                span for span in state.modality_token_spans if span.modality == "video"
            ]
            source_placeholder_spans = tuple(
                (int(span.start), int(span.end)) for span in video_spans
            )
            grid_rows = _tensor_rows(state.video_grid_thw)
            if len(grid_rows) != 1 or len(grid_rows[0]) != 3:
                raise RuntimeError(
                    f"Expected one video_grid_thw row, got {grid_rows!r}"
                )
            source_grid_thw = grid_rows[0]
            merge_size = int(batch.metadata["spatial_merge_size"])
            source_grid_token_count = (
                source_grid_thw[0]
                * source_grid_thw[1]
                * source_grid_thw[2]
                // (merge_size**2)
            )
            if _span_count(source_placeholder_spans) != source_grid_token_count:
                raise RuntimeError(
                    "Qwen3-VL video spans differ from video_grid_thw token count: "
                    f"spans={source_placeholder_spans!r}, "
                    f"grid={source_grid_token_count}"
                )
            native_video_count = int(batch.metadata["video_count"])
            labels_absent = state.labels is None and (
                "labels" not in state.native_model_inputs
                or state.native_model_inputs["labels"] is None
            )
            timing = state.native_model_inputs.get("second_per_grid_ts")
            timing_values = [] if timing is None else _float_values(timing)

            with compare_generation(
                activation=activation,
                generate=_profile_generation_report,
                facade=facade,
                runtime=runtime,
                state=state,
                batch=batch,
                first_layer_audit=first_layer_audit,
                post_boundary_audits=post_boundary_audits,
                torch=torch,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=profile.min_new_tokens,
            ) as (baseline, compressed, before_metadata, metadata):
                before_invocations = int(before_metadata["projector_invocations"])
                before_captures = int(
                    before_metadata.get("signal_provider", {}).get("capture_count", 0)
                )
                before_detector_calls = _detector_calls(before_metadata)
                if int(metadata["projector_invocations"]) != before_invocations + 1:
                    raise RuntimeError(
                        f"{profile.projector_label} projector invocation count did "
                        "not increment once"
                    )
                records = metadata["projector_layout_deltas"]
                algorithm_records = metadata.get("algorithm_records", ())
                secondary_records = metadata.get(profile.secondary_layout_key, ())
                cache_records = metadata.get("cache_repairs", ())
                provider_capture_delta = (
                    int(
                        metadata.get("signal_provider", {}).get(
                            "capture_count",
                            0,
                        )
                    )
                    - before_captures
                )
                detector_call_delta = _detector_calls(metadata) - before_detector_calls
                provider_metadata = metadata.get("signal_provider", {})
                detector_metadata = (
                    provider_metadata.get("detector", {})
                    if isinstance(provider_metadata, Mapping)
                    else {}
                )
                bridge_metadata = metadata["bridge"]
                latest_splice = bridge_metadata["latest_splice"]
                latest_generation = bridge_metadata["latest_generation"]
                latest_vtp = bridge_metadata.get("latest_vtp")
                latest_metok = bridge_metadata.get("latest_metok")
                if latest_splice is None or latest_generation is None:
                    raise RuntimeError(
                        "Qwen3-VL bridge published incomplete splice/generation evidence"
                    )
                checks = profile.sample_validator(
                    baseline=baseline,
                    compressed=compressed,
                    activation_records=records,
                    algorithm_records=algorithm_records,
                    vtp_records=secondary_records,
                    secondary_records=secondary_records,
                    cache_records=cache_records,
                    latest_splice=latest_splice,
                    latest_generation=latest_generation,
                    latest_vtp=latest_vtp,
                    latest_metok=latest_metok,
                    source_grid_thw=source_grid_thw,
                    source_grid_token_count=source_grid_token_count,
                    spatial_merge_size=merge_size,
                    labels_absent=labels_absent,
                    source_placeholder_spans=source_placeholder_spans,
                    native_video_count=native_video_count,
                    expected_deepstack_branches=len(deepstack_handles),
                    provider_capture_delta=provider_capture_delta,
                    provider_metadata=provider_metadata,
                    activation_metadata=metadata,
                    auxiliary_checkpoint_paths=auxiliary_checkpoint_paths,
                    detector_call_delta=detector_call_delta,
                    detector_metadata=detector_metadata,
                    detector_registered_as_trainable_child=bool(
                        metadata.get("detector_registered_as_trainable_child", False)
                    ),
                    second_per_grid_seconds=(
                        None if len(timing_values) != 1 else timing_values[0]
                    ),
                )
                case_activation = {
                    **metadata,
                    "projector_layout_deltas": records,
                    "algorithm_records": algorithm_records,
                    profile.secondary_layout_key: secondary_records,
                    "cache_repairs": cache_records,
                }

            case = {
                **generation_case(document, video),
                "native_video": {
                    "conversation": conversation,
                    "requested_fps": float(args.video_fps),
                    "requested_max_pixels": int(args.video_max_pixels),
                    **(
                        {}
                        if not hasattr(args, "video_min_pixels")
                        else {
                            "requested_min_pixels": int(args.video_min_pixels),
                        }
                    ),
                    "enable_thinking": False,
                    "video_grid_thw": source_grid_thw,
                    "second_per_grid_ts": timing_values,
                    "spatial_merge_size": merge_size,
                    "source_grid_token_count": source_grid_token_count,
                    "video_placeholder_spans": source_placeholder_spans,
                    "pixel_values_videos_shape": _shape(state.video_pixel_values),
                    "processor_output_contract": batch.metadata[
                        "processor_output_contract"
                    ],
                    "video_metadata_present": batch.metadata["video_metadata_present"],
                    "labels_absent_for_inference": labels_absent,
                },
                "baseline": baseline,
                "compressed": compressed,
                "activation": case_activation,
                "bridge_latest_splice": latest_splice,
                "bridge_latest_vtp": latest_vtp,
                "bridge_latest_metok": latest_metok,
                "bridge_latest_generation": latest_generation,
                "checks": checks,
            }
            cases[str(doc_id)] = case
            sample_rows.append(
                {
                    **generation_sample(document, baseline, compressed),
                    "compression_applied": checks["compression_applied"],
                    "source_visual_tokens": records[0]["source_count"],
                    "result_visual_tokens": records[0]["result_count"],
                }
            )
    finally:
        if activation.installed:
            activation.uninstall()
        for post_boundary_audit in post_boundary_audits.values():
            post_boundary_audit.close()
        first_layer_audit.close()

    torch.cuda.synchronize()
    gpu_after = _gpu_snapshot(args.physical_gpu)
    report = {
        "status": "passed",
        "scope": {
            "algorithm": profile.algorithm,
            "variant": profile.variant,
            "backbone": "qwen3_vl",
            "binding": profile.binding,
            "execution_phase": "generate",
            "canonical_config": dict(profile.canonical_config),
            "quality_claim": "functional_migration_only",
        },
        "provenance": {
            "code_revision": args.code_revision,
            **(
                {}
                if profile.reference_revision is None
                else {"reference_revision": profile.reference_revision}
            ),
            "command": shlex.join(sys.argv),
            "checkpoint": checkpoint_info,
            **(
                {}
                if not auxiliary_checkpoint_info
                else {"auxiliary_checkpoints": auxiliary_checkpoint_info}
            ),
            "nextqa_parquet": str(parquet_path),
            "nextqa_parquet_size_bytes": parquet_path.stat().st_size,
            "fixed_doc_ids": list(FIXED_NEXTQA_DOC_IDS),
            "video_preprocessing": dict(profile.preprocessing_record(args)),
        },
        "environment": {
            **generation_environment(
                visible=visible,
                torch=torch,
                transformers=transformers,
                dtype="bfloat16",
            ),
            "attention_implementation": "sdpa",
        },
        "gpu": checkpoint_gpu_report(
            before=gpu_before,
            admission=admission,
            after=gpu_after,
            torch=torch,
        ),
        "structural_handles": sorted(handles),
        "deepstack_projector_handles": deepstack_handles,
        "samples": cases,
        "sample_count": len(cases),
        "elapsed_seconds": round(time.time() - started, 3),
        "support_activation_claimed": False,
        "boundaries": {
            "full_nextqa_benchmark": False,
            "paper_quality_or_efficiency_reproduction": False,
            "convergence_claim": False,
        },
    }
    if tuple(int(item) for item in cases) != FIXED_NEXTQA_DOC_IDS:
        raise RuntimeError("The completed sample set differs from the fixed manifest")
    return report, sample_rows


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    add_checkpoint_arguments(parser, estimated_peak_mib=40960)
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--video_fps", type=float, default=QWEN3_VIDEO_FPS)
    parser.add_argument("--video_max_pixels", type=int, default=QWEN3_VIDEO_MAX_PIXELS)
    return parser


def _execute(
    args: argparse.Namespace,
    *,
    profile: Qwen3VideoDiagnosticProfile | None = None,
) -> int:
    if args.max_new_tokens < 2:
        raise RuntimeError("max_new_tokens must be at least 2")
    if args.video_fps <= 0 or args.video_max_pixels <= 0:
        raise RuntimeError("video_fps and video_max_pixels must be positive")
    if hasattr(args, "video_min_pixels"):
        if args.video_min_pixels <= 0:
            raise RuntimeError("video_min_pixels must be positive")
        if args.video_min_pixels > args.video_max_pixels:
            raise RuntimeError("video_min_pixels cannot exceed video_max_pixels")
    return inference_main(args, partial(_run, profile=profile))


def main(argv: list[str] | None = None) -> int:
    return _execute(_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
