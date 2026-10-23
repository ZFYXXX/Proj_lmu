"""Three-step LongVU/LLaMA-VID SFT correctness and compressor-state reload.

Uses the established native training facades and selected-parameter checkpoint
route, plus each provider's strict compression_state_dict API. All assets are
local. The single NExT-QA test-derived fixture is not training-quality evidence.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import argparse
import gc
import importlib
import json
import math
import os
import shlex
import sys
import time

from mllm_comp.compressors.multimodal_registry import (
    builtin_multimodal_compression_contracts,
)
from mllm_comp.compressors.video.config import (
    CANONICAL_VIDEO_CONFIGS,
    VIDEO_REFERENCE_REVISION,
)
from mllm_comp.core.vocabulary import ExecutionPhase
from mllm_comp.pipeline.stage import CompressionStageConfig

from scripts.diagnostics import (
    validate_qwen25_vl_llamavid_video_compression_checkpoint as llama_checks,
    video_checkpoint_common as base,
    video_training_checkpoint_common as checkpoint_io,
)


OPTIMIZER_STEPS = 3
BACKBONES = ("llava_onevision", "qwen25_vl", "qwen3_vl")
ALGORITHMS = ("longvu", "llamavid")


def _native(backbone):
    names = {
        "llava_onevision": "llava_dycoke",
        "qwen25_vl": "qwen25_dycoke",
        "qwen3_vl": "qwen3_vqtoken",
    }
    return importlib.import_module(
        f"scripts.diagnostics.validate_{names[backbone]}_video_training_checkpoint"
    )


def _sample(args, document, video):
    if args.backbone == "llava_onevision":
        from mllm_comp.backbones.llava_onevision import LlavaOneVisionSample

        return LlavaOneVisionSample(
            sample_id="nextqa_mc_test:442:sft",
            prompt=document["prompt"],
            video=str(video),
        )
    patch = 28 if args.backbone == "qwen25_vl" else 32
    if args.backbone == "qwen25_vl":
        from mllm_comp.backbones.qwen25_vl import Qwen25VLSample as Sample
        from scripts.diagnostics import (
            validate_qwen25_vl_video_compression_checkpoint as native,
        )
    else:
        from mllm_comp.backbones.qwen3_vl import Qwen3VLSample as Sample
        from scripts.diagnostics import (
            validate_qwen3_vl_video_compression_checkpoint as native,
        )
    conversation = native._conversation(
        prompt=document["prompt"],
        video=video,
        fps=1.0,
        max_pixels=(320 if args.algorithm == "longvu" else 128) * patch**2,
    )
    if args.algorithm == "longvu":
        conversation[1]["content"][0]["min_pixels"] = 256 * patch**2
    return Sample(sample_id="nextqa_mc_test:442:sft", conversation=conversation)


def _scope(args):
    return {
        "algorithm": args.algorithm,
        "variant": f"{args.algorithm}.video",
        "backbone": args.backbone,
        "binding": f"{args.algorithm}.video.{args.backbone}",
    }


def _stage(args, phase):
    scope = _scope(args)
    return CompressionStageConfig.from_dict(
        {
            **{key: value for key, value in scope.items() if key != "backbone"},
            "name": f"{args.algorithm}-video-three-step",
            "placement": "projector_level",
            "execution_phase": phase.value,
            "plugin": args.algorithm,
            "target_modalities": ["video"],
            "conditioning_modalities": ["text"],
            "preserved_modalities": ["text", "image", "audio"],
            "coupling_policy": "independent",
            "media_alignment": "spatiotemporal",
            "config": {},
        }
    )


def _activation(args, facade, model, runtime, phase):
    scope = _scope(args)
    runtime.metadata[
        "longvu_dino_path" if args.algorithm == "longvu" else "llamavid_bert_path"
    ] = str(args.auxiliary_checkpoint)
    stage = _stage(args, phase)
    binding = builtin_multimodal_compression_contracts().binding(scope["binding"])
    activation = binding.executor_factory(
        model=model,
        runtime=runtime,
        facade=facade,
        stage=stage,
        binding=binding,
        binding_config=SimpleNamespace(config={}),
        execution_phase=phase,
    )
    activation.install()
    return activation


def _algorithm_checks(args, record, temporal, per_frame):
    source, result = int(record["source_count"]), int(record["result_count"])
    checks = {
        "canonical_config_exact": dict(record["canonical_config"])
        == dict(CANONICAL_VIDEO_CONFIGS[args.algorithm]),
        "source_count_exact": source == temporal * per_frame,
        "non_identity_reduction": source > result > 0,
        "finite_compressor_signals": all(
            record[k] is True
            for k in (
                "projected_features_finite",
                "raw_features_finite",
                "result_features_finite",
            )
        ),
    }
    if args.algorithm == "longvu":
        selected = record["selected_frame_indices_by_video"][0]
        checks.update(
            {
                "dino_finite_frozen": record["dinov2_features_finite"] is True
                and record["dino_frozen"] is True,
                "sva_query_count_exact": result == len(selected) * 144,
            }
        )
    else:
        if args.backbone == "llava_onevision":
            pass
        else:
            pass
        checks.update(
            {
                "two_tokens_per_unit": result == 2 * temporal,
                "teacher_forced_answer_excluded": record["text_policy"]
                == "user_text_without_teacher_forced_answer"
                and record["text_context_present"] is True,
                "local_bert_tokens": record["bert_local_only"] is True
                and llama_checks.llama._bert_token_shapes_exact(
                    record["bert_token_shapes"]
                ),
            }
        )
    return checks


def _forward_evidence(
    args, *, metadata, before, state, batch, output, decoder, training, torch
):
    records = metadata["algorithm_records"]
    layouts = metadata["projector_layout_deltas"]
    if len(records) != 1 or len(layouts) != 1:
        raise RuntimeError("Expected one compressor call per forward")
    record, layout = records[0], layouts[0]
    splice = metadata["bridge"]["latest_splice"]
    logits = output.logits
    source, result = int(record["source_count"]), int(record["result_count"])
    checks = {
        "compression_once": metadata["projector_invocations"]
        == before["projector_invocations"] + 1,
        "provider_once": metadata["signal_provider"]["capture_count"]
        == before["signal_provider"]["capture_count"] + 1,
        "compression_applied": splice["compression_applied"] is True,
        "layout_counts_exact": int(layout["source_count"]) == source
        and int(layout["result_count"]) == result,
        "finite_logits": bool(torch.isfinite(logits).all().item()),
        "decoder_mask_position_cache_aligned": all(decoder["invariants"].values()),
        "logits_match_decoder_length": logits.shape[0] == 1
        and logits.shape[1] == decoder["prefill"]["query_length"],
        "training_cache_disabled": state.native_model_inputs.get("use_cache") is False,
    }
    labels = state.labels
    if training:
        checks["finite_loss"] = bool(torch.isfinite(output.loss).all().item())
        checks["completion_labels_present"] = labels is not None and bool(
            (labels != -100).any().item()
        )
        checks["source_video_labels_ignored"] = all(
            bool((labels[:, span.start : span.end] == -100).all().item())
            for span in state.modality_token_spans
            if span.modality == "video"
        )
    if args.backbone == "llava_onevision":
        temporal, per_frame = int(batch.metadata["video"]["frame_count"]), 196
        checks.update(
            {
                "splice_counts_exact": splice["source_patch_count"] == source
                and splice["result_patch_count"] == result,
                "structural_tokens_preserved": len(
                    splice["structural_source_positions"]
                )
                == len(splice["structural_result_positions"]),
                "label_logit_layout_exact": len(
                    splice["sequence_representative_indices"]
                )
                == logits.shape[1],
            }
        )
    else:
        grid = base._tensor_rows(state.video_grid_thw)[0]
        merge = int(batch.metadata["spatial_merge_size"])
        temporal, per_frame = int(grid[0]), int(grid[1] * grid[2] // merge**2)
        anchors = record["representative_indices"][0]
        source_mrope = tuple(map(tuple, splice["source_visual_mrope"]))
        checks.update(
            {
                "native_grid_preserved": splice["source_video_grid_preserved"] is True
                and tuple(splice["source_video_grid_thw"]) == tuple(grid),
                "placeholders_exact": splice["source_placeholder_count"] == source
                and splice["result_placeholder_count"] == result,
                "native_mrope_exact": tuple(map(tuple, splice["result_visual_mrope"]))
                == tuple(source_mrope[i] for i in anchors),
                "compact_sequence_length": logits.shape[1]
                == state.input_ids.shape[1] - source + result,
                "sparse_grid_policy": splice["result_grid_policy"]
                == "irregular_sparse_source_anchors",
            }
        )
        if args.backbone == "qwen3_vl":
            checks.update(
                {
                    "visual_masks_compacted": splice["visual_pos_masks_compacted"]
                    is True,
                    "deepstack_three_branches": splice["deepstack_branch_count"]
                    == record["deepstack_branch_count"]
                    == 3,
                    "main_deepstack_layout_shared": splice[
                        "shared_main_deepstack_layout"
                    ]
                    is True,
                    "deepstack_result_shapes": tuple(
                        map(tuple, record["deepstack_result_shapes"])
                    )
                    == tuple(map(tuple, splice["deepstack_result_shapes"]))
                    and all(s[0] == result for s in record["deepstack_result_shapes"]),
                }
            )
    checks.update(_algorithm_checks(args, record, temporal, per_frame))
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"Provider training evidence failed: {failed}")
    return {
        "algorithm": checkpoint_io.canonical(record),
        "compression": checkpoint_io.canonical(layout),
        "splice": checkpoint_io.canonical(splice),
        "decoder": decoder,
        "checks": checks,
    }


def _parameter_evidence(named, before, torch):
    records = []
    for name, parameter in named:
        grad = parameter.grad
        records.append(
            {
                "name": name,
                "numel": parameter.numel(),
                "gradient_present": grad is not None,
                "gradient_finite": grad is None
                or bool(torch.isfinite(grad).all().item()),
                "gradient_nonzero": grad is not None and bool((grad != 0).any().item()),
                "gradient_norm": (
                    0.0 if grad is None else float(grad.float().norm().item())
                ),
                "max_parameter_delta": float(
                    (parameter.detach() - before[name]).abs().max().float().item()
                ),
            }
        )
    return records


def parameter_checks(algorithm, records):
    prefixes = (
        (
            "vision_query",
            "raw_projector.",
            "dino_projector.",
            "layers.",
            "output_projector.",
        )
        if algorithm == "longvu"
        else (
            "query_tokens",
            "bert_layers.",
            "cross_attention.",
            "vlm_att_projector.",
            "vlm_att_val_projector.",
            "content_projector.",
        )
    )
    checks = {"all_gradients_finite": all(r["gradient_finite"] for r in records)}
    for prefix in prefixes:
        checks[f"updated_{prefix}"] = any(
            r["name"].startswith(prefix)
            and r["gradient_nonzero"]
            and math.isfinite(r["max_parameter_delta"])
            and r["max_parameter_delta"] > 0
            for r in records
        )
    return checks


def _train(args, document, video, torch):
    from safetensors.torch import save_file

    native = _native(args.backbone)
    facade, processor, model, runtime = native._load_runtime(
        args.checkpoint, torch=torch, train=True
    )
    # Freeze the pretrained base, then install the new trainable child before
    # optimizer construction. Its defining stages remain in the autograd graph.
    selected = checkpoint_io.select_final_norm_weight(facade, model)
    activation = _activation(args, facade, model, runtime, ExecutionPhase.FORWARD)
    module = (
        activation.fusion_module
        if args.algorithm == "longvu"
        else activation.qformer_module
    )
    module.requires_grad_(True)
    named = list(module.named_parameters())
    optimizer = torch.optim.AdamW(
        [p for _, p in selected + named], lr=args.learning_rate, weight_decay=0.0
    )
    sample = _sample(args, document, video)
    batch, state = native._training_state(
        facade=facade, processor=processor, sample=sample, target=document["target"]
    )
    audit = base._DecoderInputAudit(
        facade.model_adapter.structural_handles(model)["llm.layer[0]"]
    )
    steps = []
    try:
        model.train()
        for step in range(1, OPTIMIZER_STEPS + 1):
            optimizer.zero_grad(set_to_none=True)
            before_parameters = {name: p.detach().clone() for name, p in named}
            before_norm = {name: p.detach().clone() for name, p in selected}
            before = activation.runtime_metadata()
            mark = audit.mark()
            output = facade.runner.forward_train(runtime, state)
            if output.loss is None or not bool(
                torch.isfinite(output.loss).all().item()
            ):
                raise RuntimeError("Missing or non-finite loss")
            output.loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            metadata = activation.runtime_metadata()
            record = _forward_evidence(
                args,
                metadata=metadata,
                before=before,
                state=state,
                batch=batch,
                output=output,
                decoder=base._decoder_summary(audit.since(mark)),
                training=True,
                torch=torch,
            )
            parameters = _parameter_evidence(named, before_parameters, torch)
            norm = checkpoint_io.parameter_records(
                selected, before=before_norm, torch=torch
            )
            checks = {
                **parameter_checks(args.algorithm, parameters),
                **checkpoint_io.target_parameter_checks(norm),
            }
            record.update(
                step=step,
                loss=float(output.loss.detach().float().item()),
                trainable_parameters=norm,
                compressor_parameters=parameters,
            )
            record["checks"].update(checks)
            checkpoint_io.append_jsonl(args.output_dir / "steps.jsonl", record)
            if not all(checks.values()):
                raise RuntimeError(
                    f"Step {step} trainable stages failed: {[k for k,v in checks.items() if not v]}"
                )
            steps.append(record)
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": record["loss"],
                        "source": record["algorithm"]["source_count"],
                        "result": record["algorithm"]["result_count"],
                    }
                ),
                flush=True,
            )
            del output, before_parameters, before_norm
        saved = checkpoint_io.save_lightweight_checkpoint(
            args.output_dir / "checkpoint-3",
            selected=selected,
            scope=_scope(args),
            canonical_config=CANONICAL_VIDEO_CONFIGS[args.algorithm],
            base_checkpoint=args.checkpoint,
            code_revision=args.code_revision,
            optimizer={"name": "AdamW", "lr": args.learning_rate, "weight_decay": 0},
            optimizer_steps=3,
            torch=torch,
        )
        state_path = args.output_dir / "checkpoint-3" / "compression_model.safetensors"
        compression_state = {
            name: tensor.detach().cpu().contiguous().clone()
            for name, tensor in activation.compression_state_dict().items()
        }
        save_file(compression_state, str(state_path))
        saved["compression_state"] = {
            "path": str(state_path),
            "tensor_names": list(compression_state),
            "size_bytes": state_path.stat().st_size,
            "trainable_parameter_count": sum(p.numel() for _, p in named),
        }
        return steps, saved
    finally:
        audit.close()
        activation.uninstall()


def _reload(args, document, video, saved, torch):
    from safetensors.torch import load_file

    native = _native(args.backbone)
    facade, processor, model, runtime = native._load_runtime(
        args.checkpoint, torch=torch, train=False
    )
    base_reload = checkpoint_io.load_lightweight_checkpoint(model, saved, torch=torch)
    activation = _activation(args, facade, model, runtime, ExecutionPhase.GENERATE)
    tensors = load_file(saved["compression_state"]["path"], device="cpu")
    activation.load_compression_state_dict(tensors)
    loaded = activation.compression_state_dict()
    exact = set(loaded) == set(tensors) and all(
        torch.equal(value.detach().cpu(), tensors[name])
        for name, value in loaded.items()
    )
    if not exact:
        raise RuntimeError("Compressor state did not reload exactly")
    sample = _sample(args, document, video)
    batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
    state = facade.state_builder.build(batch, raw_batch=sample, phase="eval")
    state = replace(
        state,
        native_model_inputs={
            **state.native_model_inputs,
            "use_cache": False,
            "return_dict": True,
        },
    )
    audit = base._DecoderInputAudit(
        facade.model_adapter.structural_handles(model)["llm.layer[0]"]
    )
    try:
        model.eval()
        before = activation.runtime_metadata()
        with torch.inference_mode():
            output = facade.runner.forward_train(runtime, state)
        evidence = _forward_evidence(
            args,
            metadata=activation.runtime_metadata(),
            before=before,
            state=state,
            batch=batch,
            output=output,
            decoder=base._decoder_summary(audit.records),
            training=False,
            torch=torch,
        )
        evidence.update(
            base_reload=base_reload,
            compression_state_exact=exact,
            compression_tensor_count=len(tensors),
        )
        return evidence
    finally:
        audit.close()
        activation.uninstall()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--algorithm", choices=ALGORITHMS, required=True)
    parser.add_argument("--backbone", choices=BACKBONES, required=True)
    for name in (
        "checkpoint",
        "auxiliary_checkpoint",
        "nextqa_parquet",
        "video_root",
        "output_dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--code_revision", required=True)
    parser.add_argument("--physical_gpu", type=int, choices=(0, 1), required=True)
    parser.add_argument("--estimated_peak_mib", type=int, required=True)
    parser.add_argument("--safety_margin_mib", type=int, default=16384)
    parser.add_argument("--learning_rate", type=float, default=0.01)
    args = parser.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(args.physical_gpu):
        raise RuntimeError("Expose only the explicitly selected GPU")
    for name in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "TRANSFORMERS_OFFLINE"):
        os.environ[name] = "1"
    if (args.output_dir / "report.json").exists() or (
        args.output_dir / "steps.jsonl"
    ).exists():
        raise RuntimeError("Refusing to overwrite existing evidence")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    report = {
        "status": "failed",
        "scope": {**_scope(args), "execution_phase": "forward"},
        "code_revision": args.code_revision,
        "reference_revision": VIDEO_REFERENCE_REVISION,
        "command": shlex.join(sys.argv),
        "checkpoint": str(args.checkpoint),
        "auxiliary_checkpoint": str(args.auxiliary_checkpoint),
        "physical_gpu": args.physical_gpu,
        "optimizer_steps": 3,
        "batch_size": 1,
        "canonical_config": dict(CANONICAL_VIDEO_CONFIGS[args.algorithm]),
    }
    try:
        for path in (
            args.checkpoint,
            args.auxiliary_checkpoint,
            args.nextqa_parquet,
            args.video_root,
        ):
            if not path.exists():
                raise RuntimeError(f"Mandatory local resource missing: {path}")
        gpu = base._gpu_snapshot(args.physical_gpu)
        report["gpu"] = {
            "before": gpu,
            "admission": base._validate_admission(
                gpu,
                estimated_peak_mib=args.estimated_peak_mib,
                safety_margin_mib=args.safety_margin_mib,
            ),
            "concurrency_decision": "One owned job on this explicitly selected device; independent jobs may use the other device.",
        }
        import torch

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats()
        torch.use_deterministic_algorithms(True, warn_only=True)
        documents = base._load_fixed_documents(args.nextqa_parquet)
        document = next(row for row in documents if row["doc_id"] == 442)
        video = base._video_path(args.video_root, document["video"])
        fixture = {
            "kind": "nextqa_mc_test_derived_sft_code_correctness_fixture",
            "task": "nextqa_mc_test",
            "split": "test",
            "doc_id": 442,
            "video_id": document["video"],
            "prompt": document["prompt"],
            "target": document["target"],
            "video_path": str(video),
            "quality_or_training_data_claim": False,
        }
        checkpoint_io.atomic_json(args.output_dir / "fixture.json", fixture)
        report["fixture"] = fixture
        report["steps"], saved = _train(args, document, video, torch)
        report["checkpoint_saved"] = saved
        gc.collect()
        torch.cuda.empty_cache()
        report["reload"] = _reload(args, document, video, saved, torch)
        report["gpu"].update(
            observed_peak_allocated_mib=torch.cuda.max_memory_allocated() / 1024**2,
            observed_peak_reserved_mib=torch.cuda.max_memory_reserved() / 1024**2,
        )
        report.update(
            status="passed",
            support_activation_claimed=False,
            boundaries={
                "full_nextqa_benchmark": False,
                "paper_quality_or_efficiency_reproduction": False,
                "convergence_claim": False,
            },
        )
    except BaseException as exc:
        report.update(error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        report["elapsed_seconds"] = time.time() - started
        checkpoint_io.atomic_json(args.output_dir / "report.json", report)
    print(
        json.dumps(
            {"status": "passed", "report": str(args.output_dir / "report.json")}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
