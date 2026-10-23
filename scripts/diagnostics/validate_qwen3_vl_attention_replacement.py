"""Run one real-image Qwen3-VL operation-replacement generation smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from mllm_comp.backbones.base import BackboneRuntime
from mllm_comp.backbones.qwen3_vl import (
    DEFAULT_SYSTEM_PROMPT,
    Qwen3VLSample,
    create_qwen3_vl_backbone,
)
from mllm_comp.config.schema import experiment_config_from_dict
from mllm_comp.pipeline.runtime_compression import activate_runtime_compression
from mllm_comp.core.vocabulary import ExecutionPhase
from qwen3_vl_operator_reference import capture_reference, finalize_reference


def _conversation(image_path: Path) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": DEFAULT_SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": "What is the dominant color?"},
            ],
        },
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--strategy",
        choices=(
            "attn_linear_attention",
            "attn_cnn",
            "attn_identity",
            "attn_lolcats_linear_window",
            "attn_nha",
            "attn_sparse_window",
            "ffn_low_rank",
            "ffn_identity",
            "ffn_two_layer_mlp",
        ),
        required=True,
    )
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--kernel_size", type=int, default=3)
    parser.add_argument("--eps", type=float, default=1.0e-6)
    parser.add_argument("--lolcats_eps", type=float, default=1.0e-12)
    parser.add_argument("--window_size", type=int, default=64)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--gate_logit_normalizer", type=float, default=8.0)
    parser.add_argument("--nha_eps", type=float, default=1.0e-6)
    parser.add_argument("--decode_window_size", type=int)
    parser.add_argument("--feature_dim", type=int, default=64)
    parser.add_argument("--feature_map", default="hedgehog_softmax_dim")
    parser.add_argument("--tie_qk_feature_maps", action="store_true")
    parser.add_argument("--window_weight_init", type=float, default=0.1)
    parser.add_argument("--freeze_window_weight", action="store_true")
    parser.add_argument("--affine_attention_factors", action="store_true")
    parser.add_argument("--rank", type=int, default=256)
    parser.add_argument("--bottleneck_dim", type=int, default=1024)
    parser.add_argument("--bias", action="store_true")
    parser.add_argument("--initialization", default="column_sampling")
    parser.add_argument("--local_left", type=int, default=127)
    parser.add_argument("--local_right", type=int, default=0)
    parser.add_argument("--prefix_size", type=int, default=0)
    parser.add_argument("--prompt_suffix_size", type=int, default=0)
    parser.add_argument("--fixed_positions", default="")
    parser.add_argument("--stride", type=int)
    parser.add_argument("--non_causal", action="store_true")
    parser.add_argument("--exclude_self", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max_new_tokens", type=int, default=8)
    parser.add_argument("--report_path", required=True)
    parser.add_argument(
        "--trace_path",
        help="Save the first complete replacement input/output tensors to this .pt file.",
    )
    parser.add_argument(
        "--trace_only",
        action="store_true",
        help="Capture tensors without requiring a non-empty generated answer.",
    )
    args = parser.parse_args(argv)
    if args.layer < 0:
        raise ValueError("--layer must be non-negative")

    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for the real Qwen3-VL smoke.")
    model_path = Path(args.model_path).resolve()
    if not model_path.is_dir():
        raise RuntimeError(f"Qwen3-VL checkpoint does not exist: {model_path}")

    started = time.time()
    facade = create_qwen3_vl_backbone()
    processor = facade.processor_adapter.load_processor(
        str(model_path),
        local_files_only=True,
    )
    model = facade.model_adapter.load_model(
        str(model_path),
        train=False,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map={"": 0},
    )
    target_kind = "ffn" if args.strategy.startswith("ffn_") else "attention"
    target_suffix = "ffn" if target_kind == "ffn" else "attn"
    target = f"llm.layer[{args.layer}].{target_suffix}"
    private: dict[str, Any] = {"strategy": args.strategy, "debug": True}
    if args.strategy == "attn_linear_attention":
        private["eps"] = args.eps
    elif args.strategy == "attn_lolcats_linear_window":
        private.update(
            {
                "window_size": args.window_size,
                "decode_window_size": (
                    args.window_size
                    if args.decode_window_size is None
                    else args.decode_window_size
                ),
                "feature_dim": args.feature_dim,
                "feature_map": args.feature_map,
                "tie_qk_feature_maps": args.tie_qk_feature_maps,
                "window_weight_init": args.window_weight_init,
                "train_window_weight": not args.freeze_window_weight,
                "affine_attention_factors": args.affine_attention_factors,
                "eps": args.lolcats_eps,
            }
        )
    elif args.strategy == "attn_cnn":
        private["kernel_size"] = args.kernel_size
    elif args.strategy == "attn_nha":
        private.update(
            {
                "num_slots": args.num_slots,
                "window_size": args.window_size,
                "gate_logit_normalizer": args.gate_logit_normalizer,
                "eps": args.nha_eps,
            }
        )
    elif args.strategy == "ffn_low_rank":
        private["rank"] = args.rank
        private["initialization"] = args.initialization
    elif args.strategy == "ffn_two_layer_mlp":
        private["bottleneck_dim"] = args.bottleneck_dim
        private["bias"] = args.bias
    elif args.strategy == "attn_sparse_window":
        private.update(
            {
                "causal": not args.non_causal,
                "include_self": not args.exclude_self,
                "local_left": args.local_left,
                "local_right": args.local_right,
                "prefix_size": args.prefix_size,
                "prompt_suffix_size": args.prompt_suffix_size,
                "fixed_positions": [
                    int(item)
                    for item in args.fixed_positions.split(",")
                    if item.strip()
                ],
                "stride": args.stride,
            }
        )
    plugin_name = (
        "qwen3_vl_ffn_replacement"
        if target_kind == "ffn"
        else "qwen3_vl_attention_replacement"
    )
    runtime = BackboneRuntime(
        model=model,
        processor=processor,
        model_path_or_id=str(model_path),
        device=torch.device(args.device),
        dtype=torch.bfloat16,
        metadata={"enable_talker": False},
    )
    experiment = experiment_config_from_dict({
        "backbone": {"name": "qwen3_vl"},
        "pipeline": {"stages": [{
            "name": f"replace_layer_{args.layer}_{target_kind}_with_{args.strategy}",
            "algorithm": plugin_name,
            "operation": {"type": "replace"},
            "criterion": {"score_source": "manual"},
            "schedule": {"targets": [{"site": target}]},
            "config": private,
        }]},
    })
    session = activate_runtime_compression(
        config=experiment, facade=facade, runtime=runtime,
        compression_mode="compressed", execution_phase=ExecutionPhase.GENERATE,
    )
    activation = session.active_records[0]
    replacement = facade.model_adapter.structural_handles(model)[target]
    if getattr(replacement, "strategy_name", None) != args.strategy:
        raise RuntimeError(f"Replacement was not installed at {target}.")

    trace: dict[str, Any] = {}

    def capture_input(
        module: Any,
        positional: tuple[Any, ...],
        keyword: dict[str, Any],
    ) -> None:
        if trace:
            return
        trace.update(
            capture_reference(
                module,
                positional,
                keyword,
                strategy=args.strategy,
                target=target,
            )
        )

    def capture_output(
        _module: Any,
        _positional: tuple[Any, ...],
        _keyword: dict[str, Any],
        result: Any,
    ) -> None:
        if "actual_output" in trace:
            return
        output = result[0] if isinstance(result, tuple) else result
        if not torch.is_tensor(output):
            raise RuntimeError("Could not capture the replacement output tensor.")
        finalize_reference(trace, output)
        trace["first_forward_debug"] = dict(getattr(_module, "last_debug", {}))

    pre_hook = replacement.register_forward_pre_hook(capture_input, with_kwargs=True)
    post_hook = replacement.register_forward_hook(capture_output, with_kwargs=True)

    with tempfile.TemporaryDirectory(prefix="qwen3-vl-replacement-") as temp_dir:
        image_path = Path(temp_dir) / "red.png"
        Image.new("RGB", (256, 256), color=(192, 64, 32)).save(image_path)
        sample = Qwen3VLSample(
            sample_id=f"{args.strategy}-image-smoke",
            conversation=_conversation(image_path),
            enable_thinking=False,
        )
        batch = facade.processor_adapter.prepare_inputs(processor, sample, phase="eval")
        state = facade.state_builder.build(batch, phase="eval")
        result = facade.runner.generate(
            runtime,
            state,
            processed_batch=batch,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    pre_hook.remove()
    post_hook.remove()
    if replacement.forward_calls < 1:
        raise RuntimeError("The installed replacement forward method was never called.")
    if (
        "input" not in trace
        or "actual_output" not in trace
        or "expected_output" not in trace
    ):
        raise RuntimeError(
            "The complete replacement validation trace was not captured."
        )

    if args.strategy == "attn_lolcats_linear_window":
        runtime_tensors: dict[str, torch.Tensor] = {}
        for attribute in (
            "_runtime_k_state",
            "_runtime_kv_state",
            "_runtime_window_keys",
            "_runtime_window_values",
            "_runtime_window_features",
            "_runtime_window_keep",
        ):
            value = getattr(replacement, attribute, None)
            if torch.is_tensor(value):
                runtime_tensors[attribute.removeprefix("_runtime_")] = (
                    value.detach().cpu().clone()
                )
        trace["lolcats_runtime_state_after_generation"] = {
            "seen_tokens": getattr(replacement, "_runtime_seen_tokens", None),
            "window_size": getattr(replacement, "_runtime_window_size", None),
            "tensors": runtime_tensors,
        }
    if args.strategy == "attn_nha":
        runtime_tensors = {}
        for attribute in (
            "_runtime_long_keys",
            "_runtime_long_values",
            "_runtime_window_keys",
            "_runtime_window_rope_keys",
            "_runtime_window_values",
            "_runtime_window_retentions",
            "_runtime_window_keep",
        ):
            value = getattr(replacement, attribute, None)
            if torch.is_tensor(value):
                runtime_tensors[attribute.removeprefix("_runtime_")] = (
                    value.detach().cpu().clone()
                )
        trace["nha_runtime_state_after_generation"] = {
            "seen_tokens": getattr(replacement, "_runtime_seen_tokens", None),
            "num_slots": getattr(replacement, "num_slots", None),
            "window_size": getattr(replacement, "window_size", None),
            "tensors": runtime_tensors,
        }

    report_path = Path(args.report_path)
    trace_path = (
        Path(args.trace_path)
        if args.trace_path
        else report_path.with_suffix(".trace.pt")
    )
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(trace, trace_path)

    lolcats_components = trace.get("lolcats_reference_components", {})
    lolcats_runtime = trace.get("lolcats_runtime_state_after_generation", {})
    lolcats_runtime_tensors = lolcats_runtime.get("tensors", {})
    nha_components = trace.get("nha_reference_components", {})
    nha_runtime = trace.get("nha_runtime_state_after_generation", {})
    nha_runtime_tensors = nha_runtime.get("tensors", {})

    generated_text = result.text.text if result.text is not None else []
    if not args.trace_only and (not generated_text or not generated_text[0].strip()):
        raise RuntimeError("Replacement smoke produced no text.")

    report = {
        "status": "validated" if trace["validation"]["allclose"] else "failed",
        "strategy": args.strategy,
        "target": target,
        "replacement_class": type(replacement).__qualname__,
        "forward_calls": replacement.forward_calls,
        "last_debug": replacement.last_debug,
        "first_forward_debug": trace.get("first_forward_debug", {}),
        "parameter_compression_ratio": getattr(
            replacement,
            "parameter_compression_ratio",
            None,
        ),
        "stage_stats": activation.runtime_metadata()["stats"],
        "text": generated_text,
        "tensor_trace": {
            "path": str(trace_path.resolve()),
            "capture_scope": "first_forward_prefill",
            "input_shape": list(trace["input"].shape),
            "actual_output_shape": list(trace["actual_output"].shape),
            "expected_output_shape": list(trace["expected_output"].shape),
            "input_dtype": str(trace["input"].dtype),
            "actual_output_dtype": str(trace["actual_output"].dtype),
            "expected_output_dtype": str(trace["expected_output"].dtype),
            "size_bytes": trace_path.stat().st_size,
            "state_dict_tensor_count": len(trace["operator_state_dict"]),
            "lolcats_reference_component_names": sorted(lolcats_components),
            "lolcats_runtime_seen_tokens": lolcats_runtime.get("seen_tokens"),
            "lolcats_runtime_window_size": lolcats_runtime.get("window_size"),
            "lolcats_runtime_state_tensor_shapes": {
                name: list(tensor.shape)
                for name, tensor in lolcats_runtime_tensors.items()
            },
            "nha_reference_component_names": sorted(nha_components),
            "nha_runtime_seen_tokens": nha_runtime.get("seen_tokens"),
            "nha_runtime_num_slots": nha_runtime.get("num_slots"),
            "nha_runtime_window_size": nha_runtime.get("window_size"),
            "nha_runtime_state_tensor_shapes": {
                name: list(tensor.shape) for name, tensor in nha_runtime_tensors.items()
            },
            "validation": trace["validation"],
        },
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(0) / (1024**3), 3),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    report_path.write_text(payload, encoding="utf-8")
    print(payload)
    if not trace["validation"]["allclose"]:
        raise RuntimeError(
            "Replacement output did not match the independent reference calculation: "
            f"strategy={args.strategy}, "
            f"max_abs_diff={trace['validation']['max_abs_diff']}."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
