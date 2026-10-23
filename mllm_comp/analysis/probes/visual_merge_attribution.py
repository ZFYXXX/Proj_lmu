"""Layer-zero visual merge attribution for Qwen2.5-VL."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import traceback
from typing import Any, Mapping, Sequence

from mllm_comp.analysis.bindings import resolve_analysis_binding, select_decoder_layers
from mllm_comp.analysis.common.config import (
    positive_integer,
    string_sequence,
    strict_boolean,
    validate_common_probe_config,
    validate_layer_selection,
)
from mllm_comp.analysis.common.sample import prepare_analysis_forward_state
from mllm_comp.analysis.common.streaming import JsonlRecordSpool
from mllm_comp.analysis.common.visual_merge_attribution import (
    ALL_ATTRIBUTION_VARIANTS,
    ATTRIBUTION_VARIANTS,
    VALUE_RESIDUAL_VARIANTS,
    apply_mrope,
    attention_counterfactuals,
    cluster_mean,
    feature_anchor_partition,
    grid_partition,
    partition_statistics,
    prepare_value_residual_factors,
    repeat_kv,
)
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.backbones.qwen25_vl.implementation import _move_value
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord
from mllm_comp.core.serialization import atomic_write_json


_DEFAULT_VARIANTS = ("pre", "post", *ATTRIBUTION_VARIANTS)


class VisualMergeAttribution:
    """Collect visual-merge and selected-layer attention evidence read-only."""

    name = "visual_merge_attribution"
    version = "0.1.0"
    supported_partitions = {"grid", "feature_anchor"}
    supported_variants = set(
        (*_DEFAULT_VARIANTS, *VALUE_RESIDUAL_VARIANTS)
    )

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="visual merge attribution")
        layers = config.get("layers", [0])
        validate_layer_selection(layers, context="visual merge attribution layers")
        if len(layers) != 1:
            raise ValueError(
                "Each visual_merge_attribution run must select exactly one decoder layer."
            )
        positive_integer(
            config, "target_tokens", 64, context="visual merge attribution"
        )
        partitions = set(
            string_sequence(
                config,
                "partitions",
                ("grid", "feature_anchor"),
                context="visual merge attribution",
            )
        )
        if not partitions or not partitions.issubset(self.supported_partitions):
            raise ValueError(
                f"Unsupported visual partition methods: {sorted(partitions - self.supported_partitions)}"
            )
        variants = set(
            string_sequence(
                config,
                "variants",
                _DEFAULT_VARIANTS,
                context="visual merge attribution",
            )
        )
        if not variants or not variants.issubset(self.supported_variants):
            raise ValueError(
                f"Unsupported visual attribution variants: {sorted(variants - self.supported_variants)}"
            )
        strict_boolean(
            config, "fp32_statistics", True, context="visual merge attribution"
        )
        strict_boolean(
            config, "save_raw", False, context="visual merge attribution"
        )
        strict_boolean(
            config,
            "run_layer0_trajectory",
            False,
            context="visual merge attribution",
        )
        tolerance = config.get("identity_tolerance", 1e-4)
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or tolerance < 0:
            raise ValueError("visual merge attribution identity_tolerance must be non-negative.")

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_VisualMergeAttributionSession":
        self.validate_config(config)
        return _VisualMergeAttributionSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )


@dataclass
class _VisualMergeAttributionSession:
    plugin: VisualMergeAttribution
    backbone: Any
    runtime: Any
    config: dict[str, Any]
    phase: str
    output_dir: str

    def __post_init__(self) -> None:
        if self.backbone.capabilities().name != "qwen25_vl":
            raise BackendIntegrationError(
                "Visual merge attribution currently supports only Qwen2.5-VL."
            )
        self.binding = resolve_analysis_binding(self.backbone, self.runtime.model)
        self.selected_layers = select_decoder_layers(
            self.binding, self.config.get("layers", [0]), context="Visual merge attribution"
        )
        if len(self.selected_layers) != 1:
            raise BackendIntegrationError(
                "Visual merge attribution requires exactly one selected decoder layer."
            )
        self.layer_index = int(self.selected_layers[0].index)
        self.attention_backends = {
            str(self.layer_index): str(
                getattr(
                    getattr(self.selected_layers[0].attention, "config", None),
                    "_attn_implementation",
                    "unknown",
                )
            )
        }
        if self.attention_backends[str(self.layer_index)] != "eager":
            raise BackendIntegrationError(
                "Visual merge attribution requires attn_implementation=eager."
            )
        self.target_tokens = int(self.config.get("target_tokens", 64))
        self.partitions = string_sequence(
            self.config,
            "partitions",
            ("grid", "feature_anchor"),
            context="visual merge attribution",
        )
        self.variants = string_sequence(
            self.config,
            "variants",
            _DEFAULT_VARIANTS,
            context="visual merge attribution",
        )
        self.save_raw = bool(self.config.get("save_raw", False))
        self.run_layer0_trajectory = bool(
            self.config.get("run_layer0_trajectory", False)
        )
        if self.run_layer0_trajectory and self.layer_index != 0:
            raise BackendIntegrationError(
                "The compressed trajectory is defined only for decoder layer 0."
            )
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.spool = JsonlRecordSpool(self.output_dir, prefix="visual_merge_attribution.samples")
        self.sample_count = 0
        self.skipped_count = 0
        self.skipped: list[dict[str, Any]] = []
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError("Cannot process after visual attribution completion.")
        try:
            report = self._process_sample(sample)
        except Exception as exc:
            failure = {
                "sample_id": sample.get("sample_id"),
                "task_name": sample.get("task_name"),
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            self.skipped_count += 1
            if len(self.skipped) < 20:
                self.skipped.append(failure)
            self.spool.append({"status": "skipped", **failure})
            return
        self.spool.append({"status": "ok", **report})
        self.sample_count += 1

    def _process_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        state = prepare_analysis_forward_state(
            backbone=self.backbone,
            runtime=self.runtime,
            sample=sample,
            probe_name="Visual merge attribution",
        )
        inputs = {
            name: _move_value(value, device=self.runtime.device, dtype=self.runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        grid = inputs.get("image_grid_thw")
        if grid is None or int(grid.shape[0]) != 1:
            raise BackendIntegrationError(
                "Visual merge attribution requires exactly one image grid."
            )
        if inputs.get("pixel_values_videos") is not None:
            raise BackendIntegrationError("Visual merge attribution does not accept video inputs.")

        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        merger = handles.get("vision_projector.merger")
        vision = handles.get("vision_encoder")
        llm = handles.get("llm")
        attention = self.selected_layers[0].attention
        if merger is None or vision is None or llm is None:
            raise BackendIntegrationError("Qwen2.5-VL lacks vision merger or decoder handles.")

        captured: dict[str, Any] = {}

        def detached(value: Any) -> Any:
            if torch.is_tensor(value):
                return value.detach()
            return value

        def merger_pre(_module: Any, args: tuple[Any, ...]) -> None:
            if args and torch.is_tensor(args[0]):
                captured["merger_input"] = detached(args[0])

        def merger_post(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            captured["visual_features"] = detached(output)

        def layer_pre(_module: Any, args: tuple[Any, ...]) -> None:
            if args and torch.is_tensor(args[0]):
                captured["layer_hidden"] = detached(args[0])

        def norm_pre(_module: Any, args: tuple[Any, ...]) -> None:
            if args and torch.is_tensor(args[0]):
                captured["pre_norm"] = detached(args[0])

        def norm_post(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            captured["post_norm"] = detached(output)

        def projection_hook(name: str):
            def callback(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
                captured[name] = detached(output)
            return callback

        def attention_pre(_module: Any, _args: tuple[Any, ...], kwargs: dict[str, Any]):
            captured["position_embeddings"] = kwargs.get("position_embeddings")
            captured["attention_mask"] = kwargs.get("attention_mask")

        def output_projection_pre(_module: Any, args: tuple[Any, ...]) -> None:
            if args and torch.is_tensor(args[0]):
                captured["attention_output_pre_projection"] = detached(args[0])

        def attention_post(_module: Any, _args: tuple[Any, ...], output: Any) -> None:
            if isinstance(output, (tuple, list)) and len(output) > 1:
                captured["attention_weights"] = detached(output[1])
            if isinstance(output, (tuple, list)) and output:
                captured["attention_output"] = detached(output[0])

        try:
            with HookSession() as hooks:
                hooks.register_forward_pre_hook(merger, merger_pre)
                hooks.register_forward_hook(merger, merger_post)
                hooks.register_forward_pre_hook(self.selected_layers[0].layer, layer_pre)
                hooks.register_forward_pre_hook(
                    self.selected_layers[0].layer.input_layernorm, norm_pre
                )
                hooks.register_forward_hook(
                    self.selected_layers[0].layer.input_layernorm, norm_post
                )
                hooks.register_forward_hook(attention.q_proj, projection_hook("q_raw"))
                hooks.register_forward_hook(attention.k_proj, projection_hook("k_raw"))
                hooks.register_forward_hook(attention.v_proj, projection_hook("v_raw"))
                hooks.register_forward_pre_hook(attention, attention_pre, with_kwargs=True)
                hooks.register_forward_hook(attention, attention_post)
                hooks.register_forward_pre_hook(attention.o_proj, output_projection_pre)
                with torch.inference_mode():
                    forward_inputs = dict(inputs)
                    forward_inputs["output_hidden_states"] = self.run_layer0_trajectory
                    outputs = self.runtime.model(**forward_inputs)
                    repeat_outputs = self.runtime.model(**inputs)
        except torch.cuda.OutOfMemoryError as exc:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise BackendIntegrationError(
                f"Layer-{self.layer_index} eager attention exceeded available GPU memory."
            ) from exc

        required = (
            "merger_input",
            "visual_features",
            "pre_norm",
            "post_norm",
            "q_raw",
            "k_raw",
            "v_raw",
            "position_embeddings",
            "attention_weights",
            "attention_output_pre_projection",
        )
        missing = [name for name in required if captured.get(name) is None]
        if missing:
            raise BackendIntegrationError(f"Qwen capture missed required tensors: {missing}.")
        logits = getattr(outputs, "logits", None)
        repeat_logits = getattr(repeat_outputs, "logits", None)
        if logits is None or repeat_logits is None:
            raise BackendIntegrationError("Qwen forward did not return logits.")

        merge_size = int(getattr(vision, "spatial_merge_size", 2))
        height = int(grid[0, 1]) // merge_size
        width = int(grid[0, 2]) // merge_size
        visual_features = captured["visual_features"].reshape(-1, captured["visual_features"].shape[-1])
        window_index, _ = vision.get_window_index(grid)
        reverse_window_index = torch.argsort(window_index).to(visual_features.device)
        visual_features = visual_features.index_select(0, reverse_window_index)
        if height * width != int(visual_features.shape[0]):
            raise BackendIntegrationError(
                f"Post-merger grid {height}x{width} does not match "
                f"{int(visual_features.shape[0])} visual features."
            )
        if int(visual_features.shape[0]) < self.target_tokens:
            return {
                "sample_id": sample["sample_id"],
                "task_name": sample["task_name"],
                "status": "inapplicable",
                "reason": "source_tokens_below_target_tokens",
                "source_tokens": int(visual_features.shape[0]),
                "target_tokens": self.target_tokens,
                "grid_shape": [height, width],
            }

        # Qwen computes its three spatial position planes from native processor
        # inputs. Preserve that exact contract for text and source visual tokens.
        rope_positions, rope_delta = self.runtime.model.model.get_rope_index(
            inputs.get("input_ids"),
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        if int(rope_positions.shape[0]) == 4:
            mrope_positions = rope_positions[1:]
        elif int(rope_positions.shape[0]) == 3:
            mrope_positions = rope_positions
        else:
            raise BackendIntegrationError("Qwen rope index did not return three M-RoPE planes.")

        attention_mask = inputs.get("attention_mask")
        seq_len = int(inputs["input_ids"].shape[-1])
        if torch.is_tensor(attention_mask) and not bool((attention_mask[0] != 0).all().item()):
            raise BackendIntegrationError(
                "Layer-zero attribution currently requires an unpadded single-sample prompt."
            )
        visual_positions = _visual_positions_from_placeholders(
            inputs["input_ids"], self.runtime.model.config.image_token_id
        )
        if len(visual_positions) != int(visual_features.shape[0]):
            raise BackendIntegrationError(
                "Image placeholder count and post-merger visual feature count disagree."
            )
        query_position = _last_valid_position(attention_mask, seq_len)
        if query_position < visual_positions[-1]:
            raise BackendIntegrationError("Final text query does not follow the image tokens.")

        heads = int(attention.num_heads)
        head_dim = int(attention.head_dim)
        kv_heads = int(attention.num_key_value_heads)
        q_raw = captured["q_raw"].reshape(1, seq_len, heads, head_dim).transpose(1, 2)[0]
        k_raw = captured["k_raw"].reshape(1, seq_len, kv_heads, head_dim).transpose(1, 2)[0]
        v_raw = captured["v_raw"].reshape(1, seq_len, kv_heads, head_dim).transpose(1, 2)[0]
        q_all, k_unused = _rotate_native(q_raw, k_raw, captured["position_embeddings"], attention)
        del k_unused
        # The model attention's native M-RoPE is confirmed against its eager
        # attention output; the helper above uses the exact captured cos/sin.
        k = _rotate_keys(k_raw, captured["position_embeddings"], attention)
        q = q_all[:, query_position : query_position + 1]
        visual_index = torch.tensor(visual_positions, device=q.device, dtype=torch.long)
        all_prefix = [index for index in range(query_position + 1) if index not in set(visual_positions)]
        text_index = torch.tensor(all_prefix, device=q.device, dtype=torch.long)
        expanded_k = repeat_kv(k, heads)
        expanded_v = repeat_kv(v_raw, heads)
        scale = head_dim**-0.5
        full_prefix_keys = expanded_k[:, : query_position + 1]
        full_prefix_values = expanded_v[:, : query_position + 1]
        manual_weights = torch.softmax(
            torch.einsum("hqd,hkd->hqk", q, full_prefix_keys) * scale,
            dim=-1,
        ).to(dtype=q.dtype)
        actual_attention = captured["attention_weights"].float()
        actual_query_weights = actual_attention[0, :, query_position, : query_position + 1]
        attention_weight_max_abs = float(
            (manual_weights[:, 0] - actual_query_weights).abs().max().item()
        )
        manual_head_output = torch.einsum(
            "hk,hkd->hd", actual_query_weights, full_prefix_values.float()
        )
        actual_head_output = captured["attention_output_pre_projection"][0, query_position].float().reshape(
            heads, head_dim
        )
        attention_output_rel = _relative_vector_error(manual_head_output, actual_head_output)
        configured_tolerance = float(self.config.get("identity_tolerance", 1e-4))
        native_tolerance = max(
            configured_tolerance,
            2.0 * float(torch.finfo(captured["q_raw"].dtype).eps),
        )
        if attention_weight_max_abs > native_tolerance or attention_output_rel > native_tolerance:
            raise BackendIntegrationError(
                f"Manual layer-{self.layer_index} Q/K/V reconstruction does not match native eager "
                f"attention (weights={attention_weight_max_abs:.3g}, "
                f"output_relative={attention_output_rel:.3g}, "
                f"tolerance={native_tolerance:.3g}, q={tuple(q.shape)}, "
                f"k={tuple(full_prefix_keys.shape)}, actual_sum={float(actual_query_weights.sum().item()):.5g}, "
                f"manual_sum={float(manual_weights.sum().item()):.5g}, "
                f"actual_top={actual_query_weights[0].topk(min(5, actual_query_weights.shape[-1])).values.tolist()}, "
                f"manual_top={manual_weights[0, 0].topk(min(5, manual_weights.shape[-1])).values.tolist()})."
            )
        text_keys = expanded_k.index_select(1, text_index)
        text_values = expanded_v.index_select(1, text_index)
        visual_keys = expanded_k.index_select(1, visual_index)
        visual_values = expanded_v.index_select(1, visual_index)
        raw_visual_hidden = captured["pre_norm"][0].index_select(0, visual_index)
        raw_visual_norm = captured["post_norm"][0].index_select(0, visual_index)
        source_positions = mrope_positions[:, 0, visual_index].to(torch.float32)
        # Normalize the model's accepted [3, batch, seq] layout to [3, tokens].
        if source_positions.ndim == 3:
            source_positions = source_positions[:, 0, :]
        mrope_section = self.runtime.model.config.text_config.rope_scaling["mrope_section"]
        detailed: dict[str, Any] = {
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "checkpoint": str(self.runtime.model_path_or_id or ""),
            "checkpoint_revision": self.runtime.revision,
            "source_tokens": int(visual_features.shape[0]),
            "target_tokens": self.target_tokens,
            "grid_shape": [height, width],
            "image_grid_thw": [int(item) for item in grid[0].tolist()],
            "visual_positions": visual_positions,
            "text_positions": all_prefix,
            "text_query_positions": [
                position for position in all_prefix if position > visual_positions[-1]
            ],
            "mrope_position_shape": list(rope_positions.shape),
            "rope_delta": rope_delta.detach().cpu().tolist() if torch.is_tensor(rope_delta) else rope_delta,
            "query_position": query_position,
            "layer_index": self.layer_index,
            "mask_shape": list(attention_mask.shape) if torch.is_tensor(attention_mask) else None,
            "logits_repeat_max_abs": float((logits.float() - repeat_logits.float()).abs().max().item()),
            "logits_shape": list(logits.shape),
            "logits_last_top_tokens": _top_logits(logits[0, query_position]),
            "attention_reconstruction": {
                "weight_max_abs": attention_weight_max_abs,
                "output_relative_l2": attention_output_rel,
                "configured_tolerance": configured_tolerance,
                "native_dtype_tolerance": native_tolerance,
            },
            "partition_results": {},
        }

        # Qwen2.5-VL's PatchMerger receives the pre-merger stream [patches,
        # vision_width] and groups every spatial_merge_unit patches internally.
        # Reconstruct that grouping before mapping it back from window order.
        merger_input = captured["merger_input"].reshape(
            -1, merge_size**2, captured["merger_input"].shape[-1]
        )
        merger_input = merger_input.index_select(0, reverse_window_index.to(merger_input.device))
        if int(merger_input.shape[0]) != height * width:
            raise BackendIntegrationError(
                "Vision merger input rows do not match post-merger spatial positions."
            )
        for partition_name in self.partitions:
            partition = (
                grid_partition(height, width, self.target_tokens)
                if partition_name == "grid"
                else feature_anchor_partition(
                    visual_features,
                    height=height,
                    width=width,
                    target_tokens=self.target_tokens,
                )
            )
            stats = partition_statistics(partition, visual_features)
            pre_projected = []
            post_projected = []
            with torch.inference_mode():
                for cluster in partition.members:
                    rows = merger_input[list(cluster)].to(dtype=merger.mlp[0].weight.dtype)
                    pre_projected.append(
                        merger(
                            rows.float()
                            .mean(dim=0, keepdim=True)
                            .to(dtype=merger.mlp[0].weight.dtype)
                        )[0].float()
                    )
                    post_projected.append(visual_features[list(cluster)].float().mean(dim=0))
            pre_projected_tensor = torch.stack(pre_projected)
            post_projected_tensor = torch.stack(post_projected)
            projector_error = _relative_vector_error(pre_projected_tensor, post_projected_tensor)
            partition_result: dict[str, Any] = {
                "partition_hash": partition.partition_hash,
                "cluster_sizes": [len(cluster) for cluster in partition.members],
                "cluster_statistics": stats["clusters"],
                "projector_pre_post_relative_l2": projector_error,
                "variants": {},
            }
            # For attention diagnostics we use the post-merger mean representative,
            # consistent with the fixed K token entering the Qwen language model.
            representative_norm = cluster_mean(raw_visual_norm, partition.members)
            representative_k_input = torch.stack(
                [
                    raw_visual_hidden[list(cluster)].mean(dim=0)
                    for cluster in partition.members
                ]
            )
            norm_of_mean = self.selected_layers[0].layer.input_layernorm(
                representative_k_input.to(dtype=raw_visual_hidden.dtype)
            ).float()
            representative_k_raw = attention.k_proj(
                norm_of_mean.to(dtype=attention.k_proj.weight.dtype)
            ).reshape(self.target_tokens, kv_heads, head_dim).transpose(0, 1)
            representative_k_raw = repeat_kv(representative_k_raw, heads)
            representative_v = torch.stack(
                [
                    visual_values.index_select(
                        1,
                        torch.tensor(cluster, device=visual_values.device, dtype=torch.long),
                    ).mean(dim=1)
                    for cluster in partition.members
                ],
                dim=1,
            )
            representative_positions = torch.stack(
                [source_positions[:, list(cluster)].mean(dim=1) for cluster in partition.members],
                dim=0,
            ).transpose(0, 1)
            # Generate the three-axis rotary factors at cluster centroids.
            native_rotary = llm.rotary_emb
            centroid_cos, centroid_sin = native_rotary(
                norm_of_mean.unsqueeze(0),
                representative_positions.unsqueeze(1),
            )
            representative_keys = _rotate_keys(
                representative_k_raw,
                (centroid_cos, centroid_sin),
                attention,
            )
            # RMSNorm and W_K realization audits compare P(mean(x)) to mean(P(x)).
            norm_mean_error = _relative_vector_error(norm_of_mean, representative_norm)
            raw_visual_k = repeat_kv(k_raw, heads).index_select(1, visual_index)
            raw_key_mean = torch.stack(
                [
                    raw_visual_k.index_select(
                        1,
                        torch.tensor(cluster, device=visual_keys.device, dtype=torch.long),
                    ).mean(dim=1)
                    for cluster in partition.members
                ],
                dim=1,
            )
            wk_relative_by_head = (
                (representative_k_raw - raw_key_mean).norm(dim=-1)
                / raw_key_mean.norm(dim=-1).clamp_min(1e-12)
            ).mean(dim=-1)
            rope_relative_by_head = (
                (
                    representative_keys
                    - torch.stack(
                        [
                            visual_keys.index_select(
                                1,
                                torch.tensor(cluster, device=visual_keys.device, dtype=torch.long),
                            ).mean(dim=1)
                            for cluster in partition.members
                        ],
                        dim=1,
                    )
                ).norm(dim=-1)
                / torch.stack(
                    [
                        visual_keys.index_select(
                            1,
                            torch.tensor(cluster, device=visual_keys.device, dtype=torch.long),
                        ).mean(dim=1)
                        for cluster in partition.members
                    ],
                    dim=1,
                ).norm(dim=-1).clamp_min(1e-12)
            ).mean(dim=-1)
            mean_rotated_keys = torch.stack(
                [
                    visual_keys.index_select(
                        1,
                        torch.tensor(cluster, device=visual_keys.device, dtype=torch.long),
                    ).mean(dim=1)
                    for cluster in partition.members
                ],
                dim=1,
            )
            rotated_mean_raw_key = _rotate_keys(
                raw_key_mean,
                (centroid_cos, centroid_sin),
                attention,
            )
            wk_delta = representative_keys.float() - rotated_mean_raw_key.float()
            rope_delta = rotated_mean_raw_key.float() - mean_rotated_keys.float()
            decomposition_residual = (
                representative_keys.float()
                - mean_rotated_keys.float()
                - wk_delta
                - rope_delta
            )
            counterfactuals = _counterfactuals_over_text_queries(
                q_all=q_all,
                query_positions=detailed["text_query_positions"],
                text_positions=all_prefix,
                expanded_keys=expanded_k,
                expanded_values=expanded_v,
                visual_keys=visual_keys,
                visual_values=visual_values,
                representative_keys=representative_keys,
                representative_values=representative_v,
                members=partition.members,
                variants=[name for name in ALL_ATTRIBUTION_VARIANTS if name in self.variants],
            )
            partition_result.update(
                {
                    "rmsnorm_mean_noncommutativity_relative_l2": norm_mean_error,
                    "wk_relative_error_by_head": [float(value.item()) for value in wk_relative_by_head],
                    "mrope_key_relative_error_by_head": [float(value.item()) for value in rope_relative_by_head],
                    "error_decomposition_closure_max_abs": float(
                        decomposition_residual.abs().max().item()
                    ),
                    "counterfactuals": counterfactuals,
                }
            )
            partition_result["variants"] = {
                name: counterfactuals[name]
                for name in ALL_ATTRIBUTION_VARIANTS
                if name in counterfactuals
            }
            if self.save_raw:
                partition_result["partition_members"] = [list(cluster) for cluster in partition.members]
                partition_result["source_mrope_positions"] = source_positions.detach().cpu().tolist()
            detailed["partition_results"][partition_name] = partition_result

        if self.run_layer0_trajectory:
            detailed["layer0_compressed_trajectory"] = self._run_layer0_trajectory(
                inputs=inputs,
                baseline_outputs=outputs,
                baseline_hidden=captured["layer_hidden"],
                visual_positions=visual_positions,
                text_positions=all_prefix,
                source_positions=source_positions,
                partition_features=visual_features,
                height=height,
                width=width,
            )

        visual_mass = actual_attention[0, :, query_position, visual_index].sum(dim=-1)
        detailed["layer_attention"] = {
            "layer_index": self.layer_index,
            "heads": int(actual_attention.shape[1]),
            "query_position": query_position,
            "visual_mass_by_head": [float(value.item()) for value in visual_mass],
            "attention_output_norm": float(captured["attention_output"][0, query_position].float().norm().item()),
            "attention_mask_shape": list(captured["attention_mask"].shape) if torch.is_tensor(captured.get("attention_mask")) else None,
        }
        return detailed

    def _run_layer0_trajectory(
        self,
        *,
        inputs: Mapping[str, Any],
        baseline_outputs: Any,
        baseline_hidden: Any,
        visual_positions: Sequence[int],
        text_positions: Sequence[int],
        source_positions: Any,
        partition_features: Any,
        height: int,
        width: int,
    ) -> dict[str, Any]:
        """Run the frozen decoder on mean-merged embeddings from layer zero.

        This is the plan's Layer-0 one-time-compression trajectory: each group
        is replaced by its mean pre-layer embedding, placed at the rounded
        three-axis M-RoPE centroid. All text embeddings and their native
        position IDs are retained. It is diagnostic only and never writes to
        model weights or feeds any Stage-2 training path.
        """
        import torch
        import torch.nn.functional as functional

        baseline_states = getattr(baseline_outputs, "hidden_states", None)
        if baseline_states is None:
            raise BackendIntegrationError(
                "Layer-0 trajectory requested but baseline hidden states were not returned."
            )
        if baseline_hidden.ndim != 3 or int(baseline_hidden.shape[0]) != 1:
            raise BackendIntegrationError("Layer-0 embedding capture has an unexpected shape.")
        original_positions, _ = self.runtime.model.model.get_rope_index(
            inputs.get("input_ids"),
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        if int(original_positions.shape[0]) == 4:
            original_positions = original_positions[1:]
        if int(original_positions.shape[0]) != 3:
            raise BackendIntegrationError("Compressed trajectory requires three M-RoPE planes.")
        original_positions = original_positions[:, 0].to(device=baseline_hidden.device)
        llm = self.backbone.model_adapter.structural_handles(self.runtime.model).get("llm")
        lm_head = getattr(self.runtime.model, "lm_head", None)
        if llm is None or lm_head is None:
            raise BackendIntegrationError("Could not resolve Qwen decoder or language-model head.")
        base_hidden_states = [state.detach() for state in baseline_states]
        base_logits = baseline_outputs.logits.detach()
        sequence_length = int(baseline_hidden.shape[1])
        visual_to_source = {position: index for index, position in enumerate(visual_positions)}
        results: dict[str, Any] = {}

        for partition_name in self.partitions:
            partition = (
                grid_partition(height, width, self.target_tokens)
                if partition_name == "grid"
                else feature_anchor_partition(
                    partition_features,
                    height=height,
                    width=width,
                    target_tokens=self.target_tokens,
                )
            )
            starts = {min(members): members for members in partition.members}
            covered = {source for members in partition.members for source in members}
            if covered != set(range(len(visual_positions))):
                raise BackendIntegrationError("Compressed trajectory partition lost visual tokens.")
            embeddings: list[Any] = []
            positions: list[Any] = []
            source_to_compact: dict[int, int] = {}
            for source_position in range(sequence_length):
                visual_index = visual_to_source.get(source_position)
                if visual_index is None:
                    source_to_compact[source_position] = len(embeddings)
                    embeddings.append(baseline_hidden[0, source_position])
                    positions.append(original_positions[:, source_position])
                    continue
                members = starts.get(visual_index)
                if members is None:
                    continue
                device_indices = torch.tensor(
                    members, device=baseline_hidden.device, dtype=torch.long
                )
                cluster_embedding = baseline_hidden[0].index_select(
                    0,
                    torch.tensor(
                        [visual_positions[index] for index in members],
                        device=baseline_hidden.device,
                    ),
                ).float().mean(dim=0).to(dtype=baseline_hidden.dtype)
                source_to_compact[source_position] = len(embeddings)
                embeddings.append(cluster_embedding)
                centroid = source_positions[:, device_indices].float().mean(dim=1).round().long()
                positions.append(centroid)
            compressed_embeddings = torch.stack(embeddings).unsqueeze(0)
            compressed_positions = torch.stack(positions, dim=1).unsqueeze(1).long()
            compact_text = [
                (source, source_to_compact[source])
                for source in text_positions
                if source in source_to_compact
            ]
            if not compact_text:
                raise BackendIntegrationError("No text positions survived visual compression.")
            compact_text_indices = torch.tensor(
                [compact for _, compact in compact_text],
                device=baseline_hidden.device,
                dtype=torch.long,
            )
            baseline_text_indices = torch.tensor(
                [source for source, _ in compact_text],
                device=base_logits.device,
                dtype=torch.long,
            )
            compressed_length = int(compressed_embeddings.shape[1])
            attention_mask = torch.ones(
                (1, compressed_length), device=baseline_hidden.device, dtype=torch.long
            )
            cache_position = torch.arange(
                compressed_length, device=baseline_hidden.device, dtype=torch.long
            )
            with torch.inference_mode():
                compressed_output = llm(
                    inputs_embeds=compressed_embeddings,
                    attention_mask=attention_mask,
                    position_ids=compressed_positions,
                    cache_position=cache_position,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True,
                )
                compressed_logits = lm_head(compressed_output.last_hidden_state)

            compressed_text_logits = compressed_logits.index_select(
                1, compact_text_indices
            ).float()
            baseline_text_logits = base_logits.index_select(
                1, baseline_text_indices
            ).float()
            teacher_log_probs = functional.log_softmax(baseline_text_logits, dim=-1)
            student_log_probs = functional.log_softmax(compressed_text_logits, dim=-1)
            teacher_probs = teacher_log_probs.exp()
            kl = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
            hidden_drift: dict[str, float] = {}
            for layer_index, (teacher_state, student_state) in enumerate(
                zip(base_hidden_states, compressed_output.hidden_states)
            ):
                teacher_text = teacher_state.index_select(1, baseline_text_indices).float()
                student_text = student_state.index_select(1, compact_text_indices).float()
                numerator = (teacher_text - student_text).norm(dim=-1)
                denominator = teacher_text.norm(dim=-1).clamp_min(1e-12)
                hidden_drift[str(layer_index)] = float((numerator / denominator).mean().item())
            top1_agreement = (
                baseline_text_logits.argmax(dim=-1)
                == compressed_text_logits.argmax(dim=-1)
            ).float().mean()
            final_source, final_compact = compact_text[-1]
            final_logit_error = _relative_vector_error(
                compressed_logits[0, final_compact].float(),
                base_logits[0, final_source].float(),
            )
            results[partition_name] = {
                "partition_hash": partition.partition_hash,
                "source_tokens": len(visual_positions),
                "compressed_tokens": int(self.target_tokens),
                "sequence_length_before": sequence_length,
                "sequence_length_after": compressed_length,
                "text_positions_preserved": len(compact_text) == len(text_positions),
                "mrope_text_positions_exact": all(
                    torch.equal(
                        compressed_positions[:, 0, compact],
                        original_positions[:, source],
                    )
                    for source, compact in compact_text
                ),
                "mean_teacher_to_compressed_text_kl": float(kl.mean().item()),
                "text_top1_agreement": float(top1_agreement.item()),
                "final_query_logits_relative_l2": final_logit_error,
                "hidden_state_relative_drift_by_layer": hidden_drift,
            }
        return results

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Visual attribution session was already finished.")
        self._finished = True
        payload_path = Path(self.output_dir) / "visual_merge_attribution.samples.jsonl"
        aggregate_by_task: dict[str, dict[str, dict[str, Any]]] = {}
        trajectory_by_task_partition: dict[str, dict[str, dict[str, Any]]] = {}
        applicable_count = 0
        inapplicable_count = 0
        gate_failures: list[dict[str, Any]] = []
        with payload_path.open("w", encoding="utf-8") as output:
            for line in _read_lines_from_spool(self.spool.path):
                output.write(line + "\n")
                report = json.loads(line)
                if report.get("status") != "ok":
                    if report.get("status") == "inapplicable":
                        inapplicable_count += 1
                    continue
                applicable_count += 1
                if not _all_finite(report):
                    gate_failures.append(
                        {"sample_id": report.get("sample_id"), "reason": "non_finite_value"}
                    )
                trajectory = report.get("layer0_compressed_trajectory")
                if trajectory is not None:
                    for partition_name, values in trajectory.items():
                        if not values.get("text_positions_preserved") or not values.get(
                            "mrope_text_positions_exact"
                        ):
                            gate_failures.append(
                                {
                                    "sample_id": report.get("sample_id"),
                                    "partition": partition_name,
                                    "reason": "compressed_trajectory_text_position_mismatch",
                                }
                            )
                        if int(values.get("compressed_tokens", -1)) != self.target_tokens:
                            gate_failures.append(
                                {
                                    "sample_id": report.get("sample_id"),
                                    "partition": partition_name,
                                    "reason": "compressed_trajectory_token_count_mismatch",
                                }
                            )
                        task = str(report.get("task_name", ""))
                        state = trajectory_by_task_partition.setdefault(task, {}).setdefault(
                            partition_name,
                            {
                                "sample_count": 0,
                                "mean_teacher_to_compressed_text_kl_sum": 0.0,
                                "text_top1_agreement_sum": 0.0,
                                "final_query_logits_relative_l2_sum": 0.0,
                                "hidden_state_relative_drift_by_layer_sum": {},
                            },
                        )
                        state["sample_count"] += 1
                        for metric in (
                            "mean_teacher_to_compressed_text_kl",
                            "text_top1_agreement",
                            "final_query_logits_relative_l2",
                        ):
                            state[f"{metric}_sum"] += float(values[metric])
                        for layer, value in values[
                            "hidden_state_relative_drift_by_layer"
                        ].items():
                            drift = state["hidden_state_relative_drift_by_layer_sum"]
                            drift[layer] = drift.get(layer, 0.0) + float(value)
                repeat_error = float(report.get("logits_repeat_max_abs", float("inf")))
                repeat_tolerance = float(
                    report.get("attention_reconstruction", {}).get(
                        "native_dtype_tolerance", 0.0
                    )
                )
                if repeat_error > repeat_tolerance:
                    gate_failures.append(
                        {
                            "sample_id": report.get("sample_id"),
                            "reason": "repeat_logits_mismatch",
                            "max_abs": repeat_error,
                            "tolerance": repeat_tolerance,
                        }
                    )
                if report.get("attention_reconstruction", {}).get(
                    "weight_max_abs", float("inf")
                ) > report.get("attention_reconstruction", {}).get(
                    "native_dtype_tolerance", 0
                ):
                    gate_failures.append(
                        {"sample_id": report.get("sample_id"), "reason": "attention_reconstruction"}
                    )
                for partition_name, partition in report.get("partition_results", {}).items():
                    closure = float(partition.get("error_decomposition_closure_max_abs", float("inf")))
                    if closure > 1e-4:
                        gate_failures.append(
                            {
                                "sample_id": report.get("sample_id"),
                                "partition": partition_name,
                                "reason": "error_decomposition_closure",
                                "max_abs": closure,
                            }
                        )
                    for oracle_name, metric_name in (
                        ("O-Z", "visual_attention_mass_abs_error"),
                        ("O-M", "attention_output_relative_error"),
                    ):
                        oracle = partition.get("variants", {}).get(oracle_name, {})
                        oracle_error = float(
                            oracle.get(metric_name, {}).get("mean", float("inf"))
                        )
                        if oracle_error > 1e-5:
                            gate_failures.append(
                                {
                                    "sample_id": report.get("sample_id"),
                                    "partition": partition_name,
                                    "variant": oracle_name,
                                    "reason": "oracle_reconstruction",
                                    "metric": metric_name,
                                    "mean": oracle_error,
                                    "tolerance": 1e-5,
                                }
                            )
                    task = str(report.get("task_name", ""))
                    task_state = aggregate_by_task.setdefault(task, {}).setdefault(
                        partition_name,
                        {
                            "sample_count": 0,
                            "projector_pre_post_relative_l2_sum": 0.0,
                            "rmsnorm_mean_noncommutativity_relative_l2_sum": 0.0,
                            "error_decomposition_closure_max_abs": 0.0,
                            "variants": {},
                        },
                    )
                    task_state["sample_count"] += 1
                    task_state["projector_pre_post_relative_l2_sum"] += float(
                        partition["projector_pre_post_relative_l2"]
                    )
                    task_state["rmsnorm_mean_noncommutativity_relative_l2_sum"] += float(
                        partition["rmsnorm_mean_noncommutativity_relative_l2"]
                    )
                    task_state["error_decomposition_closure_max_abs"] = max(
                        task_state["error_decomposition_closure_max_abs"], closure
                    )
                    for variant_name, variant_report in partition.get("variants", {}).items():
                        metric_state = task_state["variants"].setdefault(
                            variant_name,
                            {
                                "attention_output_relative_error_sum": 0.0,
                                "visual_attention_mass_abs_error_sum": 0.0,
                                "text_attention_l1_error_sum": 0.0,
                                "cluster_log_partition_abs_error_sum": 0.0,
                            },
                        )
                        for metric in metric_state:
                            metric_state[metric] += float(
                                variant_report[metric.removesuffix("_sum")]["mean"]
                            )
        for task_partitions in aggregate_by_task.values():
            for values in task_partitions.values():
                count = max(1, int(values["sample_count"]))
                for metric in (
                    "projector_pre_post_relative_l2_sum",
                    "rmsnorm_mean_noncommutativity_relative_l2_sum",
                ):
                    values[metric.removesuffix("_sum")] = values.pop(metric) / count
                for variant_values in values["variants"].values():
                    for metric in list(variant_values):
                        variant_values[metric.removesuffix("_sum")] = (
                            variant_values.pop(metric) / count
                        )
        for task_partitions in trajectory_by_task_partition.values():
            for values in task_partitions.values():
                count = max(1, int(values["sample_count"]))
                for metric in (
                    "mean_teacher_to_compressed_text_kl_sum",
                    "text_top1_agreement_sum",
                    "final_query_logits_relative_l2_sum",
                ):
                    values[metric.removesuffix("_sum")] = values.pop(metric) / count
                values["hidden_state_relative_drift_by_layer"] = {
                    layer: value / count
                    for layer, value in values.pop(
                        "hidden_state_relative_drift_by_layer_sum"
                    ).items()
                }
        self.spool.close()
        aggregate_path = Path(self.output_dir) / "visual_merge_attribution.aggregate.json"
        aggregate = {
            "sample_count": self.sample_count,
            "applicable_sample_count": applicable_count,
            "inapplicable_sample_count": inapplicable_count,
            "skipped_sample_count": self.skipped_count,
            "skipped_samples": self.skipped,
            "pilot_gate": (
                "pass"
                if applicable_count > 0 and self.skipped_count == 0 and not gate_failures
                else "needs_review"
            ),
            "pilot_gate_failures": gate_failures,
            "aggregates_by_task_partition": aggregate_by_task,
            "layer0_compressed_trajectory_by_task_partition": trajectory_by_task_partition,
            "configured_partitions": list(self.partitions),
            "configured_variants": list(self.variants),
            "configured_layers": [self.layer_index],
            "attention_backends": self.attention_backends,
        }
        artifact = {
            "schema_version": 1,
            "artifact_kind": "aggregate",
            "probe_name": self.config.get("name", self.plugin.name),
            "method": self.plugin.name,
            "method_version": self.plugin.version,
            "phase": self.phase,
            "backbone_name": "qwen25_vl",
            "checkpoint": str(self.runtime.model_path_or_id or ""),
            "checkpoint_revision": self.runtime.revision,
            "payload": aggregate,
        }
        atomic_write_json(aggregate_path, artifact, ensure_ascii=False)
        sample_artifact_path = Path(self.output_dir) / "visual_merge_attribution.samples.json"
        sample_artifact_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_kind": "per_sample",
                    "probe_name": self.config.get("name", self.plugin.name),
                    "method": self.plugin.name,
                    "method_version": self.plugin.version,
                    "phase": self.phase,
                    "backbone_name": "qwen25_vl",
                    "checkpoint": str(self.runtime.model_path_or_id or ""),
                    "checkpoint_revision": self.runtime.revision,
                    "payload": {"samples_jsonl": str(payload_path)},
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return AnalysisRecord(
            name=str(self.config.get("name", self.plugin.name)),
            phase=self.phase,
            method=self.plugin.name,
            summary={
                "successful_samples": self.sample_count,
                "skipped_samples": self.skipped_count,
                "target_tokens": self.target_tokens,
                "partitions": list(self.partitions),
                "variants": list(self.variants),
            },
            artifact_paths={
                "aggregate": str(aggregate_path),
                "per_sample": str(payload_path),
            },
            sample_count=self.sample_count,
        )

    def close(self) -> None:
        if self._closed:
            return
        if not self._finished:
            self.spool.close()
        self._closed = True


def _visual_positions_from_placeholders(input_ids: Any, image_token_id: int) -> list[int]:
    positions = (input_ids[0] == int(image_token_id)).nonzero(as_tuple=True)[0]
    return [int(value) for value in positions.detach().cpu().tolist()]


def _counterfactuals_over_text_queries(
    *,
    q_all: Any,
    query_positions: Sequence[int],
    text_positions: Sequence[int],
    expanded_keys: Any,
    expanded_values: Any,
    visual_keys: Any,
    visual_values: Any,
    representative_keys: Any,
    representative_values: Any,
    members: Sequence[Sequence[int]],
    variants: Sequence[str],
) -> dict[str, Any]:
    """Evaluate each post-image text query with its exact causal text prefix."""

    import torch

    if not query_positions:
        raise BackendIntegrationError(
            "Visual attribution found no text queries following the image tokens."
        )
    value_ranks = sorted(
        {
            int(name.removeprefix("O-Z-VR"))
            for name in variants
            if name.startswith("O-Z-VR")
        }
    )
    value_residual_factors = (
        prepare_value_residual_factors(
            visual_keys, visual_values, members, value_ranks
        )
        if value_ranks
        else None
    )
    query_reports = []
    for position in query_positions:
        visible_text = [index for index in text_positions if index <= int(position)]
        if not visible_text:
            raise BackendIntegrationError(
                f"Text query at position {position} has no causally visible text keys."
            )
        indices = torch.tensor(visible_text, device=expanded_keys.device, dtype=torch.long)
        report = attention_counterfactuals(
            queries=q_all[:, int(position) : int(position) + 1],
            text_keys=expanded_keys.index_select(1, indices),
            text_values=expanded_values.index_select(1, indices),
            visual_keys=visual_keys,
            visual_values=visual_values,
            representative_keys=representative_keys,
            representative_values=representative_values,
            members=members,
            variants=variants,
            value_residual_factors=value_residual_factors,
        )
        query_reports.append({"query_position": int(position), **report})

    def average_metric(path: str, metric: str) -> dict[str, Any]:
        reports = [item[path][metric] for item in query_reports]
        head_ids = sorted(reports[0]["heads"], key=int)
        return {
            "mean": sum(float(item["mean"]) for item in reports) / len(reports),
            "heads": {
                head: sum(float(item["heads"][head]) for item in reports) / len(reports)
                for head in head_ids
            },
        }

    aggregate: dict[str, Any] = {"query_count": len(query_reports)}
    for variant in variants:
        metric_names = query_reports[0][variant].keys()
        aggregate[variant] = {
            metric: average_metric(variant, metric) for metric in metric_names
        }
    diagnostic_names = [
        metric
        for metric, value in query_reports[0]["diagnostics"].items()
        if isinstance(value, Mapping) and "mean" in value
    ]
    aggregate["diagnostics"] = {
        metric: average_metric("diagnostics", metric)
        for metric in diagnostic_names
    }
    aggregate["per_text_query"] = [
        {
            "query_position": item["query_position"],
            **{
                variant: item[variant]
                for variant in variants
            },
            "diagnostics": item["diagnostics"],
        }
        for item in query_reports
    ]
    return aggregate


def _all_finite(value: Any) -> bool:
    """Check a JSON report recursively for NaN/Inf numerical values."""
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(item) for item in value)
    return True


def _last_valid_position(attention_mask: Any, sequence_length: int) -> int:
    if attention_mask is None:
        return sequence_length - 1
    valid = (attention_mask[0] != 0).nonzero(as_tuple=True)[0]
    if not len(valid):
        raise BackendIntegrationError("Visual attribution found an empty attention mask.")
    return int(valid[-1].item())


def _rotate_native(q_raw: Any, k_raw: Any, position_embeddings: Any, attention: Any) -> tuple[Any, Any]:
    from mllm_comp.analysis.common.visual_merge_attribution import apply_mrope

    cos, sin = position_embeddings
    section = attention.rope_scaling["mrope_section"]
    return (
        apply_mrope(q_raw, cos, sin, section),
        apply_mrope(k_raw, cos, sin, section),
    )


def _rotate_keys(k_raw: Any, position_embeddings: Any, attention: Any) -> Any:
    cos, sin = position_embeddings
    return apply_mrope(k_raw, cos, sin, attention.rope_scaling["mrope_section"])


def _relative_vector_error(left: Any, right: Any, eps: float = 1e-12) -> float:
    left = left.float()
    right = right.float()
    return float(((left - right).norm() / right.norm().clamp_min(eps)).item())


def _top_logits(values: Any, limit: int = 20) -> list[dict[str, Any]]:
    logits, indices = values.float().topk(min(limit, int(values.shape[-1])))
    return [
        {"token_id": int(index), "logit": float(value)}
        for index, value in zip(indices.detach().cpu().tolist(), logits.detach().cpu().tolist())
    ]


def _read_lines_from_spool(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        yield from (line.rstrip("\n") for line in handle)


__all__ = ["VisualMergeAttribution"]
