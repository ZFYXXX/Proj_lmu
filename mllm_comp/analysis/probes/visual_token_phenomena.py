"""Causal, answer-supervised discovery of visual-token compression phenomena."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import time
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
from mllm_comp.analysis.common.sample import prepare_teacher_forced_analysis_forward_state
from mllm_comp.analysis.common.streaming import JsonlRecordSpool
from mllm_comp.analysis.common.visual_merge_attribution import (
    feature_anchor_partition,
    grid_partition,
    partition_statistics,
)
from mllm_comp.analysis.common.visual_token_phenomena import (
    answer_logit_metrics,
    answer_predictor_positions,
    compact_answer_logit_metrics,
    jaccard,
    media_fingerprint,
    safe_budget,
    spatial_coverage,
    spearman_correlation,
    top_fraction_indices,
)
from mllm_comp.analysis.common.visual_token_discovery import joint_deletion_plan, save_region_map
from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings, primary_task_score
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.backbones.qwen25_vl.implementation import _move_value
from mllm_comp.core.errors import BackendIntegrationError
from mllm_comp.core.manifest import AnalysisRecord
from mllm_comp.core.serialization import atomic_write_json


_INTERVENTIONS = {"zero", "global_mean"}
_COMPRESSION_OPERATORS = {
    "prune",
    "representative",
    "cluster_mean",
    "attention_weighted_merge",
    "residual_absorption",
}


class VisualTokenPhenomena:
    """Test visual-token scores and compression hypotheses against answer loss."""

    name = "visual_token_phenomena"
    version = "0.3.0"
    requires_teacher_forcing = True

    def validate_config(self, config: dict[str, Any]) -> None:
        validate_common_probe_config(config, context="visual token phenomena")
        validate_layer_selection(
            config.get("layers", [0, 2, 4, 8]),
            context="visual token phenomenon layers",
        )
        positive_integer(config, "target_units", 64, context="visual token phenomena")
        positive_integer(
            config,
            "max_intervention_units",
            int(config.get("target_units", 64)),
            context="visual token phenomena",
        )
        positive_integer(config, "spatial_bins", 4, context="visual token phenomena")
        interventions = set(
            string_sequence(
                config,
                "interventions",
                ("zero", "global_mean"),
                context="visual token phenomena",
            )
        )
        if not interventions or not interventions.issubset(_INTERVENTIONS):
            raise ValueError(f"Unsupported causal interventions: {sorted(interventions)}")
        operators = set(
            string_sequence(
                config,
                "compression_operators",
                tuple(sorted(_COMPRESSION_OPERATORS)),
                context="visual token phenomena",
            )
        )
        if not operators or not operators.issubset(_COMPRESSION_OPERATORS):
            raise ValueError(f"Unsupported compression operators: {sorted(operators)}")
        for field, default in {
            "run_causal_interventions": True, "run_unit_deletion": False,
            "run_joint_deletion": False, "save_region_maps": False,
            "run_generation": False, "generation_native_check": True,
            "run_budget_frontier": True, "run_layer_trajectory": True, "save_raw": False,
            "run_pair_geometry_interventions": False,
            "run_group_geometry_interventions": False,
            "run_group_geometry_generation": False,
        }.items():
            strict_boolean(config, field, default, context="visual token phenomena")
        if config.get("run_joint_deletion", False) and not config.get("run_unit_deletion", False):
            raise ValueError("Joint deletion requires measured singleton physical deletions.")
        if config.get("run_group_geometry_generation", False) and not config.get("run_group_geometry_interventions", False):
            raise ValueError("Group-geometry generation requires group-geometry interventions.")
        positive_integer(config, "generation_max_new_tokens", 24, context="visual token phenomena")
        positive_integer(config, "group_generation_max_new_tokens", 16, context="visual token phenomena")
        if config.get("run_generation", False) and config.get("intervention_layer", 0) != 0:
            raise ValueError("Free generation currently requires input-layer deletion.")
        repeats = config.get("stability_repeats", 0)
        if isinstance(repeats, bool) or not isinstance(repeats, int) or not 0 <= repeats <= 5:
            raise ValueError("stability_repeats must be an integer in [0,5].")
        fractions = config.get("joint_deletion_fractions", [0.25, 0.5, 0.75])
        if not isinstance(fractions, (list, tuple)) or not fractions or any(
            isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1
            for value in fractions
        ):
            raise ValueError("joint_deletion_fractions must be non-empty fractions in (0,1).")
        if "sample_ids" in config:
            string_sequence(config, "sample_ids", (), context="visual token phenomena")
        _positive_int_sequence(config.get("budgets", [32, 64, 96, 128, 192, 256]), "budgets")
        _positive_int_sequence(config.get("random_seeds", [11, 23, 47]), "random_seeds")
        for field, default in (("safe_nll_relative_limit", 0.05), ("safe_kl_limit", 0.05)):
            value = config.get(field, default)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or float(value) < 0:
                raise ValueError(f"visual token phenomena {field} must be non-negative numeric.")

    def open(
        self,
        *,
        backbone: Any,
        runtime: Any,
        config: dict[str, Any],
        phase: str,
        output_dir: str,
    ) -> "_VisualTokenPhenomenaSession":
        self.validate_config(config)
        return _VisualTokenPhenomenaSession(
            plugin=self,
            backbone=backbone,
            runtime=runtime,
            config=config,
            phase=phase,
            output_dir=output_dir,
        )


@dataclass
class _VisualTokenPhenomenaSession:
    plugin: VisualTokenPhenomena
    backbone: Any
    runtime: Any
    config: dict[str, Any]
    phase: str
    output_dir: str

    def __post_init__(self) -> None:
        if self.backbone.capabilities().name != "qwen25_vl":
            raise BackendIntegrationError("Visual token phenomena currently supports Qwen2.5-VL only.")
        self.binding = resolve_analysis_binding(self.backbone, self.runtime.model)
        self.selected_layers = select_decoder_layers(
            self.binding,
            self.config.get("layers", [0, 2, 4, 8]),
            context="Visual token phenomena",
        )
        self.layer_indices = tuple(int(layer.index) for layer in self.selected_layers)
        self.intervention_layer = int(self.config.get("intervention_layer", 0))
        if self.intervention_layer not in self.layer_indices:
            raise BackendIntegrationError("intervention_layer must be included in layers.")
        for selected in self.selected_layers:
            backend = str(getattr(getattr(selected.attention, "config", None), "_attn_implementation", ""))
            if backend != "eager":
                raise BackendIntegrationError("Visual token phenomena requires eager decoder attention.")
        self.target_units = int(self.config.get("target_units", 64))
        self.max_intervention_units = min(
            self.target_units,
            int(self.config.get("max_intervention_units", self.target_units)),
        )
        self.interventions = string_sequence(
            self.config,
            "interventions",
            ("zero", "global_mean"),
            context="visual token phenomena",
        )
        self.operators = string_sequence(
            self.config,
            "compression_operators",
            tuple(sorted(_COMPRESSION_OPERATORS)),
            context="visual token phenomena",
        )
        self.budgets = tuple(sorted(set(int(v) for v in self.config.get("budgets", [32, 64, 96, 128, 192, 256]))))
        self.random_seeds = tuple(int(v) for v in self.config.get("random_seeds", [11, 23, 47]))
        self.run_causal = bool(self.config.get("run_causal_interventions", True))
        self.run_unit_deletion = bool(self.config.get("run_unit_deletion", False))
        self.run_joint_deletion = bool(self.config.get("run_joint_deletion", False))
        self.run_generation = bool(self.config.get("run_generation", False))
        self.run_pair_geometry = bool(self.config.get("run_pair_geometry_interventions", False))
        self.run_group_geometry = bool(self.config.get("run_group_geometry_interventions", False))
        self.run_group_generation = bool(self.config.get("run_group_geometry_generation", False))
        self.stability_repeats = int(self.config.get("stability_repeats", 0))
        self.sample_ids = set(self.config.get("sample_ids", ()))
        self.unselected_samples = 0
        self.run_frontier = bool(self.config.get("run_budget_frontier", True))
        self.run_layers = bool(self.config.get("run_layer_trajectory", True))
        self.save_raw = bool(self.config.get("save_raw", True))
        if self.config.get("save_region_maps", False):
            import importlib.util
            if importlib.util.find_spec("matplotlib") is None:
                raise BackendIntegrationError("save_region_maps requires the analysis-plots optional dependency.")
        self.implementation_hashes = {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "discovery_helpers": hashlib.sha256(Path(__file__).parents[1].joinpath("common/visual_token_discovery.py").read_bytes()).hexdigest(),
            "generation_helpers": hashlib.sha256(Path(__file__).parents[1].joinpath("common/visual_token_generation.py").read_bytes()).hexdigest(),
        }
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        self.spool = JsonlRecordSpool(self.output_dir, prefix="visual_token_phenomena.samples")
        self.successful = 0
        self.skipped: list[dict[str, Any]] = []
        self._finished = False
        self._closed = False

    def process(self, sample: Mapping[str, Any]) -> None:
        if self._finished or self._closed:
            raise BackendIntegrationError("Cannot process after visual phenomenon completion.")
        if self.sample_ids and str(sample.get("sample_id")) not in self.sample_ids:
            self.unselected_samples += 1
            return
        try:
            result = self._process_sample(sample)
        except Exception as exc:
            failure = {
                "status": "skipped",
                "sample_id": sample.get("sample_id"),
                "task_name": sample.get("task_name"),
                "reason": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            self.skipped.append({k: v for k, v in failure.items() if k != "traceback"})
            self.spool.append(failure)
            return
        self.spool.append({"status": "ok", **result})
        self.successful += 1

    def _process_sample(self, sample: Mapping[str, Any]) -> dict[str, Any]:
        import torch

        if self.run_generation and not callable(sample.get("score_prediction")):
            raise BackendIntegrationError("Free generation requires the native task scorer.")
        state, objective = prepare_teacher_forced_analysis_forward_state(
            backbone=self.backbone,
            runtime=self.runtime,
            sample=sample,
            probe_name="Visual token phenomena",
            output_attentions=True,
            output_hidden_states=True,
        )
        inputs = {
            name: _move_value(value, device=self.runtime.device, dtype=self.runtime.dtype)
            for name, value in state.native_model_inputs.items()
        }
        grid = inputs.get("image_grid_thw")
        if grid is None or int(grid.shape[0]) != 1 or inputs.get("pixel_values_videos") is not None:
            raise BackendIntegrationError("Visual token phenomena requires one image and no video.")
        attention_mask = inputs.get("attention_mask")
        if attention_mask is not None and not bool((attention_mask[0] != 0).all().item()):
            raise BackendIntegrationError("Visual token phenomena requires an unpadded batch-size-one sample.")
        handles = self.backbone.model_adapter.structural_handles(self.runtime.model)
        merger = handles.get("vision_projector.merger")
        vision = handles.get("vision_encoder")
        llm = handles.get("llm")
        lm_head = getattr(self.runtime.model, "lm_head", None)
        if merger is None or vision is None or llm is None or lm_head is None:
            raise BackendIntegrationError("Qwen structural handles are incomplete.")

        captured: dict[str, Any] = {}

        def merger_post(_module: Any, _args: Any, output: Any) -> None:
            captured["visual_features_window"] = output.detach()

        def layer_input_hook(index: int):
            def callback(_module: Any, args: tuple[Any, ...]) -> None:
                if args:
                    captured[f"layer_{index}_input"] = args[0].detach()
            return callback

        def vision_attention_pre(_module: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            hidden = args[0] if args else kwargs.get("hidden_states")
            if hidden is not None:
                captured["vision_attention_hidden"] = hidden.detach()
            cu_seqlens = kwargs.get("cu_seqlens")
            if cu_seqlens is not None:
                captured["vision_attention_cu_seqlens"] = cu_seqlens.detach()
            captured["vision_position_embeddings"] = tuple(
                value.detach() for value in kwargs.get("position_embeddings", ())
            )

        with HookSession() as hooks:
            hooks.register_forward_hook(merger, merger_post)
            hooks.register_forward_pre_hook(
                vision.blocks[-1].attn,
                vision_attention_pre,
                with_kwargs=True,
            )
            for selected in self.selected_layers:
                hooks.register_forward_pre_hook(selected.layer, layer_input_hook(int(selected.index)))
            with torch.inference_mode():
                baseline = self.runtime.model(**inputs)

        logits = getattr(baseline, "logits", None)
        hidden_states = getattr(baseline, "hidden_states", None)
        attentions = getattr(baseline, "attentions", None)
        labels = inputs.get("labels")
        if logits is None or hidden_states is None or attentions is None or labels is None:
            raise BackendIntegrationError("Teacher baseline omitted logits, hidden states, attentions, or labels.")
        predictor_positions, targets = answer_predictor_positions(labels)
        baseline_metrics = answer_logit_metrics(
            logits,
            logits,
            predictor_positions=predictor_positions,
            targets=targets,
        )
        visual_positions = _visual_positions(inputs["input_ids"], self.runtime.model.config.image_token_id)
        first_answer_label = int(torch.nonzero(labels[0] != -100, as_tuple=False)[0].item())
        visual_position_set = set(visual_positions)
        prompt_query_positions = [
            position
            for position in range(first_answer_label)
            if position not in visual_position_set
        ]
        if not prompt_query_positions:
            raise BackendIntegrationError("No prompt text positions are available for query-only attention scores.")
        merge_size = int(getattr(vision, "spatial_merge_size", 2))
        height = int(grid[0, 1]) // merge_size
        width = int(grid[0, 2]) // merge_size
        if len(visual_positions) != height * width:
            raise BackendIntegrationError("Visual placeholder count does not match the post-merger grid.")
        if len(visual_positions) < self.target_units:
            raise BackendIntegrationError("Source visual-token count is below target_units.")
        window_index, _ = vision.get_window_index(grid)
        reverse_window_index = torch.argsort(window_index).to(logits.device)
        visual_features = captured["visual_features_window"].reshape(-1, captured["visual_features_window"].shape[-1])
        visual_features = visual_features.index_select(0, reverse_window_index)
        vision_attention_scores = _vision_attention_received_scores(
            attention=vision.blocks[-1].attn,
            hidden=captured.get("vision_attention_hidden"),
            position_embeddings=captured.get("vision_position_embeddings"),
            cu_seqlens=captured.get("vision_attention_cu_seqlens"),
            reverse_window_index=reverse_window_index,
            merge_unit=merge_size**2,
        )
        intervention_state = captured[f"layer_{self.intervention_layer}_input"]
        visual_index = torch.tensor(visual_positions, device=intervention_state.device, dtype=torch.long)
        visual_hidden = intervention_state[0].index_select(0, visual_index)
        partition = grid_partition(height, width, self.target_units)
        partition_stats = partition_statistics(partition, visual_features)

        score_vectors = self._score_vectors(
            baseline=baseline,
            hidden_states=hidden_states,
            attentions=attentions,
            visual_positions=visual_positions,
            predictor_positions=predictor_positions,
            prompt_query_positions=prompt_query_positions,
            members=partition.members,
            visual_features=visual_features,
            vision_attention_scores=vision_attention_scores,
        )
        for seed in self.random_seeds:
            rng = random.Random(f"{seed}:{sample['sample_id']}:visual-score")
            values = list(range(self.target_units))
            rng.shuffle(values)
            score_vectors[f"random_seed_{seed}"] = [float(value) for value in values]

        interventions: dict[str, list[dict[str, Any]]] = {}
        rope_positions, _ = self.runtime.model.model.get_rope_index(
            inputs.get("input_ids"),
            inputs.get("image_grid_thw"),
            inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        if int(rope_positions.shape[0]) == 4:
            rope_positions = rope_positions[1:]
        source_positions = rope_positions[:, 0].to(intervention_state.device)
        pair_geometry: dict[str, Any] | None = None
        if self.run_pair_geometry:
            pair_geometry = self._run_pair_geometry(
                sample_id=str(sample["sample_id"]),
                hidden=intervention_state,
                visual_hidden=visual_hidden,
                visual_positions=visual_positions,
                source_positions=source_positions,
                height=height,
                width=width,
                llm=llm,
                lm_head=lm_head,
                reference_logits=logits,
                predictor_positions=predictor_positions,
                targets=targets,
            )
        group_geometry: dict[str, Any] | None = None
        if self.run_group_geometry:
            if self.run_group_generation and not callable(sample.get("score_prediction")):
                raise BackendIntegrationError("Group-geometry generation requires the native task scorer.")
            group_geometry = self._run_group_geometry(
                sample_id=str(sample["sample_id"]), hidden=intervention_state,
                visual_positions=visual_positions, source_positions=source_positions,
                height=height, width=width, llm=llm, lm_head=lm_head,
                reference_logits=logits, predictor_positions=predictor_positions,
                targets=targets, sample=sample,
                prompt_length=int(objective["prefix_token_counts"][0]),
                run_generation=self.run_group_generation,
            )
        if self.run_causal:
            global_mean = visual_hidden.float().mean(dim=0).to(dtype=visual_hidden.dtype)
            selected_layer = next(
                item.layer for item in self.selected_layers if int(item.index) == self.intervention_layer
            )
            for kind in self.interventions:
                rows: list[dict[str, Any]] = []
                for unit_index, members in enumerate(partition.members[: self.max_intervention_units]):
                    replacement = (
                        torch.zeros_like(global_mean) if kind == "zero" else global_mean
                    )
                    unit_source_positions = [visual_positions[index] for index in members]

                    def replace_input(_module: Any, args: tuple[Any, ...], positions=unit_source_positions, value=replacement):
                        changed = args[0].clone()
                        changed[0, positions] = value.to(device=changed.device, dtype=changed.dtype)
                        return (changed, *args[1:])

                    candidate_inputs = dict(inputs)
                    candidate_inputs["output_attentions"] = False
                    candidate_inputs["output_hidden_states"] = False
                    with HookSession() as hooks:
                        hooks.register_forward_pre_hook(selected_layer, replace_input)
                        with torch.inference_mode():
                            candidate = self.runtime.model(**candidate_inputs)
                    metrics = answer_logit_metrics(
                        logits,
                        candidate.logits,
                        predictor_positions=predictor_positions,
                        targets=targets,
                    )
                    rows.append({"unit": unit_index, "members": list(members), **metrics})
                interventions[kind] = rows

        def delete_units(units: Sequence[int]) -> tuple[dict[str, Any], Any]:
            removed = {int(index) for unit in units for index in partition.members[unit]}
            kept_groups = tuple(
                (index,) for index in range(len(visual_positions)) if index not in removed
            )
            compact = _compact_layer_state(
                hidden=intervention_state, position_ids=source_positions,
                visual_positions=visual_positions, members=kept_groups,
                operator="retain", token_scores=torch.ones(len(visual_positions), device=intervention_state.device),
                budget=len(kept_groups),
            )
            if intervention_state.is_cuda:
                torch.cuda.synchronize(intervention_state.device)
            started = time.perf_counter()
            with torch.inference_mode():
                candidate_logits = _run_decoder_suffix(
                    llm=llm, lm_head=lm_head, hidden=compact["hidden"],
                    position_ids=compact["position_ids"], start_layer=self.intervention_layer,
                )
            if candidate_logits.is_cuda:
                torch.cuda.synchronize(candidate_logits.device)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            compact_predictors = [compact["source_to_compact"][position] for position in predictor_positions]
            metrics = compact_answer_logit_metrics(
                logits, candidate_logits, source_predictor_positions=predictor_positions,
                compact_predictor_positions=compact_predictors, targets=targets,
            )
            first_metrics = compact_answer_logit_metrics(
                logits, candidate_logits, source_predictor_positions=predictor_positions[:1],
                compact_predictor_positions=compact_predictors[:1], targets=targets[:1],
            )
            selected_logits = candidate_logits[0, compact_predictors].detach().float().clone()
            return {
                "deleted_visual_tokens": len(removed), "retained_visual_tokens": len(kept_groups),
                "deleted_visual_indices": sorted(removed),
                "sequence_length_after": int(compact["hidden"].shape[1]), "elapsed_ms": elapsed_ms,
                "first_answer_token_kl": first_metrics["teacher_to_candidate_answer_kl"],
                "first_answer_token_nll_delta": first_metrics["answer_nll_delta"],
                **metrics,
            }, selected_logits

        if self.run_unit_deletion:
            identity = _compact_layer_state(
                hidden=intervention_state,
                position_ids=source_positions,
                visual_positions=visual_positions,
                members=tuple((index,) for index in range(len(visual_positions))),
                operator="retain",
                token_scores=torch.ones(len(visual_positions), device=intervention_state.device),
                budget=len(visual_positions),
            )
            with torch.inference_mode():
                identity_logits = _run_decoder_suffix(
                    llm=llm,
                    lm_head=lm_head,
                    hidden=identity["hidden"],
                    position_ids=identity["position_ids"],
                    start_layer=self.intervention_layer,
                )
            predictor_index = torch.tensor(predictor_positions, device=logits.device)
            identity_max_abs = float(
                (
                    logits[0].index_select(0, predictor_index).float()
                    - identity_logits[0]
                    .index_select(0, predictor_index.to(identity_logits.device))
                    .float()
                )
                .abs()
                .max()
                .item()
            )
            identity_tolerance = float(self.config.get("identity_max_abs_logit_tolerance", 0.05))
            if identity_max_abs > identity_tolerance:
                raise BackendIntegrationError(
                    "Uncompressed suffix identity check exceeded tolerance: "
                    f"max_abs_logit_error={identity_max_abs:.6g}, tolerance={identity_tolerance:.6g}."
                )
            identity_check = {
                "max_abs_logit_error": identity_max_abs,
                "tolerance": identity_tolerance,
                "passed": True,
            }
            rows = []
            for unit_index, members in enumerate(partition.members):
                metrics, _ = delete_units((unit_index,))
                rows.append({"unit": unit_index, "members": list(members), **metrics})
            interventions["physical_delete"] = rows
        else:
            identity_check = None

        stability: dict[str, Any] = {"baseline_repeats": [], "single_unit_repeats": []}
        if self.stability_repeats:
            repeat_inputs = {**inputs, "output_attentions": False, "output_hidden_states": False}
            for repeat in range(self.stability_repeats):
                with torch.inference_mode():
                    repeated = self.runtime.model(**repeat_inputs)
                delta = float((logits[0, predictor_positions].float() - repeated.logits[0, predictor_positions].float()).abs().max().item())
                stability["baseline_repeats"].append({
                    "repeat_id": repeat + 1, "max_abs_logit_error": delta,
                    **answer_logit_metrics(logits, repeated.logits, predictor_positions=predictor_positions, targets=targets),
                })
            if self.run_unit_deletion:
                singletons = interventions["physical_delete"]
                effects = [item["answer_nll_delta"] for item in singletons]
                sentinels = sorted({round(i * (self.target_units - 1) / 7) for i in range(8)} | {
                    min(range(self.target_units), key=lambda i: effects[i]),
                    max(range(self.target_units), key=lambda i: effects[i]),
                })
                for unit in sentinels:
                    first_logits = None
                    for repeat in range(self.stability_repeats):
                        metrics, repeated_logits = delete_units((unit,))
                        if first_logits is None:
                            first_logits = repeated_logits
                        stability["single_unit_repeats"].append({
                            "unit": unit, "repeat_id": repeat + 1,
                            "max_abs_logit_error_from_repeat1": float((repeated_logits - first_logits).abs().max().item()),
                            "nll_delta_difference_from_initial": metrics["answer_nll_delta"] - effects[unit],
                            **metrics,
                        })

        joint_rows: list[dict[str, Any]] = []
        if self.run_joint_deletion:
            effects = [item["answer_nll_delta"] for item in interventions["physical_delete"]]
            plan = joint_deletion_plan(
                effects=effects, members=partition.members, scores=score_vectors,
                fractions=self.config.get("joint_deletion_fractions", [0.25, 0.5, 0.75]),
                seeds=self.random_seeds, sample_id=str(sample["sample_id"]),
                prompt_attention_key=f"text_visual_attention_layer_{self.intervention_layer}_prompt",
            )
            cache: dict[tuple[int, ...], dict[str, Any]] = {}
            for condition in plan:
                units = condition["units"]
                cache_hit = units in cache
                if not cache_hit:
                    cache[units], _ = delete_units(units)
                metrics = cache[units]
                additive = sum(effects[index] for index in units)
                joint_rows.append({
                    **condition, **metrics, "diagnostic_cache_hit": cache_hit,
                    "sum_singleton_nll_deltas": additive,
                    "max_selected_singleton_nll_delta": max(effects[index] for index in units),
                    "max_abs_selected_singleton_nll_delta": max(abs(effects[index]) for index in units),
                    "joint_minus_sum_singletons": metrics["answer_nll_delta"] - additive,
                    "oracle_ranking": condition["ranking"].startswith("singleton_oracle") or str(condition["matched_to"]).startswith("singleton_oracle"),
                    "position_policy": "retain original M-RoPE for all surviving visual and text tokens",
                })

        generation_rows: list[dict[str, Any]] = []
        if self.run_generation:
            from transformers import GenerationConfig

            prompt_length = int(objective["prefix_token_counts"][0])
            prompt_inputs = {
                name: _move_value(value, device=self.runtime.device, dtype=self.runtime.dtype)
                for name, value in sample["processor_outputs"].items()
            }
            if prompt_length != prompt_inputs["input_ids"].shape[1] or not torch.equal(
                inputs["input_ids"][:, :prompt_length], prompt_inputs["input_ids"]
            ):
                raise BackendIntegrationError("Gold-free prompt does not match the teacher-forced prefix.")
            prompt_hidden = intervention_state[:, :prompt_length]
            prompt_positions = source_positions[:, :prompt_length]
            eos_ids = self.runtime.model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids)
            max_new = int(self.config.get("generation_max_new_tokens", 24))
            tokenizer = self.runtime.processor.tokenizer

            def rollout(units: Sequence[int]) -> dict[str, Any]:
                removed = {index for unit in units for index in partition.members[unit]}
                compact = _compact_layer_state(
                    hidden=prompt_hidden, position_ids=prompt_positions,
                    visual_positions=visual_positions,
                    members=tuple((i,) for i in range(len(visual_positions)) if i not in removed),
                    operator="retain", token_scores=torch.ones(len(visual_positions), device=prompt_hidden.device),
                    budget=len(visual_positions) - len(removed),
                )
                decoded = greedy_decode_embeddings(
                    prompt_hidden=compact["hidden"], prompt_position_ids=compact["position_ids"],
                    token_embedding=self.runtime.model.get_input_embeddings(),
                    forward_logits=lambda hidden, positions: _run_decoder_suffix(
                        llm=llm, lm_head=lm_head, hidden=hidden, position_ids=positions,
                        start_layer=0, last_token_only=True,
                    ),
                    max_new_tokens=max_new, eos_token_ids=eos_ids,
                )
                text = tokenizer.decode(decoded["generated_token_ids"], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                stops = sample.get("generation_stops", [])
                stops = [stops] if isinstance(stops, str) else list(stops or [])
                for stop in stops:
                    if stop:
                        text = text.split(stop, 1)[0]
                text = text.strip()
                metrics = sample["score_prediction"](text)
                return {
                    **decoded, "prediction": text, "native_task_metrics": metrics,
                    "primary_task_score": primary_task_score(str(sample["task_name"]), metrics),
                    "deleted_visual_tokens": len(removed),
                    "retained_visual_tokens": len(visual_positions) - len(removed),
                    "units": list(units), "generation_stops": stops,
                }

            dense_generation = rollout(())
            if self.config.get("generation_native_check", True):
                native_config = GenerationConfig(
                    max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0,
                    eos_token_id=eos_ids, pad_token_id=tokenizer.pad_token_id,
                    bos_token_id=tokenizer.bos_token_id, use_cache=False,
                )
                with torch.inference_mode():
                    native_tokens = self.runtime.model.generate(**prompt_inputs, generation_config=native_config)
                native_ids = native_tokens[0, prompt_length:].tolist()
                if native_ids != dense_generation["generated_token_ids"]:
                    raise BackendIntegrationError(
                        f"Native/custom dense generation mismatch: native={native_ids}, custom={dense_generation['generated_token_ids']}."
                    )
                dense_generation["native_generation_identity_passed"] = True
            generation_rows.append({"ranking": "baseline", "fraction_units": 0.0, "selection_uses_answer": False, **dense_generation})
            generation_cache: dict[tuple[int, ...], dict[str, Any]] = {}
            for condition in joint_rows:
                ranking = condition["ranking"]
                if ranking == "singleton_oracle_high":
                    continue
                if ranking == "size_matched_random" and condition["matched_to"] != "singleton_oracle_abs_low":
                    continue
                units = tuple(condition["units"])
                if units not in generation_cache:
                    generation_cache[units] = rollout(units)
                generated = generation_cache[units]
                generation_rows.append({
                    "ranking": ranking, "fraction_units": condition["fraction_units"],
                    "random_seed": condition["random_seed"], "matched_to": condition["matched_to"],
                    "selection_uses_answer": ranking.startswith("singleton_oracle"),
                    "primary_task_score_delta": generated["primary_task_score"] - dense_generation["primary_task_score"],
                    "prediction_matches_baseline": generated["prediction"] == dense_generation["prediction"],
                    "answer_nll_delta": condition["answer_nll_delta"],
                    "teacher_to_candidate_answer_kl": condition["teacher_to_candidate_answer_kl"],
                    "first_answer_token_kl": condition["first_answer_token_kl"],
                    **generated,
                })

        region_map = None
        if self.config.get("save_region_maps", False) and self.run_unit_deletion:
            effects = [item["answer_nll_delta"] for item in interventions["physical_delete"]]
            filename = str(sample["sample_id"]).replace(":", "_").replace("/", "_")
            region_map = save_region_map(
                media=sample.get("media"), members=partition.members, effects=effects,
                height=height, width=width,
                path=str(Path(self.output_dir) / "region_maps" / f"{filename}.png"),
                title=f"{sample['sample_id']} | {sample.get('metadata', {}).get('prompt', '')} | target: {sample.get('target_text')}",
            )

        causal_vectors = {
            kind: [float(row["answer_nll_delta"]) for row in rows]
            for kind, rows in interventions.items()
            if len(rows) == self.target_units
        }
        correlations = {
            kind: {
                name: spearman_correlation(values, effects)
                for name, values in score_vectors.items()
            }
            for kind, effects in causal_vectors.items()
        }
        topk_curves = {
            kind: _additive_topk_curves(score_vectors, effects)
            for kind, effects in causal_vectors.items()
        }
        spatial = {}
        centroids = [item["centroid"] for item in partition_stats["clusters"]]
        for kind, effects in causal_vectors.items():
            selected = top_fraction_indices(effects, 0.25)
            spatial[kind] = {
                "causal_top_quartile_units": list(selected),
                **spatial_coverage(
                    selected,
                    centroids,
                    height=height,
                    width=width,
                    bins=int(self.config.get("spatial_bins", 4)),
                ),
            }

        frontier: list[dict[str, Any]] = []
        if self.run_frontier or self.run_layers:
            trajectory_layers = self.layer_indices if self.run_layers else (0,)
            trajectory_budgets = self.budgets if self.run_frontier else tuple(
                value for value in self.budgets if value in {64, 128}
            )
            for layer_index in trajectory_layers:
                layer_hidden = captured[f"layer_{layer_index}_input"]
                token_scores = _visual_attention_token_scores(
                    attentions[layer_index], visual_positions, predictor_positions
                )
                layer_features = layer_hidden[0].index_select(
                    0, torch.tensor(visual_positions, device=layer_hidden.device)
                )
                for budget in trajectory_budgets:
                    if budget > len(visual_positions):
                        continue
                    try:
                        layer_partition = feature_anchor_partition(
                            layer_features,
                            height=height,
                            width=width,
                            target_tokens=budget,
                        )
                    except ValueError:
                        continue
                    for operator in self.operators:
                        compact = _compact_layer_state(
                            hidden=layer_hidden,
                            position_ids=source_positions,
                            visual_positions=visual_positions,
                            members=layer_partition.members,
                            operator=operator,
                            token_scores=token_scores,
                            budget=budget,
                        )
                        start = time.perf_counter()
                        with torch.inference_mode():
                            compact_logits = _run_decoder_suffix(
                                llm=llm,
                                lm_head=lm_head,
                                hidden=compact["hidden"],
                                position_ids=compact["position_ids"],
                                start_layer=layer_index,
                            )
                        if torch.cuda.is_available():
                            torch.cuda.synchronize(compact_logits.device)
                        elapsed_ms = (time.perf_counter() - start) * 1000.0
                        compact_predictors = [
                            compact["source_to_compact"][position]
                            for position in predictor_positions
                        ]
                        metrics = compact_answer_logit_metrics(
                            logits,
                            compact_logits,
                            source_predictor_positions=predictor_positions,
                            compact_predictor_positions=compact_predictors,
                            targets=targets,
                        )
                        frontier.append(
                            {
                                "layer": layer_index,
                                "budget": budget,
                                "operator": operator,
                                "prefill_suffix_ms": elapsed_ms,
                                "sequence_length_before": int(layer_hidden.shape[1]),
                                "sequence_length_after": int(compact["hidden"].shape[1]),
                                **metrics,
                            }
                        )

        safe_by_layer_operator: dict[str, int | None] = {}
        for layer_index in self.layer_indices:
            for operator in self.operators:
                rows = [
                    row for row in frontier
                    if int(row["layer"]) == layer_index and row["operator"] == operator
                ]
                if rows:
                    safe_by_layer_operator[f"layer_{layer_index}:{operator}"] = safe_budget(
                        rows,
                        nll_relative_limit=float(self.config.get("safe_nll_relative_limit", 0.05)),
                        kl_limit=float(self.config.get("safe_kl_limit", 0.05)),
                    )

        return {
            "sample_id": sample["sample_id"],
            "task_name": sample["task_name"],
            "checkpoint": str(self.runtime.model_path_or_id or ""),
            "checkpoint_revision": self.runtime.revision,
            "implementation_hashes": self.implementation_hashes,
            "input_ids_sha256": hashlib.sha256(inputs["input_ids"].detach().cpu().numpy().tobytes()).hexdigest(),
            "original_position_ids_sha256": hashlib.sha256(source_positions.detach().cpu().numpy().tobytes()).hexdigest(),
            "image_fingerprint": media_fingerprint(sample.get("media"), sample.get("raw_doc")),
            "prompt": sample.get("metadata", {}).get("prompt"),
            "target_text": sample.get("target_text"),
            "target_source": sample.get("target_source"),
            "target_references": sample.get("target_references"),
            "objective": objective,
            "source_tokens": len(visual_positions),
            "grid_shape": [height, width],
            "target_units": self.target_units,
            "intervened_units": max((len(rows) for rows in interventions.values()), default=0),
            "intervention_units_by_kind": {kind: len(rows) for kind, rows in interventions.items()},
            "uncompressed_suffix_identity": identity_check,
            "stability": stability,
            "joint_deletion": joint_rows,
            "free_generation": generation_rows,
            "discovery_sampling": sample.get("metadata", {}).get("discovery_sampling", {}),
            "region_map": region_map,
            "baseline": baseline_metrics,
            "scores": score_vectors,
            "interventions": interventions,
            "pair_geometry_interventions": pair_geometry,
            "group_geometry_interventions": group_geometry,
            "score_causal_spearman": correlations,
            "topk_additive_effect_curves": topk_curves,
            "spatial_causal_coverage": spatial,
            "budget_layer_frontier": frontier,
            "safe_budget_by_layer_operator": safe_by_layer_operator,
            "partition_hash": partition.partition_hash,
            "cluster_statistics": partition_stats["clusters"],
        }

    def _run_pair_geometry(
        self,
        *,
        sample_id: str,
        hidden: Any,
        visual_hidden: Any,
        visual_positions: Sequence[int],
        source_positions: Any,
        height: int,
        width: int,
        llm: Any,
        lm_head: Any,
        reference_logits: Any,
        predictor_positions: Sequence[int],
        targets: Sequence[int],
    ) -> dict[str, Any]:
        """Counterfactually test feature collapse and spatial-position binding.

        One pair is chosen without labels from a feature-similarity × grid-distance
        quadrant. All conditions retain sequence length, attention mask, and text
        positions; only the selected pair's features and/or 3-axis M-RoPE positions
        change. This is a mechanism screen, not a compression method.
        """
        import torch
        import torch.nn.functional as functional

        count = int(visual_hidden.shape[0])
        if count < 2 or height * width != count:
            return {"status": "skipped", "reason": "invalid_visual_grid"}
        normalized = functional.normalize(visual_hidden.float(), dim=-1)
        similarities = normalized @ normalized.T
        row = torch.arange(count, device=visual_hidden.device)
        y = torch.div(row, width, rounding_mode="floor").float()
        x = (row % width).float()
        coords = torch.stack((y, x), dim=-1)
        distances = torch.cdist(coords, coords, p=1)
        upper = torch.triu(torch.ones_like(similarities, dtype=torch.bool), diagonal=1)
        pair_i, pair_j = torch.where(upper)
        pair_sim = similarities[pair_i, pair_j]
        pair_dist = distances[pair_i, pair_j]
        sim_low, sim_high = torch.quantile(pair_sim, torch.tensor([0.25, 0.75], device=pair_sim.device))
        dist_low, dist_high = torch.quantile(pair_dist, torch.tensor([0.25, 0.75], device=pair_dist.device))
        quadrant = str(int.from_bytes(hashlib.sha256(sample_id.encode("utf-8")).digest()[:2], "big") % 4)
        categories = {
            "0": ("similar_near", pair_sim >= sim_high, pair_dist <= dist_low),
            "1": ("similar_far", pair_sim >= sim_high, pair_dist >= dist_high),
            "2": ("dissimilar_near", pair_sim <= sim_low, pair_dist <= dist_low),
            "3": ("dissimilar_far", pair_sim <= sim_low, pair_dist >= dist_high),
        }
        category, sim_mask, dist_mask = categories[quadrant]
        eligible = torch.where(sim_mask & dist_mask)[0]
        if eligible.numel() == 0:
            return {
                "status": "skipped", "reason": "empty_similarity_distance_quadrant",
                "assigned_pair_category": category,
            }
        # Stable tie-break and selection, independent of answers and labels.
        eligible_pairs = torch.stack((pair_i[eligible], pair_j[eligible]), dim=-1).cpu().tolist()
        eligible_pairs.sort(key=lambda pair: (pair[0], pair[1]))
        choice_seed = hashlib.sha256((sample_id + ":pair-geometry-v1").encode("utf-8")).digest()
        left, right = eligible_pairs[int.from_bytes(choice_seed[:4], "big") % len(eligible_pairs)]
        left_pos, right_pos = int(visual_positions[left]), int(visual_positions[right])
        original_left = hidden[0, left_pos].float()
        original_right = hidden[0, right_pos].float()
        mean = (original_left + original_right) * 0.5
        mean_norm = mean.norm()
        if float(mean_norm.item()) <= 1e-12:
            return {"status": "skipped", "reason": "near_zero_pair_mean", "pair": [left, right]}
        normmatched_left = mean * (original_left.norm() / mean_norm)
        normmatched_right = mean * (original_right.norm() / mean_norm)
        centroid_position = (source_positions[:, left_pos].float() + source_positions[:, right_pos].float()) * 0.5

        def evaluate(name: str, replace_features: bool, replace_positions: bool, preserve_norm: bool = False) -> dict[str, Any]:
            candidate_hidden = hidden.clone()
            candidate_positions = source_positions
            if replace_features:
                candidate_hidden[0, left_pos] = (normmatched_left if preserve_norm else mean).to(candidate_hidden.dtype)
                candidate_hidden[0, right_pos] = (normmatched_right if preserve_norm else mean).to(candidate_hidden.dtype)
            if replace_positions:
                candidate_positions = source_positions.clone().float()
                candidate_positions[:, left_pos] = centroid_position
                candidate_positions[:, right_pos] = centroid_position
            if candidate_hidden.is_cuda:
                torch.cuda.synchronize(candidate_hidden.device)
            started = time.perf_counter()
            with torch.inference_mode():
                candidate_logits = _run_decoder_suffix(
                    llm=llm, lm_head=lm_head, hidden=candidate_hidden,
                    position_ids=(
                        candidate_positions.unsqueeze(1)
                        if candidate_positions.ndim == 2 else candidate_positions
                    ),
                    start_layer=self.intervention_layer,
                )
            if candidate_logits.is_cuda:
                torch.cuda.synchronize(candidate_logits.device)
            metrics = answer_logit_metrics(
                reference_logits, candidate_logits,
                predictor_positions=predictor_positions, targets=targets,
            )
            return {
                "condition": name, "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "sequence_length": int(candidate_hidden.shape[1]),
                "feature_changed": replace_features,
                "feature_norm_matched": bool(replace_features and preserve_norm),
                "visual_position_changed": replace_positions,
                "text_position_unchanged": True,
                **metrics,
            }

        conditions = [
            evaluate("feature_mean", True, False),
            evaluate("feature_normmatched_mean", True, False, preserve_norm=True),
            evaluate("position_centroid", False, True),
            evaluate("feature_normmatched_mean_plus_position_centroid", True, True, preserve_norm=True),
        ]
        return {
            "status": "ok",
            "pair_selection_uses_answer": False,
            "pair_category": category,
            "pair_visual_indices": [left, right],
            "pair_sequence_positions": [left_pos, right_pos],
            "pair_feature_cosine": float(similarities[left, right].item()),
            "pair_grid_manhattan_distance": float(distances[left, right].item()),
            "pair_original_feature_norms": [float(original_left.norm().item()), float(original_right.norm().item())],
            "pair_mean_norm": float(mean_norm.item()),
            "source_position_ids": [source_positions[:, left_pos].detach().cpu().tolist(), source_positions[:, right_pos].detach().cpu().tolist()],
            "centroid_position_ids": centroid_position.detach().cpu().tolist(),
            "text_position_ids_unchanged": True,
            "sequence_length_unchanged": True,
            "conditions": conditions,
        }

    def _run_group_geometry(
        self,
        *,
        sample_id: str,
        hidden: Any,
        visual_positions: Sequence[int],
        source_positions: Any,
        height: int,
        width: int,
        llm: Any,
        lm_head: Any,
        reference_logits: Any,
        predictor_positions: Sequence[int],
        targets: Sequence[int],
        sample: Mapping[str, Any],
        prompt_length: int,
        run_generation: bool,
    ) -> dict[str, Any]:
        """Test feature direction and content-position binding on 16-token groups."""
        import torch
        import torch.nn.functional as functional

        count = len(visual_positions)
        group_size = 16
        if height * width != count or count < 32:
            return {"status": "skipped", "reason": "requires_grid_and_at_least_32_visual_tokens"}
        # A 16-token contiguous core is placed inside a 20-token canvas. The
        # matched-random control is sampled from that same canvas and selected
        # without labels to match norm, feature cosine, and bounding-box span.
        canvas_shapes = [(4, 5), (5, 4), (2, 10), (10, 2)]
        feasible = [shape for shape in canvas_shapes if shape[0] <= height and shape[1] <= width]
        if not feasible:
            return {"status": "skipped", "reason": "no_20_token_canvas_fits_grid"}
        canvas_h, canvas_w = min(
            feasible,
            key=lambda shape: abs(math.log((shape[0] / shape[1]) / (height / width))),
        )
        core_h, core_w = min(canvas_h, 4), min(canvas_w, 4)
        starts = [(y, x) for y in range(height - canvas_h + 1) for x in range(width - canvas_w + 1)]
        start_seed = hashlib.sha256((sample_id + ":group-core-v1").encode("utf-8")).digest()
        canvas_y, canvas_x = starts[int.from_bytes(start_seed[:4], "big") % len(starts)]
        coherent = [
            (canvas_y + y) * width + canvas_x + x
            for y in range(core_h) for x in range(core_w)
        ]
        canvas = [
            (canvas_y + y) * width + canvas_x + x
            for y in range(canvas_h) for x in range(canvas_w)
        ]
        token_indices = torch.tensor(visual_positions, device=hidden.device, dtype=torch.long)
        features = hidden[0].index_select(0, token_indices).float()
        norms = features.norm(dim=-1)
        normalized = functional.normalize(features, dim=-1)
        all_sim = normalized @ normalized.T
        coords = torch.stack((torch.arange(count, device=hidden.device) // width,
                              torch.arange(count, device=hidden.device) % width), dim=-1).float()

        def stats(indices: Sequence[int]) -> dict[str, float]:
            idx = torch.tensor(indices, device=hidden.device, dtype=torch.long)
            pair = all_sim[idx][:, idx]
            pair_mask = ~torch.eye(len(indices), device=hidden.device, dtype=torch.bool)
            xy = coords[idx]
            span = xy.max(dim=0).values - xy.min(dim=0).values
            return {
                "norm_mean": float(norms[idx].mean().item()),
                "norm_std": float(norms[idx].std(unbiased=False).item()),
                "mean_pair_cosine": float(pair[pair_mask].mean().item()),
                "bbox_height": float((span[0] + 1).item()),
                "bbox_width": float((span[1] + 1).item()),
            }

        target_stats = stats(coherent)
        generator = random.Random("group-matched:" + sample_id)
        candidates: list[tuple[float, tuple[int, ...], dict[str, float]]] = []
        canvas_list = list(canvas)
        norm_scale = max(float(norms.std(unbiased=False).item()), 1e-6)
        for _ in range(2048):
            candidate = tuple(sorted(generator.sample(canvas_list, group_size)))
            if candidate == tuple(coherent):
                continue
            candidate_stats = stats(candidate)
            score = (
                abs(candidate_stats["norm_mean"] - target_stats["norm_mean"]) / norm_scale
                + abs(candidate_stats["norm_std"] - target_stats["norm_std"]) / norm_scale
                + abs(candidate_stats["mean_pair_cosine"] - target_stats["mean_pair_cosine"]) / 0.1
                + abs(candidate_stats["bbox_height"] - target_stats["bbox_height"])
                + abs(candidate_stats["bbox_width"] - target_stats["bbox_width"])
            )
            candidates.append((score, candidate, candidate_stats))
        if not candidates:
            return {"status": "skipped", "reason": "matched_random_sampling_failed"}
        candidates.sort(key=lambda item: (item[0], item[1]))
        match_score, matched_random, random_stats = candidates[0]

        dense_generation = None
        if run_generation:
            from mllm_comp.analysis.common.visual_token_generation import greedy_decode_embeddings, primary_task_score

            eos_ids = self.runtime.model.generation_config.eos_token_id
            eos_ids = [eos_ids] if isinstance(eos_ids, int) else list(eos_ids or [])
            max_new_tokens = int(self.config.get("group_generation_max_new_tokens", 24))
            tokenizer = self.runtime.processor.tokenizer

            def rollout(generation_hidden: Any, generation_positions: Any) -> dict[str, Any]:
                prefix_hidden = generation_hidden[:, :prompt_length]
                prefix_positions = generation_positions[:, :prompt_length]
                if prefix_positions.ndim == 2:
                    prefix_positions = prefix_positions.unsqueeze(1)
                decoded = greedy_decode_embeddings(
                    prompt_hidden=prefix_hidden,
                    prompt_position_ids=prefix_positions,
                    token_embedding=self.runtime.model.get_input_embeddings(),
                    forward_logits=lambda state, pos: _run_decoder_suffix(
                        llm=llm, lm_head=lm_head, hidden=state, position_ids=pos,
                        start_layer=self.intervention_layer, last_token_only=True,
                    ),
                    max_new_tokens=max_new_tokens,
                    eos_token_ids=eos_ids,
                )
                prediction = tokenizer.decode(
                    decoded["generated_token_ids"], skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                stops = sample.get("generation_stops", [])
                stops = [stops] if isinstance(stops, str) else list(stops or [])
                for stop in stops:
                    if stop:
                        prediction = prediction.split(stop, 1)[0]
                prediction = prediction.strip()
                task_metrics = sample["score_prediction"](prediction)
                return {
                    **decoded, "prediction": prediction,
                    "native_task_metrics": task_metrics,
                    "primary_task_score": primary_task_score(str(sample["task_name"]), task_metrics),
                }

            dense_generation = rollout(hidden, source_positions)

        def evaluate(group_name: str, members: Sequence[int], intervention: str) -> dict[str, Any]:
            positions = [int(visual_positions[index]) for index in members]
            candidate_hidden = hidden.clone()
            candidate_positions = source_positions
            if intervention in {"direction_collapse", "direction_collapse_plus_position_permutation"}:
                group = features[torch.tensor(members, device=features.device)]
                shared_direction = functional.normalize(group.mean(dim=0), dim=0, eps=1e-12)
                replacement = shared_direction.unsqueeze(0) * group.norm(dim=-1, keepdim=True)
                candidate_hidden[0, positions] = replacement.to(candidate_hidden.dtype)
            if intervention in {"position_permutation", "direction_collapse_plus_position_permutation"}:
                candidate_positions = source_positions.clone()
                old_positions = source_positions[:, positions].clone()
                order = list(range(len(positions)))
                permutation_rng = random.Random(f"{sample_id}:{group_name}:position-permutation-v1")
                permutation_rng.shuffle(order)
                if all(order[index] == index for index in range(len(order))):
                    order = order[1:] + order[:1]
                order_tensor = torch.tensor(order, device=source_positions.device, dtype=torch.long)
                candidate_positions[:, positions] = old_positions.index_select(1, order_tensor)
            if candidate_hidden.is_cuda:
                torch.cuda.synchronize(candidate_hidden.device)
            started = time.perf_counter()
            with torch.inference_mode():
                candidate_logits = _run_decoder_suffix(
                    llm=llm, lm_head=lm_head, hidden=candidate_hidden,
                    position_ids=(candidate_positions.unsqueeze(1) if candidate_positions.ndim == 2 else candidate_positions),
                    start_layer=self.intervention_layer,
                )
            if candidate_logits.is_cuda:
                torch.cuda.synchronize(candidate_logits.device)
            row = {
                "group": group_name, "intervention": intervention,
                "elapsed_ms": (time.perf_counter() - started) * 1000.0,
                "changed_visual_tokens": group_size,
                "sequence_length": int(candidate_hidden.shape[1]),
                "text_position_unchanged": True,
                **answer_logit_metrics(reference_logits, candidate_logits,
                    predictor_positions=predictor_positions, targets=targets),
            }
            if run_generation:
                generation = rollout(candidate_hidden, candidate_positions)
                row["free_generation"] = {
                    **generation,
                    "prediction_matches_dense": generation["prediction"] == dense_generation["prediction"],
                    "primary_task_score_delta": generation["primary_task_score"] - dense_generation["primary_task_score"],
                }
            return row

        conditions = [
            evaluate(group_name, members, intervention)
            for group_name, members in (("spatial_block", coherent), ("matched_random", matched_random))
            for intervention in ("direction_collapse", "position_permutation", "direction_collapse_plus_position_permutation")
        ]
        return {
            "status": "ok", "selection_uses_answer": False,
            "group_size": group_size, "canvas_shape": [canvas_h, canvas_w],
            "spatial_block_indices": coherent,
            "matched_random_indices": list(matched_random),
            "matched_random_selection_score": float(match_score),
            "spatial_block_statistics": target_stats,
            "matched_random_statistics": random_stats,
            "position_permutation_preserves_position_multiset": True,
            "text_positions_unchanged": True,
            "sequence_length_unchanged": True,
            "conditions": conditions,
            "dense_free_generation": dense_generation,
        }

    def _score_vectors(
        self,
        *,
        baseline: Any,
        hidden_states: Sequence[Any],
        attentions: Sequence[Any],
        visual_positions: Sequence[int],
        predictor_positions: Sequence[int],
        prompt_query_positions: Sequence[int],
        members: Sequence[Sequence[int]],
        visual_features: Any,
        vision_attention_scores: Any,
    ) -> dict[str, list[float]]:
        import torch
        import torch.nn.functional as functional

        visual_index = torch.tensor(visual_positions, device=visual_features.device)
        normalized = functional.normalize(visual_features.float(), dim=-1)
        similarity = normalized @ normalized.transpose(0, 1)
        similarity.fill_diagonal_(-1.0)
        uniqueness = 1.0 - similarity.max(dim=-1).values
        center = visual_features.float().mean(dim=0)
        diversity = 1.0 - functional.cosine_similarity(
            visual_features.float(), center.unsqueeze(0), dim=-1
        )
        layer_zero = hidden_states[0][0].index_select(0, visual_index.to(hidden_states[0].device)).float()
        token_scores: dict[str, Any] = {
            "feature_norm": layer_zero.norm(dim=-1),
            "feature_diversity": diversity,
            "feature_uniqueness": uniqueness,
            "vision_attention_received": vision_attention_scores,
        }
        for layer_index in self.layer_indices:
            token_scores[f"text_visual_attention_layer_{layer_index}_prompt"] = _visual_attention_token_scores(
                attentions[layer_index], visual_positions, prompt_query_positions
            )
            token_scores[f"text_visual_attention_layer_{layer_index}_gold_context"] = _visual_attention_token_scores(
                attentions[layer_index], visual_positions, predictor_positions
            )
            before = hidden_states[layer_index][0].index_select(
                0, visual_index.to(hidden_states[layer_index].device)
            ).float()
            after = hidden_states[layer_index + 1][0].index_select(
                0, visual_index.to(hidden_states[layer_index + 1].device)
            ).float()
            token_scores[f"transition_layer_{layer_index}"] = (
                (after - before).norm(dim=-1) / before.norm(dim=-1).clamp_min(1e-12)
            )
        return {
            name: [
                float(values[list(cluster)].float().mean().item())
                for cluster in members
            ]
            for name, values in token_scores.items()
        }

    def finish(self) -> AnalysisRecord:
        if self._finished:
            raise BackendIntegrationError("Visual token phenomena session already finished.")
        self._finished = True
        records = list(self.spool)
        successful = [record for record in records if record.get("status") == "ok"]
        if not successful:
            reasons = "; ".join(item.get("reason", "") for item in records[:3])
            raise BackendIntegrationError("Visual token phenomena produced no successful samples. " + reasons)
        aggregate = _aggregate_records(successful, self.config)
        jsonl_path = Path(self.output_dir) / "visual_token_phenomena.samples.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as output:
            for record in records:
                output.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.spool.close()
        aggregate_path = Path(self.output_dir) / "visual_token_phenomena.aggregate.json"
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
        return AnalysisRecord(
            name=str(self.config.get("name", self.plugin.name)),
            phase=self.phase,
            method=self.plugin.name,
            summary={
                "successful_samples": len(successful),
                "skipped_samples": len(records) - len(successful),
                "unselected_samples": self.unselected_samples,
                "target_units": self.target_units,
                "layers": list(self.layer_indices),
                "budgets": list(self.budgets),
            },
            artifact_paths={"aggregate": str(aggregate_path), "per_sample": str(jsonl_path)},
            sample_count=len(successful),
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self._finished:
            self.spool.close()


def _positive_int_sequence(value: Any, field: str) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"visual token phenomena {field} must be a non-empty integer sequence.")
    result = tuple(int(item) for item in value)
    if any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in value):
        raise ValueError(f"visual token phenomena {field} must contain positive integers.")
    return result


def _visual_positions(input_ids: Any, image_token_id: int) -> list[int]:
    return [
        index for index, token in enumerate(input_ids[0].tolist())
        if int(token) == int(image_token_id)
    ]


def _visual_attention_token_scores(attention: Any, visual_positions: Sequence[int], query_positions: Sequence[int]) -> Any:
    import torch
    visual_index = torch.tensor(visual_positions, device=attention.device, dtype=torch.long)
    query_index = torch.tensor(query_positions, device=attention.device, dtype=torch.long)
    selected = attention[0].index_select(1, query_index).index_select(2, visual_index)
    return selected.float().mean(dim=(0, 1))


def _vision_attention_received_scores(
    *,
    attention: Any,
    hidden: Any,
    position_embeddings: Any,
    cu_seqlens: Any,
    reverse_window_index: Any,
    merge_unit: int,
) -> Any:
    """Average key-received attention from Qwen's final full-attention vision block.

    This is a descriptive ranking, not an importance label. The final vision block
    is full-attention in Qwen2.5-VL; cu_seqlens is still honored so the helper does
    not silently mix independent packed images if the backbone changes later.
    Scores are reduced from patch tokens to post-merger tokens and restored from
    window order to the original spatial order.
    """

    import torch
    import torch.nn.functional as functional

    if hidden is None or position_embeddings is None or len(position_embeddings) != 2:
        raise BackendIntegrationError("Final vision attention did not expose hidden states and rotary embeddings.")
    if cu_seqlens is None:
        raise BackendIntegrationError("Final vision attention did not expose cu_seqlens.")
    sequence_length = int(hidden.shape[0])
    cos, sin = (value.float() for value in position_embeddings)
    # The hook captured an inference tensor. Project it under inference_mode as
    # Linear's backward-save path rejects inference tensors even when the result
    # is only consumed by this read-only diagnostic.
    with torch.inference_mode():
        qkv = attention.qkv(hidden).reshape(sequence_length, 3, attention.num_heads, attention.head_dim)
    query, key = qkv[:, 0], qkv[:, 1]

    def rotate_half(value: Any) -> Any:
        first, second = value.chunk(2, dim=-1)
        return torch.cat((-second, first), dim=-1)

    cos = cos.unsqueeze(-2)
    sin = sin.unsqueeze(-2)
    query = query.float() * cos + rotate_half(query.float()) * sin
    key = key.float() * cos + rotate_half(key.float()) * sin
    boundaries = [int(value) for value in cu_seqlens.tolist()]
    received = torch.zeros(sequence_length, device=hidden.device, dtype=torch.float32)
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start:
            continue
        q = query[start:end].transpose(0, 1)
        k = key[start:end].transpose(0, 1)
        weights = torch.matmul(q, k.transpose(-1, -2)) * float(attention.scaling)
        weights = functional.softmax(weights, dim=-1, dtype=torch.float32)
        received[start:end] = weights.mean(dim=(0, 1))
    if sequence_length % merge_unit:
        raise BackendIntegrationError("Vision patch count is not divisible by the spatial merge unit.")
    post_merger = received.reshape(-1, merge_unit).mean(dim=-1)
    if post_merger.numel() != reverse_window_index.numel():
        raise BackendIntegrationError("Vision-attention score count does not match post-merger token count.")
    return post_merger.index_select(0, reverse_window_index.to(post_merger.device)).detach()


def _additive_topk_curves(
    scores: Mapping[str, Sequence[float]],
    causal_effects: Sequence[float],
) -> dict[str, list[dict[str, float]]]:
    """Screen rankings using additive sums of individually measured effects."""
    result: dict[str, list[dict[str, float]]] = {}
    for name, values in scores.items():
        rows = []
        for fraction in (0.1, 0.25, 0.5):
            selected = top_fraction_indices(values, fraction)
            rows.append(
                {
                    "fraction": fraction,
                    "unit_count": len(selected),
                    "additive_answer_nll_delta": sum(float(causal_effects[index]) for index in selected),
                }
            )
        result[name] = rows
    return result


def _compact_layer_state(
    *,
    hidden: Any,
    position_ids: Any,
    visual_positions: Sequence[int],
    members: Sequence[Sequence[int]],
    operator: str,
    token_scores: Any,
    budget: int,
) -> dict[str, Any]:
    import torch

    if operator == "retain":
        groups = tuple(tuple(int(value) for value in group) for group in members)
        kept_visual = [group[0] for group in groups if len(group) == 1]
        if len(kept_visual) != len(groups) or len(set(kept_visual)) != len(kept_visual) or any(
            index < 0 or index >= len(visual_positions) for index in kept_visual
        ):
            raise ValueError("retain requires distinct, in-range singleton visual groups.")
        kept_visual_set = set(kept_visual)
        removed_source = {
            position for index, position in enumerate(visual_positions) if index not in kept_visual_set
        }
        kept_source = [position for position in range(int(hidden.shape[1])) if position not in removed_source]
        index = torch.tensor(kept_source, device=hidden.device, dtype=torch.long)
        return {
            "hidden": hidden.index_select(1, index),
            "position_ids": position_ids.index_select(1, index.to(position_ids.device)).unsqueeze(1).long(),
            "source_to_compact": {source: compact for compact, source in enumerate(kept_source)},
        }
    visual_to_source = {position: index for index, position in enumerate(visual_positions)}
    if operator == "prune":
        kept = sorted(
            sorted(range(len(visual_positions)), key=lambda index: (-float(token_scores[index]), index))[:budget]
        )
        groups = tuple((index,) for index in kept)
    else:
        groups = tuple(tuple(int(value) for value in group) for group in members)
    starts = {min(group): group for group in groups}
    outputs: list[Any] = []
    positions: list[Any] = []
    source_to_compact: dict[int, int] = {}
    for source_position in range(int(hidden.shape[1])):
        visual_index = visual_to_source.get(source_position)
        if visual_index is None:
            source_to_compact[source_position] = len(outputs)
            outputs.append(hidden[0, source_position])
            positions.append(position_ids[:, source_position])
            continue
        group = starts.get(visual_index)
        if group is None:
            continue
        source_indices = torch.tensor(
            [visual_positions[index] for index in group], device=hidden.device, dtype=torch.long
        )
        values = hidden[0].index_select(0, source_indices)
        weights = token_scores[list(group)].float().clamp_min(0)
        if operator in {"prune", "retain"}:
            merged = values[0]
            centroid = position_ids[:, visual_positions[group[0]]]
        elif operator == "representative":
            representative = max(group, key=lambda index: (float(token_scores[index]), -index))
            merged = hidden[0, visual_positions[representative]]
            centroid = position_ids[:, visual_positions[representative]]
        elif operator == "cluster_mean":
            merged = values.float().mean(dim=0).to(dtype=values.dtype)
            centroid = position_ids[:, source_indices].float().mean(dim=1).round().long()
        else:
            if float(weights.sum().item()) <= 0:
                weights = torch.ones_like(weights)
            normalized = weights / weights.sum()
            weighted = (values.float() * normalized[:, None]).sum(dim=0)
            if operator == "attention_weighted_merge":
                merged = weighted.to(dtype=values.dtype)
            elif operator == "residual_absorption":
                representative_local = int(weights.argmax().item())
                anchor = values[representative_local].float()
                residual = weighted - anchor
                merged = (anchor + residual).to(dtype=values.dtype)
            else:
                raise ValueError(f"Unknown compression operator {operator!r}.")
            centroid = (
                position_ids[:, source_indices].float() * normalized.to(position_ids.device)[None, :]
            ).sum(dim=1).round().long()
        source_to_compact[source_position] = len(outputs)
        outputs.append(merged)
        positions.append(centroid)
    return {
        "hidden": torch.stack(outputs).unsqueeze(0),
        "position_ids": torch.stack(positions, dim=1).unsqueeze(1).long(),
        "source_to_compact": source_to_compact,
    }


def _run_decoder_suffix(*, llm: Any, lm_head: Any, hidden: Any, position_ids: Any, start_layer: int, last_token_only: bool = False) -> Any:
    import torch
    from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask

    sequence_length = int(hidden.shape[1])
    attention_mask = torch.ones((1, sequence_length), device=hidden.device, dtype=torch.long)
    cache_position = torch.arange(sequence_length, device=hidden.device, dtype=torch.long)
    mask_kwargs = {
        "config": llm.config,
        "input_embeds": hidden,
        "attention_mask": attention_mask,
        "cache_position": cache_position,
        "past_key_values": None,
        "position_ids": None,
    }
    masks = {"full_attention": create_causal_mask(**mask_kwargs)}
    if getattr(llm, "has_sliding_layers", False):
        masks["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
    position_embeddings = llm.rotary_emb(hidden, position_ids)
    state = hidden
    for layer in llm.layers[start_layer:]:
        output = layer(
            state,
            attention_mask=masks[layer.attention_type],
            position_ids=None,
            past_key_values=None,
            output_attentions=False,
            use_cache=False,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )
        state = output[0]
    normalized = llm.norm(state)
    return lm_head(normalized[:, -1:] if last_token_only else normalized)


def _bootstrap_mean(values: Sequence[float], *, seed: int = 42, samples: int = 1000) -> dict[str, float] | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(samples):
        means.append(sum(rng.choice(finite) for _ in finite) / len(finite))
    means.sort()
    return {
        "mean": sum(finite) / len(finite),
        "ci95_low": means[int(0.025 * (samples - 1))],
        "ci95_high": means[int(0.975 * (samples - 1))],
        "sample_count": len(finite),
    }


def _aggregate_records(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for record in records:
        by_task.setdefault(str(record["task_name"]), []).append(record)
    correlations: dict[str, Any] = {}
    for task, task_records in by_task.items():
        state: dict[str, list[float]] = {}
        for record in task_records:
            for intervention, scores in record.get("score_causal_spearman", {}).items():
                for score, value in scores.items():
                    if value is not None:
                        state.setdefault(f"{intervention}:{score}", []).append(float(value))
        correlations[task] = {
            name: _bootstrap_mean(values) for name, values in sorted(state.items())
        }

    image_groups: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for record in records:
        image_groups.setdefault(
            (str(record["task_name"]), str(record["image_fingerprint"])), []
        ).append(record)
    query_dependence: dict[str, list[dict[str, Any]]] = {}
    for (task, fingerprint), group in image_groups.items():
        if len(group) < 2:
            continue
        pair_rows = []
        for left_index in range(len(group)):
            for right_index in range(left_index + 1, len(group)):
                left = group[left_index]
                right = group[right_index]
                for kind in set(left.get("interventions", {})) & set(right.get("interventions", {})):
                    left_values = [float(row["answer_nll_delta"]) for row in left["interventions"][kind]]
                    right_values = [float(row["answer_nll_delta"]) for row in right["interventions"][kind]]
                    if len(left_values) != len(right_values) or not left_values:
                        continue
                    pair_rows.append(
                        {
                            "intervention": kind,
                            "left_sample": left["sample_id"],
                            "right_sample": right["sample_id"],
                            "top_quartile_jaccard": jaccard(
                                top_fraction_indices(left_values, 0.25),
                                top_fraction_indices(right_values, 0.25),
                            ),
                            "rank_spearman": spearman_correlation(left_values, right_values),
                        }
                    )
        if pair_rows:
            query_dependence.setdefault(task, []).append(
                {"image_fingerprint": fingerprint, "question_count": len(group), "pairs": pair_rows}
            )

    safe_counts: dict[str, dict[str, int]] = {}
    for record in records:
        task = str(record["task_name"])
        task_counts = safe_counts.setdefault(task, {})
        for key, value in record.get("safe_budget_by_layer_operator", {}).items():
            label = f"{key}:K{value}" if value is not None else f"{key}:none"
            task_counts[label] = task_counts.get(label, 0) + 1
    return {
        "successful_samples": len(records),
        "task_sample_counts": {task: len(values) for task, values in by_task.items()},
        "score_causal_spearman_bootstrap": correlations,
        "query_dependence": query_dependence,
        "same_image_groups": sum(len(values) for values in query_dependence.values()),
        "unique_images_by_task": {task: len({row["image_fingerprint"] for row in values}) for task, values in by_task.items()},
        "joint_deletion_conditions": sum(len(row.get("joint_deletion", ())) for row in records),
        "stability_max_baseline_logit_error": max((value["max_abs_logit_error"] for row in records for value in row.get("stability", {}).get("baseline_repeats", ())), default=None),
        "stability_max_singleton_logit_error": max((value["max_abs_logit_error_from_repeat1"] for row in records for value in row.get("stability", {}).get("single_unit_repeats", ())), default=None),
        "safe_budget_counts": safe_counts,
        "decision_thresholds": {
            "safe_nll_relative_limit": float(config.get("safe_nll_relative_limit", 0.05)),
            "safe_kl_limit": float(config.get("safe_kl_limit", 0.05)),
        },
        "methodological_notes": {
            "query_dependence": "Exploratory overlap only; requires same-question, paraphrase, and signal-size controls. No fixed Jaccard decision thresholds.",
            "score_correlation_bootstrap": "Legacy per-question exploratory intervals; scientific comparisons must resample independent images.",
            "joint_deletion": "Actual compact suffix reruns; random controls exactly match the removed unit-size histogram. Singleton rankings use gold-answer outcomes and are diagnostic oracles.",
            "interaction": "Joint loss minus summed singleton losses measures nonadditivity of the loss, not by itself semantic redundancy or complementarity.",
            "stability": "Identical-condition BF16 repeats measure repeatability, not FP32 accuracy or paraphrase reliability.",
            "topk_curves": "additive screening curves from individual interventions; not joint-ablation task curves",
            "drop_intervention": "zero hidden-state ablation with sequence length and M-RoPE held fixed",
            "global_mean_intervention": "selected unit replaced by the image-wide visual mean",
            "residual_absorption": "anchor plus its full attention-weighted residual; algebraically matches attention-weighted merge",
            "primary_metrics": ["answer_nll", "teacher_logit_kl", "answer_flip"],
            "attention_l2_primary": False,
        },
    }


__all__ = ["VisualTokenPhenomena"]
