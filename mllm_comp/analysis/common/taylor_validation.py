"""Causal Mask/Ablation validation for hierarchical Taylor rankings."""

from __future__ import annotations

import math
import random
from typing import Any, Callable, Mapping, Sequence

from mllm_comp.analysis.bindings import AnalysisBackboneBinding
from mllm_comp.analysis.common.interventions import (
    install_attention_edge_block,
    install_attention_head_ablation,
    install_ffn_neuron_ablation,
    install_layer_update_skip,
    install_sublayer_update_skip,
)
from mllm_comp.analysis.common.token_mask import TokenGroupLayout
from mllm_comp.analysis.common.taylor import taylor_path_groups
from mllm_comp.analysis.hooks import HookSession
from mllm_comp.core.errors import BackendIntegrationError


VALIDATION_KINDS = ("path", "head", "ffn_neuron", "sublayer", "layer")
def validate_causal_validation_config(config: Mapping[str, Any]) -> dict[str, Any]:
    raw = config.get("causal_validation", {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise ValueError("hierarchical Taylor causal_validation must be a mapping.")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError("causal_validation.enabled must be a boolean.")
    count = raw.get("component_count", 1)
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("causal_validation.component_count must be positive.")
    kinds_raw = raw.get("kinds", VALIDATION_KINDS)
    if not isinstance(kinds_raw, (list, tuple)) or not kinds_raw:
        raise ValueError("causal_validation.kinds must be a non-empty list.")
    if any(not isinstance(value, str) for value in kinds_raw):
        raise ValueError("causal_validation.kinds entries must be strings.")
    kinds = tuple(dict.fromkeys(kinds_raw))
    unknown = sorted(set(kinds) - set(VALIDATION_KINDS))
    if unknown:
        raise ValueError(f"Unsupported causal-validation kinds: {unknown}")
    seed = raw.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("causal_validation.seed must be an integer.")
    return {
        "enabled": enabled,
        "component_count": count,
        "kinds": kinds,
        "seed": seed,
    }


def select_causal_validation_candidates(
    report: Mapping[str, Any],
    *,
    kinds: Sequence[str],
    component_count: int,
    seed: int,
    sample_id: str,
) -> dict[str, Any]:
    """Select low-ranked components and same-kind deterministic random controls."""

    result: dict[str, Any] = {}
    for kind in kinds:
        candidates = _candidates(report, kind)
        selected_count = min(component_count, len(candidates))
        if selected_count <= 0:
            result[kind] = {
                "status": "unavailable",
                "reason": "no_finite_candidates",
            }
            continue
        ordered = sorted(candidates, key=lambda item: (item["score"], item["id"]))
        selected = ordered[:selected_count]
        selected_ids = {item["id"] for item in selected}
        pool = [item for item in candidates if item["id"] not in selected_ids]
        if len(pool) < selected_count:
            result[kind] = {
                "status": "unavailable",
                "reason": "insufficient_disjoint_random_controls",
                "candidate_count": len(candidates),
                "selected": selected,
            }
            continue
        rng = random.Random(f"{seed}:{sample_id}:{kind}:matched_control_v1")
        random_control = sorted(
            rng.sample(pool, selected_count),
            key=lambda item: item["id"],
        )
        result[kind] = {
            "status": "ready",
            "candidate_count": len(candidates),
            "selected": selected,
            "random_control": random_control,
        }
    return result


def run_causal_validation(
    *,
    binding: AnalysisBackboneBinding,
    layout: TokenGroupLayout,
    state: Any,
    forward: Callable[[], Any],
    baseline_loss: float,
    selections: Mapping[str, Any],
    target_modality: str = "vision",
) -> dict[str, Any]:
    """Measure native teacher-forced loss deltas under actual interventions."""

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - real runtime requires torch
        raise BackendIntegrationError("Taylor causal validation requires PyTorch.") from exc
    native_inputs = getattr(state, "native_model_inputs", None)
    if not isinstance(native_inputs, dict):
        raise BackendIntegrationError(
            "Taylor causal validation requires mutable native model inputs."
        )
    original_output_attentions = native_inputs.get("output_attentions")
    native_inputs["output_attentions"] = False
    results: dict[str, Any] = {}
    try:
        for kind, selection in selections.items():
            if selection.get("status") != "ready":
                results[kind] = dict(selection)
                continue
            arms: dict[str, Any] = {}
            for arm in ("selected", "random_control"):
                candidates = selection[arm]
                with HookSession() as hooks:
                    intervention_evidence = _install_candidates(
                        hooks,
                        binding=binding,
                        layout=layout,
                        candidates=candidates,
                        target_modality=target_modality,
                    )
                    with torch.inference_mode():
                        output = forward()
                    loss = _loss_value(output)
                    del output
                empty_path_masks = [
                    evidence
                    for evidence in intervention_evidence
                    if evidence.get("kind") == "path"
                    and evidence.get("newly_blocked_edge_count") == 0
                ]
                arms[arm] = {
                    "status": (
                        "unavailable" if empty_path_masks else "finite"
                    ),
                    "components": candidates,
                    "loss": loss,
                    "loss_delta": loss - baseline_loss,
                    "absolute_loss_delta": abs(loss - baseline_loss),
                    "intervention_evidence": intervention_evidence,
                    "reason": (
                        "no_safely_blockable_causally_visible_edge"
                        if empty_path_masks
                        else None
                    ),
                }
            if any(arms[arm]["status"] != "finite" for arm in arms):
                results[kind] = {
                    "status": "unavailable",
                    "reason": "matched_intervention_arm_was_not_causally_realized",
                    "candidate_count": selection["candidate_count"],
                    "component_count": len(selection["selected"]),
                    "selected": arms["selected"],
                    "random_control": arms["random_control"],
                }
                continue
            results[kind] = {
                "status": "finite",
                "intervention": _intervention_name(kind),
                "candidate_count": selection["candidate_count"],
                "component_count": len(selection["selected"]),
                "selected": arms["selected"],
                "random_control": arms["random_control"],
            }
    finally:
        native_inputs["output_attentions"] = original_output_attentions
    return {
        "objective": "teacher_forced_answer_cross_entropy_v1",
        "metric": "native_loss_delta_v1",
        "baseline_loss": float(baseline_loss),
        "selection": "lowest_normalized_taylor_with_disjoint_random_control_v1",
        "kinds": results,
    }


def _candidates(report: Mapping[str, Any], kind: str) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for layer_key, layer in report.get("layers", {}).items():
        layer_index = int(layer_key)
        if kind == "path":
            for path, values in layer.get("path", {}).items():
                for index, score in enumerate(values.get("normalized_score", ()) or ()):
                    _append_candidate(
                        candidates,
                        kind=kind,
                        layer=layer_index,
                        index=index,
                        subtype=str(path),
                        score=score,
                    )
        elif kind == "head":
            values = layer.get("head", {}).get("all", {})
            for index, score in enumerate(values.get("normalized_score", ()) or ()):
                _append_candidate(
                    candidates,
                    kind=kind,
                    layer=layer_index,
                    index=index,
                    score=score,
                )
        elif kind == "ffn_neuron":
            values = (
                layer.get("ffn_neuron", {}).get("taylor", {}).get("all", {})
            )
            for index, score in enumerate(values.get("normalized_score", ()) or ()):
                _append_candidate(
                    candidates,
                    kind=kind,
                    layer=layer_index,
                    index=index,
                    score=score,
                )
        elif kind == "sublayer":
            for subtype, groups in layer.get("sublayer", {}).items():
                score = groups.get("all", {}).get("normalized_score")
                _append_candidate(
                    candidates,
                    kind=kind,
                    layer=layer_index,
                    subtype=str(subtype),
                    score=score,
                )
        elif kind == "layer":
            score = layer.get("layer", {}).get("all", {}).get("normalized_score")
            _append_candidate(
                candidates,
                kind=kind,
                layer=layer_index,
                score=score,
            )
        else:
            raise ValueError(f"Unsupported causal-validation kind {kind!r}.")
    return candidates


def _append_candidate(
    values: list[dict[str, Any]],
    *,
    kind: str,
    layer: int,
    score: Any,
    index: int | None = None,
    subtype: str | None = None,
) -> None:
    if score is None or not math.isfinite(float(score)):
        return
    identity = f"layer:{layer}/{kind}"
    if subtype is not None:
        identity += f":{subtype}"
    if index is not None:
        identity += f":{index}"
    values.append(
        {
            "id": identity,
            "kind": kind,
            "layer": int(layer),
            "index": index,
            "subtype": subtype,
            "score": float(score),
        }
    )


def _install_candidates(
    hooks: HookSession,
    *,
    binding: AnalysisBackboneBinding,
    layout: TokenGroupLayout,
    candidates: Sequence[Mapping[str, Any]],
    target_modality: str,
) -> list[dict[str, Any]]:
    layers = {layer.index: layer for layer in binding.layers}
    grouped: dict[tuple[str, int, str | None], list[int | None]] = {}
    for candidate in candidates:
        key = (
            str(candidate["kind"]),
            int(candidate["layer"]),
            str(candidate["subtype"])
            if candidate.get("subtype") is not None
            else None,
        )
        grouped.setdefault(key, []).append(
            int(candidate["index"]) if candidate.get("index") is not None else None
        )
    evidence_items: list[dict[str, Any]] = []
    for (kind, layer_index, subtype), indices in grouped.items():
        layer = layers[layer_index]
        integer_indices = [int(index) for index in indices if index is not None]
        if kind == "path":
            assert subtype is not None
            try:
                target_group, source_group = taylor_path_groups(
                    target_modality
                )[subtype]
            except KeyError as exc:
                raise BackendIntegrationError(
                    f"Unknown {target_modality} Taylor path {subtype!r}."
                ) from exc
            complement = f"non{target_modality}"
            if target_group == "text":
                target_group = complement
            if source_group == "text":
                source_group = complement
            evidence: dict[str, Any] = {
                "kind": "path",
                "layer": layer_index,
                "subtype": subtype,
            }
            install_attention_edge_block(
                hooks,
                attention=layer.attention,
                layout=layout,
                source_group=source_group,
                target_group=target_group,
                head_indices=integer_indices,
                num_heads=binding.attention_geometry.num_query_heads,
                evidence=evidence,
            )
            evidence_items.append(evidence)
        elif kind == "head":
            install_attention_head_ablation(
                hooks,
                output_projection=layer.attention_output_projection,
                head_indices=integer_indices,
                num_heads=binding.attention_geometry.num_query_heads,
                head_dim=binding.attention_geometry.head_dim,
            )
        elif kind == "ffn_neuron":
            install_ffn_neuron_ablation(
                hooks,
                down_projection=layer.ffn_down_projection,
                neuron_indices=integer_indices,
            )
        elif kind == "sublayer":
            target = layer.attention if subtype == "attention" else layer.ffn
            install_sublayer_update_skip(
                hooks,
                sublayer=target,
                layout=layout,
                target_group="all",
            )
        elif kind == "layer":
            install_layer_update_skip(
                hooks,
                layer=layer.layer,
                layout=layout,
                target_group="all",
            )
        else:  # pragma: no cover - selection validator owns this vocabulary
            raise BackendIntegrationError(
                f"Unsupported causal-validation candidate kind {kind!r}."
            )
    return evidence_items


def _loss_value(output: Any) -> float:
    loss = getattr(output, "loss", None)
    if loss is None and isinstance(output, Mapping):
        loss = output.get("loss")
    if loss is None or getattr(loss, "ndim", None) != 0:
        raise BackendIntegrationError(
            "Causal Taylor intervention returned no scalar native loss."
        )
    value = float(loss.detach().cpu().item())
    if not math.isfinite(value):
        raise BackendIntegrationError(
            "Causal Taylor intervention produced a non-finite loss."
        )
    return value


def _intervention_name(kind: str) -> str:
    return {
        "path": "per_head_safe_partial_4d_attention_edge_mask_v2",
        "head": "pre_o_projection_head_zero_ablation_v1",
        "ffn_neuron": "pre_down_projection_neuron_zero_ablation_v1",
        "sublayer": "residual_update_zero_ablation_v1",
        "layer": "decoder_layer_identity_ablation_v1",
    }[kind]


__all__ = [
    "VALIDATION_KINDS",
    "run_causal_validation",
    "select_causal_validation_candidates",
    "validate_causal_validation_config",
]
