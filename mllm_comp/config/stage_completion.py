"""Complete unambiguous v2 requests from the selected backbone's declarations.

This runs before the existing typed Stage parser. It only supplies omitted
identity and modality fields; algorithm parameters and explicit choices retain
their existing validation path.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

from mllm_comp.core.errors import ConfigError, RegistryError
from mllm_comp.core.vocabulary import CompressionMethod, Modality

if TYPE_CHECKING:
    from mllm_comp.compressors.contracts import (
        BackbonePlacementBinding,
        CompressionContractRegistry,
    )


_COMPLETE_FIELDS = frozenset({
    "algorithm", "variant", "binding", "plugin", "placement", "execution_phase",
    "method", "object_family", "target_modalities", "conditioning_modalities",
    "preserved_modalities", "coupling_policy", "media_alignment",
})


def complete_stage_requests(
    stages: Sequence[Mapping[str, Any]], *, backbone_name: str, extensions: Sequence[str],
) -> list[dict[str, Any]]:
    """Fill unique single-binding choices using one registry for this config.

    Complete snapshots and explicit compound recipes need no lookup. In
    particular, reloading an expanded snapshot does not call research
    registrations merely to reconstruct fields that are already present.
    """
    result = [dict(stage) for stage in stages]
    for stage in result:
        if (stage.get("plugin") in {
                "qwen3_vl_attention_replacement", "qwen3_vl_ffn_replacement"}
                and not any(stage.get(name) for name in ("algorithm", "variant", "binding", "bindings"))):
            # Existing recipes and persisted effective_config snapshots used
            # plugin-only names for these shared-LLM module replacements.
            # Normalize their identity here; execution has one binding path.
            stage["algorithm"] = stage["plugin"]
            for name in ("variant", "binding", "modalities", "target_modalities"):
                if not stage.get(name):
                    stage.pop(name, None)
            scope = stage.get("modalities", stage.get("target_modalities", []))
            if set(scope) != {"image"}:
                stage["coupling_policy"] = "synchronized"
    incomplete = [
        index for index, stage in enumerate(result)
        if not stage.get("bindings")
        and any(stage.get(name) for name in ("algorithm", "variant", "binding"))
        and not _COMPLETE_FIELDS.issubset(stage)
    ]
    if not incomplete:
        return result
    if not isinstance(backbone_name, str) or not backbone_name:
        raise ConfigError("Completing a compression stage requires backbone.name.")

    from mllm_comp.compressors.extensions import compression_contracts

    registry = compression_contracts(extensions)
    for index in incomplete:
        stage = result[index]
        label = f"stage {stage.get('name', index)!r}"
        try:
            binding = _select_binding(stage, backbone_name=backbone_name, registry=registry)
            variant = registry.variant(binding.variant)
            algorithm = registry.algorithm(variant.algorithm)
        except (RegistryError, TypeError, ValueError) as exc:
            raise ConfigError(f"{label}: {exc}") from exc

        if "placement" not in stage:
            placements = {target.placement.value for target in binding.targets}
            stage["placement"] = _unique_choice(placements, "placement", label)
        if "execution_phase" not in stage:
            phases = {phase.value for phase in binding.supported_phases}
            stage["execution_phase"] = _unique_choice(phases, "execution_phase", label)

        defaults = {
            "algorithm": algorithm.name,
            "variant": variant.name,
            "binding": binding.name,
            "plugin": algorithm.name,
            "method": algorithm.method.value,
            "object_family": {
                CompressionMethod.TOKEN_COMPRESSION: "tensor_slot",
                CompressionMethod.OPERATION_COMPRESSION: "module_unit",
            }[algorithm.method],
            "target_modalities": sorted(value.value for value in variant.target_modalities),
            "conditioning_modalities": sorted(value.value for value in variant.conditioning_modalities),
            "preserved_modalities": sorted(value.value for value in variant.preserved_modalities),
            "coupling_policy": variant.coupling_policy.value,
            "media_alignment": variant.media_alignment.value,
        }
        for name, value in defaults.items():
            stage.setdefault(name, value)
    return result


def _select_binding(
    stage: Mapping[str, Any], *, backbone_name: str, registry: CompressionContractRegistry,
) -> BackbonePlacementBinding:
    algorithm_name = stage.get("algorithm")
    variant_name = stage.get("variant")
    binding_name = stage.get("binding")
    if binding_name:
        binding = registry.binding(binding_name)
        variant = registry.variant(binding.variant)
        if binding.backbone != backbone_name:
            raise ValueError(f"binding {binding.name!r} targets backbone {binding.backbone!r}, not {backbone_name!r}.")
        if algorithm_name and algorithm_name != variant.algorithm:
            raise ValueError(f"binding {binding.name!r} belongs to algorithm {variant.algorithm!r}, not {algorithm_name!r}.")
        if variant_name and variant_name != variant.name:
            raise ValueError(f"binding {binding.name!r} belongs to variant {variant.name!r}, not {variant_name!r}.")
        return binding

    if algorithm_name:
        registry.algorithm(algorithm_name)
    if (algorithm_name in {"qwen3_vl_attention_replacement", "qwen3_vl_ffn_replacement"}
            and variant_name is None
            and not stage.get("modalities") and not stage.get("target_modalities")):
        # Keep the historical omitted-scope interpretation stable now that the
        # same OP also advertises an explicit image-only interpretation.
        variant_name = f"{algorithm_name}.shared_llm"
    if variant_name:
        variant = registry.variant(variant_name)
        if algorithm_name and algorithm_name != variant.algorithm:
            raise ValueError(f"variant {variant_name!r} belongs to algorithm {variant.algorithm!r}, not {algorithm_name!r}.")
        algorithm_name = variant.algorithm
    candidates = [
        binding for binding in registry.bindings()
        if binding.backbone == backbone_name
        and registry.variant(binding.variant).algorithm == algorithm_name
        and (variant_name is None or binding.variant == variant_name)
    ]
    declared = candidates
    for field_name in ("target_modalities", "modalities"):
        if field_name not in stage:
            continue
        values = stage[field_name]
        if not isinstance(values, (list, tuple, set)):
            raise ValueError(f"{field_name} must be a list.")
        modalities = frozenset(Modality(value) for value in values)
        candidates = [binding for binding in candidates if binding.target_modalities == modalities]
    if len(candidates) == 1:
        return candidates[0]
    choices = ", ".join(
        f"variant={binding.variant!r}, binding={binding.name!r}"
        for binding in sorted(candidates or declared, key=lambda item: item.name)
    ) or "none"
    raise ValueError(
        f"algorithm {algorithm_name!r} has {len(candidates)} matching bindings on {backbone_name!r}; "
        f"specify variant, target_modalities or binding. Candidates: {choices}."
    )


def _unique_choice(values: set[str], field_name: str, label: str) -> str:
    if len(values) != 1:
        raise ConfigError(f"{label}: specify {field_name}; candidates: {sorted(values)}.")
    return next(iter(values))
