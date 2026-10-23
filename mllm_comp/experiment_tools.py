"""Read-only capability discovery and comparison of recorded experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from mllm_comp.analysis.run_comparison import compare_runs
from mllm_comp.compressors.extensions import compression_contracts
from mllm_comp.compressors.support_report import _support_matrix_row
from mllm_comp.core.serialization import to_jsonable
from mllm_comp.core.vocabulary import CompressionMethod, ExecutionPhase, Modality
from mllm_comp.registry.catalog import CapabilityKind, iter_capability_specs


def discover_capabilities(
    *,
    kind: CapabilityKind | str = CapabilityKind.COMPRESSOR,
    name: str | None = None,
    method: CompressionMethod | str | None = None,
    variant: str | None = None,
    binding: str | None = None,
    backbone: str | None = None,
    modality: Modality | str | None = None,
    phase: ExecutionPhase | str | None = None,
    extensions: Sequence[str] = (),
) -> dict[str, Any]:
    """Project declarations and exact evidence without constructing providers.

    Compression items describe one real binding and phase. A binding without
    support evidence remains discoverable, including explicit research
    extensions. Neither an available executor nor an active support unit is a
    guarantee that arbitrary parameters or combinations will execute.
    """

    kind = CapabilityKind(kind)
    if kind is not CapabilityKind.COMPRESSOR:
        if any(value is not None for value in (method, variant, binding, backbone, modality, phase)) or extensions:
            raise ValueError("Method, binding, backbone, modality, phase and extension filters require kind='compressor'.")
        items = [
            {
                "name": spec.name,
                "kind": spec.kind.value,
                "status": spec.status.value,
                "interfaces": sorted(spec.interfaces),
                "override_fields": sorted(spec.override_fields),
                "metadata": to_jsonable(spec.metadata),
            }
            for spec in iter_capability_specs(kind=kind)
            if name is None or spec.name == name
        ]
        return {"kind": kind.value, "items": sorted(items, key=lambda item: item["name"])}

    method = None if method is None else CompressionMethod(method)
    modality = None if modality is None else Modality(modality)
    phase = None if phase is None else ExecutionPhase(phase)
    registry = compression_contracts(extensions)
    items = []
    for bound in sorted(registry.bindings(), key=lambda item: item.name):
        variant_descriptor = registry.variant(bound.variant)
        algorithm = registry.algorithm(variant_descriptor.algorithm)
        if (
            (name is not None and algorithm.name != name)
            or (method is not None and algorithm.method is not method)
            or (variant is not None and bound.variant != variant)
            or (binding is not None and bound.name != binding)
            or (backbone is not None and bound.backbone != backbone)
            or (modality is not None and modality not in variant_descriptor.target_modalities)
        ):
            continue
        for bound_phase in sorted(bound.supported_phases, key=str):
            if phase is not None and bound_phase is not phase:
                continue
            units = registry.support.find(
                algorithm=algorithm.name, variant=bound.variant,
                backbone=bound.backbone, target_modality=modality, phase=bound_phase,
            )
            exact_rows = [
                {
                    **_support_matrix_row(unit, bound).to_dict(),
                    "evidence": to_jsonable(unit.evidence),
                    "inference_active": unit.inference_active,
                    "train_verified": unit.train_verified,
                    "train_step_verified": unit.train_step_verified,
                }
                for unit in sorted(units, key=lambda item: item.identity.target_modality.value)
                if unit.binding == bound.name
            ]
            items.append({
                "algorithm": algorithm.name,
                "algorithm_version": algorithm.version,
                "method": algorithm.method.value,
                "operations": sorted(str(value) for value in algorithm.operations),
                "capability_status": str(algorithm.metadata.get("capability_status", "target")),
                "variant": bound.variant,
                "binding": bound.name,
                "backbone": bound.backbone,
                "phase": bound_phase.value,
                "target_modalities": sorted(str(value) for value in variant_descriptor.target_modalities),
                "conditioning_modalities": sorted(str(value) for value in variant_descriptor.conditioning_modalities),
                "preserved_modalities": sorted(str(value) for value in variant_descriptor.preserved_modalities),
                "token_unit": None if variant_descriptor.token_unit is None else str(variant_descriptor.token_unit),
                "supported_rules": sorted(variant_descriptor.supported_rules),
                "config_defaults": to_jsonable(dict(variant_descriptor.config_defaults)),
                "targets": [
                    {"name": target.name, "placement": str(target.placement), "site": target.target_site}
                    for target in bound.ordered_targets()
                ],
                "required_handles": sorted(bound.required_handles),
                "executor_available": callable(bound.executor_factory),
                "support": to_jsonable(exact_rows),
            })
    registered = {algorithm.name for algorithm in registry.algorithms()}
    catalog_only = []
    if all(value is None for value in (method, variant, binding, modality, phase)):
        for spec in iter_capability_specs(kind=kind):
            if spec.name in registered or (name is not None and spec.name != name):
                continue
            if backbone is not None and backbone not in spec.metadata.get("backbones", []):
                continue
            catalog_only.append({
                "name": spec.name, "kind": kind.value, "status": spec.status.value,
                "activation_phase": spec.activation_phase,
                "interfaces": sorted(spec.interfaces), "metadata": to_jsonable(spec.metadata),
            })
    return {
        "kind": kind.value, "extensions": list(extensions), "items": items,
        "catalog_only": sorted(catalog_only, key=lambda item: item["name"]),
    }


__all__ = ["compare_runs", "discover_capabilities"]
