"""Reversible runtime scope: projector-only by default; legacy via ``full``.

Select by *physical binding targets*, not an algorithm's name or paper title.
Hybrid algorithms may expose a projector-only ablation; that is not their
complete paper method. The unfiltered registry remains available for auditing.
"""
from dataclasses import replace
import os

from .contracts import CompressionContractRegistry
from mllm_comp.core.vocabulary import CompressionMethod, Modality, Placement


def projector_binding(binding, registry):
    variant = registry.variant(binding.variant)
    algorithm = registry.algorithm(variant.algorithm)
    return (
        variant.target_modalities == frozenset({Modality.IMAGE})
        and algorithm.method is CompressionMethod.TOKEN_COMPRESSION
        and not any(stage["placement"] == Placement.ENCODER_LEVEL.value
            for stage in algorithm.metadata.get("paper_compression_stages", ()))
        and all(target.placement is Placement.PROJECTOR_LEVEL for target in binding.targets)
    )


def apply_method_profile(registry, profile=None):
    profile = profile or os.environ.get("MLLM_COMP_METHOD_PROFILE", "projector")
    if profile == "full":
        return registry
    if profile != "projector":
        raise ValueError("MLLM_COMP_METHOD_PROFILE must be 'projector' or 'full'")
    bindings = [b for b in registry.bindings() if projector_binding(b, registry)]
    names = {b.name for b in bindings}
    variants = {b.variant for b in bindings}
    algorithms = {registry.variant(v).algorithm for v in variants}
    result = CompressionContractRegistry()
    for descriptor in registry.algorithms():
        if descriptor.name in algorithms:
            result.register_algorithm(descriptor)
    for descriptor in registry.variants():
        if descriptor.name in variants:
            result.register_variant(descriptor)
    for binding in bindings:
        result.register_binding(replace(binding, conflicts=binding.conflicts & names,
            metadata={**binding.metadata, "method_profile": "projector"}))
    result.extend_support(unit for unit in registry.support.find() if unit.binding in names)
    return result
