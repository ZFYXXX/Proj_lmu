"""An independently named method that reuses LTBM and its real rule consumers."""

from mllm_comp.compressors.contracts import CompressionContractRegistry
from mllm_comp.compressors.extensions import register_method
from mllm_comp.compressors.projector_level.local_temporal_bipartite_merging import (
    LocalTemporalBipartiteProjectorStrategy,
)


def register(registry: CompressionContractRegistry) -> None:
    register_method(
        registry,
        algorithm="researcher_ltbm_audio",
        reuse_bindings=["local_temporal_bipartite_merging_audio.audio.qwen25_omni"],
        strategy_factory=LocalTemporalBipartiteProjectorStrategy,
        supported_rules=("grouper", "aggregator"),
    )
