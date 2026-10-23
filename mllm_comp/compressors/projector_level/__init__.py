"""Projector-level compression namespace."""

from .fastslm_hta import (
    FastSLMHTAAudio,
    FastSLMHTASpec,
    build_fastslm_hta,
    fastslm_hierarchical_temporal_abstract,
    fastslm_hta_keep_count,
    fastslm_window_budgets,
    load_fastslm_hta_checkpoint,
)
from .local_temporal_bipartite_merging import (
    LocalTemporalBipartiteMergingAudio,
    LocalTemporalBipartiteMergingSpec,
    local_temporal_bipartite_merge,
    ltbm_keep_count,
)
from .segmentwise_topk import (
    SegmentwiseTopKAudio,
    SegmentwiseTopKSpec,
    realized_keep_count,
    segment_boundaries,
    segment_budgets,
    segmentwise_random_indices,
    segmentwise_topk_indices,
    segmentwise_uniform_indices,
    target_keep_count,
)
from .similarity_attention_merge_prune import (
    SimilarityAttentionMergePruneAudio,
    SimilarityAttentionMergePruneSpec,
    sa_map_keep_count,
    similarity_attention_merge_prune,
)
from .uniform_average_pooling import (
    UniformAveragePoolingAudio,
    UniformAveragePoolingSpec,
    uniform_average_keep_count,
    uniform_average_pool,
)
from .uniform_sampling import (
    UniformSamplingAudio,
    UniformSamplingSpec,
    uniform_sample,
    uniform_sampling_keep_count,
)
from .unsupervised_segmentation import (
    UnsupervisedSegmentationAudio,
    UnsupervisedSegmentationSpec,
    unsupervised_segment_merge,
    unsupervised_segmentation_boundaries,
)

__all__ = [
    "FastSLMHTAAudio",
    "FastSLMHTASpec",
    "LocalTemporalBipartiteMergingAudio",
    "LocalTemporalBipartiteMergingSpec",
    "SegmentwiseTopKAudio",
    "SegmentwiseTopKSpec",
    "SimilarityAttentionMergePruneAudio",
    "SimilarityAttentionMergePruneSpec",
    "UniformAveragePoolingAudio",
    "UniformAveragePoolingSpec",
    "UniformSamplingAudio",
    "UniformSamplingSpec",
    "UnsupervisedSegmentationAudio",
    "UnsupervisedSegmentationSpec",
    "build_fastslm_hta",
    "fastslm_hierarchical_temporal_abstract",
    "fastslm_hta_keep_count",
    "fastslm_window_budgets",
    "local_temporal_bipartite_merge",
    "load_fastslm_hta_checkpoint",
    "ltbm_keep_count",
    "realized_keep_count",
    "segment_boundaries",
    "segment_budgets",
    "segmentwise_random_indices",
    "segmentwise_topk_indices",
    "segmentwise_uniform_indices",
    "sa_map_keep_count",
    "similarity_attention_merge_prune",
    "target_keep_count",
    "uniform_average_keep_count",
    "uniform_average_pool",
    "uniform_sample",
    "uniform_sampling_keep_count",
    "unsupervised_segment_merge",
    "unsupervised_segmentation_boundaries",
]
