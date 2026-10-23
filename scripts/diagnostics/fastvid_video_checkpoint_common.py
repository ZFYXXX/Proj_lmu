"""Independent FastVID diagnostic checks shared across native video layouts."""

from typing import Any, Mapping

from mllm_comp.compressors.video.config import canonical_video_config_record

CANONICAL_FASTVID_CONFIG = canonical_video_config_record("fastvid")


def fastvid_algorithm_checks(
    record: Mapping[str, Any] | None,
    *,
    source_count: int,
    grid: Mapping[str, Any],
) -> dict[str, bool]:
    record = {} if record is None else record
    temporal = int(grid["temporal_slice_count"])
    projected_tokens_per_frame = int(grid["per_temporal_slice_source_count"])
    raw_tokens_per_frame = int(grid["raw_tokens_per_frame"])
    expected_source = int(grid["source_grid_token_count"])
    projected_shape = tuple(record.get("projected_features_shape", ()))
    raw_shape = tuple(record.get("raw_features_shape", ()))
    global_shape = tuple(record.get("frame_global_features_shape", ()))
    saliency_shape = tuple(record.get("token_saliency_attention_shape", ()))
    return {
        "canonical_source_raster_exact": (
            source_count == expected_source
            and int(record.get("source_count", 0)) == expected_source
            and int(record.get("frame_count", 0)) == temporal
            and int(record.get("projected_tokens_per_frame", 0))
            == projected_tokens_per_frame
        ),
        "canonical_config_exact": dict(record.get("canonical_config", {}))
        == CANONICAL_FASTVID_CONFIG,
        "projected_signal_shape_exact": (
            len(projected_shape) == 2
            and int(projected_shape[0]) == source_count
            and int(projected_shape[1]) > 0
        ),
        "raw_signal_shape_exact": (
            len(raw_shape) == 2
            and int(raw_shape[0]) == temporal * raw_tokens_per_frame
            and int(raw_shape[1]) > 0
            and int(record.get("raw_tokens_per_frame", 0)) == raw_tokens_per_frame
        ),
        "frame_global_signal_shape_exact": (
            len(global_shape) == 2
            and int(global_shape[0]) == temporal
            and int(global_shape[1]) > 0
        ),
        "saliency_signal_shape_exact": saliency_shape == (source_count,),
        "all_runtime_signals_finite": all(
            bool(record.get(name))
            for name in (
                "projected_features_finite",
                "raw_features_finite",
                "frame_global_features_finite",
                "token_saliency_attention_finite",
            )
        ),
        "algorithm_representatives_source_ordered": bool(
            record.get("representatives_source_ordered")
        ),
    }
