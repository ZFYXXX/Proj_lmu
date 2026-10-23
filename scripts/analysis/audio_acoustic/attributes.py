"""Input-only acoustic strata; no fitted readout and no use of QA outcomes."""
import numpy as np


ATTRIBUTE_FIELDS = {
    "pitch": {"median_f0_hz": "Hz", "f0_iqr_cents": "cents", "voiced_fraction": "fraction"},
    "envelope": {"mean_log_rms": "natural log RMS", "log_rms_std": "natural log RMS",
                 "log_rms_range_p90_p10": "natural log RMS"},
    "rhythm": {"proxy_pause_fraction": "fraction", "proxy_pause_mean_s": "seconds",
               "proxy_pause_rate_hz": "segments / second"},
}


def input_terciles(values, clusters):
    """One observation per audio for cutoffs; repeated questions retain the same bin."""
    values = np.asarray(values, dtype=float)
    unique = {}
    for value, cluster in zip(values, clusters):
        if np.isfinite(value):
            unique.setdefault(cluster, value)
    finite = np.array(list(unique.values()))
    labels = np.full(len(values), "unavailable", dtype=object)
    cutoffs, groups = [], []
    if len(finite):
        if np.ptp(finite) == 0:
            labels[np.isfinite(values)] = "constant"
            groups = ["constant"]
        else:
            cutoffs = np.quantile(finite, [1 / 3, 2 / 3]).tolist()
            labels[np.isfinite(values) & (values <= cutoffs[0])] = "low"
            labels[np.isfinite(values) & (values > cutoffs[0]) & (values <= cutoffs[1])] = "middle"
            labels[np.isfinite(values) & (values > cutoffs[1])] = "high"
            groups = ["low", "middle", "high"]
    if np.any(labels == "unavailable"):
        groups.append("unavailable")
    return dict(cutoffs=cutoffs, labels=labels.tolist(), groups=groups,
                cutoff_audio_count=len(finite), values=[float(v) if np.isfinite(v) else None for v in values])


def prepare_attributes(experiment, baseline):
    clusters = [row.get("audio_path") or f"doc:{row['doc_index']}" for row in baseline]
    profiles = [row.get("acoustic_proxy_summary", {}) for row in baseline]
    attributes = {
        name: dict(input_terciles([p.get(name) for p in profiles], clusters), unit=unit)
        for name, unit in ATTRIBUTE_FIELDS[experiment].items()
    }
    duration = input_terciles([p.get("duration_s") for p in profiles], clusters)
    tokens = input_terciles([row["projector_tokens"] for row in baseline], clusters)
    return dict(
        attributes=attributes,
        control_bins={"duration_s": duration, "native_tokens": tokens},
        records=[dict(doc_index=row["doc_index"], audio_path=row.get("audio_path"),
                      task_name=row.get("task_name"), sub_category=row.get("sub_category"),
                      duration_s=profiles[i].get("duration_s"), native_tokens=row["projector_tokens"],
                      valid_f0_frames=profiles[i].get("valid_f0_frames"))
                 for i, row in enumerate(baseline)],
        grouping="Unique-audio input terciles, before consulting outcomes; ties stay together; constant and unavailable explicit",
        scope="Original-input attribute distributions and native QA accuracy; no token compression, decoding probe or causal isolation; low activity is not speaking rate",
    )
