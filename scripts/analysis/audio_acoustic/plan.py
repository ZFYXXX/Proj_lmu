"""Explicit, separately runnable experimental conditions."""
from itertools import combinations


EXPERIMENTS = (
    "observe", "band_importance", "complementarity", "energy_importance",
    "pitch", "envelope", "rhythm",
    "frequency_energy", "representation_spectrum", "spectral_resolution",
)

def build_suite_conditions(**kwargs):
    """Evaluate identical interventions once; retain each experiment's local names."""
    shared, experiments = [], {}
    for experiment in EXPERIMENTS:
        experiments[experiment] = []
        for condition in build_conditions(experiment, **kwargs):
            operation = {k: v for k, v in condition.items() if k != "id"}
            source = next((row for row in shared
                           if {k: v for k, v in row.items() if k != "id"} == operation), None)
            if source is None:
                source = condition.copy()
                shared.append(source)
            experiments[experiment].append(dict(condition, source_condition_id=source["id"]))
    if len({row["id"] for row in shared}) != len(shared):
        raise ValueError("Different shared operations have the same condition ID")
    return shared, experiments


def build_conditions(experiment, *, cutoffs=(6000, 4000, 3000, 2000, 1000)):
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}")
    if any(not 0 < c < 8000 for c in cutoffs):
        raise ValueError("Cutoffs must lie inside (0, 8000) Hz")
    rows = []

    def add(identifier, family, **kwargs):
        rows.append(dict(id=identifier, family=family, **kwargs))

    # The identity traverses the same waveform/filter plumbing. It is a
    # numerical and generation control, not another untreated cached answer.
    if experiment in {"band_importance", "complementarity", "energy_importance", "frequency_energy"}:
        add("wave_identity", "wave", operation="identity", rms_match=False)
    if experiment in {"band_importance", "energy_importance"}:
        for index in range(8):
            for rms in (False, True):
                add(f"band{index}_{'rms' if rms else 'raw'}", "wave",
                    operation="bandstop", bands=[index], band_width_hz=1000,
                    rms_match=rms)
            if experiment == "energy_importance":
                add(f"band{index}_matched_energy", "wave", operation="matched_energy",
                    bands=[index], band_width_hz=1000, rms_match=False)
    if experiment == "complementarity":
        for bands in [[i] for i in range(4)] + [list(pair) for pair in combinations(range(4), 2)]:
            add("remove_" + "_".join(map(str, bands)), "wave", operation="bandstop",
                bands=bands, band_width_hz=2000, rms_match=False)
    if experiment == "frequency_energy":
        for cutoff in cutoffs:
            for rms in (False, True):
                add(f"lowpass{cutoff:g}_{'rms' if rms else 'raw'}", "wave",
                    operation="lowpass", cutoff_hz=cutoff, rms_match=rms)
    if experiment == "spectral_resolution":
        for groups in (128, 64, 32, 16):
            for method in (("mean",) if groups == 128 else ("mean", "center")):
                add(f"mel{groups}_{method}", "mel", groups=groups, method=method)
    if experiment == "representation_spectrum":
        for ratio in (1.0, 0.75, 0.5, 0.25):
            add(f"time_lowpass{ratio:g}", "time_lowpass", nyquist_ratio=ratio)
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Repeated parameter values produce duplicate condition IDs")
    return rows
