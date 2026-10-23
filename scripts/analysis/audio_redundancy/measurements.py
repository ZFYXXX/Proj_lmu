"""Per-audio rank and cross-fitted discrete observations."""
import numpy as np
import torch

from .algorithms import discrete_measurement, local_ranks, rank_measurement, spectrum_stats


def observe(capture, settings, bank, fold, *, example=False, seed=42):
    stages, discrete, bases, arrays = [], {}, {}, {}
    duration = capture.mel_length * capture.mel_step_ms / 1000
    if example:
        arrays.update(mel=capture.mel.float().cpu().numpy().T,
                      mel_time_ms=np.arange(capture.mel_length) * capture.mel_step_ms)
    for name in settings["stages"]:
        hidden = capture.hidden[name]
        times, chunks = capture.layouts[name]
        row = dict(name=name, tokens=len(hidden), channels=hidden.shape[1], tokens_per_second=len(hidden) / duration)
        if example:
            arrays[f"time_{name}"] = times
            arrays[f"chunks_{name}"] = chunks
        if "A" in settings["experiments"]:
            raw, raw_s = rank_measurement(hidden, centered=False)
            if name in settings["intervention_stages"]:
                x = hidden.float()
                _, singular, vh = torch.linalg.svd(x - x.mean(0), full_matrices=False)
                bases[name] = vh.T
                s = singular.cpu().numpy()
                centered = spectrum_stats(s, *x.shape, centered=True)
            else:
                centered, s = rank_measurement(hidden)
            local = local_ranks(hidden, times, chunks, settings["windows_ms"])
            row["rank"] = dict(raw=raw, centered=centered, local={})
            for window in settings["windows_ms"]:
                selected = [r for r in local if r["window_ms"] == window]
                valid = [r for r in selected if r["effective_rank"] is not None]
                row["rank"]["local"][str(window)] = dict(windows=len(selected), valid_windows=len(valid),
                    **{key: float(np.mean([r[key] for r in valid])) if valid else None for key in (
                        "tokens", "rank_capacity", "effective_rank", "normalized_effective_rank", "participation_ratio", "k90", "k95")})
            if example:
                arrays[f"singular_{name}"] = s
                arrays[f"singular_raw_{name}"] = raw_s
                for window in settings["windows_ms"]:
                    valid = [r for r in local if r["window_ms"] == window and r["effective_rank"] is not None]
                    arrays[f"local_{name}_{window:g}"] = np.array([
                        [r["center_ms"], r["effective_rank"], r["normalized_effective_rank"], r["tokens"]] for r in valid]).reshape(-1, 4)
        if bank is not None:
            row["discrete"] = {}
            for k in settings["codebook_sizes"]:
                result = discrete_measurement(hidden, chunks, bank.get(fold, name, k), duration, seed=seed)
                discrete[(name, k)] = result
                row["discrete"][str(k)] = result["stats"]
                if example:
                    for field in ("codes", "entropy", "surprisal"):
                        arrays[f"{field}_{name}_K{k}"] = result[field]
        stages.append(row)
    if example and "H" in settings["experiments"]:
        arrays["singular_Mel"] = rank_measurement(capture.mel)[1]
    return dict(duration_seconds=duration, stages=stages, held_out_fold=fold), discrete, bases, arrays
