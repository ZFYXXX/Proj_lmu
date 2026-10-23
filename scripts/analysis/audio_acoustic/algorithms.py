"""Native-chunk-local measurements and length-preserving time smoothing."""
import numpy as np


def chunk_indices(chunk_ids):
    chunk_ids = np.asarray(chunk_ids)
    cuts = np.r_[0, np.flatnonzero(np.diff(chunk_ids) != 0) + 1, len(chunk_ids)]
    return [np.arange(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def time_lowpass(hidden, chunks, ratio):
    import torch
    if not 0 < ratio <= 1:
        raise ValueError("Temporal cutoff must be a fraction of token Nyquist")
    if ratio == 1:
        return hidden.clone()
    changed = hidden.clone()
    for indices in chunk_indices(chunks):
        values = hidden[indices.tolist()].float()
        extended = torch.cat([values, values.flip(0)], dim=0)
        frequency = torch.fft.rfftfreq(len(extended), device=hidden.device) / 0.5
        gain = (frequency <= ratio).float()
        filtered = torch.fft.irfft(torch.fft.rfft(extended, dim=0) * gain[:, None],
                                   n=len(extended), dim=0)[:len(values)]
        changed[indices.tolist()] = filtered.to(hidden.dtype)
    return changed


def temporal_spectrum(hidden, chunks, bins=16):
    """FFT along time, summed over channels; never assign Hz to hidden dimensions."""
    import torch
    edges = np.linspace(0, 1, bins + 1)
    aggregate = {"raw": np.zeros(bins), "demean": np.zeros(bins)}
    for indices in chunk_indices(chunks):
        if len(indices) < 4:
            continue
        values = hidden[indices.tolist()].detach().float()
        freq = np.fft.rfftfreq(len(values)) / 0.5
        assignment = np.minimum(np.searchsorted(edges, freq, side="right") - 1, bins - 1)
        window = torch.hann_window(len(values), device=values.device)[:, None]
        for key, signal in (("raw", values), ("demean", values - values.mean(0))):
            power = torch.fft.rfft(signal * window, dim=0).abs().square().sum(-1)
            power[1: (-1 if len(values) % 2 == 0 else None)] *= 2
            # Sum spectral energy / window energy => duration-weighted power.
            power = power / window.square().sum().clamp_min(1e-20)
            aggregate[key] += np.bincount(assignment, weights=power.cpu().numpy(), minlength=bins)
    return dict(nyquist_fraction_edges=edges, axis="time_within_native_chunk",
                **{key: (value / value.sum() if value.sum() > 1e-20 else None)
                   for key, value in aggregate.items()})


def representation_drift(values, reference):
    import torch
    current = values.detach().float()
    reference = reference.detach().float()
    if current.shape != reference.shape:
        raise ValueError("Acoustic interventions must preserve the native token layout")
    cosine = torch.nn.functional.cosine_similarity(current, reference, dim=-1)
    return dict(relative_l2=float((current - reference).norm() / reference.norm().clamp_min(1e-12)),
                mean_cosine=float(cosine.mean()),
                alignment="native_grid")
