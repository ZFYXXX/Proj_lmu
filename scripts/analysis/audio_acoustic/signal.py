"""Linear-power measurements and length-preserving acoustic interventions."""
from contextlib import contextmanager

import numpy as np


def frames(wave, n_fft=400, hop=160):
    wave = np.asarray(wave, dtype=np.float64)
    if wave.ndim != 1 or not len(wave) or not np.isfinite(wave).all():
        raise ValueError("Expected finite nonempty mono audio")
    padded = np.pad(wave, (n_fft // 2, n_fft // 2), mode="reflect" if len(wave) > 1 else "edge")
    return np.lib.stride_tricks.sliding_window_view(padded, n_fft)[::hop]


def spectrum(wave, sr=16000, n_fft=400, hop=160):
    window = np.hanning(n_fft + 1)[:-1]
    fft = np.fft.rfft(frames(wave, n_fft, hop) * window, axis=-1)
    psd = np.mean(np.abs(fft) ** 2, axis=0) / (sr * np.sum(window ** 2))
    psd[1:-1] *= 2
    hz = np.fft.rfftfreq(n_fft, 1 / sr)
    power = psd * sr / n_fft
    total = float(power.sum())
    fractions = power / total if total > 0 else np.zeros_like(power)
    edges = np.arange(0, sr / 2 + 1, 1000)
    bands = [float(fractions[(hz >= a) & ((hz < b) if b < sr / 2 else (hz <= b))].sum())
             for a, b in zip(edges[:-1], edges[1:])]
    rolloff = {str(q): float(hz[min(np.searchsorted(np.cumsum(fractions), q), len(hz) - 1)])
               if total else None for q in (0.85, 0.95, 0.99)}
    return dict(hz=hz, psd=psd, power_fraction=fractions,
                band_edges_hz=edges, band_fraction=bands, rolloff_hz=rolloff,
                rms=float(np.sqrt(np.mean(np.asarray(wave, dtype=float) ** 2))),
                duration_s=len(wave) / sr, has_energy=total > 0)


def _rise(hz, edge, transition):
    position = np.clip((hz - edge + transition / 2) / transition, 0, 1)
    return 0.5 - 0.5 * np.cos(np.pi * position)


def band_window(hz, low, high, nyquist, transition=200.0):
    left = np.ones_like(hz) if low <= 0 else _rise(hz, low, transition)
    right = np.ones_like(hz) if high >= nyquist else 1 - _rise(hz, high, transition)
    return left * right


def filter_wave(wave, condition, sr=16000):
    """Even extension + smooth real FFT gain; no resampling or hard crop seams.

    Matched-energy attenuation removes the same *linear* power fraction from
    each band of a given clip: half the smallest removable band power. Gains
    are solved with Parseval's identity, not from log-Mel values.
    """
    source = np.asarray(wave, dtype=np.float64)
    if source.ndim != 1 or not len(source) or not np.isfinite(source).all():
        raise ValueError("Expected finite nonempty mono audio")
    extended = np.concatenate([source, source[::-1]])
    transform = np.fft.rfft(extended)
    hz = np.fft.rfftfreq(len(extended), 1 / sr)
    gain = np.ones_like(hz)
    operation = condition["operation"]
    transition = float(condition.get("transition_hz", 200.0))
    attenuation, matched_fraction = 1.0, None
    if operation in {"bandstop", "matched_energy"}:
        width = condition["band_width_hz"]
        windows = [band_window(hz, i * width, (i + 1) * width, sr / 2, transition)
                   for i in range(int(sr / 2 / width))]
        if operation == "matched_energy":
            power = np.abs(transform) ** 2
            power[1:-1] *= 2
            power /= max(float(power.sum()), 1e-30)
            removals = [float(np.sum(power * (2 * w - w * w))) for w in windows]
            matched_fraction = 0.5 * min(removals)
            w = windows[condition["bands"][0]]
            lo, hi = 0.0, 1.0
            for _ in range(48):
                mid = (lo + hi) / 2
                removed = float(np.sum(power * (2 * mid * w - (mid * w) ** 2)))
                if removed < matched_fraction:
                    lo = mid
                else:
                    hi = mid
            attenuation = (lo + hi) / 2
        removed_window = np.clip(sum(windows[index] for index in condition["bands"]), 0, 1)
        gain = 1 - attenuation * removed_window
    elif operation == "lowpass":
        gain = 1 - _rise(hz, condition["cutoff_hz"], transition)
    elif operation != "identity":
        raise ValueError(f"Unknown waveform operation: {operation}")
    filtered = np.fft.irfft(transform * gain, n=len(extended))[:len(source)]
    original_power = float(np.mean(source ** 2))
    filtered_power = float(np.mean(filtered ** 2))
    rms_gain = 1.0
    if condition.get("rms_match") and filtered_power > 1e-30:
        rms_gain = np.sqrt(original_power / filtered_power)
        filtered *= rms_gain
    return filtered.astype(np.float32), dict(
        removed_power_fraction=(1 - filtered_power / original_power) if original_power else None,
        requested_removed_power_fraction=matched_fraction, attenuation=attenuation,
        energy_match_degenerate=matched_fraction is not None and matched_fraction <= 1e-6,
        rms_gain=float(rms_gain), peak=float(np.max(np.abs(filtered))),
        identity=operation == "identity", samples=len(source), transition_hz=transition,
        filter="even_extension_zero_phase_raised_cosine", clipped=False)


def acoustic_targets(wave, sr=16000):
    """Input-only ACF/activity proxies; not human F0, VAD, or syllable labels."""
    blocks = frames(wave, n_fft=640, hop=160)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1))
    db = 20 * np.log10(np.maximum(rms, 1e-8))
    active = (db > max(float(db.max()) - 40, -80)) & (rms > 1e-7)
    centered = blocks - blocks.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2048, axis=1)
    acf = np.fft.irfft(np.abs(fft) ** 2, n=2048, axis=1)[:, :641]
    # Restrict the documented proxy to 50..800 Hz. Window-length bias is
    # corrected by the number of overlapping samples at each lag.
    acf[:, :640] /= np.arange(640, 0, -1)[None, :]
    acf /= np.maximum(acf[:, :1], 1e-20)
    lag_min, lag_max = int(sr / 800), int(sr / 50)
    candidates = acf[:, lag_min:lag_max + 1]
    # Earliest local peak within 90% of the maximum reduces octave-down ties.
    peaks = (candidates >= np.roll(candidates, 1, axis=1)) & (candidates >= np.roll(candidates, -1, axis=1))
    peaks[:, (0, -1)] = False
    good = peaks & (candidates >= 0.9 * candidates.max(axis=1, keepdims=True))
    peak_index = np.where(good.any(axis=1), good.argmax(axis=1), candidates.argmax(axis=1))
    lag = peak_index + lag_min
    periodicity = candidates[np.arange(len(lag)), peak_index]
    voiced = active & (periodicity >= 0.6)
    f0 = np.where(voiced, sr / lag, np.nan)
    time_ms = np.arange(len(rms)) * 10.0
    changes = np.diff(np.r_[False, ~active, False].astype(int))
    starts, ends = np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)
    pauses = (ends - starts) * 0.01
    pauses = pauses[pauses >= 0.2]
    return dict(time_ms=time_ms, log_f0=np.log(np.maximum(f0, 1e-12)),
                voiced=voiced.astype(float), log_rms=np.log(np.maximum(rms, 1e-8)),
                active=active.astype(float),
                rhythm=np.array([float((~active).mean()), float(pauses.mean()) if len(pauses) else 0,
                                 len(pauses) / max(len(wave) / sr, 1e-9)]),
                provenance="waveform_acf_50_800Hz_and_relative_RMS_activity_proxies")


def acoustic_profile(wave, targets):
    """Original-input descriptors only, independent of model predictions."""
    log_f0 = targets["log_f0"][np.isfinite(targets["log_f0"])]
    # A few isolated periodic frames do not define a clip's pitch distribution.
    usable_pitch = len(log_f0) >= 5
    log_rms = targets["log_rms"]
    return dict(
        duration_s=len(wave) / 16000,
        valid_f0_frames=len(log_f0),
        median_f0_hz=float(np.exp(np.median(log_f0))) if usable_pitch else None,
        f0_iqr_cents=float(np.diff(np.quantile(log_f0, [0.25, 0.75]))[0] * 1200 / np.log(2))
        if usable_pitch else None,
        voiced_fraction=float(np.mean(targets["voiced"])),
        mean_log_rms=float(np.mean(log_rms)),
        log_rms_std=float(np.std(log_rms)),
        log_rms_range_p90_p10=float(np.diff(np.quantile(log_rms, [0.1, 0.9]))[0]),
        proxy_pause_fraction=float(targets["rhythm"][0]),
        proxy_pause_mean_s=float(targets["rhythm"][1]),
        proxy_pause_rate_hz=float(targets["rhythm"][2]),
        provenance=targets["provenance"],
    )


def reduce_mel_power(power, groups, method="mean"):
    import torch
    n_mels = power.shape[-2]
    if not 1 <= groups <= n_mels:
        raise ValueError("Mel group count outside the native frequency dimension")
    output = power.clone()
    for indices in np.array_split(np.arange(n_mels), groups):
        start, end = int(indices[0]), int(indices[-1]) + 1
        value = (power[..., start:end, :].mean(dim=-2, keepdim=True) if method == "mean"
                 else power[..., start + (end - start - 1) // 2:start + (end - start - 1) // 2 + 1, :])
        if method not in {"mean", "center"}:
            raise ValueError("Unknown Mel reduction")
        output[..., start:end, :] = value
    return output


@contextmanager
def mel_intervention(extractor, condition):
    if not condition or condition["family"] != "mel":
        yield
        return
    import torch
    original = extractor._torch_extract_fbank_features

    def extract(waveform, device="cpu"):
        wave = torch.from_numpy(waveform).to(device, torch.float32)
        if extractor.dither != 0:
            raise ValueError("Paired Mel intervention requires dither=0")
        stft = torch.stft(wave, extractor.n_fft, extractor.hop_length,
                          window=torch.hann_window(extractor.n_fft, device=device), return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2
        filters = torch.from_numpy(extractor.mel_filters).to(device, torch.float32)
        mel = reduce_mel_power(filters.T @ magnitudes, condition["groups"], condition["method"])
        log = torch.clamp(mel, min=1e-10).log10()
        maximum = log.amax(dim=(-2, -1), keepdim=True)
        return ((torch.maximum(log, maximum - 8.0) + 4.0) / 4.0).detach().cpu().numpy()

    extractor._torch_extract_fbank_features = extract
    try:
        yield
    finally:
        extractor._torch_extract_fbank_features = original


@contextmanager
def audio_intervention(adapter, condition, capture, on_wave=None):
    """Intercept the existing decoder's mono waveform, without rewriting audio files."""
    original = adapter.media_loader
    loader = original
    if loader is None:
        from qwen_omni_utils import process_mm_info
        loader = process_mm_info

    def load(*args, **kwargs):
        audios, images, videos = loader(*args, **kwargs)
        if audios is None or len(audios) != 1:
            raise ValueError("Acoustic MMSU experiments require one decoded audio per request")
        wave = np.asarray(audios[0], dtype=np.float32)
        if wave.ndim != 1:
            raise ValueError("The native loader must return mono audio")
        capture["original_wave"] = wave
        if condition and condition["family"] == "wave":
            wave, capture["filter"] = filter_wave(wave, condition)
            audios = [wave]
        capture["input_wave"] = wave
        if on_wave is not None:
            on_wave(capture["original_wave"], wave)
        return audios, images, videos

    # The adapter is a frozen dataclass. This request-local field swap is
    # restored even on failure; no platform classes or model weights change.
    object.__setattr__(adapter, "media_loader", load)
    try:
        yield
    finally:
        object.__setattr__(adapter, "media_loader", original)
