"""Waveform weak teachers and domain-specific interventions (16 kHz mono)."""
from __future__ import annotations
import numpy as np
from scipy.signal import find_peaks
from .data import validate_speech

SAMPLE_RATE = 16000
HOP = 320  # 20 ms weak-target grid; native token coordinates are kept separately.

def read_audio(path):
    import soundfile as sf
    import librosa
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y) < 1024 or not np.isfinite(y).all():
        raise ValueError(f"Invalid/too short audio: {path}")
    return np.asarray(y, dtype=np.float32)

def world_features(y):
    import pyworld
    wave = np.asarray(y, dtype=np.float64)
    f0, times = pyworld.dio(wave, SAMPLE_RATE, f0_floor=50, f0_ceil=800, frame_period=20.0)
    f0 = pyworld.stonemask(wave, f0, times, SAMPLE_RATE)
    return f0, times

def key_from_chroma(chroma):
    # Krumhansl-Schmuckler templates; a weak key estimate, not annotated truth.
    major = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
    minor = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])
    profile = np.asarray(chroma).mean(axis=0)
    if np.std(profile) < 1e-8:
        return None
    scores = [np.corrcoef(profile, np.roll(template, shift))[0,1]
              for template in (major, minor) for shift in range(12)]
    return int(np.argmax(scores))

def weak_targets(y, domain, speech=None):
    import librosa
    duration = len(y) / SAMPLE_RATE
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=HOP)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=SAMPLE_RATE, hop_length=HOP)
    valid = times < duration
    times, rms = times[valid], rms[valid]
    flux = librosa.onset.onset_strength(y=y, sr=SAMPLE_RATE, hop_length=HOP)
    flux = flux[:len(times)]
    onsets = librosa.onset.onset_detect(onset_envelope=flux, sr=SAMPLE_RATE,
                                       hop_length=HOP, units="time")
    arrays = {"time": times, "log_energy": np.log(np.maximum(rms, 1e-7)),
              "onset_strength": flux, "onset_times": onsets}
    meta = {"duration": duration, "weak_labels": True,
            "onset_teacher": "librosa positive spectral flux + onset_detect",
            "label_hop_seconds": HOP / SAMPLE_RATE}
    if domain in ("music", "speech"):
        f0, ft = world_features(y)
        # Never interpolate across unvoiced gaps.
        right = np.minimum(np.searchsorted(ft,times),len(ft)-1)
        left = np.maximum(right-1,0)
        indices = np.where(np.abs(ft[left]-times)<np.abs(ft[right]-times),left,right)
        f0 = f0[indices]
        arrays["log_f0"] = np.where(f0 > 0, np.log2(np.maximum(f0, 1)), np.nan)
        meta["f0_teacher"] = "WORLD DIO + StoneMask; voiced frames only"
    if domain == "music":
        chroma = librosa.feature.chroma_cqt(y=y, sr=SAMPLE_RATE, hop_length=HOP, tuning=0)
        chroma = chroma[:, :len(times)].T
        if len(chroma) != len(times):
            raise ValueError("CQT and weak-target frame counts differ")
        arrays["chroma"] = chroma.astype(np.float32)
        pitch = chroma.argmax(1).astype(float)
        pitch[(rms < 1e-5) | (chroma.max(1) < 1e-4)] = np.nan
        arrays["pitch_class"] = pitch
        bpm, beats = librosa.beat.beat_track(onset_envelope=flux, sr=SAMPLE_RATE,
                                            hop_length=HOP, units="time", trim=False)
        arrays["beat_times"] = np.asarray(beats)
        bpm = float(np.asarray(bpm).reshape(-1)[0])
        meta["bpm"] = bpm if bpm > 0 and len(beats) >= 3 else None
        meta["key"] = key_from_chroma(chroma) if rms.max() >= 1e-5 else None
        centered = flux - flux.mean()
        ac = librosa.autocorrelate(centered)
        lo, hi = int(60/240*SAMPLE_RATE/HOP), min(len(ac), int(60/40*SAMPLE_RATE/HOP))
        meta["periodicity"] = float(np.max(ac[lo:hi])/ac[0]) if hi>lo and ac[0]>1e-12 else None
        meta["tonal_teacher"] = "librosa CQT chroma + Krumhansl-Schmuckler key estimate"
        meta["beat_teacher"] = "librosa dynamic-programming beat_track"
        meta["f0_limitation"] = "dominant monophonic F0 proxy; polyphonic music is not ground-truth pitch"
    if domain == "speech":
        speech = speech or {}
        validate_speech({"id": speech.get("id", "unknown"), **speech}, duration)
        meta["speech_alignment"] = speech
        words = speech.get("words", [])
        meta["words_per_second"] = len(words) / duration if words else None
        phones = speech.get("phones", [])
        meta["phonemes_per_second"] = len(phones) / duration if phones else None
        meta["rate_definition"] = "aligned lexical tokens / full clip duration, pauses included"
    return arrays, meta

def variant_specs(domain, experiments):
    names = []
    if domain == "sound":
        if "S1" in experiments:
            names += ["gain_6db", "noise_20db", "shift_250ms"]
        if "S2" in experiments:
            names += ["shuffle_250ms", "shuffle_500ms", "shuffle_1000ms"]
    if domain == "music":
        if "M1" in experiments:
            names += ["pitch_-2", "pitch_-1", "pitch_1", "pitch_2"]
        if "M2" in experiments:
            names += ["stretch_0.8", "stretch_1.2", "beat_phase_shuffle"]
    if domain == "speech" and "SP2" in experiments:
        names += ["world_resynthesis", "pitch_flatten", "energy_normalize",
                  "stretch_0.8", "stretch_1.2"]
    return names

def shuffle_chunks(y, boundaries, rng):
    chunks = [x for x in np.split(y, boundaries) if len(x)]
    if len(chunks) < 2:
        return None
    order = rng.permutation(len(chunks))
    if np.array_equal(order, np.arange(len(chunks))):
        order = np.roll(order, 1)
    return np.concatenate([chunks[i] for i in order])

def intervene(y, name, targets, seed):
    import librosa
    rng = np.random.default_rng(seed)
    info = {"name": name, "time_scale": 1.0, "reference_validity": "unverified_after_intervention"}
    if name == "gain_6db":
        changed = y * 10**(6/20)
        info["reference_validity"] = "event_identity_presumed"
    elif name == "noise_20db":
        rms = np.sqrt(np.mean(y*y))
        changed = y + rng.normal(size=len(y)).astype(np.float32)*rms/10
        info["reference_validity"] = "event_identity_presumed"
    elif name == "shift_250ms":
        changed = np.roll(y, round(.25*SAMPLE_RATE))
        info["reference_validity"] = "circular_shift_event_identity_presumed"
    elif name.startswith("shuffle_"):
        step = round(int(name.split("_")[1].removesuffix("ms"))/1000*SAMPLE_RATE)
        changed = shuffle_chunks(y, np.arange(step, len(y), step), rng)
    elif name.startswith("pitch_") and name != "pitch_flatten":
        info["semitones"] = int(name.split("_")[1])
        changed = librosa.effects.pitch_shift(y=y, sr=SAMPLE_RATE, n_steps=info["semitones"])
    elif name.startswith("stretch_"):
        rate = float(name.split("_")[1])
        info["time_scale"] = rate  # original time = changed time * rate
        changed = librosa.effects.time_stretch(y=y, rate=rate)
    elif name == "beat_phase_shuffle":
        beats = targets.get("beat_times", np.array([]))
        if len(beats) < 3:
            return None, {**info, "status": "skipped", "reason": "fewer_than_three_weak_beats"}
        # Shuffle half-beat segments; disrupt phase ordering instead of calling arbitrary cuts beats.
        cuts = np.sort(np.r_[beats, (beats[1:]+beats[:-1])/2])
        cuts = np.unique(np.round(cuts*SAMPLE_RATE).astype(int))
        changed = shuffle_chunks(y, cuts[(cuts>0)&(cuts<len(y))], rng)
    elif name in ("pitch_flatten", "world_resynthesis"):
        import pyworld
        f0, times = world_features(y)
        wave = y.astype(np.float64)
        spectrum = pyworld.cheaptrick(wave, f0, times, SAMPLE_RATE)
        aperiodicity = pyworld.d4c(wave, f0, times, SAMPLE_RATE)
        if name == "pitch_flatten":
            if not np.any(f0>0):
                return None, {**info, "status": "skipped", "reason": "no_voiced_frames"}
            f0[f0>0] = np.median(f0[f0>0])
        changed = pyworld.synthesize(f0, spectrum, aperiodicity, SAMPLE_RATE, frame_period=20.0)
        changed = librosa.util.fix_length(changed, size=len(y))
        info["control"] = "world_resynthesis" if name == "pitch_flatten" else "clean"
    elif name == "energy_normalize":
        from scipy.ndimage import gaussian_filter1d
        rms = np.exp(targets["log_energy"])
        active = rms > max(float(rms.max())*.05, 1e-6)
        target = np.median(rms[active]) if active.any() else 1e-6
        gain = np.where(active, np.clip(target/np.maximum(rms, 1e-6), .2, 5), 1)
        gain = gaussian_filter1d(gain, 2)
        changed = y * np.interp(np.arange(len(y))/SAMPLE_RATE, targets["time"], gain)
    else:
        raise ValueError(name)
    if changed is None:
        return None, {**info, "status": "skipped", "reason": "too_short_for_chunk_shuffle"}
    if not np.isfinite(changed).all():
        raise ValueError(f"Nonfinite intervention: {name}")
    return np.asarray(changed, dtype=np.float32), {**info, "status": "ready"}

def event_frame_labels(times, events, tolerance):
    if not len(events):
        return np.zeros(len(times), dtype=float)
    return (np.min(np.abs(np.asarray(times)[:,None]-np.asarray(events)[None,:]), axis=1) <= tolerance).astype(float)

def interval_labels(times, intervals, stress=False):
    import re
    labels = np.full(len(times), "", dtype=object)
    for item in intervals:
        mask = (times >= item["start"]) & (times < item["end"])
        value = item["label"]
        if value.lower() in {"sil", "sp", "spn", "<eps>"}:
            continue
        if stress:
            match = re.search(r"([012])$", value)
            if not match:
                continue
            value = match.group(1)
        else:
            value = re.sub(r"[012]$", "", value)
        labels[mask] = value
    return labels
