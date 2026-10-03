"""Tempo, beats, downbeats and the 48 Hz feature table (the base layer: librosa only).

Audio is decoded at 24 kHz so that a hop of 500 samples is exactly 1/48 s: feature frame i is t = i / 48.
"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SR = 24000
RATE = 48
HOP = SR // RATE            # 500
N_FFT = 2048
BANDS = [(40, 120), (120, 300), (300, 700), (700, 1500), (1500, 3000), (3000, 6000), (6000, 10000)]
BPM_RANGE = (70.0, 180.0)


# ---------------------------------------------------------------- small signal helpers

def norm(v: np.ndarray, q: float = 0.98) -> np.ndarray:
    """Scale so the q-quantile maps to 1, then clip to 0..1."""
    ref = float(np.quantile(v, q)) if len(v) else 0.0
    return np.clip(v / ref, 0, 1) if ref > 1e-9 else np.zeros_like(v)


def follow(v: np.ndarray, attack: float, release: float) -> np.ndarray:
    """Envelope follower: rises by `attack`, falls by `release` (fractions per frame)."""
    out = np.empty_like(v)
    s = 0.0
    for i, x in enumerate(v):
        s += (x - s) * (attack if x > s else release)
        out[i] = s
    return out


def onset_of(v: np.ndarray) -> np.ndarray:
    """Positive part of the signal above its own slow average: how much it just jumped."""
    return norm(np.maximum(0.0, v - follow(v, 0.08, 0.08)))


def r3(v) -> list[float]:
    return [round(float(x), 3) for x in v]


# ---------------------------------------------------------------- result

@dataclass
class AudioResult:
    duration: float
    bpm: float
    beats: list[float]
    downbeats: list[float]
    features: dict           # rate, loud, bands, kick, flux, cent
    onsets: list[float]      # onset times (s), used to snap word timings
    raw: dict                # unnormalised summaries for style analysis


def _fold_tempo(bpm: float) -> float:
    lo, hi = BPM_RANGE
    while bpm < lo:
        bpm *= 2
    while bpm > hi:
        bpm /= 2
    return bpm


def _fix_offbeat(beat_frames: np.ndarray, kick: np.ndarray, n: int) -> np.ndarray:
    """The tracker sometimes locks onto the off-beats (hi-hats between the kicks). If the bass onsets sit
    clearly stronger half a beat later, shift the whole grid by half a beat."""
    if len(beat_frames) < 8:
        return beat_frames
    half = int(round(np.median(np.diff(beat_frames)) / 2))

    def strength(frames: np.ndarray) -> float:
        f = frames[(frames >= 2) & (frames < n - 2)]
        return float(np.mean([kick[i - 2:i + 3].max() for i in f])) if len(f) else 0.0

    on, off = strength(beat_frames), strength(beat_frames + half)
    if off > on * 1.5 and off - on > 0.05:
        shifted = beat_frames + half
        return shifted[shifted < n]
    return beat_frames


def _downbeat_phase(beat_frames: np.ndarray, kick: np.ndarray, loud: np.ndarray) -> int:
    """Which of the 4 beat phases carries the downbeat: strongest kick + loudness change on it."""
    if len(beat_frames) < 8:
        return 0
    k = kick[np.clip(beat_frames, 0, len(kick) - 1)]
    rise = np.maximum(0, np.diff(loud, prepend=loud[0]))
    r = np.array([rise[max(0, f - 2):f + 3].max() for f in np.clip(beat_frames, 0, len(rise) - 1)])
    score = k + 0.5 * norm(r)
    return int(np.argmax([score[p::4].mean() for p in range(4)]))


def analyze_audio(y: np.ndarray, bpm_override: float | None = None) -> AudioResult:
    duration = len(y) / SR
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP, center=True)) ** 2   # power
    freqs = librosa.fft_frequencies(sr=SR, n_fft=N_FFT)
    n = S.shape[1]

    # loudness
    rms = librosa.feature.rms(S=np.sqrt(S), frame_length=N_FFT, hop_length=HOP)[0][:n]
    loud = follow(norm(rms), 0.6, 0.15)

    # bands (RMS inside each band)
    raw_bands = []
    for lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        raw_bands.append(norm(np.sqrt(S[m].mean(axis=0))))
    bands = [follow(b, 0.6, 0.18) for b in raw_bands]

    # onsets
    kick = follow(onset_of(raw_bands[0]), 0.9, 0.2)
    flux = follow(norm(np.sum([onset_of(b) for b in raw_bands], axis=0)), 0.9, 0.25)

    # centroid of the 300 Hz - 6 kHz bands on a log-frequency axis
    logs = np.array([np.log2(np.sqrt(lo * hi)) for lo, hi in BANDS[2:6]])
    w = np.array(raw_bands[2:6])
    wsum = w.sum(axis=0)
    cent = np.where(wsum > 1e-4, (w * logs[:, None]).sum(axis=0) / np.maximum(wsum, 1e-9), np.nan)
    known = cent[~np.isnan(cent)]
    if len(known):
        lo_c, hi_c = np.quantile(known, [0.05, 0.95])
        cent = np.where(np.isnan(cent), 0.5, np.clip((cent - lo_c) / max(1e-6, hi_c - lo_c), 0, 1))
    else:
        cent = np.full(n, 0.5)
    cent = follow(cent, 0.12, 0.12)

    # tempo and beats
    oenv = librosa.onset.onset_strength(S=librosa.power_to_db(S), sr=SR, hop_length=HOP)
    if bpm_override:
        bpm = float(bpm_override)
        _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, bpm=bpm, units="frames")
    else:
        tempo, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, units="frames")
        est = float(np.atleast_1d(tempo)[0])
        bpm = _fold_tempo(est)
        if abs(bpm - est) > 1:
            _, beat_frames = librosa.beat.beat_track(onset_envelope=oenv, sr=SR, hop_length=HOP, bpm=bpm,
                                                     units="frames")
        # refine the tempo from the beats themselves: median span of 8 beats (a single inter-beat
        # interval is quantised to 1/48 s, about +-3 BPM)
        if len(beat_frames) > 16:
            span = (beat_frames[8:] - beat_frames[:-8]) / 8.0
            bpm = _fold_tempo(60.0 * RATE / float(np.median(span)))
    beat_frames = _fix_offbeat(np.asarray(beat_frames), kick, n)
    beats = beat_frames / RATE
    phase = _downbeat_phase(beat_frames, kick, loud)
    downbeats = beats[phase::4]

    onset_frames = librosa.onset.onset_detect(onset_envelope=oenv, sr=SR, hop_length=HOP, units="frames",
                                              backtrack=False)
    rms_db = librosa.amplitude_to_db(rms, ref=1.0)
    raw = dict(rms_db_mean=float(np.mean(rms_db[rms_db > -60])) if np.any(rms_db > -60) else -60.0,
               rms_db_p95=float(np.quantile(rms_db, 0.95)),
               onset_rate=len(onset_frames) / max(1.0, duration),
               centroid_hz=float(np.mean(librosa.feature.spectral_centroid(S=np.sqrt(S), sr=SR))))

    return AudioResult(
        duration=round(duration, 3), bpm=round(bpm, 2),
        beats=[round(float(b), 4) for b in beats], downbeats=[round(float(b), 4) for b in downbeats],
        features=dict(rate=RATE, loud=r3(loud), bands=[r3(b) for b in bands], kick=r3(kick), flux=r3(flux),
                      cent=r3(cent)),
        onsets=[round(float(f) / RATE, 4) for f in onset_frames], raw=raw)
