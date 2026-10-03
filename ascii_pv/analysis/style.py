"""Rule-based style: three 0..1 axes plus a few tags (the ML layer adds CLAP tags in M5).

  energy   how hard it drives: loudness, onset rate, tempo
  valence  how bright/happy it sounds: major vs minor key, spectral brightness, tempo
  density  how busy it is: onset rate, percussive share
"""

from __future__ import annotations

import librosa
import numpy as np

from ascii_pv.analysis.audio import SR

# Krumhansl-Kessler key profiles
MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def lin(x: float, lo: float, hi: float) -> float:
    return float(np.clip((x - lo) / (hi - lo), 0, 1))


def key_of(y: np.ndarray) -> tuple[str, float]:
    """-> ("A minor", mode) where mode is +1 clearly major .. -1 clearly minor."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR).mean(axis=1)
    best = (-2.0, "", 0)
    corr = {}
    for mode, prof in (("major", MAJOR), ("minor", MINOR)):
        for k in range(12):
            r = float(np.corrcoef(chroma, np.roll(prof, k))[0, 1])
            corr[(mode, k)] = r
            if r > best[0]:
                best = (r, mode, k)
    r, mode, k = best
    other = max(v for (m, _), v in corr.items() if m != mode)
    margin = np.clip((r - other) * 5, 0, 1)
    return f"{NOTES[k]} {mode}", float(margin if mode == "major" else -margin)


def analyze_style(y: np.ndarray, bpm: float, raw: dict) -> dict:
    harm, perc = librosa.effects.hpss(y)
    perc_share = float(np.sum(perc ** 2) / (np.sum(harm ** 2) + np.sum(perc ** 2) + 1e-9))
    key, mode = key_of(harm)

    loud = lin(raw["rms_db_mean"], -24, -8)
    onset = lin(raw["onset_rate"], 1.0, 5.0)
    tempo = lin(bpm, 70, 170)
    bright = lin(raw["centroid_hz"], 1200, 3200)

    energy = 0.4 * loud + 0.35 * onset + 0.25 * tempo
    valence = 0.5 * (mode + 1) / 2 + 0.3 * bright + 0.2 * tempo
    density = 0.6 * onset + 0.4 * lin(perc_share, 0.15, 0.5)

    tags = []
    tags.append("energetic" if energy > 0.62 else "calm" if energy < 0.38 else "mid-energy")
    tags.append("bright" if valence > 0.58 else "dark" if valence < 0.42 else "neutral")
    tags.append("busy" if density > 0.6 else "sparse" if density < 0.35 else "")
    if perc_share > 0.35:
        tags.append("percussive")
    return dict(energy=round(energy, 3), valence=round(valence, 3), density=round(density, 3),
                key=key, mode=round(mode, 3), percussive=round(perc_share, 3),
                tags=[t for t in tags if t])
