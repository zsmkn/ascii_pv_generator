"""Beats and downbeats with beat_this (CPJKU). Falls back to the librosa result when unavailable."""

from __future__ import annotations

import numpy as np


def beats_downbeats(y: np.ndarray, sr: int, device: str = "cuda") -> tuple[list[float], list[float]]:
    from beat_this.inference import Audio2Beats

    a2b = Audio2Beats(checkpoint_path="final0", device=device, dbn=False)
    beats, downbeats = a2b(y, sr)
    return [round(float(b), 4) for b in beats], [round(float(d), 4) for d in downbeats]


def bpm_of(beats: list[float]) -> float:
    b = np.asarray(beats)
    if len(b) < 17:
        return 60.0 / float(np.median(np.diff(b))) if len(b) > 1 else 120.0
    return 60.0 / float(np.median((b[8:] - b[:-8]) / 8))
