"""Audio analysis on synthetic signals (no ffmpeg needed: analyze_audio takes samples)."""

import numpy as np
import pytest

librosa = pytest.importorskip("librosa")

from ascii_pv.analysis.audio import RATE, SR, analyze_audio  # noqa: E402


def drum_loop(bpm: float, seconds: float, accent_every: int = 4) -> np.ndarray:
    """Kick on every beat (louder on the bar), noisy hat on the off-beats."""
    rng = np.random.default_rng(0)
    y = np.zeros(int(seconds * SR), np.float32)
    period = 60.0 / bpm
    kick_t = np.arange(int(0.12 * SR)) / SR
    kick = np.sin(2 * np.pi * 60 * kick_t * (1 + 2 * np.exp(-kick_t * 30))) * np.exp(-kick_t * 25)
    hat = rng.standard_normal(int(0.03 * SR)) * np.exp(-np.arange(int(0.03 * SR)) / SR * 150) * 0.3
    for i in range(int(seconds / period)):
        s = int(i * period * SR)
        amp = 1.0 if i % accent_every == 0 else 0.6
        y[s:s + len(kick)] += (amp * kick)[: len(y) - s]
        h = int((i + 0.5) * period * SR)
        y[h:h + len(hat)] += hat[: max(0, len(y) - h)]
    return y


@pytest.mark.parametrize("bpm", [96.0, 128.0, 150.0])
def test_tempo_and_beats(bpm):
    r = analyze_audio(drum_loop(bpm, 40))
    assert abs(r.bpm - bpm) < 1.5, r.bpm
    ibi = np.diff(r.beats)
    assert abs(np.median(ibi) - 60 / bpm) < 0.03


def test_features_shape_and_range():
    r = analyze_audio(drum_loop(120, 20))
    f = r.features
    n = len(f["loud"])
    assert f["rate"] == RATE and abs(n - 20 * RATE) <= 2
    assert len(f["bands"]) == 7 and all(len(b) == n for b in f["bands"])
    for k in ("loud", "kick", "flux", "cent"):
        v = np.asarray(f[k])
        assert v.min() >= 0 and v.max() <= 1


def test_downbeats_land_on_accents():
    bpm = 120.0
    r = analyze_audio(drum_loop(bpm, 40))
    bar = 4 * 60 / bpm
    # accented kicks are at multiples of the bar length
    phase = [(d % bar) / bar for d in r.downbeats[2:-2]]
    assert np.median([min(p, 1 - p) for p in phase]) < 0.1
