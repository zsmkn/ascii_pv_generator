"""Lyric sync against the separated vocal track.

A lyric file timed on another cut of the song is off by a constant (offset) and often a speed factor (scale):
t_true = scale * t_file + offset. With the vocals isolated:

  1. global   grid-search scale x offset maximising the correlation between "a lyric line is up" and "someone
              is singing". Where the singing pauses (breaks, breaths between phrases) is what pins the timing;
              vocal onsets alone are useless in rap, where the voice barely stops.
  2. confidence   repeat the search with shuffled line gaps (same count, same spacing statistics); the real fit
              must clearly beat that null, otherwise nothing is changed
  3. fine     near the global fit (+-FINE_SCALE, +-FINE_OFFSET), maximise how many line starts sit on a vocal
              onset. The span correlation is broad and can settle ~1% off in scale, which drifts >1 s by the end
              of a song; onsets pin it exactly. Only an accurately timed file (e.g. a Spotify LRC) carries that
              information, so it is used only when it clearly beats the same lines randomly jittered.
  4. local    each line start snaps to a clear vocal onset within +-LOCAL s (if there is one), keeping order

Everything here is numpy only; the vocal envelope comes from analysis/ml/stems.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import maximum_filter1d, uniform_filter1d

SCALES = np.arange(0.85, 1.1501, 0.0025)
OFFSETS = np.arange(-12.0, 12.0001, 0.02)
LOCAL = 0.4
MIN_Z = 3.0          # how far above the null the fit must be (in null standard deviations)
FINE_SCALE = 0.015
FINE_OFFSET = 1.0
FINE_TOL = 0.1       # a line start "hits" an onset within +-0.1 s
FINE_JITTER = 0.5


@dataclass
class SyncResult:
    scale: float
    offset: float
    score: float
    null_mean: float
    null_std: float
    applied: bool

    @property
    def z(self) -> float:
        return (self.score - self.null_mean) / max(1e-9, self.null_std)

    def describe(self, what: str = "lyric sync") -> str:
        verdict = "applied" if self.applied else "not applied (not confident)"
        return (f"{what}: scale {self.scale:.4f}, offset {self.offset:+.2f}s, "
                f"z={self.z:.1f} -> {verdict}")


def onset_envelope(vocal_rms: np.ndarray, rate: int) -> np.ndarray:
    """Where the voice starts: positive jumps of the (log) vocal level, z-scored, widened a little."""
    db = 20 * np.log10(np.maximum(vocal_rms, 1e-4))
    smooth = uniform_filter1d(db, max(1, rate // 24))
    rise = np.maximum(0.0, np.diff(smooth, prepend=smooth[0]))
    slow = uniform_filter1d(rise, rate)                       # remove the local average
    env = np.maximum(0.0, rise - slow)
    env = (env - env.mean()) / (env.std() + 1e-9)
    return maximum_filter1d(env, max(1, rate // 12))          # +-40 ms tolerance


def vocal_activity(vocal_rms: np.ndarray, rate: int) -> np.ndarray:
    """z-scored "singing now" curve: vocal level above its 35th percentile, smoothed over 1/4 s."""
    db = 20 * np.log10(np.maximum(vocal_rms, 1e-4))
    act = uniform_filter1d((db > np.quantile(db, 0.35)).astype(float), max(1, rate // 4))
    return (act - act.mean()) / (act.std() + 1e-9)


def _best(act: np.ndarray, rate: int, starts: np.ndarray, ends: np.ndarray) -> tuple[float, float, float]:
    """Max over (scale, offset) of the correlation between the lyric-span mask and the activity curve."""
    n = len(act)
    cum = np.concatenate([[0.0], np.cumsum(act)])
    best = (-np.inf, 1.0, 0.0)
    for a in SCALES:
        s0 = np.clip(np.rint((a * starts[:, None] + OFFSETS[None, :]) * rate), 0, n).astype(int)
        s1 = np.clip(np.rint((a * ends[:, None] + OFFSETS[None, :]) * rate), 0, n).astype(int)
        k = np.maximum(0, s1 - s0).sum(axis=0)                  # frames covered by lyrics
        inside = (cum[s1] - cum[s0]).sum(axis=0)                # sum of activity under the lyrics
        p = k / n
        score = np.where((p > 0.05) & (p < 0.95), inside / n / np.sqrt(p * (1 - p) + 1e-12), -np.inf)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), float(a), float(OFFSETS[j]))
    return best


def _spans(starts: np.ndarray, lens: np.ndarray) -> np.ndarray:
    """Line ends: at most ~5 s sung, and never past the next line (spans must not overlap, or the mask
    correlation counts the overlap twice)."""
    e = starts + np.minimum(lens, 5.0)
    e[:-1] = np.minimum(e[:-1], starts[1:])
    return e


def fit(starts: list[float], ends: list[float], act: np.ndarray, rate: int, trials: int = 12,
        seed: int = 0) -> SyncResult:
    s = np.asarray(starts, float)
    e = _spans(s, np.asarray(ends, float) - s)
    if len(s) < 6:
        return SyncResult(1.0, 0.0, 0.0, 0.0, 1.0, False)
    score, a, b = _best(act, rate, s, e)
    rng = np.random.default_rng(seed)
    gaps, lens = np.diff(s), e - s
    null = []
    for _ in range(trials):
        fs = s[0] + np.concatenate([[0], np.cumsum(rng.permutation(gaps))])
        null.append(_best(act, rate, fs, _spans(fs, rng.permutation(lens)))[0])
    res = SyncResult(a, b, score, float(np.mean(null)), float(np.std(null)) + 1e-6, False)
    res.applied = res.z >= MIN_Z
    return res


def _best_onset(hit: np.ndarray, rate: int, starts: np.ndarray, scale: float, offset: float
                ) -> tuple[float, float, float]:
    """Max over the neighbourhood of (scale, offset) of the mean onset strength at the line starts."""
    A = np.arange(scale - FINE_SCALE, scale + FINE_SCALE + 1e-9, 0.0005)
    B = np.arange(offset - FINE_OFFSET, offset + FINE_OFFSET + 1e-9, 0.01)
    best = (-np.inf, scale, offset)
    for a in A:
        idx = np.clip(np.rint((a * starts[None, :] + B[:, None]) * rate).astype(int), 0, len(hit) - 1)
        score = hit[idx].mean(axis=1)
        j = int(np.argmax(score))
        if score[j] > best[0]:
            best = (float(score[j]), float(a), float(B[j]))
    return best


def fine(starts: list[float], env: np.ndarray, rate: int, scale: float, offset: float, trials: int = 12,
         seed: int = 0) -> SyncResult:
    """Stage 3: tighten a global fit on the vocal onsets (env from onset_envelope)."""
    s = np.asarray(starts, float)
    if len(s) < 6:
        return SyncResult(scale, offset, 0.0, 0.0, 1.0, False)
    # triangle kernel of half-width FINE_TOL: tolerant, but still peaks on exact alignment (a flat window makes
    # a plateau and argmax lands on its edge); clipped so one huge onset does not outweigh many ordinary ones
    w = max(1, int(FINE_TOL * rate))
    hit = uniform_filter1d(uniform_filter1d(np.minimum(np.maximum(env, 0.0), 4.0), w), w)
    score, a, b = _best_onset(hit, rate, s, scale, offset)
    # null: the same lines, each jittered by up to +-FINE_JITTER. The coarse structure is already established
    # (stage 2); this asks whether the file's per-line precision is real. Sloppy timing (ASR-made LRCs) scores
    # the same jittered or not and is left alone.
    rng = np.random.default_rng(seed)
    null = [_best_onset(hit, rate, s + rng.uniform(-FINE_JITTER, FINE_JITTER, len(s)), scale, offset)[0]
            for _ in range(trials)]
    res = SyncResult(a, b, score, float(np.mean(null)), float(np.std(null)) + 1e-6, False)
    res.applied = res.z >= MIN_Z
    return res


def refine(starts: list[float], env: np.ndarray, rate: int, window: float = LOCAL) -> list[float]:
    """Snap each start to the strongest onset nearby (only if clearly there), keeping the order."""
    out, prev = [], -np.inf
    w = int(window * rate)
    for t in starts:
        c = int(round(t * rate))
        lo, hi = max(0, c - w), min(len(env), c + w + 1)
        new = t
        if hi > lo:
            seg = env[lo:hi]
            k = int(np.argmax(seg))
            if seg[k] > 1.5:
                new = (lo + k) / rate
        new = max(new, prev + 0.15)
        out.append(round(new, 3))
        prev = new
    return out
