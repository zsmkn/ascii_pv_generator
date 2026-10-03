"""Musical events for the Director (M10): build-ups, drops and breaks, on the bar grid.

Per bar, an intensity in 0..1 averages loudness, the bass stem, the drum stem and drum-hit density (the stems
come from demucs; without the ML layer the mix's bass band, loudness and onsets stand in). Then:

  drop    a downbeat where the next two bars are clearly more intense than the two before, and intense in
          absolute terms (the bass and drums come in: the moment a PV should explode)
  build   the bars right before a drop whose intensity keeps rising (drums getting denser, risers): tension
  break   a downbeat where intensity collapses (everything but a pad drops out): room for a calm shot

Events are dicts so analysis.json stays plain: {"kind", "t" (drop / break) or "start"/"end" (build), "strength"}.
"""

from __future__ import annotations

import numpy as np

DROP_JUMP = 0.18      # intensity rise (0..1) across a downbeat that counts as a drop
DROP_LEVEL = 0.35     # quantile of the song's intensity the two bars after a drop must reach
BREAK_FALL = 0.22
MIN_GAP = 8           # bars between two drops (or two breaks)
BUILD_MAX = 8         # bars
BUILD_RISE = 0.08     # total rise over a build


def intensity_curve(features: dict, stems: dict | None) -> np.ndarray:
    """0..1 per feature frame: mean of loudness, bass, drums and drum-hit density."""
    loud = np.asarray(features["loud"], float)
    if stems:
        parts = [loud, np.asarray(stems["bass"], float), np.asarray(stems["drums"], float),
                 np.clip(np.asarray(stems["kick"], float) + np.asarray(stems["snare"], float), 0, 1)]
    else:
        parts = [loud, np.asarray(features["bands"][0], float), loud,
                 np.clip(np.asarray(features["kick"], float) + np.asarray(features["flux"], float), 0, 1)]
    n = min(len(p) for p in parts)
    return np.mean([p[:n] for p in parts], axis=0)


def detect(features: dict, stems: dict | None, downbeats: list[float], duration: float, bpm: float
           ) -> list[dict]:
    """Candidates are the downbeats; each is judged on fixed-length windows (bars of 4 beats at the song's tempo)
    rather than on neighbouring downbeats, which the tracker sometimes places irregularly."""
    rate = features["rate"]
    curve = intensity_curve(features, stems)
    cum = np.concatenate([[0.0], np.cumsum(curve)])
    n = len(curve)
    bar = 4 * 60.0 / bpm

    def mean(t0: float, t1: float) -> float:
        a, b = int(np.clip(t0 * rate, 0, n)), int(np.clip(t1 * rate, 0, n))
        return (cum[b] - cum[a]) / (b - a) if b > a else np.nan

    cand = np.asarray([d for d in downbeats if 2 * bar <= d <= duration - 2 * bar], float)
    if len(cand) < 4:
        return []
    before = np.array([mean(t - 2 * bar, t) for t in cand])
    after = np.array([mean(t, t + 2 * bar) for t in cand])
    jump = np.nan_to_num(after - before)
    level = float(np.quantile(curve, DROP_LEVEL))
    events: list[dict] = []
    for k in _peaks(np.where(after >= level, jump, 0.0), cand, DROP_JUMP, MIN_GAP * bar):
        t = float(cand[k])
        events.append({"kind": "drop", "t": round(t, 3), "strength": round(float(jump[k]), 3)})
        # build: bar-long windows walking back from the drop while the intensity keeps rising
        I = [mean(t - (j + 1) * bar, t - j * bar) for j in range(BUILD_MAX + 1)]   # I[0] = the bar before
        m = 0
        while m < BUILD_MAX and t - (m + 2) * bar >= 0 and I[m] >= I[m + 1] - 0.03:
            m += 1
        if m >= 2 and I[0] - I[m - 1] >= BUILD_RISE:
            events.append({"kind": "build", "start": round(t - m * bar, 3), "end": round(t, 3),
                           "strength": round(float(I[0] - I[m - 1]), 3)})
    for k in _peaks(-jump, cand, BREAK_FALL, MIN_GAP * bar):
        events.append({"kind": "break", "t": round(float(cand[k]), 3), "strength": round(float(-jump[k]), 3)})
    return sorted(events, key=lambda e: e.get("t", e.get("start")))


def _peaks(score: np.ndarray, times: np.ndarray, thr: float, gap: float) -> list[int]:
    """Indices above thr, strongest first, at least `gap` seconds apart."""
    out: list[int] = []
    for k in np.argsort(-score):
        if score[k] < thr:
            break
        if all(abs(times[k] - times[j]) >= gap for j in out):
            out.append(int(k))
    return sorted(out)
