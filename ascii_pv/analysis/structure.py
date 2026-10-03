"""Section boundaries and labels (base layer: librosa only).

1. Beat-synchronous chroma + MFCC -> self-similarity -> checkerboard novelty; peaks become boundaries,
   snapped to downbeats, at least MIN_BARS apart.
2. Segments are clustered by their mean features into repeat groups (A, B, C, ...).
3. Labels from rules: lyric-free start/end -> intro/outro, lyric-free middle -> instrumental; among sung
   segments the loud, repeating group that holds hook lines is the chorus; a sung group that occurs once late
   in the song is the bridge; the segment just before a chorus that is louder than the verse is a prechorus.

Labels are hints for the Director (where to cut, how intense), not ground truth; storyboard.json can override.
"""

from __future__ import annotations

import bisect

import librosa
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from ascii_pv.analysis.audio import HOP, SR
from ascii_pv.core.analysis import LyricLine, Section

MIN_BARS = 4
KERNEL_BEATS = 16


def _checkerboard(n: int) -> np.ndarray:
    x = np.arange(-n, n) + 0.5
    g = np.exp(-(x / (n / 2)) ** 2)
    k = np.outer(np.sign(x) * g, np.sign(x) * g)
    return k


def _novelty(ssm: np.ndarray, half: int) -> np.ndarray:
    n = len(ssm)
    k = _checkerboard(half)
    pad = np.pad(ssm, half, mode="constant")
    return np.array([np.sum(pad[i:i + 2 * half, i:i + 2 * half] * k) for i in range(n)])


def _snap(t: float, grid: list[float]) -> float:
    i = bisect.bisect_left(grid, t)
    cands = [grid[j] for j in (i - 1, i) if 0 <= j < len(grid)]
    return min(cands, key=lambda g: abs(g - t)) if cands else t


def boundaries(y: np.ndarray, beats: list[float], downbeats: list[float], bpm: float, duration: float) -> list[float]:
    if len(beats) < KERNEL_BEATS * 2:
        return [0.0, duration]
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)[1:]
    frames = librosa.time_to_frames(beats, sr=SR, hop_length=HOP)
    feats = []
    for f in (chroma, mfcc):
        s = librosa.util.sync(f, frames, aggregate=np.median)
        s = (s - s.mean(axis=1, keepdims=True)) / (s.std(axis=1, keepdims=True) + 1e-6)
        feats.append(s)
    x = np.vstack(feats).T                                   # (beats+1, dims)
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    ssm = median_filter(x @ x.T, size=(1, 5))                # smooth along diagonals a little
    nov = _novelty(ssm, KERNEL_BEATS // 2)
    nov = np.maximum(0, nov)
    nov /= nov.max() + 1e-9
    beat_times = [0.0] + list(beats)                          # sync() puts a segment before the first beat
    min_gap_beats = MIN_BARS * 4
    peaks, _ = find_peaks(nov, distance=min_gap_beats, height=0.15, prominence=0.1)
    grid = downbeats or beats
    cuts = sorted({round(_snap(beat_times[min(p, len(beat_times) - 1)], grid), 3) for p in peaks})
    min_gap = MIN_BARS * 4 * 60.0 / bpm
    out = [0.0]
    for c in cuts:
        if c - out[-1] >= min_gap and duration - c >= min_gap * 0.5:
            out.append(c)
    out.append(duration)
    return out


def _segment_features(y: np.ndarray, bounds: list[float]) -> np.ndarray:
    """Per-segment mean of song-normalised chroma + MFCC. Normalising over the whole song first matters:
    raw means of one song's segments are all nearly parallel, so everything would look like one group."""
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)
    mfcc = librosa.feature.mfcc(y=y, sr=SR, hop_length=HOP, n_mfcc=13)[1:]
    f = np.vstack([chroma, mfcc])
    f = (f - f.mean(axis=1, keepdims=True)) / (f.std(axis=1, keepdims=True) + 1e-6)
    rows = []
    for a, b in zip(bounds, bounds[1:]):
        fa, fb = librosa.time_to_frames([a, b], sr=SR, hop_length=HOP)
        rows.append(f[:, fa:max(fb, fa + 1)].mean(axis=1))
    x = np.array(rows)
    x = x - x.mean(axis=0, keepdims=True)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _silhouette(d: np.ndarray, lab: np.ndarray) -> float:
    s = []
    for i in range(len(lab)):
        same = (lab == lab[i])
        same[i] = False
        if not same.any():
            s.append(0.0)
            continue
        a = d[i, same].mean()
        b = min(d[i, lab == g].mean() for g in set(lab) if g != lab[i])
        s.append((b - a) / max(a, b, 1e-9))
    return float(np.mean(s))


def _groups(feats: np.ndarray) -> list[int]:
    """Average-linkage clustering on cosine distance; the group count (2..6) with the best silhouette."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import pdist, squareform

    n = len(feats)
    if n < 3:
        return list(range(n))
    dv = pdist(feats, "cosine")
    d = squareform(dv)
    z = linkage(dv, "average")
    best, best_lab = -2.0, np.zeros(n, int)
    for k in range(2, min(6, n - 1) + 1):
        lab = fcluster(z, k, "maxclust")
        if len(set(lab)) < 2:
            continue
        sc = _silhouette(d, lab)
        if sc > best:
            best, best_lab = sc, lab
    # renumber in order of first appearance: A, B, C ...
    order: dict[int, int] = {}
    return [order.setdefault(int(g), len(order)) for g in best_lab]


def label_sections(y: np.ndarray, bounds: list[float], loud: list[float], rate: int,
                   lyrics: list[LyricLine]) -> list[Section]:
    n = len(bounds) - 1
    if n <= 0:
        return []
    feats = _segment_features(y, bounds)
    group = _groups(feats)
    lv = np.asarray(loud)
    energy = [float(lv[int(a * rate):max(int(a * rate) + 1, int(b * rate))].mean()) for a, b in zip(bounds, bounds[1:])]

    def sung(a: float, b: float) -> float:
        """Share of the segment covered by lyric lines."""
        cover = sum(max(0.0, min(b, ln.end) - max(a, ln.start)) for ln in lyrics)
        return cover / max(1e-6, b - a)

    def hooks(a: float, b: float) -> int:
        return sum(1 for ln in lyrics if ln.hook and a <= ln.start < b)

    sung_share = [sung(a, b) for a, b in zip(bounds, bounds[1:])]
    has_lyrics = bool(lyrics)
    vocal = [s > 0.2 for s in sung_share] if has_lyrics else [True] * n
    labels = [""] * n

    # intro / outro / instrumental
    first = next((i for i in range(n) if vocal[i]), n)
    last = max((i for i in range(n) if vocal[i]), default=-1)
    for i in range(n):
        if not vocal[i]:
            labels[i] = "intro" if i < first else "outro" if i > last else "instrumental"
    if not has_lyrics:
        if n >= 3 and energy[0] < np.median(energy):
            labels[0] = "intro"
        if n >= 3 and energy[-1] < np.median(energy):
            labels[-1] = "outro"

    # chorus: score repeat groups among sung segments
    sung_idx = [i for i in range(n) if not labels[i]]
    if sung_idx:
        mean_e = np.mean([energy[i] for i in sung_idx])
        scores = {}
        for g in {group[i] for i in sung_idx}:
            members = [i for i in sung_idx if group[i] == g]
            e = np.mean([energy[i] for i in members])
            h = sum(hooks(bounds[i], bounds[i + 1]) for i in members)
            scores[g] = (e / (mean_e + 1e-9)) + 0.35 * (len(members) - 1) + 0.25 * min(h, 4)
        # a chorus group must not cover most of the sung time (that is just "the song")
        sung_time = sum(bounds[i + 1] - bounds[i] for i in sung_idx)
        share = {g: sum(bounds[i + 1] - bounds[i] for i in sung_idx if group[i] == g) / sung_time for g in scores}
        allowed = [g for g in scores if share[g] <= 0.6] or list(scores)
        chorus_g = max(allowed, key=scores.get)
        # ... and needs to stand out (louder or hookier than the rest)
        if len(scores) == 1 or scores[chorus_g] > 1.1:
            for i in sung_idx:
                if group[i] == chorus_g:
                    labels[i] = "chorus"
        # per-segment evidence the group vote misses:
        #  - a hook-dense segment that is not quieter than usual is a chorus even inside a big group
        #  - a segment glued to a chorus and as loud as it is the chorus's other half
        #    ("hook-dense" is relative: in songs that repeat nearly every line it means little)
        med_e = float(np.median([energy[i] for i in sung_idx]))
        hook_share = sum(ln.hook for ln in lyrics) / max(1, len(lyrics))
        need = max(0.75, hook_share + 0.25)
        for i in sung_idx:
            lines_in = [ln for ln in lyrics if bounds[i] <= ln.start < bounds[i + 1]]
            if len(lines_in) >= 3 and sum(ln.hook for ln in lines_in) / len(lines_in) >= need and energy[i] >= med_e:
                labels[i] = "chorus"
        chorus_e = [energy[i] for i in sung_idx if labels[i] == "chorus"]
        spread = max(energy[i] for i in sung_idx) - min(energy[i] for i in sung_idx)
        if chorus_e and spread >= 0.2:   # in a flat song "as loud as the chorus" says nothing
            ce = float(np.mean(chorus_e))
            core = [lab == "chorus" for lab in labels]   # snapshot: no chaining along the song
            for i in sung_idx:
                if labels[i]:
                    continue
                near = [j for j in (i - 1, i + 1) if 0 <= j < n and core[j]]
                if near and energy[i] >= ce * 0.95 and energy[i] >= med_e:
                    labels[i] = "chorus"
        verse_e = np.mean([energy[i] for i in sung_idx if labels[i] != "chorus"] or [mean_e])
        counts = {g: sum(1 for i in sung_idx if group[i] == g) for g in set(group)}
        for i in sung_idx:
            if labels[i]:
                continue
            nxt = labels[i + 1] if i + 1 < n else ""
            if nxt == "chorus" and energy[i] > verse_e * 1.05 and i > 0 and labels[i - 1] != "chorus":
                labels[i] = "prechorus"
            elif counts[group[i]] == 1 and i > n * 0.55 and any(labels[j] == "chorus" for j in range(i)):
                labels[i] = "bridge"
            else:
                labels[i] = "verse"

    return [Section(round(a, 3), round(b, 3), lab, round(e, 3), chr(65 + g))
            for a, b, lab, e, g in zip(bounds, bounds[1:], labels, energy, group)]
