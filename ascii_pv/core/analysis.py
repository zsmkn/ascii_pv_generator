"""The song as data: what the analyzer writes to analysis.json and what every frame reads.

Time series are sampled at `features.rate` Hz (48 by default) and normalised to 0..1:
  loud   overall RMS
  bands  7 bands, bass (40-120 Hz) .. air (6-10 kHz)
  kick   onset strength in the bass band
  flux   onset strength over all bands
  cent   spectral centroid, 0 low .. 1 high

With the ML layer, `stems` (same rate) adds the demucs stems: drums / bass / other levels and drum-stem hits
(kick, snare). `events` lists build-ups, drops and breaks (analysis/events.py). Features falls back to mix
features for the stem fields, so templates can use them unconditionally.
"""

from __future__ import annotations

import bisect
import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

N_BANDS = 7


@dataclass
class Word:
    start: float
    end: float
    text: str


@dataclass
class LyricLine:
    start: float
    end: float
    text: str
    words: list[Word] = field(default_factory=list)
    hook: bool = False                                   # the line repeats (chorus / refrain material)
    motifs: list[str] = field(default_factory=list)      # visual motifs its words trigger (love, fire, ...)

    def word_at(self, t: float) -> int:
        """Index of the word being sung at t (the last started one), -1 before the first."""
        return bisect.bisect_right([w.start for w in self.words], t) - 1


@dataclass
class Section:
    start: float
    end: float
    label: str          # intro / verse / prechorus / chorus / bridge / outro / instrumental
    energy: float = 0.5
    group: str = ""     # repeat group (A, B, ...): segments that sound alike share a letter


@dataclass
class Features:
    loud: float
    bands: list[float]
    kick: float
    flux: float
    cent: float
    vocal: float
    drums: float = 0.0        # drum stem level (mix loudness without stems)
    bass: float = 0.0         # bass stem level (the 40-120 Hz band without stems)
    other: float = 0.0        # everything but drums, bass, vocals (loudness without stems)
    drum_kick: float = 0.0    # kick hits on the drum stem (mix kick without stems)
    drum_snare: float = 0.0   # snare / hat hits on the drum stem (mix flux without stems)


@dataclass
class Analysis:
    duration: float
    bpm: float
    beats: list[float]
    downbeats: list[float]
    sections: list[Section]
    style: dict
    features: dict                      # {"rate": 48, "loud": [...], "bands": [[...] * 7], ...}
    lyrics: list[LyricLine] = field(default_factory=list)
    vocal: dict | None = None           # {"rate": 48, "rms": [...]}; None without the ML layer
    source: dict = field(default_factory=dict)  # audio path, sha256, lyric path, ...
    stems: dict | None = None           # {"rate": 48, "drums", "bass", "other", "kick", "snare"}; ML layer only
    events: list[dict] = field(default_factory=list)   # [{"kind": build|drop|break, "t" | "start"+"end", ...}]

    # ------------------------------------------------------------ lookup

    def _idx(self, t: float, rate: float, n: int) -> int:
        return max(0, min(n - 1, int(t * rate)))

    def feat(self, t: float) -> Features:
        f = self.features
        i = self._idx(t, f["rate"], len(f["loud"]))
        v = 0.0
        if self.vocal:
            v = self.vocal["rms"][self._idx(t, self.vocal["rate"], len(self.vocal["rms"]))]
        if self.stems:
            s = self.stems
            j = self._idx(t, s["rate"], len(s["drums"]))
            extra = (s["drums"][j], s["bass"][j], s["other"][j], s["kick"][j], s["snare"][j])
        else:
            extra = (f["loud"][i], f["bands"][0][i], f["loud"][i], f["kick"][i], f["flux"][i])
        return Features(f["loud"][i], [b[i] for b in f["bands"]], f["kick"][i], f["flux"][i], f["cent"][i], v, *extra)

    def build_progress(self, t: float) -> float | None:
        """0 -> 1 through a build-up containing t, None outside builds."""
        for e in self.events:
            if e["kind"] == "build" and e["start"] <= t < e["end"]:
                return (t - e["start"]) / max(1e-6, e["end"] - e["start"])
        return None

    def since(self, kind: str, t: float) -> float:
        """Seconds since the last `kind` event (drop / break) at or before t; inf if none."""
        best = math.inf
        for e in self.events:
            if e["kind"] == kind and e["t"] <= t:
                best = min(best, t - e["t"])
        return best

    def beat_info(self, t: float) -> tuple[int, float, float]:
        """(beat index, phase 0..1 inside the beat, time of that beat). Extrapolates past both ends."""
        b = self.beats
        period = 60.0 / self.bpm
        if not b:
            i = math.floor(t / period)
            return i, (t / period) - i, i * period
        if t < b[0]:
            i = math.floor((t - b[0]) / period)
            bt = b[0] + i * period
            return i, (t - bt) / period, bt
        i = bisect.bisect_right(b, t) - 1
        if i + 1 < len(b):
            span = b[i + 1] - b[i]
        else:
            k = math.floor((t - b[-1]) / period)
            i, bt = len(b) - 1 + k, b[-1] + k * period
            return i, (t - bt) / period, bt
        return i, (t - b[i]) / span if span > 0 else 0.0, b[i]

    def bar_index(self, t: float) -> int:
        if self.downbeats:
            return bisect.bisect_right(self.downbeats, t) - 1
        return self.beat_info(t)[0] // 4

    def travel(self, t: float, base: float = 0.3) -> float:
        """Distance travelled by t when speed = base + loudness: lets templates move faster in loud passages
        and still be pure functions of t (the integral is precomputed once)."""
        cum = self.__dict__.get("_travel")
        if cum is None:
            f = self.features
            loud = np.asarray(f["loud"], float)
            cum = np.concatenate([[0.0], np.cumsum(base + loud) / f["rate"]])
            self.__dict__["_travel"] = cum
        x = t * self.features["rate"]
        i = int(np.clip(np.floor(x), 0, len(cum) - 2))
        return float(cum[i] + (cum[i + 1] - cum[i]) * (x - i)) + max(0.0, t - self.duration) * base

    def section_at(self, t: float) -> Section | None:
        for s in self.sections:
            if s.start <= t < s.end:
                return s
        return None

    def line_at(self, t: float) -> int:
        """Index of the lyric line active at t, or -1."""
        i = bisect.bisect_right([ln.start for ln in self.lyrics], t) - 1
        if i >= 0 and t < self.lyrics[i].end:
            return i
        return -1

    # ------------------------------------------------------------ io

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Analysis":
        d = dict(d)
        d["sections"] = [Section(**s) for s in d.get("sections", [])]
        d["lyrics"] = [LyricLine(**{**ln, "words": [Word(**w) for w in ln.get("words", [])]})
                       for ln in d.get("lyrics", [])]
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Analysis":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------ synthetic

    @classmethod
    def synthetic(cls, duration: float = 60.0, bpm: float = 120.0, seed: int = 0, rate: int = 48) -> "Analysis":
        """A fake but musically shaped song (kicks on the beat, louder choruses) for tests and demos."""
        rng = random.Random(seed)
        period = 60.0 / bpm
        beats = [round(i * period, 4) for i in range(int(duration / period) + 1)]
        downbeats = beats[::4]
        # 8-bar sections: intro verse chorus verse chorus outro ...
        bar = period * 4
        labels = ["intro", "verse", "chorus", "verse", "chorus", "bridge", "chorus", "outro"]
        sections, t, k = [], 0.0, 0
        while t < duration:
            end = min(duration, t + bar * 8)
            lab = labels[min(k, len(labels) - 1)] if end < duration else "outro"
            group = {"intro": "A", "verse": "B", "chorus": "C", "bridge": "D", "outro": "E"}[lab]
            sections.append(Section(round(t, 4), round(end, 4), lab, 0.9 if lab == "chorus" else 0.5, group))
            t, k = end, k + 1
        n = int(duration * rate)
        loud, kick, flux, cent = [], [], [], []
        bands = [[] for _ in range(N_BANDS)]
        for i in range(n):
            tt = i / rate
            sec = next((s for s in sections if s.start <= tt < s.end), sections[-1])
            base = 0.35 + 0.5 * sec.energy
            ph = (tt % period) / period
            k_ = math.exp(-ph * 9)
            loud.append(round(min(1.0, base * (0.7 + 0.3 * k_)), 3))
            kick.append(round(k_, 3))
            flux.append(round(min(1.0, k_ * 0.8 + rng.random() * 0.2), 3))
            cent.append(round(0.5 + 0.3 * math.sin(tt * 0.3), 3))
            for b in range(N_BANDS):
                wobble = 0.5 + 0.5 * math.sin(tt * (1.3 + b * 0.7) + b)
                e = base * (k_ if b == 0 else 0.4 + 0.6 * wobble) * (1.0 - b * 0.06)
                bands[b].append(round(min(1.0, max(0.0, e + rng.uniform(-0.05, 0.05))), 3))
        words = "hello world this is a demo line for the ascii pv generator".split()
        lyrics = []
        for s in sections:
            if s.label in ("intro", "outro"):
                continue
            t0 = s.start + bar
            while t0 + bar * 2 <= s.end:
                ws = [words[(int(t0) + j) % len(words)] for j in range(4)]
                wt = [Word(round(t0 + j * period, 4), round(t0 + (j + 1) * period, 4), w) for j, w in enumerate(ws)]
                lyrics.append(LyricLine(wt[0].start, round(t0 + bar * 2 - 0.05, 4), " ".join(ws), wt))
                t0 += bar * 2
        return cls(duration=duration, bpm=bpm, beats=beats, downbeats=downbeats, sections=sections,
                   style={"energy": 0.6, "valence": 0.5, "density": 0.5, "tags": []},
                   features=dict(rate=rate, loud=loud, bands=bands, kick=kick, flux=flux, cent=cent),
                   lyrics=lyrics, source={"synthetic": True, "seed": seed})
