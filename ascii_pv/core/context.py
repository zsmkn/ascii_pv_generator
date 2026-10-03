"""FrameCtx: everything a template may read for one frame. Templates must be pure functions of it."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from ascii_pv.core.analysis import Analysis, Features, LyricLine, Section
from ascii_pv.core.palette import Palette
from ascii_pv.core.storyboard import Shot


def hash32(*parts: int) -> int:
    """Stable integer hash (unlike hash(), identical across runs and processes)."""
    v = 0x811C9DC5
    for p in parts:
        v ^= p & 0xFFFFFFFF
        v = (v * 0x01000193) & 0xFFFFFFFF
        v ^= v >> 15
        v = (v * 0x2C1B3C6D) & 0xFFFFFFFF
        v ^= v >> 12
    return v


def str_seed(s: str) -> int:
    return hash32(*s.encode("utf-8"))


@dataclass
class FrameCtx:
    t: float                  # song time (s)
    frame: int                # int(t * fps): the unit for per-frame randomness
    shot: Shot
    local_t: float            # t - shot.start
    u: float                  # shot progress 0..1
    beat_idx: int
    beat_phase: float         # 0..1 inside the current beat
    beat_time: float          # when the current beat started
    bar_idx: int
    feat: Features
    section: Section | None
    line_idx: int             # active lyric line, -1 if none
    line: LyricLine | None
    word_idx: int             # word being sung in `line`, -1 if none
    word_progress: float      # 0..1 through the current word
    palette: Palette
    seed: int
    bpm: float
    lyric_mask: object = None  # filled in by the typography layer (M2)
    live: dict = field(default_factory=dict)   # values a template exposes to its HUD panels this frame (M10)

    @property
    def energy(self) -> float:
        return self.feat.loud

    @property
    def bands(self) -> list[float]:
        return self.feat.bands

    @property
    def kick(self) -> float:
        return self.feat.kick

    def pulse(self, decay: float = 0.14) -> float:
        """1 on each beat, decaying exponentially (decay in seconds)."""
        return math.exp(-max(0.0, self.t - self.beat_time) / decay)

    def rng(self, salt: int = 0, per_frame: bool = True) -> random.Random:
        """Deterministic RNG. per_frame=False gives one stream for the whole shot."""
        return random.Random(hash32(self.seed, str_seed(self.shot.id), salt, self.frame if per_frame else -1))


def make_ctx(t: float, analysis: Analysis, shot: Shot, palette: Palette, seed: int, fps: int) -> FrameCtx:
    bi, phase, bt = analysis.beat_info(t)
    li = analysis.line_at(t)
    line = analysis.lyrics[li] if li >= 0 else None
    wi, wp = -1, 0.0
    if line is not None and line.words:
        wi = line.word_at(t)
        if wi >= 0:
            w = line.words[wi]
            wp = min(1.0, max(0.0, (t - w.start) / max(1e-3, w.end - w.start)))
    local = t - shot.start
    return FrameCtx(t=t, frame=int(math.floor(t * fps)), shot=shot, local_t=local,
                    u=min(1.0, max(0.0, local / max(1e-6, shot.dur))),
                    beat_idx=bi, beat_phase=phase, beat_time=bt, bar_idx=analysis.bar_index(t),
                    feat=analysis.feat(t), section=analysis.section_at(t),
                    line_idx=li, line=line, word_idx=wi, word_progress=wp,
                    palette=palette, seed=seed, bpm=analysis.bpm)
