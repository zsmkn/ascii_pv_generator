"""storyboard.json: the Director's output and the user's editing surface.

Anything marked `locked: true` is kept as-is when `plan --lock` regenerates the rest.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Shot:
    id: str
    start: float
    end: float
    template: str
    params: dict = field(default_factory=dict)
    fx: dict = field(default_factory=dict)
    transition_in: str = "cut"
    locked: bool = False
    hud: list[str] = field(default_factory=list)   # HUD panels, e.g. ["formula:torus", "data"] (templates/hud.py)

    @property
    def dur(self) -> float:
        return self.end - self.start


@dataclass
class LyricCue:
    line: int                     # index into Analysis.lyrics
    mode: str = "line"            # takeover / line / word_flash / stack / embed
    font: str = "bit5x5"
    fill: str = "binary"          # binary / hex / self / ramp / outline / block
    enter: str = "assemble"
    exit: str = "explode"
    blend: str = "overlay"        # overlay / mask / knockout / obstacle / echo
    offset: float = 0.0           # per-line timing correction (seconds)
    locked: bool = False


@dataclass
class Storyboard:
    seed: int = 0
    palette: str = "amber"
    fps: int = 24
    grid: tuple[int, int] = (160, 45)          # MP4 cell grid (cols, rows)
    lyric_offset: float = 0.0                  # seconds; > 0 shows every lyric earlier (fixes a late LRC)
    shots: list[Shot] = field(default_factory=list)
    lyrics: list[LyricCue] = field(default_factory=list)
    overlays: dict = field(default_factory=lambda: {"sprite": {"enabled": False}, "subtitle": False})
    assets: dict = field(default_factory=dict)   # {"character": {"full": png, "bust": png, "face": png}, ...}

    def __post_init__(self) -> None:
        self.shots.sort(key=lambda s: s.start)
        self._starts = [s.start for s in self.shots]

    def shot_at(self, t: float) -> Shot | None:
        i = bisect.bisect_right(self._starts, t) - 1
        if i < 0:
            return self.shots[0] if self.shots else None
        return self.shots[i]

    # ------------------------------------------------------------ io

    def to_dict(self) -> dict:
        d = asdict(self)
        d["grid"] = list(self.grid)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Storyboard":
        d = dict(d)
        d["shots"] = [Shot(**s) for s in d.get("shots", [])]
        d["lyrics"] = [LyricCue(**c) for c in d.get("lyrics", [])]
        if "grid" in d:
            d["grid"] = tuple(d["grid"])
        return cls(**d)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Storyboard":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def single(cls, template: str, duration: float, **kw) -> "Storyboard":
        """One shot covering the whole song: handy for previews and tests."""
        params = kw.pop("params", {})
        return cls(shots=[Shot("s01", 0.0, duration, template, params)], **kw)
