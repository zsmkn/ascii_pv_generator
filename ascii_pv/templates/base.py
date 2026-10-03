"""Template interface and registry.

A template draws the background picture of a shot. It declares metadata that the Director scores
against the song (energy/valence ranges, section types, tags, lyric motifs), and implements
`draw(canvas, ctx)` as a pure function of ctx. Expensive per-shot work goes in __init__.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx
from ascii_pv.core.storyboard import Shot


@dataclass(frozen=True)
class TemplateMeta:
    name: str
    description: str = ""
    energy: tuple[float, float] = (0.0, 1.0)
    valence: tuple[float, float] = (0.0, 1.0)
    sections: frozenset[str] = frozenset()     # preferred section labels; empty = any
    tags: frozenset[str] = frozenset()         # style tags it suits (electronic, ballad, ...)
    motifs: frozenset[str] = frozenset()       # lyric motifs that trigger it (love, rain, ...)
    needs_image: bool = False
    min_beats: int = 4
    lyric_blends: frozenset[str] = field(default_factory=lambda: frozenset({"overlay"}))
    busy: float = 0.5        # 0 calm .. 1 busy / high contrast: how hard it competes with lettering (M10)
    hud: str = "corners"     # where HUD panels fit around the picture: sides / top / corners / none
    hud_topics: tuple[str, ...] = ()   # panels that suit it, best first (e.g. "formula:torus")
    easter_egg: bool = False # never picked normally; at most one easter egg a song (planner._easter_egg)


class Template:
    meta: ClassVar[TemplateMeta]
    defaults: ClassVar[dict] = {}

    def __init__(self, shot: Shot, analysis: Analysis, assets: dict | None = None):
        self.shot = shot
        self.analysis = analysis
        self.assets = assets or {}
        self.params = {**self.defaults, **shot.params}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        raise NotImplementedError


REGISTRY: dict[str, type[Template]] = {}


def register(cls: type[Template]) -> type[Template]:
    name = cls.meta.name
    if name in REGISTRY:
        raise ValueError(f"template {name!r} registered twice")
    REGISTRY[name] = cls
    return cls


def create(shot: Shot, analysis: Analysis, assets: dict | None = None) -> Template:
    try:
        cls = REGISTRY[shot.template]
    except KeyError:
        raise KeyError(f"unknown template {shot.template!r}; available: {', '.join(sorted(REGISTRY))}") from None
    return cls(shot, analysis, assets)
