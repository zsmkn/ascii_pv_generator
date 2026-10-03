"""Tunnel: rings rushing toward the viewer, one ring per beat; the walls brighten on kicks."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx
from ascii_pv.templates.base import Template, TemplateMeta, register

RING_CHARS = np.array(list(" .:-=+*#%@"))


@register
class Tunnel(Template):
    meta = TemplateMeta(
        "tunnel", "Rings rushing toward the viewer, one per beat; spins slowly, flares on kicks.",
        busy=0.9, energy=(0.55, 1.0), sections=frozenset({"chorus", "prechorus"}),
        tags=frozenset({"energetic", "bright", "busy", "percussive"}),
        motifs=frozenset({"speed", "fly", "star", "time", "city"}),
        lyric_blends=frozenset({"overlay", "mask", "knockout"}), 
    )
    defaults = {"twist": 0.6, "sides": 0}   # sides 0 = round, 4 = square, 6 = hexagon

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        cx, cy = (c.w - 1) / 2, (c.h - 1) / 2
        yy, xx = np.mgrid[0:c.h, 0:c.w]
        dx, dy = (xx - cx) / 2.0, (yy - cy)          # cells are ~2x taller than wide
        ang = np.arctan2(dy, dx)
        sides = int(self.params["sides"])
        r = np.hypot(dx, dy)
        if sides >= 3:                                # polygon distance
            seg = math.tau / sides
            r = r * np.cos((ang % seg) - seg / 2) / math.cos(seg / 2)
        r = np.maximum(r, 0.6)
        K = math.hypot(cx / 2.0, cy) * 1.05           # screen edge sits at depth ~1; ring i at radius K / i
        depth = K / r
        beats = ctx.beat_idx + ctx.beat_phase
        z = depth + beats                             # one ring passes per beat
        twist = float(self.params["twist"]) * ctx.local_t * 0.4
        stripe = ((ang + twist) / (math.tau / 12)) % 1.0
        near = np.clip(1.4 / depth, 0, 1)              # 1 near the screen edge, 0 at the vanishing point
        ring = (z % 1.0) < 0.08 + 0.1 * near
        spoke = (stripe < 0.03 + 0.03 * near) & (depth > 1.1)
        wall = ring | spoke
        lev = np.clip((0.1 + 0.8 * near) * (0.55 + 0.45 * ctx.energy) + 0.35 * ctx.kick * ctx.pulse(0.1) * near,
                      0, 1)
        lev = np.where(ring, lev, lev * 0.5)
        idx = np.clip((lev * (len(RING_CHARS) - 1)).astype(int), 1, len(RING_CHARS) - 1)
        ys, xs = np.nonzero(wall & (depth < 30))
        bg = np.array(pal.bg, float)
        col = np.where((ring & (near > 0.8))[ys, xs, None], np.array(pal.accent, float), np.array(pal.ui, float))
        c.ch[ys, xs] = RING_CHARS[idx[ys, xs]]
        c.fg[ys, xs] = (bg + (col - bg) * lev[ys, xs, None]).astype(np.uint8)
