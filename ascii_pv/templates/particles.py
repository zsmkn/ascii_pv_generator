"""Particles: a drifting swarm whose look follows a lyric motif (embers rise, rain falls, bubbles pop...).
Positions are closed-form in t (wrap-around motion with a sway), so any frame can be drawn directly."""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.typography.layer import _u16

# kind -> (chars from small to big, vertical speed in rows per beat (negative = up), sway in cells)
KINDS = {
    "embers": ("..'*^", -1.2, 2.0),
    "rain": ("..:||", 3.5, 0.2),
    "bubbles": (".o0O@", -0.7, 1.5),
    "snow": (".:+*", 0.5, 2.5),
    "stars": (".+*", 0.0, 0.0),
    "petals": (".,*%", 0.8, 3.0),
}
MOTIF_KIND = {"fire": "embers", "blood": "embers", "water": "rain", "pop": "bubbles", "dream": "snow",
              "star": "stars", "night": "stars", "love": "petals", "light": "stars", "fly": "embers"}


@register
class Particles(Template):
    meta = TemplateMeta(
        "particles", "Motif-driven swarm: embers rise, rain falls, bubbles float, stars twinkle.",
        busy=0.5, energy=(0.0, 0.75), sections=frozenset({"intro", "verse", "bridge", "outro", "instrumental"}),
        tags=frozenset({"calm", "bright", "sparse", "neutral"}), motifs=frozenset(MOTIF_KIND),
        lyric_blends=frozenset({"overlay", "obstacle"}), 
    )
    defaults = {"kind": "embers", "count": 0.06}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        chars, vy, sway = KINDS.get(str(self.params["kind"]), KINDS["embers"])
        seed = str_seed(self.shot.id) & 0xFFFF
        n = max(8, int(c.w * c.h * float(self.params["count"])))
        i = np.arange(n)
        x0 = _u16(i, seed) / 65535.0 * c.w
        y0 = _u16(i, seed + 1) / 65535.0 * c.h
        size = _u16(i, seed + 2) % len(chars)
        spd = 0.5 + (_u16(i, seed + 3) / 65535.0)
        phase = _u16(i, seed + 4) / 65535.0 * 6.283
        beats = ctx.local_t * ctx.bpm / 60
        y = (y0 + vy * spd * beats * (0.6 + 0.2 * size)) % c.h
        x = (x0 + np.sin(ctx.local_t * 1.3 * spd + phase) * sway) % c.w
        twinkle = 0.5 + 0.5 * np.sin(ctx.local_t * 3 * spd + phase)
        lev = np.clip((0.25 + 0.6 * size / max(1, len(chars) - 1)) * (0.5 + 0.5 * twinkle)
                      * (0.6 + 0.5 * ctx.energy) + 0.3 * ctx.kick * ctx.pulse(0.12), 0, 1)
        X, Y = x.astype(int) % c.w, y.astype(int) % c.h
        order = np.argsort(lev)
        X, Y, lev, size = X[order], Y[order], lev[order], size[order]
        if ctx.lyric_mask is not None and self.params.get("avoid_lyrics", True):
            keep = ~ctx.lyric_mask[Y, X]           # particles flow around the lettering
            X, Y, lev, size = X[keep], Y[keep], lev[keep], size[keep]
        bg = np.array(pal.bg, float)
        big = (size >= len(chars) - 1)[:, None]
        col = np.where(big, np.array(pal.accent, float), np.array(pal.ui, float))
        c.ch[Y, X] = np.array(list(chars))[size]
        c.fg[Y, X] = (bg + (col - bg) * lev[:, None]).astype(np.uint8)
