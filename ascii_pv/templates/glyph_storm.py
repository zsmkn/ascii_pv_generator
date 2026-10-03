"""Glyph storm: a drifting field of code glyphs; denser and brighter with the music, rows sway on the beat."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.typography.layer import ALPHABET, _u16


@register
class GlyphStorm(Template):
    meta = TemplateMeta(
        "glyph_storm", "Drifting field of code glyphs; density follows loudness, rows sway on the beat.",
        busy=0.9,  energy=(0.5, 1.0), sections=frozenset({"chorus", "prechorus", "bridge"}),
        tags=frozenset({"energetic", "dark", "busy"}), motifs=frozenset({"pulse", "light"}),
        lyric_blends=frozenset({"overlay", "mask"}),
    )
    defaults = {"density": 0.22, "sway": 7.0, "rate": 7.0}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        p = self.params
        pal = ctx.palette
        t = ctx.t
        yy, xx = np.mgrid[0:c.h, 0:c.w]
        tick = int(t * float(p["rate"]))
        k = _u16(xx * 37 + yy * 911 + tick * (3 + xx % 7), str_seed(self.shot.id) & 0xFFFF)
        density = float(p["density"]) * (0.45 + 0.9 * ctx.energy) + 0.15 * ctx.kick * ctx.pulse(0.1)
        show = k / 65535.0 < density
        sway = float(p["sway"]) * (0.4 + 0.6 * ctx.energy)
        shift = (np.sin(t * 2.6 + np.arange(c.h) * 0.31 + ctx.beat_phase * math.tau * 0.25) * sway).astype(int)
        xs = (xx + shift[:, None]) % c.w
        chars = ALPHABET[(k + ctx.frame + (xx // 7) * 13) % len(ALPHABET)]
        # a bright scan band rolls down the screen, one pass per bar
        bar_u = ((ctx.beat_idx % 4) + ctx.beat_phase) / 4
        band = (yy - int(bar_u * c.h)) % c.h < 2
        lev = np.where(band, 0.75, np.where(k % 17 == 0, 0.5, 0.22 + 0.2 * ctx.energy))
        bg = np.array(pal.bg, float)
        rgb = (bg + (np.array(pal.ui, float) - bg) * lev[..., None]).astype(np.uint8)
        c.ch[yy[show], xs[show]] = chars[show]
        c.fg[yy[show], xs[show]] = rgb[show]
