"""Matrix rain: glyph columns falling at tempo-locked speeds; heads flash on kicks."""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.typography.layer import ALPHABET, _u16


@register
class MatrixRain(Template):
    meta = TemplateMeta(
        "matrix_rain", "Falling glyph columns; speed locked to the tempo, heads flash on kicks.",
        busy=0.7, energy=(0.3, 0.9), valence=(0.0, 0.6), sections=frozenset({"verse", "prechorus", "bridge"}),
        tags=frozenset({"dark", "busy", "mid-energy"}), motifs=frozenset({"water", "night", "dream", "pulse"}),
        lyric_blends=frozenset({"overlay", "mask"}), 
    )
    defaults = {"density": 0.55, "speed": 1.0, "trail": 0.45}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        p = self.params
        pal = ctx.palette
        seed = str_seed(self.shot.id) & 0xFFFF
        cols = np.arange(c.w)
        active = (_u16(cols, seed) / 65535.0) < float(p["density"])
        # rows per second: 2..6 beats' worth, fixed per column so the picture is a pure function of t
        speed = (0.5 + _u16(cols, seed + 1) / 65535.0) * ctx.bpm / 60 * 4 * float(p["speed"])
        trail = np.maximum(4, (float(p["trail"]) * c.h * (0.5 + _u16(cols, seed + 2) / 65535.0))).astype(int)
        span = c.h + trail
        head = (ctx.local_t * speed + _u16(cols, seed + 3) % 200) % span          # (W,)
        yy = np.arange(c.h)[:, None]
        dist = head[None, :] - yy                                                # rows behind the head
        on = active[None, :] & (dist >= 0) & (dist < trail[None, :])
        tick = int(ctx.t * 12)
        xx = np.broadcast_to(cols[None, :], (c.h, c.w))
        k = _u16(xx * 131 + yy * 17 + (tick + (xx % 5)) // 2, seed + 4)
        chars = ALPHABET[k % len(ALPHABET)]
        fade = np.clip(1 - dist / np.maximum(1, trail[None, :]), 0, 1)
        lev = np.clip(0.12 + 0.6 * fade ** 1.6 * (0.6 + 0.4 * ctx.energy), 0, 1)
        is_head = on & (dist < 1)
        ys, xs = np.nonzero(on)
        bg = np.array(pal.bg, float)
        ui = np.array(pal.ui, float)
        rgb = (bg + (ui - bg) * lev[ys, xs][:, None]).astype(np.uint8)
        c.ch[ys, xs] = chars[ys, xs]
        c.fg[ys, xs] = rgb
        hy, hx = np.nonzero(is_head)
        c.fg[hy, hx] = pal.accent_at(0.55 + 0.45 * ctx.kick * ctx.pulse(0.12))
