"""Spectrum bars: the seven analysis bands spread across the width, with 1/8-cell precision."""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx
from ascii_pv.templates.base import Template, TemplateMeta, register

EIGHTHS = " ▁▂▃▄▅▆▇█"
BAND_HZ = [80, 190, 460, 1000, 2100, 4200, 7700]       # centres of the analysis bands (analysis/audio.py BANDS)


@register
class Blank(Template):
    meta = TemplateMeta("blank", "Background colour only.", min_beats=1, busy=0.0, )

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pass


@register
class SpectrumBars(Template):
    meta = TemplateMeta(
        "spectrum_bars", "Band energies as vertical bars; the bass side flares on kicks.",
        busy=0.8, sections=frozenset({"intro", "verse", "outro", "instrumental"}),
        lyric_blends=frozenset({"overlay", "knockout"}), 
    )
    defaults = {"mirror": True, "gap": 1, "height": 0.8, "annotate": True}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        gap = int(self.params["gap"])
        mirror = bool(self.params["mirror"])
        max_h = max(1.0, c.h * float(self.params["height"]))
        bands = np.asarray(ctx.bands, float)
        # One bar per 2 columns + gap; bands interpolated across the bars (mirrored: bass in the middle).
        step = 2 + gap
        n = max(1, c.w // step)
        pos = np.linspace(0, 1, n)
        if mirror:
            pos = np.abs(pos * 2 - 1)  # 0 (bass) in the middle, 1 (air) at both edges
        levels = np.interp(pos * (len(bands) - 1), np.arange(len(bands)), bands)
        wobble = 0.08 * np.sin(ctx.t * 3.0 + np.arange(n) * 0.7)
        levels = np.clip(levels + wobble * ctx.energy, 0, 1)
        flare = ctx.pulse(0.12) * ctx.kick
        base_y = c.h - 1
        x0 = (c.w - n * step + gap) // 2
        annotate = bool(self.params["annotate"])
        if annotate:                                     # level grid behind the bars, like an analyser's
            for frac, label in ((0.25, "-12dB"), (0.5, "-6dB"), (0.75, "-3dB")):
                y = base_y - int(frac * max_h)
                c.put(0, y, "┄" * c.w, pal.ui_at(0.22))
                c.put(1, y, label, pal.ui_at(0.45))
        for i, lv in enumerate(levels):
            height = lv * max_h
            full = int(height)
            frac = int((height - full) * 8)
            x = x0 + i * step
            for k in range(full):
                y = base_y - k
                level = 0.25 + 0.75 * (k / max_h) + flare * 0.3
                c.put(x, y, "██", pal.ramp(min(1.0, level)))
            if frac and full < max_h:
                c.put(x, base_y - full, EIGHTHS[frac] * 2, pal.ramp(min(1.0, 0.25 + 0.75 * full / max_h)))

        if annotate:
            self._peak(c, ctx, levels, pos, x0, step, base_y, max_h)

    def _peak(self, c: Canvas, ctx: FrameCtx, levels: np.ndarray, pos: np.ndarray, x0: int, step: int, base_y: int,
              max_h: float) -> None:
        """A callout on the loudest bar: brackets on its top and a box beside it with its frequency and level."""
        pal = ctx.palette
        i = int(np.argmax(levels))
        hz = float(np.exp(np.interp(pos[i] * (len(BAND_HZ) - 1), np.arange(len(BAND_HZ)), np.log(BAND_HZ))))
        label = f"PEAK {hz:5.0f}Hz {levels[i]:.2f}"
        w = len(label) + 2
        top = base_y - int(levels[i] * max_h)
        lx = x0 + i * step
        col = pal.accent_at(0.9)
        c.put(lx - 1, top - 1, "[", col)
        c.put(lx + 2, top - 1, "]", col)
        right = lx + 4 + w <= c.w
        bx = lx + 5 if right else lx - 3 - w
        y = top - 3
        if y < 0 or bx < 0:
            return
        c.put(bx, y, "┌" + "─" * (w - 2) + "┐", col)
        c.put(bx, y + 1, "│", col)
        c.put(bx + 1, y + 1, label, pal.ui_at(0.95))
        c.put(bx + w - 1, y + 1, "│", col)
        c.put(bx, y + 2, "└" + "─" * (w - 2) + "┘", col)
        if right:
            c.put(lx + 3, y + 2, "─" * 2, col)
        else:
            c.put(bx + w, y + 2, "─" * 2, col)
