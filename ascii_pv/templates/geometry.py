"""Geometric templates (M8, second round): helix, vortex, kaleidoscope, radar."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core import noise
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.templates.waves import SHADE, _paint
from ascii_pv.typography.layer import _u16


def _polar(c: Canvas, cx: float | None = None, cy: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(r, theta) per cell; r = 1 at half the screen height (cells are ~2x taller than wide)."""
    yy, xx = np.mgrid[0:c.h, 0:c.w].astype(float)
    cx = c.w / 2 if cx is None else cx
    cy = c.h / 2 if cy is None else cy
    dx, dy = (xx - cx) / 2, yy - cy
    return np.hypot(dx, dy) / (c.h / 2), np.arctan2(dy, dx)


@register
class Helix(Template):
    meta = TemplateMeta(
        "helix", "A DNA double helix turning across the screen, base pairs between the strands; kicks swell it.",
        busy=0.4, energy=(0.15, 0.75), sections=frozenset({"intro", "verse", "bridge", "instrumental"}),
        tags=frozenset({"calm", "mid-energy", "neutral", "bright"}), motifs=frozenset({"blood", "dream", "time"}),
        lyric_blends=frozenset({"overlay", "knockout"}),
    )
    defaults = {"turns": 2.2, "rungs": 4}
    BASES = ("AT", "TA", "GC", "CG")

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        x = np.arange(w)
        k = math.tau * float(self.params["turns"]) / w               # phase per column
        S = self.analysis.travel(ctx.t, 0.5) * 1.6
        ph = x * k - S
        A = h * 0.3 * (1 + 0.12 * ctx.kick * ctx.pulse(0.15))
        mid = h / 2
        y1, y2 = mid + A * np.sin(ph), mid - A * np.sin(ph)
        d1 = np.cos(ph)                                              # > 0: strand 1 is in front
        seed = str_seed(self.shot.id) & 0xFFFF
        # base pairs (behind both strands' front halves)
        step = int(self.params["rungs"])
        drift = S / k                                                 # columns the helix has moved
        first = int(math.floor(drift / step))
        for g in range(-first - 1, -first + w // step + 2):           # rungs travel with the strands
            xi = int(round(g * step + drift))
            if not 0 <= xi < w:
                continue
            a, b = sorted((y1[xi], y2[xi]))
            rows = np.arange(int(math.ceil(a)) + 1, int(math.floor(b)))
            if len(rows) < 1:
                continue
            pair = self.BASES[int(_u16(np.array([g & 0xFFFF]), seed)[0]) % 4]
            mid_r = (a + b) / 2
            ch = np.where(rows < mid_r, pair[0], pair[1])
            ch = np.where(np.abs(rows - mid_r) < 0.6, "-", np.where((rows == rows[0]) | (rows == rows[-1]), ch, ":"))
            _paint(c, rows, np.full(len(rows), xi), ch, np.full(len(rows), 0.35 + 0.15 * ctx.energy), ctx)
        # back strand first, then the front one
        for y, d in ((y1, d1), (y2, -d1)):
            for front in (False, True):
                m = (d > 0) == front
                yi = np.rint(y[m]).astype(int)
                ok = (yi >= 0) & (yi < h)
                dd = d[m][ok]
                ch = np.where(dd > 0.55, "@", np.where(dd > 0, "O", np.where(dd > -0.5, "o", ".")))
                _paint(c, yi[ok], x[m][ok], ch, 0.3 + 0.7 * (dd + 1) / 2, ctx, dd > 0.8)


@register
class Vortex(Template):
    meta = TemplateMeta(
        "vortex", "Spiral arms wheeling around a dark core; kicks wind the spiral tighter.",
        busy=0.85, energy=(0.55, 1.0), sections=frozenset({"chorus", "prechorus", "bridge"}),
        tags=frozenset({"energetic", "dark", "busy"}), motifs=frozenset({"dream", "star", "fall", "break"}),
        lyric_blends=frozenset({"overlay", "mask", "knockout"}),
    )
    defaults = {"arms": 3, "twist": 5.0}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        r, th = _polar(c)
        arms = int(self.params["arms"])
        twist = float(self.params["twist"]) * (1 + 0.35 * ctx.kick * ctx.pulse(0.18))
        spin = self.analysis.travel(ctx.t, 0.4) * 1.8
        v = np.sin(arms * th + np.log(r + 0.04) * twist - spin)
        core = 1 - np.exp(-(r / 0.18) ** 2)                          # black hole in the middle
        edge = np.clip(1.6 - r, 0, 1)
        lev = np.clip((v + 0.2) / 1.2, 0, 1) ** 1.6 * core * edge * (0.65 + 0.45 * ctx.energy)
        idx = np.clip((lev * (len(SHADE) - 1)).astype(int), 0, len(SHADE) - 1)
        ys, xs = np.nonzero(idx > 0)
        _paint(c, ys, xs, SHADE[idx[ys, xs]], lev[ys, xs], ctx, (lev > 0.8)[ys, xs])


@register
class Kaleidoscope(Template):
    meta = TemplateMeta(
        "kaleidoscope", "Mirror-symmetric noise patterns in 6 or 8 wedges, slowly turning and flowing outward.",
        busy=0.9, energy=(0.4, 1.0), sections=frozenset({"chorus", "bridge", "prechorus", "instrumental"}),
        tags=frozenset({"bright", "energetic", "busy"}), motifs=frozenset({"dream", "light", "dance"}),
        lyric_blends=frozenset({"overlay", "mask", "knockout"}),
    )
    defaults = {"wedges": 6, "zoom": 1.0}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        r, th = _polar(c)
        n = int(self.params["wedges"])
        trav = self.analysis.travel(ctx.t, 0.3)
        wedge = math.tau / n
        t2 = np.abs(np.mod(th + trav * 0.15, wedge) - wedge / 2)    # fold into one mirrored wedge
        z = float(self.params["zoom"]) * 2.4
        u, v = r * np.cos(t2) * z, r * np.sin(t2) * z
        seed = str_seed(self.shot.id) & 0xFFFF
        f = noise.fbm(u * 0.9 - trav * 0.5, v * 0.9 + 3.1, octaves=2, seed=seed)
        petals = 0.5 + 0.5 * np.cos(t2 * n * 1.5 + r * 4 - trav * 1.8)      # bold shapes that show the symmetry
        f = 0.55 * f + 0.45 * petals * (0.5 + 0.5 * np.sin(r * 6 - trav * 2.2))
        lev = np.clip((f - 0.3) / 0.45, 0, 1) ** 1.3 * np.clip(1.6 - r * 0.6, 0, 1) * (0.8 + 0.35 * ctx.energy)
        lev = np.clip(lev + 0.15 * ctx.kick * ctx.pulse(0.12) * (lev > 0.3), 0, 1)
        idx = np.clip((lev * (len(SHADE) - 1)).astype(int), 0, len(SHADE) - 1)
        ys, xs = np.nonzero(idx > 0)
        _paint(c, ys, xs, SHADE[idx[ys, xs]], lev[ys, xs], ctx, (lev > 0.7)[ys, xs])


@register
class Radar(Template):
    meta = TemplateMeta(
        "radar", "A radar scope: the sweep turns once a bar, contacts light up as it passes and fade behind it.",
        busy=0.4, energy=(0.2, 0.8), sections=frozenset({"intro", "verse", "prechorus", "bridge"}),
        tags=frozenset({"dark", "mid-energy", "percussive", "neutral"}), motifs=frozenset({"pulse", "fight", "city"}),
        lyric_blends=frozenset({"overlay", "knockout"}),
    )
    defaults = {"contacts": 9, "rings": 3}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        pal = ctx.palette
        r, th = _polar(c)
        R = 0.92
        rn = r / R
        sweep = math.tau * (((ctx.beat_idx % 4) + ctx.beat_phase) / 4) - math.pi / 2
        behind = np.mod(sweep - th, math.tau)                        # angle since the sweep passed
        inside = rn <= 1
        # grid: range rings, crosshair, rim
        rings = int(self.params["rings"])
        ring = inside & (np.abs(rn * rings - np.rint(rn * rings)) < 0.06 * rings) & (rn > 0.1)
        yy, xx = np.mgrid[0:h, 0:w]
        cross = inside & ((np.abs(xx - w / 2) < 0.5) | (np.abs(yy - h / 2) < 0.5)) & ((xx + yy) % 2 == 0)
        _paint(c, *np.nonzero(ring), np.full(ring.sum(), "."), np.full(ring.sum(), 0.3), ctx)
        _paint(c, *np.nonzero(cross), np.full(cross.sum(), "+"), np.full(cross.sum(), 0.22), ctx)
        rim = (np.abs(rn - 1) < 0.035)
        _paint(c, *np.nonzero(rim), np.full(rim.sum(), "o"), np.full(rim.sum(), 0.4), ctx)
        # the afterglow behind the sweep and the sweep line itself
        glow = np.exp(-behind * 2.2) * inside
        g = glow > 0.12
        ch = np.where(behind < 0.07, "#", np.where(glow > 0.5, ":", "."))
        _paint(c, yy[g], xx[g], ch[g], 0.25 + 0.75 * glow[g], ctx, behind[g] < 0.07)
        # contacts: drifting targets, bright when the sweep has just passed them
        seed = str_seed(self.shot.id) & 0xFFFF
        k = np.arange(int(self.params["contacts"]))
        cr = 0.2 + 0.75 * _u16(k, seed) / 65535
        ca = _u16(k, seed + 1) / 65535 * math.tau + ctx.t * (0.03 + 0.05 * _u16(k, seed + 2) / 65535)
        lit = np.exp(-np.mod(sweep - ca, math.tau) * 0.9)
        cx = np.rint(w / 2 + np.cos(ca) * cr * R * h).astype(int)
        cy = np.rint(h / 2 + np.sin(ca) * cr * R * h / 2).astype(int)
        ok = (cx >= 0) & (cx < w) & (cy >= 0) & (cy < h) & (lit > 0.1)
        hot = lit > 0.75
        _paint(c, cy[ok], cx[ok], np.where(hot, "@", "o")[ok], (0.35 + 0.65 * lit)[ok], ctx, hot[ok])
        c.fg[h // 2, w // 2] = pal.accent_at(0.9)
        c.ch[h // 2, w // 2] = "+"
