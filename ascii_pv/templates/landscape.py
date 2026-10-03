"""Scenery templates (M8, second round): ridgelines, rain, ocean, fireworks."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core import noise
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.templates.waves import _paint
from ascii_pv.typography.layer import _u16


def _bands_at(a, t: float) -> np.ndarray:
    f = a.features
    i = max(0, min(len(f["loud"]) - 1, int(t * f["rate"])))
    return np.array([b[i] for b in f["bands"]], float)


@register
class Ridgelines(Template):
    meta = TemplateMeta(
        "ridgelines", "Stacked ridgelines (Unknown Pleasures): each line is the spectrum a moment ago, scrolling back.",
        busy=0.5, energy=(0.25, 0.85), sections=frozenset({"intro", "verse", "bridge", "outro", "instrumental"}),
        tags=frozenset({"dark", "calm", "mid-energy", "neutral"}), motifs=frozenset({"pulse", "night", "city"}),
        lyric_blends=frozenset({"overlay", "knockout"}),
    )
    defaults = {"lines": 0, "amp": 1.0, "step": 0.12}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        n = int(self.params["lines"]) or max(6, h // 2)
        top, bottom = h * 0.2, h * 0.94
        gap = (bottom - top) / max(1, n - 1)
        step = float(self.params["step"])
        xn = (np.arange(w) - w / 2) / (w / 2)
        window = np.exp(-(xn / 0.38) ** 2)                          # the ridges rise in the middle
        rows = np.arange(h)[:, None]
        now = int(ctx.t / step)
        seed = str_seed(self.shot.id) & 0xFFFF
        for i in range(n):                                          # back (oldest) to front (now)
            k = now - (n - 1 - i)                                   # snapshot index: a line keeps its shape
            bands = _bands_at(self.analysis, k * step)
            prof = np.interp(np.abs(xn) * 6, np.arange(7), bands)
            detail = noise.value(np.arange(w) * 0.16, np.full(w, k * 1.7), seed)
            hgt = (0.75 * prof + 0.4 * detail ** 1.5) * window * float(self.params["amp"]) * gap * 5.5
            base = top + i * gap
            y = base - hgt
            # hidden lines: the area under this ridge hides the ridges behind it
            hide = (rows > y[None, :]) & (rows <= base + 0.5)
            c.ch[hide] = " "
            r = np.floor(y).astype(int)
            frac = y - r
            slope = np.gradient(y)
            ch = np.where(frac < 0.33, "‾", np.where(frac < 0.66, "-", "_"))
            ch = np.where(slope < -0.7, "/", np.where(slope > 0.7, "\\", ch))
            ok = (r >= 0) & (r < h)
            lev = (0.3 + 0.7 * (i + 1) / n) * (0.6 + 0.4 * ctx.energy)
            peak = hgt > gap * 2.5
            xs = np.nonzero(ok)[0]
            _paint(c, r[ok], xs, ch[ok], np.full(len(xs), lev) + 0.25 * peak[ok], ctx, peak[ok] & (i > n * 0.6))
            # steep flanks: join the columns with vertical strokes
            lo, hi = np.minimum(y[:-1], y[1:]), np.maximum(y[:-1], y[1:])
            fill = (rows > lo[None, :] + 0.5) & (rows < hi[None, :] - 0.5)
            ys, xs = np.nonzero(fill)
            _paint(c, ys, xs, np.full(len(ys), "|"), np.full(len(ys), lev * 0.8), ctx)


@register
class Rain(Template):
    meta = TemplateMeta(
        "rain", "Slanted rain streaks that splash on the ground; heavier and windier in loud passages.",
        busy=0.5, energy=(0.0, 0.7), sections=frozenset({"intro", "verse", "bridge", "outro"}),
        tags=frozenset({"calm", "dark", "sparse"}), motifs=frozenset({"water", "night", "break"}),
        lyric_blends=frozenset({"overlay", "obstacle", "knockout"}),
    )
    defaults = {"density": 0.5, "wind": 0.35, "length": 3}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        seed = str_seed(self.shot.id) & 0xFFFF
        ground = h - 1
        L = int(self.params["length"])
        n_max = int(w * h * 0.03)
        n = int(n_max * float(self.params["density"]) * (0.5 + 0.7 * ctx.energy))
        k = np.arange(n)
        x0 = _u16(k, seed) % w
        speed = (16 + _u16(k, seed + 1) % 14) * h / 45                # rows per second
        cycle = h + L + 6
        pos = (_u16(k, seed + 2) / 65535 * cycle + self.analysis.travel(ctx.t, 0.8) * speed) % cycle
        wind = float(self.params["wind"]) * (0.7 + 0.6 * ctx.energy)
        streak = "/" if wind > 0.15 else "|"
        pal = ctx.palette
        for j in range(L):                                           # the streak, head brightest
            y = np.floor(pos - j).astype(int)
            x = (x0 + np.rint(y * wind)).astype(int) % w
            ok = (y >= 0) & (y < ground)
            _paint(c, y[ok], x[ok], np.full(ok.sum(), streak), np.full(ok.sum(), 0.85 - 0.22 * j), ctx)
        # splashes where the drops land
        age = pos - ground                                           # rows past the ground
        land = (age >= 0) & (age < 6)
        xl = ((x0 + np.rint(ground * wind)).astype(int) % w)[land]
        stage = age[land].astype(int)
        for s, offs in ((0, [(0, ".")]), (1, [(0, "o")]), (2, [(-1, "("), (1, ")")]), (3, [(-2, "."), (2, ".")]),
                        (4, [(-2, "."), (2, ".")])):
            m = stage == s
            for dx, ch in offs:
                xx = (xl[m] + dx) % w
                lev = np.full(m.sum(), 0.9 - 0.15 * s)
                _paint(c, np.full(m.sum(), ground - (1 if s == 1 else 0)), xx, np.full(m.sum(), ch), lev, ctx,
                       np.full(m.sum(), s <= 1))
        floor = c.ch[ground] == " "
        c.ch[ground, floor] = "_"
        c.fg[ground, floor] = pal.ui_at(0.25)


@register
class Ocean(Template):
    meta = TemplateMeta(
        "ocean", "Moonlit sea: perspective waves rolling in, the moon's reflection shimmering on the water.",
        busy=0.5, energy=(0.0, 0.6), sections=frozenset({"intro", "verse", "bridge", "outro", "instrumental"}),
        tags=frozenset({"calm", "sparse", "bright"}), motifs=frozenset({"water", "night", "dream", "star"}),
        lyric_blends=frozenset({"overlay", "knockout"}),
    )
    defaults = {"moon": 0.72, "swell": 1.0}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        seed = str_seed(self.shot.id) & 0xFFFF
        horizon = int(h * 0.42)
        pal = ctx.palette
        # stars, twinkling
        k = np.arange(int(w * horizon * 0.015))
        sx, sy = _u16(k, seed) % w, _u16(k, seed + 1) % max(1, horizon - 1)
        tw = 0.5 + 0.5 * np.sin(ctx.t * (1 + _u16(k, seed + 2) % 4) + k)
        _paint(c, sy, sx, np.where(tw > 0.85, "+", "."), 0.2 + 0.5 * tw, ctx)
        # moon
        mx, my, R = float(self.params["moon"]) * w, horizon * 0.42, max(2.0, h * 0.09)
        yy, xx = np.mgrid[0:horizon, 0:w]
        d = np.hypot((xx - mx) / 2, yy - my) / R
        disc = d < 1
        ch = np.where(d < 0.65, "@", np.where(d < 0.88, "%", "*"))
        _paint(c, yy[disc], xx[disc], ch[disc], 0.95 - 0.3 * d[disc], ctx, d[disc] < 0.65)
        glow = (d >= 1.15) & (d < 1.5) & (((xx + yy) % 3) == 0)
        _paint(c, yy[glow], xx[glow], ".", np.full(glow.sum(), 0.3), ctx)
        # sea
        sy, sx = np.mgrid[horizon:h, 0:w].astype(float)
        z = (sy - horizon + 1) / (h - horizon)                       # 0 far .. 1 near
        trav = self.analysis.travel(ctx.t, 0.4)
        k1 = 0.9 / (z + 0.15)
        v = (np.sin(sx / k1 * 0.6 + trav * 2.2 + sy * 1.3) + 0.6 * np.sin(sx / k1 * 1.3 - trav * 1.4 + sy * 0.7)
             + 0.3 * np.sin(sx * 0.05 + trav)) * float(self.params["swell"])
        u = np.clip((v + 1.9) / 3.8, 0, 1)
        ch = np.array(list(" .-~^"))[np.clip((u * 4.99).astype(int), 0, 4)]
        lev = (0.2 + 0.55 * z) * (0.5 + 0.6 * u) * (0.8 + 0.3 * ctx.energy) + 0.1 * ctx.kick * ctx.pulse(0.2)
        # the moon on the water: a shimmering column widening toward us
        col = np.abs(sx - mx) < 1.5 + 9 * z
        shim = col & (np.sin(sy * 1.9 + ctx.t * 4.0 + sx * 0.7) > 0.1)
        ch = np.where(shim, np.where(z > 0.5, "=", "-"), ch)
        lev = np.where(shim, 0.65 + 0.35 * z, lev)
        m = ch != " "
        _paint(c, sy[m].astype(int), sx[m].astype(int), ch[m], lev[m], ctx, shim[m])
        c.put(0, horizon, "_" * w, pal.ui_at(0.3))


@register
class Fireworks(Template):
    meta = TemplateMeta(
        "fireworks", "Rockets launched on the beats burst into falling sparks; more of them when it is loud.",
        busy=0.6, energy=(0.5, 1.0), sections=frozenset({"chorus", "prechorus", "outro"}),
        tags=frozenset({"energetic", "bright", "busy"}), motifs=frozenset({"light", "fire", "star", "fly"}),
        lyric_blends=frozenset({"overlay", "obstacle", "knockout"}),
    )
    defaults = {"rate": 0.45, "size": 1.0}
    RISE, LIFE = 0.7, 2.2

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        seed = str_seed(shot.id) & 0xFFFF
        f = analysis.features
        beats = [b for b in analysis.beats if shot.start - self.RISE - self.LIFE <= b <= shot.end]
        self.launches = []
        for b in beats:
            i = min(len(f["loud"]) - 1, int(b * f["rate"]))
            hb = int(_u16(np.array([int(b * 1000)]), seed)[0])
            if hb % 100 < 100 * float(self.params["rate"]) * (0.5 + f["loud"][i]):
                self.launches.append((b, (hb % 997) / 997, 0.15 + (hb % 89) / 89 * 0.3, hb))

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        w, h = c.w, c.h
        pal = ctx.palette
        roles = [pal.accent, pal.ui, pal.anom, pal.err]
        for tb, fx, fy, hb in self.launches:
            age = ctx.t - tb
            if age < 0 or age > self.RISE + self.LIFE:
                continue
            x, yb = 4 + fx * (w - 8), fy * h
            if age < self.RISE:                                       # the rocket climbing
                y = h - 1 - (h - 1 - yb) * (age / self.RISE) ** 0.7
                c.put(x, y, "^", pal.ui_at(0.9))
                c.put(x, y + 1, "|", pal.ui_at(0.5))
                c.put(x, y + 2, ".", pal.ui_at(0.3))
                continue
            a = age - self.RISE
            n = 70 + hb % 40
            j = np.arange(n)
            th = _u16(j, hb) / 65535 * math.tau
            shell = np.where(j % 3 == 0, 0.55, 1.0)                    # an inner and an outer shell of sparks
            v = shell * (7 + 2 * _u16(j, hb + 1) / 65535) * float(self.params["size"]) * h / 45
            fade = max(0.0, 1 - a / self.LIFE) ** 0.7
            base = np.array(roles[hb % 4], float)
            bg = np.array(pal.bg, float)
            for lag, ch_set in ((0.0, "@*+"), (0.1, "*+."), (0.22, "..'")):
                aa = max(0.0, a - lag)
                px = x + np.cos(th) * v * aa * 2 * (1 - 0.35 * aa / self.LIFE)
                py = yb + np.sin(th) * v * aa * (1 - 0.35 * aa / self.LIFE) + 3.0 * aa * aa
                xi, yi = np.rint(px).astype(int), np.rint(py).astype(int)
                ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
                ch = ch_set[0] if a < 0.35 else ch_set[1] if a < 1.2 else ch_set[2]
                lev = fade * (1.0 - 0.3 * lag / 0.1)
                c.ch[yi[ok], xi[ok]] = ch
                c.fg[yi[ok], xi[ok]] = (bg + (base - bg) * max(0.0, lev)).astype(np.uint8)
