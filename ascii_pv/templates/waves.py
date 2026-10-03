"""Signal-and-field templates: oscilloscope, starfield, plasma, flames."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core import noise
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.typography.layer import _u16

SHADE = np.array(list(" .:-=+*#%@"))
FLAME = np.array(list(" .'^;|!*#@"))        # vertical strokes: flames should not look striped


def _paint(c: Canvas, ys: np.ndarray, xs: np.ndarray, chars, lev: np.ndarray, ctx: FrameCtx,
           accent: np.ndarray | None = None) -> None:
    pal = ctx.palette
    bg = np.array(pal.bg, float)
    col = np.array(pal.ui, float)
    rgb = bg + (col - bg) * np.clip(lev, 0, 1)[..., None]
    if accent is not None:
        acc = np.array(pal.accent, float)
        rgb = np.where(accent[..., None], bg + (acc - bg) * np.clip(lev, 0, 1)[..., None], rgb)
    c.ch[ys, xs] = chars
    c.fg[ys, xs] = rgb.astype(np.uint8)


@register
class Oscilloscope(Template):
    meta = TemplateMeta(
        "oscilloscope", "Phosphor trace built from the band energies: wave, lissajous or radial spectrum.",
        busy=0.6, energy=(0.2, 0.9), sections=frozenset({"intro", "verse", "bridge", "outro", "instrumental"}),
        tags=frozenset({"calm", "mid-energy", "dark", "neutral", "sparse"}), motifs=frozenset({"pulse", "light"}),
        lyric_blends=frozenset({"overlay", "knockout"}), hud="corners", hud_topics=('plot:lissajous', 'formula:euler', 'data', 'formula:trig'),
    )
    defaults = {"shape": "wave", "trail": 6}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        shape = str(self.params["shape"])
        bands = np.asarray(ctx.bands)
        n = 900
        s = np.linspace(0, 1, n)
        trail = int(self.params["trail"])
        for echo in range(trail, -1, -1):              # older copies first, dimmer
            t = ctx.t - echo * 0.018
            if shape == "radial":
                ang = s * math.tau
                k = np.interp((np.abs((s * 2) % 2 - 1)) * 6, np.arange(7), bands)
                r = 0.35 + 0.45 * k * (0.7 + 0.3 * np.sin(ang * 3 + t * 2))
                x = 0.5 + 0.5 * r * np.cos(ang + t * 0.3)
                y = 0.5 + 0.5 * r * np.sin(ang + t * 0.3)
            elif shape == "lissajous":
                a, b = 3 + int(bands[2] * 3), 2 + int(bands[4] * 3)
                x = 0.5 + 0.42 * np.sin(a * s * math.tau + t * 1.3) * (0.4 + 0.6 * ctx.energy)
                y = 0.5 + 0.42 * np.sin(b * s * math.tau + t * 0.9) * (0.4 + 0.6 * ctx.energy)
            else:
                x = s
                y = 0.5 + sum(bands[i] * 0.22 / (1 + i * 0.3) * np.sin(s * math.tau * (1 + i * 1.7) + t * (3 + i))
                              for i in range(7))
            X = np.clip((x * (c.w - 1)).astype(int), 0, c.w - 1)
            Y = np.clip((y * (c.h - 1)).astype(int), 0, c.h - 1)
            head = echo == 0
            lev = np.full(n, (1 - echo / (trail + 1)) * (0.35 + 0.65 * ctx.energy) + (0.3 * ctx.kick if head else 0))
            chars = np.where(head, "*", ".") if shape != "wave" else np.where(head, "~", "-")
            _paint(c, Y, X, np.full(n, chars), lev, ctx, np.full(n, head) if head else None)


@register
class Starfield(Template):
    meta = TemplateMeta(
        "starfield", "Stars rushing past; the ride speeds up in loud passages (distance = integral of loudness).",
        busy=0.7, energy=(0.3, 1.0), sections=frozenset({"intro", "prechorus", "chorus", "bridge", "outro"}),
        tags=frozenset({"energetic", "bright", "calm"}), motifs=frozenset({"star", "fly", "speed", "night", "dream"}),
        lyric_blends=frozenset({"overlay", "obstacle", "knockout"}), 
    )
    defaults = {"count": 1100, "speed": 1.0}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        n = int(self.params["count"])
        seed = str_seed(self.shot.id) & 0xFFFF
        i = np.arange(n)
        sx = _u16(i, seed) / 65535 * 2 - 1
        sy = _u16(i, seed + 1) / 65535 * 2 - 1
        z0 = _u16(i, seed + 2) / 65535
        travel = self.analysis.travel(ctx.t) * 0.35 * float(self.params["speed"])
        z = (z0 - travel) % 1.0 + 0.02                        # 1 far .. 0.02 near
        X = (c.w / 2 + sx / z * c.w * 0.12).astype(int)
        Y = (c.h / 2 + sy / z * c.h * 0.12).astype(int)
        ok = (X >= 0) & (X < c.w) & (Y >= 0) & (Y < c.h)
        near = np.clip(1 - z, 0, 1)
        # near stars leave a short streak pointing back to the vanishing point
        for back, dimk in ((0.06, 0.45), (0.03, 0.7)):
            zb = z + back
            tx = (c.w / 2 + sx / zb * c.w * 0.12).astype(int)
            ty = (c.h / 2 + sy / zb * c.h * 0.12).astype(int)
            s_ok = ok & (near > 0.7) & (tx >= 0) & (tx < c.w) & (ty >= 0) & (ty < c.h)
            _paint(c, ty[s_ok], tx[s_ok], np.full(s_ok.sum(), "."), (near[s_ok] ** 2) * dimk, ctx)
        chars = np.where(near > 0.9, "@", np.where(near > 0.75, "*", np.where(near > 0.45, "+", ".")))
        lev = 0.3 + 0.7 * near ** 1.5 + 0.2 * ctx.kick * ctx.pulse(0.1)
        order = np.argsort(near[ok])
        _paint(c, Y[ok][order], X[ok][order], chars[ok][order], lev[ok][order], ctx, (near > 0.9)[ok][order])


@register
class Plasma(Template):
    meta = TemplateMeta(
        "plasma", "Demoscene sine plasma in a character ramp; flows faster and brighter with the music.",
        busy=0.9, energy=(0.2, 0.8), sections=frozenset({"verse", "bridge", "intro", "instrumental"}),
        tags=frozenset({"calm", "bright", "neutral", "mid-energy"}), motifs=frozenset({"dream", "light", "water"}),
        lyric_blends=frozenset({"overlay", "mask", "knockout"}), 
    )
    defaults = {"scale": 1.0, "floor": 0.1}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        yy, xx = np.mgrid[0:c.h, 0:c.w].astype(float)
        sc = float(self.params["scale"]) * 0.12
        t = self.analysis.travel(ctx.t) * 0.8
        x, y = xx * sc, yy * sc * 2
        v = (np.sin(x + t) + np.sin((y + t) * 0.8) + np.sin((x + y + t) * 0.6)
             + np.sin(np.hypot(x - c.w * sc / 2, y - c.h * sc) * 1.1 - t * 1.3)) / 4      # -1..1
        u = (v + 1) / 2
        floor = float(self.params["floor"])
        lev = np.clip((u - floor) / (1 - floor), 0, 1) ** 0.8 * (0.7 + 0.4 * ctx.energy)             + 0.15 * ctx.kick * ctx.pulse(0.1)
        lev = np.clip(lev, 0, 1)
        idx = np.clip((lev * (len(SHADE) - 1)).astype(int), 0, len(SHADE) - 1)
        ys, xs = np.nonzero(idx > 0)
        _paint(c, ys, xs, SHADE[idx[ys, xs]], lev[ys, xs], ctx, (lev > 0.85)[ys, xs])


@register
class Flames(Template):
    meta = TemplateMeta(
        "flames", "Noise fire rising from the bottom; the bass pushes it higher.",
        busy=0.8, energy=(0.4, 1.0), sections=frozenset({"chorus", "prechorus", "bridge", "outro"}),
        tags=frozenset({"energetic", "dark", "busy"}), motifs=frozenset({"fire", "blood", "light"}),
        lyric_blends=frozenset({"overlay", "mask", "knockout"}), 
    )
    defaults = {"height": 0.65}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        seed = str_seed(self.shot.id) & 0xFFFF
        yy, xx = np.mgrid[0:c.h, 0:c.w].astype(float)
        rise = self.analysis.travel(ctx.t) * 3
        # noise stretched vertically (tall tongues) and scrolling upward
        n = noise.fbm(xx * 0.14, yy * 0.07 + rise, octaves=4, seed=seed)
        height = float(self.params["height"]) * (0.6 + 0.5 * ctx.bands[0] + 0.25 * ctx.kick * ctx.pulse(0.15))
        hgt = (c.h - 1 - yy) / (c.h * max(0.1, height))            # 0 at the bottom, 1 at the flame tips
        heat = np.clip(n * 1.35 - hgt * 1.0 + 0.1, 0, 1)
        idx = np.clip((heat * (len(FLAME) - 1)).astype(int), 0, len(FLAME) - 1)
        ys, xs = np.nonzero(idx > 0)
        pal = ctx.palette
        h = heat[ys, xs]
        # colour ramp: dark ui -> ui -> accent (white hot) -> err tinge at the edges
        rgb = np.where((h > 0.8)[:, None], np.array(pal.accent, float), np.array(pal.ui, float))
        bg = np.array(pal.bg, float)
        edge = (h < 0.35)[:, None]
        rgb = np.where(edge, np.array(pal.err, float), rgb)
        rgb = bg + (rgb - bg) * np.clip(0.25 + h, 0, 1)[:, None]
        c.ch[ys, xs] = FLAME[idx[ys, xs]]
        c.fg[ys, xs] = rgb.astype(np.uint8)
