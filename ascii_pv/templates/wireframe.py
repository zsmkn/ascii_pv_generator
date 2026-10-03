"""Wireframe mesh: a rotating cube / torus / sphere in perspective, edges drawn with depth-shaded glyphs.
Rotation speeds up with loudness; the mesh swells on each kick."""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx
from ascii_pv.templates.base import Template, TemplateMeta, register

STEPS = 28          # samples per edge
DEPTH_CHARS = np.array(list(".:-=+*#"))


@lru_cache(None)
def mesh(form: str) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (N,3) in -1..1, edges (E,2))"""
    if form == "cube":
        v = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], float) * 0.8
        e = [(i, j) for i in range(8) for j in range(i + 1, 8) if np.sum(v[i] != v[j]) == 1]
        return v, np.array(e)
    if form == "torus":
        R, r, nu, nv = 0.7, 0.3, 16, 8
        v = np.array([[(R + r * math.cos(b)) * math.cos(a), r * math.sin(b), (R + r * math.cos(b)) * math.sin(a)]
                      for a in np.linspace(0, math.tau, nu, endpoint=False)
                      for b in np.linspace(0, math.tau, nv, endpoint=False)])
        e = []
        for i in range(nu):
            for j in range(nv):
                a = i * nv + j
                e += [(a, i * nv + (j + 1) % nv), (a, ((i + 1) % nu) * nv + j)]
        return v, np.array(e)
    # sphere (lat/long)
    nlat, nlon = 7, 14
    v = [[0, 1, 0]]
    for i in range(1, nlat):
        th = math.pi * i / nlat
        for j in range(nlon):
            ph = math.tau * j / nlon
            v.append([math.sin(th) * math.cos(ph), math.cos(th), math.sin(th) * math.sin(ph)])
    v.append([0, -1, 0])
    v = np.array(v) * 0.9
    e = [(0, 1 + j) for j in range(nlon)]
    for i in range(nlat - 1):
        base = 1 + i * nlon
        for j in range(nlon):
            e.append((base + j, base + (j + 1) % nlon))
            if i < nlat - 2:
                e.append((base + j, base + nlon + j))
    last = len(v) - 1
    e += [(1 + (nlat - 2) * nlon + j, last) for j in range(nlon)]
    return v, np.array(e)


def rotate(v: np.ndarray, a: float, b: float) -> np.ndarray:
    ca, sa, cb, sb = math.cos(a), math.sin(a), math.cos(b), math.sin(b)
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    x, z = x * ca + z * sa, z * ca - x * sa
    y, z = y * cb - z * sb, y * sb + z * cb
    return np.stack([x, y, z], axis=1)


@register
class Wireframe(Template):
    meta = TemplateMeta(
        "wireframe_mesh", "Rotating wireframe (cube / torus / sphere); spins with loudness, swells on kicks.",
        energy=(0.2, 0.8), sections=frozenset({"verse", "intro", "bridge"}),
        tags=frozenset({"calm", "mid-energy", "dark", "neutral"}), motifs=frozenset({"star", "time", "dream"}),
        lyric_blends=frozenset({"overlay", "knockout"}), hud="sides", hud_topics=('formula:torus', 'code:donut', 'plot:sin', 'data'),
    )
    defaults = {"form": "torus", "size": 0.9, "spin": 0.5, "annotate": True}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        v, e = mesh(str(self.params["form"]))
        # angle integrates speed over time only approximately (pure function of t): base spin + energy wobble
        spin = float(self.params["spin"])
        a = ctx.local_t * spin * 1.3 + 0.4 * math.sin(ctx.local_t * 0.7) * ctx.energy
        b = 0.55 + 0.35 * math.sin(ctx.local_t * spin * 0.6)
        scale = float(self.params["size"]) * (1 + 0.12 * ctx.kick * ctx.pulse(0.12))
        p = rotate(v * scale, a, b)
        ctx.live.update(A=a, B=b, R=0.7, r=0.3, form=self.params["form"])
        persp = 3.2 / (3.2 + p[:, 2])
        half = min(c.w / 2, c.h)     # cells are ~2x taller than wide
        sx = c.w / 2 + p[:, 0] * persp * half * 0.9
        sy = c.h / 2 + p[:, 1] * persp * half * 0.45
        u = np.linspace(0, 1, STEPS)[None, :]
        i0, i1 = e[:, 0], e[:, 1]
        X = sx[i0][:, None] + (sx[i1] - sx[i0])[:, None] * u
        Y = sy[i0][:, None] + (sy[i1] - sy[i0])[:, None] * u
        Z = p[i0, 2][:, None] + (p[i1, 2] - p[i0, 2])[:, None] * u
        depth = np.clip((1.2 - Z) / 2.4, 0, 1)          # 1 = near
        X, Y, depth = np.rint(X).ravel().astype(int), np.rint(Y).ravel().astype(int), depth.ravel()
        ok = (X >= 0) & (X < c.w) & (Y >= 0) & (Y < c.h)
        X, Y, depth = X[ok], Y[ok], depth[ok]
        order = np.argsort(depth)                        # near samples drawn last
        X, Y, depth = X[order], Y[order], depth[order]
        chars = DEPTH_CHARS[np.minimum(len(DEPTH_CHARS) - 1, (depth * len(DEPTH_CHARS)).astype(int))]
        lev = 0.2 + 0.8 * depth * (0.6 + 0.4 * ctx.energy)
        bg = np.array(pal.bg, float)
        rgb = (bg + (np.array(pal.ui, float) - bg) * lev[:, None]).astype(np.uint8)
        c.ch[Y, X] = chars
        c.fg[Y, X] = rgb
        # vertices
        vx, vy = np.rint(sx).astype(int), np.rint(sy).astype(int)
        near = p[:, 2] < 0
        okv = (vx >= 0) & (vx < c.w) & (vy >= 0) & (vy < c.h) & near
        c.ch[vy[okv], vx[okv]] = "@"
        c.fg[vy[okv], vx[okv]] = pal.accent_at(0.6 + 0.4 * ctx.pulse(0.15))
        if self.params.get("annotate", True):
            self._annotate(c, ctx, v, e, p, sx, sy)

    def _annotate(self, c: Canvas, ctx: FrameCtx, v: np.ndarray, e: np.ndarray, p: np.ndarray, sx: np.ndarray,
                  sy: np.ndarray) -> None:
        """Tracking brackets around the mesh, its stats, and live coordinates on the two nearest vertices."""
        pal = ctx.palette
        col = pal.accent_at(0.75)
        x0, x1 = int(sx.min()) - 2, int(sx.max()) + 2
        y0, y1 = int(sy.min()) - 1, int(sy.max()) + 1
        for (x, y, s) in ((x0, y0, "┌─"), (x1 - 1, y0, "─┐"), (x0, y1, "└─"), (x1 - 1, y1, "─┘")):
            c.put(x, y, s, col)
        tag = f"OBJ:{str(self.params['form']).upper()}  V={len(v)} E={len(e)}"
        c.put((x0 + x1 - len(tag)) // 2, y0 - 1, tag, pal.ui_at(0.6))
        c.put(x1 - 21, y1 + 1, f"Z {p[:, 2].min():+.2f}..{p[:, 2].max():+.2f}", pal.ui_at(0.6))
        for k in np.argsort(p[:, 2])[:2]:                       # nearest vertices
            x, y = int(round(sx[k])) + 2, int(round(sy[k]))
            c.put(x, y, f"({p[k, 0]:+.2f},{p[k, 1]:+.2f},{p[k, 2]:+.2f})", pal.ui_at(0.7))
