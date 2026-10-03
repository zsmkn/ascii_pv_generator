"""Transitions between two shots. Each takes the outgoing frame `a`, the incoming frame `b` (same size), the
progress u (0..1) and writes the mix into `b`.

cut       nothing (b as is)
flash     b, washed toward the accent colour at the start (for hits into a chorus)
dissolve  cells switch from a to b in a random order
crt_on    b opens from a bright horizontal line, like a CRT powering on
glitch    b, with a's rows torn through it early on
wipe      a hard edge sweeps left to right
blocks    4x2 tiles flip from a to b in random order
slide     b pushes a off to the left
zoom      a rushes toward the viewer and breaks up into b
reflow    a's glyphs fly across the screen and land where b's are (ref B's reflow, sorted instead of Hilbert)
"""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.palette import Palette
from ascii_pv.typography.layer import _u16

DURATION = {"cut": 0.0, "flash": 0.35, "dissolve": 0.6, "crt_on": 0.5, "glitch": 0.4, "wipe": 0.45,
            "blocks": 0.5, "slide": 0.5, "zoom": 0.55, "reflow": 0.8}


def _smooth(u: float) -> float:
    return u * u * (3 - 2 * u)


def _copy_cells(dst: Canvas, src: Canvas, m: np.ndarray) -> None:
    dst.ch[m] = src.ch[m]
    dst.fg[m] = src.fg[m]
    dst.bg[m] = src.bg[m]
    dst.bold[m] = src.bold[m]


def apply(kind: str, a: Canvas | None, b: Canvas, u: float, pal: Palette, seed: int = 0) -> None:
    u = min(1.0, max(0.0, u))
    if kind == "cut" or u >= 1.0:
        return
    h, w = b.h, b.w
    if kind == "flash":
        k = (1 - u) ** 2
        lit = b.ch != " "
        acc = np.array(pal.accent, float)
        b.fg[lit] = (b.fg[lit] + (acc - b.fg[lit]) * k).astype(np.uint8)
        # a burst of glyphs across the screen at the very start
        if u < 0.35:
            n = int(w * h * 0.25 * (1 - u / 0.35))
            idx = _u16(np.arange(n), seed + 1)
            ys, xs = idx % h, (idx // 3) % w
            b.ch[ys, xs] = "#"
            b.fg[ys, xs] = pal.accent_at(1 - u)
    elif kind == "dissolve" and a is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        order = _u16(yy * 7919 + xx, seed + 2) / 65535.0
        _copy_cells(b, a, order >= u)
    elif kind == "wipe" and a is not None:
        edge = int(u * (w + 4)) - 2
        xx = np.broadcast_to(np.arange(w)[None, :], (h, w))
        _copy_cells(b, a, xx > edge)
        col = min(w - 1, max(0, edge))
        b.ch[:, col] = "|"
        b.fg[:, col] = pal.accent
    elif kind == "crt_on":
        half = max(0.5, u ** 1.5 * h / 2)
        yy = np.arange(h)[:, None]
        hidden = np.abs(yy - (h - 1) / 2) > half
        hidden = np.broadcast_to(hidden, (h, w))
        b.ch[hidden] = " "
        b.bg[hidden] = pal.bg
        if u < 0.5:
            mid = h // 2
            b.ch[mid, :] = "="
            b.fg[mid, :] = pal.accent_at(1 - u)
    elif kind == "glitch" and a is not None:
        rows = _u16(np.arange(h), seed + 3 + int(u * 20)) % 100 < (1 - u) * 60
        for y in np.nonzero(rows)[0]:
            shift = int(_u16(np.array([y]), seed + 4 + int(u * 20))[0] % 17) - 8
            for arr_b, arr_a in ((b.ch, a.ch), (b.fg, a.fg)):
                arr_b[y] = np.roll(arr_a[y], shift, axis=0)
    elif kind == "blocks" and a is not None:
        yy, xx = np.mgrid[0:h, 0:w]
        tile = (yy // 2) * 1000 + (xx // 4)
        order = _u16(tile, seed + 5) / 65535.0
        _copy_cells(b, a, order >= u)
    elif kind == "slide" and a is not None:
        off = int(_smooth(u) * w)
        if off < w:
            keep = np.zeros((h, w), bool)
            keep[:, : w - off] = True
            for dst, src in ((b.ch, a.ch), (b.fg, a.fg), (b.bg, a.bg), (b.bold, a.bold)):
                new = np.roll(dst, w - off, axis=1) if off else dst.copy()
                old = np.roll(src, -off, axis=1)
                dst[:] = np.where(keep[..., None] if dst.ndim == 3 else keep, old, new)
    elif kind == "zoom" and a is not None:
        s = 1 + 2.5 * u * u
        yy, xx = np.mgrid[0:h, 0:w]
        sy = np.clip(((yy - h / 2) / s + h / 2).astype(int), 0, h - 1)
        sx = np.clip(((xx - w / 2) / s + w / 2).astype(int), 0, w - 1)
        order = _u16(yy * 7919 + xx, seed + 6) / 65535.0
        show_a = order >= _smooth(u)
        for dst, src in ((b.ch, a.ch), (b.fg, a.fg), (b.bg, a.bg)):
            zoomed = src[sy, sx]
            m = show_a[..., None] if dst.ndim == 3 else show_a
            dst[:] = np.where(m, zoomed, dst)
    elif kind == "reflow" and a is not None:
        _reflow(a, b, u, pal, seed)


def _reflow(a: Canvas, b: Canvas, u: float, pal: Palette, seed: int) -> None:
    """a's glyphs fly to b's glyph positions (matched in reading order), b fades in underneath."""
    ay, ax = np.nonzero(a.ch != " ")
    by, bx = np.nonzero(b.ch != " ")
    n = min(len(ay), len(by), 2500)
    if n == 0:
        return
    pick_a = np.linspace(0, len(ay) - 1, n).astype(int)
    pick_b = np.linspace(0, len(by) - 1, n).astype(int)
    ay, ax, by, bx = ay[pick_a], ax[pick_a], by[pick_b], bx[pick_b]
    chars = np.where(u < 0.5, a.ch[ay, ax], b.ch[by, bx])
    cols = np.where(u < 0.5, a.fg[ay, ax].T, b.fg[by, bx].T).T
    # each particle leaves at its own moment and travels on an ease-in-out curve with a little lift
    k = _u16(np.arange(n), seed + 7) / 65535.0
    p = np.clip((u - k * 0.35) / 0.65, 0, 1)
    e = p * p * (3 - 2 * p)
    x = np.rint(ax + (bx - ax) * e).astype(int)
    y = np.rint(ay + (by - ay) * e - np.sin(e * np.pi) * 3).astype(int)
    fade = _smooth(min(1.0, max(0.0, (u - 0.4) / 0.6)))
    b.dim(1 - fade)                                   # b appears as the particles arrive
    ok = (x >= 0) & (x < b.w) & (y >= 0) & (y < b.h)
    b.ch[y[ok], x[ok]] = chars[ok]
    b.fg[y[ok], x[ok]] = cols[ok]
