"""Post effects on a finished background canvas (applied before the lyric layer, so lettering stays legible).

glitch    rows slip sideways, cells take the error colour, blocks of noise flash in; `amount` 0..1, bursts on kicks
scan      a brighter scan line rolls down the screen (phosphor look)
shake     the picture jolts on hard kicks            zoom    a push-in on the kick
split     red / cyan ghosts of the glyphs            strobe  the background flashes toward the accent (mild)
noise     analog speckle and a tearing row
M10:
shockwave  a ring of block glyphs racing out from the centre at a drop (glyph_boost: denser glyphs inside it)
(They all act on the background only: the lyric layer is drawn afterwards and stays steady.)
"""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.palette import Palette
from ascii_pv.typography.layer import _u16

BLOCK = np.array(list("█▓▒░#@%$"))


def glitch(c: Canvas, amount: float, frame: int, kick: float, pal: Palette, seed: int = 0) -> None:
    if amount <= 0.01:
        return
    a = min(1.0, amount * (0.6 + 0.8 * kick))
    h, w = c.h, c.w
    r = lambda i, salt: int(_u16(np.array([frame * 131 + i]), seed + salt)[0])  # noqa: E731
    # row slips
    if r(0, 1) % 100 < a * 45:
        for i in range(1 + int(a * 6)):
            row = r(i, 2) % h
            shift = int((r(i, 3) % 21 - 10) * a)
            if shift:
                for arr in (c.ch, c.fg, c.bg, c.bold):
                    arr[row] = np.roll(arr[row], shift, axis=0)
    # error-colour speckle
    if r(1, 4) % 100 < a * 50:
        n = int(a * w * h * 0.01)
        idx = _u16(np.arange(n) + frame * 977, seed + 5)
        ys, xs = idx % h, (idx // 7) % w
        c.fg[ys, xs] = pal.err_at(0.8)
    # noise blocks (strong glitch only)
    if a > 0.55 and r(2, 6) % 100 < (a - 0.55) * 120:
        for i in range(1 + int((a - 0.55) * 8)):
            bw, bh = 3 + r(i, 7) % 10, 1 + r(i, 8) % 3
            x, y = r(i, 9) % max(1, w - bw), r(i, 10) % max(1, h - bh)
            k = _u16(np.arange(bw * bh) + i * 31 + frame, seed + 11).reshape(bh, bw)
            c.ch[y:y + bh, x:x + bw] = BLOCK[k % len(BLOCK)]
            c.fg[y:y + bh, x:x + bw] = pal.err_at(0.9) if i % 3 == 0 else pal.ui_at(0.8)


def scan(c: Canvas, t: float, amount: float, pal: Palette) -> None:
    if amount <= 0.01:
        return
    row = int(t * 9) % c.h
    lit = c.ch[row] != " "
    c.fg[row, lit] = np.maximum(c.fg[row, lit], np.array(pal.ui_at(0.5 + 0.5 * amount), np.uint8))


# ---------------------------------------------------------------- M8: beat-driven camera and signal effects
# All take `hit` = kick * beat pulse (0..1): they fire on the drums and settle between them.

def _jitter(frame: int, salt: int) -> int:
    return int(_u16(np.array([frame]), salt)[0])


def shake(c: Canvas, amount: float, hit: float, frame: int) -> None:
    """The whole picture jolts a cell or two on hard kicks."""
    s = amount * hit
    if s < 0.15:
        return
    dx = _jitter(frame, 21) % 5 - 2 if s > 0.5 else 0      # hard hits: up to 2 cells
    if dx == 0:
        dx = 1 if frame % 2 else -1                          # always move at least one cell
    dy = _jitter(frame, 22) % 3 - 1 if s > 0.4 else 0
    for arr in (c.ch, c.fg, c.bg, c.bold):
        arr[:] = np.roll(np.roll(arr, dx, axis=1), dy, axis=0)


def zoom(c: Canvas, amount: float, hit: float) -> None:
    """A quick push-in on the kick: the picture scales up around the centre."""
    s = 1 + 0.12 * amount * hit
    if s < 1.01:
        return
    yy, xx = np.mgrid[0:c.h, 0:c.w]
    sy = np.clip(((yy - c.h / 2) / s + c.h / 2).astype(int), 0, c.h - 1)
    sx = np.clip(((xx - c.w / 2) / s + c.w / 2).astype(int), 0, c.w - 1)
    for arr in (c.ch, c.fg, c.bg, c.bold):
        arr[:] = arr[sy, sx]


def split(c: Canvas, amount: float, hit: float, pal: Palette) -> None:
    """RGB split: red and cyan ghosts of the glyphs slide out sideways, into empty cells only."""
    off = int(round(1 + 2 * amount * hit))
    if amount <= 0.01:
        return
    lit = c.ch != " "
    for dx, col in ((-off, (255, 40, 60)), (off, (40, 220, 255))):
        ghost = np.roll(lit, dx, axis=1) & ~lit
        src = np.roll(c.ch, dx, axis=1)
        c.ch[ghost] = src[ghost]
        base = np.array(pal.bg, float)
        c.fg[ghost] = (base + (np.array(col, float) - base) * (0.25 + 0.35 * amount)).astype(np.uint8)


def strobe(c: Canvas, amount: float, hit: float, pal: Palette) -> None:
    """The background flashes toward the accent colour on the strongest kicks (kept mild)."""
    k = amount * hit
    if k < 0.35:
        return
    tint = np.array(pal.accent, float)
    c.bg[:] = (c.bg * (1 - 0.18 * k) + tint * 0.18 * k).astype(np.uint8)


def noise(c: Canvas, amount: float, frame: int, pal: Palette) -> None:
    """Analog noise: sparse speckle and one tearing row."""
    if amount <= 0.01:
        return
    n = int(c.w * c.h * 0.02 * amount)
    idx = _u16(np.arange(n) + frame * 7919, 31)
    ys, xs = idx % c.h, (idx // 3) % c.w
    c.ch[ys, xs] = np.array(list(".,'`:"))[idx % 5]
    c.fg[ys, xs] = pal.ui_at(0.3)
    if _jitter(frame, 32) % 100 < 40 * amount:
        row = _jitter(frame, 33) % c.h
        shift = _jitter(frame, 34) % 9 - 4
        for arr in (c.ch, c.fg):
            arr[row] = np.roll(arr[row], shift, axis=0)


# ---------------------------------------------------------------- M10: the drop shockwave

DENSITY = " .:-=+*#%@"
_DENS_IDX = {ch: i for i, ch in enumerate(DENSITY)}


def glyph_boost(c: Canvas, amount: float, hit: float, mask: np.ndarray | None = None) -> None:
    """Hits make the glyphs denser (". : - = + * # % @"), not the screen brighter: an ASCII-native flash.
    Glyphs outside the ramp count as its middle. `mask` limits it to some cells."""
    k = amount * hit
    if k < 0.15:
        return
    lit = c.ch != " "
    if mask is not None:
        lit &= mask
    if not lit.any():
        return
    vals, inv = np.unique(c.ch[lit], return_inverse=True)
    idx = np.array([_DENS_IDX.get(v, 5) for v in vals])[inv]
    step = int(round(1 + 3 * min(1.0, k)))
    c.ch[lit] = np.array(list(DENSITY))[np.minimum(len(DENSITY) - 1, idx + step)]
    c.fg[lit] = np.minimum(255, c.fg[lit] * (1 + 0.2 * k)).astype(np.uint8)


def shockwave(c: Canvas, k: float, pal: Palette) -> None:
    """The drop: a ring of block glyphs races out from the centre (k 1 -> 0 as it expands), glyphs inside it
    hit hard. The background colour is left alone."""
    if k <= 0.02:
        return
    yy, xx = np.mgrid[0:c.h, 0:c.w]
    d = np.hypot((xx - c.w / 2) / 2, yy - c.h / 2)          # cells are ~2x taller than wide
    r = (1 - k) * (d.max() + 3)
    band = np.abs(d - r)
    ring = band < 1.6
    c.ch[ring] = np.array(list("█▓▒░"))[np.minimum(3, (band[ring] * 2.2).astype(int))]
    c.fg[ring] = pal.accent_at(0.55 + 0.45 * k)
    glyph_boost(c, 1.0, k, mask=d < r - 1.6)
