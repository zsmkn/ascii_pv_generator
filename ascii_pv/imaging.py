"""RGBA image -> terminal cells, and the image moved by the music.

Modes (cells are ~1:2, so aspect is corrected everywhere):
  halfblock  '▀' per cell: fg = upper pixel, bg = lower pixel -> twice the vertical resolution, true colour
  ramp       luminance -> " .:-=+*#%@"
  glyph      code glyphs (0/1 by default) whose brightness and colour follow the picture
  edge       Sobel direction -> - | / \\ where the edges are strong; a faint fill inside
Colour: "source" keeps the picture's colours; "palette" maps luminance onto the palette ramp.

CellImage is a small bundle of arrays (chars, fg, bg, has_bg, alpha) at a fixed cell size; react() returns a
shifted / flared copy for one frame and blit() composites it onto a canvas.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.palette import Palette

RAMP = np.array(list(" .:-=+*#%@"))
GLYPHS = np.array(list("01"))
EDGE_CHARS = np.array(list("-/|\\"))


@dataclass
class CellImage:
    chars: np.ndarray      # (h, w) str
    fg: np.ndarray         # (h, w, 3) uint8
    bg: np.ndarray         # (h, w, 3) uint8 (used where has_bg)
    has_bg: np.ndarray     # (h, w) bool: the cell paints its background too (halfblock)
    alpha: np.ndarray      # (h, w) bool: the cell belongs to the picture
    lum: np.ndarray        # (h, w) float 0..1

    @property
    def w(self) -> int:
        return self.chars.shape[1]

    @property
    def h(self) -> int:
        return self.chars.shape[0]


@lru_cache(maxsize=32)
def _load(path: str) -> Image.Image:
    return Image.open(path).convert("RGBA")


def fit_cells(img_w: int, img_h: int, max_w: int, max_h: int) -> tuple[int, int]:
    """Largest (cols, rows) with the picture's aspect, given cells twice as tall as wide."""
    aspect = img_w / img_h * 2           # cols per row
    cols = min(max_w, int(max_h * aspect))
    rows = min(max_h, int(cols / aspect))
    return max(1, cols), max(1, rows)


def _tint(lum: np.ndarray, pal: Palette) -> np.ndarray:
    bg = np.array(pal.bg, float)
    ui = np.array(pal.ui, float)
    acc = np.array(pal.accent, float)
    lo = bg + (ui - bg) * np.clip(lum / 0.7, 0, 1)[..., None]
    hi = ui + (acc - ui) * np.clip((lum - 0.7) / 0.3, 0, 1)[..., None]
    return np.where((lum < 0.7)[..., None], lo, hi).astype(np.uint8)


@lru_cache(maxsize=64)
def cell_image(path: str, cols: int, rows: int, mode: str, color: str, pal: Palette) -> CellImage:
    src = _load(path)
    if mode == "halfblock":
        im = np.asarray(src.resize((cols, rows * 2), Image.LANCZOS), np.float32)
        top, bot = im[0::2], im[1::2]
        ta, ba = top[..., 3] > 110, bot[..., 3] > 110
        lum_t = (top[..., :3] @ [0.299, 0.587, 0.114]) / 255
        lum_b = (bot[..., :3] @ [0.299, 0.587, 0.114]) / 255
        if color == "palette":
            ct, cb = _tint(lum_t, pal), _tint(lum_b, pal)
        else:
            ct, cb = top[..., :3].astype(np.uint8), bot[..., :3].astype(np.uint8)
        chars = np.where(ta, "▀", np.where(ba, "▄", " "))
        fg = np.where(ta[..., None], ct, cb)
        bg = cb
        has_bg = ta & ba
        return CellImage(chars, fg, bg, has_bg, ta | ba, (lum_t + lum_b) / 2)

    small = src.resize((cols, rows), Image.LANCZOS)
    sharp = small.convert("RGB").filter(ImageFilter.UnsharpMask(radius=1, percent=160, threshold=2))
    im = np.dstack([np.asarray(sharp, np.float32), np.asarray(small.getchannel("A"), np.float32)])
    alpha = im[..., 3] > 110
    rgb = im[..., :3]
    lum = (rgb @ [0.299, 0.587, 0.114]) / 255
    # stretch contrast inside the picture so dark and pale images both use the whole ramp
    if alpha.any():
        lo, hi = np.quantile(lum[alpha], [0.03, 0.97])
        lum = np.clip((lum - lo) / max(1e-3, hi - lo), 0, 1)
    if color == "palette":
        fg = _tint(0.25 + 0.75 * lum, pal)
    else:
        # boost dim source colours so glyphs on a dark background stay visible
        gain = 1.0 / np.maximum(0.35, rgb.max(axis=2, keepdims=True) / 255)
        fg = np.clip(rgb * np.minimum(gain, 2.2), 0, 255).astype(np.uint8)
    if mode == "ramp":
        chars = RAMP[np.clip((lum * (len(RAMP) - 1) + 0.5).astype(int), 1, len(RAMP) - 1)]
    elif mode == "glyph":
        k = (np.arange(rows)[:, None] * 131 + np.arange(cols)[None, :] * 17) % 97
        chars = GLYPHS[k % len(GLYPHS)]
        fg = (fg * (0.35 + 0.65 * lum[..., None])).astype(np.uint8)
    elif mode == "edge":
        # gradients at 2x (4x vertically: rows are twice as tall), max-pooled per cell so thin lines survive
        big = src.resize((cols * 2, rows * 4), Image.LANCZOS)
        a_big = np.asarray(big.getchannel("A"), np.float32) / 255
        g = np.asarray(big.convert("L").filter(ImageFilter.SMOOTH), np.float32) * a_big
        gx = np.zeros_like(g)
        gy = np.zeros_like(g)
        gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
        gy[1:-1] = g[2:] - g[:-2]

        def per_cell(arr: np.ndarray) -> np.ndarray:
            return arr.reshape(rows, 4, cols, 2).transpose(0, 2, 1, 3).reshape(rows, cols, 8)

        mag = per_cell(np.hypot(gx, gy))
        k = mag.argmax(axis=2)[..., None]
        m = mag.max(axis=2)
        ex = np.take_along_axis(per_cell(gx), k, axis=2)[..., 0]
        ey = np.take_along_axis(per_cell(gy), k, axis=2)[..., 0]
        # the edge runs perpendicular to the gradient; y points down and a cell is half as wide as tall
        ang = (np.degrees(np.arctan2(-ex, ey * 0.5)) + 180) % 180
        d = ((ang + 22.5) // 45).astype(int) % 4                 # index into EDGE_CHARS: - / | \
        strong = m > (np.quantile(m[alpha], 0.62) if alpha.any() else m.mean())
        fill = RAMP[np.clip((lum * 4).astype(int), 0, 4)]        # faint body: " .:-="
        chars = np.where(strong, EDGE_CHARS[d], fill)
        fg = np.where(strong[..., None], fg, (fg * 0.35).astype(np.uint8))
    else:
        raise ValueError(f"unknown image mode {mode!r}")
    chars = np.where(alpha, chars, " ")
    return CellImage(chars, fg, np.zeros_like(fg), np.zeros(alpha.shape, bool), alpha & (chars != " "), lum)


def react(ci: CellImage, t: float, bands: list[float], kick: float, flux: float, energy: float, bpm: float,
          amount: float = 1.0, seed: int = 0) -> tuple[CellImage, int]:
    """The picture moved by the music. Returns (image, extra left margin in cells).

    rows    each row slides with the band matching its height (hair = highs, feet = bass), as a wave
    kick    the whole picture brightens and hops one cell up
    flux    a few cells flash to white
    quiet   some cells drop out, and the figure fills back in as the music swells
    """
    if amount <= 0:
        return ci, 0
    h, w = ci.h, ci.w
    m = 2
    phase = t * math.tau * (bpm / 60) / 2
    out_ch = np.full((h, w + 2 * m), " ", dtype=ci.chars.dtype)
    out_fg = np.zeros((h, w + 2 * m, 3), np.uint8)
    out_bg = np.zeros((h, w + 2 * m, 3), np.uint8)
    out_hb = np.zeros((h, w + 2 * m), bool)
    out_a = np.zeros((h, w + 2 * m), bool)
    for r in range(h):
        u = r / max(1, h - 1)
        e = bands[min(len(bands) - 1, int((1 - u) * len(bands)))]
        sh = int(round(e ** 1.5 * 2.2 * amount * math.sin(math.tau * u * 2.0 - phase)))
        a, b = m + sh, m + sh + w
        out_ch[r, a:b], out_fg[r, a:b], out_bg[r, a:b] = ci.chars[r], ci.fg[r], ci.bg[r]
        out_hb[r, a:b], out_a[r, a:b] = ci.has_bg[r], ci.alpha[r]
    rng = np.random.default_rng((seed * 7919 + int(t * 48)) & 0xFFFFFFFF)
    # quiet passages: drop cells (a new pattern every half beat)
    drop = max(0.0, min(1.0, (0.32 - energy) / 0.32)) * 0.3 * amount
    if drop > 0.01:
        prng = np.random.default_rng((seed * 131 + int(t * bpm / 30)) & 0xFFFFFFFF)
        gone = (prng.random(out_a.shape) < drop) & out_a
        out_ch[gone] = " "
        out_hb[gone] = False
        out_a[gone] = False
    # kick flare
    flare = min(1.0, kick * 0.9) * amount
    if flare > 0.02:
        lut = np.clip(np.arange(256) * (1 + 0.55 * flare) + 30 * flare, 0, 255).astype(np.uint8)
        out_fg = lut[out_fg]
        out_bg = np.where(out_hb[..., None], lut[out_bg], out_bg)
    # onset sparkles
    n = int(28 * max(0.0, flux - 0.35) / 0.65 * amount)
    if n:
        ys, xs = np.nonzero(out_a)
        if len(ys):
            pick = rng.choice(len(ys), size=min(n, len(ys)), replace=False)
            out_fg[ys[pick], xs[pick]] = 255
    hop = 1 if kick * amount > 0.6 else 0
    if hop:
        out_ch, out_fg, out_bg = np.roll(out_ch, -1, 0), np.roll(out_fg, -1, 0), np.roll(out_bg, -1, 0)
        out_hb, out_a = np.roll(out_hb, -1, 0), np.roll(out_a, -1, 0)
        out_a[-1] = False
    return CellImage(out_ch, out_fg, out_bg, out_hb, out_a, np.pad(ci.lum, ((0, 0), (m, m)))), m


def blit(c: Canvas, ci: CellImage, x: int, y: int, reveal: np.ndarray | None = None, dim: float = 0.0) -> None:
    """Draw the picture's cells at (x, y); `reveal` (h, w) bool hides cells that are not in yet."""
    ys, xs = np.nonzero(ci.alpha if reveal is None else ci.alpha & reveal)
    X, Y = xs + x, ys + y
    ok = (X >= 0) & (X < c.w) & (Y >= 0) & (Y < c.h)
    ys, xs, X, Y = ys[ok], xs[ok], X[ok], Y[ok]
    fg = ci.fg[ys, xs]
    if dim > 0:
        fg = (fg * (1 - dim) + np.array(c.default_bg) * dim).astype(np.uint8)
    c.ch[Y, X] = ci.chars[ys, xs]
    c.fg[Y, X] = fg
    hb = ci.has_bg[ys, xs]
    c.bg[Y[hb], X[hb]] = ci.bg[ys[hb], xs[hb]]
    c.bold[Y, X] = False
