"""Canvas -> RGB pixels with a monospace font (used by `preview --png` and the MP4 renderer).

Every (char, bold) is rasterised once into a coverage mask of exactly one cell; a frame is then one numpy
gather + blend: pixel = bg + (fg - bg) * mask[char]. Block elements are drawn as exact rectangles so they tile
with no gaps; characters Consolas lacks fall back to Segoe UI Symbol, or Microsoft JhengHei / YaHei for Chinese
and Japanese. Wide characters get a two-cell glyph split into left and right halves.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_pv.core.canvas import Canvas, char_width

DEFAULT_FONT = os.environ.get("ASCII_PV_FONT", "C:/Windows/Fonts/consola.ttf")
DEFAULT_FONT_BOLD = os.environ.get("ASCII_PV_FONT_BOLD", "C:/Windows/Fonts/consolab.ttf")
FALLBACK_FONT = os.environ.get("ASCII_PV_FONT_FALLBACK", "C:/Windows/Fonts/seguisym.ttf")
CJK_FONTS = [p for p in (os.environ.get("ASCII_PV_FONT_CJK", ""), "C:/Windows/Fonts/msjh.ttc",
                         "C:/Windows/Fonts/msyh.ttc") if p]

# (x0, y0, x1, y1) as fractions of the cell
BLOCKS = {"█": (0, 0, 1, 1), "▀": (0, 0, 1, .5), "▄": (0, .5, 1, 1), "▌": (0, 0, .5, 1), "▐": (.5, 0, 1, 1),
          **{ch: (0, 1 - (i + 1) / 8, 1, 1) for i, ch in enumerate("▁▂▃▄▅▆▇")}}
SHADES = {"░": 0.25, "▒": 0.5, "▓": 0.75}


@lru_cache(None)
def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(None)
def _has_glyph(path: str, ch: str) -> bool:
    f = _font(path, 16)

    def bits(s: str) -> bytes:
        m = f.getmask(s)
        return bytes(m.size) + bytes(m)

    return bits(ch) != bits("\U000F0000")   # a private-use point: always the .notdef box


@lru_cache(None)
def fit_font_size(cell_w: int, cell_h: int, path: str = DEFAULT_FONT) -> int:
    """Largest size whose advance and line height fit the cell."""
    size = 4
    for s in range(4, 200):
        f = _font(path, s)
        asc, desc = f.getmetrics()
        if f.getlength("M") > cell_w + 0.01 or asc + desc > cell_h:
            break
        size = s
    return size


@lru_cache(None)
def cell_size(size: int, path: str = DEFAULT_FONT) -> tuple[int, int]:
    """Cell (w, h) of the font at `size` (for callers that pick a font size rather than a cell)."""
    f = _font(path, size)
    asc, desc = f.getmetrics()
    return int(round(f.getlength("M"))), asc + desc


class Rasterizer:
    def __init__(self, cell_w: int, cell_h: int, font_size: int | None = None):
        self.cw, self.ch = cell_w, cell_h
        self.size = font_size or fit_font_size(cell_w, cell_h)
        f = _font(DEFAULT_FONT, self.size)
        asc, desc = f.getmetrics()
        self.y_off = (cell_h - (asc + desc)) // 2
        self._index: dict[tuple, int] = {}
        self._masks: list[np.ndarray] = []
        self._atlas: np.ndarray | None = None
        self._key(" ", False)                         # index 0 = blank

    # ------------------------------------------------------------ atlas

    def _draw(self, ch: str, bold: bool, width_cells: int) -> np.ndarray:
        cw, chh = self.cw * width_cells, self.ch
        if ch in BLOCKS:
            x0, y0, x1, y1 = BLOCKS[ch]
            m = np.zeros((chh, cw), np.float32)
            m[round(y0 * chh):round(y1 * chh), round(x0 * cw):round(x1 * cw)] = 1
            return m
        if ch in SHADES:
            return np.full((chh, cw), SHADES[ch], np.float32)
        if ch == " ":
            return np.zeros((chh, cw), np.float32)
        path = DEFAULT_FONT_BOLD if bold else DEFAULT_FONT
        if not _has_glyph(path, ch):
            for alt in [FALLBACK_FONT] + CJK_FONTS:
                if os.path.exists(alt) and _has_glyph(alt, ch):
                    path = alt
                    break
        img = Image.new("L", (cw, chh))
        f = _font(path, self.size)
        adv = f.getlength(ch)
        y = self.y_off
        if path in CJK_FONTS:                         # its own metrics: centre the square ideograph box
            _, top, _, bottom = f.getbbox("國")
            y = (chh - (bottom - top)) / 2 - top
        ImageDraw.Draw(img).text((max(0.0, (cw - adv) / 2), y), ch, font=f, fill=255)
        return np.asarray(img, np.float32) / 255.0

    def _key(self, ch: str, bold: bool, half: str = "") -> int:
        k = (ch, bold, half)
        i = self._index.get(k)
        if i is None:
            if half:
                full = self._draw(ch, bold, 2)
                m = full[:, :self.cw] if half == "L" else full[:, self.cw:]
            else:
                m = self._draw(ch, bold, 1)
            i = self._index[k] = len(self._masks)
            self._masks.append(np.ascontiguousarray(m))
            self._atlas = None
        return i

    def _indices(self, c: Canvas) -> np.ndarray:
        chars, bold = c.ch, c.bold
        idx = np.zeros(chars.shape, np.int32)
        for b in (False, True):
            sel = bold == b
            if not sel.any():
                continue
            uniq, inv = np.unique(chars[sel], return_inverse=True)
            lut = np.array([self._key(u, b) if u != "" and char_width(u) == 1 else 0 for u in uniq], np.int32)
            idx[sel] = lut[inv.ravel()]
        # wide characters: head = left half, the continuation cell to its right = right half
        wide_y, wide_x = np.nonzero(np.vectorize(lambda s: s != "" and char_width(s) == 2, otypes=[bool])(chars))
        for y, x in zip(wide_y, wide_x):
            idx[y, x] = self._key(str(chars[y, x]), bool(bold[y, x]), "L")
            if x + 1 < c.w:
                idx[y, x + 1] = self._key(str(chars[y, x]), bool(bold[y, x]), "R")
        return idx

    # ------------------------------------------------------------ render

    def render(self, c: Canvas) -> np.ndarray:
        """(h * cell_h, w * cell_w, 3) uint8"""
        idx = self._indices(c)
        if self._atlas is None:
            self._atlas = np.round(np.stack(self._masks) * 255).astype(np.uint16)
        H, W = c.h * self.ch, c.w * self.cw
        # coverage in pixel space, then an integer blend (x / 255 as (x + 1 + (x >> 8)) >> 8)
        a = self._atlas[idx].transpose(0, 2, 1, 3).reshape(H, W)[..., None]
        fg = np.repeat(np.repeat(c.fg, self.ch, axis=0), self.cw, axis=1).astype(np.uint16)
        bg = np.repeat(np.repeat(c.bg, self.ch, axis=0), self.cw, axis=1).astype(np.uint16)
        x = fg * a + bg * (255 - a)
        return ((x + 1 + (x >> 8)) >> 8).astype(np.uint8)


def to_image(c: Canvas, size: int = 16) -> Image.Image:
    """Preview helper: font size in, image out."""
    cw, chh = cell_size(size)
    return Image.fromarray(Rasterizer(cw, chh, size).render(c), "RGB")
