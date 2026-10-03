"""Lettering fonts: the built-in bitmap font and TrueType faces rasterised onto the cell grid.

  bit5x5   5-row bitmap font (font5.py), scaled by whole cells: blocky, retro
  impact / black / serif / script / mono   Windows TrueType faces (or any .ttf path): smooth big letters

A TTF line is rendered at high resolution with square pixels (a cell is OVERSAMPLE wide x 2*OVERSAMPLE tall)
and box-filtered down to cells, then thresholded. Each word is rendered on its own so every ink cell
knows which word it belongs to.

Chinese / Japanese lines (core/cjk.py) use a CJK face whatever the cue's font says (the bitmap font and the
Latin faces have no such glyphs): Microsoft JhengHei Bold, or YaHei Bold when JhengHei lacks a character
(simplified forms). A thin face loses strokes on the cell grid, so bold sans it is. Their size is set by a
square ideograph instead of "H", and glued words (no space between them in the text) are drawn with no gap.
"""

from __future__ import annotations

import os
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ascii_pv.core.cjk import has_cjk

WIN = os.environ.get("WINDIR", "C:/Windows") + "/Fonts/"
TTF = {"impact": WIN + "impact.ttf", "black": WIN + "ariblk.ttf", "serif": WIN + "georgiab.ttf",
       "script": WIN + "segoeprb.ttf", "mono": WIN + "consolab.ttf"}
CJK_FACES = (WIN + "msjhbd.ttc", WIN + "msyhbd.ttc")
BITMAP = {"bit5x5"}
MIN_TTF_ROWS = 6          # below this a TTF face turns to mush; the layout falls back to the bitmap font
MIN_CJK_ROWS = 9          # an ideograph needs more rows than a capital to keep its strokes apart
OVERSAMPLE = 8            # render pixels per cell column


def font_path(name: str) -> str | None:
    if name in BITMAP:
        return None
    p = TTF.get(name, name)
    return p if os.path.exists(p) else None


def is_ttf(name: str) -> bool:
    return font_path(name) is not None


@lru_cache(maxsize=1024)
def _covers(path: str, text: str) -> bool:
    """The face has a real glyph (not the .notdef box) for every non-space character of `text`."""
    f = _face(path, 32)

    def bits(ch: str) -> bytes:
        m = f.getmask(ch)
        return bytes(m.size) + bytes(m)

    notdef = bits("\U000F0000")                       # a private-use point: always the .notdef box
    return all(bits(ch) != notdef for ch in set(text) if not ch.isspace())


def cjk_path(name: str, text: str) -> str | None:
    """The face for a CJK line: a user .ttf that covers it, else the first built-in CJK face that does."""
    own = name not in TTF and name not in BITMAP and os.path.exists(name)
    for p in ((name,) if own else ()) + CJK_FACES:
        if os.path.exists(p) and _covers(p, text):
            return p
    return next((p for p in CJK_FACES if os.path.exists(p)), None)


def resolve(name: str, words: tuple[str, ...]) -> str | None:
    """The TTF face used for `words` with the cue font `name` (None = the bitmap font)."""
    if has_cjk("".join(words)):
        return cjk_path(name, "".join(words))
    return font_path(name)


@lru_cache(None)
def _face(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


@lru_cache(maxsize=512)
def _advance(path: str, word: str) -> float:
    """Advance width of `word` at size 100, in pixels."""
    return _face(path, 100).getlength(word.upper())


CAP_FILL = 0.88          # capitals fill this share of the line's rows (lyrics are drawn in capitals)


@lru_cache(None)
def _cap(path: str, ref: str = "H") -> tuple[float, float]:
    """(top, height) of the reference glyph ("H", or a square ideograph for CJK lines) at size 100, in pixels
    from the drawing origin."""
    _, top, _, bottom = _face(path, 100).getbbox(ref)
    return float(top), float(bottom - top)


def ref_glyph(words: tuple[str, ...]) -> str:
    """The glyph that sets the letter size: a square ideograph when the text has CJK in it (pass the whole
    lyric line's words, so a Latin-only part of a mixed line keeps the same size)."""
    return "國" if has_cjk("".join(words)) else "H"


def _gaps(words: tuple[str, ...], glue: tuple[bool, ...] | None) -> list[bool]:
    """Per word after the first: is there a space before it?"""
    return [not (glue and glue[i]) for i in range(1, len(words))]


def advance_per_row(path: str, text: str, ref: str = "H") -> float:
    """Cells of width `text` takes per row of line height."""
    return _advance(path, text) / _cap(path, ref)[1] * 2 * CAP_FILL


def ttf_cols(path: str, words: tuple[str, ...], rows: int, glue: tuple[bool, ...] | None = None,
             ref: str | None = None) -> int:
    """Width in cells of `words` (one line) when its line height is `rows` cells. A row is two cell-widths
    tall, so the line is 2*rows square units high and each column is one unit wide."""
    space = (_advance(path, " ") or 30.0) * 1.2
    adv = sum(_advance(path, w) for w in words) + space * sum(_gaps(words, glue))
    return max(1, int(np.ceil(adv / _cap(path, ref or ref_glyph(words))[1] * rows * 2 * CAP_FILL)))


@lru_cache(maxsize=256)
def ttf_line(path: str, words: tuple[str, ...], rows: int, glue: tuple[bool, ...] | None = None,
             ref: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """(mask (rows, cols) bool, word index per cell (rows, cols) int, -1 outside)."""
    ref = ref or ref_glyph(words)
    cols = ttf_cols(path, words, rows, glue, ref)
    H = rows * OVERSAMPLE * 2                          # render height in px (a cell is OVERSAMPLE x 2*OVERSAMPLE)
    W = cols * OVERSAMPLE
    top100, cap100 = _cap(path, ref)
    size = max(4, int(100 * H * CAP_FILL / cap100))
    face = _face(path, size)
    y0 = H * (1 - CAP_FILL) / 2 - top100 * size / 100      # centre the capitals vertically
    space = face.getlength(" ") * 1.2
    img = np.zeros((H, W), np.float32)
    owner = np.full((H, W), -1, np.int32)
    gaps = _gaps(words, glue)
    x = 0.0
    for i, w in enumerate(words):
        if i and gaps[i - 1]:
            x += space
        im = Image.new("L", (W, H))
        ImageDraw.Draw(im).text((x, y0), w.upper(), font=face, fill=255)
        a = np.asarray(im, np.float32)
        owner[(a > 100) & (owner < 0)] = i
        img = np.maximum(img, a)
        x += face.getlength(w.upper())
    # box filter down to cells
    cell = img.reshape(rows, OVERSAMPLE * 2, cols, OVERSAMPLE).mean(axis=(1, 3))
    mask = cell > 96
    own = owner.reshape(rows, OVERSAMPLE * 2, cols, OVERSAMPLE).max(axis=(1, 3))
    return mask, np.where(mask, own, -1)
