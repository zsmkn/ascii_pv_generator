"""Text -> InkMask: which cells of an area a lyric line covers when drawn as big letters.

Layout: the words are broken into 1..MAX_LINES lines (minimising the widest line) and the biggest letters that
fit win.
  bitmap (bit5x5)  each font pixel becomes px_w x px_h cells, both integers (fractional scales make strokes
                   uneven); px_w between px_h and 2*px_h (cells are ~1:2, so 2*px_h is a square pixel)
  TrueType         whole rows per line (at least fonts.MIN_TTF_ROWS); below that the bitmap font takes over
  CJK              a CJK face (fonts.resolve), at least fonts.MIN_CJK_ROWS per line; never the bitmap font.
                   Glued words (core/cjk.py: no space between them) are drawn without a gap, and lines break
                   at the text's own spaces when that costs little width.
If nothing fits, layout() returns None and the caller splits the line into chunks (chunks()).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy import ndimage

from ascii_pv.core.cjk import has_cjk
from ascii_pv.typography import fonts
from ascii_pv.typography.font5 import HEIGHT, render_text, text_bits_width

MAX_LINES = 4
LINE_GAP = 1.2           # blank rows between bitmap text lines, in font pixels
TTF_GAP = 0.25           # blank rows between TrueType lines, as a share of the line height
SPACE_BITS = text_bits_width(" ") + 2


@dataclass(frozen=True)
class InkMask:
    xs: np.ndarray          # (N,) cell x of each ink cell
    ys: np.ndarray          # (N,) cell y
    word: np.ndarray        # (N,) word index in the source line
    edge: np.ndarray        # (N,) True if a 4-neighbour is not ink
    depth: np.ndarray       # (N,) cells to the nearest edge (0 on the edge)
    order: np.ndarray       # (N,) 0..1 reading position (left to right, top to bottom)
    mask: np.ndarray        # (H, W) bool
    bbox: tuple[int, int, int, int]   # x0, y0, x1, y1 (exclusive)
    px_h: float             # rows per bitmap-font pixel (TTF: rows per line / 5): how big the letters came out
    font: str = "bit5x5"
    cjk: bool = False       # Chinese / Japanese glyphs (thin, dense strokes)

    @property
    def n(self) -> int:
        return len(self.xs)


def _best_split(widths: list[float], space: float, k: int, glue: tuple[bool, ...] | None = None,
                cut_cost: float = 0.0) -> list[tuple[int, int]]:
    """Split words into k consecutive runs minimising the widest run (brute force; lines are short). Glued
    words have no space between them; cutting between two of them costs `cut_cost` of width."""
    n = len(widths)
    if k >= n:
        return [(i, i + 1) for i in range(n)]
    gap = [0.0] + [0.0 if glue and glue[i] else space for i in range(1, n)]

    def run_w(a: int, b: int) -> float:
        return sum(widths[a:b]) + sum(gap[a + 1:b])

    best, best_cuts = None, None
    for cuts in combinations(range(1, n), k - 1):
        edges = (0, *cuts, n)
        m = max(run_w(a, b) for a, b in zip(edges, edges[1:]))
        m += cut_cost * sum(1 for c in cuts if glue and glue[c])
        if best is None or m < best:
            best, best_cuts = m, edges
    return list(zip(best_cuts, best_cuts[1:]))


def _sub_glue(glue: tuple[bool, ...] | None, a: int, b: int) -> tuple[bool, ...] | None:
    """Glue of words[a:b] as a line of its own (its first word follows nothing)."""
    return (False, *glue[a + 1:b]) if glue else None


def _layout_bitmap(words: tuple[str, ...], area_w: int, area_h: int, max_px_h: int):
    widths = [text_bits_width(w) for w in words]
    best, best_key = None, None
    for k in range(1, min(MAX_LINES, len(words)) + 1):
        runs = _best_split(widths, SPACE_BITS, k)
        bits_w = max(sum(widths[a:b]) + SPACE_BITS * (b - a - 1) for a, b in runs)
        px_w_max = area_w // bits_w
        px_h = min(int(area_h // (HEIGHT * k + LINE_GAP * (k - 1))), px_w_max, max_px_h)
        if px_h < 1:
            continue
        px_w = min(2 * px_h, px_w_max)
        key = (px_h, px_w, -k)
        if best_key is None or key > best_key:
            best, best_key = ("bit5x5", runs, px_w, px_h), key
    return best


def _layout_ttf(path: str, words: tuple[str, ...], area_w: int, area_h: int, max_rows: int,
                glue: tuple[bool, ...] | None = None, min_rows: int = fonts.MIN_TTF_ROWS):
    # cells of width per row of height, measured on the line's own reference glyph (so CJK sizes match)
    ref = fonts.ref_glyph(words)
    per_row = [fonts.advance_per_row(path, w, ref) for w in words]
    space = fonts.advance_per_row(path, " ", ref) * 1.2
    cut_cost = float(np.median(per_row)) if glue and any(glue) else 0.0
    best, best_rows = None, 0
    for k in range(1, min(MAX_LINES, len(words)) + 1):
        runs = _best_split(per_row, space, k, glue, cut_cost)
        widest = max(sum(per_row[a:b]) + sum(0.0 if glue and glue[i] else space for i in range(a + 1, b))
                     for a, b in runs)
        rows = int(min(area_h / (k + TTF_GAP * (k - 1)), area_w / max(widest, 1e-6), max_rows))
        # the estimate can be off by a column or two: shrink until every run really fits
        while rows >= min_rows and any(
                fonts.ttf_cols(path, words[a:b], rows, _sub_glue(glue, a, b), ref) > area_w for a, b in runs):
            rows -= 1
        if rows >= min_rows and rows > best_rows:
            best, best_rows = (path, runs, rows, rows), rows
    return best


@lru_cache(maxsize=1024)
def layout(words: tuple[str, ...], area_w: int, area_h: int, max_px_h: int = 99, font: str = "bit5x5",
           glue: tuple[bool, ...] | None = None):
    """-> (font used: "bit5x5" or a TTF path, runs, px_w, px_h) or None when nothing readable fits. For TTF
    px_w = px_h = rows."""
    if not words:
        return None
    path = fonts.resolve(font, words)
    if has_cjk("".join(words)):
        if path is None:
            return None
        lay = _layout_ttf(path, words, area_w, area_h, max_px_h * HEIGHT, glue, fonts.MIN_CJK_ROWS)
        if lay is None and len(words) == 1:            # a single glyph: smaller beats not at all
            lay = _layout_ttf(path, words, area_w, area_h, max_px_h * HEIGHT, glue, fonts.MIN_TTF_ROWS)
        return lay
    if path is not None:
        lay = _layout_ttf(path, words, area_w, area_h, max_px_h * HEIGHT, glue)
        if lay is not None:
            return lay
    return _layout_bitmap(words, area_w, area_h, max_px_h)


def fits(words: tuple[str, ...], area_w: int, area_h: int, font: str = "bit5x5",
         glue: tuple[bool, ...] | None = None) -> bool:
    return layout(words, area_w, area_h, font=font, glue=glue) is not None


def chunks(words: tuple[str, ...], area_w: int, area_h: int, font: str = "bit5x5",
           glue: tuple[bool, ...] | None = None) -> list[tuple[int, int]]:
    """Split into consecutive word runs that each fit: as few runs as possible, then as few cuts inside a glued
    phrase as possible (CJK: break at the text's spaces), then runs of even length (no one-word leftover at
    the end). A single word that never fits stands alone."""
    n = len(words)

    def ok(a: int, b: int) -> bool:
        return b - a == 1 or fits(words[a:b], area_w, area_h, font, _sub_glue(glue, a, b))

    # best[b] = (runs, glued cuts, sum of squared run lengths) for words[:b]; a longer run that does not fit
    # means no longer one will, so each start stops at its first failure
    best: list[tuple | None] = [None] * (n + 1)
    best[0] = (0, 0, 0)
    prev = [0] * (n + 1)
    for a in range(n):
        if best[a] is None:
            continue
        r, g, sq = best[a]
        cut = 1 if a and glue and glue[a] else 0
        for b in range(a + 1, n + 1):
            if not ok(a, b):
                break
            cand = (r + 1, g + cut, sq + (b - a) ** 2)
            if best[b] is None or cand < best[b]:
                best[b], prev[b] = cand, a
    out, b = [], n
    while b > 0:
        out.append((prev[b], b))
        b = prev[b]
    return out[::-1]


def _bitmap_line(text: str, first_word: int, px_w: int, px_h: int) -> tuple[np.ndarray, np.ndarray]:
    bits, owner = render_text(text)
    char_word, wi = [], first_word
    for ch in text:
        char_word.append(-1 if ch == " " else wi)
        if ch == " ":
            wi += 1
    col_word = np.array([char_word[o] if o >= 0 else -1 for o in owner], int)
    sub = bits[np.arange(HEIGHT * px_h) // px_h][:, np.arange(bits.shape[1] * px_w) // px_w]
    words = np.broadcast_to(col_word[np.arange(bits.shape[1] * px_w) // px_w], sub.shape)
    return sub, np.where(sub, words, -1)


@lru_cache(maxsize=256)
def ink_mask(words: tuple[str, ...], area: tuple[int, int, int, int], canvas: tuple[int, int],
             max_px_h: int = 99, font: str = "bit5x5", glue: tuple[bool, ...] | None = None) -> InkMask | None:
    ax, ay, aw, ah = area
    W, H = canvas
    lay = layout(words, aw, ah, max_px_h, font, glue)
    if lay is None:
        return None
    used, runs, px_w, px_h = lay
    ttf = used != "bit5x5"
    ref = fonts.ref_glyph(words)
    lines = []
    for a, b in runs:
        if ttf:
            m, own = fonts.ttf_line(used, words[a:b], px_h, _sub_glue(glue, a, b), ref)
            own = np.where(own >= 0, own + a, -1)
        else:
            m, own = _bitmap_line(" ".join(words[a:b]), a, px_w, px_h)
        lines.append((m, own))
    line_h = lines[0][0].shape[0]
    gap = int(round(TTF_GAP * px_h)) if ttf else int(round(LINE_GAP * px_h))
    gap = gap if len(lines) > 1 else 0
    block_h = line_h * len(lines) + gap * (len(lines) - 1)
    top = ay + (ah - block_h) // 2

    mask = np.zeros((H, W), bool)
    word_at = np.full((H, W), -1, int)
    for li, (m, own) in enumerate(lines):
        left = ax + (aw - m.shape[1]) // 2
        y0 = top + li * (line_h + gap)
        ys_, xs_ = np.nonzero(m)
        X, Y = xs_ + left, ys_ + y0
        ok = (X >= 0) & (X < W) & (Y >= 0) & (Y < H)
        mask[Y[ok], X[ok]] = True
        word_at[Y[ok], X[ok]] = own[ys_[ok], xs_[ok]]

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    pad = np.pad(mask, 1)
    inner = pad[:-2, 1:-1] & pad[2:, 1:-1] & pad[1:-1, :-2] & pad[1:-1, 2:]
    edge = ~inner[ys, xs]
    depth = ndimage.distance_transform_cdt(mask, metric="taxicab")[ys, xs] - 1
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    order = ((ys - y0) // max(1, line_h) * (x1 - x0) + (xs - x0)) / max(1, (x1 - x0) * len(runs))
    mask.setflags(write=False)
    word = word_at[ys, xs]
    word = np.where(word < 0, 0, word)
    return InkMask(xs=xs, ys=ys, word=word, edge=edge, depth=depth, order=np.clip(order, 0, 1), mask=mask,
                   bbox=(int(x0), int(y0), int(x1), int(y1)), px_h=float(px_h if not ttf else px_h / HEIGHT),
                   font=used, cjk=has_cjk("".join(words)))
