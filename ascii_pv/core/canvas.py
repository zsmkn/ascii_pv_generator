"""Cell grid shared by every backend.

A cell is (char, fg RGB, bg RGB, bold). Storage is numpy so that full-screen templates can work
on whole arrays; `put` and friends are the convenient per-string path.

Wide (East Asian W/F) characters take two cells: the head holds the character and the cell to its
right holds '' (a continuation). Backends skip continuations.
"""

from __future__ import annotations

import unicodedata
from functools import lru_cache

import numpy as np

RGB = tuple[int, int, int]


@lru_cache(maxsize=4096)
def char_width(ch: str) -> int:
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def text_width(s: str) -> int:
    return sum(char_width(c) for c in s)


def crop(s: str, n: int) -> str:
    """Longest prefix of s that fits in n cells."""
    out, used = [], 0
    for ch in s:
        k = char_width(ch)
        if used + k > n:
            break
        out.append(ch)
        used += k
    return "".join(out)


class Canvas:
    def __init__(self, w: int, h: int, bg: RGB = (0, 0, 0), fg: RGB = (200, 200, 200)):
        self.w, self.h = w, h
        self.default_bg, self.default_fg = bg, fg
        self.ch = np.full((h, w), " ", dtype="<U1")
        self.fg = np.empty((h, w, 3), np.uint8)
        self.bg = np.empty((h, w, 3), np.uint8)
        self.bold = np.zeros((h, w), bool)
        self.fg[:] = fg
        self.bg[:] = bg
        self.clip: tuple[int, int, int, int] = (0, 0, w, h)  # x0, y0, x1, y1 (exclusive)

    # ------------------------------------------------------------ cells

    def inside(self, x: int, y: int) -> bool:
        x0, y0, x1, y1 = self.clip
        return x0 <= x < x1 and y0 <= y < y1

    def _set(self, x: int, y: int, ch: str, fg: RGB | None, bg: RGB | None, bold: bool) -> None:
        # Writing over half of a wide character blanks the other half. (A continuation written
        # right after its own head needs no fix-up.)
        if ch != "":
            if self.ch[y, x] == "" and x > 0:
                self.ch[y, x - 1] = " "
            elif x + 1 < self.w and self.ch[y, x + 1] == "" and char_width(str(self.ch[y, x]) or " ") == 2:
                self.ch[y, x + 1] = " "
        self.ch[y, x] = ch
        if fg is not None:
            self.fg[y, x] = fg
        if bg is not None:
            self.bg[y, x] = bg
        self.bold[y, x] = bold

    def put(self, x: float, y: float, s: str, fg: RGB | None = None, bg: RGB | None = None,
            bold: bool = False) -> None:
        """Write a string starting at (x, y). Parts outside the clip rect are dropped."""
        x, y = int(round(x)), int(round(y))
        x0, y0, x1, y1 = self.clip
        if not y0 <= y < y1:
            return
        for ch in str(s):
            k = char_width(ch)
            if k == 0:
                continue
            if x >= x0 and x + k <= x1:
                self._set(x, y, ch, fg, bg, bold)
                if k == 2:
                    self._set(x + 1, y, "", fg, bg, bold)
            x += k
            if x >= x1:
                break

    def get(self, x: int, y: int) -> str:
        return str(self.ch[y, x]) if 0 <= x < self.w and 0 <= y < self.h else " "

    def center(self, y: float, s: str, fg: RGB | None = None, bg: RGB | None = None, bold: bool = False) -> None:
        self.put((self.w - text_width(s)) // 2, y, s, fg, bg, bold)

    def fill(self, x: int, y: int, w: int, h: int, ch: str = " ", fg: RGB | None = None,
             bg: RGB | None = None) -> None:
        cx0, cy0, cx1, cy1 = self.clip
        xa, ya = max(cx0, int(x)), max(cy0, int(y))
        xb, yb = min(cx1, int(x + w)), min(cy1, int(y + h))
        if xa >= xb or ya >= yb:
            return
        self.ch[ya:yb, xa:xb] = ch
        if fg is not None:
            self.fg[ya:yb, xa:xb] = fg
        if bg is not None:
            self.bg[ya:yb, xa:xb] = bg
        self.bold[ya:yb, xa:xb] = False

    def clear(self) -> None:
        self.ch[:] = " "
        self.fg[:] = self.default_fg
        self.bg[:] = self.default_bg
        self.bold[:] = False

    def line(self, x0: float, y0: float, x1: float, y1: float, ch: str = ".", fg: RGB | None = None) -> None:
        steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) * 1.5))
        for i in range(steps + 1):
            u = i / steps
            self.put(x0 + (x1 - x0) * u, y0 + (y1 - y0) * u, ch, fg)

    def box(self, x: int, y: int, w: int, h: int, fg: RGB | None = None, title: str = "") -> None:
        if w < 2 or h < 2:
            return
        self.put(x, y, "+" + "-" * (w - 2) + "+", fg)
        self.put(x, y + h - 1, "+" + "-" * (w - 2) + "+", fg)
        for yy in range(y + 1, y + h - 1):
            self.put(x, yy, "|", fg)
            self.put(x + w - 1, yy, "|", fg)
        if title:
            self.put(x + 2, y, f" {crop(title, w - 6)} ", fg)

    def dim(self, factor: float, mask: np.ndarray | None = None) -> None:
        """Fade cells toward the canvas background (factor 0 = unchanged, 1 = invisible). Painted cell
        backgrounds (half-block pictures) fade too."""
        f = min(1.0, max(0.0, factor))
        base = np.array(self.default_bg, float)
        bg = (self.bg * (1 - f) + base * f).astype(np.uint8)
        fg = (self.fg * (1 - f) + base * f).astype(np.uint8)
        if mask is None:
            self.fg[:], self.bg[:] = fg, bg
        else:
            self.fg[mask], self.bg[mask] = fg[mask], bg[mask]

    def blit(self, other: "Canvas", x: int = 0, y: int = 0, transparent: bool = True) -> None:
        """Copy another canvas onto this one. With transparent=True blank cells are skipped."""
        for yy in range(other.h):
            ty = y + yy
            if not 0 <= ty < self.h:
                continue
            for xx in range(other.w):
                tx = x + xx
                if not self.inside(tx, ty):
                    continue
                ch = str(other.ch[yy, xx])
                if transparent and ch == " ":
                    continue
                self.ch[ty, tx] = ch
                self.fg[ty, tx] = other.fg[yy, xx]
                self.bg[ty, tx] = other.bg[yy, xx]
                self.bold[ty, tx] = other.bold[yy, xx]

    # ------------------------------------------------------------ output

    def plain(self) -> str:
        return "\n".join("".join(str(c) for c in row) for row in self.ch)

    def ansi(self) -> str:
        """Full-frame 24-bit ANSI. (The terminal backend does its own diffed output.)"""
        out = ["\x1b[H"]
        chars = self.ch.tolist()
        fg = self.fg.tolist()
        bg = self.bg.tolist()
        bold = self.bold.tolist()
        last = None
        for y in range(self.h):
            out.append(f"\x1b[{y + 1};1H")
            for x in range(self.w):
                ch = chars[y][x]
                if ch == "":
                    continue
                style = (tuple(fg[y][x]), tuple(bg[y][x]), bold[y][x])
                if style != last:
                    (fr, fg_, fb), (br, bg_, bb), b = style
                    out.append(f"\x1b[{'1' if b else '0'};38;2;{fr};{fg_};{fb};48;2;{br};{bg_};{bb}m")
                    last = style
                out.append(ch)
        out.append("\x1b[0m")
        return "".join(out)
