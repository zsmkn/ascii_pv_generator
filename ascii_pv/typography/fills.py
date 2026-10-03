"""What glyph each resting ink cell shows.

  binary   0 / 1                     hex      0-9 A-F
  self     the letters of the word itself, cycling (LOVE is made of L, O, V, E)
  ramp     by depth into the stroke: edge '@', then '#', '*', '+'
  block    by depth: '█' edge, '▓', '▒' inside
  outline  edge cells only ('#'); the inside stays empty
  solid    '#'
Digits and letters only read as a texture when strokes are at least ~3 cells thick; thinner letters get solid
fills (binary/hex/self fall back to '#'). Chinese / Japanese glyphs have many thin strokes: a texture would
blur them, so they need most of their cells inside a stroke (only a huge single character gets there).
"""

from __future__ import annotations

import re

import numpy as np

from ascii_pv.typography.inkmask import InkMask

FILLS = {"binary": np.array(list("01")), "hex": np.array(list("0123456789ABCDEF"))}
RAMP = np.array(list("@#*+"))
BLOCK = np.array(list("█▓▒"))
TEXTURED = {"binary", "hex", "self"}
NAMES = ("binary", "hex", "self", "ramp", "block", "outline", "solid")


CJK_INTERIOR = 0.4        # share of a CJK mask's cells that must lie inside a stroke for a texture to read


def thick(ink: InkMask) -> bool:
    """Strokes are thick enough for a texture to read."""
    if ink.cjk:
        return float((ink.depth >= 1).mean()) >= CJK_INTERIOR
    return ink.px_h >= 2 or (ink.font != "bit5x5" and ink.depth.max() >= 2)


def resting(fill: str, ink: InkMask, words: tuple[str, ...], k: np.ndarray) -> np.ndarray:
    if fill in TEXTURED and not thick(ink):
        fill = "solid"
    if fill in FILLS:
        chars = FILLS[fill]
        return chars[k % len(chars)]
    if fill == "self":
        letters = [re.sub(r"[^A-Z0-9]", "", w.upper()) or "#" for w in words]
        return np.array([letters[w][kk % len(letters[w])] for w, kk in zip(ink.word, k)])
    if fill == "ramp":
        return RAMP[np.minimum(ink.depth, len(RAMP) - 1)]
    if fill == "block":
        return BLOCK[np.minimum(ink.depth, len(BLOCK) - 1)]
    if fill == "outline":
        return np.where(ink.edge, "#", " ")
    return np.full(ink.n, "#")


def flickers(fill: str, ink: InkMask) -> bool:
    """Resting bits flip now and then (binary / hex only)."""
    return fill in FILLS and thick(ink)
