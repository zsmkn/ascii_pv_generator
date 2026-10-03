"""bit5x5: a 5-row bitmap font, variable width (letters 5, punctuation narrower).

Lowercase maps to uppercase. Unknown characters draw as a small box.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

HEIGHT = 5
GAP = 1  # blank columns between glyphs

_SRC = {
    "A": ".###. #...# ##### #...# #...#",
    "B": "####. #...# ####. #...# ####.",
    "C": ".#### #.... #.... #.... .####",
    "D": "####. #...# #...# #...# ####.",
    "E": "##### #.... ####. #.... #####",
    "F": "##### #.... ####. #.... #....",
    "G": ".#### #.... #..## #...# .###.",
    "H": "#...# #...# ##### #...# #...#",
    "I": "##### ..#.. ..#.. ..#.. #####",
    "J": "..### ...#. ...#. #..#. .##..",
    "K": "#...# #..#. ###.. #..#. #...#",
    "L": "#.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #...# #...#",
    "N": "#...# ##..# #.#.# #..## #...#",
    "O": ".###. #...# #...# #...# .###.",
    "P": "####. #...# ####. #.... #....",
    "Q": ".###. #...# #.#.# #..#. .##.#",
    "R": "####. #...# ####. #..#. #...#",
    "S": ".#### #.... .###. ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #.#.# ##.## #...#",
    "X": "#...# .#.#. ..#.. .#.#. #...#",
    "Y": "#...# .#.#. ..#.. ..#.. ..#..",
    "Z": "##### ...#. ..#.. .#... #####",
    "0": ".###. #..## #.#.# ##..# .###.",
    "1": "..#.. .##.. ..#.. ..#.. .###.",
    "2": "####. ....# .###. #.... #####",
    "3": "####. ....# .###. ....# ####.",
    "4": "#..#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ####.",
    "6": ".###. #.... ####. #...# .###.",
    "7": "##### ...#. ..#.. .#... .#...",
    "8": ".###. #...# .###. #...# .###.",
    "9": ".###. #...# .#### ....# .###.",
    ".": ". . . . #",
    ",": ".. .. .. .# #.",
    "'": "# # . . .",
    '"': "#.# #.# ... ... ...",
    "!": "# # # . #",
    "?": "####. ....# ..##. ..... ..#..",
    "-": "... ... ### ... ...",
    ":": ". # . # .",
    ";": ".. .# .. .# #.",
    "(": ".# #. #. #. .#",
    ")": "#. .# .# .# #.",
    "&": ".##.. #..#. .##.# #..#. .##.#",
    "/": "..# ..# .#. #.. #..",
    "*": "..... #.#.# .###. #.#.# .....",
    "+": "..... ..#.. .###. ..#.. .....",
    "=": "... ### ... ### ...",
    "#": ".#.#. ##### .#.#. ##### .#.#.",
    " ": "... ... ... ... ...",
}
_UNKNOWN = "### #.# #.# #.# ###"


@lru_cache(None)
def glyph(ch: str) -> np.ndarray:
    """(5, width) bool array."""
    src = _SRC.get(ch.upper(), _SRC.get(ch, _UNKNOWN))
    rows = src.split(" ")
    assert len(rows) == HEIGHT, ch
    return np.array([[c == "#" for c in r] for r in rows], dtype=bool)


def render_text(text: str) -> tuple[np.ndarray, np.ndarray]:
    """Bitmap of one line + for each column the index of the character it belongs to (-1 in gaps)."""
    cols, owner = [], []
    for i, ch in enumerate(text):
        g = glyph(ch)
        if cols:
            cols.append(np.zeros((HEIGHT, GAP), bool))
            owner.extend([-1] * GAP)
        cols.append(g)
        owner.extend([i] * g.shape[1])
    if not cols:
        return np.zeros((HEIGHT, 0), bool), np.zeros(0, int)
    return np.hstack(cols), np.array(owner, int)


def text_bits_width(text: str) -> int:
    return sum(glyph(c).shape[1] for c in text) + GAP * max(0, len(text) - 1)
