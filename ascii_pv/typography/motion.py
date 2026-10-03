"""How lyric particles arrive and leave. Every function is vectorised over the ink cells and a pure function of
time, so any frame can be drawn directly.

enter(kind, ...)  u = arrival progress per cell (1 = at rest)
  assemble   fly in from a random spot along a swirl (ref A's title)
  decode     appear in place left to right, scrambling briefly
  rain       fall from above the screen into place, column by column
  scan       a bright front sweeps left to right, leaving the letters behind it
  glitch_in  cells blink in at random with a sideways jitter that settles
leave(kind, ...)  v = leaving progress 0..1 for the whole piece
  explode    spin outward and thin out
  fade       drop out in random order, in place
  melt       drip down, each column at its own speed
  shatter    break into 4x2 tiles that fly apart and fall
  crt_off    collapse to a horizontal line, then to a dot
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

ENTERS = ("assemble", "decode", "rain", "scan", "glitch_in")
EXITS = ("explode", "fade", "melt", "shatter", "crt_off")
SHARDS = np.array(list("▓▒░"))


@dataclass
class Frame:
    x: np.ndarray
    y: np.ndarray
    visible: np.ndarray
    u: np.ndarray                  # arrival progress; >= 0.98 counts as landed
    level: np.ndarray              # brightness multiplier
    glyph: np.ndarray | None = None  # per-cell glyph override ("" or None = keep the cell's own glyph)

    @property
    def landed(self) -> np.ndarray:
        return self.u >= 0.98


def enter(kind: str, tx: np.ndarray, ty: np.ndarray, order: np.ndarray, delay: np.ndarray, ox: np.ndarray,
          oy: np.ndarray, k: np.ndarray, age, dur: float, W: int, H: int, bbox) -> Frame:
    n = len(tx)
    ones = np.ones(n)
    age = np.broadcast_to(np.asarray(age, float), (n,))
    dur = max(1e-3, dur)
    if kind == "decode":
        appear = age / (dur * 0.6) - order
        u = np.clip(appear / 0.4 + delay * 0.2, 0, 1)
        return Frame(tx.copy(), ty.copy(), appear >= 0, u, ones.copy())
    if kind == "rain":
        x0, _, x1, _ = bbox
        col = (tx - x0) / max(1, x1 - x0)
        u = np.clip((age - (col * 0.5 + delay * 0.15) * dur) / (dur * 0.5), 0, 1)
        fall = (1 - u ** 2) * (ty + 3 + (k % 7))           # accelerate into place from above the screen
        return Frame(tx.copy(), ty - fall, age >= 0, u, ones.copy())
    if kind == "scan":
        front = age / (dur * 0.85)
        u = np.clip((front - order) / 0.08, 0, 1)
        lit = (u > 0) & (u < 1)
        level = np.where(lit, 1.4, 1.0)
        return Frame(tx.copy(), ty.copy(), u > 0, u, level, np.where(lit, "#", "") if lit.any() else None)
    if kind == "glitch_in":
        start = (k % 100) / 100 * 0.7
        u = np.clip((age / dur - start) / 0.3, 0, 1)
        flicker = ((k + (age * 30).astype(int)) % 3 == 0) & (u < 1)
        jitter = ((k % 9) - 4) * 3 * (1 - u)
        return Frame(tx + jitter, ty.copy(), (u > 0) & ~flicker, u, ones.copy())
    # assemble
    u = np.clip((age - delay * dur * 0.45) / (dur * 0.55), 0, 1)
    e = 1 - (1 - u) ** 3
    swirl = np.sin(u * math.pi) * (1 - u)
    ph = age * 2 + np.arange(n) * 0.7
    x = ox + (tx - ox) * e + np.sin(ph) * swirl * W * 0.18
    y = oy + (ty - oy) * e + np.cos(ph) * swirl * H * 0.18
    return Frame(x, y, age >= 0, u, ones.copy())


def leave(kind: str, f: Frame, tx: np.ndarray, ty: np.ndarray, k: np.ndarray, v: float, W: int, H: int,
          bbox) -> Frame:
    if v <= 0:
        return f
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    rest = np.zeros(len(tx))                       # leaving cells no longer count as landed
    if kind == "fade":
        return Frame(f.x, f.y, f.visible & ((k % 100) / 100 >= v), f.u, f.level * (1 - 0.6 * v), f.glyph)
    if kind == "melt":
        speed = 0.5 + (((tx * 7) % 11) / 11)       # per column
        drop = v ** 2 * H * 0.7 * speed
        glyph = np.where(drop > 1, np.where(k % 2 == 0, "|", ":"), "")
        return Frame(f.x, ty + drop, f.visible & ((k % 100) / 100 >= v * 0.8), f.u,
                     f.level * (1 - 0.5 * v), glyph)
    if kind == "shatter":
        tile = (tx // 4).astype(int) * 131 + (ty // 2).astype(int) * 17
        ang = (tile * 2654435761 % 360) * math.pi / 180
        spd = 0.5 + (tile * 40503 % 100) / 100
        x = tx + np.cos(ang) * spd * v * W * 0.35
        y = ty + np.sin(ang) * spd * v * H * 0.25 + v ** 2 * H * 0.35
        glyph = np.full(len(tx), SHARDS[min(2, int(v * 3))])
        return Frame(x, y, f.visible & ((k % 100) / 100 >= v * 0.85), rest, f.level, glyph)
    if kind == "crt_off":
        if v < 0.6:
            s = 1 - v / 0.6
            y = cy + (ty - cy) * s
            x = tx
            glyph = np.where(s < 0.35, "-", "")
        else:
            s = 1 - (v - 0.6) / 0.4
            x = cx + (tx - cx) * s
            y = np.full(len(tx), cy)
            glyph = np.full(len(tx), "-" if s > 0.1 else "*")
        return Frame(x, y, f.visible & (v < 0.98), rest, f.level * (1 + 0.6 * v), glyph)
    # explode
    ang = np.arctan2((ty - cy) * 2, tx - cx) + v * 0.75
    dist = np.hypot(tx - cx, (ty - cy) * 2) + v * (W * 0.25 + k % 40)
    return Frame(cx + np.cos(ang) * dist, cy + np.sin(ang) * dist / 2, f.visible & ((k % 100) / 100 >= v * 0.9),
                 rest, f.level)
