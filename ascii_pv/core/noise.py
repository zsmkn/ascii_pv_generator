"""Deterministic value noise (vectorised), for flames, plasma and other organic textures."""

from __future__ import annotations

import numpy as np


def _lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    v = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263 + seed * 2147483647) & 0xFFFFFFFF
    v = (v ^ (v >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((v ^ (v >> 16)) & 0xFFFF) / 65535.0


def value(x: np.ndarray, y: np.ndarray, seed: int = 0) -> np.ndarray:
    """Smooth noise in 0..1 at float coordinates (lattice spacing 1)."""
    x0, y0 = np.floor(x), np.floor(y)
    fx, fy = x - x0, y - y0
    sx, sy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)
    a = _lattice(x0, y0, seed)
    b = _lattice(x0 + 1, y0, seed)
    c = _lattice(x0, y0 + 1, seed)
    d = _lattice(x0 + 1, y0 + 1, seed)
    return (a + (b - a) * sx) + ((c + (d - c) * sx) - (a + (b - a) * sx)) * sy


def fbm(x: np.ndarray, y: np.ndarray, octaves: int = 4, seed: int = 0) -> np.ndarray:
    """Fractal sum of `octaves` noise layers, normalised to 0..1."""
    total, amp, norm = np.zeros_like(x, dtype=float), 1.0, 0.0
    for o in range(octaves):
        total += amp * value(x * (2 ** o), y * (2 ** o), seed + o * 101)
        norm += amp
        amp *= 0.5
    return total / norm
