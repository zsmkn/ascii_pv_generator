"""Pixel post-processing for the video (the terminal cannot do these): CRT phosphor look.

bloom     bright glyphs glow: threshold, blur at quarter resolution, add back
trail     phosphor persistence: the previous frame lingers, decaying (needs the previous output frame)
scan      every other pixel row slightly darker
vignette  corners fall off
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageFilter


@dataclass
class PostFX:
    bloom: float = 0.45
    trail: float = 0.35
    scan: float = 0.10
    vignette: float = 0.22

    def apply(self, px: np.ndarray, prev: np.ndarray | None) -> np.ndarray:
        f = px.astype(np.float32)
        if self.trail > 0 and prev is not None:
            f = np.maximum(f, prev.astype(np.float32) * self.trail)
        if self.bloom > 0:
            f += self.bloom * _bloom(f)
        if self.scan > 0 or self.vignette > 0:
            f *= _shade(px.shape[0], px.shape[1], self.scan, self.vignette)[..., None]
        return np.clip(f, 0, 255).astype(np.uint8)


def _bloom(f: np.ndarray) -> np.ndarray:
    h, w, _ = f.shape
    h4, w4 = h // 4, w // 4
    small = f[:h4 * 4, :w4 * 4].reshape(h4, 4, w4, 4, 3).mean(axis=(1, 3))
    lum = small.max(axis=2, keepdims=True)
    bright = small * np.clip((lum - 110) / 145, 0, 1)
    img = Image.fromarray(bright.astype(np.uint8), "RGB").filter(ImageFilter.GaussianBlur(3))
    glow = np.asarray(img, np.float32)
    glow = np.repeat(np.repeat(glow, 4, axis=0), 4, axis=1)
    out = np.zeros_like(f)
    out[:h4 * 4, :w4 * 4] = glow
    return out


@lru_cache(8)
def _shade(h: int, w: int, scan: float, vignette: float) -> np.ndarray:
    y = np.linspace(-1, 1, h)[:, None]
    x = np.linspace(-1, 1, w)[None, :]
    m = 1 - vignette * np.clip((x ** 2 + y ** 2) / 2, 0, 1) ** 1.5
    rows = np.ones((h, 1))
    rows[1::2] = 1 - scan
    return (m * rows).astype(np.float32)
