"""image_ascii: the cover / character picture as characters, moving with the music.

params
  asset    key in storyboard.assets ("character", "cover")
  crop     full / bust / face
  mode     halfblock / ramp / glyph / edge          (see ascii_pv/imaging.py)
  color    source / palette
  align    center / left / right
  scale    share of the canvas height the picture may use
  reveal   scan (rows appear top to bottom over the first bar) / dissolve / none
  react    how strongly it moves with the music (0 = still)
"""

from __future__ import annotations

import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.imaging import _load, blit, cell_image, fit_cells, react
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.typography.layer import _u16


def place(c: Canvas, cols: int, rows: int, align: str) -> tuple[int, int]:
    y = (c.h - rows) // 2
    if align == "left":
        return max(1, c.w // 20), y
    if align == "right":
        return c.w - cols - max(1, c.w // 20), y
    return (c.w - cols) // 2, y


def reveal_mask(kind: str, h: int, w: int, u: float, seed: int) -> np.ndarray | None:
    if kind == "none" or u >= 1:
        return None
    if kind == "dissolve":
        yy, xx = np.mgrid[0:h, 0:w]
        return _u16(yy * 7919 + xx, seed) / 65535.0 < u
    rows = int(u * (h + 1))                   # scan
    m = np.zeros((h, w), bool)
    m[:rows] = True
    return m


def draw_picture(c: Canvas, ctx: FrameCtx, path: str, p: dict, age: float, seed: int, dim: float = 0.0) -> None:
    src = _load(path)
    scale = float(p.get("scale", 0.95))
    cols, rows = fit_cells(src.width, src.height, int(c.w * 0.9), max(4, int(c.h * scale)))
    ci = cell_image(path, cols, rows, str(p.get("mode", "halfblock")), str(p.get("color", "source")), ctx.palette)
    moved, margin = react(ci, ctx.t, ctx.bands, ctx.kick * ctx.pulse(0.15), ctx.feat.flux, ctx.energy, ctx.bpm,
                          float(p.get("react", 1.0)), seed)
    x, y = place(c, cols, rows, str(p.get("align", "center")))
    bar = 240.0 / max(40.0, ctx.bpm)
    mask = reveal_mask(str(p.get("reveal", "scan")), moved.h, moved.w, age / bar, seed)
    blit(c, moved, x - margin, y, mask, dim)


@register
class ImageAscii(Template):
    meta = TemplateMeta(
        "image_ascii", "The cover / character picture in characters (halfblock / ramp / glyph / edge), dancing.",
        energy=(0.0, 1.0), sections=frozenset({"intro", "outro", "bridge", "instrumental", "verse"}),
        needs_image=True, lyric_blends=frozenset({"overlay"}), busy=0.5, 
    )
    defaults = {"asset": "character", "crop": "full", "mode": "halfblock", "color": "source", "align": "center",
                "scale": 0.95, "reveal": "scan", "react": 1.0, "backdrop": True}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        p = self.params
        crops = self.assets.get(str(p["asset"])) or {}
        path = crops.get(str(p["crop"])) or crops.get("full")
        seed = str_seed(self.shot.id) & 0xFFFF
        if p.get("backdrop", True):
            # a faint dot grid gives the figure a floor to stand on
            ys, xs = np.mgrid[1:c.h:2, 2:c.w:4]
            c.ch[ys, xs] = "."
            c.fg[ys, xs] = ctx.palette.ui_at(0.12 + 0.1 * ctx.pulse(0.2))
        if path:
            draw_picture(c, ctx, path, p, ctx.local_t, seed)
