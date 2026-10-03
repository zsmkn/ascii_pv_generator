"""Frame composition, shared by the terminal player and the video renderer.

Per frame: background template -> shot FX (glitch, scan, ...; M10 drop shockwave) -> HUD panels
-> transition
-> (M6) character sprite -> lyric typography.
"""

from __future__ import annotations

import bisect

import numpy as np

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, make_ctx, str_seed
from ascii_pv.core.palette import get_palette
from ascii_pv.core.storyboard import Shot, Storyboard
from ascii_pv.fx import post, transitions
from ascii_pv.templates import REGISTRY, create
from ascii_pv.templates.base import Template
from ascii_pv.templates.hud import draw_hud
from ascii_pv.typography.layer import LyricLayer


DROP_FLASH = 0.6     # seconds a drop's shockwave takes to cross the screen


class Renderer:
    def __init__(self, analysis: Analysis, board: Storyboard, assets: dict | None = None):
        self.analysis = analysis
        self.board = board
        self.assets = {**board.assets, **(assets or {})}
        self.palette = get_palette(board.palette)
        self._templates: dict[str, Template] = {}
        self.lyrics = LyricLayer(analysis, board)
        self._starts = [s.start for s in board.shots]
        # beat index of the first downbeat, so that bar phase = beats since the last downbeat / 4
        db = analysis.downbeats
        self._db_beat = bisect.bisect_left(analysis.beats, db[0] - 1e-3) if db and analysis.beats else 0

    def _template(self, shot: Shot) -> Template:
        tpl = self._templates.get(shot.id)
        if tpl is None:
            tpl = self._templates[shot.id] = create(shot, self.analysis, self.assets)
        return tpl

    def _background(self, shot: Shot, t: float, w: int, h: int, mask: np.ndarray | None) -> tuple[Canvas, FrameCtx]:
        pal = self.palette
        c = Canvas(w, h, bg=pal.bg, fg=pal.ui)
        ctx = make_ctx(t, self.analysis, shot, pal, self.board.seed, self.board.fps)
        ctx.lyric_mask = mask
        self._template(shot).draw(c, ctx)
        c.clip = (0, 0, w, h)
        fx = shot.fx
        hit = ctx.kick * ctx.pulse(0.15)
        post.zoom(c, float(fx.get("zoom", 0.0)), hit)
        post.shake(c, float(fx.get("shake", 0.0)), hit, ctx.frame)
        post.split(c, float(fx.get("split", 0.0)), hit, pal)
        post.noise(c, float(fx.get("noise", 0.0)), ctx.frame, pal)
        post.glitch(c, float(fx.get("glitch", 0.0)), ctx.frame, hit, pal, str_seed(shot.id) & 0xFFFF)
        post.scan(c, t, float(fx.get("scan", 0.0)), pal)
        post.strobe(c, float(fx.get("strobe", 0.0)), hit, pal)
        self._shockwave(c, ctx, float(fx.get("shockwave", fx.get("build", 0.0))))   # "build": older boards
        if shot.hud:                       # after the FX: the readouts stay legible however hard the picture shakes
            meta = REGISTRY[shot.template].meta if shot.template in REGISTRY else None
            draw_hud(c, ctx, self.analysis, shot.hud, meta.hud if meta else "corners")
        return c, ctx

    def _shockwave(self, c: Canvas, ctx: FrameCtx, amount: float) -> None:
        """M10: a drop sends a ring of block glyphs racing out from the centre (and one hard jolt)."""
        d = self.analysis.since("drop", ctx.t)
        if amount > 0.01 and d < DROP_FLASH:
            k = amount * (1 - d / DROP_FLASH)
            post.shockwave(c, k, self.palette)
            post.shake(c, 1.0, k, ctx.frame)

    def _sprite(self, c: Canvas, ctx: FrameCtx) -> None:
        """The character over the picture during the sections listed in overlays.sprite."""
        sp = self.board.overlays.get("sprite") or {}
        sec = ctx.section
        if not sp.get("enabled") or sec is None or sec.label not in sp.get("sections", ["chorus"]):
            return
        if ctx.shot.template == "image_ascii":            # she is already the picture
            return
        crops = self.assets.get(sp.get("asset", "character")) or {}
        path = crops.get(sp.get("crop", "bust")) or crops.get("full")
        if not path:
            return
        from ascii_pv.templates.image_ascii import draw_picture

        params = {"mode": "halfblock", "color": "source", "align": "right", "scale": 0.9, "reveal": "scan",
                  **sp}
        # sides alternate between repeats of the section so the figure does not sit still all song
        if sp.get("align") == "alternate":
            idx = sum(1 for s in self.analysis.sections if s.label == sec.label and s.start < sec.start)
            params["align"] = "right" if idx % 2 == 0 else "left"
        draw_picture(c, ctx, path, params, ctx.t - sec.start, str_seed(sec.group or "s") & 0xFFFF,
                     float(sp.get("dim", 0.0)))

    def frame(self, t: float, w: int, h: int, lyric_offset: float = 0.0) -> Canvas:
        """lyric_offset > 0 shows the lyrics earlier (the player's [ / ] keys)."""
        if not self.board.shots:
            return Canvas(w, h, bg=self.palette.bg, fg=self.palette.ui)
        i = max(0, bisect.bisect_right(self._starts, t) - 1)
        shot = self.board.shots[i]
        lt = t + self.board.lyric_offset + lyric_offset
        mask = self.lyrics.mask_at(lt, w, h)
        c, ctx = self._background(shot, t, w, h, mask)

        kind = shot.transition_in
        dur = float(shot.fx.get("transition_s", transitions.DURATION.get(kind, 0.0)))
        if dur > 0 and t - shot.start < dur:
            prev = self._background(self.board.shots[i - 1], t, w, h, mask)[0] if i > 0 else None
            transitions.apply(kind, prev, c, (t - shot.start) / dur, self.palette, str_seed(shot.id) & 0xFFFF)

        self._sprite(c, ctx)
        bar_phase = (((ctx.beat_idx - self._db_beat) % 4) + ctx.beat_phase) / 4
        self.lyrics.draw(c, lt, self.palette, ctx.beat_phase, bar_phase, ctx.kick, ctx.energy)
        if self.board.overlays.get("subtitle"):
            self._subtitle(c, lt)
        return c

    def _subtitle(self, c: Canvas, lt: float) -> None:
        """Plain one-line caption on the bottom row (debugging / accessibility; off by default)."""
        i = self.analysis.line_at(lt)
        if i < 0:
            return
        from ascii_pv.core.canvas import crop, text_width

        text = crop(self.analysis.lyrics[i].text, c.w - 4)
        c.fill(0, c.h - 1, c.w, 1, " ", bg=self.palette.bg)
        c.put((c.w - text_width(text)) // 2, c.h - 1, text, self.palette.ui_at(0.85))
