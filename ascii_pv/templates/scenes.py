"""Motif scenes and machines: life_grid, heart, clock, skyline, boot_log."""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core.canvas import Canvas, crop
from ascii_pv.core.context import FrameCtx, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.templates.waves import _paint
from ascii_pv.typography.layer import _u16

GLIDER = np.array([[0, 1, 0], [0, 0, 1], [1, 1, 1]], bool)


@register
class LifeGrid(Template):
    """Conway's Game of Life, one generation per beat; a glider is dropped in on every downbeat.

    Generations are stepped forward from the shot start, so a frame is still a pure function of t: the template
    caches the last state and only recomputes from scratch when time goes backwards (seek)."""
    meta = TemplateMeta(
        "life_grid", "Game of Life, one generation per beat; gliders dropped in on the downbeats.",
        busy=0.6, energy=(0.2, 0.75), sections=frozenset({"intro", "verse", "bridge", "instrumental"}),
        tags=frozenset({"calm", "mid-energy", "dark", "sparse", "neutral"}), motifs=frozenset({"dream", "time", "pulse"}),
        lyric_blends=frozenset({"overlay", "obstacle", "knockout"}), hud="corners", hud_topics=('code:life', 'data', 'log', 'hex'),
    )
    defaults = {"density": 0.28}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._cache: tuple | None = None       # (w, h, generation, grid)

    def _seed_grid(self, w: int, h: int) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w]
        return _u16(yy * 7919 + xx, str_seed(self.shot.id) & 0xFFFF) / 65535 < float(self.params["density"])

    @staticmethod
    def _step(g: np.ndarray) -> np.ndarray:
        n = sum(np.roll(np.roll(g, dy, 0), dx, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dy or dx)
        return (n == 3) | (g & (n == 2))

    def _state(self, w: int, h: int, gen: int) -> np.ndarray:
        if self._cache and self._cache[:2] == (w, h) and self._cache[2] <= gen:
            _, _, g0, grid = self._cache
        else:
            g0, grid = 0, self._seed_grid(w, h)
        a = self.analysis
        first = a.beat_info(self.shot.start)[0]
        for g in range(g0, gen):
            grid = self._step(grid)
            b = first + g + 1
            if b % 4 == 0:                      # a downbeat: drop a glider somewhere
                x = int(_u16(np.array([b]), 3)[0]) % max(1, w - 3)
                y = int(_u16(np.array([b]), 4)[0]) % max(1, h - 3)
                grid[y:y + 3, x:x + 3] |= np.rot90(GLIDER, b % 4)
        self._cache = (w, h, gen, grid)
        return grid

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        first = self.analysis.beat_info(self.shot.start)[0]
        gen = max(0, ctx.beat_idx - first)
        g = self._state(c.w, c.h, gen)
        if ctx.lyric_mask is not None:
            g = g & ~ctx.lyric_mask                 # cells die where the lettering is
        ys, xs = np.nonzero(g)
        born = ctx.beat_phase < 0.25
        lev = np.full(len(ys), 0.45 + 0.4 * ctx.energy + (0.25 * ctx.kick if born else 0))
        _paint(c, ys, xs, np.full(len(ys), "o"), lev, ctx)


@register
class Heart(Template):
    meta = TemplateMeta(
        "heart", "A heart that beats with the song, particles circling it.",
        busy=0.4, energy=(0.1, 0.8), sections=frozenset({"verse", "chorus", "bridge", "outro"}),
        tags=frozenset({"bright", "calm", "mid-energy"}), motifs=frozenset({"love"}),
        lyric_blends=frozenset({"overlay", "knockout"}), 
    )
    defaults = {"size": 0.8}

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        beat = math.exp(-ctx.beat_phase * 5)
        s = float(self.params["size"]) * (0.9 + 0.12 * beat + 0.08 * ctx.kick)
        yy, xx = np.mgrid[0:c.h, 0:c.w].astype(float)
        R = min(c.w / 2, c.h) * 0.42 * s
        x = (xx - c.w / 2) / (R * 2) * 1.3          # cells are 1:2
        y = -(yy - c.h * 0.48) / R * 1.3
        f = (x * x + y * y - 1) ** 3 - x * x * y ** 3  # the heart curve: inside where f < 0
        inside = f < 0
        depth = np.clip(-f * 6, 0, 1)
        chars = np.where(depth > 0.6, "@", np.where(depth > 0.3, "#", np.where(depth > 0.08, "*", "+")))
        ys, xs = np.nonzero(inside)
        lev = 0.35 + 0.5 * depth[ys, xs] + 0.25 * beat
        pal = ctx.palette
        bg = np.array(pal.bg, float)
        col = np.array(pal.err, float) * 0.6 + np.array(pal.accent, float) * 0.4
        c.ch[ys, xs] = chars[ys, xs]
        c.fg[ys, xs] = (bg + (col - bg) * np.clip(lev, 0, 1)[:, None]).astype(np.uint8)
        # orbiting sparks
        n = 40
        a = np.arange(n) / n * math.tau + ctx.local_t * 0.8
        rr = R * (1.35 + 0.15 * np.sin(np.arange(n) * 1.7 + ctx.t * 2))
        px = (c.w / 2 + np.cos(a) * rr * 2).astype(int)
        py = (c.h * 0.48 + np.sin(a) * rr).astype(int)
        ok = (px >= 0) & (px < c.w) & (py >= 0) & (py < c.h)
        _paint(c, py[ok], px[ok], np.full(ok.sum(), "."), np.full(ok.sum(), 0.5 + 0.4 * beat), ctx)


@register
class Clock(Template):
    meta = TemplateMeta(
        "clock", "An analog clock face; the second hand ticks on every beat, the minute hand per bar.",
        busy=0.3, energy=(0.1, 0.8), sections=frozenset({"intro", "verse", "bridge", "outro"}),
        tags=frozenset({"calm", "neutral", "mid-energy", "dark"}), motifs=frozenset({"time"}),
        lyric_blends=frozenset({"overlay", "knockout"}), 
    )

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        cx, cy = c.w / 2, c.h / 2
        R = min(c.w / 2, c.h) * 0.45
        pal = ctx.palette
        # rim and hour marks
        a = np.linspace(0, math.tau, 240, endpoint=False)
        _paint(c, (cy + np.sin(a) * R).astype(int), (cx + np.cos(a) * R * 2).astype(int), np.full(240, "o"),
               np.full(240, 0.65), ctx)
        for h in range(12):
            ang = h / 12 * math.tau - math.pi / 2
            c.put(cx + math.cos(ang) * R * 1.8, cy + math.sin(ang) * R * 0.9, "#" if h % 3 == 0 else "+",
                  pal.ui_at(0.8))
        tick = ctx.beat_idx + min(1.0, ctx.beat_phase * 6)          # snaps forward on each beat
        hands = [(tick / 60, 0.92, "*", pal.err_at(0.95)), (tick / 4 / 60, 0.78, "#", pal.ui_at(1.0)),
                 (tick / 4 / 60 / 12, 0.5, "@", pal.accent)]
        for turns, length, ch, col in hands:
            ang = turns * math.tau - math.pi / 2
            c.line(cx, cy, cx + math.cos(ang) * R * 2 * length, cy + math.sin(ang) * R * length, ch, col)
        c.put(cx, cy, "@", pal.accent_at(0.6 + 0.4 * ctx.pulse(0.15)))


@register
class Skyline(Template):
    meta = TemplateMeta(
        "skyline", "A city skyline scrolling past at night; windows blink with the drums.",
        busy=0.4, energy=(0.2, 0.9), sections=frozenset({"intro", "verse", "prechorus", "bridge", "outro"}),
        tags=frozenset({"mid-energy", "dark", "neutral", "energetic"}), motifs=frozenset({"city", "night", "speed"}),
        lyric_blends=frozenset({"overlay", "knockout"}), 
    )

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        seed = str_seed(self.shot.id) & 0xFFFF
        scroll = self.analysis.travel(ctx.t) * 8
        for layer, (speed, dim, base) in enumerate(((0.35, 0.28, 0.45), (1.0, 0.7, 0.62))):
            x_world = np.arange(c.w) + scroll * speed
            bld = (x_world // 7).astype(int)               # one building every 7 columns
            height = (_u16(bld, seed + layer) % 100) / 100 * c.h * base + c.h * 0.12
            top = (c.h - height).astype(int)
            yy = np.arange(c.h)[:, None]
            body = yy >= top[None, :]
            wall = (x_world.astype(int) % 7 == 0)[None, :] | (yy == top[None, :])
            lit_seed = _u16(bld[None, :] * 131 + yy, seed + 10 + layer)
            window = body & ~wall & ((x_world.astype(int) % 7) % 2 == 1)[None, :] & (yy % 2 == 1)
            on = window & (lit_seed % 100 < 35 + 40 * ctx.energy)
            blink = window & ((lit_seed + ctx.beat_idx * 7) % 100 < 12 * ctx.kick * 3)
            ys, xs = np.nonzero(body & wall)
            _paint(c, ys, xs, np.where(yy[ys, 0] == top[xs], "_", "|"), np.full(len(ys), dim), ctx)
            ys, xs = np.nonzero(on | blink)
            _paint(c, ys, xs, np.full(len(ys), "#"), np.full(len(ys), dim + 0.25), ctx, blink[ys, xs])
        # a moon
        c.put(c.w * 0.8, 2, "(", pal.accent_at(0.7))


BOOT = ["[ OK ] mounting /dev/audio", "[ OK ] decoding stream", "[ OK ] tempo lock {bpm} BPM",
        "[ OK ] key: {key}", "[ OK ] {beats} beats indexed", "[ OK ] lyrics: {lines} lines",
        "[ .. ] loading shaders", "[ OK ] palette: {palette}", "[ OK ] style: {tags}", "> run --pv \"{title}\""]


@register
class BootLog(Template):
    meta = TemplateMeta(
        "boot_log", "A terminal booting up: the song's own facts typed out line by line on the beat.",
        busy=0.5, energy=(0.0, 0.7), sections=frozenset({"intro"}),
        tags=frozenset({"dark", "mid-energy", "calm", "neutral"}), motifs=frozenset({"pulse", "time"}),
        lyric_blends=frozenset({"overlay"}), 
    )

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        a = self.analysis
        pal = ctx.palette
        title = a.source.get("audio", "track")
        title = title.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        facts = dict(bpm=f"{a.bpm:.0f}", key=a.style.get("key", "?"), beats=len(a.beats), lines=len(a.lyrics),
                     palette=pal.name, tags=", ".join(a.style.get("tags", [])) or "-", title=title[:40])
        lines = [s.format(**facts) for s in BOOT] + [ln.text for ln in a.lyrics[:4]]
        per_line = 60.0 / max(40, a.bpm)                    # one line per beat
        shown = ctx.local_t / per_line
        x0, y0 = 3, 2
        for i, s in enumerate(lines[: int(shown) + 1]):
            y = y0 + i
            if y >= c.h - 2:
                break
            typed = s if i < int(shown) else s[: int((shown - int(shown)) * len(s) * 1.6)]
            col = pal.ui_at(0.85) if not s.startswith(">") else pal.accent
            if s.startswith("[ OK ]"):
                c.put(x0, y, "[ OK ]", pal.anom_at(0.9))
                c.put(x0 + 6, y, crop(typed[6:], c.w - x0 - 8), col)
            else:
                c.put(x0, y, crop(typed, c.w - x0 - 2), col)
        # progress bar on the last row, filling once per bar
        bar_u = ((ctx.beat_idx % 4) + ctx.beat_phase) / 4
        wbar = c.w - 12
        c.put(3, c.h - 2, "[" + "#" * int(bar_u * wbar) + "." * (wbar - int(bar_u * wbar)) + "]", pal.ui_at(0.6))
        if int(ctx.t * 2) % 2 == 0:
            c.put(x0 + 1 + len(lines[min(len(lines) - 1, int(shown))]) % (c.w - 6), y0 + min(int(shown), c.h - 5),
                  "_", pal.accent)
