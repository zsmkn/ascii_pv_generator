"""Retro game templates (M8, third round): tetris, invaders, pong, runner (easter eggs: at most one a song).

Each one plays a little game in time with the music: pieces land, aliens step and the ball is returned on the
beats. The game is simulated once per shot in __init__ (deterministic, seeded by the shot id) and a frame only
looks the state up at t, so frames stay pure functions of t.
"""

from __future__ import annotations

import bisect
import math
import numpy as np

from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import FrameCtx, hash32, str_seed
from ascii_pv.templates.base import Template, TemplateMeta, register


# ---------------------------------------------------------------- shared helpers

def beat_grid(a, t0: float, t1: float, div: int = 1) -> list[float]:
    """Beat times (and `div` subdivisions of each beat) covering [t0, t1], extrapolated past the analysed beats."""
    period = 60.0 / max(1.0, a.bpm)
    beats = [b for b in a.beats if t0 - 2 * period <= b <= t1 + 2 * period]
    if len(beats) < 2:
        k0 = math.floor(t0 / period) - 1
        beats = [k * period for k in range(k0, k0 + int((t1 - t0) / period) + 4)]
    while beats[0] > t0:
        beats.insert(0, beats[0] - period)
    while beats[-1] < t1:
        beats.append(beats[-1] + period)
    out = []
    for b0, b1 in zip(beats, beats[1:]):
        out += [b0 + (b1 - b0) * k / div for k in range(div)]
    return out + [beats[-1]]


def sprite(c: Canvas, x: int, y: int, lines: list[str], fg, bold: bool = False) -> None:
    """Draw lines of text; spaces are transparent."""
    for j, line in enumerate(lines):
        yy = y + j
        if not 0 <= yy < c.h:
            continue
        for i, ch in enumerate(line):
            xx = x + i
            if ch != " " and 0 <= xx < c.w:
                c.ch[yy, xx] = ch
                c.fg[yy, xx] = fg
                c.bold[yy, xx] = bold


DIGITS = {
    "0": ["███", "█ █", "█ █", "█ █", "███"], "1": [" █ ", "██ ", " █ ", " █ ", "███"],
    "2": ["███", "  █", "███", "█  ", "███"], "3": ["███", "  █", "███", "  █", "███"],
    "4": ["█ █", "█ █", "███", "  █", "  █"], "5": ["███", "█  ", "███", "  █", "███"],
    "6": ["███", "█  ", "███", "█ █", "███"], "7": ["███", "  █", "  █", "  █", "  █"],
    "8": ["███", "█ █", "███", "█ █", "███"], "9": ["███", "█ █", "███", "  █", "███"],
}


def big_number(c: Canvas, x: int, y: int, n: int, fg) -> None:
    for i, d in enumerate(str(n)):
        sprite(c, x + i * 4, y, DIGITS[d], fg)


def _rng(seed: int, *k: int) -> int:
    return hash32(seed, *k)


# ---------------------------------------------------------------- tetris

PIECES = {"I": [(0, 0), (0, 1), (0, 2), (0, 3)], "O": [(0, 0), (0, 1), (1, 0), (1, 1)],
          "T": [(0, 0), (0, 1), (0, 2), (1, 1)], "S": [(0, 1), (0, 2), (1, 0), (1, 1)],
          "Z": [(0, 0), (0, 1), (1, 1), (1, 2)], "J": [(0, 0), (1, 0), (1, 1), (1, 2)],
          "L": [(0, 2), (1, 0), (1, 1), (1, 2)]}
KINDS = list(PIECES)


def _rotations(cells: list[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    out, cur = [], cells
    for _ in range(4):
        r0, c0 = min(r for r, _ in cur), min(c for _, c in cur)
        norm = sorted((r - r0, c - c0) for r, c in cur)
        if norm not in out:
            out.append(norm)
        cur = [(c, -r) for r, c in cur]
    return out


ROTS = {k: _rotations(v) for k, v in PIECES.items()}


@register
class Tetris(Template):
    meta = TemplateMeta(
        "tetris", "A game of Tetris: a piece lands on every beat, lines clear with a flash.",
        busy=0.45, energy=(0.3, 0.9), sections=frozenset({"verse", "prechorus", "chorus", "bridge"}),
        tags=frozenset({"retro", "bright", "energetic", "percussive"}), motifs=frozenset({"pop", "break", "time"}),
        lyric_blends=frozenset({"overlay", "knockout"}), easter_egg=True,
    )
    defaults = {"every": 1}
    W, H = 10, 20

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        seed = str_seed(shot.id) & 0xFFFF
        every = int(self.params["every"]) or (2 if analysis.bpm > 140 else 1)
        self.times = beat_grid(analysis, shot.start, shot.end + 1)[::every]
        board = np.zeros((self.H, self.W), np.int8)
        self.steps = []           # per piece: (kind, rot cells, row, col, board before, cleared rows)
        lines = 0
        for k in range(len(self.times)):
            kind = KINDS[_rng(seed, k) % 7]
            row, col, cells = self._best(board, kind)
            if row is None or board[:4].any():                       # topped out: a fresh game
                board[:] = 0
                row, col, cells = self._best(board, kind)
            before = board.copy()
            for r, c in cells:
                board[row + r, col + c] = KINDS.index(kind) + 1
            full = [r for r in range(self.H) if board[r].all()]
            if full:
                lines += len(full)
                board = np.vstack([np.zeros((len(full), self.W), np.int8), np.delete(board, full, axis=0)])
            self.steps.append((kind, cells, row, col, before, full, lines))

    def _best(self, board: np.ndarray, kind: str):
        """The usual greedy player: lowest stack, fewest holes, smooth surface, lines cleared."""
        best, pick = -1e9, (None, None, None)
        heights_of = lambda b: np.where(b.any(0), self.H - (b > 0).argmax(0), 0)   # noqa: E731
        for cells in ROTS[kind]:
            cw = max(c for _, c in cells) + 1
            for col in range(self.W - cw + 1):
                row = -1
                while all(row + 1 + r < self.H and not board[row + 1 + r, col + c] for r, c in cells):
                    row += 1
                if row < 0:
                    continue
                b = board.copy()
                for r, c in cells:
                    b[row + r, col + c] = 1
                full = int(b.all(1).sum())
                hts = heights_of(b)
                holes = int((np.maximum.accumulate(b > 0, axis=0) & (b == 0)).sum())   # empty under a block
                score = -0.51 * hts.sum() + 0.76 * full - 0.36 * holes - 0.18 * np.abs(np.diff(hts)).sum()
                if score > best:
                    best, pick = score, (row, col, cells)
        return pick

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        big = c.h >= self.H * 2 + 2 and c.w >= 100
        sx, sy = (4, 2) if big else (2, 1)
        tile = ["┌──┐", "└──┘"] if big else ["[]"]
        ww, wh = self.W * sx, self.H * sy
        x0, y0 = (c.w - ww) // 2, max(0, (c.h - wh - 1) // 2)
        k = max(0, min(len(self.steps) - 1, bisect.bisect_right(self.times, ctx.t) - 1))
        kind, cells, row, col, before, full, lines = self.steps[k]
        colors = [pal.ui_at(0.85), pal.accent_at(0.9), pal.anom_at(0.85), pal.err_at(0.8)]

        def cell(r: int, cc: int, v: int, flash: bool = False) -> None:
            fg = pal.accent_at(1.0) if flash else colors[(v - 1) % 4]
            for j in range(sy):
                c.put(x0 + cc * sx, y0 + r * sy + j, "▓" * sx if flash else tile[j], fg)

        # well
        for r in range(wh):
            c.put(x0 - 2, y0 + r, "<!", pal.ui_at(0.5))
            c.put(x0 + ww, y0 + r, "!>", pal.ui_at(0.5))
        c.put(x0 - 2, y0 + wh, "<!" + "=" * ww + "!>", pal.ui_at(0.5))
        for r in range(self.H):
            for cc in range(self.W):
                c.put(x0 + cc * sx + sx // 2, y0 + r * sy + sy // 2, ".", pal.ui_at(0.15))
        # the board; the rows the last piece completed flash for a moment before they go
        prev = self.steps[k - 1] if k > 0 else None
        if prev is not None and prev[5] and ctx.t - self.times[k] < 0.18:
            b = prev[4].copy()
            for r, cc in prev[1]:
                b[prev[2] + r, prev[3] + cc] = KINDS.index(prev[0]) + 1
            for r, cc in zip(*np.nonzero(b)):
                cell(r, cc, int(b[r, cc]), flash=r in prev[5])
        else:
            for r, cc in zip(*np.nonzero(before)):
                cell(r, cc, int(before[r, cc]))
            # the falling piece: slides to its column, then drops onto the stack by the next beat
            t0 = self.times[k]
            t1 = self.times[k + 1] if k + 1 < len(self.times) else t0 + 0.5
            p = min(1.0, max(0.0, (ctx.t - t0) / max(1e-3, t1 - t0)))
            spawn = (self.W - (max(cc for _, cc in cells) + 1)) // 2
            cur_col = round(spawn + (col - spawn) * min(1.0, p / 0.35))
            cur_row = math.floor(-1 + (row + 1) * p ** 2)
            v = KINDS.index(kind) + 1
            for r, cc in cells:
                if cur_row + r >= 0:
                    cell(cur_row + r, cur_col + cc, v)
        # side panel
        px = x0 + ww + 4
        if px + 12 < c.w:
            nxt = self.steps[k + 1][0] if k + 1 < len(self.steps) else kind
            c.put(px, y0, "NEXT", pal.ui_at(0.6))
            for r, cc in ROTS[nxt][0]:
                c.put(px + cc * 2, y0 + 2 + r, "[]", colors[KINDS.index(nxt) % 4])
            c.put(px, y0 + 6, f"LINES {lines - len(full):04d}", pal.ui_at(0.7))
            c.put(px, y0 + 8, f"SCORE {k * 40 + (lines - len(full)) * 100:06d}", pal.ui_at(0.7))


# ---------------------------------------------------------------- space invaders

ALIENS = [("/o\\", "\\o/"), ("{@}", "}@{"), ("<=>", ">=<")]


@register
class Invaders(Template):
    meta = TemplateMeta(
        "invaders", "Space Invaders: the formation steps on every beat, the cannon fires on the kicks.",
        busy=0.45, energy=(0.4, 1.0), sections=frozenset({"verse", "prechorus", "chorus", "bridge"}),
        tags=frozenset({"retro", "energetic", "percussive", "dark"}), motifs=frozenset({"fight", "star", "fly"}),
        lyric_blends=frozenset({"overlay", "knockout"}), easter_egg=True,
    )
    defaults = {"rows": 5, "cols": 11}
    BULLET = 0.3          # s from the cannon to the formation

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        self.seed = str_seed(shot.id) & 0xFFFF
        self.beats = beat_grid(analysis, shot.start, shot.end + 1)
        f = analysis.features
        self.fires = []        # (t_fire, wave, target (r, c)) ; dead[(wave, r, c)] = t_hit
        self.dead: dict[tuple[int, int, int], float] = {}
        self.waves = [self.beats[0]]
        R, C = int(self.params["rows"]), int(self.params["cols"])
        alive = {(r, cc) for r in range(R) for cc in range(C)}
        for k, b in enumerate(self.beats[1:], 1):
            i = min(len(f["kick"]) - 1, int(b * f["rate"]))
            if f["kick"][i] < 0.35 and k % 2:
                continue
            if not alive:
                self.waves.append(b)
                alive = {(r, cc) for r in range(R) for cc in range(C)}
                continue
            cols = sorted({cc for _, cc in alive})
            tc = cols[_rng(self.seed, k) % len(cols)]
            tr = max(r for r, cc in alive if cc == tc)                # the lowest alien in that column
            alive.discard((tr, tc))
            self.fires.append((b, len(self.waves) - 1, (tr, tc)))
            self.dead[(len(self.waves) - 1, tr, tc)] = b + self.BULLET

    def _formation(self, c: Canvas, step: int) -> tuple[int, int]:
        """Top-left of the formation after `step` beat steps: across, down a row, back..."""
        C = int(self.params["cols"])
        span = max(1, (c.w - C * 6 - 4) // 2)
        sweep, within = divmod(step, span)
        x = 2 + (within if sweep % 2 == 0 else span - within) * 2
        return x, 3 + sweep % max(1, c.h // 3)

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        wave = max(0, bisect.bisect_right(self.waves, ctx.t) - 1)
        w0 = self.waves[wave]
        step = max(0, bisect.bisect_right(self.beats, ctx.t) - 1 - bisect.bisect_left(self.beats, w0))
        fx, fy = self._formation(c, step)
        R, C = int(self.params["rows"]), int(self.params["cols"])
        colors = [pal.anom_at(0.9), pal.accent_at(0.9), pal.ui_at(0.85)]
        frame = step % 2
        for r in range(R):
            kind = 0 if r == 0 else 1 if r < 3 else 2
            for cc in range(C):
                x, y = fx + cc * 6, fy + r * 2
                td = self.dead.get((wave, r, cc), math.inf)
                if ctx.t < td:
                    sprite(c, x, y, [ALIENS[kind][frame]], colors[kind])
                elif ctx.t < td + 0.35:                              # explosion
                    sprite(c, x, y, ["\\*/" if ctx.t < td + 0.15 else "* *"], pal.err_at(0.9))
        # cannon: glides under its next target, fires on the beat
        gy = c.h - 2
        upcoming = [fr for fr in self.fires if fr[0] >= ctx.t - self.BULLET]
        prev = [fr for fr in self.fires if fr[0] < ctx.t]

        def target_x(fr):
            tf, wv, (tr, tc) = fr
            st = max(0, bisect.bisect_right(self.beats, tf) - 1 - bisect.bisect_left(self.beats, self.waves[wv]))
            return self._formation(c, st)[0] + tc * 6 + 1

        nxt = upcoming[0] if upcoming else None
        cx = c.w / 2
        if nxt is not None:
            x1 = target_x(nxt)
            x0 = target_x(prev[-1]) if prev else c.w / 2
            t0 = prev[-1][0] if prev else self.shot.start
            p = min(1.0, max(0.0, (ctx.t - t0) / max(1e-3, nxt[0] - t0)))
            cx = x0 + (x1 - x0) * (1 - (1 - p) ** 3)
        sprite(c, int(cx) - 2, gy, ["_/^\\_"], pal.ui_at(0.95), bold=True)
        # bullets in flight
        for tf, wv, (tr, tc) in self.fires:
            if tf <= ctx.t < tf + self.BULLET:
                bx = target_x((tf, wv, (tr, tc)))
                ty = fy + tr * 2
                by = gy - 1 - (gy - 1 - ty) * (ctx.t - tf) / self.BULLET
                c.put(bx, by, "|", pal.accent_at(1.0))
        # bunkers
        for i in range(4):
            bx = int(c.w * (i + 1) / 5) - 3
            sprite(c, bx, gy - 3, ["/###\\", "#####"], pal.ui_at(0.35))
        c.put(0, c.h - 1, "_" * c.w, pal.ui_at(0.3))


# ---------------------------------------------------------------- pong

@register
class Pong(Template):
    meta = TemplateMeta(
        "pong", "Pong: the ball is returned exactly on the beats; now and then a miss scores a point.",
        busy=0.3, energy=(0.2, 0.8), sections=frozenset({"intro", "verse", "bridge", "instrumental"}),
        tags=frozenset({"retro", "sparse", "percussive", "neutral"}), motifs=frozenset({"pulse", "pop", "time"}),
        lyric_blends=frozenset({"overlay", "knockout"}), easter_egg=True,
    )
    defaults = {"every": 1, "miss": 13}

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        self.seed = str_seed(shot.id) & 0xFFFF
        every = int(self.params["every"]) or 1
        self.hits = beat_grid(analysis, shot.start, shot.end + 1)[::every]

    def _segments(self, top: float, bot: float):
        """Per hit segment: (y start, unfolded dy, missed?) and the score before it."""
        span = bot - top
        y, out, score = (top + bot) / 2, [], [0, 0]
        miss_every = int(self.params["miss"])
        for k in range(len(self.hits)):
            dy = ((_rng(self.seed, k) % 1000) / 1000 - 0.5) * 2.4 * span
            missed = k % miss_every == miss_every - 1
            out.append((y, dy, missed, tuple(score)))
            if missed:
                score[(k + 1) % 2] += 1
                y = (top + bot) / 2
            else:
                y = _fold(y + dy, top, bot)
        return out

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        top, bot = 1, c.h - 2
        segs = self._segments(top, bot)
        c.put(0, 0, "▀" * c.w, pal.ui_at(0.5))
        c.put(0, c.h - 1, "▄" * c.w, pal.ui_at(0.5))
        for y in range(1, c.h - 1, 2):
            c.put(c.w // 2, y, "│", pal.ui_at(0.3))
        k = max(0, min(len(self.hits) - 2, bisect.bisect_right(self.hits, ctx.t) - 1))
        y0, dy, missed, score = segs[k]
        if c.w >= 40 and c.h >= 14:
            big_number(c, c.w // 2 - 8, 2, score[0], pal.ui_at(0.6))
            big_number(c, c.w // 2 + 5, 2, score[1], pal.ui_at(0.6))
        t0, t1 = self.hits[k], self.hits[k + 1]
        p = (ctx.t - t0) / max(1e-3, t1 - t0)
        left_px, right_px = 3, c.w - 4
        going_right = k % 2 == 0
        xa, xb = (left_px + 1, right_px - 1) if going_right else (right_px - 1, left_px + 1)
        bx = xa + (xb - xa) * p
        if missed:
            bx = xa + (xb - xa) * p * 1.15
        by = _fold(y0 + dy * p, top, bot)
        for lag in (0.06, 0.03):                                     # a short trail
            pp = max(0.0, p - lag)
            c.put(xa + (xb - xa) * pp, _fold(y0 + dy * pp, top, bot), ".", pal.ui_at(0.4))
        if 0 <= bx < c.w:
            c.put(bx, by, "█", pal.accent_at(1.0), bold=True)
        # paddles: the receiver meets the ball (or just misses it), the other drifts to the middle
        ph = max(2, (bot - top) // 6)
        arrive = _fold(y0 + dy, top, bot)
        recv = arrive if not missed else _fold(arrive + ph * 2.2, top + ph, bot - ph)
        ease = 1 - (1 - min(1.0, p / 0.8)) ** 2
        mid = (top + bot) / 2
        prev_y = by if p < 0.05 else mid
        ry = prev_y + (recv - prev_y) * ease
        sy = by + (mid - by) * min(1.0, p * 1.5)
        for x, yc in ((right_px if going_right else left_px, ry), (left_px if going_right else right_px, sy)):
            for j in range(-ph // 2, ph - ph // 2):
                c.put(x, yc + j, "█", pal.ui_at(0.95))


def _fold(y: float, top: float, bot: float) -> float:
    """Reflect y back into [top, bot] (a ball bouncing off the walls)."""
    span = bot - top
    v = (y - top) % (2 * span)
    return top + (v if v <= span else 2 * span - v)


# ---------------------------------------------------------------- runner

RUN = [["  O  ", " /|\\ ", "/ | \\", "  |  ", " / \\ "], ["  O  ", " /|\\ ", "/ | \\", "  |  ", "  |\\ "]]
JUMP = ["  O  ", "\\ | /", "  |  ", " / \\ ", "     "]
SMALL_RUN = [[" o ", "/|\\", "/ \\"], [" o ", "/|\\", " | "]]
SMALL_JUMP = ["\\o/", " | ", "/ \\"]
CACTI = [["|", "|"], ["|_|", " | ", " | "], ["| |", "|_|", " | ", " | "]]


@register
class Runner(Template):
    meta = TemplateMeta(
        "runner", "An endless runner: obstacles arrive on the beats and the runner hops them; hills scroll behind.",
        busy=0.35, energy=(0.3, 0.9), sections=frozenset({"intro", "verse", "prechorus", "bridge", "outro"}),
        tags=frozenset({"retro", "bright", "mid-energy", "sparse"}), motifs=frozenset({"speed", "fly", "city"}),
        lyric_blends=frozenset({"overlay", "knockout"}), easter_egg=True,
    )
    defaults = {"speed": 1.0}

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        seed = str_seed(shot.id) & 0xFFFF
        beats = beat_grid(analysis, shot.start, shot.end + 4)
        f = analysis.features
        self.obstacles = []    # (beat time it reaches the runner, cactus kind)
        last = -1e9
        for k, b in enumerate(beats):
            i = min(len(f["kick"]) - 1, int(b * f["rate"]))
            gap = b - last
            if gap > 0.9 * 60 / max(1, analysis.bpm) and (f["kick"][i] > 0.3 or _rng(seed, k) % 3 == 0):
                self.obstacles.append((b, _rng(seed, k, 5) % len(CACTI)))
                last = b
        self.period = 60.0 / max(1.0, analysis.bpm)

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        w, h = c.w, c.h
        big = h >= 30
        gy = int(h * 0.75)
        v = w / (self.period * 6) * float(self.params["speed"])      # columns per second: ~6 beats across
        px = int(w * 0.18)
        dist = (ctx.t - self.shot.start) * v
        # far hills and clouds (parallax)
        xs = np.arange(w)
        hill = (np.sin((xs + dist * 0.15) * 0.045) * 0.6 + np.sin((xs + dist * 0.15) * 0.11 + 1.3) * 0.4 + 1) / 2
        hy = (gy - 2 - hill * h * 0.25).astype(int)
        slope = np.gradient(hy.astype(float))
        hch = np.where(slope < -0.3, "/", np.where(slope > 0.3, "\\", "_"))
        for x in range(w):
            if 0 <= hy[x] < gy:
                c.put(x, hy[x], hch[x], pal.ui_at(0.25))
        for k in range(4):
            cx = int((w * (k + 0.5) / 4 - dist * 0.08 + w * 10) % (w + 12)) - 6
            c.put(cx, 2 + (k * 3) % max(1, int(h * 0.25)), "~~~~", pal.ui_at(0.3))
        # ground, with speckles rushing by
        c.put(0, gy, "_" * w, pal.ui_at(0.7))
        for x in range(w):
            if hash32(int(x + dist) // 1, 17) % 9 == 0 and gy + 1 < h:
                c.put(x, gy + 1, "." if hash32(int(x + dist), 3) % 2 else "-", pal.ui_at(0.35))
        # obstacles: each is at the runner exactly on its beat
        jump_y = 0.0
        half = self.period * 0.55
        for tb, kind in self.obstacles:
            ox = px + 2 + (tb - ctx.t) * v
            if -4 < ox < w + 2:
                shape = CACTI[kind]
                sprite(c, int(ox), gy - len(shape) + 1, shape, pal.accent_at(0.8))
            d = ctx.t - tb
            if abs(d) < half:
                jump_y = max(jump_y, 1 - (d / half) ** 2)
        lift = int(round(jump_y * (6 if big else 3)))
        if big:
            fig = JUMP if lift > 0 else RUN[int(ctx.beat_phase * 2) % 2]
        else:
            fig = SMALL_JUMP if lift > 0 else SMALL_RUN[int(ctx.beat_phase * 2) % 2]
        sprite(c, px, gy - len(fig) + 1 - lift, fig, pal.anom_at(1.0), bold=True)
        score = int((ctx.t - self.shot.start) * 10)
        c.put(w - 18, 1, f"HI 00{score + 1200:05d} {score:05d}", pal.ui_at(0.6))
