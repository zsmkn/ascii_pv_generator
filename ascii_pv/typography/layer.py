"""The lyric layer: each sung line becomes big letters made of glyph particles.

Per line one LyricCue (defaults when the storyboard has none) chooses:
  mode    takeover    full screen; the picture dims and a glyph storm runs behind the letters
          line        a band in the lower part of the screen over the running picture
          word_flash  one huge word at a time, swapped as each word is sung (fast lines)
          stack       the whole line laid out, each word dropping in only when it is sung (slow lines)
          embed       big letters in the middle that belong to the picture (use with blend mask / knockout)
  font    bit5x5 / impact / black / serif / script / mono (typography/fonts.py)
  fill    binary / hex / self / ramp / block / outline / solid (typography/fills.py)
  enter   assemble / decode / rain / scan / glitch_in; exit explode / fade / melt / shatter / crt_off (motion.py)
  blend   overlay   letters drawn over the picture
          mask      the picture shows only through the letters; everything else dims
          knockout  the letters are holes cut into the picture, with a thin rim
          echo      overlay, and the previous line lingers faintly behind as texture
          obstacle  overlay; templates that read ctx.lyric_mask (particles) flow around the letters

Timing: window [t_in, t_out] per line, t_in = line start - PRE (letters assemble just before the singing),
t_out = the next line's t_in or the line end. While held: unsung words show only their outline (in stack mode
they are absent), a word fills in and flashes when it is sung, the sung word stays in the accent colour, and a
'#' sweep crosses the letters once per bar. Kicks lift brightness.
Readability: a line that does not fit is split into chunks shown one after another; letters over a busy picture
get a dark halo (line) or a drop shadow (takeover, word_flash, embed).

Everything is a pure function of (lyric time, canvas size); per-piece particle data is cached.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.context import hash32
from ascii_pv.core.palette import Palette
from ascii_pv.core.cjk import glue, tokens
from ascii_pv.core.storyboard import LyricCue, Storyboard
from ascii_pv.typography import fills, motion
from ascii_pv.typography.inkmask import InkMask, chunks, fits, ink_mask

PRE = 0.35                 # s the letters start assembling before the line is sung
MIN_LIT = 0.3              # s the last word of a piece stays lit before the next piece may take over
LIT_BEFORE_EXIT = 0.2      # s of that before the exit animation may start (it then gets the rest, at least 0.1 s)
FILL_IN = 0.12             # s a sung word takes to fill in
STACK_DROP = 0.3           # s a stack-mode word takes to drop into place (it lights when it lands)


def shown_after(mode: str) -> float:
    """How long after its start a sung word is on screen: at once (it fills in where it stands), or once it has
    dropped into place (stack). MIN_LIT and LIT_BEFORE_EXIT count from then."""
    return STACK_DROP if mode == "stack" else 0.0


def switch_time(last_word: float, next_word: float, lead: float, min_lit: float = MIN_LIT) -> float:
    """When the next piece replaces the current one: `lead` before its first word (time to assemble), but not
    before the current piece's last word has been lit for MIN_LIT, and never after the next word is sung.
    With dense words (CJK, rap) the next piece assembles while its first word is already being sung."""
    return max(next_word - lead, min(last_word + min_lit, next_word))
ALPHABET = np.array(list("0123456789ABCDEF<>[]{}();:=/\\|+-*#"))
FILLS = fills.FILLS        # kept for callers that use the old name
THIN = np.array(list(" .,'`:;-_~^\"|!/\\()"))   # picture glyphs too light to draw a letter with (mask blend)
MODES = ("takeover", "line", "word_flash", "stack", "embed")
BLENDS = ("overlay", "mask", "knockout", "echo", "obstacle")


def _u16(a: np.ndarray, salt: int) -> np.ndarray:
    """Vectorised integer hash -> 0..65535."""
    v = (a.astype(np.uint64) * np.uint64(0x9E3779B1) + np.uint64(salt * 0x85EBCA77 + 0x165667B1)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(15)
    v = (v * np.uint64(0x2C1B3C6D)) & np.uint64(0xFFFFFFFF)
    v ^= v >> np.uint64(12)
    return (v & np.uint64(0xFFFF)).astype(np.int64)


@dataclass
class Piece:
    """One displayable unit: a whole line, a chunk of it, or (word_flash) one word."""
    line: int
    words: tuple[str, ...]
    word_times: tuple[float, ...]   # start time of each word (lyric time)
    t_in: float
    t_out: float
    cue: LyricCue
    glue: tuple[bool, ...] | None = None   # per word: drawn with no gap before it (CJK, see core/cjk.py)


@dataclass(frozen=True)
class Particles:
    ink: InkMask
    ox: np.ndarray       # origin x
    oy: np.ndarray       # origin y
    delay: np.ndarray    # 0..1 start delay share
    k: np.ndarray        # per-particle hash
    fill: np.ndarray     # resting glyph


class LyricLayer:
    def __init__(self, analysis: Analysis, board: Storyboard):
        self.analysis = analysis
        self.board = board
        cues = {c.line: c for c in board.lyrics}
        self.lines = analysis.lyrics
        self.cues = [cues.get(i, LyricCue(line=i)) for i in range(len(self.lines))]
        self._windows = self._line_windows()
        self._piece_cache: dict[tuple, list[Piece]] = {}

    # ------------------------------------------------------------ timing

    def _line_windows(self) -> list[tuple[float, float]]:
        out = []
        starts = [ln.start + c.offset for ln, c in zip(self.lines, self.cues)]
        lasts = [(ln.words[-1].start if ln.words else ln.start) + c.offset + shown_after(c.mode)
                 for ln, c in zip(self.lines, self.cues)]
        ins = [starts[0] - PRE if starts else 0.0]
        ins += [switch_time(lasts[i - 1], starts[i], PRE) for i in range(1, len(self.lines))]
        for i, ln in enumerate(self.lines):
            nxt = ins[i + 1] if i + 1 < len(self.lines) else math.inf
            t_out = min(ln.end + self.cues[i].offset, nxt)
            out.append((ins[i], max(ins[i] + 0.2, t_out)))
        return out

    @staticmethod
    def area(mode: str, w: int, h: int) -> tuple[int, int, int, int]:
        if mode in ("takeover", "word_flash", "embed"):
            return (3, 2, w - 6, h - 4)
        if mode == "stack":
            top = int(h * 0.18)
            return (3, top, w - 6, int(h * 0.64))
        top = int(h * 0.58)
        return (2, top, w - 4, h - top - 2)

    @staticmethod
    def max_px(mode: str) -> int:
        return 6 if mode == "word_flash" else 99

    def _pieces(self, i: int, w: int, h: int) -> list[Piece]:
        key = (i, w, h)
        if key in self._piece_cache:
            return self._piece_cache[key]
        ln, cue = self.lines[i], self.cues[i]
        words = tuple(wd.text for wd in ln.words) or tokens(ln.text)
        times = tuple(wd.start + cue.offset for wd in ln.words) or tuple([ln.start + cue.offset] * len(words))
        g = glue(words, ln.text)
        g = g if any(g) else None
        t_in, t_out = self._windows[i]
        _, _, aw, ah = self.area(cue.mode, w, h)
        if cue.mode == "word_flash":
            runs = [(j, j + 1) for j in range(len(words))]
        elif fits(words, aw, ah, cue.font, g):
            runs = [(0, len(words))]
        else:
            runs = chunks(words, aw, ah, cue.font, g)
        lead = 0.08 if cue.mode == "word_flash" else PRE * 0.6
        pieces = []
        # word_flash shows one word at a time: it is lit as soon as it lands, so it keeps its short lead
        min_lit = 0.0 if cue.mode == "word_flash" else MIN_LIT + shown_after(cue.mode)
        cuts = [switch_time(times[b - 1], times[b], lead, min_lit) for _, b in runs[:-1]]
        for j, (a, b) in enumerate(runs):
            p_in = t_in if j == 0 else cuts[j - 1]
            p_out = t_out if j == len(runs) - 1 else cuts[j]
            sub = (False, *g[a + 1:b]) if g else None
            pieces.append(Piece(i, words[a:b], times[a:b], p_in, max(p_in + 0.12, p_out), cue, sub))
        self._piece_cache[key] = pieces
        return pieces

    def active(self, lt: float, w: int, h: int) -> Piece | None:
        for i, (a, b) in enumerate(self._windows):
            if a <= lt < b:
                for p in self._pieces(i, w, h):
                    if p.t_in <= lt < p.t_out:
                        return p
        return None

    def previous(self, p: Piece, w: int, h: int) -> Piece | None:
        """The piece shown just before `p` (for echo)."""
        pieces = self._pieces(p.line, w, h)
        j = pieces.index(p)
        if j > 0:
            return pieces[j - 1]
        return self._pieces(p.line - 1, w, h)[-1] if p.line > 0 else None

    # ------------------------------------------------------------ geometry

    @staticmethod
    @lru_cache(maxsize=96)
    def _particles(words: tuple[str, ...], area: tuple[int, int, int, int], size: tuple[int, int], fill: str,
                   font: str, max_px: int, seed: int, glue: tuple[bool, ...] | None = None) -> Particles | None:
        ink = ink_mask(words, area, size, max_px, font, glue)
        if ink is None:
            return None
        w, h = size
        idx = np.arange(ink.n)
        k = _u16(idx, seed)
        ox = _u16(idx, seed + 1) % w
        oy = _u16(idx, seed + 2) % h
        delay = (_u16(idx, seed + 3) % 1000) / 1000.0
        return Particles(ink, ox, oy, delay, k, fills.resting(fill, ink, words, k))

    def _parts(self, p: Piece, w: int, h: int) -> Particles | None:
        cue = p.cue
        return self._particles(p.words, self.area(cue.mode, w, h), (w, h), cue.fill, cue.font,
                               self.max_px(cue.mode), self._seed(p), p.glue)

    def mask_at(self, lt: float, w: int, h: int) -> np.ndarray | None:
        p = self.active(lt, w, h)
        if p is None:
            return None
        parts = self._parts(p, w, h)
        return parts.ink.mask if parts else None

    def _seed(self, p: Piece) -> int:
        return hash32(self.board.seed, p.line, len(p.words), int(p.t_in * 100)) & 0xFFFF

    # ------------------------------------------------------------ drawing

    def draw(self, c: Canvas, lt: float, pal: Palette, beat_phase: float, bar_phase: float, kick: float,
             energy: float) -> None:
        p = self.active(lt, c.w, c.h)
        if p is None:
            return
        cue = p.cue
        parts = self._parts(p, c.w, c.h)
        if parts is None:
            return
        ink = parts.ink
        age, span = lt - p.t_in, p.t_out - p.t_in
        calm = cue.enter in ("decode", "scan")
        enter_dur = min(0.45 if calm else 0.7, 0.5 * span)
        if cue.mode == "word_flash":
            enter_dur = min(0.18, 0.4 * span)
        lead = p.word_times[0] - p.t_in if p.word_times else PRE
        if lead < PRE * 0.6:                   # came in late (switch_time): assemble fast, the words are due
            enter_dur = min(enter_dur, max(0.2, lead + 0.2))
        leave_dur = min(0.3 if cue.exit == "fade" else 0.45, 0.25 * span)
        drop = STACK_DROP
        if cue.mode == "stack" and p.word_times:   # little room after the last word: every word drops faster
            drop = min(STACK_DROP, max(0.1, p.t_out - max(p.word_times) - LIT_BEFORE_EXIT - 0.1))
        if p.word_times:                       # the exit waits until the last word has been seen lit
            shown = max(p.word_times) + (drop if cue.mode == "stack" else shown_after(cue.mode))
            leave_dur = min(leave_dur, max(0.1, p.t_out - shown - LIT_BEFORE_EXIT))
        v = min(1.0, max(0.0, (lt - (p.t_out - leave_dur)) / leave_dur)) if leave_dur > 0 else 0.0
        fade_in = min(1.0, age / max(0.05, enter_dur * 0.5)) * (1 - v)

        # what happens to the picture behind the letters
        if cue.mode == "takeover":
            self._storm(c, lt, pal, energy, fade=fade_in)
        elif cue.blend == "mask":
            outside = ~ink.mask
            c.dim(0.85 * fade_in, outside)
        elif cue.mode in ("line", "stack") and cue.blend != "knockout":
            x0, y0, x1, y1 = ink.bbox
            halo = np.zeros((c.h, c.w), bool)
            halo[max(0, y0 - 1):y1 + 1, max(0, x0 - 2):x1 + 2] = True
            c.dim(0.9 * (1 - v), halo)
        if cue.blend == "echo":
            prev = self.previous(p, c.w, c.h)
            if prev is not None:
                self._ghost(c, prev, pal)

        tx, ty, k = ink.xs.astype(float), ink.ys.astype(float), parts.k
        times = np.array(p.word_times)
        wi = np.clip(ink.word, 0, len(times) - 1)
        aw = lt - times[wi]                              # age of each cell's word since it was sung
        cur = int(np.searchsorted(times, lt, side="right")) - 1
        sung = aw >= 0

        cell_age = aw + 0.05 if cue.mode == "stack" else age   # stack: every word arrives on its own
        dur = drop if cue.mode == "stack" else enter_dur
        f = motion.enter(cue.enter, tx, ty, ink.order, parts.delay, parts.ox, parts.oy, k, cell_age, dur,
                         c.w, c.h, ink.bbox)
        f = motion.leave(cue.exit, f, tx, ty, k, v, c.w, c.h, ink.bbox)
        landed = f.landed
        visible = f.visible.copy()
        if cue.mode == "stack":
            visible &= sung
        else:
            # unsung: outline only; sung: the inside fills in over 0.12 s. CJK strokes are too thin for an
            # outline (it doubles every stroke): unsung CJK shows whole, just dimmer
            shell = ink.edge | ink.cjk
            visible &= ~landed | shell | (sung & ((k % 1000) / 1000 < np.clip(aw / FILL_IN, 0, 1)))
        visible &= ~(landed & (parts.fill == " "))       # outline fill: the inside stays empty

        frame = int(lt * 24)
        chars = parts.fill.copy()
        if fills.flickers(cue.fill, ink):
            flick = (k + int(lt * 6)) % 7 == 0           # a few resting bits flip, ~6 times a second
            chars = np.where(flick & landed, fills.FILLS["binary"][(k + frame) % 2], chars)
        chars = np.where(landed, chars, ALPHABET[(k + frame) % len(ALPHABET)])
        if f.glyph is not None:
            chars = np.where(f.glyph != "", f.glyph, chars)

        level = np.where(landed, np.where(sung, 0.9, 0.38), np.where(k % 3 == 0, 0.7, 0.45)) * f.level
        accent = landed & sung & (wi == cur)
        chars = np.where(accent & (aw < 0.15), "#", chars)
        if v == 0 and cue.mode != "word_flash":
            x0, _, x1, _ = ink.bbox
            sweep = x0 - 6 + bar_phase * (x1 - x0 + 12)
            hit = landed & sung & (np.abs(tx - sweep) < 2)
            chars = np.where(hit, "#", chars)
            accent = accent | hit
        level = np.clip(level + 0.25 * kick * math.exp(-beat_phase * 4), 0, 1)

        X, Y = np.rint(f.x).astype(int), np.rint(f.y).astype(int)
        ok = visible & (X >= 0) & (X < c.w) & (Y >= 0) & (Y < c.h)
        X, Y, level, accent, chars, landed_ok, edge_ok = (X[ok], Y[ok], level[ok], accent[ok], chars[ok],
                                                          landed[ok], ink.edge[ok])
        bg = np.array(pal.bg, float)
        col = np.where(accent[:, None], np.array(pal.accent, float), np.array(pal.ui, float))
        rgb = (bg + (col - bg) * level[:, None]).astype(np.uint8)

        if cue.blend == "knockout":
            # holes in the picture: resting cells go blank, the rim stays as a thin outline
            hole = landed_ok & ~edge_ok
            c.ch[Y[hole], X[hole]] = " "
            c.bg[Y[hole], X[hole]] = pal.bg
            rim = landed_ok & edge_ok
            c.ch[Y[rim], X[rim]] = "#"                  # a bright rim: the holes alone vanish in a busy picture
            rim_lev = np.maximum(level[rim], 0.8)[:, None]
            rim_col = np.where(accent[rim][:, None], np.array(pal.accent, float), np.array(pal.ui, float))
            c.fg[Y[rim], X[rim]] = (bg + (rim_col - bg) * rim_lev).astype(np.uint8)
            fly = ~landed_ok
            X, Y, chars, rgb, accent = X[fly], Y[fly], chars[fly], rgb[fly], accent[fly]
            level = level[fly]
        elif cue.blend == "mask":
            # the letters are windows onto the picture: keep its glyphs where they carry some ink. Thin ones
            # (". ' ^" of a sparse or dark picture) would dissolve the letters into it, and the rim always keeps
            # the fill so the outline reads
            under = c.ch[Y, X]
            keep = landed_ok & ~edge_ok & ~accent & ~np.isin(under, THIN)
            chars = np.where(keep, under, chars)
        if cue.mode in ("takeover", "word_flash", "embed") and cue.blend in ("overlay", "echo", "obstacle"):
            sx, sy = X + 1, Y + 1                       # drop shadow keeps letters apart from a busy picture
            s_ok = (sx < c.w) & (sy < c.h)
            c.ch[sy[s_ok], sx[s_ok]] = " "
            c.bg[sy[s_ok], sx[s_ok]] = pal.bg
        order = np.argsort(level, kind="stable")          # bright cells win where particles overlap
        c.ch[Y[order], X[order]] = chars[order]
        c.fg[Y[order], X[order]] = rgb[order]
        c.bold[Y[order], X[order]] = accent[order]

    def _ghost(self, c: Canvas, p: Piece, pal: Palette) -> None:
        """A finished piece left behind as faint texture."""
        parts = self._parts(p, c.w, c.h)
        if parts is None:
            return
        ink = parts.ink
        chars = np.where(parts.fill == " ", ".", parts.fill)
        c.ch[ink.ys, ink.xs] = chars
        c.fg[ink.ys, ink.xs] = pal.ui_at(0.16)

    def _storm(self, c: Canvas, lt: float, pal: Palette, energy: float, fade: float) -> None:
        """Takeover background: dim what is there and overrun it with sparse drifting glyphs."""
        c.dim(0.8 * fade)
        if fade <= 0.02:
            return
        density = (0.08 + 0.12 * energy) * fade
        tick = int(lt * 7)
        yy, xx = np.mgrid[0:c.h, 0:c.w]
        k = _u16(xx * 37 + yy * 911 + tick * (3 + xx % 7), 7)
        show = k / 65535 < density
        shift = (np.sin(lt * 4 + np.arange(c.h) * 0.31) * 6).astype(int)
        xs = (xx + shift[:, None]) % c.w
        chars = ALPHABET[(k + int(lt * 24) + (xx // 7) * 13) % len(ALPHABET)]
        lev = np.where(k % 17 == 0, 0.32, 0.16) * fade
        bg = np.array(pal.bg, float)
        rgb = (bg + (np.array(pal.ui, float) - bg) * lev[..., None]).astype(np.uint8)
        c.ch[yy[show], xs[show]] = chars[show]
        c.fg[yy[show], xs[show]] = rgb[show]
        c.bold[yy[show], xs[show]] = False
