"""The terminal player: audio clock -> Renderer -> diffed terminal output, plus keyboard control."""

from __future__ import annotations

import bisect
import time

from ascii_pv.backends.audio_player import open_clock
from ascii_pv.backends.terminal import Terminal, read_keys
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas, crop
from ascii_pv.core.storyboard import Storyboard
from ascii_pv.render import Renderer

MIN_W, MIN_H = 60, 20
HUD_SECONDS = 2.5
HELP = ["CONTROLS",
        "SPACE          play / pause",
        "LEFT / RIGHT   back / forward 5 s",
        "1 - 9          jump to section",
        ", / .          previous / next lyric line",
        "[ / ]          lyrics earlier / later 0.1 s",
        "+ / -          volume",
        "R              restart",
        "H              this help",
        "Q / ESC        quit"]


def _mmss(t: float) -> str:
    return f"{int(t) // 60:02d}:{t % 60:04.1f}"


class Player:
    def __init__(self, analysis: Analysis, board: Storyboard, audio: str | None, fps: int = 30,
                 mute: bool = False, lyric_offset: float = 0.0, stats: bool = False):
        self.a = analysis
        self.renderer = Renderer(analysis, board)
        self.clock, self.note = open_clock(audio, analysis.duration, mute)
        self.fps = fps
        self.offset = lyric_offset
        self.stats = stats
        self.help = False
        self.hud_until = time.perf_counter() + 4.0
        self._render_ms = 0.0
        self._bytes = 0

    # ------------------------------------------------------------ controls

    def _seek(self, t: float) -> None:
        self.clock.seek(max(0.0, min(self.a.duration - 0.05, t)))

    def handle(self, key: str) -> bool:
        """Returns False to quit."""
        k = key.lower()
        now = self.clock.time()
        if k in ("q", "\x1b", "\x03"):
            return False
        if k == " ":
            self.clock.pause() if self.clock.playing else self.clock.play()
        elif key == "LEFT":
            self._seek(now - 5)
        elif key == "RIGHT":
            self._seek(now + 5)
        elif k == "r":
            self._seek(0)
            self.clock.play()
        elif k.isdigit() and k != "0":
            i = int(k) - 1
            if i < len(self.a.sections):
                self._seek(self.a.sections[i].start)
        elif k in ",.":
            starts = [ln.start for ln in self.a.lyrics]
            if starts:
                i = bisect.bisect_right(starts, now + self.offset + 0.05) - 1
                i = max(0, min(len(starts) - 1, i + (1 if k == "." else -1)))
                self._seek(starts[i] - self.offset - 0.4)
        elif k == "[":
            self.offset = round(self.offset + 0.1, 2)
        elif k == "]":
            self.offset = round(self.offset - 0.1, 2)
        elif k in "+=":
            self.clock.set_volume(self.clock.volume + 0.05)
        elif k in "-_":
            self.clock.set_volume(self.clock.volume - 0.05)
        elif k == "h":
            self.help = not self.help
        self.hud_until = time.perf_counter() + HUD_SECONDS
        return True

    # ------------------------------------------------------------ overlays

    def _hud(self, c: Canvas, t: float) -> None:
        pal = self.renderer.palette
        sec = self.a.section_at(t)
        state = "PLAYING" if self.clock.playing else "PAUSED"
        s = (f" {_mmss(t)} / {_mmss(self.a.duration)}  {state}  {sec.label if sec else '-'}  "
             f"{self.a.bpm:.0f} BPM  lyrics {self.renderer.board.lyric_offset + self.offset:+.1f}s  vol {self.clock.volume * 100:.0f}%  H help ")
        if self.stats:
            s += f" render {self._render_ms:.1f}ms  {self._bytes / 1024:.0f}KB "
        if self.note:
            s += f" [{self.note}] "
        c.fill(0, c.h - 1, c.w, 1, " ", bg=pal.bg)
        c.put(1, c.h - 1, crop(s, c.w - 2), pal.ui_at(0.9))

    def _help(self, c: Canvas) -> None:
        pal = self.renderer.palette
        w = min(c.w - 4, max(len(s) for s in HELP) + 6)
        h = len(HELP) + 2
        x, y = (c.w - w) // 2, (c.h - h) // 2
        c.fill(x, y, w, h, " ", bg=pal.bg)
        c.box(x, y, w, h, pal.ui_at(0.9))
        for i, s in enumerate(HELP):
            c.put(x + 3, y + 1 + i, s, pal.accent if i == 0 else pal.ui_at(0.85))

    def _too_small(self, w: int, h: int) -> Canvas:
        pal = self.renderer.palette
        c = Canvas(w, h, bg=pal.bg, fg=pal.ui)
        c.center(h // 2 - 1, "enlarge the window", pal.ui)
        c.center(h // 2 + 1, f"{w} x {h}  (minimum {MIN_W} x {MIN_H})", pal.ui_at(0.5))
        return c

    # ------------------------------------------------------------ loop

    def run(self, start: float = 0.0, stop_after: float | None = None) -> dict:
        """Play until the song ends or the user quits. Returns simple timing stats."""
        frames, worst, total_bytes = 0, 0.0, 0
        self._seek(start)
        self.clock.play()
        period = 1.0 / self.fps
        end_seen = None
        try:
            with Terminal() as term:
                next_frame = time.perf_counter()
                while True:
                    for key in read_keys():
                        if not self.handle(key):
                            return self._report(frames, worst, total_bytes)
                    w, h = term.size()
                    w -= 1   # leave the last column alone: some consoles scroll when it is written
                    t = self.clock.time()
                    t0 = time.perf_counter()
                    if w < MIN_W or h < MIN_H:
                        c = self._too_small(max(1, w), max(1, h))
                    else:
                        c = self.renderer.frame(t, w, h, self.offset)
                        if self.help:
                            self._help(c)
                        if self.help or not self.clock.playing or t0 < self.hud_until:
                            self._hud(c, t)
                    self._render_ms = (time.perf_counter() - t0) * 1000
                    self._bytes = term.draw(c)
                    frames += 1
                    total_bytes += self._bytes
                    worst = max(worst, (time.perf_counter() - t0) * 1000)
                    if stop_after is not None and t >= stop_after:
                        return self._report(frames, worst, total_bytes)
                    # the song ended: hold the last frame a moment, then leave
                    if t >= self.a.duration - 0.06 and not self.clock.playing:
                        end_seen = end_seen or time.perf_counter()
                        if time.perf_counter() - end_seen > 1.5:
                            return self._report(frames, worst, total_bytes)
                    next_frame += period
                    delay = next_frame - time.perf_counter()
                    if delay > 0:
                        time.sleep(delay)
                    else:
                        next_frame = time.perf_counter()   # running late: drop frames, keep the clock
        finally:
            self.clock.close()

    @staticmethod
    def _report(frames: int, worst_ms: float, total_bytes: int) -> dict:
        return {"frames": frames, "worst_frame_ms": round(worst_ms, 1), "bytes": total_bytes}
