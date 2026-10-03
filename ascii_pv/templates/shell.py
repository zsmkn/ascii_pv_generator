"""shell_typing (M8): someone at a terminal, analysing this very song. Keystrokes land on 16th notes (long file
names are tab-completed), Enter on the beat, and the output (the song's real facts: duration, tempo, key, sections, drops, lyrics) appears at once and
scrolls up. The session is scripted once per shot in __init__; a frame replays the events up to t.
"""

from __future__ import annotations

import bisect
import re
from pathlib import Path

from ascii_pv.core.canvas import Canvas, crop
from ascii_pv.core.context import FrameCtx
from ascii_pv.templates.base import Template, TemplateMeta, register
from ascii_pv.templates.retro import beat_grid


def _mmss(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:05.2f}"


@register
class ShellTyping(Template):
    meta = TemplateMeta(
        "shell_typing", "A shell session analysing the song itself: keys on 16th notes, Enter on the beat.",
        busy=0.4, energy=(0.0, 0.75), sections=frozenset({"intro", "verse", "bridge", "outro", "instrumental"}),
        tags=frozenset({"dark", "calm", "neutral", "mid-energy", "sparse"}), motifs=frozenset({"pulse", "time"}),
        lyric_blends=frozenset({"overlay"}),
    )
    defaults = {"rate": 2}

    def __init__(self, shot, analysis, assets=None):
        super().__init__(shot, analysis, assets)
        a = analysis
        src = a.source.get("audio") or "track.mp3"
        name = Path(str(src).replace("\\", "/")).name
        stem = Path(name).stem
        self.slug = re.sub(r"[^a-z0-9]+", "-", stem.lower()).strip("-")[:24] or "song"
        self.grid = beat_grid(a, shot.start, shot.end + 2, 4)                 # 16th notes
        self.events: list[tuple[float, str, object]] = []
        self.times: list[float] = []
        rate = max(1, int(self.params["rate"]))
        g = 0

        def at(i: int) -> float:
            return self.grid[min(i, len(self.grid) - 1)]

        def command(cmd: str, out: list) -> None:
            nonlocal g
            self._add(at(g), "prompt", None)
            g += 2
            # long file names are tab-completed: three keys, then the rest at once
            head, sep, tail = cmd.partition(name) if len(name) > 8 else (cmd, "", "")
            keys = [head[k:k + rate] for k in range(0, len(head), rate)]
            if sep:
                keys += [name[:3][k:k + rate] for k in range(0, 3, rate)] + [name[3:]]
                keys += [tail[k:k + rate] for k in range(0, len(tail), rate)]
            for key in keys:
                if key:
                    self._add(at(g), "keys", key)
                    g += 1
            g = (g + 3) // 4 * 4                                              # Enter on the next beat
            self._add(at(g), "enter", None)
            for line in out:
                if isinstance(line, tuple):                                    # ("bar", label, beats)
                    _, label, beats = line
                    self._add(at(g), "bar", (label, at(g), at(g + beats * 4)))
                    g += beats * 4
                else:
                    self._add(at(g), "out", line)
                    g += 1
            g += 4

        size = 0
        try:
            size = Path(src).stat().st_size
        except OSError:
            size = int(a.duration * 16000)
        secs = " ".join(dict.fromkeys(s.label for s in a.sections)) or "-"
        drops = ", ".join(_mmss(e["t"]) for e in a.events if e["kind"] == "drop")[:60] or "none"
        files = [f"-rw-r--r-- 1 pv pv {size:>9} Oct  2 04:12 {name}"]
        if a.lyrics:
            files.append(f"-rw-r--r-- 1 pv pv {sum(len(ln.text) + 10 for ln in a.lyrics):>9} Oct  2 04:12 lyrics.lrc")
        files += ["-rw-r--r-- 1 pv pv    807523 Oct  2 04:13 analysis.json",
                  "-rw-r--r-- 1 pv pv     18336 Oct  2 04:13 storyboard.json"]
        h, rem = divmod(a.duration, 3600)
        script = [
            (f"cd ~/pv/{self.slug}", []),
            ("ls -la", [f"total {len(files) * 4}"] + files),
            (f"ffprobe -hide_banner {name}", [
                f"Input #0, mp3, from '{name}':",
                f"  Duration: {int(h):02d}:{int(rem // 60):02d}:{rem % 60:05.2f}, start: 0.000000, bitrate: 320 kb/s",
                "  Stream #0:0: Audio: mp3, 44100 Hz, stereo, fltp, 320 kb/s"]),
            (f"ascii-pv analyze {name}", [
                f"decoding {name} ...", ("bar", "beats   ", 2), ("bar", "demucs  ", 2),
                f"tempo    {a.bpm:.1f} bpm   key {a.style.get('key', '?')}   beats {len(a.beats)}",
                f"sections {secs}", f"drops    {drops}",
                f"style    {', '.join(a.style.get('tags', [])[:4]) or '-'}"]),
        ]
        if a.lyrics:
            script.append(("head -n 3 lyrics.lrc", [f"[{_mmss(ln.start)}]{ln.text}" for ln in a.lyrics[:3]]))
        script += [
            (f"python3 -c \"print(60 / {a.bpm:.2f})\"", [f"{60 / max(1.0, a.bpm):.6f}"]),
            ("uptime", [f" 04:13:07 up {int(shot.start) // 60} min,  1 user,  load average: "
                        f"{a.style.get('energy', 0.5):.2f}, {a.style.get('density', 0.5):.2f}, 0.42"]),
        ]
        k = 0
        while g < len(self.grid) - 8 and k < 40:                               # loop the session if the shot is long
            cmd, out = script[k % len(script)]
            command(cmd, out)
            k += 1
        self.events.sort(key=lambda e: e[0])
        self.times = [e[0] for e in self.events]

    def _add(self, t: float, kind: str, payload) -> None:
        self.events.append((t, kind, payload))

    def draw(self, c: Canvas, ctx: FrameCtx) -> None:
        pal = ctx.palette
        n = bisect.bisect_right(self.times, ctx.t)
        lines: list[list[tuple[str, tuple]]] = []
        typing = False
        user, path, dim, text = pal.anom_at(0.9), pal.accent_at(0.85), pal.ui_at(0.55), pal.ui_at(0.9)
        for t, kind, payload in self.events[:n]:
            if kind == "prompt":
                lines.append([("pv@ascii", user), (":", dim), (f"~/pv/{self.slug}", path), ("$ ", dim)])
                typing = True
            elif kind == "keys":
                lines[-1].append((payload, text))
            elif kind == "enter":
                typing = False
            elif kind == "out":
                lines.append([(payload, dim)])
            elif kind == "bar":
                label, t0, t1 = payload
                u = min(1.0, max(0.0, (ctx.t - t0) / max(1e-3, t1 - t0)))
                width = 30
                done = int(u * width)
                lines.append([(label, dim), ("[" + "#" * done + "." * (width - done) + "]", pal.accent_at(0.8)),
                              (f" {int(u * 100):3d}%", dim)])
        x0, y0 = 2, 1
        rows = max(1, c.h - 2)
        shown = lines[-rows:]
        for i, segs in enumerate(shown):
            x = x0
            for s, col in segs:
                s = crop(s, c.w - x - 1)
                c.put(x, y0 + i, s, col)
                x += len(s)
            if i == len(shown) - 1:
                blink = typing or ctx.beat_phase < 0.5                          # solid while typing
                if blink and x < c.w - 1:
                    c.put(x, y0 + i, "█", pal.ui_at(0.8))
