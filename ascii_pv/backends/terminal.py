"""Terminal output (diffed 24-bit ANSI) and keyboard input.

Only cells that changed since the last frame are written: rows are scanned for changed runs (small unchanged
gaps are rewritten rather than paying for a cursor move), colours are emitted only when they change, and the
whole frame goes out in one write. Keyboard: msvcrt on Windows (POSIX is a later milestone).
"""

from __future__ import annotations

import os
import sys

import numpy as np

from ascii_pv.backends.winconsole import setup_console
from ascii_pv.core.canvas import Canvas

GAP = 4   # unchanged cells tolerated inside one run


class Terminal:
    def __init__(self):
        self.vt = setup_console()
        self.out = sys.stdout.buffer
        self._prev: tuple | None = None
        self._size: tuple[int, int] | None = None

    def __enter__(self) -> "Terminal":
        # alternate screen, hide cursor, no autowrap, clear
        self.out.write(b"\x1b[?1049h\x1b[?25l\x1b[?7l\x1b[2J")
        self.out.flush()
        return self

    def __exit__(self, *exc) -> None:
        self.out.write(b"\x1b[0m\x1b[?7h\x1b[?25h\x1b[?1049l")
        self.out.flush()

    def size(self) -> tuple[int, int]:
        try:
            w, h = os.get_terminal_size(sys.stdout.fileno())
        except OSError:
            w, h = 120, 40
        return w, h

    def invalidate(self) -> None:
        self._prev = None

    def draw(self, c: Canvas) -> int:
        """Write the frame; returns the number of bytes sent."""
        if self._size != (c.w, c.h):
            self._size = (c.w, c.h)
            self._prev = None
            self.out.write(b"\x1b[0m\x1b[2J")
        if self._prev is None:
            changed = np.ones((c.h, c.w), bool)
        else:
            pch, pfg, pbg, pbold = self._prev
            changed = (c.ch != pch) | np.any(c.fg != pfg, axis=2) | np.any(c.bg != pbg, axis=2) | (c.bold != pbold)
        self._prev = (c.ch.copy(), c.fg.copy(), c.bg.copy(), c.bold.copy())

        parts: list[str] = []
        # track fg, bg and bold separately: the background rarely changes, so most cells need only an fg code
        cur_fg = cur_bg = cur_b = None
        rows = np.nonzero(changed.any(axis=1))[0]
        if len(rows):
            chars = c.ch.tolist()
            fg = c.fg.tolist()
            bg = c.bg.tolist()
            bold = c.bold.tolist()
        for y in rows:
            xs = np.nonzero(changed[y])[0]
            # group changed columns into runs, bridging small gaps
            runs, start, prev = [], xs[0], xs[0]
            for x in xs[1:]:
                if x - prev > GAP:
                    runs.append((start, prev))
                    start = x
                prev = x
            runs.append((start, prev))
            row_ch, row_fg, row_bg, row_b = chars[y], fg[y], bg[y], bold[y]
            for a, b in runs:
                # never start a run on the continuation half of a wide char
                while a > 0 and row_ch[a] == "":
                    a -= 1
                parts.append(f"\x1b[{y + 1};{a + 1}H")
                for x in range(a, b + 1):
                    ch = row_ch[x]
                    if ch == "":
                        continue
                    f, b_, bo = row_fg[x], row_bg[x], row_b[x]
                    if bo != cur_b:
                        parts.append("\x1b[1m" if bo else "\x1b[22m")
                        cur_b = bo
                    if b_ != cur_bg:
                        parts.append(f"\x1b[48;2;{b_[0]};{b_[1]};{b_[2]}m")
                        cur_bg = b_
                    if f != cur_fg:
                        parts.append(f"\x1b[38;2;{f[0]};{f[1]};{f[2]}m")
                        cur_fg = f
                    parts.append(ch)
        if not parts:
            return 0
        data = "".join(parts).encode("utf-8")
        self.out.write(data)
        self.out.flush()
        return len(data)


# ---------------------------------------------------------------- keyboard

def read_keys() -> list[str]:
    """Non-blocking: every key pressed since the last call. Arrows come back as LEFT/RIGHT/UP/DOWN."""
    keys: list[str] = []
    if sys.platform != "win32":
        return keys
    import msvcrt

    while msvcrt.kbhit():
        ch = msvcrt.getwch()
        if ch in ("\x00", "\xe0"):
            code = msvcrt.getwch()
            keys.append({"K": "LEFT", "M": "RIGHT", "H": "UP", "P": "DOWN"}.get(code, ""))
        else:
            keys.append(ch)
    return [k for k in keys if k]
