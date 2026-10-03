"""HUD: framed panels of maths, code and live analysis readouts in a template's empty areas (M10).

A shot's `hud` ({"panels": ["formula:torus", "code:donut", ...]}) lists panels; the template's
TemplateMeta.hud names a layout (where its picture leaves room: "sides", "top", "corners", "none") and each
panel takes the next slot. Everything is a pure function of the frame: panel contents come from t, the analysis
and the live values the template wrote into ctx.live (e.g. the torus' rotation angles).

Panels draw an opaque box (so they read over busy pictures), unfold their frame in the first 0.4 s of the shot
and decode their text character by character.

  formula:<topic>   pi (Leibniz partial sums converging) · trig · euler · torus (parametric form + live
                    rotation matrix) · fourier (band amplitudes) · mandel (an orbit of z <- z^2 + c)
  plot:<kind>       sin (sin / cos on axes) · lissajous · signal (the last 4 s of kick and bass)
  data              tempo, beat, stem meters, centroid, time code, section, next event
  code:<topic>      donut · fft · life · rain · render (a listing; the line being "run" follows the beat)
  log               an analysis log scrolling with the beats
  hex               a hex dump of the "frame buffer"
"""

from __future__ import annotations

import math

import numpy as np

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas, crop
from ascii_pv.core.context import FrameCtx, hash32, str_seed

# (x0, y0, x1, y1) as fractions of the screen; slots in the upper half come first (lyric shots only use those)
LAYOUTS: dict[str, list[tuple[float, float, float, float]]] = {
    "sides": [(0.01, 0.05, 0.27, 0.47), (0.73, 0.05, 0.99, 0.47), (0.73, 0.53, 0.99, 0.95), (0.01, 0.53, 0.27, 0.95)],
    "top": [(0.01, 0.03, 0.33, 0.40), (0.67, 0.03, 0.99, 0.40), (0.35, 0.03, 0.65, 0.24)],
    "corners": [(0.01, 0.03, 0.25, 0.33), (0.75, 0.03, 0.99, 0.33), (0.75, 0.67, 0.99, 0.97), (0.01, 0.67, 0.25, 0.97)],
    "none": [],
}
UNFOLD = 0.4          # s: the frame draws itself
DECODE = 0.002        # s per character of text decoding after that
GARBLE = "01#%&*+=<>/\\"
PI_DIGITS = ("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798"
             "21480865132823066470938446095505822317253594081284811174502841027019385211055596446229489549303819")

TITLES = {"formula": "f(x)", "plot": "PLOT", "data": "SIGNAL", "code": "SRC", "log": "LOG", "hex": "MEM"}


# ---------------------------------------------------------------- drawing helpers

class Panel:
    """A box on the canvas with a text cursor; text decodes in after the frame has unfolded."""

    def __init__(self, c: Canvas, ctx: FrameCtx, x: int, y: int, w: int, h: int, title: str, salt: int):
        self.c, self.ctx, self.x, self.y, self.w, self.h = c, ctx, x, y, w, h
        self.salt = salt
        self.n = 0                                    # characters written so far (decode order)
        pal = ctx.palette
        u = min(1.0, ctx.local_t / UNFOLD)
        c.fill(x, y, w, h, " ", bg=pal.bg)
        border = pal.ui_at(0.45)
        per = 2 * (w + h - 2)
        shown = int(per * u)
        cells = ([(x + i, y) for i in range(w)] + [(x + w - 1, y + j) for j in range(1, h)]
                 + [(x + w - 1 - i, y + h - 1) for i in range(1, w)] + [(x, y + h - 1 - j) for j in range(1, h - 1)])
        for k, (cx, cy) in enumerate(cells[:shown]):
            corner = {(x, y): "┌", (x + w - 1, y): "┐", (x, y + h - 1): "└", (x + w - 1, y + h - 1): "┘"}
            ch = corner.get((cx, cy), "─" if cy in (y, y + h - 1) else "│")
            c.put(cx, cy, ch, border)
        if u >= 1.0 and w > len(title) + 6:
            c.put(x + 2, y, f"┤{title}├", pal.accent_at(0.85))
        self.ready = u >= 1.0
        self.t0 = ctx.local_t - UNFOLD

    @property
    def inner(self) -> tuple[int, int]:
        return self.w - 4, self.h - 2

    def text(self, row: int, col: int, s: str, level: float = 0.75, accent: bool = False) -> None:
        if not self.ready or row < 0 or row >= self.h - 2:
            return
        s = crop(s, self.w - 4 - col)
        pal = self.ctx.palette
        fg = pal.accent_at(level) if accent else pal.ui_at(level)
        done = int(self.t0 / DECODE) - self.n                  # characters of this string already decoded
        self.n += len(s)
        if done < len(s):
            out = []
            for i, ch in enumerate(s):
                if i < done or ch == " ":
                    out.append(ch)
                else:
                    out.append(GARBLE[hash32(self.salt, self.n + i, self.ctx.frame) % len(GARBLE)])
            s = "".join(out)
        self.c.put(self.x + 2 + col, self.y + 1 + row, s, fg)

    def meter(self, row: int, label: str, v: float, width: int | None = None) -> None:
        w = width or max(4, self.inner[0] - len(label) - 7)
        k = int(round(max(0.0, min(1.0, v)) * w))
        self.text(row, 0, label, 0.55)
        self.text(row, len(label) + 1, "▪" * k + "·" * (w - k), 0.8, accent=v > 0.75)
        self.text(row, len(label) + 2 + w, f"{v:4.2f}", 0.7)


# ---------------------------------------------------------------- panel contents

def _formula(p: Panel, topic: str, ctx: FrameCtx, a: Analysis) -> None:
    t = ctx.t
    live = ctx.live
    if topic == "pi":
        k = int(t * 40) % 4000 + 1
        n = np.arange(k)
        s = 4.0 * float(np.sum((-1.0) ** n / (2 * n + 1)))
        rows = ["π = 4·Σ (-1)^n / (2n+1)", "      n = 0..k", "", f"k   = {k}", f"S_k = {s:.10f}",
                f"err = {abs(s - math.pi):.3e}", ""]
        off = int(t * 6) % (len(PI_DIGITS) - 40)
        rows += [f"π[{off:03d}..] {PI_DIGITS[off:off + 40]}"]
    elif topic == "trig":
        th = (t * 1.3) % math.tau
        rows = [f"θ = {th:6.4f} rad ({math.degrees(th):5.1f}°)", "", f"sin θ = {math.sin(th):+.4f}",
                f"cos θ = {math.cos(th):+.4f}", f"tan θ = {math.tan(th):+.4f}" if abs(math.cos(th)) > 1e-3 else
                "tan θ = ∞", "", f"sin²θ + cos²θ = {math.sin(th) ** 2 + math.cos(th) ** 2:.4f}"]
    elif topic == "euler":
        th = (t * 0.9) % math.tau
        rows = ["e^(iθ) = cos θ + i·sin θ", "", f"θ = {th:.4f}", f"  = {math.cos(th):+.4f} {math.sin(th):+.4f}i",
                f"|z| = {abs(complex(math.cos(th), math.sin(th))):.4f}", "", "e^(iπ) + 1 = 0"]
    elif topic == "torus":
        A, B = live.get("A", t * 0.7), live.get("B", 0.5)
        R, r = live.get("R", 0.7), live.get("r", 0.3)
        ca, sa, cb, sb = math.cos(A), math.sin(A), math.cos(B), math.sin(B)
        m = [[ca, 0, sa], [sa * sb, cb, -ca * sb], [-sa * cb, sb, ca * cb]]
        rows = ["x = (R + r·cos θ)·cos φ", "y = r·sin θ", "z = (R + r·cos θ)·sin φ",
                f"R = {R:.2f}   r = {r:.2f}", f"A = {A % math.tau:.3f}  B = {B % math.tau:.3f}", "Rx(B)·Ry(A) ="]
        rows += [f"│{m[i][0]:+.3f} {m[i][1]:+.3f} {m[i][2]:+.3f}│" for i in range(3)]
    elif topic == "heart":
        u = (t * 1.1) % math.tau
        x = 16 * math.sin(u) ** 3
        y = 13 * math.cos(u) - 5 * math.cos(2 * u) - 2 * math.cos(3 * u) - math.cos(4 * u)
        rows = ["x(u) = 16·sin³u", "y(u) = 13·cos u - 5·cos 2u", "       - 2·cos 3u - cos 4u", "",
                f"u = {u:.4f}", f"x = {x:+8.3f}", f"y = {y:+8.3f}", f"r = {math.hypot(x, y):8.3f}"]
    elif topic == "fourier":
        rows = ["f(t) = Σ a[n]·sin(2π·f[n]·t + φ[n])", ""]
        for name, v in zip(["40", "120", "300", "700", "1.5k", "3k", "6k"], ctx.bands):
            rows.append(f"a({name:>4}Hz) {v:.3f} " + "▪" * int(v * 8))
    else:                                                                   # mandel
        c0 = complex(-0.745 + 0.02 * math.cos(t * 0.3), 0.11 + 0.02 * math.sin(t * 0.37))
        z = 0j
        rows = ["z ← z² + c", f"c = {c0.real:+.4f} {c0.imag:+.4f}i", ""]
        for i in range(6):
            z = z * z + c0
            rows.append(f"z{i + 1} = {z.real:+.4f} {z.imag:+.4f}i  |z|={abs(z):.3f}")
    for i, s in enumerate(rows):
        p.text(i, 0, s, 0.8, accent=("=" in s and i > 0 and any(ch.isdigit() for ch in s)))


def _plot(p: Panel, kind: str, ctx: FrameCtx, a: Analysis) -> None:
    iw, ih = p.inner
    gw, gh = iw - 5, ih - 2
    if gw < 6 or gh < 3:
        return
    c, pal = p.c, ctx.palette
    ox, oy = p.x + 2 + 5, p.y + 1
    # axes
    for j in range(gh):
        c.put(ox - 1, oy + j, "│", pal.ui_at(0.35))
    c.put(ox - 1, oy + gh, "└" + "─" * gw, pal.ui_at(0.35))
    if not p.ready:
        return
    xs = np.linspace(0, 1, gw * 2)
    if kind == "signal":
        rate = a.features["rate"]
        span = 4.0
        times = ctx.t - span + xs * span
        idx = np.clip((times * rate).astype(int), 0, len(a.features["loud"]) - 1)
        src = a.stems or {"kick": a.features["kick"], "bass": a.features["bands"][0]}
        series = [(np.asarray(src["kick"])[idx], "*", pal.accent_at(0.9)), (np.asarray(src["bass"])[idx], "·",
                                                                             pal.ui_at(0.8))]
        p.text(0, 0, " 1.0", 0.5)
        p.text(gh - 1, 0, " 0.0", 0.5)
        p.text(gh + 1, 5, "t-4s     kick *  bass ·", 0.5)
    elif kind == "lissajous":
        u = np.linspace(0, math.tau, gw * 4)
        X = (np.sin(3 * u + ctx.t) + 1) / 2
        Y = (np.sin(2 * u) + 1) / 2
        px = np.clip((X * (gw - 1)).astype(int), 0, gw - 1)
        py = np.clip(((1 - Y) * (gh - 1)).astype(int), 0, gh - 1)
        for x_, y_ in zip(px, py):
            c.put(ox + x_, oy + y_, "•", pal.accent_at(0.85))
        p.text(gh + 1, 5, "x=sin(3u+t)  y=sin(2u)", 0.5)
        return
    else:                                                                   # sin / cos
        ph = ctx.t * 2.0
        series = [((np.sin(xs * math.tau - ph) + 1) / 2, "*", pal.accent_at(0.9)),
                  ((np.cos(xs * math.tau - ph) + 1) / 2, "·", pal.ui_at(0.7))]
        p.text(0, 0, "+1", 0.5)
        p.text(gh // 2, 0, " 0", 0.5)
        p.text(gh - 1, 0, "-1", 0.5)
        p.text(gh + 1, 5, "0      π       2π   sin * cos ·", 0.5)
    for vals, ch, col in series:
        px = np.clip((xs * (gw - 1)).astype(int), 0, gw - 1)
        py = np.clip(((1 - np.clip(vals, 0, 1)) * (gh - 1)).round().astype(int), 0, gh - 1)
        for x_, y_ in zip(px, py):
            c.put(ox + x_, oy + y_, ch, col)


def _data(p: Panel, ctx: FrameCtx, a: Analysis) -> None:
    f = ctx.feat
    beat = ctx.beat_idx % 4
    rows_left = p.inner[1]
    m, s = divmod(ctx.t, 60)
    hz = 300 * (20 ** f.cent)                                   # centroid 0..1 over ~300 Hz .. 6 kHz
    p.text(0, 0, f"T  {int(m):02d}:{s:06.3f}   F {ctx.frame}", 0.8, accent=True)
    p.text(1, 0, f"BPM {ctx.bpm:6.2f}  BEAT " + "".join("■" if i <= beat else "□" for i in range(4)), 0.75)
    p.meter(2, "KICK ", f.drum_kick)
    p.meter(3, "SNARE", f.drum_snare)
    p.meter(4, "BASS ", f.bass)
    p.meter(5, "LOUD ", f.loud)
    p.text(6, 0, f"CENTROID {hz:7.1f} Hz", 0.7)
    sec = ctx.section
    if sec is not None and rows_left > 7:
        p.text(7, 0, f"SECTION {sec.label}{sec.group}  E={sec.energy:.2f}", 0.7)
    if rows_left > 8:
        nxt = next((e for e in a.events if e.get("t", e.get("start", 0)) > ctx.t), None)
        bp = a.build_progress(ctx.t)
        if bp is not None:
            p.text(8, 0, f"BUILD {int(bp * 100):3d}% " + "▲" * int(bp * 8), 0.9, accent=True)
        elif nxt:
            dt = nxt.get("t", nxt.get("start")) - ctx.t
            p.text(8, 0, f"NEXT {nxt['kind'].upper()} in {dt:5.1f}s", 0.7)


CODE = {
    "donut": ["for (θ = 0; θ < 2π; θ += 0.07) {", "  for (φ = 0; φ < 2π; φ += 0.02) {", "    cθ = cos(θ), sθ = sin(θ);",
              "    cφ = cos(φ), sφ = sin(φ);", "    x = (R2 + R1·cθ)·(cB·cφ + sA·sB·sφ)", "        - R1·cA·sB·sθ;",
              "    y = (R2 + R1·cθ)·(sB·cφ - sA·cB·sφ)", "        + R1·cA·cB·sθ;",
              "    z = K2 + cA·(R2 + R1·cθ)·sφ + R1·sA·sθ;", "    ooz = 1/z;",
              "    xp = W/2 + K1·ooz·x;  yp = H/2 - K1·ooz·y;",
              "    L = cφ·cθ·sB - cA·cθ·sφ - sA·sθ", "        + cB·(cA·sθ - cθ·sA·sφ);", "    if (L > 0 && ooz > zbuf[xp][yp]) {",
              "      zbuf[xp][yp] = ooz;", "      out[xp][yp] = \".,-~:;=!*#$@\"[L*8];", "    }", "  }", "}"],
    "fft": ["def fft(x):", "    n = len(x)", "    if n == 1:", "        return x", "    even = fft(x[0::2])",
            "    odd  = fft(x[1::2])", "    w = exp(-2j*pi*arange(n//2)/n)", "    return concat([even + w*odd,",
            "                   even - w*odd])", "", "S = abs(fft(frame * hann(N)))**2", "bands = [S[lo:hi].mean()",
            "         for lo, hi in BANDS]"],
    "life": ["def step(g):", "    n = sum(roll(roll(g, dy, 0), dx, 1)", "            for dy in (-1, 0, 1)",
             "            for dx in (-1, 0, 1)) - g", "    return (n == 3) | (g & (n == 2))", "",
             "g = random(H, W) < density", "while playing:", "    g = step(g)", "    draw(g)"],
    "rain": ["for col in columns:", "    col.y += col.speed * dt", "    if col.y > H + col.len:", "        col.reset()",
             "    for k in range(col.len):", "        ch = GLYPHS[hash(col, k, t)]", "        glow = 1 - k / col.len",
             "        put(col.x, col.y - k, ch, glow)"],
    "render": ["while t < duration:", "    a = analysis.feat(t)", "    shot = board.shot_at(t)",
               "    c = template.draw(shot, a)", "    c = fx(c, kick=a.kick)", "    c = lyrics.draw(c, t)",
               "    out.write(c)", "    t += 1 / fps"],
}


def _code(p: Panel, topic: str, ctx: FrameCtx) -> None:
    lines = CODE.get(topic, CODE["render"])
    rows = p.inner[1]
    cur = ctx.beat_idx % len(lines)
    top = max(0, min(cur - rows // 2, len(lines) - rows)) if len(lines) > rows else 0
    for i in range(min(rows, len(lines))):
        k = top + i
        mark = ">" if k == cur else " "
        p.text(i, 0, f"{k + 1:2d}{mark}{lines[k]}", 0.95 if k == cur else 0.6, accent=k == cur)


def _log(p: Panel, ctx: FrameCtx, a: Analysis) -> None:
    rows = p.inner[1]
    beats = [b for b in a.beats if b <= ctx.t][-rows:]
    rate = a.features["rate"]
    out = []
    for b in beats:
        i = min(len(a.features["loud"]) - 1, int(b * rate))
        h = hash32(int(b * 1000), 77)
        m, s = divmod(b, 60)
        tag = ["onset", "fft", "zbuf", "sync", "beat"][h % 5]
        val = {"onset": f"kick={a.features['kick'][i]:.2f}", "fft": f"peak={40 * 2 ** (h % 8):d}Hz",
               "zbuf": f"px={1200 + h % 900}", "sync": f"dt={(h % 41 - 20) / 1000:+.3f}s",
               "beat": f"E={a.features['loud'][i]:.2f}"}[tag]
        out.append(f"[{int(m):02d}:{s:06.3f}] {tag:<5} {val}")
    for i, s in enumerate(out):
        p.text(i, 0, s, 0.5 + 0.4 * (i + 1) / max(1, len(out)), accent=i == len(out) - 1)


def _hex(p: Panel, ctx: FrameCtx) -> None:
    rows = p.inner[1]
    per = max(2, (p.inner[0] - 8) // 4)
    for i in range(rows):
        addr = (ctx.frame // 2 + i) * per * 16 % 0xFFFF
        bs = [hash32(addr, k, ctx.frame // 3) & 0xFF for k in range(per)]
        asc = "".join(chr(b) if 33 <= b < 127 else "." for b in bs)
        p.text(i, 0, f"{addr:04X} " + " ".join(f"{b:02X}" for b in bs) + " " + asc, 0.55)


# ---------------------------------------------------------------- entry point

def draw_hud(c: Canvas, ctx: FrameCtx, a: Analysis, panels: list[str], layout: str) -> None:
    slots = LAYOUTS.get(layout, [])
    for k, (spec, (x0, y0, x1, y1)) in enumerate(zip(panels, slots)):
        x, y = int(x0 * c.w), int(y0 * c.h)
        w, h = int(x1 * c.w) - x, int(y1 * c.h) - y
        if w < 16 or h < 6:
            continue
        kind, _, topic = spec.partition(":")
        title = TITLES.get(kind, kind.upper()) + (f":{topic}" if topic else "")
        p = Panel(c, ctx, x, y, w, h, title.upper(), str_seed(spec) + k)
        if kind == "formula":
            _formula(p, topic or "pi", ctx, a)
        elif kind == "plot":
            _plot(p, topic or "sin", ctx, a)
        elif kind == "data":
            _data(p, ctx, a)
        elif kind == "code":
            _code(p, topic or "render", ctx)
        elif kind == "log":
            _log(p, ctx, a)
        elif kind == "hex":
            _hex(p, ctx)
