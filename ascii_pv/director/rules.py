"""The Director's rules, kept apart from the planning loop so they are easy to read and tune.

  palette_for       style -> palette
  shot_bars         how many bars a shot lasts in a section
  score_template    how well a template suits a shot (style, section, lyric motifs, variety)
  sample_params     per-template parameter variation
  transition_for    cut type between two shots
  hud_for           maths / code / readout panels around the picture
  fx_for            glitch / scan / camera amounts (+ the M10 drop shockwave)
  lyric_cue_for     how each lyric line is lettered
"""

from __future__ import annotations

import bisect
import math
import random
from collections import Counter

from ascii_pv.core.analysis import Analysis, LyricLine, Section
from ascii_pv.core.cjk import has_cjk, word_count
from ascii_pv.core.storyboard import LyricCue
from ascii_pv.templates.base import TemplateMeta
from ascii_pv.templates.particles import MOTIF_KIND

# ---------------------------------------------------------------- palette


def palette_for(style: dict) -> str:
    e, v = style.get("energy", 0.5), style.get("valence", 0.5)
    if v >= 0.6:
        return "synthwave" if e >= 0.6 else "phosphor"
    if v <= 0.4:
        return "amber" if e >= 0.5 else "deepsea"
    return "amber" if e >= 0.6 else "deepsea"


# ---------------------------------------------------------------- cutting


def shot_bars(sec: Section, bpm: float) -> int:
    """Bars per shot: busy, loud sections cut faster."""
    if sec.label in ("chorus", "prechorus") or sec.energy >= 0.7:
        bars = 4
    elif sec.label in ("intro", "outro") or sec.energy < 0.35:
        bars = 8
    else:
        bars = 8 if bpm < 110 else 4
    return bars


# ---------------------------------------------------------------- template choice


def _fit(x: float, lo_hi: tuple[float, float]) -> float:
    lo, hi = lo_hi
    if lo <= x <= hi:
        return 1.0
    d = lo - x if x < lo else x - hi
    return math.exp(-d * 6)


def score_template(meta: TemplateMeta, *, sec: Section, style: dict, motifs: Counter, history: list[str],
                   group_choice: str | None, has_image: bool = False) -> float:
    if meta.needs_image and not has_image:
        return -math.inf
    if meta.name == "blank" or meta.easter_egg:
        return -math.inf
    energy = 0.6 * sec.energy + 0.4 * style.get("energy", 0.5)
    s = 1.2 * _fit(energy, meta.energy) + 0.6 * _fit(style.get("valence", 0.5), meta.valence)
    if meta.sections:
        s += 1.0 if sec.label in meta.sections else 0.0
    else:
        s += 0.4
    s += 0.25 * len(set(style.get("tags", [])) & meta.tags)
    hits = sum(n for m, n in motifs.items() if m in meta.motifs)
    s += 0.9 * min(1.0, hits / 2)
    if history and history[-1] == meta.name:
        s -= 1.5
    elif meta.name in history[-3:]:
        s -= 0.5
    s -= 0.45 * history.count(meta.name)       # spread the song over many looks
    if group_choice == meta.name:
        s += 1.6          # a repeated section keeps its look
    return s


EGG_CHANCE = 0.5          # share of songs that get one easter-egg shot


def pick(scores: dict[str, float], rng: random.Random, temperature: float = 0.35) -> str:
    names = [n for n, v in scores.items() if v > -math.inf]
    top = max(scores[n] for n in names)
    weights = [math.exp((scores[n] - top) / temperature) for n in names]
    return rng.choices(names, weights)[0]


def sample_params(name: str, rng: random.Random, sec: Section, motifs: Counter,
                  style_tags: tuple[str, ...] = ()) -> dict:
    if name == "wireframe_mesh":
        return {"form": rng.choice(["torus", "cube", "sphere"]), "spin": round(rng.uniform(0.35, 0.8), 2)}
    if name == "glyph_storm":
        return {"density": round(0.12 + 0.14 * sec.energy + rng.uniform(-0.03, 0.03), 3)}
    if name == "tunnel":
        return {"sides": rng.choice([0, 0, 4, 6]), "twist": round(rng.uniform(0.3, 1.0), 2)}
    if name == "matrix_rain":
        return {"density": round(rng.uniform(0.4, 0.7), 2), "speed": round(rng.uniform(0.8, 1.3), 2)}
    if name == "particles":
        kinds = [MOTIF_KIND[m] for m, _ in motifs.most_common() if m in MOTIF_KIND]
        kind = kinds[0] if kinds else rng.choice(["embers", "snow", "stars", "bubbles"])
        return {"kind": kind, "count": round(0.04 + 0.05 * sec.energy, 3)}
    if name == "image_ascii":
        tags = set(style_tags)
        modes = ["halfblock", "halfblock", "glyph"] if "energetic" in tags else ["halfblock", "ramp", "halfblock"]
        if "dark" in tags:
            modes += ["edge", "glyph"]
        return {"mode": rng.choice(modes), "crop": rng.choice(["full", "bust", "bust", "face"]),
                "reveal": rng.choice(["scan", "dissolve"]), "react": round(0.6 + 0.6 * sec.energy, 2)}
    if name == "oscilloscope":
        return {"shape": rng.choice(["wave", "lissajous", "radial"]), "trail": rng.choice([4, 6, 8])}
    if name == "starfield":
        return {"count": rng.choice([800, 1100, 1400]), "speed": round(0.7 + 0.8 * sec.energy, 2)}
    if name == "plasma":
        return {"scale": round(rng.uniform(0.7, 1.4), 2)}
    if name == "flames":
        return {"height": round(0.5 + 0.35 * sec.energy, 2)}
    if name == "life_grid":
        return {"density": round(rng.uniform(0.2, 0.35), 2)}
    if name == "ridgelines":
        return {"amp": round(0.8 + 0.5 * sec.energy, 2), "step": rng.choice([0.08, 0.12, 0.16])}
    if name == "rain":
        return {"density": round(0.35 + 0.4 * sec.energy, 2), "wind": rng.choice([0.0, 0.25, 0.4, 0.55])}
    if name == "ocean":
        return {"moon": rng.choice([0.25, 0.7, 0.75]), "swell": round(0.7 + 0.6 * sec.energy, 2)}
    if name == "fireworks":
        return {"rate": round(0.3 + 0.35 * sec.energy, 2), "size": round(rng.uniform(0.8, 1.2), 2)}
    if name == "helix":
        return {"turns": rng.choice([1.6, 2.2, 2.8]), "rungs": rng.choice([3, 4])}
    if name == "vortex":
        return {"arms": rng.choice([2, 3, 4, 5]), "twist": round(rng.uniform(3.5, 7.0), 1)}
    if name == "kaleidoscope":
        return {"wedges": rng.choice([6, 6, 8]), "zoom": round(rng.uniform(0.8, 1.3), 2)}
    if name == "radar":
        return {"contacts": rng.choice([6, 9, 12]), "rings": rng.choice([3, 4])}
    if name == "tetris":
        return {"every": 2 if sec.energy < 0.45 else 1}
    if name == "invaders":
        return {"rows": rng.choice([4, 5]), "cols": rng.choice([9, 11])}
    if name == "pong":
        return {"every": 2 if sec.energy < 0.4 else 1, "miss": rng.choice([9, 13, 17])}
    if name == "runner":
        return {"speed": round(0.8 + 0.4 * sec.energy, 2)}
    if name == "shell_typing":
        return {"rate": 1 if sec.energy < 0.3 else 2}
    if name == "spectrum_bars":
        return {"height": round(0.4 + 0.4 * sec.energy, 2), "mirror": rng.random() < 0.7}
    return {}


# ---------------------------------------------------------------- HUD panels


def hud_for(meta: TemplateMeta, free: bool, rng: random.Random) -> list[str]:
    """Panels of maths / code / readouts around the picture (templates/hud.py). The template's signature panel
    comes first; shots with lyrics get one or two, in the upper half only, so the lettering keeps its room."""
    from ascii_pv.templates.hud import LAYOUTS

    slots = LAYOUTS.get(meta.hud, [])
    if not slots or not meta.hud_topics:
        return []
    upper = sum(1 for s in slots if s[3] <= 0.5)
    n = min(len(slots), 4, len(meta.hud_topics) + 1) if free else min(upper, 1 if meta.hud == "corners" else 2)
    rest = list(meta.hud_topics[1:])
    rng.shuffle(rest)
    pool = [meta.hud_topics[0], *rest] + [p for p in ("data", "log", "hex") if p not in meta.hud_topics]
    return pool[:n]


# ---------------------------------------------------------------- transitions & fx


def transition_for(prev: Section | None, sec: Section, first_in_section: bool, index: int,
                   rng: random.Random | None = None, at_drop: bool = False) -> str:
    """Hard cuts inside a section, a hit into the chorus, softer moves elsewhere; `rng` picks among fitting ones.
    A drop always hits with a flash."""
    rng = rng or random.Random(index)
    if index == 0:
        return "crt_on"
    if at_drop:
        return "flash"
    if not first_in_section:
        return rng.choice(["glitch", "blocks", "cut"]) if sec.energy >= 0.75 else rng.choice(["cut", "cut", "blocks"])
    if sec.label == "chorus":
        return rng.choice(["flash", "flash", "zoom"])
    if prev is not None and prev.energy < 0.3 and sec.energy > prev.energy + 0.2:
        return "crt_on"
    if prev is not None and prev.label == "chorus":
        return rng.choice(["wipe", "slide"])
    return rng.choice(["dissolve", "reflow", "blocks"])


def sprite_overlay(has_image: bool, style: dict) -> dict:
    """The character joins the choruses (alternating sides); a low glitch keeps it part of the picture."""
    if not has_image:
        return {"enabled": False}
    return {"enabled": True, "asset": "character", "crop": "bust", "mode": "halfblock", "color": "source",
            "align": "alternate", "scale": 0.85, "sections": ["chorus"], "reveal": "scan", "react": 1.0}


def fx_for(sec: Section, style: dict, progress: float, shockwave: bool = False) -> dict:
    """Glitch grows with loudness and, a little, with how far into the song we are; the camera effects (zoom,
    shake, strobe) belong to loud sections, split to dark ones, analog noise to the quiet edges of the song.
    shockwave: the shot starts at a drop with no lyric on screen (render.Renderer._shockwave)."""
    tags = set(style.get("tags", []))
    loud = sec.label == "chorus" or sec.energy >= 0.72
    g = max(0.0, sec.energy - 0.55) * 0.9 + (0.12 if sec.label == "chorus" else 0.0) + 0.1 * progress
    if "dark" in tags:
        g *= 1.2
    fx = {"glitch": round(min(0.8, g), 3), "scan": 0.35 if style.get("valence", 0.5) < 0.5 else 0.15}
    if loud:
        fx["zoom"] = 0.6 if sec.label == "chorus" else 0.35
        fx["shake"] = 0.7 if sec.energy >= 0.75 else 0.4
        if "energetic" in tags:
            fx["strobe"] = 0.5
    elif sec.label == "prechorus":
        fx["zoom"] = 0.3
    if "dark" in tags and sec.energy >= 0.55:
        fx["split"] = 0.45 if loud else 0.25
    elif "percussive" in tags and loud:
        fx["split"] = 0.3
    if sec.label in ("intro", "outro") or sec.energy < 0.3:
        fx["noise"] = 0.35 if "dark" in tags else 0.2
    if shockwave:
        fx["shockwave"] = 1.0
    return fx


# ---------------------------------------------------------------- lyrics

DENSE_WPS = 2.8          # words per second above which a line is "rap speed"


def words_per_second(ln: LyricLine) -> float:
    return word_count(ln.text) / max(0.3, ln.end - ln.start)


def fonts_for(style: dict, rng: random.Random) -> tuple[str, str]:
    """(font for big moments, font for the running lines), from the song's style."""
    tags = set(style.get("tags", []))
    e, v = style.get("energy", 0.5), style.get("valence", 0.5)
    if "calm" in tags or e < 0.4:
        big = rng.choice(["serif", "script"])
        return big, rng.choice(["serif", "bit5x5"])
    if "dark" in tags or v < 0.4:
        return rng.choice(["impact", "black"]), rng.choice(["bit5x5", "impact"])
    return rng.choice(["black", "impact", "bit5x5"]), rng.choice(["bit5x5", "black"])


def section_look(sec: Section, style: dict, rng: random.Random) -> dict:
    """Fill / enter / exit shared by every line of one repeat group (a chorus keeps its lettering)."""
    loud = sec.label in ("chorus", "prechorus") or sec.energy >= 0.65
    dark = style.get("valence", 0.5) < 0.42
    if loud:
        return {"fill": rng.choice(["binary", "binary", "hex", "ramp"]),
                "enter": rng.choice(["assemble", "assemble", "glitch_in", "rain"]),
                "exit": rng.choice(["explode", "shatter"] + (["melt"] if dark else []))}
    return {"fill": rng.choice(["binary", "self", "block", "ramp"]),
            "enter": rng.choice(["decode", "scan", "rain"]),
            "exit": rng.choice(["fade", "crt_off"] + (["melt"] if dark else ["explode"]))}


def lettering(a: Analysis, shots: list, style: dict, rng: random.Random) -> list[LyricCue]:
    """One LyricCue per lyric line."""
    from ascii_pv.templates import REGISTRY

    big_font, line_font = fonts_for(style, rng)
    looks: dict[str, dict] = {}
    takeovers: dict[float, int] = {}          # section start -> takeovers used (keep them special)
    starts = [s.start for s in shots]
    cues = []
    for i, ln in enumerate(a.lyrics):
        sec = a.section_at(ln.start + 0.01) or Section(0.0, a.duration, "verse", 0.5, "-")
        look = looks.setdefault(sec.group or sec.label, section_look(sec, style, rng))
        shot = shots[max(0, bisect.bisect_right(starts, ln.start) - 1)] if shots else None
        blends = REGISTRY[shot.template].meta.lyric_blends if shot and shot.template in REGISTRY else frozenset()
        n_words, wps = word_count(ln.text), words_per_second(ln)     # CJK: two characters ~ one word
        before = [o for o in a.lyrics[:i] if sec.start <= o.start]
        first_in_chorus = sec.label == "chorus" and not before
        repeat_in_section = any(o.text.lower() == ln.text.lower() for o in before)
        short_hook = ln.hook and n_words <= 5 and sec.label in ("chorus", "prechorus", "bridge")

        mode, blend, font = "line", "overlay", line_font
        enter, exit_, fill = look["enter"], look["exit"], look["fill"]
        if short_hook and repeat_in_section:
            mode, font = "word_flash", big_font                  # the hook again: hammer it word by word
        elif first_in_chorus or (i == 0 and n_words <= 6) or (short_hook and not takeovers.get(sec.start)):
            mode, font = "takeover", big_font
            takeovers[sec.start] = takeovers.get(sec.start, 0) + 1
        elif wps < 1.3 and sec.energy < 0.55 and n_words <= 8:
            mode, font = "stack", big_font                       # slow line: words pile up as sung
        elif "mask" in blends and n_words <= 6 and rng.random() < 0.35:
            mode, blend, font = "embed", "mask", big_font        # letters as windows onto a busy picture
        elif "knockout" in blends and n_words <= 5 and not has_cjk(ln.text) and rng.random() < 0.25:
            # (holes with a rim need thick strokes; CJK strokes are one or two cells)
            mode, blend, font = "embed", "knockout", big_font
        elif sec.label == "verse" and style.get("valence", 0.5) < 0.45 and rng.random() < 0.35:
            blend = "echo"
        if wps >= DENSE_WPS and mode in ("line", "embed"):
            enter, exit_ = "decode", "fade"                       # rap speed: no flying letters
        if mode == "word_flash":
            enter, exit_ = "glitch_in", "fade"
        if blend == "mask":
            fill = "solid"                                        # the picture provides the texture
        cues.append(LyricCue(line=i, mode=mode, font=font, fill=fill, enter=enter, exit=exit_, blend=blend))
    return cues
