"""plan(analysis) -> Storyboard.

1. Cut each section into shots of `shot_bars` bars on the downbeat grid (a short remainder joins the last shot).
2. For each shot, score every template against the section, the song's style, the lyric motifs sung in the shot
   and recent history; draw one with a seeded softmax. Sections of the same repeat group reuse the template and
   parameters chosen the first time that group appeared, so a chorus looks like the chorus each time.
3. Transitions and FX from the section boundaries and energy; lyric cues line by line.
   M10: drops always start a new shot; with no lyric on screen it opens on a shockwave. Shots with lyrics get
   fewer HUD panels (only the plain templates have them at all).
4. `lock`: shots and cues marked locked in a previous storyboard are kept; generated shots are cut around them.
"""

from __future__ import annotations

import random
from collections import Counter

from ascii_pv.core.analysis import Analysis, Section
from ascii_pv.core.context import hash32, str_seed
from ascii_pv.core.storyboard import Shot, Storyboard
from ascii_pv.director import rules
from ascii_pv.templates import REGISTRY

MIN_SHOT = 2.0   # seconds


def _free(a: Analysis, t0: float, t1: float) -> bool:
    """No lyric line on screen anywhere in [t0, t1)."""
    return not any(ln.start < t1 and ln.end > t0 for ln in a.lyrics)


def _cuts(sec: Section, a: Analysis) -> list[float]:
    """Shot boundaries inside a section: every `shot_bars` bars on the downbeat grid, and always at a drop."""
    bar = 240.0 / a.bpm
    forced = sorted(e["t"] for e in a.events if e["kind"] == "drop" and sec.start + bar <= e["t"] <= sec.end - bar)
    forced = [t for k, t in enumerate(forced) if k == 0 or t - forced[k - 1] >= bar]
    out = [sec.start]
    for s0, s1 in zip([sec.start, *forced], [*forced, sec.end]):
        grid = [d for d in a.downbeats if s0 + 0.5 * bar < d < s1 - 0.5 * bar]
        last = s0
        for d in grid:
            step = rules.shot_bars(sec, a.bpm) * bar
            if d - last >= step - 0.25 * bar and s1 - d >= min(step * 0.5, MIN_SHOT * 2):
                out.append(d)
                last = d
        out.append(s1)
    return out


def plan(a: Analysis, seed: int = 0, palette: str | None = None, lock: Storyboard | None = None,
         assets: dict | None = None) -> Storyboard:
    """assets: {"character": {"full": png, "bust": png, "face": png}} from analysis/image.prepare (or the lock's)."""
    assets = assets if assets is not None else (lock.assets if lock else {})
    has_image = bool(assets.get("character"))
    rng = random.Random(hash32(seed, 0x5EED))
    style = a.style
    sections = a.sections or [Section(0.0, a.duration, "verse", 0.5, "A")]
    locked = sorted([s for s in (lock.shots if lock else []) if s.locked], key=lambda s: s.start)

    shots: list[Shot] = []
    history: list[str] = []
    group_pick: dict[str, tuple[str, dict]] = {}   # "<group>:<shot # in section>" -> (template, params)
    n = 0
    for si, sec in enumerate(sections):
        bounds = _cuts(sec, a)
        for j, (t0, t1) in enumerate(zip(bounds, bounds[1:])):
            if t1 - t0 < 0.05:
                continue
            motifs = Counter(m for ln in a.lyrics if t0 <= ln.start < t1 for m in ln.motifs)
            free = _free(a, t0, t1)
            at_drop = any(e["kind"] == "drop" and abs(e["t"] - t0) < 0.05 for e in a.events)
            gkey = f"{sec.group}:{j}"
            scores = {name: rules.score_template(cls.meta, sec=sec, style=style, motifs=motifs, history=history,
                                                 group_choice=group_pick.get(gkey, ("", {}))[0],
                                                 has_image=has_image)
                      for name, cls in REGISTRY.items()}
            again = group_pick.get(gkey)
            first_or_last = (si == 0 and j == 0) or (si == len(sections) - 1 and t1 >= sections[-1].end - 1e-6)
            if has_image and first_or_last and sec.label in ("intro", "outro", "verse", "instrumental"):
                # open on the character (revealed row by row) and end on her
                name = "image_ascii"
                params = {"mode": "halfblock", "crop": "full", "reveal": "scan" if si == 0 else "dissolve",
                          "react": 0.8}
                forced = True
            elif again and (not history or history[-1] != again[0]):
                forced = False
                name, params = again[0], dict(again[1])   # same look as last time this part came round
            else:
                forced = False
                name = rules.pick(scores, rng)
                params = rules.sample_params(name, rng, sec, motifs, tuple(style.get("tags", [])))
            if not forced:                                 # a forced opener/closer is not the group's look
                group_pick.setdefault(gkey, (name, params))
            history.append(name)
            prev = sections[si - 1] if si > 0 else None
            n += 1
            fx = rules.fx_for(sec, style, t0 / max(1.0, a.duration), shockwave=free and at_drop)
            hud = rules.hud_for(REGISTRY[name].meta, free, random.Random(hash32(seed, n, 0x4D0)))
            if hud and REGISTRY[name].meta.hud == "top" and "height" in params:
                params = {**params, "height": min(float(params["height"]), 0.55)}   # room for the panels
            shots.append(Shot(f"s{n:02d}", round(t0, 3), round(t1, 3), name, params, fx,
                              rules.transition_for(prev, sec, j == 0, len(shots), rng, at_drop=at_drop),
                              hud=hud))

    song = str_seed(str(a.source.get("audio_sha256", a.duration)))          # which songs get one varies too
    _easter_egg(a, shots, random.Random(hash32(seed, song, 0xE66)))
    if locked:
        shots = _merge_locked(shots, locked)

    cues = rules.lettering(a, shots, style, random.Random(hash32(seed, 0x1E77)))
    if lock:
        kept = {c.line: c for c in lock.lyrics if c.locked}
        cues = [kept.get(c.line, c) for c in cues]

    pal = palette or (lock.palette if lock else rules.palette_for(style))
    return Storyboard(seed=seed, palette=pal, fps=lock.fps if lock else 24,
                      lyric_offset=lock.lyric_offset if lock else 0.0,
                      grid=lock.grid if lock else (160, 45), shots=shots, lyrics=cues,
                      overlays={**(lock.overlays if lock else {"subtitle": False}),
                                "sprite": rules.sprite_overlay(has_image, style)},
                      assets=assets)


def _easter_egg(a: Analysis, shots: list[Shot], rng: random.Random) -> None:
    """About every other song, one shot (not the first or last, preferably with no lyric on screen) becomes an
    easter-egg template: the retro games, which carry few glyphs and would wear thin as regulars."""
    eggs = [cls.meta for cls in REGISTRY.values() if cls.meta.easter_egg]
    if not eggs or len(shots) < 3 or rng.random() >= rules.EGG_CHANCE:
        return
    inner = [s for s in shots[1:-1] if s.dur >= MIN_SHOT * 2]
    pool = [s for s in inner if _free(a, s.start, s.end)] or inner
    if not pool:
        return
    shot = pool[rng.randrange(len(pool))]
    sec = a.section_at(shot.start + 0.01) or Section(0.0, a.duration, "verse", 0.5, "-")
    fits = [m for m in eggs if m.energy[0] <= sec.energy <= m.energy[1]] or eggs
    meta = fits[rng.randrange(len(fits))]
    shot.template, shot.hud = meta.name, []
    shot.params = rules.sample_params(meta.name, rng, sec, Counter(), tuple(a.style.get("tags", [])))


def _merge_locked(gen: list[Shot], locked: list[Shot]) -> list[Shot]:
    """Keep locked shots exactly; trim generated shots so nothing overlaps them."""
    out: list[Shot] = []
    for s in gen:
        pieces = [(s.start, s.end)]
        for L in locked:
            nxt = []
            for a0, a1 in pieces:
                if L.end <= a0 or L.start >= a1:
                    nxt.append((a0, a1))
                    continue
                if L.start - a0 >= 0.25:
                    nxt.append((a0, L.start))
                if a1 - L.end >= 0.25:
                    nxt.append((L.end, a1))
            pieces = nxt
        for k, (a0, a1) in enumerate(pieces):
            out.append(Shot(s.id if k == 0 else f"{s.id}{chr(98 + k - 1)}", round(a0, 3), round(a1, 3), s.template,
                            dict(s.params), dict(s.fx), s.transition_in if a0 == s.start else "cut"))
    out.extend(locked)
    out.sort(key=lambda s: s.start)
    ids = [s.id for s in out]
    for i, s in enumerate(out):            # generated ids must not collide with locked ones
        if ids.count(s.id) > 1 and not s.locked:
            s.id = f"{s.id}_{i}"
    return out
