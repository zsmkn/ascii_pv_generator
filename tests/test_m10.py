"""M10: stems in the feature table, build / drop / break events, and the Director's rules for lyric-free shots."""

import math

import numpy as np

from ascii_pv.analysis.events import detect
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.storyboard import Shot, Storyboard
from ascii_pv.director.planner import plan
from ascii_pv.render import Renderer

RATE = 48


def stepped_song(bpm=120.0, bars=32, drop_bar=16, build_bars=4):
    """Feature table that is quiet, rises over `build_bars`, then everything comes in at `drop_bar`."""
    bar = 240.0 / bpm
    dur = bars * bar
    n = int(dur * RATE)
    t = np.arange(n) / RATE
    level = np.where(t < (drop_bar - build_bars) * bar, 0.15,
                     np.where(t < drop_bar * bar, 0.15 + 0.3 * (t - (drop_bar - build_bars) * bar)
                              / (build_bars * bar), 0.9))
    feats = {"rate": RATE, "loud": level.tolist(), "bands": [level.tolist()] * 7, "kick": level.tolist(),
             "flux": level.tolist(), "cent": [0.5] * n}
    stems = {"rate": RATE, "drums": level.tolist(), "bass": np.where(t < drop_bar * bar, 0.0, 0.9).tolist(),
             "other": [0.5] * n, "kick": level.tolist(), "snare": level.tolist()}
    downbeats = [round(k * bar, 3) for k in range(bars)]
    return feats, stems, downbeats, dur, bar


def test_detect_finds_the_drop_and_its_build():
    feats, stems, downbeats, dur, bar = stepped_song()
    ev = detect(feats, stems, downbeats, dur, 120.0)
    drops = [e for e in ev if e["kind"] == "drop"]
    assert len(drops) == 1 and abs(drops[0]["t"] - 16 * bar) < 1e-6
    builds = [e for e in ev if e["kind"] == "build"]
    assert builds and builds[0]["end"] == drops[0]["t"] and builds[0]["start"] <= 13 * bar


def test_detect_works_without_stems_and_finds_breaks():
    feats, _, downbeats, dur, bar = stepped_song()
    flipped = {**feats, **{k: feats[k][::-1] for k in ("loud", "kick", "flux")}, "bands": [feats["loud"][::-1]] * 7}
    ev = detect(flipped, None, downbeats, dur, 120.0)
    assert any(e["kind"] == "break" for e in ev)


def test_features_fall_back_to_the_mix_without_stems():
    a = Analysis.synthetic(30.0)
    assert a.stems is None
    f = a.feat(5.0)
    i = int(5.0 * a.features["rate"])
    assert f.drums == a.features["loud"][i] and f.bass == a.features["bands"][0][i]
    assert f.drum_kick == a.features["kick"][i]


def test_event_queries():
    a = Analysis.synthetic(30.0)
    a.events = [{"kind": "build", "start": 4.0, "end": 8.0, "strength": 0.2},
                {"kind": "drop", "t": 8.0, "strength": 0.4}]
    assert a.build_progress(3.9) is None and abs(a.build_progress(6.0) - 0.5) < 1e-9
    assert a.since("drop", 7.9) == math.inf and abs(a.since("drop", 8.25) - 0.25) < 1e-9


def _song_with_drop(lyrics: bool):
    a = Analysis.synthetic(60.0)
    if not lyrics:
        a.lyrics = []
    sec = next(s for s in a.sections if s.label == "chorus")
    drop = next(d for d in a.downbeats if d > sec.start + 4)                 # a drop inside the chorus
    a.events = [{"kind": "build", "start": drop - 4.0, "end": drop, "strength": 0.2},
                {"kind": "drop", "t": drop, "strength": 0.4}]
    return a, drop


def test_drop_cuts_and_opens_on_a_shockwave_without_heavier_fx():
    a, drop = _song_with_drop(lyrics=False)
    b = plan(a, seed=1)
    at = [s for s in b.shots if abs(s.start - drop) < 1e-3]
    assert at and at[0].transition_in == "flash" and at[0].fx.get("shockwave") == 1.0
    others = [s for s in b.shots if s is not at[0]]
    assert all("shockwave" not in s.fx and "drive" not in s.fx for s in others)     # no stem-driven camera


def test_shots_with_lyrics_keep_the_old_rules():
    a, _ = _song_with_drop(lyrics=True)
    b = plan(a, seed=1)
    for s in b.shots:
        if any(ln.start < s.end and ln.end > s.start for ln in a.lyrics):
            assert "shockwave" not in s.fx


def test_stem_fx_frames_are_pure_functions_of_t():
    a, drop = _song_with_drop(lyrics=False)
    fx = {"shockwave": 1.0, "glitch": 0.5}
    board = Storyboard(shots=[Shot("s01", 0.0, a.duration, "plasma", {}, fx)])
    r = Renderer(a, board)
    ts = [drop - 0.5, drop + 0.1, drop + 0.3]
    first = [r.frame(t, 80, 24).ch.copy() for t in ts]
    r.frame(1.0, 80, 24)
    again = [r.frame(t, 80, 24).ch for t in reversed(ts)][::-1]
    assert all((x == y).all() for x, y in zip(first, again))
    wave = r.frame(drop + 0.15, 80, 24)
    calm = Renderer(a, Storyboard(shots=[Shot("s01", 0.0, a.duration, "plasma", {}, {})])).frame(drop + 0.15, 80, 24)
    assert np.isin(wave.ch, list("█▓▒░")).sum() > np.isin(calm.ch, list("█▓▒░")).sum() + 20   # the shockwave ring
    assert abs(wave.bg.astype(float).mean() - calm.bg.astype(float).mean()) < 1.0       # no white-out


def test_every_template_renders_with_its_hud_at_any_size():
    from ascii_pv.templates import REGISTRY
    from ascii_pv.templates.hud import CODE

    a = Analysis.synthetic(20.0)
    for name, cls in REGISTRY.items():
        if cls.meta.needs_image:
            continue
        panels = list(cls.meta.hud_topics) + ["log", "hex", "code:" + next(iter(CODE)), "plot:signal"]
        board = Storyboard(shots=[Shot("s01", 0.0, 20.0, name, {}, {}, hud=panels)])
        r = Renderer(a, board)
        for w, h in ((160, 45), (80, 24), (40, 12)):
            for t in (0.1, 0.5, 3.0):
                c = r.frame(t, w, h)
                assert c.ch.shape == (h, w)


def test_hud_is_a_pure_function_of_t():
    a = Analysis.synthetic(20.0)
    board = Storyboard(shots=[Shot("s01", 0.0, 20.0, "wireframe_mesh", {}, {},
                                   hud=["formula:torus", "code:donut", "plot:sin", "data"])])
    r = Renderer(a, board)
    x = r.frame(5.0, 160, 45).ch.copy()
    r.frame(9.0, 160, 45)
    assert (r.frame(5.0, 160, 45).ch == x).all()
    assert "R = 0.70" in "".join(x[3]) or any("R = 0.70" in "".join(row) for row in x)   # the torus formula


def test_easter_eggs_are_rare():
    from ascii_pv.templates import REGISTRY

    eggs = {n for n, c in REGISTRY.items() if c.meta.easter_egg}
    assert {"tetris", "invaders", "pong", "runner"} <= eggs
    a = Analysis.synthetic(120.0)
    a.lyrics = []
    counts = [sum(s.template in eggs for s in plan(a, seed=seed).shots) for seed in range(20)]
    assert max(counts) <= 1                                   # at most one a song
    assert 3 <= sum(counts) <= 17                             # roughly every other song
