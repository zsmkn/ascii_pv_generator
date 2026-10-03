"""M8: new templates, post effects and transitions."""

import numpy as np
import pytest

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.palette import get_palette
from ascii_pv.core.storyboard import Shot, Storyboard
from ascii_pv.fx import post, transitions
from ascii_pv.render import Renderer

NEW = ["oscilloscope", "starfield", "plasma", "flames", "life_grid", "heart", "clock", "skyline", "boot_log"]


@pytest.fixture(scope="module")
def song():
    return Analysis.synthetic(duration=40, bpm=120, seed=8)


@pytest.mark.parametrize("name", NEW)
def test_new_templates_draw_something_and_are_pure(song, name):
    r1 = Renderer(song, Storyboard(shots=[Shot("x", 0, 40, name)]))
    r2 = Renderer(song, Storyboard(shots=[Shot("x", 0, 40, name)]))
    a = r1.frame(12.3, 100, 30)
    assert (a.ch != " ").sum() > 20
    r1.frame(30.0, 100, 30)                      # go forward, then back (life_grid caches its state)
    assert r1.frame(12.3, 100, 30).ansi() == a.ansi() == r2.frame(12.3, 100, 30).ansi()


def test_life_grid_steps_one_generation_per_beat(song):
    r = Renderer(song, Storyboard(shots=[Shot("x", 0, 40, "life_grid")]))
    a = r.frame(10.05, 80, 24).plain()
    assert r.frame(10.3, 80, 24).plain() == a    # same beat (120 bpm: beats every 0.5 s)
    assert r.frame(10.55, 80, 24).plain() != a   # next beat, next generation


def test_travel_is_monotonic(song):
    ts = np.linspace(0, 45, 200)
    d = [song.travel(t) for t in ts]
    assert all(b > a for a, b in zip(d, d[1:]))


def _busy(w=60, h=20, ch="x"):
    c = Canvas(w, h)
    c.ch[:, ::2] = ch
    c.fg[:] = (200, 150, 50)
    return c


@pytest.mark.parametrize("fx", ["shake", "zoom", "split", "strobe", "noise"])
def test_post_effects_change_the_picture_on_a_hit(fx):
    pal = get_palette("amber")
    c = _busy()
    before = (c.ch.copy(), c.fg.copy(), c.bg.copy())
    fn = getattr(post, fx)
    args = {"shake": (1.0, 1.0, 7), "zoom": (1.0, 1.0), "split": (1.0, 1.0, pal), "strobe": (1.0, 1.0, pal),
            "noise": (1.0, 7, pal)}[fx]
    fn(c, *args)
    assert not (np.array_equal(before[0], c.ch) and np.array_equal(before[1], c.fg)
                and np.array_equal(before[2], c.bg))
    quiet = _busy()
    fn(quiet, *[0.0 if isinstance(v, float) else v for v in args])
    assert np.array_equal(quiet.ch, _busy().ch)


@pytest.mark.parametrize("kind", ["blocks", "slide", "zoom", "reflow"])
def test_new_transitions_go_from_a_to_b(kind):
    pal = get_palette("amber")
    a, b = _busy(ch="a"), _busy(ch="b")
    early = _busy(ch="b")
    transitions.apply(kind, a, early, 0.05, pal)
    assert (early.ch == "a").sum() > (early.ch == "b").sum() * 0.5 or kind == "reflow"
    late = _busy(ch="b")
    transitions.apply(kind, a, late, 0.999, pal)
    assert (late.ch == "b").sum() >= (b.ch == "b").sum() * 0.9


def test_shell_typing_types_whole_commands_on_the_grid():
    from ascii_pv.core.analysis import Analysis
    from ascii_pv.core.storyboard import Shot
    from ascii_pv.templates import create

    a = Analysis.synthetic(40.0)
    a.source["audio"] = "some very long song title.mp3"
    for rate in (1, 2, 3):
        tpl = create(Shot("s01", 5.0, 35.0, "shell_typing", {"rate": rate}, {}), a)
        typed, line = [], ""
        for _, kind, p in tpl.events:
            if kind == "prompt":
                line = ""
            elif kind == "keys":
                line += p
            elif kind == "enter":
                typed.append(line)
        assert "ffprobe -hide_banner some very long song title.mp3" in typed
        assert all(t0 <= t1 for t0, t1 in zip(tpl.times, tpl.times[1:]))
