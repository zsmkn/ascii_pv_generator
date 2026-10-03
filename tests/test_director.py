import dataclasses

import pytest

from ascii_pv.core.analysis import Analysis, LyricLine, Word
from ascii_pv.core.storyboard import Shot
from ascii_pv.director import rules
from ascii_pv.director.planner import plan
from ascii_pv.render import Renderer


@pytest.fixture(scope="module")
def song():
    return Analysis.synthetic(duration=90, bpm=124, seed=5)


def test_shots_tile_the_song(song):
    b = plan(song, seed=1)
    assert b.shots[0].start == 0 and b.shots[-1].end == pytest.approx(song.duration)
    for s0, s1 in zip(b.shots, b.shots[1:]):
        assert s0.end == pytest.approx(s1.start) and s1.dur > 0.5
    assert len({s.id for s in b.shots}) == len(b.shots)
    assert b.shots[0].transition_in == "crt_on"


def test_cuts_land_on_downbeats(song):
    b = plan(song, seed=1)
    db = set(round(d, 3) for d in song.downbeats)
    sec_starts = set(round(s.start, 3) for s in song.sections)
    for s in b.shots[1:]:
        assert round(s.start, 3) in db | sec_starts


def test_deterministic_and_seed_sensitive(song):
    a1, a2 = plan(song, seed=7).to_dict(), plan(song, seed=7).to_dict()
    assert a1 == a2
    others = [plan(song, seed=s).to_dict()["shots"] for s in range(8)]
    assert len({str(o) for o in others}) > 1


def test_repeated_groups_share_their_look(song):
    b = plan(song, seed=3)
    seen = {}
    for s in b.shots:
        sec = song.section_at(s.start + 0.01)
        idx = sum(1 for o in b.shots if sec.start - 1e-3 <= o.start < s.start)
        key = (sec.label, idx)
        if sec.label == "chorus":
            seen.setdefault(key, []).append((s.template, s.params))
    for looks in seen.values():
        assert all(lk == looks[0] for lk in looks)


def test_lock_keeps_locked_shots_and_cues(song):
    first = plan(song, seed=1)
    mine = Shot("mine", 20.0, 30.0, "spectrum_bars", {"height": 0.9}, locked=True)
    first.shots = [s for s in first.shots if s.end <= 20.0 or s.start >= 30.0] + [mine]
    first.lyrics[0] = dataclasses.replace(first.lyrics[0], mode="takeover", fill="hex", locked=True)
    again = plan(song, seed=99, lock=first)
    kept = [s for s in again.shots if s.id == "mine"]
    assert kept and kept[0].start == 20.0 and kept[0].params == {"height": 0.9}
    for s in again.shots:
        if s.id != "mine":
            assert s.end <= 20.0 + 1e-6 or s.start >= 30.0 - 1e-6
    assert again.lyrics[0].fill == "hex" and again.lyrics[0].locked


def test_dense_lines_get_calm_lettering():
    a = Analysis.synthetic(duration=60, bpm=120, seed=1)
    fast = LyricLine(20.0, 22.0, "one two three four five six seven eight",
                     [Word(20 + i * 0.25, 20 + (i + 1) * 0.25, w) for i, w in
                      enumerate("one two three four five six seven eight".split())])
    slow = LyricLine(30.0, 34.0, "we are shadow bound now", [Word(30 + i, 31 + i, w) for i, w in
                                                             enumerate("we are shadow bound now".split())])
    a.lyrics = [slow, fast]
    fast.start, fast.end = 36.0, 38.0
    for w_, t in zip(fast.words, [36 + i * 0.25 for i in range(8)]):
        w_.start, w_.end = t, t + 0.25
    import random
    for seed in range(6):
        cues = rules.lettering(a, [], {"energy": 0.5, "valence": 0.5, "tags": []}, random.Random(seed))
        cf = cues[1]
        if cf.mode in ("line", "embed"):
            assert (cf.enter, cf.exit) == ("decode", "fade")


def test_palette_follows_style():
    assert rules.palette_for({"energy": 0.8, "valence": 0.7}) == "synthwave"
    assert rules.palette_for({"energy": 0.3, "valence": 0.3}) == "deepsea"


def test_planned_song_renders_everywhere(song):
    b = plan(song, seed=2)
    r = Renderer(song, b)
    for s in b.shots:                                   # start of each shot = inside its transition
        for dt in (0.01, 0.2, s.dur / 2):
            r.frame(s.start + dt, 90, 28)
