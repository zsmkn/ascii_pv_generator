import pytest

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.storyboard import LyricCue, Shot, Storyboard
from ascii_pv.render import Renderer
from ascii_pv.templates import REGISTRY


@pytest.fixture(scope="module")
def song():
    return Analysis.synthetic(duration=40, bpm=120, seed=1)


def test_synthetic_song_shape(song):
    f = song.features
    assert len(f["bands"]) == 7 and len(f["loud"]) == 40 * 48
    assert song.sections[0].label == "intro"
    assert song.lyrics and all(ln.words for ln in song.lyrics)


def test_beat_info(song):
    i, phase, bt = song.beat_info(1.25)          # 120 bpm: beats every 0.5 s
    assert i == 2 and abs(phase - 0.5) < 1e-6 and bt == 1.0
    i, phase, bt = song.beat_info(song.duration + 1.0)   # extrapolated past the end
    assert abs(bt - round(bt * 2) / 2) < 1e-6 and 0 <= phase < 1


def test_analysis_roundtrip(tmp_path, song):
    song.save(tmp_path / "a.json")
    back = Analysis.load(tmp_path / "a.json")
    assert back.to_dict() == song.to_dict()


def test_storyboard_roundtrip_and_lookup(tmp_path):
    sb = Storyboard(shots=[Shot("b", 10, 20, "blank"), Shot("a", 0, 10, "spectrum_bars")],
                    lyrics=[LyricCue(line=0, mode="takeover")])
    sb.save(tmp_path / "s.json")
    back = Storyboard.load(tmp_path / "s.json")
    assert back.to_dict() == sb.to_dict()
    assert back.shot_at(5).id == "a" and back.shot_at(15).id == "b" and back.shot_at(-1).id == "a"


def test_builtin_templates_registered():
    assert {"blank", "spectrum_bars"} <= set(REGISTRY)


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_template_renders_any_size(song, name):
    r = Renderer(song, Storyboard.single(name, song.duration))
    for w, h in [(20, 6), (80, 24), (160, 45)]:
        c = r.frame(12.3, w, h)
        assert (c.w, c.h) == (w, h)


def test_frames_are_pure_functions_of_t(song):
    r1 = Renderer(song, Storyboard.single("spectrum_bars", song.duration))
    r2 = Renderer(song, Storyboard.single("spectrum_bars", song.duration))
    a = r1.frame(17.0, 100, 30).ansi()
    r1.frame(3.0, 100, 30)  # rendering other times in between must not matter
    assert r1.frame(17.0, 100, 30).ansi() == a == r2.frame(17.0, 100, 30).ansi()


def test_unknown_template_lists_available(song):
    r = Renderer(song, Storyboard.single("nope", song.duration))
    with pytest.raises(KeyError, match="spectrum_bars"):
        r.frame(0, 10, 5)
