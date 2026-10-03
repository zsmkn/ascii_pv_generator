import io
import time

from ascii_pv.backends.audio_player import SilentClock
from ascii_pv.backends.raster import cell_size, to_image
from ascii_pv.backends.terminal import Terminal
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.director.planner import plan
from ascii_pv.render import Renderer


class Buf(io.BytesIO):
    pass


def _term():
    t = Terminal.__new__(Terminal)
    t.out = Buf()
    t._prev = None
    t._size = None
    return t


def test_terminal_diff_writes_only_changes():
    t = _term()
    c = Canvas(20, 5)
    c.put(0, 0, "hello")
    assert t.draw(c) > 0
    assert t.draw(c) == 0                     # identical frame: nothing sent
    c.put(10, 3, "x")
    t.out.seek(0)
    t.out.truncate()
    n = t.draw(c)
    data = t.out.getvalue().decode()
    assert 0 < n < 60 and "x" in data and "hello" not in data


def test_silent_clock():
    clk = SilentClock(10.0)
    clk.seek(2.0)
    assert clk.time() == 2.0 and not clk.playing
    clk.play()
    time.sleep(0.05)
    assert 2.03 < clk.time() < 2.5
    clk.pause()
    t = clk.time()
    time.sleep(0.02)
    assert clk.time() == t


def test_raster_size():
    c = Canvas(10, 4)
    c.put(0, 0, "A█▁")
    img = to_image(c, 14)
    cw, ch = cell_size(14)
    assert img.size == (10 * cw, 4 * ch)


def test_planned_storyboard_covers_song_and_renders():
    a = Analysis.synthetic(duration=40, bpm=120, seed=2)
    b = plan(a)
    assert b.shots[0].start == 0 and b.shots[-1].end == a.duration
    assert len(b.lyrics) == len(a.lyrics)
    r = Renderer(a, b)
    for t in range(0, 40, 3):
        r.frame(t + 0.3, 100, 30)
