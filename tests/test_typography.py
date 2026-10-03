import numpy as np
import pytest

from ascii_pv.core.analysis import Analysis, LyricLine, Word
from ascii_pv.core.storyboard import LyricCue, Storyboard
from ascii_pv.render import Renderer
from ascii_pv.typography.font5 import glyph, render_text, text_bits_width
from ascii_pv.typography.inkmask import chunks, ink_mask, layout
from ascii_pv.typography.layer import PRE, LyricLayer


def test_font_covers_letters_digits_punct():
    for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'!?-:;()&/":
        g = glyph(ch)
        assert g.shape[0] == 5 and g.any()
    assert glyph("a").tolist() == glyph("A").tolist()
    bits, owner = render_text("HI")
    assert bits.shape[1] == text_bits_width("HI") == 5 + 1 + 5
    assert owner.tolist().count(-1) == 1


def test_layout_prefers_big_letters_and_wraps():
    font, runs, px_w, px_h = layout(("IS", "THIS", "REAL?"), 116, 36)
    assert font == "bit5x5" and px_h >= 2 and px_h <= px_w <= 2 * px_h
    assert layout(tuple("a very long line that cannot possibly fit in a tiny box".split()), 40, 8) is None


def test_chunks_fit_and_cover_all_words():
    words = tuple("Playing with the night shade, twenty-seven".split())
    runs = chunks(words, 116, 15)
    assert runs[0][0] == 0 and runs[-1][1] == len(words)
    assert all(b > a for a, b in runs)


def test_ink_mask_consistency():
    m = ink_mask(("HELLO", "WORLD"), (2, 2, 116, 36), (120, 40))
    assert m is not None and m.n == int(m.mask.sum())
    assert set(np.unique(m.word)) == {0, 1}
    x0, y0, x1, y1 = m.bbox
    assert (m.xs >= x0).all() and (m.xs < x1).all() and (m.ys >= y0).all() and (m.ys < y1).all()
    assert m.edge.any() and not m.edge.all()     # big letters have an inside


def _song_with_lines():
    a = Analysis.synthetic(duration=30, bpm=120, seed=3)
    words = [Word(10.0, 10.5, "IS"), Word(10.5, 11.0, "THIS"), Word(11.0, 12.0, "REAL?")]
    a.lyrics = [LyricLine(10.0, 13.0, "IS THIS REAL?", words, hook=True),
                LyricLine(13.0, 16.0, "NEXT", [Word(13.0, 16.0, "NEXT")])]
    return a


def test_windows_do_not_overlap_and_start_early():
    a = _song_with_lines()
    layer = LyricLayer(a, Storyboard())
    (a0, b0), (a1, b1) = layer._windows
    assert a0 == pytest.approx(10.0 - PRE) and b0 <= a1 and a1 == pytest.approx(13.0 - PRE)


@pytest.mark.parametrize("mode", ["takeover", "line"])
def test_lyrics_draw_and_settle(mode):
    a = _song_with_lines()
    board = Storyboard.single("blank", a.duration, lyrics=[LyricCue(line=0, mode=mode)])
    r = Renderer(a, board)
    before = r.frame(9.0, 120, 40)
    assert before.plain().strip() == ""                       # nothing before the window
    settled = r.frame(11.6, 120, 40)                           # all three words sung, letters landed
    layer = r.lyrics
    p = layer.active(11.6, 120, 40)
    ink = layer._parts(p, 120, 40).ink
    drawn = np.array([settled.ch[y, x] != " " for x, y in zip(ink.xs, ink.ys)])
    assert drawn.mean() > 0.9                                 # the letters are there


def test_lyric_offset_shifts_lyric_time_only():
    """The offset moves which lyric is up; beat-driven effects (sweep, kick lift) still follow the music."""
    a = _song_with_lines()
    r = Renderer(a, Storyboard.single("blank", a.duration))
    assert r.lyrics.active(9.0 + 1.5, 120, 40) is not None and r.lyrics.active(9.0, 120, 40) is None
    shown = r.frame(9.0, 120, 40, lyric_offset=1.5)
    assert shown.plain().strip() != ""


# ---------------------------------------------------------------- v2

from ascii_pv.typography import fills, motion  # noqa: E402
from ascii_pv.typography.fonts import font_path, ttf_line  # noqa: E402


def test_ttf_fonts_render_capitals_filling_the_rows():
    for name in ("impact", "black", "serif"):
        path = font_path(name)
        if path is None:
            pytest.skip("Windows fonts not available")
        m, own = ttf_line(path, ("HI", "YOU"), 10)
        assert m.shape[0] == 10 and m[1:-1].any(axis=1).all()
        assert set(np.unique(own[m])) == {0, 1}


def test_ttf_layout_falls_back_to_bitmap_when_small():
    ink = ink_mask(tuple("one two three four five six seven eight nine".split()), (0, 0, 60, 8), (60, 8),
                   font="impact")
    assert ink is None or ink.font == "bit5x5"


def test_fills():
    ink = ink_mask(("HELLO",), (2, 2, 116, 36), (120, 40))
    k = np.arange(ink.n)
    for name in fills.NAMES:
        chars = fills.resting(name, ink, ("HELLO",), k)
        assert chars.shape == (ink.n,)
    assert set(fills.resting("self", ink, ("HELLO",), k)) <= set("HELO")
    out = fills.resting("outline", ink, ("HELLO",), k)
    assert (out[~ink.edge] == " ").all() and (out[ink.edge] == "#").all()


@pytest.mark.parametrize("kind", motion.ENTERS)
def test_enter_kinds_land(kind):
    ink = ink_mask(("HEY",), (2, 2, 116, 36), (120, 40))
    n = ink.n
    rng = np.random.default_rng(0)
    args = (ink.xs.astype(float), ink.ys.astype(float), ink.order, rng.random(n), rng.integers(0, 120, n),
            rng.integers(0, 40, n), np.arange(n))
    early = motion.enter(kind, *args, 0.01, 0.6, 120, 40, ink.bbox)
    done = motion.enter(kind, *args, 5.0, 0.6, 120, 40, ink.bbox)
    assert done.landed.all() and done.visible.all()
    assert not early.landed.all()
    assert np.allclose(done.x, ink.xs) and np.allclose(done.y, ink.ys)


@pytest.mark.parametrize("kind", motion.EXITS)
def test_exit_kinds_leave(kind):
    ink = ink_mask(("HEY",), (2, 2, 116, 36), (120, 40))
    n = ink.n
    f = motion.Frame(ink.xs.astype(float), ink.ys.astype(float), np.ones(n, bool), np.ones(n), np.ones(n))
    late = motion.leave(kind, f, ink.xs.astype(float), ink.ys.astype(float), np.arange(n), 0.99, 120, 40, ink.bbox)
    assert late.visible.mean() < 0.5 or not late.landed.any()


@pytest.mark.parametrize("mode", ["takeover", "line", "word_flash", "stack", "embed"])
@pytest.mark.parametrize("blend", ["overlay", "mask", "knockout", "echo"])
def test_every_mode_and_blend_draws(mode, blend):
    a = _song_with_lines()
    board = Storyboard.single("glyph_storm", a.duration,
                              lyrics=[LyricCue(line=0, mode=mode, blend=blend, font="impact", fill="ramp"),
                                      LyricCue(line=1, mode=mode, blend=blend)])
    r = Renderer(a, board)
    for t in (9.8, 10.3, 11.6, 12.5, 14.0):
        r.frame(t, 120, 40)
    assert r.lyrics.active(11.6, 120, 40) is not None


def test_word_flash_shows_one_word_at_a_time():
    a = _song_with_lines()
    r = Renderer(a, Storyboard.single("blank", a.duration, lyrics=[LyricCue(line=0, mode="word_flash")]))
    assert r.lyrics.active(10.7, 120, 40).words == ("THIS",)
    assert r.lyrics.active(11.5, 120, 40).words == ("REAL?",)


def test_subtitle_overlay():
    a = _song_with_lines()
    b = Storyboard.single("blank", a.duration)
    b.overlays["subtitle"] = True
    c = Renderer(a, b).frame(11.0, 120, 40)
    assert "IS THIS REAL?" in c.plain().splitlines()[-1]


def test_mask_blend_letters_stay_solid_on_a_sparse_picture():
    # a dark picture of thin dots: the letters must not be drawn with its ". ' ^" (they vanish into it)
    from ascii_pv.core.canvas import Canvas
    from ascii_pv.core.palette import get_palette
    from ascii_pv.typography.layer import THIN

    a = _song_with_lines()
    layer = LyricLayer(a, Storyboard(lyrics=[LyricCue(line=0, mode="embed", blend="mask", font="impact",
                                                      fill="solid")]))
    pal = get_palette("deepsea")
    for t in (10.9, 11.5):                                       # "REAL?" still unsung, then all sung
        c = Canvas(120, 40, bg=pal.bg, fg=pal.ui)
        c.ch[:] = np.where((np.arange(120)[None, :] + np.arange(40)[:, None]) % 2 == 0, ".", "^")
        layer.draw(c, t, pal, 0.5, 0.5, 0.0, 0.5)
        ink = layer._parts(layer.active(t, 120, 40), 120, 40).ink
        drawn = ink.edge | (ink.word < 2) | (t > 11.0)          # unsung words show only their outline
        assert not np.isin(c.ch[ink.ys[drawn], ink.xs[drawn]], THIN).any()


def test_switch_waits_for_the_last_word():
    from ascii_pv.typography.layer import MIN_LIT, switch_time

    assert switch_time(10.0, 12.0, PRE) == pytest.approx(12.0 - PRE)          # room: the usual lead
    assert switch_time(10.0, 10.5, PRE) == pytest.approx(10.0 + MIN_LIT)     # the last word gets its moment
    assert switch_time(10.0, 10.1, PRE) == pytest.approx(10.1)               # never after the next word


def test_every_word_of_a_dense_chunked_line_lights_before_the_switch():
    from ascii_pv.analysis.lyrics import estimate_words

    a = Analysis.synthetic(duration=30, bpm=120, seed=3)
    text = "沒解開的誤會 時間的路費 心臟的枯萎只能忘記"                     # ~0.2 s a character
    a.lyrics = [LyricLine(10.0, 14.5, text, estimate_words(text, 10.0, 14.5)),
                LyricLine(14.5, 16.0, "next", estimate_words("next", 14.5, 16.0))]
    layer = LyricLayer(a, Storyboard(lyrics=[LyricCue(line=0, mode="line")]))
    pieces = layer._pieces(0, 160, 45)
    assert len(pieces) >= 2
    for p in pieces:
        assert max(p.word_times) < p.t_out                                    # lit before it goes
    assert all(x.t_out <= y.t_in + 1e-9 for x, y in zip(pieces, pieces[1:]))
    # ... and is still whole on screen just after its last word (the exit waits for it)
    from ascii_pv.core.canvas import Canvas
    from ascii_pv.core.palette import get_palette

    pal = get_palette("deepsea")
    for p in pieces:
        c = Canvas(160, 45, bg=pal.bg, fg=pal.ui)
        layer.draw(c, max(p.word_times) + 0.05, pal, 0.5, 0.5, 0.0, 0.5)
        ink = layer._parts(p, 160, 45).ink
        assert (c.ch[ink.ys, ink.xs] != " ").mean() > 0.95


@pytest.mark.parametrize("room", [1.0, 0.45])
def test_stack_last_word_lands_before_the_exit(room):
    # stack: every word drops in when sung; the last one must land (and light) before the line leaves, even
    # when the next line follows soon (then the words drop faster)
    from ascii_pv.analysis.lyrics import estimate_words
    from ascii_pv.core.canvas import Canvas
    from ascii_pv.core.palette import get_palette
    from ascii_pv.typography.layer import LIT_BEFORE_EXIT

    a = Analysis.synthetic(duration=30, bpm=120, seed=3)
    words = estimate_words("終究和她擦肩而過", 10.0, 12.0)
    last = words[-1].start
    a.lyrics = [LyricLine(10.0, last + room, "終究和她擦肩而過", words),
                LyricLine(last + room + PRE, 18.0, "next", estimate_words("next", last + room + PRE, 18.0))]
    layer = LyricLayer(a, Storyboard(lyrics=[LyricCue(line=0, mode="stack", exit="crt_off")]))
    p = layer._pieces(0, 160, 45)[-1]
    ink = layer._parts(p, 160, 45).ink
    pal = get_palette("deepsea")
    t = p.t_out - 0.1 - 0.02 if room < 0.5 else last + 0.3 + LIT_BEFORE_EXIT / 2   # just before the exit
    c = Canvas(160, 45, bg=pal.bg, fg=pal.ui)
    layer.draw(c, t, pal, 0.5, 0.5, 0.0, 0.5)
    mine = ink.word == len(p.words) - 1
    assert (c.ch[ink.ys[mine], ink.xs[mine]] != " ").mean() > 0.95
