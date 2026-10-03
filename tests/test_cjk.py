"""Chinese / Japanese lyrics: tokens, timing, layout, fills, rules and the rasterizer."""

import numpy as np
import pytest

from ascii_pv.analysis.lyrics import _key, estimate_words, load_lyrics, motifs_of
from ascii_pv.core.analysis import Analysis, LyricLine
from ascii_pv.core.cjk import glue, has_cjk, tokens, word_count
from ascii_pv.core.storyboard import LyricCue, Storyboard
from ascii_pv.render import Renderer
from ascii_pv.typography import fills, fonts
from ascii_pv.typography.inkmask import chunks, ink_mask, layout
from ascii_pv.typography.layer import LyricLayer

pytestmark = pytest.mark.skipif(fonts.cjk_path("impact", "中") is None, reason="no CJK font installed")


# ---------------------------------------------------------------- text

def test_tokens_split_cjk_and_keep_english_as_is():
    assert tokens("遍地 emo 的連鎖") == ("遍", "地", "emo", "的", "連", "鎖")
    assert tokens("(我陪她哭過 痛過)") == ("(我", "陪", "她", "哭", "過", "痛", "過)")
    assert tokens("為何只剩下了難過？") == ("為", "何", "只", "剩", "下", "了", "難", "過？")
    assert tokens("「你好」，世界。") == ("「你", "好」，", "世", "界。")
    for s in ("So, let me hold you close (Oh, yeah)", "24/7 rock-n-roll", "Isn't it strange?"):
        assert tokens(s) == tuple(s.split(" "))                  # English: unchanged


def test_glue_follows_the_spaces_of_the_text():
    words = tokens("哭過 痛過 emo 的")
    assert glue(words, "哭過 痛過 emo 的") == (False, True, False, True, False, False)
    assert not any(glue(tuple("a b c".split()), "a b c"))
    assert word_count("祝她越來越好") == 3 and word_count("just free talk") == 3


def test_cjk_timing_one_syllable_per_character_with_breaths_at_spaces():
    ws = estimate_words("哭過 痛過", 10.0, 12.0)
    assert [w.text for w in ws] == ["哭", "過", "痛", "過"]
    d = np.diff([w.start for w in ws])
    assert d[1] > d[0] and d[1] > d[2]                           # the space is a short pause


def test_repeats_and_motifs_of_chinese_lines(tmp_path):
    assert _key("為何只剩下了難過？") != _key("終究和她擦肩而過")
    assert _key("為何只剩下了難過？") == _key("為何只剩下了難過")
    assert "water" in motifs_of("我陪她哭過") and "fly" in motifs_of("祝她飛向海闊天高")
    lrc = tmp_path / "zh.lrc"
    lrc.write_text("[00:01.00]哭過 痛過\n[00:04.00]終究和她擦肩而過\n[00:08.00]哭過 痛過\n[00:12.00]end\n",
                   encoding="utf-8")
    lines = load_lyrics(lrc)
    assert [ln.hook for ln in lines[:3]] == [True, False, True]
    assert [w.text for w in lines[1].words] == list("終究和她擦肩而過")


# ---------------------------------------------------------------- layout

def test_cjk_layout_uses_a_cjk_face_and_glued_words_sit_closer():
    words = tokens("祝她越來越好")
    lay = layout(words, 156, 17, font="bit5x5", glue=glue(words, "祝她越來越好"))
    assert lay is not None and lay[0] in fonts.CJK_FACES and lay[3] >= fonts.MIN_CJK_ROWS
    path = lay[0]
    glued = fonts.ttf_cols(path, words, 10, glue(words, "祝她越來越好"))
    spaced = fonts.ttf_cols(path, words, 10)
    assert glued < spaced


def test_cjk_chunks_have_no_one_word_leftover():
    text = "吵鬧的夜店只想和她 just free talk"
    words = tokens(text)
    _, _, aw, ah = LyricLayer.area("line", 160, 45)
    runs = chunks(words, aw, ah, "impact", glue(words, text))
    assert len(runs) >= 2 and runs[0][0] == 0 and runs[-1][1] == len(words)
    assert all(b - a >= 2 for a, b in runs)
    assert words[runs[-1][0]:] == ("just", "free", "talk")


def test_texture_fills_need_thick_cjk_strokes():
    words = tokens("沒解開的誤會")
    small = ink_mask(words, (2, 26, 156, 17), (160, 45), 99, "impact", glue(words, "沒解開的誤會"))
    assert small.cjk and not fills.thick(small)                  # binary / hex would blur these strokes
    huge = ink_mask(("哭",), (3, 2, 154, 41), (160, 45), 6, "impact")
    assert huge.cjk and fills.thick(huge)


# ---------------------------------------------------------------- rendering

def _song():
    a = Analysis.synthetic(duration=30, bpm=120, seed=3)
    a.lyrics = [LyricLine(10.0, 13.0, "哭過 痛過 愛過 恨過", estimate_words("哭過 痛過 愛過 恨過", 10.0, 13.0)),
                LyricLine(13.0, 16.0, "遍地 emo 的連鎖", estimate_words("遍地 emo 的連鎖", 13.0, 16.0))]
    return a


@pytest.mark.parametrize("mode", ["takeover", "line", "word_flash", "stack", "embed"])
@pytest.mark.parametrize("blend", ["overlay", "mask", "knockout", "echo"])
def test_every_mode_and_blend_draws_cjk(mode, blend):
    a = _song()
    board = Storyboard.single("glyph_storm", a.duration,
                              lyrics=[LyricCue(line=0, mode=mode, blend=blend, font="bit5x5", fill="binary"),
                                      LyricCue(line=1, mode=mode, blend=blend, font="serif", fill="self")])
    r = Renderer(a, board)
    for t in (9.8, 11.0, 12.5, 14.0, 15.5):
        c = r.frame(t, 120, 40)
        assert not any(has_cjk(ch) for ch in c.ch.ravel())     # big letters are made of ASCII glyphs
    p = r.lyrics.active(11.0, 120, 40)
    assert p is not None and r.lyrics._parts(p, 120, 40) is not None


def test_subtitle_rasterises_cjk_inside_the_cell():
    from ascii_pv.backends.raster import Rasterizer

    a = _song()
    b = Storyboard.single("blank", a.duration)
    b.overlays["subtitle"] = True
    c = Renderer(a, b).frame(11.0, 60, 10)
    assert "哭" in "".join(c.ch[-1])
    img = Rasterizer(8, 16).render(c)
    row = img[-16:].astype(int).sum(axis=2)
    lit = np.nonzero(row.max(axis=1) > 200)[0]
    assert len(lit) and lit.min() >= 1 and lit.max() <= 14            # drawn, and not clipped by the cell


def test_rules_keep_knockout_away_from_cjk():
    from ascii_pv.director.planner import plan

    a = _song()
    a.lyrics = a.lyrics * 6
    for seed in range(6):
        board = plan(a, seed=seed)
        assert all(c.blend != "knockout" for c in board.lyrics)
