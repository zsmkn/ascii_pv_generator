"""Plain-text lyric alignment (analysis/align.py) on synthetic CTC emissions: no models needed."""

import numpy as np
import pytest

from ascii_pv.analysis import align as A
from ascii_pv.analysis.lyrics import load_lyrics

FPS = 50.0
VOCAB = {"-": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "o": 6, "'": 7}


def emission(segments, seconds):
    """Log-probs: blank everywhere, except (char, t0, t1) segments where that char is near certain."""
    T = int(seconds * FPS)
    p = np.full((T, len(VOCAB)), 0.01)
    p[:, 0] = 0.9
    for ch, t0, t1 in segments:
        a, b = int(t0 * FPS), int(t1 * FPS)
        p[a:b] = 0.01
        p[a:b, VOCAB[ch]] = 0.9
    return np.log(p / p.sum(1, keepdims=True))


def sung(word, t, dur=0.1):
    return [(c, t + k * dur, t + (k + 1) * dur - 0.02) for k, c in enumerate(word)]


# ---------------------------------------------------------------- text

def test_parse_plain_headers_meta_stanzas_and_backing_vocals():
    lines = A.parse_plain("[ar:Someone]\n[Verse 1]\nab cd\n\nChorus:\nbad (ode, ace)\n(dab dab)\n")
    assert [ln.text for ln in lines] == ["ab cd", "bad (ode, ace)", "(dab dab)"]
    assert lines[0].section == "Verse 1" and lines[1].section == "Chorus"
    assert [ln.stanza for ln in lines] == [0, 1, 1]
    assert lines[1].optional == [False, True, True]
    assert lines[2].optional == [True, True]


def test_spoken_words():
    assert A.spoken("24/7") == "twenty four seven"
    assert A.spoken("We're,") == "we're"
    assert A.spoken("R&B") == "r and b"
    assert A.word_key("Don't!") == "dont"


def test_text_lines_of_timed_files_expand_repeats(tmp_path):
    lrc = tmp_path / "a.lrc"
    lrc.write_text("[ti:x]\n[00:05.00]first\n[00:01.00][00:09.00]chorus\n", encoding="utf-8")
    assert [ln.text for ln in A.text_lines(lrc)] == ["chorus", "first", "chorus"]
    assert not A.is_plain_text(lrc)
    txt = tmp_path / "a.txt"
    txt.write_text("first\nsecond\n", encoding="utf-8")
    assert A.is_plain_text(txt) and [ln.text for ln in A.text_lines(txt)] == ["first", "second"]
    untimed = tmp_path / "b.lrc"
    untimed.write_text("[ar:x]\nfirst\n", encoding="utf-8")
    assert A.is_plain_text(untimed)


# ---------------------------------------------------------------- matching

def test_match_words_tolerates_misheard_words():
    pairs = A.match_words(["link", "link", "up", "now"], ["blink", "link", "up", "wow", "now"])
    assert [(i, j) for i, j, _ in pairs] == [(0, 0), (1, 1), (2, 2), (3, 4)]
    assert pairs[0][2] == 1 and pairs[1][2] == 2


def test_anchors_need_a_run_and_skip_unclear_repeats():
    lines = [A.text_line(s) for s in ["dab ace", "bad ode", "bad ode", "cab dad"]]
    heard = [(1.0, 1.2, "dab"), (1.3, 1.5, "ace"), (4.0, 4.2, "bad"), (4.3, 4.5, "ode"),
             (9.0, 9.2, "cab"), (9.3, 9.5, "dad")]
    win = A.anchors(lines, heard)
    # "bad ode" twice, heard once: which copy is unknown -> neither is anchored; the others are
    assert set(win) == {0, 1, 6, 7}
    heard2 = heard[:4] + [(6.0, 6.2, "bad"), (6.3, 6.5, "ode")] + heard[4:]
    assert set(A.anchors(lines, heard2)) == set(range(8))          # both copies heard: anchored in order


# ---------------------------------------------------------------- alignment

def test_align_finds_words_skips_absent_backing_vocals_and_exports_lrc(tmp_path):
    segs = sung("bad", 1.0) + sung("ace", 1.5) + sung("cab", 4.0) + sung("ode", 4.6)
    logp = emission(segs, 8.0)
    lines = A.parse_plain("bad ace (dab)\ncab ode\n")
    out, info = A.align_lines(lines, logp, FPS, VOCAB, None, 8.0)
    assert info.skipped == 1                                       # "(dab)" is not sung
    assert abs(out[0].start - 1.0) < 0.05 and abs(out[1].start - 4.0) < 0.05
    assert abs(out[0].words[1].start - 1.5) < 0.05 and abs(out[1].words[1].start - 4.6) < 0.05
    assert out[0].end <= out[1].start
    p = tmp_path / "auto.lrc"
    p.write_text(A.to_lrc(out, "t"), encoding="utf-8")
    back = load_lyrics(p)                                          # the export reads back as an enhanced LRC
    assert [round(ln.start, 2) for ln in back] == [round(ln.start, 2) for ln in out]
    assert [w.text for w in back[1].words] == ["cab", "ode"]


def _word_start(path, tk, word):
    states = [2 * k + 1 for k, w in enumerate(tk.word) if w == word]
    return np.nonzero(np.isin(path, states))[0][0] / FPS


def test_anchor_window_picks_the_right_repeat():
    # "ace bad" is sung twice but written once; the Whisper anchor says which time
    logp = emission(sung("ace", 1.0) + sung("bad", 1.4) + sung("ace", 5.0) + sung("bad", 5.4), 8.0)
    tk = A.build_tokens(["ace", "bad"], [False, False], [0, 0], VOCAB)
    for lo, hi, want in ((0.5, 2.0, 1.0), (4.5, 6.0, 5.0)):
        path = A.viterbi(logp, tk, FPS, {0: (lo, hi), 1: (lo + 0.4, hi + 0.4)})
        assert abs(_word_start(path, tk, 0) - want) < 0.05


def test_star_absorbs_unscripted_singing():
    # something sung between the lines that is not in the text ("ddd") must not drag "cab" onto it
    segs = sung("bad", 1.0) + sung("ddd", 2.5) + sung("cab", 4.0)
    out, _ = A.align_lines(A.parse_plain("bad\ncab\n"), emission(segs, 6.0), FPS, VOCAB, None, 6.0)
    assert abs(out[1].start - 4.0) < 0.05


def test_stray_first_letter_is_dropped():
    # an isolated "c" long before the rest of "cab" (a held note read as its first letter)
    segs = sung("bad", 0.5) + [("c", 1.5, 1.56)] + sung("cab", 3.0)
    out, _ = A.align_lines(A.parse_plain("bad\ncab\n"), emission(segs, 5.0), FPS, VOCAB, None, 5.0)
    assert abs(out[1].start - 3.0) < 0.15


def test_snap_moves_a_line_start_onto_the_onset():
    segs = sung("bad", 1.0) + sung("ace", 3.0) + sung("ode", 3.5)
    out, _ = A.align_lines(A.parse_plain("bad\nace ode\n"), emission(segs, 6.0), FPS, VOCAB, None, 6.0)
    env = np.zeros(6 * 48)
    env[int(2.8 * 48)] = 5.0                                       # the voice really starts at 2.8 s
    assert A.snap_starts(out, env, 48) >= 1
    assert abs(out[1].start - 2.8) < 0.03 and out[1].words[0].start == out[1].start


def test_skipped_backing_vocal_stays_with_its_line():
    # "(dab)" opens the second line but is not sung: the line starts at "ace", not halfway through the gap
    segs = sung("bad", 1.0) + sung("ace", 3.0)
    lines = A.parse_plain("bad\n(dab) ace\n")
    out, _ = A.align_lines(lines, emission(segs, 6.0), FPS, VOCAB, None, 6.0)
    assert abs(out[1].start - 3.0) < 0.05
    # ... unless Whisper heard it: then the line starts where Whisper heard it
    heard = [(1.0, 1.3, "bad"), (2.5, 2.7, "dab"), (3.0, 3.3, "ace")]
    out, _ = A.align_lines(lines, emission(segs, 6.0), FPS, VOCAB, heard, 6.0)
    assert abs(out[1].start - 2.5) < 0.05 and out[1].words[1].start > out[1].start


def test_vocables():
    assert all(map(A.vocable, ["Whoa-ooh,", "Shh", "(Mm,", "yeah", "nah", "oh", "Ooh", "hey!"]))
    assert not any(map(A.vocable, ["dab", "love", "you", "I", "hold", "oh-love"]))


def test_unheard_ad_lib_moves_to_the_strongest_onset():
    # "oh oh" between two sung lines leaves no trace in the CTC: it goes to the clear vocal onset
    segs = sung("bad", 0.5) + sung("cab", 5.0)
    out, _ = A.align_lines(A.parse_plain("bad\noh oh\ncab\n"), emission(segs, 7.0), FPS, VOCAB, None, 7.0,
                           onset_env=_env({3.4: 6.0, 2.0: 2.5}), onset_rate=48)
    assert abs(out[1].start - 3.4) < 0.03 and out[1].words[-1].start < out[2].start


def _env(peaks, seconds=7.0):
    env = np.zeros(int(seconds * 48))
    for t, z in peaks.items():
        env[int(t * 48)] = z
    return env


# ---------------------------------------------------------------- Chinese

try:
    import pypinyin  # noqa: F401
    no_pinyin = False
except ImportError:
    no_pinyin = True
needs_pinyin = pytest.mark.skipif(no_pinyin, reason="pypinyin not installed")


@needs_pinyin
def test_chinese_is_compared_as_pinyin():
    assert A.romanized(("哭", "過")) == A.romanized(("哭", "过")) == ("ku", "guo")      # traditional = simplified
    assert A.romanized(("長", "大")) == ("zhang", "da")                                # polyphone in context
    assert A.romanized(("遍", "地", "emo")) == ("bian", "di", "emo")
    assert A.keys_of(["女"]) == ["nu"]                                                # ü is written u
    assert A.text_line("(有懦弱過 也勇敢過)").words == ["(有", "懦", "弱", "過", "也", "勇", "敢", "過)"]


@needs_pinyin
def test_whisper_phrases_are_spread_over_their_characters():
    words, keys = A.split_heard([(1.0, 2.0, "也勇敢过"), (2.5, 3.0, "just")])
    assert [w for _, _, w in words] == ["也", "勇", "敢", "过", "just"]
    assert words[1][:2] == (1.25, 1.5) and keys == ["ye", "yong", "gan", "guo", "just"]


@needs_pinyin
def test_traditional_lyrics_anchor_on_simplified_whisper():
    lines = [A.text_line(s) for s in ["哭過 痛過", "遍地的連鎖"]]
    heard = [(1.0, 2.0, "哭过痛过"), (4.0, 5.5, "遍地的连锁")]
    win = A.anchors(lines, heard)
    assert set(win) == set(range(9))
    assert win[4][0] == pytest.approx(4.0 - A.EARLY)                  # 遍 where Whisper heard it


def test_heard_words_go_to_the_sung_line_before_a_backing_echo():
    # "ab cd (ab cd)" then "ab cd" again, heard twice: the second hearing belongs to the second line
    lines = [A.text_line("bad ode (bad ode)"), A.text_line("bad ode")]
    heard = [(1.0, 1.3, "bad"), (1.3, 1.6, "ode"), (5.0, 5.3, "bad"), (5.3, 5.6, "ode")]
    win = A.anchors(lines, heard)
    assert win[4][0] == pytest.approx(5.0 - A.EARLY) and 2 not in win


@needs_pinyin
def test_auto_lrc_keeps_chinese_characters_together(tmp_path):
    from ascii_pv.core.analysis import LyricLine, Word

    text = "哭過 痛過 emo"
    words = [Word(1.0 + i * 0.3, 1.2 + i * 0.3, w) for i, w in enumerate(["哭", "過", "痛", "過", "emo"])]
    lrc = A.to_lrc([LyricLine(1.0, 3.0, text, words)])
    assert "<00:01.00>哭<00:01.30>過 <00:01.60>痛" in lrc
    p = tmp_path / "auto.lrc"
    p.write_text(lrc, encoding="utf-8")
    back = load_lyrics(p)
    assert back[0].text == text and [w.text for w in back[0].words] == ["哭", "過", "痛", "過", "emo"]


def test_a_held_word_starts_where_it_was_heard():
    # "ace baaaad": CTC marks the onset of the held vowel and its end, blanks in between. Whisper heard "bad"
    # at 1.0 s, so the word starts there; with no anchor the long gap still reads as a stray first letter
    segs = sung("ace", 0.4) + [("b", 1.0, 1.06), ("a", 1.08, 1.14), ("d", 2.0, 2.06)]
    lines = A.parse_plain("ace bad\n")
    heard = [(0.4, 0.7, "ace"), (1.0, 2.1, "bad")]
    out, _ = A.align_lines(lines, emission(segs, 4.0), FPS, VOCAB, heard, 4.0)
    assert abs(out[0].words[1].start - 1.0) < 0.05
    out, _ = A.align_lines(lines, emission(segs, 4.0), FPS, VOCAB, None, 4.0)
    assert out[0].words[1].start > 1.5
