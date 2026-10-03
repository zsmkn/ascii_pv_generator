import numpy as np

from ascii_pv.analysis.sync import fine, fit, onset_envelope, refine, vocal_activity

RATE = 48


def fake_vocals(starts, duration, rng, lens=None):
    """Vocal RMS: silence with a little noise, each sung line a loud block of 1.2-2 s."""
    rms = 0.002 + 0.001 * rng.random(int(duration * RATE))
    for i, s in enumerate(starts):
        a = int(s * RATE)
        L = lens[i] if lens is not None else rng.uniform(1.2, 2.0)
        rms[a:a + int(L * RATE)] = 0.2 + 0.05 * rng.random()
    return rms


def test_recovers_scale_and_offset():
    rng = np.random.default_rng(0)
    true = np.cumsum(rng.uniform(2.0, 4.5, 40)) + 5.0          # irregular like real lyrics
    lens = rng.uniform(1.2, 2.0, 40)
    rms = fake_vocals(true, true[-1] + 10, rng, lens)
    scale, offset = 1.08, -2.4
    file_s = (true - offset) / scale                             # what a mistimed LRC would say
    file_e = (true + lens - offset) / scale
    res = fit(list(file_s), list(file_e), vocal_activity(rms, RATE), RATE, trials=6)
    assert res.applied and res.z > 3
    assert abs(res.scale - scale) < 0.01 and abs(res.offset - offset) < 0.3


def test_refuses_when_there_is_nothing_to_find():
    rng = np.random.default_rng(1)
    rms = 0.002 + 0.2 * rng.random(200 * RATE)                  # noise, no structure
    starts = np.cumsum(rng.uniform(2, 4, 40))
    assert not fit(list(starts), list(starts + 1.5), vocal_activity(rms, RATE), RATE, trials=6).applied


def fake_words(starts, duration, rng):
    """Vocal RMS with word-level onsets: every line is 4-7 words separated by short dips, like real singing."""
    rms = 0.002 + 0.001 * rng.random(int(duration * RATE))
    for s in starts:
        t = s
        for _ in range(rng.integers(4, 8)):
            L = rng.uniform(0.15, 0.4)
            rms[int(t * RATE):int((t + L) * RATE)] = 0.1 + 0.15 * rng.random()
            t += L + rng.uniform(0.05, 0.15)
    return rms


def test_fine_tightens_a_slightly_wrong_scale():
    rng = np.random.default_rng(3)
    true = np.cumsum(rng.uniform(2.0, 4.5, 50)) + 5.0
    env = onset_envelope(fake_words(true, true[-1] + 10, rng), RATE)
    res = fine(list(true), env, RATE, 0.99, 0.5, trials=6)        # coarse fit ~1 % off: >1 s drift at the end
    assert res.applied
    assert abs(res.scale - 1.0) < 0.001 and abs(res.offset) < 0.05


def test_fine_refuses_inaccurate_timing():
    rng = np.random.default_rng(4)
    true = np.cumsum(rng.uniform(2.0, 4.5, 50)) + 5.0
    env = onset_envelope(fake_words(true, true[-1] + 10, rng), RATE)
    sloppy = true + rng.uniform(-0.8, 0.8, len(true))           # right on average, wrong per line (Gemini-like)
    assert not fine(list(sloppy), env, RATE, 1.0, 0.0, trials=6).applied


def test_refine_snaps_and_keeps_order():
    rng = np.random.default_rng(2)
    true = [5.0, 9.0, 13.5, 13.9]
    env = onset_envelope(fake_vocals(true, 20, rng), RATE)
    out = refine([5.25, 8.8, 13.6, 13.6], env, RATE)
    assert abs(out[0] - 5.0) < 0.06 and abs(out[1] - 9.0) < 0.06
    assert all(b > a for a, b in zip(out, out[1:]))


def test_shift_lines_moves_words_and_trims_ends():
    from ascii_pv.analysis.pipeline import _shift_lines
    from ascii_pv.core.analysis import LyricLine, Word

    lines = [LyricLine(1.0, 3.0, "a b", [Word(1.0, 2.0, "a"), Word(2.0, 3.0, "b")]),
             LyricLine(3.0, 5.0, "c", [Word(3.0, 5.0, "c")])]
    _shift_lines(lines, [1.5, 3.2], duration=6.0)
    assert (lines[0].start, lines[0].end) == (1.5, 3.2)
    assert [w.start for w in lines[0].words] == [1.5, 2.5] and lines[0].words[-1].end <= 3.2
    assert lines[1].start == 3.2 and lines[1].end == 5.2


def test_clap_merge_adds_tags_and_nudges_valence():
    from ascii_pv.analysis.ml.clap import merge_into_style

    style = {"energy": 0.5, "valence": 0.5, "tags": ["mid-energy"]}
    clap = {"genre": [["ambient", 0.6], ["rock", 0.1]], "mood": [["sad", 0.7], ["happy", 0.05]]}
    out = merge_into_style(style, clap)
    assert "calm" in out["tags"] and "dark" in out["tags"] and "energetic" not in out["tags"]
    assert out["valence"] < 0.5 and out["clap"] is clap


def test_shifted_line_keeps_its_words_inside():
    from ascii_pv.analysis.lyrics import estimate_words
    from ascii_pv.analysis.pipeline import _shift_lines
    from ascii_pv.core.analysis import LyricLine

    lines = [LyricLine(10.0, 12.0, "捉摸不透的話", estimate_words("捉摸不透的話", 10.0, 12.0)),
             LyricLine(12.0, 14.0, "next one", estimate_words("next one", 12.0, 14.0))]
    _shift_lines(lines, [10.6, 12.0], 20.0)                  # the first line starts later, the second does not
    ln = lines[0]
    assert ln.start == 10.6 and ln.end == 12.0
    assert ln.words[0].start == 10.6 and ln.words[-1].start < ln.end
    assert all(a.start < b.start for a, b in zip(ln.words, ln.words[1:]))
