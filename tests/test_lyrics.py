from ascii_pv.analysis.lyrics import clean, lemma, load_lyrics, motifs_of, parse_lrc, parse_srt, syllables


def write(tmp_path, name, text, encoding="utf-8"):
    p = tmp_path / name
    p.write_bytes(text.encode(encoding))
    return p


def test_parse_lrc_basic_meta_and_end_markers():
    entries, meta = parse_lrc("[ti:Song]\n[00:01.50]Hello world\n[00:03.00]\n[01:02.345]Next line\n")
    assert meta["ti"] == "Song"
    assert [(e["t"], e["text"]) for e in entries] == [(1.5, "Hello world"), (3.0, ""), (62.345, "Next line")]


def test_parse_lrc_multi_stamp_and_offset():
    entries, _ = parse_lrc("[offset:500]\n[00:10.00][00:20.00]Chorus\n")
    assert [e["t"] for e in entries] == [9.5, 19.5]


def test_parse_lrc_enhanced_word_times():
    entries, _ = parse_lrc("[00:01.00]<00:01.00>Hello <00:01.60>big <00:02.10>world\n")
    assert entries[0]["text"] == "Hello big world"
    assert entries[0]["words"] == [(1.0, "Hello"), (1.6, "big"), (2.1, "world")]


def test_parse_srt():
    srt = "1\n00:00:01,000 --> 00:00:03,500\n<i>Hello</i>\nworld\n\n2\n00:00:04,000 --> 00:00:05,000\nBye\n"
    assert parse_srt(srt) == [{"t": 1.0, "end": 3.5, "text": "Hello world"},
                              {"t": 4.0, "end": 5.0, "text": "Bye"}]


def test_clean_typographic_punctuation():
    assert clean("‘Cause we’re  “here” — now…") == "'Cause we're \"here\" - now..."


def test_load_lyrics_timing_hooks_motifs(tmp_path):
    p = write(tmp_path, "a.lrc", "﻿[00:01.00]I love the fire\n[00:03.00]Burning, burning\n"
                                 "[00:05.00]I love the fire\n[00:30.00]Last\n")
    lines = load_lyrics(p, duration=40)
    assert [ln.text for ln in lines] == ["I love the fire", "Burning, burning", "I love the fire", "Last"]
    assert lines[0].end == 3.0                      # held until the next line
    assert lines[2].end < 30.0                      # capped across the instrumental gap
    assert lines[0].hook and lines[2].hook and not lines[1].hook
    assert lines[0].motifs == ["love", "fire"] and lines[1].motifs == ["fire"]
    for ln in lines:                                # words fill the line in order
        assert ln.words[0].start == ln.start
        assert all(a.start < b.start for a, b in zip(ln.words, ln.words[1:]))
        assert ln.words[-1].end <= ln.end


def test_offset_scale_and_duration_clip(tmp_path):
    p = write(tmp_path, "a.lrc", "[00:10.00]one\n[00:20.00]two\n[00:50.00]gone\n")
    lines = load_lyrics(p, duration=40, offset=1.0, scale=0.5)
    assert [(ln.start, ln.text) for ln in lines] == [(6.0, "one"), (11.0, "two"), (26.0, "gone")]
    lines = load_lyrics(p, duration=30)
    assert [ln.text for ln in lines] == ["one", "two"]


def test_non_utf8_falls_back(tmp_path):
    p = write(tmp_path, "a.lrc", "[00:01.00]caf\xe9 night\n", encoding="cp1252")
    assert load_lyrics(p)[0].text == "café night"


def test_onset_snapping(tmp_path):
    p = write(tmp_path, "a.lrc", "[00:00.00]one two three four\n[00:04.00]x\n")
    free = load_lyrics(p)[0].words
    snapped = load_lyrics(p, onsets=[free[1].start + 0.05])[0].words
    assert abs(snapped[1].start - (free[1].start + 0.05)) < 1e-6


def test_syllables_and_lemma():
    assert syllables("love") == 1 and syllables("vitality") >= 3 and syllables("fire") == 1
    assert lemma("Burning,") == "burn" and lemma("stars") == "star" and lemma("running") == "run"
    assert motifs_of("Pop, pop, sugar on top") == ["pop"]
