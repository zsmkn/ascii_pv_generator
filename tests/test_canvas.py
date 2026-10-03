from ascii_pv.core.canvas import Canvas, crop, text_width


def test_put_and_plain():
    c = Canvas(10, 2)
    c.put(2, 0, "abc")
    assert c.plain().splitlines()[0] == "  abc     "


def test_clip_drops_outside():
    c = Canvas(10, 3)
    c.clip = (2, 1, 6, 2)
    c.put(0, 1, "0123456789")
    c.put(0, 0, "zzzz")
    rows = c.plain().splitlines()
    assert rows[0].strip() == ""
    assert rows[1] == "  2345    "


def test_wide_chars_take_two_cells():
    assert text_width("a愛b") == 4
    c = Canvas(6, 1)
    c.put(0, 0, "a愛b")
    assert c.get(1, 0) == "愛" and c.get(2, 0) == "" and c.get(3, 0) == "b"
    assert c.plain() == "a愛b  "


def test_overwriting_half_a_wide_char_blanks_the_other_half():
    c = Canvas(6, 1)
    c.put(0, 0, "愛")
    c.put(1, 0, "x")  # hits the continuation
    assert c.get(0, 0) == " " and c.get(1, 0) == "x"
    c.put(2, 0, "愛")
    c.put(2, 0, "y")  # hits the head
    assert c.get(2, 0) == "y" and c.get(3, 0) == " "


def test_wide_char_rewritten_in_place_survives():
    c = Canvas(4, 1)
    c.put(0, 0, "愛")
    c.put(0, 0, "愛")
    assert c.get(0, 0) == "愛" and c.get(1, 0) == ""


def test_crop():
    assert crop("a愛b", 2) == "a"
    assert crop("a愛b", 3) == "a愛"


def test_ansi_skips_continuations():
    c = Canvas(4, 1)
    c.put(0, 0, "愛")
    out = c.ansi()
    assert out.count("愛") == 1
