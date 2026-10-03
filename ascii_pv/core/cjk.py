"""Chinese / Japanese / Korean lyric text: where the "words" are.

The rest of the program times and lays out lyrics word by word, and English words are separated by spaces.
CJK text has no spaces between words, so every ideograph / kana / hangul syllable is its own token (one sung
syllable each); a run of Latin letters or digits inside it stays one token ("遍地 emo 的連鎖" -> 遍 地 emo 的 連 鎖).
Punctuation sticks to its neighbour: closing marks to the token before, opening marks to the token after.

Tokens of one space-separated chunk are "glued": they are drawn without a gap. English text has no glue, so
tokens(english) == english.split(" ") and nothing changes for it.
"""

from __future__ import annotations

from functools import lru_cache

# opening marks attach to the next token, everything else that is not a letter / digit / CJK to the previous one
OPENING = set("(（[［{｛「『【〔〈《“‘")


def is_cjk(ch: str) -> bool:
    """An ideograph, kana or hangul syllable: one sung syllable, drawn as a square glyph."""
    o = ord(ch)
    return (0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF or 0xF900 <= o <= 0xFAFF or 0x20000 <= o <= 0x2FA1F
            or 0x3040 <= o <= 0x30FF or 0x31F0 <= o <= 0x31FF or 0xAC00 <= o <= 0xD7AF or o == 0x3005)


def has_cjk(text: str) -> bool:
    return any(is_cjk(ch) for ch in text)


def has_kana(text: str) -> bool:
    """Japanese hiragana / katakana (Chinese has none)."""
    return any(0x3040 <= ord(ch) <= 0x30FF or 0x31F0 <= ord(ch) <= 0x31FF for ch in text)


def cjk_count(text: str) -> int:
    return sum(1 for ch in text if is_cjk(ch))


def _word_char(ch: str) -> bool:
    return ch.isalnum() and not is_cjk(ch)


def _split_chunk(chunk: str) -> list[str]:
    if not has_cjk(chunk):
        return [chunk]
    # 1. raw pieces: single CJK characters, runs of letters / digits (with ' and - inside), single other marks
    raw: list[str] = []
    i = 0
    while i < len(chunk):
        ch = chunk[i]
        if is_cjk(ch):
            raw.append(ch)
            i += 1
        elif _word_char(ch):
            j = i + 1
            while j < len(chunk) and (_word_char(chunk[j]) or (chunk[j] in "'-" and j + 1 < len(chunk)
                                                              and _word_char(chunk[j + 1]))):
                j += 1
            raw.append(chunk[i:j])
            i = j
        else:
            raw.append(ch)
            i += 1
    # 2. marks join a neighbour
    out: list[str] = []
    prefix = ""
    for p in raw:
        if is_cjk(p[0]) or _word_char(p[0]):
            out.append(prefix + p)
            prefix = ""
        elif p in OPENING or not out:
            prefix += p
        else:
            out[-1] += p
    if prefix:
        if out:
            out[-1] += prefix
        else:
            out.append(prefix)
    return out


@lru_cache(maxsize=4096)
def tokens(text: str) -> tuple[str, ...]:
    """Display / timing units of a lyric line (see the module docstring)."""
    out: list[str] = []
    for chunk in text.split(" "):
        if chunk:
            out.extend(_split_chunk(chunk))
    return tuple(out)


@lru_cache(maxsize=4096)
def glue(words: tuple[str, ...], text: str) -> tuple[bool, ...]:
    """Per word: True when it follows the previous word with no space in `text` (drawn without a gap).
    Words that cannot be found in order in the text are not glued."""
    out, pos, prev_end = [], 0, -1
    for w in words:
        at = text.find(w, pos)
        if at < 0:
            out.append(False)
            continue
        out.append(at == prev_end and bool(out))
        prev_end = pos = at + len(w)
    return tuple(out)


def word_count(text: str) -> float:
    """Roughly how many English-sized words a line holds (two CJK syllables ~ one word), for rules that were
    tuned on English lines."""
    cjk = cjk_count(text)
    if not cjk:
        return float(len(text.split(" ")))
    latin = sum(1 for t in tokens(text) if not is_cjk(t[0]) and any(_word_char(c) for c in t))
    return latin + cjk / 2
