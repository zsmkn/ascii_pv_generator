"""Plain-text lyrics -> timed lines and words, aligned to the separated vocals (M9).

Two models look at the vocal track (analysis/ml/asr.py runs them; everything here is numpy):

  CTC   a wav2vec2 aligner (torchaudio MMS_FA) gives per-frame character log-probabilities. Forced alignment of
        the known text on them is precise where the voice is clear, but in stacked / processed choruses the model
        barely hears the words and the alignment drifts freely.
  ASR   Whisper transcribes the vocals with word timestamps. It hears sung words far better, but its timestamps
        are loose (a word after a pause often starts where the pause starts) and it drops or invents lines.

So Whisper decides roughly *where* each word is and CTC decides *exactly when*:

  1. match     the Whisper words are matched to the lyric words (edit-distance DP, fuzzy word equality). A lyric
               word matched exactly next to another exact match, or nearly ("blink" / "link") between two matched
               words, becomes an anchor: [whisper start - EARLY, whisper end + LATE]. Lines repeated back to back
               are matched again as a group (see anchors).
  2. viterbi   our own CTC Viterbi over blank / character states. Leaving an anchor window costs PENALTY nats per
               second (a soft prior: clear acoustic evidence still wins). Parenthesised words (backing vocals)
               and a "something else is sung here" star token between lines are optional: skip edges jump them.
               The star has a fixed per-frame cost, unlike torchaudio's (log-prob 0, which lets it swallow
               the real words and squeeze them into single frames).
  3. assemble  word and line times from the path; skipped / untokenisable words are interpolated (skipped words
               at the edges of a line stay with it); a stray first letter (or first word) far ahead of the rest
               is dropped.
  4. snap      each line start moves onto a clear vocal onset between the previous line and its second word;
               an ad-lib line neither model heard ("Whoa-ooh", "Shh") goes to the strongest onset in its gap.

Measured against Spotify LRCs (text only as input, one global offset per song for the different audio edits):
Invisible Things / Like I Do / Deserve You / Happier: median line-start error 0.05-0.12 s, 74-93 % within 0.3 s,
91-98 % within 0.5 s. The parameters were tuned on the first song only and sit on flat ground (halving or
doubling PENALTY, STAR_COST, EARLY changes nothing). What is left: ad-lib lines in a run of ad-libs, backing
vocals overlapping the lead (where "when the line starts" is itself ambiguous), and text that is not in the
audio at all (a music-video edit with cut sections pulls the unanchored lines around it).
On the Gemini-made LRCs (timing drifts by seconds mid-song) aligning their text beats their own timestamps.

Chinese: every character is a word (core/cjk.py). Both models get the text as toneless pinyin (pypinyin, whole
line at once so polyphones read in context): MMS_FA was trained on romanised text, and comparing sounds lets the
traditional lyrics match Whisper's simplified output and its homophone slips (恨過 heard as 寒过). Whisper
returns space-separated phrases for Chinese; each is spread evenly over its characters (anchors are windows,
CTC sets the exact times). Japanese (kana) is not supported: it would need its own romanisation.
"""

from __future__ import annotations

import difflib
from functools import lru_cache
import re
from dataclasses import dataclass, field

import numpy as np

from ascii_pv.analysis.lyrics import META, TS, WORD_TS, clean
from ascii_pv.core.cjk import glue, has_cjk, is_cjk, tokens

STAR_COST = -2.5      # log-prob per frame of "sung, but not in the text"
PENALTY = 4.0         # nats per second outside an anchor window
EARLY, LATE = 0.5, 0.6
MAX_INTRA = 0.35     # s: a longer silence between two characters of one word means the first one was a stray
HELD_SLACK = 0.15    # s: ... unless it lies where Whisper heard the word (at most this much before its start)
LINE_GAP = 1.2       # s: ... and between the first and second word of a line
SNAP_BACK = 1.0      # s: how far before the aligned start a line's onset may be searched
SNAP_MIN = 2.0       # onset strength (z) that counts as clear
LOW = 0.05           # line confidence below this is reported as low
OPTIONAL_WEIGHT = 0.75   # a heard word matched to a parenthesised (optional) lyric word is worth this much
HOLD = 2.0           # a line stays up at most this long after its last word (unless the next line comes first)

# ---------------------------------------------------------------- text

SECTION = re.compile(r"^\s*[\[(]?\s*(intro|verse|pre-?chorus|chorus|hook|bridge|outro|refrain|interlude|break|"
                     r"post-?chorus|drop|instrumental)\b[^\])]*[\])]?\s*:?\s*$", re.I)
HEADER = re.compile(r"^\s*\[([^\]]*)\]\s*$")


@dataclass
class TextLine:
    text: str                                    # display text (cleaned)
    words: list[str]                             # display tokens (text.split(" "))
    optional: list[bool]                         # per word: inside parentheses (backing vocals, ad-libs)
    section: str | None = None                   # the [Chorus] / Verse 2: header above it, if any
    stanza: int = 0                              # blank-line separated block index


def has_timestamps(text: str) -> bool:
    return any(TS.match(ln.strip()) for ln in text.splitlines())


def is_plain_text(path) -> bool:
    """A lyric file without any timing: .txt, or an LRC / unknown file with no timestamps (SRT always has them)."""
    from pathlib import Path

    from ascii_pv.analysis.lyrics import SRT_TIME, read_text

    text = read_text(path)
    if Path(path).suffix.lower() == ".srt" or SRT_TIME.search(text):
        return False
    return not has_timestamps(text)


def parse_plain(text: str) -> list[TextLine]:
    """One lyric line per text line. Blank lines separate stanzas; section headers are kept as hints;
    LRC metadata tags and stray (empty) timestamps are dropped."""
    out: list[TextLine] = []
    section, stanza, gap = None, 0, False
    for raw in text.splitlines():
        s = raw.strip()
        if not s:
            gap = True
            continue
        if META.match(s) and not TS.match(s):
            continue                                                  # [ar:Lauv] (a header has no "tag:")
        if (m := HEADER.match(s)) or SECTION.match(s):
            section = (m.group(1) if m else s).strip("[]():  ")
            gap = True
            continue
        s = clean(WORD_TS.sub("", TS.sub("", s)))
        if not s:
            continue
        if gap and out:
            stanza += 1
        gap = False
        out.append(text_line(s, section, stanza))
    return out


def text_line(s: str, section: str | None = None, stanza: int = 0) -> TextLine:
    words = list(tokens(s))
    opt, depth = [], 0
    for w in words:                                      # "(I love the most, I love the most)" -> all optional
        opt.append(depth > 0 or w.startswith("("))
        depth = max(0, depth + w.count("(") - w.count(")"))
    return TextLine(s, words, opt, section, stanza)


def text_lines(path) -> list[TextLine]:
    """The lyric text of any lyric file, timing ignored (plain text, or `--retime` of an LRC / SRT). An LRC line
    stamped several times ("[00:10][01:20]chorus") is sung several times, so it is expanded in time order."""
    from pathlib import Path

    from ascii_pv.analysis.lyrics import SRT_TIME, parse_lrc, parse_srt, read_text

    text = read_text(path)
    if Path(path).suffix.lower() == ".srt" or SRT_TIME.search(text):
        return [text_line(e["text"]) for e in parse_srt(text)]
    if has_timestamps(text):
        return [text_line(e["text"]) for e in parse_lrc(text)[0] if e["text"]]
    return parse_plain(text)


_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen " \
        "seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        return _ONES[n // 100] + " hundred" + ("" if n % 100 == 0 else " " + number_words(n % 100))
    return " ".join(_ONES[int(d)] for d in str(n))


def spoken(word: str) -> str:
    """Display word -> what is sung, lowercase letters and apostrophes ("24/7" -> "twenty four seven")."""
    w = word.lower().replace("&", " and ")
    w = re.sub(r"\d+", lambda m: " " + number_words(int(m.group())) + " ", w)
    w = re.sub(r"[^a-z' ]", " ", w)
    return re.sub(r"\s+", " ", w).strip(" '")


def word_key(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", spoken(word))


def _pinyin(han: str) -> list[str]:
    """Toneless pinyin, one syllable per character of `han` (CJK characters only); ü is written u."""
    from pypinyin import lazy_pinyin

    py = lazy_pinyin(han, errors=lambda x: [""] * len(x))
    if len(py) != len(han):                                  # odd input: character by character
        py = [(lazy_pinyin(ch, errors=lambda x: [""] * len(x)) or [""])[0] for ch in han]
    return [x.replace("v", "u") for x in py]


@lru_cache(maxsize=4096)
def romanized(words: tuple[str, ...]) -> tuple[str, ...]:
    """What each word of a line sounds like in Latin letters: spoken() for English, pinyin for Chinese
    characters (computed over the whole line so polyphones get their context)."""
    if not any(has_cjk(w) for w in words):
        return tuple(spoken(w) for w in words)
    py = iter(_pinyin("".join(ch for w in words for ch in w if is_cjk(ch))))
    out = []
    for w in words:
        parts = [next(py) if is_cjk(ch) else ch for ch in w]
        latin = spoken("".join(p if not is_cjk(ch) else " " for p, ch in zip(parts, w)))
        han = " ".join(p for p, ch in zip(parts, w) if is_cjk(ch) and p)
        out.append(" ".join(x for x in (han, latin) if x))
    return tuple(out)


def keys_of(words) -> list[str]:
    return [re.sub(r"[^a-z0-9]", "", r) for r in romanized(tuple(words))]


def split_heard(heard: list[tuple[float, float, str]]) -> tuple[list[tuple[float, float, str]], list[str]]:
    """Whisper's words, Chinese phrases spread evenly over their characters -> (words, keys)."""
    out, keys = [], []
    for s, e, text in heard:
        toks = tokens(text) if has_cjk(text) else (text,)
        if not toks:
            continue
        step = (e - s) / len(toks)
        for i, tk in enumerate(toks):
            out.append((s + i * step, s + (i + 1) * step, tk))
        keys.extend(keys_of(toks))
    return out, keys


# ---------------------------------------------------------------- 1. match Whisper words

def _similar(a: str, b: str) -> int:
    """2 = same word, 1 = probably the same word misheard, 0 = different."""
    if not a or not b:
        return 0
    if a == b:
        return 2
    return 1 if difflib.SequenceMatcher(None, a, b).ratio() >= 0.6 else 0


def match_words(lyric: list[str], heard: list[str], gap: float = -0.6,
                weight: list[float] | None = None) -> list[tuple[int, int, int]]:
    """Global alignment of lyric word keys with heard word keys -> [(lyric i, heard j, similarity 1|2)].
    weight: per lyric word, scales what a match is worth (optional words count less, so a heard word that
    could be either the sung line or a backing echo of it goes to the line)."""
    n, m = len(lyric), len(heard)
    ul, li = np.unique(np.array(lyric, dtype=object), return_inverse=True) if n else ([], [])
    uh, hi = np.unique(np.array(heard, dtype=object), return_inverse=True) if m else ([], [])
    usim = np.array([[_similar(a, b) for b in uh] for a in ul], dtype=np.int8).reshape(len(ul), len(uh))
    sim = usim[np.asarray(li)[:, None], np.asarray(hi)[None, :]] if n and m else np.zeros((n, m), np.int8)
    gain = np.where(sim == 2, 2.0, np.where(sim == 1, 1.0, -1.0))
    if weight is not None:
        gain = np.where(gain > 0, gain * np.asarray(weight, float)[:, None], gain)
    D = np.zeros((n + 1, m + 1))
    D[:, 0] = np.arange(n + 1) * gap
    D[0, :] = np.arange(m + 1) * gap
    P = np.zeros((n + 1, m + 1), np.int8)
    for i in range(1, n + 1):                                   # row-wise: the left move is a running max
        diag = D[i - 1, :-1] + gain[i - 1]
        up = D[i - 1, 1:] + gap
        row = np.maximum(diag, up)
        src = np.where(diag >= up, 0, 1).astype(np.int8)
        for j in range(1, m + 1):
            left = D[i, j - 1] + gap
            if left > row[j - 1]:
                D[i, j], P[i, j] = left, 2
            else:
                D[i, j], P[i, j] = row[j - 1], src[j - 1]
    out = []
    i, j = n, m
    while i > 0 and j > 0:
        if P[i, j] == 0:
            if sim[i - 1, j - 1]:
                out.append((i - 1, j - 1, int(sim[i - 1, j - 1])))
            i, j = i - 1, j - 1
        elif P[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return out[::-1]


def _trusted(pairs: list[tuple[int, int, int]]) -> dict[int, int]:
    """Trusted pairs: an exact match next to another exact match, or a near match (misheard / misspelt:
    "blink" vs "link") with matched words on both sides. Lone matches are often a common word paired by chance."""
    got = {i: (j, s) for i, j, s in pairs}

    def linked(i, j, d):
        return i + d in got and got[i + d][0] == j + d

    out = {}
    for i, (j, s) in got.items():
        if s == 2 and ((linked(i, j, -1) and got[i - 1][1] == 2) or (linked(i, j, 1) and got[i + 1][1] == 2)):
            out[i] = j
        elif s == 1 and linked(i, j, -1) and linked(i, j, 1):
            out[i] = j
    return out


def anchors(lines: list[TextLine], heard: list[tuple[float, float, str]]) -> dict[int, tuple[float, float]]:
    """Lyric word index -> (lo, hi) seconds where it should be sung, from the Whisper words.

    A line repeated right next to itself ("So, let me hold you close" x2) is ambiguous: Whisper may hear only
    one of the copies, and the global matching can hand it to the wrong one. Such a group is matched again
    on its own, against the Whisper words between its anchored neighbours, and keeps anchors only if every
    copy is clearly there (otherwise the neighbours alone bound it)."""
    heard, hk = split_heard(heard)
    line_keys = [keys_of(ln.words) for ln in lines]
    keys = [k for lk in line_keys for k in lk]
    line_of = [li for li, ln in enumerate(lines) for _ in ln.words]
    first = np.cumsum([0] + [len(ln.words) for ln in lines])
    core = [" ".join(k for k, o in zip(lk, ln.optional) if not o) for lk, ln in zip(line_keys, lines)]
    amb = [bool(core[li]) and any(core[k] == core[li] for k in range(max(0, li - 2), min(len(lines), li + 3))
                                  if k != li) for li in range(len(lines))]
    wt = [OPTIONAL_WEIGHT if o else 1.0 for ln in lines for o in ln.optional]
    match = {i: j for i, j in _trusted(match_words(keys, hk, weight=wt)).items() if not amb[line_of[i]]}

    li = 0
    while li < len(lines):
        if not amb[li]:
            li += 1
            continue
        g1 = li
        while g1 + 1 < len(lines) and amb[g1 + 1]:
            g1 += 1
        w0, w1 = first[li], first[g1 + 1]
        j0 = max((j for i, j in match.items() if i < w0), default=-1)
        j1 = min((j for i, j in match.items() if i >= w1), default=len(hk))
        idx = [i for i in range(w0, w1) if keys[i]]
        local = _trusted(match_words([keys[i] for i in idx], hk[j0 + 1:j1], weight=[wt[i] for i in idx]))
        got = {idx[a]: j0 + 1 + b for a, b in local.items()}
        ok = all(sum(1 for i in range(first[k], first[k + 1]) if i in got) * 2 >= len(lines[k].words)
                 for k in range(li, g1 + 1))
        if ok:
            match.update(got)
        li = g1 + 1
    return {i: (heard[j][0] - EARLY, heard[j][1] + LATE) for i, j in match.items()}


# ---------------------------------------------------------------- 2. CTC Viterbi

@dataclass
class Tokens:
    ids: list[int] = field(default_factory=list)          # vocabulary index per token (-1 = star)
    word: list[int] = field(default_factory=list)         # lyric word index per token (-1 = star)
    optional: list[bool] = field(default_factory=list)


def build_tokens(keys: list[str], optional: list[bool], line_of: list[int], vocab: dict[str, int]) -> Tokens:
    """Characters of every word in order, with an optional star between lines (and at both ends)."""
    tk = Tokens()

    def star():
        tk.ids.append(-1)
        tk.word.append(-1)
        tk.optional.append(True)

    star()
    for wi, key in enumerate(keys):
        if wi and line_of[wi] != line_of[wi - 1]:
            star()
        for c in key:
            if c in vocab:
                tk.ids.append(vocab[c])
                tk.word.append(wi)
                tk.optional.append(optional[wi])
    star()
    return tk


def optional_units(tk: Tokens) -> list[tuple[int, int]]:
    """[(first token, last token)] of each skippable unit, in order."""
    units = []
    k = 0
    while k < len(tk.ids):
        if not tk.optional[k]:
            k += 1
            continue
        v = k
        while tk.word[k] >= 0 and v + 1 < len(tk.ids) and tk.word[v + 1] == tk.word[k]:
            v += 1
        units.append((k, v))
        k = v + 1
    return units


def viterbi(logp: np.ndarray, tk: Tokens, fps: float, windows: dict[int, tuple[float, float]],
            blank: int = 0) -> np.ndarray:
    """Best CTC path -> state index per frame. States: 2k = blank before token k, 2k+1 = token k, 2L = final blank.
    windows: word index -> (lo, hi) seconds (soft)."""
    T = len(logp)
    L = len(tk.ids)
    S = 2 * L + 1
    ids = np.array(tk.ids)
    state_tok = np.full(S, blank)
    state_tok[1::2] = np.where(ids < 0, blank, ids)
    is_star = np.zeros(S, bool)
    is_star[1::2] = ids < 0
    # soft windows per state, in frames
    lo = np.full(S, -np.inf)
    hi = np.full(S, np.inf)
    for k, w in enumerate(tk.word):
        if w in windows:
            lo[2 * k + 1], hi[2 * k + 1] = windows[w][0] * fps, windows[w][1] * fps
    has_win = np.isfinite(lo)
    pen_rate = PENALTY / fps
    # the s-2 jump (skip the blank between two tokens) is allowed between different tokens
    allow2 = np.zeros(S, bool)
    allow2[3::2] = ids[1:] != ids[:-1]
    # skip edges: each optional unit (a star, or the tokens of one optional word) can be jumped, from the blank
    # before it or the token before it, to the blank after it. Consecutive units chain (one unit's target is the
    # next one's source).
    units = optional_units(tk)
    src_a = np.full(S, -1)
    src_b = np.full(S, -1)
    for u, v in units:
        src_a[2 * (v + 1)] = 2 * u
        src_b[2 * (v + 1)] = 2 * u - 1 if u > 0 else -1
    has_a = src_a >= 0
    has_b = src_b >= 0
    ia = np.where(has_a, src_a, 0)
    ib = np.where(has_b, src_b, 0)

    NEG = -1e18
    bp = np.zeros((T, S), np.int8)
    cur = np.full(S, NEG)

    def emit(t: int) -> np.ndarray:
        e = logp[t, state_tok]
        e = np.where(is_star, STAR_COST, e)
        if has_win.any():
            out = np.maximum(0.0, np.maximum(lo - t, t - hi))
            e = e - np.where(has_win, out * pen_rate, 0.0)
        return e

    # start in the first blank / token, or past leading optional units; end likewise at the other side
    start = np.zeros(S, bool)
    start[:2] = True
    end = np.zeros(S, bool)
    end[-2:] = True
    by_start = {u: v for u, v in units}
    by_end = {v: u for u, v in units}
    k = 0
    while k in by_start:
        k = by_start[k] + 1
        start[2 * k: 2 * k + 2] = True
    k = L - 1
    while k in by_end:
        k = by_end[k] - 1
        end[max(0, 2 * k + 1): 2 * k + 3] = True
    cur[start] = 0.0
    cur = cur + emit(0)
    for t in range(1, T):
        c1 = np.concatenate([[NEG], cur[:-1]])
        c2 = np.where(allow2, np.concatenate([[NEG, NEG], cur[:-2]]), NEG)
        c3 = np.where(has_a, cur[ia], NEG)
        c4 = np.where(has_b, cur[ib], NEG)
        stack = np.stack([cur, c1, c2, c3, c4])
        arg = stack.argmax(0)
        bp[t] = arg
        cur = stack[arg, np.arange(S)] + emit(t)

    s = int(np.argmax(np.where(end, cur, NEG)))
    path = np.empty(T, np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        b = bp[t, s]
        s = s if b == 0 else s - 1 if b == 1 else s - 2 if b == 2 else src_a[s] if b == 3 else src_b[s]
    return path


# ---------------------------------------------------------------- 3. assemble

@dataclass
class AlignInfo:
    words: int = 0
    anchored: int = 0              # lyric words with a Whisper anchor
    skipped: int = 0               # optional words the path jumped
    snapped: int = 0               # line starts moved onto a vocal onset
    line_score: list[float] = field(default_factory=list)   # mean char probability per line (0..1)

    def describe(self) -> str:
        sc = np.array(self.line_score) if self.line_score else np.zeros(1)
        return (f"lyric align: {self.words} words, {self.anchored} anchored by ASR, {self.skipped} optional "
                f"skipped, {self.snapped} starts snapped to onsets, line confidence median {np.median(sc):.2f}, "
                f"{int((sc < LOW).sum())} low")


def _fill(times: list[float | None], lo: float, hi: float) -> list[float]:
    """Interpolate missing times between known neighbours (lo / hi bound the ends)."""
    known = [i for i, t in enumerate(times) if t is not None]
    out = list(times)
    for i, t in enumerate(times):
        if t is not None:
            continue
        a = max((k for k in known if k < i), default=None)
        b = min((k for k in known if k > i), default=None)
        ta = times[a] if a is not None else lo
        tb = times[b] if b is not None else hi
        ia = a if a is not None else -1
        ib = b if b is not None else len(times)
        out[i] = ta + (tb - ta) * (i - ia) / (ib - ia)
    return out


def snap_starts(out, env: np.ndarray, rate: int) -> int:
    """Move each line's first word onto a clear vocal onset. CTC knows which word is where, but the first letter
    of a line is often pulled back onto the tail of the previous phrase. The onset must lie after the previous
    line's last word and before this line's second word; a one-syllable first word takes the latest such onset
    (its own, right before the second word), a longer one the earliest. -> number of lines moved."""
    from ascii_pv.analysis.lyrics import syllables

    moved = 0
    prev_end = 0.0
    for ln in out:
        ws = ln.words
        s0 = ws[0].start
        lo = max(prev_end, s0 - SNAP_BACK)
        hi = ws[1].start - 0.06 if len(ws) > 1 else s0 + 0.4
        a, b = int(lo * rate), min(len(env) - 1, int(hi * rate))
        if b - a > 2:
            seg = env[a:b]
            pk = np.nonzero((seg[1:-1] > SNAP_MIN) & (seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]))[0] + 1
            if len(pk):
                k = pk[-1] if syllables(ws[0].text) == 1 else pk[0]
                new = round((a + k) / rate, 3)
                if abs(new - s0) > 1e-3:
                    moved += 1
                ws[0].start = ln.start = new
                ws[0].end = max(ws[0].end, round(new + 0.05, 3))
        last = ws[-1]
        prev_end = min(last.end, last.start + 0.5)          # a word's end is loose (held note / trailing blank)
    return moved


VOCABLES = re.compile(r"(o+h*|a+h+|u+h+|e+h+|m+|h+m+|w+h*o+a*h*|w+o+|y+e+a+h*|n+a+h*|l+a+|h+e+y+|s+h+|h+a+|a+y+|o+o+h*)")


def vocable(word: str) -> bool:
    """An ad-lib sound rather than a word: "Whoa-ooh", "Mm", "Shh", "yeah", "nah"."""
    parts = [p for p in re.split(r"[^a-z]+", word.lower()) if p]
    return bool(parts) and all(VOCABLES.fullmatch(p) for p in parts)


def place_unheard(out, unheard: set[int], env: np.ndarray, rate: int) -> None:
    """Ad-lib lines neither model heard ("Whoa-ooh", "Shh") only sit where their neighbours squeezed them.
    Put each on the strongest vocal onset between the previous line's last word and the next line."""
    for k, ln in enumerate(out):
        if id(ln) not in unheard or k == 0 or k + 1 >= len(out):
            continue
        last = out[k - 1].words[-1]
        lo = min(last.end, last.start + 0.5)
        hi = out[k + 1].start - 0.2
        a, b = int(lo * rate), min(len(env) - 1, int(hi * rate))
        if b - a < 3:
            continue
        p = int(np.argmax(env[a:b]))
        if env[a + p] < SNAP_MIN:
            continue
        d = (a + p) / rate - ln.start
        for w in ln.words:
            w.start = round(min(w.start + d, hi), 3)
            w.end = round(max(w.start + 0.05, min(w.end + d, hi + 0.15)), 3)
        ln.start = ln.words[0].start


def align_lines(lines: list[TextLine], logp: np.ndarray, fps: float, vocab: dict[str, int],
                heard: list[tuple[float, float, str]] | None, duration: float,
                onset_env: np.ndarray | None = None, onset_rate: int = 48):
    """-> (list[LyricLine], AlignInfo). logp: (frames, vocab) CTC log-probs of the vocal track; onset_env:
    sync.onset_envelope of the vocal level (for snap_starts)."""
    from collections import Counter

    from ascii_pv.analysis.lyrics import _key, motifs_of
    from ascii_pv.core.analysis import LyricLine, Word

    keys, opt, line_of = [], [], []
    for li, ln in enumerate(lines):
        for r, o in zip(romanized(tuple(ln.words)), ln.optional):
            keys.append(r.replace(" ", ""))
            opt.append(o)
            line_of.append(li)
    info = AlignInfo(words=len(keys))
    win = anchors(lines, heard) if heard else {}
    info.anchored = len(win)
    # a line wholly in parentheses is sung for real when Whisper clearly heard it and it is not an echo of the
    # lead next to it (an intro hook, not "(I hold you close)" behind "let me hold you close", where what
    # Whisper heard was the lead): then it may not be skipped
    first = 0
    line_keys = [set(keys_of(ln.words)) - {""} for ln in lines]
    for li, ln in enumerate(lines):
        idx = range(first, first + len(ln.words))
        first += len(ln.words)
        near = (line_keys[li - 1] if li else set()) | (line_keys[li + 1] if li + 1 < len(lines) else set())
        echo = 2 * len(line_keys[li] & near) >= len(line_keys[li])
        if all(ln.optional) and not echo and 2 * sum(i in win for i in idx) >= len(idx):
            for i in idx:
                opt[i] = False
    tk = build_tokens(keys, opt, line_of, vocab)
    path = viterbi(logp, tk, fps, win)

    # token k occupies the frames where path == 2k+1
    n = len(keys)
    w_start: list[float | None] = [None] * n
    w_end: list[float | None] = [None] * n
    w_prob: list[list[float]] = [[] for _ in range(n)]
    tok_state = path % 2 == 1
    frames = np.nonzero(tok_state)[0]
    toks = path[frames] // 2
    gap = int(MAX_INTRA * fps)
    for f, k in zip(frames, toks):
        wi = tk.word[k]
        if wi < 0:
            continue
        t0 = f / fps
        # a word's characters come close together; one stray early character (a held note or breath before
        # the word, read as its first letter) is dropped by restarting the word after a long internal gap.
        # Unless Whisper heard the word right there: then the gap is the word itself held ("goooes": CTC marks
        # the onset and the end of a long vowel, with blanks in between)
        held = (w_start[wi] is not None and wi in win and wi > 0 and line_of[wi - 1] == line_of[wi]
                and win[wi][0] + EARLY - HELD_SLACK <= w_start[wi] <= win[wi][1])   # (a line's first word: stray)
        if w_start[wi] is None or (f - w_end[wi] * fps > gap and not held):
            w_start[wi] = t0
        w_end[wi] = t0 + 1 / fps
        w_prob[wi].append(float(np.exp(logp[f, tk.ids[k]])))
    info.skipped = sum(1 for i in range(n) if w_start[i] is None and opt[i])
    starts = _fill(w_start, 0.0, duration)
    ends = [e if e is not None else s for s, e in zip(starts, w_end)]

    out = []
    unheard: set[int] = set()
    first = 0
    for li, ln in enumerate(lines):
        idx = list(range(first, first + len(ln.words)))
        first += len(ln.words)
        sung = [i for i in idx if w_start[i] is not None]
        probs = [p for i in sung for p in w_prob[i]]
        info.line_score.append(round(float(np.mean(probs)) if probs else 0.0, 3))
        # skipped words at the edges of a sung line (a backing "(I don't deserve you)" the path jumped) stay
        # with the line instead of being spread over the gap to the neighbouring line; leading ones start
        # where Whisper heard them, if it did
        if sung:
            lead = idx[:idx.index(sung[0])]
            t1 = starts[sung[0]]
            t0 = win[lead[0]][0] + EARLY if lead and lead[0] in win else t1
            t0 = min(max(t0, t1 - LINE_GAP * 3), t1)
            for j, i in enumerate(lead):
                starts[i] = ends[i] = t0 + (t1 - t0) * j / len(lead)
            for i in idx[idx.index(sung[-1]) + 1:]:
                starts[i] = ends[i] = ends[sung[-1]]
        # the same stray problem one level up: a first word far ahead of the rest of its line (a held note
        # before it) moves up to just before the second word
        if len(sung) >= 2 and sung[0] == idx[0]:
            a, b = sung[0], sung[1]
            if starts[b] - ends[a] > LINE_GAP:
                dur = min(ends[a] - starts[a], 0.4)
                starts[a], ends[a] = starts[b] - dur - 0.05, starts[b] - 0.05
        words = []
        for j, i in enumerate(idx):
            s = starts[i]
            e = max(ends[i], starts[idx[j + 1]] if j + 1 < len(idx) else ends[i])
            words.append(Word(round(s, 3), round(max(s + 0.05, e), 3), ln.words[j]))
        s = words[0].start
        out.append(LyricLine(s, words[-1].end, ln.text, words, motifs=motifs_of(ln.text)))
        if info.line_score[-1] < LOW and not any(i in win for i in idx) and all(map(vocable, ln.words)):
            unheard.add(id(out[-1]))
    out.sort(key=lambda ln: ln.start)
    counts = Counter(_key(ln.text) for ln in out)
    for ln in out:
        ln.hook = counts[_key(ln.text)] >= 2
    if onset_env is not None:
        info.snapped = snap_starts(out, onset_env, onset_rate)
        place_unheard(out, unheard, onset_env, onset_rate)
    for i, ln in enumerate(out):                            # stay up a little after the last word
        nxt = out[i + 1].start if i + 1 < len(out) else duration
        ln.end = round(max(min(ln.words[-1].end + HOLD, nxt), min(nxt, ln.start + 0.3)), 3)
    return out, info


def _ts(t: float) -> str:
    m, s = divmod(max(0.0, t), 60)
    return f"{int(m):02d}:{s:05.2f}"


def to_lrc(lines, title: str | None = None) -> str:
    """Enhanced LRC (line + word timestamps) for inspection, hand editing and reuse as a normal lyric file."""
    out = [f"[ti:{title}]"] if title else []
    out.append("[re:ascii-pv auto align]")
    for ln in lines:
        g = glue(tuple(w.text for w in ln.words), ln.text)      # CJK: no space between glued characters
        body = "".join(("" if g[i] or i == 0 else " ") + f"<{_ts(w.start)}>{w.text}" for i, w in enumerate(ln.words))
        out.append(f"[{_ts(ln.start)}]{body}")
    return "\n".join(out) + "\n"
