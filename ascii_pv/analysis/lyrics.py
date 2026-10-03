"""Lyrics: LRC / SRT parsing, word timing, hooks and visual motifs (English, plus Chinese / Japanese text).

Word timing:
  - enhanced LRC (<mm:ss.xx> before each word) is used as is;
  - otherwise each line's time is split across its words by English syllable count (a CJK character is one
    syllable and its own word, see core/cjk.py), with a short pause after commas (and at the spaces of a CJK
    line), and each word start is snapped to a nearby audio onset when one is close.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from collections import Counter
from pathlib import Path

from ascii_pv.core.analysis import LyricLine, Word
from ascii_pv.core.cjk import cjk_count, glue, has_cjk, tokens, word_count

MAX_SNAP = 0.08          # s: how far a word start may move to meet an onset
TAIL = 0.92              # a line's words fill this share of its time (the rest is breath before the next)

# ---------------------------------------------------------------- reading

PUNCT_MAP = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'", "“": '"', "”": '"', "„": '"',
                           "–": "-", "—": "-", "‐": "-", "‑": "-", "…": "...", " ": " ", "　": " "})


def clean(s: str) -> str:
    """Typographic punctuation -> ASCII (the bitmap fonts only carry ASCII), collapse spaces."""
    s = unicodedata.normalize("NFC", s).translate(PUNCT_MAP)
    return re.sub(r"\s+", " ", s).strip()


def read_text(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        print(f"warning: {path} is not UTF-8; reading it as cp1252", file=sys.stderr)
        return raw.decode("cp1252", errors="replace")


# ---------------------------------------------------------------- LRC

TS = re.compile(r"\[(\d+):(\d{1,2})(?:[.:](\d{1,3}))?\]")
WORD_TS = re.compile(r"<(\d+):(\d{1,2})(?:[.:](\d{1,3}))?>")
META = re.compile(r"^\[([a-zA-Z]+):(.*)\]\s*$")


def _secs(m: re.Match) -> float:
    frac = m.group(3) or "0"
    return int(m.group(1)) * 60 + int(m.group(2)) + int(frac) / (10 ** len(frac))


def parse_lrc(text: str) -> tuple[list[dict], dict]:
    """-> (entries [{t, text, words: [(t, word)] | None}], metadata). Empty-text entries mark line ends."""
    meta: dict[str, str] = {}
    entries = []
    for raw in text.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        mm = META.match(raw)
        if mm and not TS.match(raw):
            meta[mm.group(1).lower()] = mm.group(2).strip()
            continue
        stamps = []
        pos = 0
        while (m := TS.match(raw, pos)):
            stamps.append(_secs(m))
            pos = m.end()
        if not stamps:
            continue
        body = raw[pos:]
        words = None
        if WORD_TS.search(body):
            words = []
            parts = WORD_TS.split(body)  # [pre, mm, ss, frac, text, mm, ss, frac, text, ...]
            for i in range(1, len(parts), 4):
                wt = int(parts[i]) * 60 + int(parts[i + 1]) + int(parts[i + 2] or 0) / (10 ** len(parts[i + 2] or "0"))
                for w in clean(parts[i + 3]).split(" "):
                    if w:
                        words.append((wt, w))
            body = WORD_TS.sub("", body)
        for t in stamps:  # "[00:10.00][01:20.00]chorus" repeats one text at several times
            entries.append({"t": t, "text": clean(body), "words": words})
    entries.sort(key=lambda e: e["t"])
    off = meta.get("offset")
    if off:
        try:
            shift = -int(off) / 1000.0     # LRC convention: positive offset = lyrics come sooner
            for e in entries:
                e["t"] += shift
                if e["words"]:
                    e["words"] = [(t + shift, w) for t, w in e["words"]]
        except ValueError:
            pass
    return entries, meta


# ---------------------------------------------------------------- SRT

SRT_TIME = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->\s*(\d+):(\d+):(\d+)[,.](\d+)")


def parse_srt(text: str) -> list[dict]:
    """-> [{t, end, text}]"""
    out = []
    for block in re.split(r"\n\s*\n", text.replace("\r\n", "\n")):
        lines = [ln for ln in block.strip().split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            m = SRT_TIME.search(ln)
            if m:
                g = [int(x) for x in m.groups()]
                t0 = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 10 ** len(m.group(4))
                t1 = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 10 ** len(m.group(8))
                body = clean(re.sub(r"<[^>]+>|\{[^}]+\}", "", " ".join(lines[i + 1:])))
                if body:
                    out.append({"t": t0, "end": t1, "text": body})
                break
    return sorted(out, key=lambda e: e["t"])


# ---------------------------------------------------------------- word timing

def syllables(word: str) -> int:
    if (n := cjk_count(word)):
        return n
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w:
        return 1
    if len(w) <= 3:
        return 1
    w = re.sub(r"(?:[^laeiouy]es|ed|[^laeiouy]e)$", "", w)
    w = re.sub(r"^y", "", w)
    return max(1, len(re.findall(r"[aeiouy]{1,2}", w)))


def _nearest(sorted_ts: list[float], t: float) -> float | None:
    import bisect

    i = bisect.bisect_left(sorted_ts, t)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_ts) and (best is None or abs(sorted_ts[j] - t) < abs(best - t)):
            best = sorted_ts[j]
    return best


PAUSE_AFTER = ",;:，、；：。！？!?"


def estimate_words(text: str, start: float, end: float, onsets: list[float] | None = None) -> list[Word]:
    toks = tokens(text)
    if not toks or not text:
        return []
    weights = [syllables(tk) + (0.6 if tk[-1:] in PAUSE_AFTER else 0.0) for tk in toks]
    if has_cjk(text):                    # "哭過 痛過 愛過": the spaces of a CJK line are short breaths
        g = glue(toks, text)
        weights = [w + (0.4 if i + 1 < len(toks) and not g[i + 1] else 0.0) for i, w in enumerate(weights)]
    span = (end - start) * TAIL
    total = sum(weights)
    starts, acc = [], 0.0
    for wgt in weights:
        starts.append(start + span * acc / total)
        acc += wgt
    if onsets:
        for i in range(1, len(starts)):
            o = _nearest(onsets, starts[i])
            if o is not None and abs(o - starts[i]) <= MAX_SNAP and o > starts[i - 1] + 0.05:
                starts[i] = o
    ends = starts[1:] + [start + span]
    return [Word(round(s, 3), round(max(s + 0.05, e), 3), tk) for s, e, tk in zip(starts, ends, toks)]


def max_line_len(text: str) -> float:
    """Upper bound on how long a line stays up when the next timestamp is far away (instrumental gap)."""
    return max(4.0, 0.6 * word_count(text) + 1.5)


# ---------------------------------------------------------------- motifs & hooks

MOTIFS: dict[str, set[str]] = {
    "love": {"love", "lover", "heart", "kiss", "baby", "babe", "darling", "honey", "mine", "yours"},
    "fire": {"fire", "burn", "flame", "blaze", "hot", "heat", "spark", "ignite"},
    "water": {"rain", "tear", "cry", "ocean", "sea", "wave", "water", "drip", "drown", "flood"},
    "night": {"night", "dark", "shadow", "moon", "midnight", "shade", "black"},
    "star": {"star", "sky", "galaxy", "space", "universe", "planet", "cosmos", "heaven"},
    "light": {"light", "shine", "bright", "glow", "sun", "blink", "flash", "neon"},
    "time": {"time", "clock", "hour", "minute", "second", "forever", "lifetime", "tomorrow", "today", "o'clock"},
    "speed": {"run", "race", "drive", "fast", "speed", "vroom", "rush", "gas", "chase", "overtake", "car", "coupe"},
    "city": {"city", "street", "traffic", "town", "road", "highway", "downtown"},
    "pop": {"pop", "bubble", "candy", "sugar", "sweet", "lollipop", "bubblegum", "bounce", "jump"},
    "blood": {"blood", "vein", "bleed", "bleeding", "pain", "wound", "scar", "hurt"},
    "dream": {"dream", "illusion", "illusory", "sleep", "fantasy", "wish", "real"},
    "fly": {"fly", "wing", "float", "weightless", "free", "high", "sky"},
    "break": {"break", "broke", "broken", "shatter", "crash", "fall", "falling"},
    "pulse": {"pulse", "beat", "heartbeat", "rhythm", "drum", "echo", "code"},
    "dance": {"dance", "move", "moving", "groove", "step", "stepping", "shake"},
    "fight": {"fight", "weapon", "war", "battle", "ground", "sharp", "hunt", "hunting", "blade"},
}
_WORD2MOTIF = {w: m for m, ws in MOTIFS.items() for w in ws}

# Chinese (traditional and simplified) / Japanese: words found anywhere in the line (no spaces to split on)
MOTIFS_CJK: dict[str, set[str]] = {
    "love": {"愛", "爱", "戀", "恋", "吻", "心", "寶貝", "宝贝", "親愛", "亲爱"},
    "fire": {"火", "燃", "焰", "燒", "烧", "熱", "热", "炎"},
    "water": {"雨", "淚", "泪", "哭", "海", "浪", "水", "洋", "溺"},
    "night": {"夜", "黑", "暗", "影", "月"},
    "star": {"星", "宇宙", "銀河", "银河", "天空"},
    "light": {"光", "亮", "陽", "阳", "閃", "闪", "燈", "灯"},
    "time": {"時間", "时间", "鐘", "钟", "永遠", "永远", "明天", "今天", "秒", "時光", "时光"},
    "speed": {"奔", "速", "飆", "飙", "衝", "冲", "車", "车", "追"},
    "city": {"城", "街", "馬路", "马路", "公路", "都市"},
    "pop": {"糖", "甜", "泡"},
    "blood": {"血", "痛", "傷", "伤", "疤", "疼"},
    "dream": {"夢", "梦", "幻", "睡", "願", "愿"},
    "fly": {"飛", "飞", "翅", "翼", "飄", "飘", "自由"},
    "break": {"碎", "破", "崩", "墜", "坠"},
    "pulse": {"心跳", "節奏", "节奏", "鼓", "脈", "脉", "編碼", "编码"},
    "dance": {"舞", "搖", "摇"},
    "fight": {"戰", "战", "鬥", "斗", "劍", "剑", "刀", "槍", "枪"},
}
_CJK2MOTIF = sorted(((w, m) for m, ws in MOTIFS_CJK.items() for w in ws), key=lambda x: -len(x[0]))


def lemma(word: str) -> str:
    w = re.sub(r"[^a-z']", "", word.lower()).strip("'")
    for cand in (w, w.removesuffix("'s")):
        if cand in _WORD2MOTIF:
            return cand
    for suf, rep in (("ies", "y"), ("ing", ""), ("ing", "e"), ("ed", ""), ("ed", "e"), ("es", ""), ("s", "")):
        if w.endswith(suf) and len(w) - len(suf) >= 3:
            cand = w[: -len(suf)] + rep
            if cand in _WORD2MOTIF:
                return cand
            if len(cand) > 3 and cand[-1] == cand[-2] and cand[:-1] in _WORD2MOTIF:  # running -> run
                return cand[:-1]
    return w


def motifs_of(text: str) -> list[str]:
    found = []
    if has_cjk(text):
        for w, m in _CJK2MOTIF:
            if w in text and m not in found:
                found.append(m)
    for tok in text.split(" "):
        m = _WORD2MOTIF.get(lemma(tok))
        if m and m not in found:
            found.append(m)
    return found


def _key(text: str) -> str:
    """Comparison key for repeated lines: letters, digits (any script) and spaces only."""
    return re.sub(r"[^\w ]|_", "", text.lower()).strip()


# ---------------------------------------------------------------- entry point

def load_lyrics(path: str | Path, onsets: list[float] | None = None, duration: float | None = None,
                offset: float = 0.0, scale: float = 1.0) -> list[LyricLine]:
    """Parse LRC/SRT into timed lines with words, hooks and motifs.

    offset/scale correct a lyric file made for another version of the song: t' = t * scale + offset.
    """
    text = read_text(path)
    suffix = Path(path).suffix.lower()
    rows: list[dict] = []
    if suffix == ".srt" or (suffix != ".lrc" and SRT_TIME.search(text)):
        for e in parse_srt(text):
            rows.append({"start": e["t"], "end": e["end"], "text": e["text"], "words": None})
    else:
        entries, _ = parse_lrc(text)
        for i, e in enumerate(entries):
            if not e["text"]:
                continue
            # still in the lyric file's own time base here; clipping to the song happens after fix()
            cap = e["t"] + max_line_len(e["text"])
            end = min(entries[i + 1]["t"], cap) if i + 1 < len(entries) else cap
            rows.append({"start": e["t"], "end": end, "text": e["text"], "words": e["words"]})

    def fix(t: float) -> float:
        return t * scale + offset

    lines = []
    for r in rows:
        s, e = fix(r["start"]), fix(r["end"])
        if duration is not None:
            if s >= duration:
                continue
            e = min(e, duration)
        if e - s < 0.2:
            continue
        if r["words"]:
            ws = [(fix(t), w) for t, w in r["words"]]
            words = [Word(round(t, 3), round(max(t + 0.05, (ws[j + 1][0] if j + 1 < len(ws) else e)), 3), w)
                     for j, (t, w) in enumerate(ws)]
        else:
            words = estimate_words(r["text"], s, e, onsets)
        lines.append(LyricLine(round(s, 3), round(e, 3), r["text"], words, motifs=motifs_of(r["text"])))

    counts = Counter(_key(ln.text) for ln in lines)
    for ln in lines:
        ln.hook = counts[_key(ln.text)] >= 2
    return lines
