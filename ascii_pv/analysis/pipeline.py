"""analyze(song, lyrics) -> work/<song-id>/analysis.json, cached by input hash and options."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from ascii_pv.analysis.audio import SR, analyze_audio
from ascii_pv.analysis.ffmpeg import decode
from ascii_pv.analysis.lyrics import load_lyrics
from ascii_pv.analysis.structure import boundaries, label_sections
from ascii_pv.analysis.style import analyze_style
from ascii_pv.core.analysis import Analysis, LyricLine
from ascii_pv.workdir import WorkDir, file_id

VERSION = 18   # bump when the analysis output changes meaning, to invalidate caches


def _options_key(song: Path, lyrics: Path | None, opts: dict) -> str:
    parts = [str(VERSION), file_id(song, 64), file_id(lyrics, 64) if lyrics else "-", json.dumps(opts, sort_keys=True)]
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


def _shift_lines(lines: list[LyricLine], starts: list[float], duration: float) -> None:
    """Move each line (and its words) to a refined start; ends stop at the next line. A line that got shorter
    that way has its words squeezed into it (they would otherwise be sung after the next line has begun)."""
    from ascii_pv.analysis.lyrics import TAIL

    for i, (ln, s) in enumerate(zip(lines, starts)):
        d = s - ln.start
        ln.start = s
        ln.end = min(ln.end + d, starts[i + 1] if i + 1 < len(starts) else duration)
        ws = [w.start + d for w in ln.words]
        room = (ln.end - s) * TAIL
        k = room / (ws[-1] - s) if ws and ws[-1] - s > room else 1.0
        ws = [s + (t - s) * k for t in ws]
        for j, w in enumerate(ln.words):
            w.start = round(ws[j], 3)
            nxt = ws[j + 1] if j + 1 < len(ws) else ln.end
            w.end = round(max(w.start + 0.05, min(w.end + d, nxt if k < 1 else ln.end)), 3)


def _align_plain(song: Path, lyrics: Path, work: WorkDir, duration: float, have: dict, dev: str, log):
    """Plain-text lyrics -> timed lines (analysis/align.py); also writes work/lyrics.auto.lrc."""
    from ascii_pv.analysis.align import align_lines, text_lines, to_lrc
    from ascii_pv.analysis.ml.asr import ctc_logp, ctc_vocab, whisper_words
    from ascii_pv.analysis.ml.stems import vocal_wave
    from ascii_pv.analysis.sync import onset_envelope

    tl = text_lines(lyrics)
    if not tl:
        log("  no lyric lines found")
        return [], None
    from ascii_pv.core.cjk import has_cjk

    language = "zh" if any(has_cjk(ln.text) for ln in tl) else "en"
    if language == "zh":
        try:
            import pypinyin  # noqa: F401
        except ImportError:
            log("  Chinese lyric timing needs pypinyin: pip install pypinyin; no lyrics")
            return [], None
    wave = vocal_wave(song, work.cache, dev, log)
    logp, fps = ctc_logp(wave, work.cache, dev, log)
    heard = None
    if have.get("clap"):                                    # transformers is there: Whisper anchors
        try:
            heard = whisper_words(wave, work.cache, dev, log, language=language)
        except Exception as e:                              # download / runtime problems: CTC alone still works
            log(f"  whisper failed ({e}); aligning without ASR anchors")
    env = onset_envelope(np.load(work.cache / "vocals_rms.npy"), 48)
    lines, info = align_lines(tl, logp, fps, ctc_vocab(), heard, duration, onset_env=env)
    log("  " + info.describe())
    out = work.root / "lyrics.auto.lrc"
    out.write_text(to_lrc(lines, song.stem), encoding="utf-8")
    log(f"  timed lyrics -> {out} (edit it and pass it as --lyrics to override)")
    return lines, dict(method="ctc+whisper" if heard else "ctc", language=language, words=info.words, anchored=info.anchored,
                       skipped=info.skipped, snapped=info.snapped, line_score=info.line_score)


def analyze(song: str | Path, lyrics: str | Path | None = None, *, bpm: float | None = None,
            lyrics_offset: float = 0.0, lyrics_scale: float = 1.0, work_root: Path = Path("work"),
            force: bool = False, ml: bool = True, device: str | None = None, sync: bool = True,
            retime: bool = False, log=print) -> tuple[Analysis, WorkDir]:
    """ml: use the ML layer where installed. sync: auto-align the lyrics to the vocals (only when no manual
    --lyrics-offset / --lyrics-scale was given). retime: ignore the lyric file's timestamps and align its text
    like plain-text lyrics (for an inaccurate LRC)."""
    from ascii_pv.analysis import ml as mllayer

    song = Path(song)
    lyrics = Path(lyrics) if lyrics else None
    work = WorkDir.for_song(song, work_root).ensure()
    have = mllayer.available() if ml else {}
    opts = dict(bpm=bpm, lyrics_offset=lyrics_offset, lyrics_scale=lyrics_scale, sync=sync, retime=retime,
                ml=sorted(k for k, v in have.items() if v))
    key = _options_key(song, lyrics, opts)
    if work.analysis.exists() and not force:
        cached = Analysis.load(work.analysis)
        if cached.source.get("key") == key:
            log(f"cached: {work.analysis}")
            return cached, work

    t0 = time.time()
    dev = mllayer.device(device) if any(have.values()) else "cpu"
    from ascii_pv.analysis.image import extract_cover

    if extract_cover(song, work.cache):
        log("cover art found -> cache/image/cover_src.png")
    log(f"decoding {song.name} ...")
    y = decode(song, SR)
    log("tempo, beats, features ...")
    au = analyze_audio(y, bpm_override=bpm)
    used: dict[str, str] = {}

    if have.get("beat_this") and not bpm:
        try:
            from ascii_pv.analysis.ml.beats import beats_downbeats, bpm_of

            log(f"beat_this on {dev} ...")
            b, d = beats_downbeats(y, SR, dev)
            if len(b) > 16 and d:
                au.beats, au.downbeats, au.bpm = b, d, round(bpm_of(b), 2)
                used["beats"] = "beat_this"
        except Exception as e:   # model download / runtime problems: keep the librosa beats
            log(f"  beat_this failed ({e}); using librosa beats")

    vocal, vocal_onsets, stems = None, None, None
    if have.get("demucs"):
        try:
            from ascii_pv.analysis.ml.stems import stem_features, stem_raw, vocal_features, vocal_rms

            rms = vocal_rms(song, work.cache, dev, log)
            vocal, vocal_onsets = vocal_features(rms)
            stems = stem_features(stem_raw(song, work.cache, dev, log), len(au.features["loud"]))
            used["vocals"] = "demucs"
        except Exception as e:
            log(f"  demucs failed ({e}); no vocal track")

    lines: list[LyricLine] = []
    sync_info = align_info = None
    from ascii_pv.analysis.align import is_plain_text, text_lines
    from ascii_pv.core.cjk import has_kana

    plain = bool(lyrics) and is_plain_text(lyrics)
    if lyrics and (retime or plain) and any(has_kana(ln.text) for ln in text_lines(lyrics)):
        # the aligner reads romanised text: Chinese goes through pinyin, Japanese would need its own romanisation
        if plain:
            log(f"lyrics {lyrics.name}: timing Japanese text from the vocals is not supported yet; "
                "use a timed LRC (no lyrics this time)")
            lyrics = None
        else:
            log("  --retime ignored: timing Japanese text from the vocals is not supported yet; "
                "using the file's timestamps")
            retime = False
    if lyrics and (retime or plain):
        log(f"lyrics {lyrics.name} ({'retime' if retime else 'plain text'}: aligning the text to the vocals) ...")
        if have.get("demucs"):
            lines, align_info = _align_plain(song, lyrics, work, au.duration, have, dev, log)
        else:
            log("  plain-text lyrics need the ML layer (demucs + torchaudio): pip install -e .[ml]; no lyrics")
    elif lyrics:
        log(f"lyrics {lyrics.name} ...")
        onsets = vocal_onsets or au.onsets
        manual = lyrics_offset != 0.0 or lyrics_scale != 1.0
        off, sc = lyrics_offset, lyrics_scale
        if sync and vocal is not None and not manual:
            from ascii_pv.analysis.sync import fine, fit, onset_envelope, refine, vocal_activity

            raw_lines = load_lyrics(lyrics)                         # the file's own timing
            vr = np.load(work.cache / "vocals_rms.npy")
            env = onset_envelope(vr, vocal["rate"])
            res = fit([ln.start for ln in raw_lines], [ln.end for ln in raw_lines], vocal_activity(vr, vocal["rate"]),
                      vocal["rate"])
            log("  " + res.describe())
            sync_info = dict(scale=res.scale, offset=res.offset, z=round(res.z, 2), applied=res.applied)
            if res.applied:
                off, sc = res.offset, res.scale
                f = fine([ln.start for ln in raw_lines], env, vocal["rate"], sc, off)
                log("  " + f.describe("onset fine-tune"))
                sync_info["fine"] = dict(scale=f.scale, offset=f.offset, z=round(f.z, 2), applied=f.applied)
                if f.applied:
                    off, sc = f.offset, f.scale
        lines = load_lyrics(lyrics, onsets=onsets, duration=au.duration, offset=off, scale=sc)
        if sync_info and sync_info["applied"]:
            _shift_lines(lines, refine([ln.start for ln in lines], env, vocal["rate"]), au.duration)
            sync_info["refined"] = True

    log("sections ...")
    bounds = boundaries(y, au.beats, au.downbeats, au.bpm, au.duration)
    sections = label_sections(y, bounds, au.features["loud"], au.features["rate"], lines)
    from ascii_pv.analysis.events import detect

    events = detect(au.features, stems, au.downbeats, au.duration, au.bpm)
    log("style ...")
    style = analyze_style(y, au.bpm, au.raw)
    if have.get("clap"):
        try:
            from ascii_pv.analysis.ml.clap import merge_into_style, tags

            log(f"CLAP on {dev} ...")
            style = merge_into_style(style, tags(y, SR, dev))
            used["style"] = "clap"
        except Exception as e:
            log(f"  CLAP failed ({e}); rule-based style only")

    a = Analysis(duration=au.duration, bpm=au.bpm, beats=au.beats, downbeats=au.downbeats, sections=sections,
                 style=style, features=au.features, lyrics=lines, vocal=vocal, stems=stems, events=events,
                 source=dict(key=key, version=VERSION, audio=str(song), lyrics=str(lyrics) if lyrics else None,
                             audio_sha256=file_id(song, 64), options=opts, raw=au.raw, onsets=au.onsets,
                             ml=dict(used, device=dev) if used else None, sync=sync_info, align=align_info))
    a.save(work.analysis)
    log(f"done in {time.time() - t0:.1f}s -> {work.analysis}")
    return a, work


def summary(a: Analysis) -> str:
    def mmss(t: float) -> str:
        return f"{int(t) // 60}:{t % 60:05.2f}"

    s = a.style
    out = [f"duration {mmss(a.duration)}   bpm {a.bpm}   beats {len(a.beats)}   first downbeat {mmss(a.downbeats[0]) if a.downbeats else '-'}",
           f"style    energy {s['energy']}  valence {s['valence']}  density {s['density']}  key {s.get('key', '?')}  "
           f"tags {', '.join(s.get('tags', []))}",
           "sections"]
    ml = a.source.get("ml")
    if ml:
        out.insert(2, "ml       " + ", ".join(f"{k}={v}" for k, v in ml.items()))
    clap = a.style.get("clap")
    if clap:
        out.insert(2, "clap     genre " + ", ".join(f"{g} {p:.2f}" for g, p in clap["genre"][:3])
                   + " | mood " + ", ".join(f"{m} {p:.2f}" for m, p in clap["mood"][:3]))
    sync = a.source.get("sync")
    if sync:
        out.insert(2, f"sync     scale {sync['scale']:.4f} offset {sync['offset']:+.2f}s z={sync['z']} "
                      f"{'applied' if sync['applied'] else 'NOT applied'}"
                      + (f" | fine {fi['scale']:.4f} {fi['offset']:+.2f}s z={fi['z']} "
                         f"{'applied' if fi['applied'] else 'not applied'}" if (fi := sync.get("fine")) else ""))
    al = a.source.get("align")
    if al:
        sc = np.array(al["line_score"] or [0.0])
        out.insert(2, f"align    plain-text lyrics, {al['method']}: {al['anchored']}/{al['words']} words anchored, "
                      f"{al['snapped']} starts on onsets, confidence median {np.median(sc):.2f}")
    if a.events:
        def fmt(e):
            if e["kind"] == "build":
                return f"build {mmss(e['start'])}-{mmss(e['end'])}"
            return f"{e['kind']} {mmss(e['t'])}"
        out.insert(2, "events   " + ", ".join(fmt(e) for e in a.events))
    for sec in a.sections:
        n = sum(1 for ln in a.lyrics if sec.start <= ln.start < sec.end)
        bar = "#" * int(sec.energy * 20)
        out.append(f"  {mmss(sec.start):>8} - {mmss(sec.end):>8}  {sec.group or '-'} {sec.label:12} e={sec.energy:.2f} {bar:20} lines={n}")
    if a.lyrics:
        hooks = sum(ln.hook for ln in a.lyrics)
        motifs: dict[str, int] = {}
        for ln in a.lyrics:
            for m in ln.motifs:
                motifs[m] = motifs.get(m, 0) + 1
        top = ", ".join(f"{m}x{c}" for m, c in sorted(motifs.items(), key=lambda kv: -kv[1])[:8])
        out.append(f"lyrics   {len(a.lyrics)} lines, {hooks} hook lines, motifs: {top or '-'}")
    return "\n".join(out)
