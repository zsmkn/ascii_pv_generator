"""Local web GUI (Gradio): the whole pipeline in four tabs.

  1 Song     upload music + lyrics (+ picture), or reopen an analysed song; analyse (and plan the first time)
  2 Direct   seed / palette / lyric offset -> plan; edit shots and lyric cues in tables and save storyboard.json
  3 Preview  any frame as an image; play in a new terminal window
  4 Render   MP4 with a progress bar, shown in the page when done

The handlers are plain functions of plain values (paths, numbers, DataFrames) so they can be tested without a
browser; build() wires them to widgets. Uploads are copied into input/ so work dirs never point at Gradio's
temporary files. Run with `ascii-pv gui`.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.palette import PALETTES
from ascii_pv.core.storyboard import LyricCue, Shot, Storyboard

WORK_ROOT = Path("work")
INPUT = Path("input")
SHOT_COLS = ["id", "start", "end", "template", "transition_in", "params", "fx", "locked"]
CUE_COLS = ["line", "text", "mode", "font", "fill", "enter", "exit", "blend", "offset", "locked"]
QUALITY = {"draft (fast, small)": (28, "veryfast"), "standard": (23, "medium"), "high": (18, "slow")}
SIZES = {"1280x720": ((1280, 720), (160, 45)), "1920x1080": ((1920, 1080), (240, 67)),
         "854x480 (quick)": ((854, 480), (106, 30))}


# ---------------------------------------------------------------- helpers

def list_works() -> list[str]:
    if not WORK_ROOT.exists():
        return []
    return sorted(p.name for p in WORK_ROOT.iterdir() if (p / "analysis.json").exists())


def _work(name: str) -> Path:
    if not name:
        raise ValueError("choose or analyse a song first")
    p = WORK_ROOT / name
    if not (p / "analysis.json").exists():
        raise ValueError(f"{p} has no analysis.json")
    return p


def keep_upload(path: str | None) -> Path | None:
    """Copy an uploaded file into input/ under its own name (Gradio's temp copies do not last)."""
    if not path:
        return None
    src = Path(path)
    if src.resolve().parent == INPUT.resolve():
        return src
    INPUT.mkdir(exist_ok=True)
    dst = INPUT / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copyfile(src, dst)
    return dst


def _summary(work: Path) -> str:
    from ascii_pv.analysis.pipeline import summary

    text = summary(Analysis.load(work / "analysis.json"))
    sb = work / "storyboard.json"
    if sb.exists():
        b = Storyboard.load(sb)
        text += f"\nstoryboard: {len(b.shots)} shots, palette {b.palette}, lyric offset {b.lyric_offset:+.2f}s"
        if b.assets:
            text += ", with picture"
    return text


# ---------------------------------------------------------------- storyboard <-> tables

def board_tables(board: Storyboard, analysis: Analysis):
    import pandas as pd

    shots = pd.DataFrame([[s.id, round(s.start, 3), round(s.end, 3), s.template, s.transition_in,
                           json.dumps(s.params), json.dumps(s.fx), s.locked] for s in board.shots], columns=SHOT_COLS)
    cues = {c.line: c for c in board.lyrics}
    rows = []
    for i, ln in enumerate(analysis.lyrics):
        c = cues.get(i, LyricCue(line=i))
        rows.append([i, ln.text, c.mode, c.font, c.fill, c.enter, c.exit, c.blend, c.offset, c.locked])
    return shots, pd.DataFrame(rows, columns=CUE_COLS)


def _truthy(v) -> bool:
    return str(v).strip().lower() in ("true", "1", "yes", "y", "x")


def tables_to_board(board: Storyboard, shots_df, cues_df) -> Storyboard:
    shots = []
    for row in shots_df.itertuples(index=False):
        r = dict(zip(SHOT_COLS, row))
        shots.append(Shot(str(r["id"]), float(r["start"]), float(r["end"]), str(r["template"]),
                          json.loads(r["params"] or "{}"), json.loads(r["fx"] or "{}"), str(r["transition_in"]),
                          _truthy(r["locked"])))
    cues = []
    for row in cues_df.itertuples(index=False):
        r = dict(zip(CUE_COLS, row))
        cues.append(LyricCue(line=int(r["line"]), mode=str(r["mode"]), font=str(r["font"]), fill=str(r["fill"]),
                             enter=str(r["enter"]), exit=str(r["exit"]), blend=str(r["blend"]),
                             offset=float(r["offset"] or 0), locked=_truthy(r["locked"])))
    d = board.to_dict()
    d["shots"] = [s.__dict__ for s in shots]
    d["lyrics"] = [c.__dict__ for c in cues]
    return Storyboard.from_dict(d)


# ---------------------------------------------------------------- handlers

def do_analyze(audio: str | None, lyrics: str | None, image: str | None, existing: str | None, use_ml: bool,
               use_sync: bool, retime: bool = False, log=print) -> tuple[str, str]:
    """-> (work name, summary). With no audio, reopens `existing`."""
    from ascii_pv.analysis.pipeline import analyze
    from ascii_pv.director.planner import plan

    if not audio:
        work = _work(existing)
        return work.name, _summary(work)
    song, lrc, img = keep_upload(audio), keep_upload(lyrics), keep_upload(image)
    a, wd = analyze(song, lrc, work_root=WORK_ROOT, ml=use_ml, sync=use_sync, retime=retime, log=log)
    sb = wd.storyboard
    if img or not sb.exists():
        from ascii_pv.cli import _image_assets

        lock = Storyboard.load(sb) if sb.exists() else None
        plan(a, lock=lock, assets=_image_assets(wd.root, str(img) if img else None, lock)).save(sb)
    return wd.root.name, _summary(wd.root)


def do_plan(name: str, seed: int, palette: str, lyric_offset: float, keep_locked: bool):
    from ascii_pv.cli import _image_assets
    from ascii_pv.director.planner import plan

    work = _work(name)
    a = Analysis.load(work / "analysis.json")
    sb = work / "storyboard.json"
    old = Storyboard.load(sb) if sb.exists() else None
    lock = old if keep_locked else None
    assets = old.assets if old and old.assets else _image_assets(work, None)
    board = plan(a, seed=int(seed), palette=None if palette == "auto" else palette, lock=lock, assets=assets)
    board.lyric_offset = float(lyric_offset)
    board.save(sb)
    shots, cues = board_tables(board, a)
    counts: dict[str, int] = {}
    for s in board.shots:
        counts[s.template] = counts.get(s.template, 0) + 1
    info = (f"palette {board.palette} · {len(board.shots)} shots: "
            + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    return info, shots, cues


def do_load(name: str):
    work = _work(name)
    a = Analysis.load(work / "analysis.json")
    sb = work / "storyboard.json"
    if not sb.exists():
        return do_plan(name, 0, "auto", 0.0, False)
    board = Storyboard.load(sb)
    shots, cues = board_tables(board, a)
    return f"loaded {sb} (palette {board.palette}, lyric offset {board.lyric_offset:+.2f}s)", shots, cues


def do_save(name: str, shots_df, cues_df) -> str:
    work = _work(name)
    sb = work / "storyboard.json"
    board = Storyboard.load(sb) if sb.exists() else Storyboard()
    try:
        new = tables_to_board(board, shots_df, cues_df)
        from ascii_pv.render import Renderer

        Renderer(Analysis.load(work / "analysis.json"), new).frame(new.shots[0].start + 0.1, 60, 20)  # validate
    except Exception as e:
        return f"not saved: {e}"
    new.save(sb)
    return f"saved {sb} ({len(new.shots)} shots, {len(new.lyrics)} lyric cues)"


def do_preview(name: str, t: float, size: str):
    from PIL import Image

    from ascii_pv.backends.raster import Rasterizer
    from ascii_pv.render import Renderer

    work = _work(name)
    a = Analysis.load(work / "analysis.json")
    sb = work / "storyboard.json"
    if not sb.exists():
        raise ValueError("no storyboard yet: plan first")
    board = Storyboard.load(sb)
    (W, H), (gw, gh) = SIZES.get(size, SIZES["1280x720"])
    c = Renderer(a, board).frame(float(t), gw, gh)
    return Image.fromarray(Rasterizer(W // gw, H // gh).render(c))


def do_play(name: str) -> str:
    """Open a new terminal window running the player (Windows Terminal if present)."""
    work = _work(name)
    exe = [sys.executable, "-m", "ascii_pv", "play", str(work.resolve())]
    cwd = str(Path.cwd())
    wt = shutil.which("wt")
    if sys.platform == "win32":
        if wt:
            subprocess.Popen([wt, "-d", cwd, *exe])
        else:
            subprocess.Popen(["cmd", "/c", "start", "ascii-pv", *exe], cwd=cwd)
        return "opened a terminal window (SPACE pause, Q quit, H help)"
    return "run in a terminal:  " + " ".join(exe)


def do_render(name: str, size: str, quality: str, start: float, end: float, post: bool, subtitle: bool,
              progress=None) -> tuple[str, str]:
    from ascii_pv.backends.video import VideoJob, render_video

    work = _work(name)
    a = Analysis.load(work / "analysis.json")
    sb = work / "storyboard.json"
    if not sb.exists():
        raise ValueError("no storyboard yet: plan first")
    (W, H), grid = SIZES.get(size, SIZES["1280x720"])
    crf, preset = QUALITY.get(quality, QUALITY["standard"])
    audio = a.source.get("audio")
    end = float(end) if end and float(end) > float(start) else None
    out = Path("out") / f"{work.name}{'' if not end and not start else f'_{start:.0f}-{end or a.duration:.0f}'}.mp4"
    job = VideoJob(analysis=str(work / "analysis.json"), storyboard=str(sb), out=str(out),
                   audio=audio if audio and Path(audio).exists() else None, size=(W, H), grid=grid,
                   start=float(start), end=end, post=bool(post), crf=crf, preset=preset, subtitle=bool(subtitle))
    lines: list[str] = []
    render_video(job, work / "cache", log=lambda *a_, **k: lines.append(" ".join(str(x) for x in a_).strip()),
                 progress=progress)
    return str(out), "\n".join(x for x in lines if x)


# ---------------------------------------------------------------- hover hints

# elem_id -> hint shown when the pointer rests on the widget (or it gets keyboard focus). Tab headers use their
# tab's elem_id (Gradio renders the header button as "<elem_id>-button").
TIPS = {
    "tab-song": "Pick the music (and lyrics / picture), or reopen a song analysed before.",
    "tab-direct": "Choose the look and edit the shot list and lyric styles.",
    "tab-preview": "Look at single frames, or play the whole PV in a terminal window.",
    "tab-render": "Render the PV to an MP4 file (saved in out/).",
    "audio": "The song. It is copied into input/; the analysis is cached in work/, so a second run is instant.",
    "lyrics": "LRC (timed lines), SRT, or plain text (one line per row). Plain text is timed automatically from "
              "the vocals. Optional: without lyrics the PV is visuals only.",
    "image": "A character or cover picture. Plain backgrounds are removed automatically. It opens and closes the "
             "PV and joins the choruses.",
    "existing": "Every song analysed before. Choosing one loads it right away: no upload, no new analysis.",
    "refresh": "Refresh this list (after analysing a song from the command line).",
    "use-ml": "Use the ML models when installed: vocal separation, beat tracking, style tags. Slower (the first "
              "run downloads models) but more accurate. Off = fast analysis from the mix only.",
    "use-sync": "Shift / stretch a timed LRC so it fits the singing. Leave on unless you have fixed the timing "
                "yourself.",
    "retime": "Throw away the lyric file's timestamps and time its text from the vocals word by word. Use it for "
              "an LRC whose timing is wrong (e.g. made by an AI).",
    "analyze-btn": "Analyse the song (tempo, beats, sections, lyrics) and make a first storyboard. Takes a few "
                   "minutes with the ML layer, much less on a second run.",
    "seed": "Another number = another cut of the same song (other templates, transitions, lyric styles).",
    "palette": "The colour scheme of the whole PV. auto = chosen from the song's mood.",
    "offset": "Move all lyrics in time. + = lyrics appear earlier. Use it when every line is a bit late or early.",
    "keep": "When planning again, keep the rows whose 'locked' column is true and re-roll everything else.",
    "plan-btn": "Make a new storyboard with the settings above (overwrites storyboard.json, except locked rows).",
    "load-btn": "Read storyboard.json again (after editing it by hand) and discard unsaved table edits.",
    "save-btn": "Write the edited tables to storyboard.json. Checked first: nothing is saved if a row is invalid.",
    "plan-info": "What the last action did.",
    "shots": "One row per shot (a stretch of the song with one background). Double-click a cell to edit; hover "
             "over a column title for what it means.",
    "cues": "How each lyric line is drawn. Double-click a cell to edit; hover over a column title for the "
            "choices.",
    "preview-time": "The moment to show. Releasing the slider shows that frame.",
    "psize": "Picture size of the preview (same grid as the MP4 at that size).",
    "prev-btn": "Draw the frame at the chosen time, exactly as it will be in the MP4.",
    "play-btn": "Open a terminal window that plays the PV with sound (SPACE pause, Q quit, H help).",
    "rsize": "Video size. 1920x1080 uses a finer character grid and takes about twice as long.",
    "quality": "draft = fast and small, for checking; high = sharpest text, slowest.",
    "render-start": "Render from this second (to check one part quickly).",
    "render-end": "Render up to this second. 0 = to the end of the song.",
    "post": "CRT look: glow, short trails and scanlines over the picture.",
    "subtitle": "Also show each lyric line as plain text on the bottom row (always readable).",
    "render-btn": "Render the MP4 (saved in out/). A 3-minute song takes about 2 minutes at 1280x720.",
}

# table elem_id -> column title -> hint
COL_TIPS = {
    "shots": {
        "id": "Shot name (also used to keep locked rows).",
        "start": "Start time in seconds.",
        "end": "End time in seconds (the next shot should start here).",
        "template": "The background animation. The list of names is above the table.",
        "transition_in": "How this shot replaces the previous one: cut, flash, dissolve, crt_on, glitch, wipe, "
                         "blocks, slide, zoom, reflow.",
        "params": "Template settings as JSON, e.g. {\"density\": 0.5}. {} = defaults.",
        "fx": "Effects as JSON, 0-1 each: glitch, scan, zoom, shake, split, strobe, noise, shockwave. "
              "{} = none.",
        "locked": "true = keep this row when planning again.",
    },
    "cues": {
        "line": "Lyric line number (fixed).",
        "text": "The lyric text (read only here: edit the lyric file to change it).",
        "mode": "takeover = full screen, line = lower band, word_flash = one huge word at a time, stack = words "
                "drop in as sung, embed = big letters inside the picture.",
        "font": "bit5x5, impact, black, serif, script, mono, or a path to a .ttf file.",
        "fill": "What the letters are made of: binary, hex, self (their own letters), ramp, block, outline, "
                "solid.",
        "enter": "How the line appears: assemble, decode, rain, scan, glitch_in.",
        "exit": "How the line leaves: explode, fade, melt, shatter, crt_off.",
        "blend": "How the letters meet the picture: overlay, mask (picture shows through), knockout (letters cut "
                 "out), echo (previous line lingers), obstacle (particles flow around them).",
        "offset": "Move only this line, in seconds (+ = earlier).",
        "locked": "true = keep this row when planning again.",
    },
}

HINT_CSS = """
#apv-tip { position: fixed; z-index: 10000; max-width: 340px; padding: 7px 10px; border-radius: 6px;
  background: #1f2430; color: #f0f2f5; font-size: 13px; line-height: 1.4; pointer-events: none;
  box-shadow: 0 4px 14px rgba(0,0,0,.25); opacity: 0; transition: opacity .12s; }
#apv-tip.on { opacity: 1; }
"""

# One document-level listener (Gradio re-renders widgets, e.g. when a tab opens), so nothing is bound per element.
HINT_JS = """
(() => {
  const TIPS = __TIPS__, COLS = __COLS__;
  let box = null, timer = null, shown = null;
  function find(el) {
    for (let n = el; n && n.nodeType === 1; n = n.parentElement) {
      if (n.tagName === 'TH') {
        const t = n.closest('[id]');
        for (let p = t; p; p = p.parentElement ? p.parentElement.closest('[id]') : null) {
          if (COLS[p.id]) {
            const name = n.textContent.trim();
            const k = Object.keys(COLS[p.id]).find(c => name === c || name.startsWith(c + ' '));
            if (k) return [n, COLS[p.id][k]];
            break;
          }
        }
      }
      if (n.id) {
        const id = n.id.endsWith('-button') && !TIPS[n.id] ? n.id.slice(0, -7) : n.id;
        if (TIPS[id]) return [n, TIPS[id]];
      }
    }
    return null;
  }
  let pending = null;
  function hide() { clearTimeout(timer); shown = pending = null; if (box) box.classList.remove('on'); }
  function show(n, text) {
    if (!box) { box = document.createElement('div'); box.id = 'apv-tip'; box.setAttribute('role', 'tooltip');
                document.body.appendChild(box); }
    box.textContent = text;
    const r = n.getBoundingClientRect(), w = box.offsetWidth, h = box.offsetHeight;
    let x = Math.min(Math.max(8, r.left), window.innerWidth - w - 8);
    let y = r.bottom + 6;
    if (y + h > window.innerHeight - 8) y = Math.max(8, r.top - h - 6);
    box.style.left = x + 'px'; box.style.top = y + 'px';
    box.classList.add('on'); shown = n;
  }
  let mx = -1, my = -1, settle = null;
  function enter(e, delay) {
    const hit = find(e.target);
    if (!hit) { if (shown || pending) hide(); return; }
    if (hit[0] === shown || hit[0] === pending) return;
    clearTimeout(timer);
    pending = hit[0];
    timer = setTimeout(() => { pending = null; show(hit[0], hit[1]); }, delay);
  }
  document.addEventListener('mousemove', e => { mx = e.clientX; my = e.clientY; enter(e, 350); });
  document.addEventListener('focusin', e => enter(e, 0));
  document.addEventListener('mousedown', hide);
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hide(); });
  // a scroll moves things under a still pointer: hide, then look again once the scrolling stops
  window.addEventListener('scroll', () => {
    hide();
    clearTimeout(settle);
    settle = setTimeout(() => {
      const el = mx >= 0 ? document.elementFromPoint(mx, my) : null;
      if (el) enter({target: el}, 350);
    }, 150);
  }, true);
})();
"""


def hint_head() -> str:
    """<head> additions: the hint box style and script, with the hint tables inlined."""
    js = HINT_JS.replace("__TIPS__", json.dumps(TIPS)).replace("__COLS__", json.dumps(COL_TIPS))
    js = js.replace("</", "<\\/")                    # a "</" inside the script would end it
    return f"<style>{HINT_CSS}</style>\n<script>{js}</script>"


# ---------------------------------------------------------------- UI

def build():
    import gradio as gr

    from ascii_pv.templates import REGISTRY

    with gr.Blocks(title="ascii-pv") as ui:
        gr.Markdown("# ascii-pv\nMusic + lyrics → ASCII PV. Work top to bottom: **Song → Direct → Preview → "
                    "Render**. Rest the pointer on any control for a hint.")
        work = gr.State("")

        with gr.Tab("1 · Song", elem_id="tab-song"):
            with gr.Row():
                audio = gr.File(label="Music (mp3 / wav / flac ...)", type="filepath", elem_id="audio")
                lyrics = gr.File(label="Lyrics (LRC / SRT / plain text, optional)", type="filepath",
                                 elem_id="lyrics")
                image = gr.File(label="Character / cover picture (optional)", type="filepath",
                                file_types=["image"], elem_id="image")
            with gr.Row():
                existing = gr.Dropdown(list_works(), label="…or reopen an analysed song", allow_custom_value=False,
                                       elem_id="existing")
                refresh = gr.Button("↻", scale=0, elem_id="refresh")
            with gr.Row():
                use_ml = gr.Checkbox(True, label="ML layer (demucs / beat_this / CLAP, if installed)", elem_id="use-ml")
                use_sync = gr.Checkbox(True, label="auto-align lyrics to the vocals", elem_id="use-sync")
                retime = gr.Checkbox(False, label="ignore the lyric file's timestamps (inaccurate LRC): align its text",
                                 elem_id="retime")
            analyze_btn = gr.Button("Analyse", variant="primary", elem_id="analyze-btn")
            summary = gr.Textbox(label="Analysis", lines=18, max_lines=40)

        with gr.Tab("2 · Direct", elem_id="tab-direct"):
            with gr.Row():
                seed = gr.Number(0, label="seed (another seed = another cut)", precision=0, elem_id="seed")
                palette = gr.Dropdown(["auto", *sorted(PALETTES)], value="auto", label="palette",
                                      elem_id="palette")
                offset = gr.Slider(-5, 5, 0, step=0.05, label="lyric offset (s, + = earlier)", elem_id="offset")
                keep = gr.Checkbox(True, label="keep rows marked locked", elem_id="keep")
            with gr.Row():
                plan_btn = gr.Button("Plan", variant="primary", elem_id="plan-btn")
                load_btn = gr.Button("Reload storyboard.json", elem_id="load-btn")
                save_btn = gr.Button("Save edits", elem_id="save-btn")
            plan_info = gr.Textbox(label="Storyboard", lines=2, elem_id="plan-info")
            gr.Markdown("Templates: " + ", ".join(sorted(REGISTRY)) +
                        ". Set **locked** to true on rows you edited, then Plan again to re-roll the rest.")
            shots = gr.Dataframe(headers=SHOT_COLS, label="Shots", interactive=True, wrap=True, elem_id="shots")
            cues = gr.Dataframe(headers=CUE_COLS, label="Lyric cues", interactive=True, wrap=True,
                                elem_id="cues")

        with gr.Tab("3 · Preview", elem_id="tab-preview"):
            with gr.Row():
                t = gr.Slider(0, 600, 30, step=0.05, label="time (s)", elem_id="preview-time")
                psize = gr.Dropdown(list(SIZES), value="1280x720", label="size", elem_id="psize")
            with gr.Row():
                prev_btn = gr.Button("Show frame", variant="primary", elem_id="prev-btn")
                play_btn = gr.Button("▶ Play in terminal", elem_id="play-btn")
            frame = gr.Image(label="Frame", type="pil")
            play_info = gr.Textbox(label="", lines=1)

        with gr.Tab("4 · Render MP4", elem_id="tab-render"):
            with gr.Row():
                rsize = gr.Dropdown(list(SIZES), value="1280x720", label="size", elem_id="rsize")
                quality = gr.Dropdown(list(QUALITY), value="standard", label="quality",
                                      elem_id="quality")
                start = gr.Number(0, label="start (s)", elem_id="render-start")
                end = gr.Number(0, label="end (s, 0 = song end)", elem_id="render-end")
            with gr.Row():
                post = gr.Checkbox(True, label="CRT look (bloom, trails, scanlines)", elem_id="post")
                subtitle = gr.Checkbox(False, label="plain subtitle line", elem_id="subtitle")
            render_btn = gr.Button("Render", variant="primary", elem_id="render-btn")
            video = gr.Video(label="Result")
            render_log = gr.Textbox(label="Log", lines=4)

        # -- wiring
        def _opened(name: str, text: str):
            dur = Analysis.load(WORK_ROOT / name / "analysis.json").duration
            return (name, text, gr.update(choices=list_works(), value=name),
                    gr.update(maximum=round(dur, 2), value=min(30.0, dur / 3)))

        def _analyze(a_, l_, i_, ex, ml, sy, rt, progress=gr.Progress()):
            progress(0.05, desc="analysing (the first ML run downloads models)")
            return _opened(*do_analyze(a_, l_, i_, ex, ml, sy, rt, log=lambda *x, **k: None))

        def _reopen(ex):
            return _opened(*do_analyze(None, None, None, ex, False, False))

        analyze_btn.click(_analyze, [audio, lyrics, image, existing, use_ml, use_sync, retime],
                          [work, summary, existing, t])
        existing.input(_reopen, [existing], [work, summary, existing, t])
        refresh.click(lambda: gr.update(choices=list_works()), None, [existing])
        plan_btn.click(do_plan, [work, seed, palette, offset, keep], [plan_info, shots, cues])
        load_btn.click(do_load, [work], [plan_info, shots, cues])
        save_btn.click(do_save, [work, shots, cues], [plan_info])
        prev_btn.click(do_preview, [work, t, psize], [frame])
        t.release(do_preview, [work, t, psize], [frame])
        play_btn.click(do_play, [work], [play_info])

        def _render(w_, s_, q_, st, en, po, su, progress=gr.Progress()):
            progress(0, desc="starting workers")
            return do_render(w_, s_, q_, st, en, po, su,
                             progress=lambda d, n: progress(d / max(1, n), desc=f"{d}/{n} frames"))

        render_btn.click(_render, [work, rsize, quality, start, end, post, subtitle], [video, render_log])
    return ui


def launch(port: int = 7860, open_browser: bool = True) -> None:
    build().launch(server_name="127.0.0.1", server_port=port, inbrowser=open_browser, head=hint_head())
