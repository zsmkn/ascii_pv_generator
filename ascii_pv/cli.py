"""Command line entry point: ascii-pv <command> ...

Commands: analyze, info, plan, play, render, all, preview, templates, gui.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ascii_pv.core.analysis import Analysis
from ascii_pv.core.palette import PALETTES
from ascii_pv.core.storyboard import Storyboard

PENDING: dict[str, str] = {}


def _size(s: str) -> tuple[int, int]:
    try:
        w, h = s.lower().split("x")
        return int(w), int(h)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected WxH, e.g. 120x40, got {s!r}") from None


AUDIO_EXT = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac", ".opus"}


def _board_for(analysis: Analysis, work: Path | None, args) -> Storyboard:
    """storyboard.json if the work dir has one; else a single template (--template) or a fresh plan."""
    if getattr(args, "template", None):
        board = Storyboard.single(args.template, analysis.duration, seed=args.seed)
    elif work is not None and (work / "storyboard.json").exists():
        board = Storyboard.load(work / "storyboard.json")
    else:
        from ascii_pv.director.planner import plan

        board = plan(analysis, seed=args.seed)
    if args.palette:
        board.palette = args.palette
    return board


def _load(args) -> tuple[Analysis, Storyboard, Path | None]:
    """Analysis + storyboard from a work dir, an audio file (analysed on the fly, cached), or --demo."""
    if getattr(args, "demo", False):
        analysis = Analysis.synthetic(duration=args.duration, bpm=args.bpm, seed=args.seed)
        return analysis, _board_for(analysis, None, args), None
    if not args.work:
        raise SystemExit("give a work dir (work/<song-id>), an audio file, or --demo")
    target = Path(args.work)
    if target.is_file() and target.suffix.lower() in AUDIO_EXT:
        from ascii_pv.analysis.pipeline import analyze

        analysis, wd = analyze(target, getattr(args, "lyrics", None), work_root=Path("work"))
        return analysis, _board_for(analysis, wd.root, args), wd.root
    analysis = Analysis.load(target / "analysis.json")
    return analysis, _board_for(analysis, target, args), target


def cmd_preview(args) -> int:
    from ascii_pv.render import Renderer

    analysis, board, _ = _load(args)
    w, h = args.size
    c = Renderer(analysis, board).frame(args.at, w, h)
    if args.png:
        from ascii_pv.backends.raster import to_image

        to_image(c).save(args.png)
        print(f"wrote {args.png}")
    elif args.plain:
        sys.stdout.write(c.plain() + "\n")
    else:
        sys.stdout.write("\x1b[2J" + c.ansi() + f"\x1b[{h + 1};1H\n")
    return 0


def cmd_play(args) -> int:
    from ascii_pv.player import Player

    analysis, board, _ = _load(args)
    if getattr(args, "subtitle", False):
        board.overlays["subtitle"] = True
    audio = args.audio or analysis.source.get("audio")
    report = Player(analysis, board, audio, fps=args.fps, mute=args.mute, lyric_offset=args.offset,
                    stats=args.stats).run(args.start, args.stop_after)
    if args.stop_after is not None:
        print(report)
    return 0


def cmd_plan(args) -> int:
    from ascii_pv.director.planner import plan

    work = Path(args.work)
    analysis = Analysis.load(work / "analysis.json")
    sb_path = work / "storyboard.json"
    lock = Storyboard.load(sb_path) if args.lock and sb_path.exists() else None
    if args.lock and lock is None:
        print("note: --lock given but there is no storyboard.json yet", file=sys.stderr)
    board = plan(analysis, seed=args.seed, palette=args.palette, lock=lock,
                 assets=_image_assets(work, args.image, lock))
    board.save(sb_path)
    counts: dict[str, int] = {}
    for s in board.shots:
        counts[s.template] = counts.get(s.template, 0) + 1
    modes: dict[str, int] = {}
    for c in board.lyrics:
        key = f"{c.mode}/{c.enter}"
        modes[key] = modes.get(key, 0) + 1
    print(f"palette {board.palette}   {len(board.shots)} shots: "
          + ", ".join(f"{k} x{v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])))
    print("lyrics  " + ", ".join(f"{k} x{v}" for k, v in sorted(modes.items())))
    print(f"wrote {sb_path}")
    return 0


def _image_assets(work: Path, image: str | None, lock: Storyboard | None = None) -> dict | None:
    """--image, else the lock's assets, else the song's embedded cover art, else none."""
    from ascii_pv.analysis.image import prepare

    if image:
        print(f"preparing {image} ...")
        return {"character": prepare(image, work / "cache")}
    if lock is not None and lock.assets:
        return lock.assets
    cover = work / "cache" / "image" / "cover_src.png"
    if cover.exists():
        return {"character": prepare(cover, work / "cache")}
    return {}


def _ensure_board(analysis: Analysis, work: Path, seed: int = 0, image: str | None = None) -> Path:
    """storyboard.json of the work dir, planned now if missing (or re-planned when a new image is given)."""
    sb = work / "storyboard.json"
    if not sb.exists() or image:
        from ascii_pv.director.planner import plan

        plan(analysis, seed=seed, assets=_image_assets(work, image)).save(sb)
        print(f"planned {sb}")
    return sb


def cmd_render(args) -> int:
    from ascii_pv.backends.video import VideoJob, render_video

    work = Path(args.work)
    analysis = Analysis.load(work / "analysis.json")
    sb = _ensure_board(analysis, work, args.seed)
    audio = args.audio or analysis.source.get("audio")
    if audio and not Path(audio).exists():
        print(f"note: audio {audio} not found; rendering without sound", file=sys.stderr)
        audio = None
    out = args.output or str(Path("out") / f"{work.name}.mp4")
    job = VideoJob(analysis=str(work / "analysis.json"), storyboard=str(sb), out=out, audio=audio,
                   size=args.size, grid=args.grid or Storyboard.load(sb).grid,
                   fps=args.fps, start=args.start, end=args.end, post=not args.no_post, crf=args.crf,
                   preset=args.preset, lyric_offset=args.offset, workers=args.workers, subtitle=args.subtitle)
    render_video(job, work / "cache")
    return 0


def cmd_all(args) -> int:
    from ascii_pv.analysis.pipeline import analyze

    analysis, wd = analyze(args.song, args.lyrics, work_root=Path("work"), retime=args.retime)
    _ensure_board(analysis, wd.root, args.seed, args.image)
    args.work = str(wd.root)
    if args.output:
        return cmd_render(args)
    args.demo = False
    args.template = None
    return cmd_play(args)


def cmd_analyze(args) -> int:
    from ascii_pv.analysis.pipeline import analyze, summary

    a, work = analyze(args.song, args.lyrics, bpm=args.bpm, lyrics_offset=args.lyrics_offset,
                      lyrics_scale=args.lyrics_scale, work_root=Path(args.work_root), force=args.force,
                      ml=not args.no_ml, device="cpu" if args.cpu else None, sync=not args.no_sync,
                      retime=args.retime)
    print(summary(a))
    print(f"work dir: {work.root}")
    return 0


def cmd_info(args) -> int:
    from ascii_pv.analysis.pipeline import summary

    print(summary(Analysis.load(Path(args.work) / "analysis.json")))
    return 0


def cmd_gui(args) -> int:
    from ascii_pv.gui import launch

    launch(port=args.port, open_browser=not args.no_browser)
    return 0


def cmd_templates(args) -> int:
    from ascii_pv.templates import REGISTRY

    for name in sorted(REGISTRY):
        m = REGISTRY[name].meta
        print(f"{name:18} {'(easter egg) ' if m.easter_egg else ''}{m.description}")
    return 0


def main(argv: list[str] | None = None) -> None:
    RETIME_HELP = ("ignore the lyric file's timestamps and time its text from the vocals, like plain-text lyrics "
                   "(for an inaccurate LRC / SRT; needs the ML layer)")
    p = argparse.ArgumentParser(prog="ascii-pv", description="Music + lyrics -> ASCII PV")
    sub = p.add_subparsers(dest="cmd", required=True)

    an = sub.add_parser("analyze", help="analyse a song (+ lyrics) into work/<song-id>/analysis.json")
    an.add_argument("song")
    an.add_argument("--lyrics", help="LRC, SRT, or plain text (one line per lyric line: timed automatically)")
    an.add_argument("--retime", action="store_true", help=RETIME_HELP)
    an.add_argument("--bpm", type=float, help="override the detected tempo")
    an.add_argument("--lyrics-offset", type=float, default=0.0, help="seconds added to every lyric time")
    an.add_argument("--lyrics-scale", type=float, default=1.0,
                    help="multiply lyric times (for a lyric file timed on another version of the song)")
    an.add_argument("--work-root", default="work")
    an.add_argument("--force", action="store_true", help="ignore the cache")
    an.add_argument("--no-ml", action="store_true", help="base layer only (librosa), even if torch is installed")
    an.add_argument("--cpu", action="store_true", help="run the ML models on the CPU")
    an.add_argument("--no-sync", action="store_true", help="do not auto-align the lyrics to the vocals")
    an.set_defaults(func=cmd_analyze)

    pn = sub.add_parser("plan", help="direct the PV: write work/<song-id>/storyboard.json")
    pn.add_argument("work")
    pn.add_argument("--seed", type=int, default=0, help="another seed gives another cut of the same song")
    pn.add_argument("--palette", choices=sorted(PALETTES), help="override the palette chosen from the style")
    pn.add_argument("--image", help="character / cover picture to put in the PV (default: embedded cover art)")
    pn.add_argument("--lock", action="store_true",
                    help="keep shots / lyric cues marked \"locked\": true in the existing storyboard.json")
    pn.set_defaults(func=cmd_plan)

    inf = sub.add_parser("info", help="print the summary of an analysed song")
    inf.add_argument("work")
    inf.set_defaults(func=cmd_info)

    pv = sub.add_parser("preview", help="render one frame to the terminal (or as plain text)")
    pv.add_argument("work", nargs="?", help="work dir, or an audio file (analysed first, cached)")
    pv.add_argument("--lyrics", help="LRC / SRT / plain text when `work` is an audio file")
    pv.add_argument("--at", type=float, default=0.0, help="song time in seconds")
    pv.add_argument("--size", type=_size, default=(120, 40), help="grid size WxH (default 120x40)")
    pv.add_argument("--plain", action="store_true", help="plain text, no colour")
    pv.add_argument("--png", help="save the frame as an image instead")
    pv.add_argument("--demo", action="store_true", help="use a synthetic song instead of a work dir")
    pv.add_argument("--duration", type=float, default=60.0, help="--demo song length")
    pv.add_argument("--bpm", type=float, default=120.0, help="--demo tempo")
    pv.add_argument("--template", help="show one template for the whole song instead of the storyboard")
    pv.add_argument("--palette", choices=sorted(PALETTES), help="override the storyboard palette")
    pv.add_argument("--seed", type=int, default=0)
    pv.set_defaults(func=cmd_preview)

    pl = sub.add_parser("play", help="play the PV in this terminal, in sync with the song")
    pl.add_argument("work", help="work dir, or an audio file (analysed first, cached)")
    pl.add_argument("--lyrics", help="LRC / SRT / plain text when `work` is an audio file")
    pl.add_argument("--audio", help="audio to play (default: the analysed file)")
    pl.add_argument("--start", type=float, default=0.0, help="start time in seconds")
    pl.add_argument("--fps", type=int, default=30)
    pl.add_argument("--offset", type=float, default=0.0, help="lyrics earlier (+) / later (-), seconds")
    pl.add_argument("--mute", action="store_true", help="no sound (the clock runs on its own)")
    pl.add_argument("--stats", action="store_true", help="show render time in the status line")
    pl.add_argument("--subtitle", action="store_true", help="also show a plain caption on the bottom row")
    pl.add_argument("--stop-after", type=float, help=argparse.SUPPRESS)   # testing: quit at this song time
    pl.add_argument("--template", help="show one template for the whole song instead of the storyboard")
    pl.add_argument("--palette", choices=sorted(PALETTES))
    pl.add_argument("--seed", type=int, default=0)
    pl.set_defaults(func=cmd_play)

    def video_args(sp):
        sp.add_argument("--size", type=_size, default=(1280, 720), help="video size WxH (default 1280x720)")
        sp.add_argument("--grid", type=_size, help="character grid WxH (default: the storyboard's, 160x45)")
        sp.add_argument("--fps", type=int, default=24)
        sp.add_argument("--start", type=float, default=0.0)
        sp.add_argument("--end", type=float)
        sp.add_argument("--offset", type=float, default=0.0, help="extra lyric offset (s), + = earlier")
        sp.add_argument("--no-post", action="store_true", help="no bloom / trails / scanlines")
        sp.add_argument("--crf", type=int, default=23, help="x264 quality (lower = better, bigger)")
        sp.add_argument("--preset", default="medium", help="x264 preset")
        sp.add_argument("--workers", type=int, default=0, help="parallel workers (default: half the CPUs)")
        sp.add_argument("--audio", help="audio to mux (default: the analysed file)")
        sp.add_argument("--seed", type=int, default=0, help="seed when a storyboard must be planned first")
        sp.add_argument("--subtitle", action="store_true", help="also show a plain caption on the bottom row")

    rd = sub.add_parser("render", help="render the PV to MP4")
    rd.add_argument("work")
    rd.add_argument("-o", "--output", help="output file (default out/<song-id>.mp4)")
    video_args(rd)
    rd.set_defaults(func=cmd_render)

    al = sub.add_parser("all", help="analyse + plan, then play (default) or render with -o")
    al.add_argument("song")
    al.add_argument("--lyrics", help="LRC, SRT, or plain text")
    al.add_argument("--retime", action="store_true", help=RETIME_HELP)
    al.add_argument("--image", help="character / cover picture to put in the PV")
    al.add_argument("-o", "--output", help="render to this MP4 instead of playing")
    video_args(al)
    al.add_argument("--mute", action="store_true")
    al.add_argument("--stats", action="store_true")
    al.add_argument("--palette", choices=sorted(PALETTES))
    al.add_argument("--stop-after", type=float, help=argparse.SUPPRESS)
    al.set_defaults(func=cmd_all)

    gu = sub.add_parser("gui", help="open the web GUI (local only) in your browser")
    gu.add_argument("--port", type=int, default=7860)
    gu.add_argument("--no-browser", action="store_true")
    gu.set_defaults(func=cmd_gui)

    tp = sub.add_parser("templates", help="list the registered templates")
    tp.set_defaults(func=cmd_templates)

    for name, when in PENDING.items():
        sp = sub.add_parser(name, help=f"(not yet implemented: {when})")
        sp.add_argument("rest", nargs=argparse.REMAINDER)
        sp.set_defaults(func=lambda a, n=name, w=when: (print(f"'{n}' arrives in {w}.", file=sys.stderr), 2)[1])

    args = p.parse_args(argv)
    from ascii_pv.backends.winconsole import setup_console

    setup_console()  # UTF-8 everywhere, even when piped: cp950 cannot encode block glyphs
    sys.exit(args.func(args))
