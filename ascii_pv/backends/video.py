"""MP4 output: Renderer -> Rasterizer -> PostFX -> ffmpeg, in parallel.

The frame range is split into one contiguous chunk per worker. Each worker renders its frames and pipes raw RGB
into its own ffmpeg (H.264, video only) -> cache/render/seg_NN.mp4. Trails need the previous output frame, so a
worker first renders the frame just before its chunk (not written) to seed them. The segments are joined with
the concat demuxer (no re-encode) and the song is muxed in, trimmed to the rendered range.
"""

from __future__ import annotations

import math
import multiprocessing as mp
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from ascii_pv.analysis.ffmpeg import tool


@dataclass
class VideoJob:
    analysis: str            # path to analysis.json
    storyboard: str          # path to storyboard.json
    out: str
    audio: str | None
    size: tuple[int, int] = (1280, 720)
    grid: tuple[int, int] = (160, 45)
    fps: int = 24
    start: float = 0.0
    end: float | None = None
    post: bool = True
    crf: int = 23
    preset: str = "medium"
    lyric_offset: float = 0.0
    workers: int = 0          # 0 = half the CPUs
    subtitle: bool = False    # plain caption line on the bottom row


# ---------------------------------------------------------------- worker

_counter = None


def _init(counter) -> None:
    global _counter
    _counter = counter


def _encoder_cmd(job: VideoJob, W: int, H: int, out: Path) -> list[str]:
    return [tool("ffmpeg"), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
            "-r", str(job.fps), "-i", "-", "-an", "-c:v", "libx264", "-preset", job.preset, "-crf", str(job.crf),
            "-pix_fmt", "yuv420p", "-threads", "2", "-movflags", "+faststart", str(out)]


def _render_chunk(job_d: dict, first: int, last: int, seg: str) -> int:
    from ascii_pv.backends.postfx import PostFX
    from ascii_pv.backends.raster import Rasterizer
    from ascii_pv.core.analysis import Analysis
    from ascii_pv.core.storyboard import Storyboard
    from ascii_pv.render import Renderer

    job = VideoJob(**job_d)
    a = Analysis.load(job.analysis)
    board = Storyboard.load(job.storyboard)
    if job.subtitle:
        board.overlays["subtitle"] = True
    r = Renderer(a, board)
    gw, gh = job.grid
    W, H = job.size
    cw, ch = W // gw, H // gh
    ras = Rasterizer(cw, ch)
    post = PostFX() if job.post else None
    pad = ((H - gh * ch) // 2, (W - gw * cw) // 2)
    bg = np.array(r.palette.bg, np.uint8)

    def frame_px(i: int) -> np.ndarray:
        t = job.start + i / job.fps
        px = ras.render(r.frame(t, gw, gh, job.lyric_offset))
        if px.shape[:2] != (H, W):                     # grid does not divide the size: centre it
            full = np.empty((H, W, 3), np.uint8)
            full[:] = bg
            full[pad[0]:pad[0] + px.shape[0], pad[1]:pad[1] + px.shape[1]] = px
            px = full
        return px

    prev = None
    if post and first > 0:
        prev = post.apply(frame_px(first - 1), None)
    proc = subprocess.Popen(_encoder_cmd(job, W, H, Path(seg)), stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        for i in range(first, last):
            px = frame_px(i)
            if post:
                px = post.apply(px, prev)
                prev = px
            proc.stdin.write(px.tobytes())
            if _counter is not None:
                with _counter.get_lock():
                    _counter.value += 1
        proc.stdin.close()
        err = proc.stderr.read().decode(errors="replace")
        if proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed on {seg}: {err}")
    except BaseException:
        proc.kill()
        raise
    return last - first


# ---------------------------------------------------------------- driver

def render_video(job: VideoJob, cache_dir: Path, log=print, progress=None) -> Path:
    """progress(done, total) is called about four times a second while frames are rendered (for GUIs)."""
    from ascii_pv.core.analysis import Analysis

    duration = Analysis.load(job.analysis).duration
    end = min(duration, job.end if job.end is not None else duration)
    n = int(math.floor((end - job.start) * job.fps + 1e-6))
    if n <= 0:
        raise ValueError("nothing to render: check --start / --end")
    workers = job.workers or max(1, (os.cpu_count() or 2) // 2)
    workers = max(1, min(workers, n // 12 or 1))
    tmp = cache_dir / "render"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    bounds = [round(k * n / workers) for k in range(workers + 1)]
    segs = [tmp / f"seg_{k:02d}.mp4" for k in range(workers)]

    log(f"rendering {n} frames ({end - job.start:.1f} s at {job.fps} fps, {job.size[0]}x{job.size[1]}) "
        f"with {workers} workers ...")
    t0 = time.time()
    counter = mp.get_context("spawn").Value("i", 0)
    jd = asdict(job)
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"), initializer=_init,
                             initargs=(counter,)) as ex:
        futs = [ex.submit(_render_chunk, jd, bounds[k], bounds[k + 1], str(segs[k])) for k in range(workers)]
        last_print = 0.0
        while not all(f.done() for f in futs):
            time.sleep(0.25)
            done = counter.value
            if progress is not None:
                progress(done, n)
            now = time.time()
            if now - last_print > 1.0:
                el = now - t0
                eta = el / done * (n - done) if done else 0
                log(f"\r  {done}/{n} frames  {done / max(el, 1e-6):5.1f} fps  eta {eta:5.0f} s", end="")
                last_print = now
        for f in futs:
            f.result()                                   # re-raise worker errors
    log(f"\r  {n}/{n} frames in {time.time() - t0:.1f} s" + " " * 20)

    listing = tmp / "segments.txt"
    listing.write_text("".join(f"file '{s.resolve().as_posix()}'\n" for s in segs), encoding="utf-8")
    out = Path(job.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [tool("ffmpeg"), "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(listing)]
    if job.audio:
        cmd += ["-ss", f"{job.start:.3f}", "-t", f"{n / job.fps:.3f}", "-i", job.audio,
                "-map", "0:v", "-map", "1:a", "-c:a", "aac", "-b:a", "256k", "-shortest"]
    cmd += ["-c:v", "copy", "-movflags", "+faststart", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"ffmpeg mux failed: {res.stderr.strip()}")
    shutil.rmtree(tmp, ignore_errors=True)
    log(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB, total {time.time() - t0:.1f} s)")
    return out
