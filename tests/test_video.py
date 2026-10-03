import json
import subprocess

import numpy as np
import pytest

from ascii_pv.analysis.ffmpeg import FFmpegNotFound, tool
from ascii_pv.backends.postfx import PostFX
from ascii_pv.backends.raster import Rasterizer
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.director.planner import plan


def test_rasterizer_cells_and_colours():
    ras = Rasterizer(8, 16)
    c = Canvas(4, 2, bg=(10, 20, 30), fg=(200, 100, 50))
    c.put(0, 0, "█")
    c.put(1, 0, "A")
    px = ras.render(c)
    assert px.shape == (32, 32, 3)
    assert (px[0:16, 0:8] == (200, 100, 50)).all()          # full block fills its cell exactly
    assert (px[16:32] == (10, 20, 30)).all()                 # blank row is pure background
    a = px[0:16, 8:16].reshape(-1, 3)
    assert (a == (200, 100, 50)).all(axis=1).any() and (a == (10, 20, 30)).all(axis=1).any()


def test_rasterizer_wide_char_spans_two_cells():
    ras = Rasterizer(8, 16)
    c = Canvas(3, 1)
    c.put(0, 0, "愛")
    px = ras.render(c)
    assert px[:, 0:8].any() and px[:, 8:16].any() and not px[:, 16:24].any()


def test_postfx_keeps_shape_and_trails():
    fx = PostFX()
    a = np.zeros((64, 64, 3), np.uint8)
    lit = a.copy()
    lit[20:30, 20:30] = 250
    out = fx.apply(a, fx.apply(lit, None))                   # dark frame after a bright one
    assert out.shape == a.shape and out[25, 25].max() > 30   # the bright square lingers


@pytest.fixture
def ffmpeg():
    try:
        return tool("ffmpeg")
    except FFmpegNotFound:
        pytest.skip("ffmpeg not installed")


def test_render_video_end_to_end(tmp_path, ffmpeg):
    from ascii_pv.backends.video import VideoJob, render_video

    a = Analysis.synthetic(duration=3.0, bpm=120, seed=1)
    a.save(tmp_path / "analysis.json")
    plan(a).save(tmp_path / "storyboard.json")
    out = tmp_path / "o.mp4"
    job = VideoJob(analysis=str(tmp_path / "analysis.json"), storyboard=str(tmp_path / "storyboard.json"),
                   out=str(out), audio=None, size=(320, 180), grid=(40, 12), fps=12, workers=2, preset="ultrafast")
    render_video(job, tmp_path / "cache", log=lambda *a, **k: None)
    info = json.loads(subprocess.run([tool("ffprobe"), "-v", "error", "-show_entries",
                                      "stream=width,height,nb_frames", "-of", "json", str(out)],
                                     capture_output=True, text=True).stdout)["streams"][0]
    assert (info["width"], info["height"]) == (320, 180) and int(info["nb_frames"]) == 36
