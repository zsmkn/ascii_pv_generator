import numpy as np
import pytest
from PIL import Image, ImageDraw

from ascii_pv.analysis.image import crops, cut_out, main_subject, prepare
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.canvas import Canvas
from ascii_pv.core.palette import get_palette
from ascii_pv.core.storyboard import Shot, Storyboard
from ascii_pv.director.planner import plan
from ascii_pv.imaging import blit, cell_image, fit_cells, react
from ascii_pv.render import Renderer


def figure_on_white(tmp_path):
    """A tall red 'figure' with an enclosed white hole, plus a small separate blob (like an inset)."""
    im = Image.new("RGB", (300, 600), (255, 255, 255))
    d = ImageDraw.Draw(im)
    d.rectangle([100, 60, 220, 560], fill=(200, 30, 60))
    d.ellipse([120, 20, 200, 100], fill=(240, 200, 180))           # head
    d.rectangle([140, 250, 180, 330], fill=(255, 255, 255))        # enclosed hole
    d.ellipse([10, 10, 40, 40], fill=(20, 20, 200))                # small inset
    p = tmp_path / "fig.jpg"
    im.save(p, quality=95)
    return p


def test_cut_out_removes_border_background_and_holes(tmp_path):
    img = cut_out(Image.open(figure_on_white(tmp_path)))
    a = np.asarray(img)[..., 3]
    assert a[5, 5] == 0                                            # the border background is gone
    assert a[290, 160] == 0                                        # the enclosed white hole too
    assert a[400, 160] > 0                                         # the figure stays


def test_main_subject_drops_small_blobs_and_crops(tmp_path):
    sub = main_subject(cut_out(Image.open(figure_on_white(tmp_path))))
    assert sub.width < 160 and sub.height > 500                    # the inset is gone from the bbox
    c = crops(sub)
    assert c["bust"].height < c["full"].height and c["face"].height < c["bust"].height


def test_prepare_writes_three_pngs(tmp_path):
    paths = prepare(figure_on_white(tmp_path), tmp_path / "cache")
    assert set(paths) == {"full", "bust", "face"}
    for p in paths.values():
        assert Image.open(p).mode == "RGBA"


def test_fit_cells_keeps_aspect():
    cols, rows = fit_cells(1000, 2000, 160, 45)                    # tall picture, cells are 1:2
    assert rows == 45 and cols == 45


@pytest.mark.parametrize("mode", ["halfblock", "ramp", "glyph", "edge"])
def test_cell_image_modes(tmp_path, mode):
    paths = prepare(figure_on_white(tmp_path), tmp_path / "cache")
    ci = cell_image(paths["full"], 20, 30, mode, "source", get_palette("amber"))
    assert ci.chars.shape == (30, 20) and ci.alpha.any() and not ci.alpha.all()
    moved, margin = react(ci, 1.0, [0.8] * 7, kick=0.9, flux=0.9, energy=0.1, bpm=120, seed=1)
    assert moved.w == ci.w + 2 * margin
    c = Canvas(40, 35)
    blit(c, moved, 5, 2)
    assert (c.ch != " ").sum() > 20


def test_director_uses_the_picture(tmp_path):
    a = Analysis.synthetic(duration=60, bpm=120, seed=4)
    assets = {"character": prepare(figure_on_white(tmp_path), tmp_path / "cache")}
    b = plan(a, seed=1, assets=assets)
    assert b.shots[0].template == "image_ascii" and b.overlays["sprite"]["enabled"]
    assert not plan(a, seed=1).overlays["sprite"]["enabled"]
    r = Renderer(a, b)
    chorus = next(s for s in a.sections if s.label == "chorus")
    c = r.frame(chorus.start + 3.0, 120, 40)
    assert (c.ch == "▀").any() or (c.ch == "▄").any()              # the sprite is on screen


def test_image_template_without_assets_is_harmless():
    a = Analysis.synthetic(duration=10, seed=1)
    Renderer(a, Storyboard(shots=[Shot("x", 0, 10, "image_ascii")])).frame(2.0, 60, 20)
