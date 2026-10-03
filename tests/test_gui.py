"""GUI handlers, exercised without a browser on a synthetic song in a temporary work root."""

import json

import pytest

from ascii_pv import gui
from ascii_pv.core.analysis import Analysis
from ascii_pv.core.storyboard import Storyboard


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(gui, "WORK_ROOT", tmp_path / "work")
    wd = tmp_path / "work" / "song-abc"
    (wd / "cache").mkdir(parents=True)
    Analysis.synthetic(duration=60, bpm=120, seed=2).save(wd / "analysis.json")
    monkeypatch.chdir(tmp_path)
    return wd.name


def test_reopen_plan_edit_save_roundtrip(work):
    assert gui.list_works() == [work]
    name, text = gui.do_analyze(None, None, None, work, False, False)
    assert name == work and "bpm 120" in text
    info, shots, cues = gui.do_plan(work, 3, "synthwave", 0.25, False)
    assert "synthwave" in info and len(shots) > 1 and len(cues) > 0
    board = Storyboard.load(gui.WORK_ROOT / work / "storyboard.json")
    assert board.palette == "synthwave" and board.lyric_offset == 0.25

    shots.loc[0, "template"] = "heart"
    shots.loc[0, "fx"] = json.dumps({"zoom": 0.5})
    shots.loc[0, "locked"] = True
    cues.loc[0, "mode"] = "word_flash"
    assert gui.do_save(work, shots, cues).startswith("saved")
    board = Storyboard.load(gui.WORK_ROOT / work / "storyboard.json")
    assert board.shots[0].template == "heart" and board.shots[0].fx == {"zoom": 0.5} and board.shots[0].locked
    assert board.lyrics[0].mode == "word_flash"

    # locked rows survive a re-plan
    _, shots2, _ = gui.do_plan(work, 9, "auto", 0.0, True)
    assert shots2.loc[shots2["id"] == board.shots[0].id, "template"].iloc[0] == "heart"


def test_bad_edits_are_refused(work):
    _, shots, cues = gui.do_plan(work, 0, "auto", 0.0, False)
    shots.loc[0, "template"] = "no_such_template"
    assert gui.do_save(work, shots, cues).startswith("not saved")
    shots.loc[0, "template"] = "blank"
    shots.loc[0, "params"] = "{broken json"
    assert gui.do_save(work, shots, cues).startswith("not saved")


def test_preview_image(work):
    gui.do_plan(work, 0, "auto", 0.0, False)
    img = gui.do_preview(work, 5.0, "854x480 (quick)")
    assert img.size == (848, 480)          # 106 x 30 cells of 8 x 16 px


def test_render_short_clip_with_progress(work):
    pytest.importorskip("subprocess")
    from ascii_pv.analysis.ffmpeg import FFmpegNotFound, tool

    try:
        tool("ffmpeg")
    except FFmpegNotFound:
        pytest.skip("ffmpeg not installed")
    gui.do_plan(work, 0, "auto", 0.0, False)
    seen = []
    out, log = gui.do_render(work, "854x480 (quick)", "draft (fast, small)", 2.0, 4.0, False, True,
                             progress=lambda d, n: seen.append((d, n)))
    assert out.endswith(".mp4") and "wrote" in log
    from pathlib import Path

    assert Path(out).exists()


def test_build_ui():
    pytest.importorskip("gradio")
    assert gui.build() is not None


def test_every_control_has_a_hover_hint():
    import gradio as gr

    ui = gui.build()
    blocks = list(ui.blocks.values())
    ids = {b.elem_id for b in blocks if getattr(b, "elem_id", None)}
    assert ids == set(gui.TIPS)                                   # no widget without a hint, no stale hint
    assert all(b.elem_id for b in blocks if isinstance(b, (gr.Button, gr.Checkbox, gr.Dropdown, gr.Tab)))
    assert list(gui.COL_TIPS["shots"]) == gui.SHOT_COLS and list(gui.COL_TIPS["cues"]) == gui.CUE_COLS
    head = gui.hint_head()
    script = head.split("<script>", 1)[1].removesuffix("</script>")
    assert "</" not in script and json.dumps(gui.TIPS["refresh"]) in script
