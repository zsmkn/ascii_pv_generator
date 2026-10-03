"""Cover / character image preparation: cut-out, main subject, and three crops.

  cut-out   an existing alpha channel is used as is; a plain background (the border is one colour, e.g. a white
            character sheet) is removed by flood-filling from the border; otherwise rembg is used if installed;
            otherwise the picture stays rectangular (fine for album covers)
  subject   the largest connected blob, plus blobs at least 20% of its size (drops chibi insets, confetti, ...)
  crops     full = the subject; bust = its top 42%; face = a square-ish box around the head (top ~18%)

Everything is written to work/<song>/cache/image/<name>_{full,bust,face}.png (RGBA). ID3 cover art is extracted by
extract_cover().
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

MAX_SIDE = 1400          # work resolution (the terminal never needs more)
BG_TOL = 22.0            # colour distance counted as "background"


def _plain_border(rgb: np.ndarray) -> np.ndarray | None:
    """The background colour if the image border is (nearly) one colour, else None."""
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]]).astype(float)
    med = np.median(border, axis=0)
    close = np.linalg.norm(border - med, axis=1) < BG_TOL
    return med if close.mean() > 0.9 else None


def cut_out(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    a = np.asarray(img)
    if a[..., 3].min() < 250:                       # already has transparency
        return img
    rgb = a[..., :3]
    bg = _plain_border(rgb)
    if bg is not None:
        near = np.linalg.norm(rgb.astype(float) - bg, axis=2) < BG_TOL
        lab, _ = ndimage.label(near)
        edge_labels = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
        bgmask = np.isin(lab, edge_labels[edge_labels > 0])
        # enclosed holes (between hair strands and arms, ...): background-coloured with a tighter tolerance and
        # big enough not to be a highlight on skin or cloth
        strict = np.linalg.norm(rgb.astype(float) - bg, axis=2) < BG_TOL * 0.55
        hl, hn = ndimage.label(strict & ~bgmask)
        if hn:
            sizes = ndimage.sum(np.ones_like(hl), hl, index=np.arange(1, hn + 1))
            big = np.nonzero(sizes >= 0.0015 * rgb.shape[0] * rgb.shape[1])[0] + 1
            bgmask |= np.isin(hl, big)
        bgmask = ndimage.binary_opening(bgmask, iterations=1)
        alpha = np.where(bgmask, 0, 255).astype(np.uint8)
        alpha = np.asarray(Image.fromarray(alpha).filter(ImageFilter.MedianFilter(3)))
        out = a.copy()
        out[..., 3] = alpha
        return Image.fromarray(out, "RGBA")
    try:
        from rembg import remove   # optional

        return remove(img).convert("RGBA")
    except Exception:
        return img                                  # keep the rectangle (e.g. an album cover)


def main_subject(img: Image.Image) -> Image.Image:
    a = np.asarray(img)
    solid = a[..., 3] > 40
    lab, n = ndimage.label(ndimage.binary_closing(solid, iterations=3))
    if n <= 1:
        return img.crop(img.getbbox() or (0, 0, img.width, img.height))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    keep = np.nonzero(sizes >= 0.2 * sizes.max())[0] + 1
    mask = np.isin(lab, keep) & solid
    out = a.copy()
    out[..., 3] = np.where(mask, a[..., 3], 0)
    ys, xs = np.nonzero(mask)
    return Image.fromarray(out, "RGBA").crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))


def crops(subject: Image.Image) -> dict[str, Image.Image]:
    w, h = subject.size
    alpha = np.asarray(subject)[..., 3] > 40
    tall = h > w * 1.3                               # a standing figure rather than a cover / portrait
    if not tall:
        return {"full": subject, "bust": subject, "face": subject}
    # head centre: centre of mass of the top 12% rows
    top = alpha[: max(1, int(h * 0.12))]
    cx = int(np.nonzero(top)[1].mean()) if top.any() else w // 2
    bust_h = int(h * 0.42)
    bw = min(w, int(bust_h * 0.9))
    bx = int(np.clip(cx - bw / 2, 0, w - bw))
    face_h = int(h * 0.19)
    fw = min(w, int(face_h * 1.1))
    fx = int(np.clip(cx - fw / 2, 0, w - fw))
    return {"full": subject, "bust": subject.crop((bx, 0, bx + bw, bust_h)),
            "face": subject.crop((fx, 0, fx + fw, face_h))}


def prepare(path: str | Path, cache: Path, name: str = "character") -> dict[str, str]:
    """-> {"full": png, "bust": png, "face": png} (paths)."""
    img = Image.open(path)
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    sub = main_subject(cut_out(img))
    out_dir = cache / "image"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for k, im in crops(sub).items():
        p = out_dir / f"{name}_{k}.png"
        im.save(p)
        paths[k] = str(p)
    return paths


def extract_cover(audio: str | Path, cache: Path) -> Path | None:
    """Embedded cover art (ID3 APIC / MP4 covr / FLAC picture) -> cache/image/cover_src.png, or None."""
    try:
        import mutagen

        f = mutagen.File(str(audio))
    except Exception:
        return None
    data = None
    if f is not None and f.tags is not None:
        for key in f.tags.keys():
            if str(key).startswith("APIC"):
                data = f.tags[key].data
                break
        if data is None and "covr" in f.tags:
            data = bytes(f.tags["covr"][0])
    if data is None and hasattr(f, "pictures") and f.pictures:
        data = f.pictures[0].data
    if not data:
        return None
    out = cache / "image" / "cover_src.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.open(io.BytesIO(data)).convert("RGB").save(out)
    return out
