"""Locate ffmpeg/ffprobe and decode audio through them.

Lookup order: $ASCII_PV_FFMPEG (path to ffmpeg.exe or its folder) -> PATH -> WinGet's package folder
(winget installs update PATH only for new shells, so a fresh install is found without a restart).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


class FFmpegNotFound(RuntimeError):
    pass


def _candidates(name: str):
    exe = name + (".exe" if sys.platform == "win32" else "")
    env = os.environ.get("ASCII_PV_FFMPEG")
    if env:
        p = Path(env)
        yield p.parent / exe if p.is_file() else p / exe
    found = shutil.which(name)
    if found:
        yield Path(found)
    if sys.platform == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", ""))
        yield local / "Microsoft" / "WinGet" / "Links" / exe
        yield from sorted((local / "Microsoft" / "WinGet" / "Packages").glob(f"*FFmpeg*/**/bin/{exe}"), reverse=True)


@lru_cache(None)
def tool(name: str = "ffmpeg") -> str:
    for p in _candidates(name):
        if p.is_file():
            return str(p)
    raise FFmpegNotFound(f"{name} not found. Install it (winget install Gyan.FFmpeg) or set ASCII_PV_FFMPEG "
                         f"to its folder.")


def decode(path: str | Path, sr: int, mono: bool = True) -> np.ndarray:
    """Decode any audio file to float32 samples: shape (n,) if mono else (n, 2)."""
    ch = 1 if mono else 2
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-vn", "-f", "f32le", "-ac", str(ch), "-ar", str(sr), "-"]
    out = subprocess.run(cmd, capture_output=True, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"ffmpeg could not decode {path}: {out.stderr.decode(errors='replace').strip()}")
    y = np.frombuffer(out.stdout, dtype=np.float32)
    return y if mono else y.reshape(-1, 2)


def probe_duration(path: str | Path) -> float:
    cmd = [tool("ffprobe"), "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)]
    out = subprocess.run(cmd, capture_output=True, check=True, text=True)
    return float(json.loads(out.stdout)["format"]["duration"])
