"""The ML analysis layer (optional: `pip install -e .[ml]`, CUDA torch recommended).

  stems.py   demucs (htdemucs): the vocal track -> vocal level curve, vocal onsets, lyric sync
  beats.py   beat_this: beats and downbeats
  clap.py    CLAP zero-shot: genre / mood tags
  asr.py     wav2vec2 (MMS_FA) character log-probs + Whisper word timestamps, for plain-text lyrics (align.py)

Each module imports its heavy dependencies lazily; `available()` says what can run here.
"""

from __future__ import annotations

import importlib.util
import os

# Windows without developer mode cannot symlink; the cache still works, the warning is just noise
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


def _has(mod: str) -> bool:
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


def available() -> dict[str, bool]:
    torch = _has("torch")
    return {"torch": torch, "demucs": torch and _has("demucs"), "beat_this": torch and _has("beat_this"),
            "clap": torch and _has("transformers")}


def device(prefer: str | None = None) -> str:
    if prefer:
        return prefer
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"
