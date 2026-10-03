"""work/<id>/ layout: one folder per song holding every intermediate file.

  analysis.json     Analyzer output
  storyboard.json   Director output (user-editable)
  cache/            stems, image crops, ink masks, ...
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

WORK_ROOT = Path("work")


def file_id(path: str | Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


@dataclass(frozen=True)
class WorkDir:
    root: Path

    @classmethod
    def for_song(cls, song: str | Path, base: Path = WORK_ROOT) -> "WorkDir":
        song = Path(song)
        return cls(base / f"{song.stem}-{file_id(song, 8)}")

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def storyboard(self) -> Path:
        return self.root / "storyboard.json"

    @property
    def cache(self) -> Path:
        return self.root / "cache"

    def ensure(self) -> "WorkDir":
        self.cache.mkdir(parents=True, exist_ok=True)
        return self
