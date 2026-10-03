"""Semantic palettes. Templates ask for a role (ui, accent, ...) and a level, never a raw colour,
so a style change only swaps the palette.

Roles:
  bg       background
  ui       the main "system" colour (text, lines, most glyphs)
  accent   highlights: the sung word, sweeps, beat flashes
  err      errors / red alerts
  anom     anomalies / warnings
"""

from __future__ import annotations

from dataclasses import dataclass

RGB = tuple[int, int, int]


def mix(c: RGB, level: float, base: RGB) -> RGB:
    level = max(0.0, min(1.0, level))
    return (int(base[0] + (c[0] - base[0]) * level),
            int(base[1] + (c[1] - base[1]) * level),
            int(base[2] + (c[2] - base[2]) * level))


@dataclass(frozen=True)
class Palette:
    name: str
    bg: RGB
    ui: RGB
    accent: RGB
    err: RGB
    anom: RGB

    def ui_at(self, level: float) -> RGB:
        return mix(self.ui, level, self.bg)

    def accent_at(self, level: float) -> RGB:
        return mix(self.accent, level, self.bg)

    def err_at(self, level: float) -> RGB:
        return mix(self.err, level, self.bg)

    def anom_at(self, level: float) -> RGB:
        return mix(self.anom, level, self.bg)

    def ramp(self, level: float) -> RGB:
        """One continuous scale: dim ui -> ui -> accent (level 0..1)."""
        level = max(0.0, min(1.0, level))
        if level < 0.7:
            return self.ui_at(0.15 + level / 0.7 * 0.85)
        return mix(self.accent, (level - 0.7) / 0.3, self.ui)


PALETTES: dict[str, Palette] = {p.name: p for p in [
    # amber phosphor (ref A / ref B "amber")
    Palette("amber", bg=(8, 6, 4), ui=(255, 176, 0), accent=(255, 236, 190), err=(255, 58, 40), anom=(70, 255, 140)),
    # green phosphor
    Palette("phosphor", bg=(3, 8, 5), ui=(92, 255, 140), accent=(220, 255, 230), err=(255, 58, 40), anom=(255, 214, 0)),
    # cold steel on navy
    Palette("deepsea", bg=(4, 7, 15), ui=(200, 214, 234), accent=(126, 152, 255), err=(255, 59, 48), anom=(255, 204, 0)),
    # neon pink / cyan
    Palette("synthwave", bg=(12, 4, 20), ui=(255, 90, 200), accent=(90, 240, 255), err=(255, 60, 60), anom=(255, 230, 90)),
    # plain white on black
    Palette("mono", bg=(0, 0, 0), ui=(210, 210, 210), accent=(255, 255, 255), err=(255, 80, 80), anom=(255, 220, 120)),
]}


def get_palette(name: str) -> Palette:
    try:
        return PALETTES[name]
    except KeyError:
        raise KeyError(f"unknown palette {name!r}; available: {', '.join(PALETTES)}") from None
