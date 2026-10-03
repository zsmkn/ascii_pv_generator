"""Built-in templates register themselves on import."""

from ascii_pv.templates import (geometry, glyph_storm, image_ascii, landscape, matrix_rain,  # noqa: F401
                                particles, retro, scenes, shell, spectrum, tunnel, waves,
                                wireframe)
from ascii_pv.templates.base import REGISTRY, Template, TemplateMeta, create, register  # noqa: F401
