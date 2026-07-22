"""Classic 2048 tile colors."""

from __future__ import annotations

# Background and grid
BG_COLOR = (250, 248, 239)
GRID_COLOR = (187, 173, 160)
PANEL_COLOR = (238, 228, 218)
TEXT_DARK = (119, 110, 101)
TEXT_LIGHT = (249, 246, 242)
ACCENT = (143, 122, 102)
BUTTON = (143, 122, 102)
BUTTON_HOVER = (120, 100, 82)
BUTTON_ACTIVE = (100, 80, 65)
BUTTON_DISABLED = (190, 180, 170)

TILE_COLORS: dict[int, tuple[int, int, int]] = {
    0: (205, 193, 180),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46),
}

TILE_TEXT_DARK = {2, 4}


def tile_bg(value: int) -> tuple[int, int, int]:
    if value in TILE_COLORS:
        return TILE_COLORS[value]
    return (60, 58, 50)


def tile_fg(value: int) -> tuple[int, int, int]:
    return TEXT_DARK if value in TILE_TEXT_DARK else TEXT_LIGHT
