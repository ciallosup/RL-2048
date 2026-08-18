"""Render a 2048 board to a Pillow image (for README GIFs)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl2048.viz import colors

HEADER_PX = 36


def _font(size: int, *, bold: bool = True, cjk: bool = False) -> ImageFont.ImageFont:
    windows = Path("C:/Windows/Fonts")
    if cjk:
        candidates = (
            windows / "msyhbd.ttc",
            windows / "msyh.ttc",
            windows / "msyhbd.ttf",
            windows / "msyh.ttf",
            windows / "simhei.ttf",
        )
    else:
        names = ("arialbd.ttf", "arial.ttf") if bold else ("arial.ttf",)
        candidates = tuple(windows / name for name in names)
    for path in candidates:
        if not path.exists():
            continue
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    try:
        return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def render_board(
    board: np.ndarray,
    *,
    tile_px: int = 64,
    gap: int = 6,
    pad: int = 10,
    caption: str | None = None,
) -> Image.Image:
    """Draw a classic 4x4 2048 grid. Optional caption sits in a header bar."""
    grid = np.asarray(board, dtype=int).reshape(4, 4)
    inner = 4 * tile_px + 5 * gap
    width = inner + 2 * pad
    height = inner + 2 * pad + (HEADER_PX if caption else 0)
    image = Image.new("RGB", (width, height), colors.BG_COLOR)
    draw = ImageDraw.Draw(image)

    top = HEADER_PX if caption else 0
    if caption:
        draw.rectangle((0, 0, width, HEADER_PX), fill=colors.BG_COLOR)
        caption_font = _font(max(13, tile_px // 5), bold=True, cjk=True)
        draw.text((pad, (HEADER_PX - tile_px // 5) // 2), caption, fill=colors.TEXT_DARK, font=caption_font)

    grid_x = pad
    grid_y = top + pad
    draw.rounded_rectangle(
        (grid_x, grid_y, grid_x + inner, grid_y + inner),
        radius=max(4, gap),
        fill=colors.GRID_COLOR,
    )

    number_font_cache: dict[int, ImageFont.ImageFont] = {}
    for row in range(4):
        for col in range(4):
            value = int(grid[row, col])
            x0 = grid_x + gap + col * (tile_px + gap)
            y0 = grid_y + gap + row * (tile_px + gap)
            x1 = x0 + tile_px
            y1 = y0 + tile_px
            draw.rounded_rectangle((x0, y0, x1, y1), radius=max(3, tile_px // 16), fill=colors.tile_bg(value))
            if value == 0:
                continue
            size = colors.tile_font_size(value, cell_px=tile_px)
            font = number_font_cache.get(size)
            if font is None:
                font = _font(size, bold=True)
                number_font_cache[size] = font
            text = str(value)
            bbox = draw.textbbox((0, 0), text, font=font)
            tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
            tx = x0 + (tile_px - tw) / 2 - bbox[0]
            ty = y0 + (tile_px - th) / 2 - bbox[1]
            draw.text((tx, ty), text, fill=colors.tile_fg(value), font=font)
    return image


def write_gif(
    boards: list[np.ndarray],
    path: Path | str,
    *,
    caption: str | None = None,
    stride: int = 1,
    duration_ms: int = 90,
    last_hold_ms: int = 1200,
    tile_px: int = 64,
) -> Path:
    """Save a looping GIF from a sequence of boards. ``stride`` keeps file size down."""
    if not boards:
        raise ValueError("Need at least one board to write a GIF.")
    sampled = list(boards[:: max(int(stride), 1)])
    if sampled[-1] is not boards[-1]:
        sampled.append(boards[-1])
    frames = [render_board(board, tile_px=tile_px, caption=caption) for board in sampled]
    durations = [duration_ms] * len(frames)
    durations[-1] = last_hold_ms
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    return out
