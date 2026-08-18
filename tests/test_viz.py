"""Visualizer helpers that do not require a Tk window."""

import pytest

from rl2048.viz.colors import tile_font_size
from rl2048.viz.layout import HARD_MIN_TILE_PX, board_geometry, board_outer_size


def test_tile_font_shrinks_for_four_digit_values():
    assert tile_font_size(2) == tile_font_size(4) == tile_font_size(64)
    assert tile_font_size(128) < tile_font_size(2)
    assert tile_font_size(1024) < tile_font_size(128)
    assert tile_font_size(1024) == tile_font_size(2048)
    assert tile_font_size(16384) < tile_font_size(1024)


def test_tile_font_scales_with_cell_size():
    assert tile_font_size(2, cell_px=200) > tile_font_size(2, cell_px=100)
    assert tile_font_size(1024, cell_px=200) > tile_font_size(1024, cell_px=100)


def test_board_grows_with_host_and_stays_square():
    small_tile, _ = board_geometry(400, 400)
    large_tile, _ = board_geometry(900, 900)
    assert large_tile > small_tile
    wide_tile, _ = board_geometry(1600, 500)
    tall_tile, _ = board_geometry(500, 1600)
    assert wide_tile == tall_tile


def test_board_fits_inside_host():
    for width, height in ((320, 240), (800, 600), (1600, 900), (1920, 1080)):
        tile, pad = board_geometry(width, height)
        assert tile >= HARD_MIN_TILE_PX
        assert board_outer_size(tile, pad) <= min(width, height)


def test_render_board_is_square_grid():
    pytest.importorskip("PIL")
    import numpy as np

    from rl2048.viz.board_image import HEADER_PX, render_board

    board = np.array([[2, 4, 8, 16], [32, 64, 128, 256], [512, 1024, 2048, 0], [0, 0, 0, 2]])
    image = render_board(board, tile_px=48, caption="test")
    assert image.size[0] == image.size[1] - HEADER_PX
    assert image.size[0] > 100
