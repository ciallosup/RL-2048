"""Square 4x4 board sizing for the visualizer."""

BOARD_N = 4
MIN_TILE_PX = 64
HARD_MIN_TILE_PX = 32
BOARD_FRAME_PAD = 8


def cell_pad(tile_px: int) -> int:
    return max(3, int(tile_px) // 18)


def board_outer_size(tile_px: int, pad: int | None = None) -> int:
    """Pixel size of the square board including frame padding and gaps."""
    tile = max(1, int(tile_px))
    p = cell_pad(tile) if pad is None else int(pad)
    return 2 * BOARD_FRAME_PAD + BOARD_N * tile + 2 * BOARD_N * p


def board_geometry(host_w: int, host_h: int) -> tuple[int, int]:
    """Return ``(tile_px, cell_pad)`` so a square 4x4 board fits in the host."""
    limit = min(max(int(host_w), 0), max(int(host_h), 0))
    if limit <= 0:
        return MIN_TILE_PX, cell_pad(MIN_TILE_PX)

    lo, hi = HARD_MIN_TILE_PX, max(HARD_MIN_TILE_PX, limit)
    best = HARD_MIN_TILE_PX
    while lo <= hi:
        mid = (lo + hi) // 2
        if board_outer_size(mid) <= limit:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best, cell_pad(best)
