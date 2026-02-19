"""Frame composition: assemble complete frames with fiducials, metadata, and datagrid.

Each frame has:
  - Black margin border
  - 4 asymmetric fiducial markers in corners (for detection/orientation)
  - Metadata rows (top and bottom)
  - Data grid (payload area)
"""

from __future__ import annotations

import numpy as np

from .config import GridConfig, FrameType, FIDUCIAL_SIZE_CELLS, METADATA_ROWS
from .color import encode_byte_to_cells
from .datagrid import paint_data_on_grid, grid_to_image


# ── Fiducial marker patterns ──
# Each fiducial is FIDUCIAL_SIZE_CELLS × FIDUCIAL_SIZE_CELLS.
# They are asymmetric so the decoder can determine orientation.
# Pattern values: 0 = black (0,0,0), 1 = white (255,255,255)
# Each corner has a unique pattern.

def _make_fiducial_tl(n: int) -> np.ndarray:
    """Top-left fiducial: full border, inner filled square."""
    pat = np.zeros((n, n), dtype=np.uint8)
    # Outer ring: white
    pat[:, :] = 0
    pat[0, :] = 1; pat[-1, :] = 1; pat[:, 0] = 1; pat[:, -1] = 1
    # Inner area: white block at top-left inner
    pat[1:n-1, 1:n-1] = 1
    # Center: black
    c = n // 2
    pat[c-1:c+1, c-1:c+1] = 0
    # Mark: top-left has extra black cell at (1,1)
    pat[1, 1] = 0
    return pat


def _make_fiducial_tr(n: int) -> np.ndarray:
    """Top-right fiducial: full border, inner with top-right mark."""
    pat = np.zeros((n, n), dtype=np.uint8)
    pat[0, :] = 1; pat[-1, :] = 1; pat[:, 0] = 1; pat[:, -1] = 1
    pat[1:n-1, 1:n-1] = 1
    c = n // 2
    pat[c-1:c+1, c-1:c+1] = 0
    # Mark: top-right corner has black cell at (1, n-2)
    pat[1, n-2] = 0
    pat[2, n-2] = 0  # Extra mark for asymmetry
    return pat


def _make_fiducial_bl(n: int) -> np.ndarray:
    """Bottom-left fiducial: full border, inner with bottom-left mark."""
    pat = np.zeros((n, n), dtype=np.uint8)
    pat[0, :] = 1; pat[-1, :] = 1; pat[:, 0] = 1; pat[:, -1] = 1
    pat[1:n-1, 1:n-1] = 1
    c = n // 2
    pat[c-1:c+1, c-1:c+1] = 0
    # Mark: bottom-left has black cells at (n-2, 1) and (n-3, 1)
    pat[n-2, 1] = 0
    pat[n-3, 1] = 0
    pat[n-2, 2] = 0
    return pat


def _make_fiducial_br(n: int) -> np.ndarray:
    """Bottom-right fiducial: full border, inner with bottom-right mark."""
    pat = np.zeros((n, n), dtype=np.uint8)
    pat[0, :] = 1; pat[-1, :] = 1; pat[:, 0] = 1; pat[:, -1] = 1
    pat[1:n-1, 1:n-1] = 1
    c = n // 2
    pat[c-1:c+1, c-1:c+1] = 0
    # Mark: bottom-right has black L-shape at (n-2, n-2), (n-3, n-2), (n-2, n-3)
    pat[n-2, n-2] = 0
    pat[n-3, n-2] = 0
    pat[n-2, n-3] = 0
    pat[n-3, n-3] = 0
    return pat


def get_fiducial_patterns(n: int = FIDUCIAL_SIZE_CELLS) -> dict[str, np.ndarray]:
    """Return the 4 fiducial patterns as binary arrays (0/1)."""
    return {
        "tl": _make_fiducial_tl(n),
        "tr": _make_fiducial_tr(n),
        "bl": _make_fiducial_bl(n),
        "br": _make_fiducial_br(n),
    }


def _paint_fiducial(grid: np.ndarray, pattern: np.ndarray,
                    row: int, col: int) -> None:
    """Paint a fiducial pattern onto the cell grid."""
    n = pattern.shape[0]
    for r in range(n):
        for c in range(n):
            color = [255, 255, 255] if pattern[r, c] else [0, 0, 0]
            grid[row + r, col + c] = color


def _paint_metadata_row(
    grid: np.ndarray,
    row_start: int,
    col_start: int,
    data: bytes,
    config: GridConfig,
    num_cols: int,
) -> None:
    """Paint metadata bytes into a row of cells."""
    cells = encode_byte_to_cells(data, config.color_levels)
    for i, c in enumerate(cells):
        if i >= num_cols:
            break
        grid[row_start, col_start + i] = c


def compose_frame(
    packet_data: bytes,
    config: GridConfig,
    frame_type: FrameType = FrameType.DATA,
) -> np.ndarray:
    """Compose a complete frame image from packet data.

    Args:
        packet_data: Raw bytes for the data area (with ECC, sized to raw_bytes_per_frame).
        config: Grid configuration.
        frame_type: Type of this frame.

    Returns:
        (height, width, 3) uint8 BGR image.
    """
    # Initialize cell grid (all black = margin effectively)
    grid = np.zeros((config.grid_rows, config.grid_cols, 3), dtype=np.uint8)

    n = FIDUCIAL_SIZE_CELLS
    patterns = get_fiducial_patterns(n)

    # Paint fiducials in the 4 corners
    _paint_fiducial(grid, patterns["tl"], 0, 0)
    _paint_fiducial(grid, patterns["tr"], 0, config.grid_cols - n)
    _paint_fiducial(grid, patterns["bl"], config.grid_rows - n, 0)
    _paint_fiducial(grid, patterns["br"], config.grid_rows - n, config.grid_cols - n)

    # Paint data onto the grid
    if frame_type == FrameType.SYNC:
        _paint_sync_pattern(grid, config)
    else:
        paint_data_on_grid(grid, packet_data, config)

    # Convert cell grid to full image
    return grid_to_image(grid, config)


def compose_sync_frame(config: GridConfig) -> np.ndarray:
    """Compose a sync frame with a checkerboard pattern in the data area."""
    return compose_frame(b"", config, frame_type=FrameType.SYNC)


def _paint_sync_pattern(grid: np.ndarray, config: GridConfig) -> None:
    """Paint a checkerboard pattern in the data area for synchronization."""
    row_start = FIDUCIAL_SIZE_CELLS + METADATA_ROWS
    col_start = FIDUCIAL_SIZE_CELLS

    for r in range(config.data_rows):
        for c in range(config.data_cols):
            if (r + c) % 2 == 0:
                grid[row_start + r, col_start + c] = [255, 255, 255]
            else:
                grid[row_start + r, col_start + c] = [0, 0, 0]
