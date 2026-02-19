"""Datagrid generation and reading.

Handles painting data bytes onto a cell grid and reading them back.
The datagrid is the rectangular area of cells that carry payload data,
excluding fiducial markers and metadata rows.
"""

from __future__ import annotations

import numpy as np

from .color import encode_byte_to_cells, decode_cells_to_bytes
from .config import GridConfig, FIDUCIAL_SIZE_CELLS, METADATA_ROWS


def paint_data_on_grid(
    grid: np.ndarray,
    data: bytes,
    config: GridConfig,
) -> None:
    """Paint data bytes onto the data area of the cell grid.

    Args:
        grid: (grid_rows, grid_cols, 3) uint8 array — the full cell grid.
               Modified in-place.
        data: Raw bytes to paint (already includes ECC).
        config: Grid configuration.
    """
    cells = encode_byte_to_cells(data, config.color_levels)

    # Data area starts after top fiducials + metadata rows
    row_start = FIDUCIAL_SIZE_CELLS + METADATA_ROWS
    col_start = FIDUCIAL_SIZE_CELLS

    idx = 0
    for r in range(config.data_rows):
        for c in range(config.data_cols):
            if idx < len(cells):
                grid[row_start + r, col_start + c] = cells[idx]
                idx += 1


def read_data_from_grid(
    grid: np.ndarray,
    config: GridConfig,
    num_bytes: int,
) -> bytes:
    """Read data bytes from the data area of the cell grid.

    Args:
        grid: (grid_rows, grid_cols, 3) uint8 array — the full cell grid.
        config: Grid configuration.
        num_bytes: Expected number of bytes to read.

    Returns:
        Decoded bytes from the grid.
    """
    row_start = FIDUCIAL_SIZE_CELLS + METADATA_ROWS
    col_start = FIDUCIAL_SIZE_CELLS

    from .color import cells_needed
    n_cells = cells_needed(num_bytes, config.color_levels)

    cells = np.zeros((n_cells, 3), dtype=np.uint8)
    idx = 0
    for r in range(config.data_rows):
        for c in range(config.data_cols):
            if idx < n_cells:
                cells[idx] = grid[row_start + r, col_start + c]
                idx += 1

    return decode_cells_to_bytes(cells, config.color_levels, num_bytes)


def grid_to_image(grid: np.ndarray, config: GridConfig) -> np.ndarray:
    """Convert a cell grid to a full-resolution image.

    Args:
        grid: (grid_rows, grid_cols, 3) uint8 cell grid.
        config: Grid configuration.

    Returns:
        (height, width, 3) uint8 image (BGR for OpenCV).
    """
    img = np.zeros((config.height, config.width, 3), dtype=np.uint8)

    # Grid starts at (margin, margin)
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            y = config.margin + r * config.cell_size
            x = config.margin + c * config.cell_size
            # Paint the cell as a solid color block
            y_end = min(y + config.cell_size, config.height)
            x_end = min(x + config.cell_size, config.width)
            # grid stores RGB, OpenCV uses BGR
            img[y:y_end, x:x_end] = grid[r, c, ::-1]

    return img


def image_to_grid(
    img: np.ndarray,
    config: GridConfig,
) -> np.ndarray:
    """Sample cell colors from a full-resolution image.

    Samples from the center of each cell for robustness.

    Args:
        img: (height, width, 3) uint8 image (BGR).
        config: Grid configuration.

    Returns:
        (grid_rows, grid_cols, 3) uint8 cell grid (RGB).
    """
    grid = np.zeros((config.grid_rows, config.grid_cols, 3), dtype=np.uint8)
    half = config.cell_size // 2

    for r in range(config.grid_rows):
        for c in range(config.grid_cols):
            cy = config.margin + r * config.cell_size + half
            cx = config.margin + c * config.cell_size + half
            cy = min(cy, img.shape[0] - 1)
            cx = min(cx, img.shape[1] - 1)
            # BGR to RGB
            grid[r, c] = img[cy, cx, ::-1]

    return grid
