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

    row_start = FIDUCIAL_SIZE_CELLS + METADATA_ROWS
    col_start = FIDUCIAL_SIZE_CELLS

    n_cells = min(len(cells), config.data_rows * config.data_cols)
    if n_cells == 0:
        return

    # Reshape cells into a 2D block and assign in one go
    full_rows = n_cells // config.data_cols
    remainder = n_cells % config.data_cols

    if full_rows > 0:
        block = cells[:full_rows * config.data_cols].reshape(
            full_rows, config.data_cols, 3
        )
        grid[row_start:row_start + full_rows,
             col_start:col_start + config.data_cols] = block

    if remainder > 0:
        grid[row_start + full_rows,
             col_start:col_start + remainder] = cells[full_rows * config.data_cols:n_cells]


def read_data_from_grid(
    grid: np.ndarray,
    config: GridConfig,
    num_bytes: int,
) -> bytes:
    """Read data bytes from the data area of the cell grid (vectorized).

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

    # Extract the full data sub-grid and flatten to (N, 3)
    data_block = grid[row_start:row_start + config.data_rows,
                      col_start:col_start + config.data_cols]  # (data_rows, data_cols, 3)
    flat = data_block.reshape(-1, 3)  # row-major = correct cell order

    # Truncate or pad to n_cells
    if flat.shape[0] >= n_cells:
        cells = flat[:n_cells]
    else:
        cells = np.zeros((n_cells, 3), dtype=np.uint8)
        cells[:flat.shape[0]] = flat

    return decode_cells_to_bytes(cells, config.color_levels, num_bytes)


def grid_to_image(grid: np.ndarray, config: GridConfig) -> np.ndarray:
    """Convert a cell grid to a full-resolution image.

    Uses np.kron for fast upscaling instead of a Python loop.

    Args:
        grid: (grid_rows, grid_cols, 3) uint8 cell grid.
        config: Grid configuration.

    Returns:
        (height, width, 3) uint8 image (BGR for OpenCV).
    """
    # grid stores RGB; OpenCV uses BGR — flip channels
    grid_bgr = grid[:, :, ::-1]  # (grid_rows, grid_cols, 3)

    # Upscale each cell to cell_size x cell_size pixels via np.repeat
    cs = config.cell_size
    upscaled = np.repeat(np.repeat(grid_bgr, cs, axis=0), cs, axis=1)
    # upscaled shape: (grid_rows*cs, grid_cols*cs, 3)

    # Place into full-size canvas at (margin, margin)
    img = np.zeros((config.height, config.width, 3), dtype=np.uint8)
    h = min(upscaled.shape[0], config.height - config.margin)
    w = min(upscaled.shape[1], config.width - config.margin)
    img[config.margin:config.margin + h,
        config.margin:config.margin + w] = upscaled[:h, :w]

    return img


def image_to_grid(
    img: np.ndarray,
    config: GridConfig,
) -> np.ndarray:
    """Sample cell colors from a full-resolution image (vectorized).

    Samples from the center of each cell for robustness.

    Args:
        img: (height, width, 3) uint8 image (BGR).
        config: Grid configuration.

    Returns:
        (grid_rows, grid_cols, 3) uint8 cell grid (RGB).
    """
    half = config.cell_size // 2

    # Build arrays of center-pixel coordinates
    ys = np.arange(config.grid_rows) * config.cell_size + config.margin + half
    xs = np.arange(config.grid_cols) * config.cell_size + config.margin + half

    # Clip to image bounds
    ys = np.clip(ys, 0, img.shape[0] - 1).astype(np.intp)
    xs = np.clip(xs, 0, img.shape[1] - 1).astype(np.intp)

    # Fancy-index: img[ys, xs] via meshgrid (row, col order)
    grid_bgr = img[np.ix_(ys, xs)]  # shape (grid_rows, grid_cols, 3)

    # BGR to RGB
    return grid_bgr[:, :, ::-1].copy()
