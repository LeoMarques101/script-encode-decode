"""Multi-level color encoding and decoding for the datagrid.

Each cell stores data across 3 RGB channels. With N color levels per channel,
each channel stores log2(N) bits, so each cell stores 3*log2(N) bits total.
"""

from __future__ import annotations

import numpy as np

from .config import COLOR_LEVEL_BITS, COLOR_LEVEL_VALUES


def get_color_values(color_levels: int) -> np.ndarray:
    """Return the discrete color values for the given number of levels."""
    return np.array(COLOR_LEVEL_VALUES[color_levels], dtype=np.uint8)


def get_thresholds(color_levels: int) -> np.ndarray:
    """Return decision thresholds (midpoints between adjacent levels)."""
    values = COLOR_LEVEL_VALUES[color_levels]
    thresholds = []
    for i in range(len(values) - 1):
        thresholds.append((values[i] + values[i + 1]) // 2)
    return np.array(thresholds, dtype=np.uint8)


def encode_byte_to_cells(data: bytes, color_levels: int) -> np.ndarray:
    """Encode raw bytes into cell color values (N_cells, 3) as uint8.

    Each cell stores `bits_per_cell` bits across 3 RGB channels.
    Returns an array of shape (num_cells, 3) with color values.
    """
    bits_per_channel = COLOR_LEVEL_BITS[color_levels]
    bits_per_cell = bits_per_channel * 3
    values = get_color_values(color_levels)

    # Convert bytes to a bit stream
    data_arr = np.frombuffer(data, dtype=np.uint8)
    bits = np.unpackbits(data_arr)

    # Pad to a multiple of bits_per_cell
    total_bits = len(bits)
    remainder = total_bits % bits_per_cell
    if remainder != 0:
        bits = np.concatenate([bits, np.zeros(bits_per_cell - remainder, dtype=np.uint8)])

    num_cells = len(bits) // bits_per_cell

    # Reshape bits into (num_cells, 3, bits_per_channel)
    bits_reshaped = bits[:num_cells * bits_per_cell].reshape(num_cells, 3, bits_per_channel)

    # Convert groups of bits to level indices using positional weighting
    # weights = [2^(bpc-1), ..., 2, 1]  (MSB first)
    weights = (1 << np.arange(bits_per_channel - 1, -1, -1)).astype(np.uint8)
    indices = (bits_reshaped * weights).sum(axis=2).astype(np.intp)  # (num_cells, 3)

    cells = values[indices]  # fancy-index into the level LUT
    return cells


def decode_cells_to_bytes(cells: np.ndarray, color_levels: int, num_bytes: int) -> bytes:
    """Decode cell color values back to raw bytes (vectorized).

    Args:
        cells: (N_cells, 3) array of uint8 color values (possibly noisy).
        color_levels: Number of discrete levels per channel.
        num_bytes: Expected number of output bytes.

    Returns:
        Decoded bytes.
    """
    bits_per_channel = COLOR_LEVEL_BITS[color_levels]
    thresholds = get_thresholds(color_levels)

    # Quantize all channels at once: searchsorted on flattened values
    flat = cells.ravel().astype(np.uint8)  # (N_cells * 3,)
    indices = np.searchsorted(thresholds, flat).astype(np.uint8)  # level index per value

    # Reshape to (N_cells, 3)
    indices = indices.reshape(-1, 3)

    # Convert each index to bits_per_channel bits (MSB first) — vectorized
    # Build a (bits_per_channel,) array of bit positions
    shifts = np.arange(bits_per_channel - 1, -1, -1, dtype=np.uint8)
    # indices shape (N, 3) → expand to (N, 3, bpc)
    bits_expanded = ((indices[..., np.newaxis] >> shifts) & 1).astype(np.uint8)
    # bits_expanded shape: (N_cells, 3, bits_per_channel)

    # Flatten to a 1-D bit stream: cell-major, then channel, then bit
    all_bits = bits_expanded.reshape(-1)

    # Take only the bits we need and pack
    needed = num_bytes * 8
    if len(all_bits) >= needed:
        bits_arr = all_bits[:needed]
    else:
        bits_arr = np.zeros(needed, dtype=np.uint8)
        bits_arr[:len(all_bits)] = all_bits

    byte_arr = np.packbits(bits_arr)
    return bytes(byte_arr[:num_bytes])


def cells_needed(num_bytes: int, color_levels: int) -> int:
    """How many cells are needed to store num_bytes."""
    bits_per_cell = COLOR_LEVEL_BITS[color_levels] * 3
    total_bits = num_bytes * 8
    return (total_bits + bits_per_cell - 1) // bits_per_cell
