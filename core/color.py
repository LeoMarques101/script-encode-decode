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
    # Unpack each byte into 8 bits (MSB first)
    bits = np.unpackbits(data_arr)

    # Pad to a multiple of bits_per_cell
    total_bits = len(bits)
    remainder = total_bits % bits_per_cell
    if remainder != 0:
        bits = np.concatenate([bits, np.zeros(bits_per_cell - remainder, dtype=np.uint8)])

    num_cells = len(bits) // bits_per_cell
    cells = np.zeros((num_cells, 3), dtype=np.uint8)

    for i in range(num_cells):
        base = i * bits_per_cell
        for ch in range(3):
            ch_bits = bits[base + ch * bits_per_channel: base + (ch + 1) * bits_per_channel]
            # Convert bits to an integer index
            level_idx = 0
            for b in ch_bits:
                level_idx = (level_idx << 1) | int(b)
            cells[i, ch] = values[level_idx]

    return cells


def decode_cells_to_bytes(cells: np.ndarray, color_levels: int, num_bytes: int) -> bytes:
    """Decode cell color values back to raw bytes.

    Args:
        cells: (N_cells, 3) array of uint8 color values (possibly noisy).
        color_levels: Number of discrete levels per channel.
        num_bytes: Expected number of output bytes.

    Returns:
        Decoded bytes.
    """
    bits_per_channel = COLOR_LEVEL_BITS[color_levels]
    bits_per_cell = bits_per_channel * 3
    values = get_color_values(color_levels)
    thresholds = get_thresholds(color_levels)

    bits = []
    for i in range(cells.shape[0]):
        for ch in range(3):
            pixel_val = int(cells[i, ch])
            # Quantize to nearest level
            level_idx = int(np.searchsorted(thresholds, pixel_val))
            # Convert index to bits (MSB first)
            for bit_pos in range(bits_per_channel - 1, -1, -1):
                bits.append((level_idx >> bit_pos) & 1)

    # Convert bits to bytes
    bits_arr = np.array(bits[:num_bytes * 8], dtype=np.uint8)
    # Pad if necessary
    if len(bits_arr) < num_bytes * 8:
        bits_arr = np.concatenate([
            bits_arr,
            np.zeros(num_bytes * 8 - len(bits_arr), dtype=np.uint8)
        ])

    byte_arr = np.packbits(bits_arr)
    return bytes(byte_arr[:num_bytes])


def cells_needed(num_bytes: int, color_levels: int) -> int:
    """How many cells are needed to store num_bytes."""
    bits_per_cell = COLOR_LEVEL_BITS[color_levels] * 3
    total_bits = num_bytes * 8
    return (total_bits + bits_per_cell - 1) // bits_per_cell
