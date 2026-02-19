"""Configuration and capacity calculations for Video Steganography."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Tuple

PROTOCOL_VERSION = 1

FIDUCIAL_SIZE_CELLS = 6  # each fiducial marker is 6x6 cells
METADATA_ROWS = 2  # top and bottom metadata rows (in cells)


class FrameType(Enum):
    START_MARKER = 0
    DATA = 1
    END_MARKER = 2
    SYNC = 3


class ECCLevel(Enum):
    LOW = "low"        # 10%
    MEDIUM = "medium"  # 25%
    HIGH = "high"      # 50%

    @property
    def ratio(self) -> float:
        return {
            ECCLevel.LOW: 0.10,
            ECCLevel.MEDIUM: 0.25,
            ECCLevel.HIGH: 0.50,
        }[self]


class Compression(Enum):
    NONE = "none"
    ZLIB = "zlib"
    LZ4 = "lz4"


COLOR_LEVEL_VALUES = {
    2: [0, 255],
    4: [0, 85, 170, 255],
    8: [0, 36, 73, 109, 146, 182, 219, 255],
}

COLOR_LEVEL_BITS = {
    2: 1,
    4: 2,
    8: 3,
}


@dataclass
class GridConfig:
    """Configuration for the video steganography system."""

    resolution: Tuple[int, int] = (1920, 1080)
    fps: int = 30
    cell_size: int = 8
    margin: int = 40
    color_levels: int = 2
    ecc_level: ECCLevel = ECCLevel.MEDIUM
    redundancy: int = 2
    codec: str = "h264"
    crf: int = 0
    compression: Compression = Compression.ZLIB

    # Sync frame interval (every N data frames)
    sync_interval: int = 50

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    @property
    def usable_width(self) -> int:
        """Pixel width available for the grid (inside margins)."""
        return self.width - 2 * self.margin

    @property
    def usable_height(self) -> int:
        """Pixel height available for the grid (inside margins)."""
        return self.height - 2 * self.margin

    @property
    def grid_cols(self) -> int:
        """Total columns in the cell grid (including fiducials and metadata)."""
        return self.usable_width // self.cell_size

    @property
    def grid_rows(self) -> int:
        """Total rows in the cell grid (including fiducials and metadata)."""
        return self.usable_height // self.cell_size

    @property
    def data_cols(self) -> int:
        """Columns available for data (excluding fiducial columns on left and right)."""
        return self.grid_cols - 2 * FIDUCIAL_SIZE_CELLS

    @property
    def data_rows(self) -> int:
        """Rows available for data (excluding fiducial rows top/bottom and metadata rows)."""
        return self.grid_rows - 2 * FIDUCIAL_SIZE_CELLS - 2 * METADATA_ROWS

    @property
    def data_cells(self) -> int:
        """Total number of cells available for data per frame."""
        return self.data_cols * self.data_rows

    @property
    def bits_per_cell(self) -> int:
        """Bits stored per cell based on color levels."""
        return COLOR_LEVEL_BITS[self.color_levels] * 3  # 3 channels (RGB)

    @property
    def raw_bytes_per_frame(self) -> int:
        """Total data bytes that fit in one frame (before ECC overhead)."""
        total_bits = self.data_cells * self.bits_per_cell
        return total_bits // 8

    @property
    def ecc_overhead_bytes(self) -> int:
        """ECC overhead bytes per frame."""
        return int(math.ceil(self.raw_bytes_per_frame * self.ecc_level.ratio))

    @property
    def payload_bytes_per_frame(self) -> int:
        """Usable payload bytes per frame after ECC overhead."""
        return self.raw_bytes_per_frame - self.ecc_overhead_bytes

    @property
    def metadata_cells(self) -> int:
        """Cells in the metadata rows (top + bottom)."""
        return self.data_cols * METADATA_ROWS * 2  # top and bottom

    def packets_needed(self, data_size: int) -> int:
        """How many packets are needed for the given data size."""
        if self.payload_bytes_per_frame <= 0:
            raise ValueError("Grid too small: no payload capacity per frame.")
        return math.ceil(data_size / self.payload_bytes_per_frame)

    def total_data_frames(self, data_size: int) -> int:
        """Total data frames including redundancy."""
        return self.packets_needed(data_size) * self.redundancy

    def total_sync_frames(self, data_size: int) -> int:
        """Number of sync frames to insert."""
        n_data = self.total_data_frames(data_size)
        if self.sync_interval <= 0:
            return 0
        return max(0, (n_data - 1) // self.sync_interval)

    def total_control_frames(self, data_size: int) -> int:
        """Start (×3) + End (×3) + sync frames."""
        return 3 + 3 + self.total_sync_frames(data_size)

    def total_frames(self, data_size: int) -> int:
        """Total frames in the output video."""
        return self.total_data_frames(data_size) + self.total_control_frames(data_size)

    def video_duration(self, data_size: int) -> float:
        """Duration of the output video in seconds."""
        return self.total_frames(data_size) / self.fps

    def summary(self, data_size: int, compressed_size: int | None = None) -> str:
        """Return a summary string of the encoding configuration."""
        pkt = self.packets_needed(compressed_size or data_size)
        total = self.total_frames(compressed_size or data_size)
        dur = self.video_duration(compressed_size or data_size)
        data_fr = self.total_data_frames(compressed_size or data_size)
        ctrl = self.total_control_frames(compressed_size or data_size)

        lines = [
            f"  Grid: {self.grid_cols}x{self.grid_rows} cells "
            f"(cell={self.cell_size}px, margin={self.margin}px)",
            f"  Data area: {self.data_cols}x{self.data_rows} cells "
            f"({self.data_cells} cells)",
            f"  Color: {self.color_levels} levels/channel "
            f"({self.bits_per_cell} bits/cell)",
            f"  Capacity: {self.raw_bytes_per_frame:,} bytes/frame",
            f"  ECC: {self.ecc_level.value} ({int(self.ecc_level.ratio * 100)}%) "
            f"-> {self.payload_bytes_per_frame:,} payload bytes/frame",
            f"  Packets: {pkt} | Frames: {data_fr} "
            f"(w/ redundancy x{self.redundancy})",
            f"  + {ctrl} control frames (start/sync/end)",
            f"  Total frames: {total} | Duration: {dur:.1f}s",
        ]
        if compressed_size is not None and compressed_size != data_size:
            ratio = compressed_size / data_size if data_size > 0 else 0
            lines.insert(0,
                f"  Compression: {self.compression.value} "
                f"(ratio: {ratio:.2f})")
        return "\n".join(lines)

    def validate(self) -> None:
        """Raise ValueError if the configuration is invalid."""
        if self.cell_size < 2:
            raise ValueError("cell_size must be >= 2")
        if self.color_levels not in COLOR_LEVEL_VALUES:
            raise ValueError(f"color_levels must be one of {list(COLOR_LEVEL_VALUES)}")
        if self.margin < 0:
            raise ValueError("margin must be >= 0")
        if self.redundancy < 1:
            raise ValueError("redundancy must be >= 1")
        if self.data_cols <= 0 or self.data_rows <= 0:
            raise ValueError(
                f"Grid too small: data area is {self.data_cols}x{self.data_rows}. "
                "Increase resolution, decrease cell_size, or decrease margin."
            )
        if self.fps <= 0:
            raise ValueError("fps must be > 0")
