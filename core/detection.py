"""Fiducial detection and perspective correction for the decoder.

Detects the 4 corner fiducial markers in a (possibly degraded) frame,
computes a homography to correct perspective/rotation/scale, then
samples cell colors from the corrected image.
"""

from __future__ import annotations

import cv2
import numpy as np
from typing import Optional

from .config import GridConfig, FIDUCIAL_SIZE_CELLS
from .framing import get_fiducial_patterns


def _build_fiducial_template(pattern: np.ndarray, cell_size: int) -> np.ndarray:
    """Build a grayscale template image from a fiducial pattern."""
    n = pattern.shape[0]
    size = n * cell_size
    tmpl = np.zeros((size, size), dtype=np.uint8)
    for r in range(n):
        for c in range(n):
            val = 255 if pattern[r, c] else 0
            y0 = r * cell_size
            x0 = c * cell_size
            tmpl[y0:y0 + cell_size, x0:x0 + cell_size] = val
    return tmpl


def detect_fiducials(
    frame_gray: np.ndarray,
    config: GridConfig,
    search_margin: int = 100,
) -> Optional[dict[str, tuple[float, float]]]:
    """Detect the 4 fiducial markers in a grayscale frame.

    Uses template matching with multi-scale search.

    Args:
        frame_gray: Grayscale uint8 image.
        config: Grid configuration.
        search_margin: Extra pixels to search beyond expected positions.

    Returns:
        Dict with keys 'tl','tr','bl','br' mapping to (x, y) center coordinates,
        or None if detection fails.
    """
    h, w = frame_gray.shape[:2]
    n = FIDUCIAL_SIZE_CELLS
    patterns = get_fiducial_patterns(n)

    fid_pixel_size = n * config.cell_size

    # Expected positions (center of each fiducial in the ideal frame)
    expected = {
        "tl": (config.margin + fid_pixel_size // 2,
               config.margin + fid_pixel_size // 2),
        "tr": (config.width - config.margin - fid_pixel_size // 2,
               config.margin + fid_pixel_size // 2),
        "bl": (config.margin + fid_pixel_size // 2,
               config.height - config.margin - fid_pixel_size // 2),
        "br": (config.width - config.margin - fid_pixel_size // 2,
               config.height - config.margin - fid_pixel_size // 2),
    }

    # Scale factors to try (handles resolution mismatch)
    scale_x = w / config.width
    scale_y = h / config.height

    results = {}

    for corner, pattern in patterns.items():
        tmpl = _build_fiducial_template(pattern, config.cell_size)

        # Scale template if needed
        if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
            new_w = max(1, int(tmpl.shape[1] * scale_x))
            new_h = max(1, int(tmpl.shape[0] * scale_y))
            tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Define search region around expected position
        ex, ey = expected[corner]
        ex_scaled = int(ex * scale_x)
        ey_scaled = int(ey * scale_y)
        tmpl_h, tmpl_w = tmpl.shape[:2]
        margin_search = int(search_margin * max(scale_x, scale_y))

        x1 = max(0, ex_scaled - tmpl_w // 2 - margin_search)
        y1 = max(0, ey_scaled - tmpl_h // 2 - margin_search)
        x2 = min(w, ex_scaled + tmpl_w // 2 + margin_search)
        y2 = min(h, ey_scaled + tmpl_h // 2 + margin_search)

        roi = frame_gray[y1:y2, x1:x2]

        if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
            # ROI too small, try full image
            roi = frame_gray
            x1, y1 = 0, 0

        # Template matching
        res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)

        if max_val < 0.3:
            return None  # Fiducial not found with sufficient confidence

        # Convert to full image coordinates (center of template match)
        cx = x1 + max_loc[0] + tmpl_w // 2
        cy = y1 + max_loc[1] + tmpl_h // 2

        results[corner] = (float(cx), float(cy))

    return results


def compute_homography(
    detected: dict[str, tuple[float, float]],
    config: GridConfig,
) -> Optional[np.ndarray]:
    """Compute homography from detected fiducial centers to ideal positions.

    Returns:
        3x3 homography matrix, or None if computation fails.
    """
    n = FIDUCIAL_SIZE_CELLS
    fid_half = (n * config.cell_size) // 2

    # Ideal corner positions (center of each fiducial in the target frame)
    ideal = {
        "tl": (config.margin + fid_half, config.margin + fid_half),
        "tr": (config.width - config.margin - fid_half,
               config.margin + fid_half),
        "bl": (config.margin + fid_half,
               config.height - config.margin - fid_half),
        "br": (config.width - config.margin - fid_half,
               config.height - config.margin - fid_half),
    }

    src_pts = np.float32([
        detected["tl"], detected["tr"], detected["br"], detected["bl"]
    ])
    dst_pts = np.float32([
        ideal["tl"], ideal["tr"], ideal["br"], ideal["bl"]
    ])

    H, status = cv2.findHomography(src_pts, dst_pts)
    if H is None:
        return None
    return H


def correct_perspective(
    frame: np.ndarray,
    H: np.ndarray,
    config: GridConfig,
) -> np.ndarray:
    """Apply homography to warp the frame to the ideal grid layout.

    Args:
        frame: Input BGR image.
        H: 3x3 homography matrix.
        config: Grid configuration.

    Returns:
        Warped BGR image of size (config.height, config.width).
    """
    return cv2.warpPerspective(
        frame, H, (config.width, config.height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def sample_grid_from_frame(
    frame_bgr: np.ndarray,
    config: GridConfig,
) -> np.ndarray:
    """Sample cell colors from a corrected frame (vectorized).

    Samples a 3x3 patch around the center of each cell and averages for noise
    robustness, using NumPy vectorized operations.

    Args:
        frame_bgr: Corrected BGR image of size (height, width, 3).
        config: Grid configuration.

    Returns:
        (grid_rows, grid_cols, 3) uint8 array in RGB.
    """
    half = config.cell_size // 2
    h, w = frame_bgr.shape[:2]

    # Center coordinates for each cell
    ys = np.arange(config.grid_rows) * config.cell_size + config.margin + half
    xs = np.arange(config.grid_cols) * config.cell_size + config.margin + half

    # Clamp to valid pixel range (leave room for ±1 patch)
    ys = np.clip(ys, 1, h - 2).astype(np.intp)
    xs = np.clip(xs, 1, w - 2).astype(np.intp)

    # Build meshgrid of center positions
    yy, xx = np.meshgrid(ys, xs, indexing='ij')  # (grid_rows, grid_cols)

    # Accumulate 3x3 patch for averaging (float32 to avoid overflow)
    acc = np.zeros((config.grid_rows, config.grid_cols, 3), dtype=np.float32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            acc += frame_bgr[yy + dy, xx + dx].astype(np.float32)
    avg_bgr = (acc / 9.0).astype(np.uint8)

    # BGR to RGB
    return avg_bgr[:, :, ::-1].copy()


def is_sync_frame(grid: np.ndarray, config: GridConfig, threshold: float = 0.7) -> bool:
    """Check if a grid looks like a sync frame (checkerboard pattern)."""
    from .config import FIDUCIAL_SIZE_CELLS, METADATA_ROWS

    row_start = FIDUCIAL_SIZE_CELLS + METADATA_ROWS
    col_start = FIDUCIAL_SIZE_CELLS

    matches = 0
    total = 0

    for r in range(min(20, config.data_rows)):
        for c in range(min(20, config.data_cols)):
            cell = grid[row_start + r, col_start + c]
            brightness = int(cell[0]) + int(cell[1]) + int(cell[2])
            expected_white = (r + c) % 2 == 0
            is_white = brightness > 384  # > 128 per channel average
            if is_white == expected_white:
                matches += 1
            total += 1

    return (matches / total) >= threshold if total > 0 else False


class FiducialTracker:
    """Tracks fiducial positions between frames to speed up detection.

    First frame: full-scan detection (normal template matching).
    Subsequent frames: search only in a small ROI around previous positions.
    If positions are stable (< threshold drift), reuses the previous homography.
    """

    def __init__(self, roi_margin: int = 60, drift_threshold: float = 2.0):
        self.roi_margin = roi_margin
        self.drift_threshold = drift_threshold
        self.last_positions: Optional[dict[str, tuple[float, float]]] = None
        self.last_homography: Optional[np.ndarray] = None
        self._templates: dict[str, np.ndarray] = {}
        self._config: Optional[GridConfig] = None

    def _ensure_templates(self, config: GridConfig, frame_h: int, frame_w: int) -> None:
        """Build and cache scaled templates for the current config."""
        if self._config is not None and self._config.cell_size == config.cell_size:
            return
        self._config = config
        n = FIDUCIAL_SIZE_CELLS
        patterns = get_fiducial_patterns(n)
        scale_x = frame_w / config.width
        scale_y = frame_h / config.height
        for corner, pattern in patterns.items():
            tmpl = _build_fiducial_template(pattern, config.cell_size)
            if abs(scale_x - 1.0) > 0.05 or abs(scale_y - 1.0) > 0.05:
                new_w = max(1, int(tmpl.shape[1] * scale_x))
                new_h = max(1, int(tmpl.shape[0] * scale_y))
                tmpl = cv2.resize(tmpl, (new_w, new_h), interpolation=cv2.INTER_AREA)
            self._templates[corner] = tmpl

    def detect(
        self, frame_gray: np.ndarray, config: GridConfig
    ) -> Optional[tuple[dict[str, tuple[float, float]], Optional[np.ndarray]]]:
        """Detect fiducials, using ROI tracking when possible.

        Returns:
            (positions_dict, homography_matrix) or None if detection fails.
        """
        h, w = frame_gray.shape[:2]
        self._ensure_templates(config, h, w)

        if self.last_positions is not None:
            result = self._detect_in_rois(frame_gray, config)
            if result is not None:
                positions = result
                H = self._compute_or_reuse_homography(positions, config)
                self.last_positions = positions
                self.last_homography = H
                return positions, H

        # Full scan fallback
        detected = detect_fiducials(frame_gray, config)
        if detected is None:
            return None
        H = compute_homography(detected, config)
        self.last_positions = detected
        self.last_homography = H
        return detected, H

    def _detect_in_rois(
        self, frame_gray: np.ndarray, config: GridConfig
    ) -> Optional[dict[str, tuple[float, float]]]:
        """Search for fiducials only in ROIs around last known positions."""
        h, w = frame_gray.shape[:2]
        results = {}

        for corner, (prev_x, prev_y) in self.last_positions.items():
            tmpl = self._templates[corner]
            tmpl_h, tmpl_w = tmpl.shape[:2]
            margin = self.roi_margin

            x1 = max(0, int(prev_x) - tmpl_w // 2 - margin)
            y1 = max(0, int(prev_y) - tmpl_h // 2 - margin)
            x2 = min(w, int(prev_x) + tmpl_w // 2 + margin)
            y2 = min(h, int(prev_y) + tmpl_h // 2 + margin)

            roi = frame_gray[y1:y2, x1:x2]
            if roi.shape[0] < tmpl_h or roi.shape[1] < tmpl_w:
                return None  # ROI too small — fall back to full scan

            res = cv2.matchTemplate(roi, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)

            if max_val < 0.3:
                return None

            cx = x1 + max_loc[0] + tmpl_w // 2
            cy = y1 + max_loc[1] + tmpl_h // 2
            results[corner] = (float(cx), float(cy))

        return results

    def _compute_or_reuse_homography(
        self, positions: dict[str, tuple[float, float]], config: GridConfig
    ) -> Optional[np.ndarray]:
        """Reuse last homography if positions haven't drifted significantly."""
        if self.last_positions is not None and self.last_homography is not None:
            max_drift = 0.0
            for corner in positions:
                if corner in self.last_positions:
                    dx = positions[corner][0] - self.last_positions[corner][0]
                    dy = positions[corner][1] - self.last_positions[corner][1]
                    max_drift = max(max_drift, (dx * dx + dy * dy) ** 0.5)
            if max_drift < self.drift_threshold:
                return self.last_homography

        return compute_homography(positions, config)

    def reset(self) -> None:
        """Clear tracking state (e.g., on scene change or config change)."""
        self.last_positions = None
        self.last_homography = None


def process_frame(
    frame_bgr: np.ndarray,
    config: GridConfig,
) -> Optional[tuple[np.ndarray, bool]]:
    """Full frame processing pipeline: detect fiducials, correct perspective, sample grid.

    Args:
        frame_bgr: Raw input BGR frame.
        config: Grid configuration.

    Returns:
        Tuple of (cell_grid_rgb, is_sync) or None if detection fails.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # Try to detect fiducials
    detected = detect_fiducials(gray, config)

    if detected is not None:
        H = compute_homography(detected, config)
        if H is not None:
            corrected = correct_perspective(frame_bgr, H, config)
        else:
            corrected = frame_bgr
    else:
        # Fall back: assume frame is already aligned (e.g., lossless path)
        corrected = frame_bgr

    # Resize if needed
    if corrected.shape[1] != config.width or corrected.shape[0] != config.height:
        corrected = cv2.resize(corrected, (config.width, config.height))

    grid = sample_grid_from_frame(corrected, config)
    sync = is_sync_frame(grid, config)

    return grid, sync
