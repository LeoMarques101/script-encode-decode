"""Tests for fiducial detection and grid sampling."""

import os
import sys
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import GridConfig
from core.framing import compose_frame, get_fiducial_patterns
from core.detection import (
    detect_fiducials,
    compute_homography,
    sample_grid_from_frame,
    process_frame,
    is_sync_frame,
)
from core.datagrid import grid_to_image, image_to_grid
import cv2


class TestFiducialPatterns:
    def test_patterns_are_unique(self):
        patterns = get_fiducial_patterns()
        keys = list(patterns.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                assert not np.array_equal(
                    patterns[keys[i]], patterns[keys[j]]
                ), f"Patterns {keys[i]} and {keys[j]} are identical"

    def test_patterns_have_correct_size(self):
        from core.config import FIDUCIAL_SIZE_CELLS
        patterns = get_fiducial_patterns()
        for name, pat in patterns.items():
            assert pat.shape == (FIDUCIAL_SIZE_CELLS, FIDUCIAL_SIZE_CELLS), \
                f"Pattern {name} has wrong shape: {pat.shape}"


class TestDetection:
    def _make_config(self, **kw):
        defaults = dict(resolution=(640, 480), cell_size=8, margin=20)
        defaults.update(kw)
        return GridConfig(**defaults)

    def test_detect_in_clean_frame(self):
        config = self._make_config()
        data = os.urandom(config.raw_bytes_per_frame)
        frame = compose_frame(data, config)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detected = detect_fiducials(gray, config)
        assert detected is not None, "Should detect fiducials in clean frame"
        assert set(detected.keys()) == {"tl", "tr", "bl", "br"}

    def test_process_frame_returns_grid(self):
        config = self._make_config()
        data = os.urandom(config.raw_bytes_per_frame)
        frame = compose_frame(data, config)

        result = process_frame(frame, config)
        assert result is not None
        grid, is_sync = result
        assert grid.shape == (config.grid_rows, config.grid_cols, 3)
        assert not is_sync

    def test_sync_frame_detected(self):
        config = self._make_config()
        from core.framing import compose_sync_frame
        frame = compose_sync_frame(config)

        result = process_frame(frame, config)
        assert result is not None
        _, is_sync = result
        assert is_sync


class TestGridRoundtrip:
    def test_grid_to_image_and_back(self):
        config = GridConfig(resolution=(320, 240), cell_size=8, margin=16)
        rows, cols = config.grid_rows, config.grid_cols
        grid = np.random.randint(0, 256, (rows, cols, 3), dtype=np.uint8)

        img = grid_to_image(grid, config)
        recovered = image_to_grid(img, config)

        # Should be identical for clean images
        np.testing.assert_array_equal(grid, recovered)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
