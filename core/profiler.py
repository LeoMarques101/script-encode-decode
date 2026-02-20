"""Performance profiler for the decoder pipeline.

Instruments each stage of decoding to measure time spent and identify bottlenecks.
Activated via the --profile flag on decoder.py.
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class StageStats:
    """Accumulated timing for a single pipeline stage."""
    total_seconds: float = 0.0
    call_count: int = 0

    @property
    def avg_ms(self) -> float:
        if self.call_count == 0:
            return 0.0
        return (self.total_seconds / self.call_count) * 1000

    @property
    def total_ms(self) -> float:
        return self.total_seconds * 1000


class DecodeProfiler:
    """Collects per-stage timing for the decoder pipeline.

    Usage::

        prof = DecodeProfiler(enabled=True)
        with prof.measure("fiducial_detection"):
            detected = detect_fiducials(...)
        ...
        prof.report()
    """

    # Canonical stage names in pipeline order
    STAGES = [
        "frame_extraction",
        "fiducial_detection",
        "homography_warp",
        "cell_sampling",
        "color_decode",
        "ecc_decode",
        "packet_assembly",
        "decompression_hash",
    ]

    STAGE_LABELS = {
        "frame_extraction": "Frame extraction",
        "fiducial_detection": "Fiducial detection",
        "homography_warp": "Homography + Warp",
        "cell_sampling": "Cell sampling (i2g)",
        "color_decode": "Color decode",
        "ecc_decode": "ECC decode (RS interl.)",
        "packet_assembly": "Packet assembly",
        "decompression_hash": "Decompress + hash",
    }

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._stats: dict[str, StageStats] = defaultdict(StageStats)
        self._wall_start: float | None = None
        self._wall_end: float | None = None
        self._total_frames: int = 0

    def start(self) -> None:
        """Mark the beginning of the decode run."""
        if self.enabled:
            self._wall_start = time.perf_counter()

    def finish(self, total_frames: int) -> None:
        """Mark the end of the decode run."""
        if self.enabled:
            self._wall_end = time.perf_counter()
            self._total_frames = total_frames

    class _Timer:
        """Context manager returned by ``measure()``."""
        __slots__ = ("_stats", "_name", "_t0")

        def __init__(self, stats: dict[str, StageStats], name: str):
            self._stats = stats
            self._name = name
            self._t0 = 0.0

        def __enter__(self):
            self._t0 = time.perf_counter()
            return self

        def __exit__(self, *exc):
            elapsed = time.perf_counter() - self._t0
            s = self._stats[self._name]
            s.total_seconds += elapsed
            s.call_count += 1

    class _NullTimer:
        """No-op context manager when profiling is disabled."""
        def __enter__(self):
            return self
        def __exit__(self, *exc):
            pass

    _null = _NullTimer()

    def measure(self, stage: str):
        """Return a context manager that times *stage*."""
        if not self.enabled:
            return self._null
        return self._Timer(self._stats, stage)

    # ── Reporting ──

    @property
    def wall_seconds(self) -> float:
        if self._wall_start is None or self._wall_end is None:
            return 0.0
        return self._wall_end - self._wall_start

    def report(self) -> str:
        """Build the profiling report table and return it as a string."""
        if not self.enabled:
            return ""

        n = self._total_frames or 1
        wall = self.wall_seconds or 1e-9

        lines: list[str] = []
        sep = "\u2550" * 58
        lines.append(f"\u2554{sep}\u2557")
        lines.append(
            f"\u2551  Decoder Performance Profile ({n} frames)"
            .ljust(59) + "\u2551"
        )
        lines.append(f"\u2560{sep}\u2563")

        header = (
            f"\u2551  {'Stage':<24}\u2502 {'Avg (ms)':>8} "
            f"\u2502 {'Total (s)':>9} \u2502 {'% Time':>6} \u2551"
        )
        lines.append(header)
        lines.append(
            f"\u2551  {'\u2500' * 24}\u253c{'\u2500' * 10}"
            f"\u253c{'\u2500' * 11}\u253c{'\u2500' * 8}\u2551"
        )

        grand_total = sum(s.total_seconds for s in self._stats.values()) or 1e-9

        bottleneck_name = ""
        bottleneck_pct = 0.0

        for stage in self.STAGES:
            s = self._stats.get(stage)
            if s is None or s.call_count == 0:
                continue
            label = self.STAGE_LABELS.get(stage, stage)
            pct = (s.total_seconds / grand_total) * 100
            lines.append(
                f"\u2551  {label:<24}\u2502 {s.avg_ms:>8.2f} "
                f"\u2502 {s.total_seconds:>9.3f} \u2502 {pct:>5.1f}% \u2551"
            )
            if pct > bottleneck_pct:
                bottleneck_pct = pct
                bottleneck_name = label

        # Total per frame
        total_avg_ms = (grand_total / n) * 1000
        lines.append(
            f"\u2551  {'\u2500' * 24}\u253c{'\u2500' * 10}"
            f"\u253c{'\u2500' * 11}\u253c{'\u2500' * 8}\u2551"
        )
        lines.append(
            f"\u2551  {'TOTAL per frame':<24}\u2502 {total_avg_ms:>8.2f} "
            f"\u2502 {grand_total:>9.3f} \u2502 {'100%':>6} \u2551"
        )

        lines.append(f"\u2560{sep}\u2563")

        fps = n / wall if wall > 0 else 0
        lines.append(
            f"\u2551  Throughput: {fps:.1f} frames/s".ljust(59) + "\u2551"
        )
        lines.append(
            f"\u2551  Wall time: {wall:.2f}s".ljust(59) + "\u2551"
        )
        if bottleneck_name:
            lines.append(
                f"\u2551  Bottleneck: {bottleneck_name} ({bottleneck_pct:.1f}%)"
                .ljust(59) + "\u2551"
            )
        lines.append(f"\u255a{sep}\u255d")

        return "\n".join(lines)
