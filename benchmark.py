#!/usr/bin/env python3
"""Benchmark suite for the video steganography decoder.

Generates test data, encodes it, then decodes with different optimization modes
and measures wall-clock time, throughput, and speedup.

Usage:
    python benchmark.py
    python benchmark.py --sizes 1KB,100KB --runs 2
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from typing import Optional

from core.config import GridConfig, ECCLevel, Compression
from core.hardware import detect_capabilities, print_capabilities
from encoder import encode as run_encode
from decoder import decode as run_decode


@dataclass
class BenchResult:
    label: str
    time_s: float
    fps: float
    speedup: float
    success: bool
    frames: int = 0


def _parse_size(s: str) -> int:
    """Parse human-readable size like '100KB' or '10MB'."""
    s = s.strip().upper()
    multipliers = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3}
    for suffix, mult in sorted(multipliers.items(), key=lambda x: -len(x[0])):
        if s.endswith(suffix):
            return int(float(s[:-len(suffix)]) * mult)
    return int(s)


def _human_size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.0f}{unit}"
        n /= 1024
    return f"{n:.0f}TB"


def _run_benchmark_case(
    video_path: str,
    output_dir: str,
    config_hint: GridConfig,
    label: str,
    baseline_time: Optional[float],
    fast_fiducial: bool = True,
    early_stop: bool = True,
    profile: bool = False,
    runs: int = 1,
) -> BenchResult:
    """Run one decode benchmark configuration."""
    times = []
    success = False
    total_frames = 0

    for _ in range(runs):
        out_path = os.path.join(output_dir, f"bench_output_{label}")
        report_path = os.path.join(output_dir, f"report_{label}.json")

        t0 = time.perf_counter()
        report = run_decode(
            input_path=video_path,
            output_path=out_path,
            report_file=report_path,
            quiet=True,
            config_hint=config_hint,
            profile=profile,
            fast_fiducial=fast_fiducial,
            early_stop=early_stop,
        )
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        success = report.hash_match
        total_frames = report.total_frames_processed

    avg_time = sum(times) / len(times)
    fps = total_frames / avg_time if avg_time > 0 else 0
    speedup = (baseline_time / avg_time) if baseline_time and avg_time > 0 else 1.0

    return BenchResult(
        label=label,
        time_s=avg_time,
        fps=fps,
        speedup=speedup,
        success=success,
        frames=total_frames,
    )


def run_benchmark(
    sizes: list[str],
    runs: int = 1,
    profile: bool = False,
) -> None:
    """Run the full benchmark suite."""
    print()
    print_capabilities()
    print()

    config = GridConfig(
        resolution=(640, 480),
        fps=30,
        cell_size=8,
        margin=20,
        color_levels=2,
        ecc_level=ECCLevel.MEDIUM,
        redundancy=2,
        codec="h264",
        crf=0,
        compression=Compression.ZLIB,
    )

    for size_str in sizes:
        size_bytes = _parse_size(size_str)
        print("=" * 68)
        print(f"  Benchmark: {_human_size(size_bytes)} random data")
        print("=" * 68)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Generate test data
            data = os.urandom(size_bytes)
            input_path = os.path.join(tmpdir, "test_input.bin")
            video_path = os.path.join(tmpdir, "encoded.mp4")

            with open(input_path, "wb") as f:
                f.write(data)

            # Encode
            print("  Encoding...", end=" ", flush=True)
            t0 = time.perf_counter()
            run_encode(input_path, video_path, config, quiet=True)
            enc_time = time.perf_counter() - t0
            vid_size = os.path.getsize(video_path)
            print(f"done ({enc_time:.2f}s, video={_human_size(vid_size)})")

            # Run benchmark cases
            results: list[BenchResult] = []

            # 1. Baseline: no optimizations
            print("  Running benchmarks...")
            baseline = _run_benchmark_case(
                video_path, tmpdir, config,
                label="baseline",
                baseline_time=None,
                fast_fiducial=False,
                early_stop=False,
                profile=profile,
                runs=runs,
            )
            results.append(baseline)
            baseline_time = baseline.time_s

            # 2. With fast fiducial only
            results.append(_run_benchmark_case(
                video_path, tmpdir, config,
                label="fast_fid",
                baseline_time=baseline_time,
                fast_fiducial=True,
                early_stop=False,
                profile=profile,
                runs=runs,
            ))

            # 3. With early stop only
            results.append(_run_benchmark_case(
                video_path, tmpdir, config,
                label="early_stop",
                baseline_time=baseline_time,
                fast_fiducial=False,
                early_stop=True,
                profile=profile,
                runs=runs,
            ))

            # 4. All optimizations
            results.append(_run_benchmark_case(
                video_path, tmpdir, config,
                label="all_opts",
                baseline_time=baseline_time,
                fast_fiducial=True,
                early_stop=True,
                profile=profile,
                runs=runs,
            ))

            # Print results table
            print()
            print(f"  {'Mode':<16} {'Time':>8} {'FPS':>8} {'Speedup':>8} {'OK':>4}")
            print(f"  {'-'*16} {'-'*8} {'-'*8} {'-'*8} {'-'*4}")
            for r in results:
                ok = "Yes" if r.success else "NO"
                print(
                    f"  {r.label:<16} {r.time_s:>7.2f}s "
                    f"{r.fps:>7.1f} {r.speedup:>7.2f}x {ok:>4}"
                )
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark suite for video steganography decoder"
    )
    parser.add_argument(
        "--sizes", default="1KB,10KB",
        help="Comma-separated test data sizes (default: 1KB,10KB)"
    )
    parser.add_argument(
        "--runs", type=int, default=1,
        help="Number of runs per benchmark case (default: 1)"
    )
    parser.add_argument(
        "--profile", action="store_true",
        help="Enable profiling for benchmark runs"
    )

    args = parser.parse_args()
    sizes = [s.strip() for s in args.sizes.split(",")]

    run_benchmark(sizes=sizes, runs=args.runs, profile=args.profile)


if __name__ == "__main__":
    main()
