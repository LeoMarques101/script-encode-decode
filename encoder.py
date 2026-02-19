#!/usr/bin/env python3
"""Video Steganography Encoder — encodes any file into a video with a visual datagrid."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from core.config import GridConfig, ECCLevel, Compression, FrameType
from core.packets import (
    compress_data,
    split_into_packets,
    build_start_frame_data,
    build_end_frame_data,
)
from core.framing import compose_frame, compose_sync_frame


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def _find_ffmpeg() -> str:
    """Find ffmpeg binary — system PATH first, imageio-ffmpeg fallback."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg:
        return ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    raise RuntimeError("ffmpeg not found in PATH and imageio-ffmpeg unavailable")


def _build_ffmpeg_cmd(output: str, config: GridConfig) -> list[str]:
    """Build the FFmpeg command for piping raw frames."""
    ffmpeg = _find_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{config.width}x{config.height}",
        "-r", str(config.fps),
        "-i", "-",
    ]

    if config.codec == "rawvideo":
        cmd += ["-vcodec", "rawvideo", "-pix_fmt", "bgr24"]
    elif config.codec == "h265":
        cmd += [
            "-vcodec", "libx265",
            "-crf", str(config.crf),
            "-preset", "medium",
            "-pix_fmt", "yuv444p",
        ]
    else:  # h264
        cmd += [
            "-vcodec", "libx264",
            "-crf", str(config.crf),
            "-preset", "medium",
            "-pix_fmt", "yuv444p",
        ]

    cmd += ["-an", output]
    return cmd


def encode(
    input_path: str,
    output_path: str,
    config: GridConfig,
    quiet: bool = False,
) -> None:
    """Main encoding pipeline."""
    config.validate()

    # ── Read input file ──
    input_file = Path(input_path)
    if not input_file.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    raw_data = input_file.read_bytes()
    file_size = len(raw_data)
    file_hash = hashlib.sha256(raw_data).hexdigest()
    filename = input_file.name

    # ── Compress ──
    compressed = compress_data(raw_data, config.compression)
    comp_size = len(compressed)
    comp_ratio = comp_size / file_size if file_size > 0 else 1.0

    # ── Split into packets ──
    packets = split_into_packets(compressed, config, file_hash)
    total_packets = len(packets)

    # ── Build control frame data ──
    start_data = build_start_frame_data(
        config, filename, file_size, file_hash, total_packets
    )
    end_data = build_end_frame_data(config, file_hash, total_packets)

    # ── Calculate frame schedule ──
    # Data frames with redundancy: distribute copies evenly
    data_frame_schedule = []  # list of packet indices in frame order
    for copy in range(config.redundancy):
        for pkt_idx in range(total_packets):
            data_frame_schedule.append(pkt_idx)

    # Insert sync frames
    sync_interval = config.sync_interval
    frames_with_sync = []
    data_idx = 0
    for i in range(len(data_frame_schedule)):
        if sync_interval > 0 and i > 0 and i % sync_interval == 0:
            frames_with_sync.append(("sync", None))
        frames_with_sync.append(("data", data_frame_schedule[i]))
        data_idx += 1

    # Full frame sequence: start(×3) + data/sync + end(×3)
    frame_sequence = []
    for _ in range(3):
        frame_sequence.append(("start", None))
    frame_sequence.extend(frames_with_sync)
    for _ in range(3):
        frame_sequence.append(("end", None))

    total_frames = len(frame_sequence)
    duration = total_frames / config.fps

    # ── Print summary ──
    if not quiet:
        print("=" * 54)
        print("  Video Steganography Encoder v1.0")
        print("=" * 54)
        print(f"  Input:  {filename} ({_human_size(file_size)})")
        print(f"  Output: {output_path} ({config.width}x{config.height} "
              f"@ {config.fps}fps)")
        print("-" * 54)
        print(config.summary(file_size, comp_size))
        print(f"  Compression ratio: {comp_ratio:.2f}")
        print(f"  Total frames: {total_frames} | Duration: {duration:.1f}s")
        print("=" * 54)

    # ── Start FFmpeg process ──
    ffmpeg_cmd = _build_ffmpeg_cmd(output_path, config)
    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    frame_size = config.width * config.height * 3

    try:
        progress = tqdm(
            total=total_frames,
            desc="Encoding",
            unit="frame",
            disable=quiet,
        )

        for frame_type, pkt_idx in frame_sequence:
            if frame_type == "start":
                img = compose_frame(start_data, config, FrameType.START_MARKER)
            elif frame_type == "end":
                img = compose_frame(end_data, config, FrameType.END_MARKER)
            elif frame_type == "sync":
                img = compose_sync_frame(config)
            else:
                img = compose_frame(packets[pkt_idx], config, FrameType.DATA)

            proc.stdin.write(img.tobytes())
            progress.update(1)

        progress.close()
    finally:
        proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg exited with code {proc.returncode}")

    out_size = os.path.getsize(output_path)
    if not quiet:
        print(f"\n  Done! Video: {_human_size(out_size)} "
              f"(ratio: {out_size / file_size:.1f}x)")


def main():
    parser = argparse.ArgumentParser(
        description="Video Steganography Encoder — encode any file into a video"
    )
    parser.add_argument("--input", "-i", required=True, help="Input file path")
    parser.add_argument("--output", "-o", default="output.mp4", help="Output video path")
    parser.add_argument("--resolution", default="1920x1080",
                        help="Video resolution WxH (default: 1920x1080)")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--cell-size", type=int, default=8,
                        help="Cell size in pixels (default: 8)")
    parser.add_argument("--margin", type=int, default=40,
                        help="Margin in pixels (default: 40)")
    parser.add_argument("--color-levels", type=int, default=2, choices=[2, 4, 8],
                        help="Color levels per channel (default: 2)")
    parser.add_argument("--ecc-level", default="medium",
                        choices=["low", "medium", "high"],
                        help="ECC level (default: medium)")
    parser.add_argument("--redundancy", type=int, default=2,
                        help="Packet redundancy count (default: 2)")
    parser.add_argument("--codec", default="h264",
                        choices=["h264", "h265", "rawvideo"],
                        help="Video codec (default: h264)")
    parser.add_argument("--crf", type=int, default=0,
                        help="Codec quality factor, 0=lossless (default: 0)")
    parser.add_argument("--compression", default="zlib",
                        choices=["none", "zlib", "lz4"],
                        help="File compression (default: zlib)")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")

    args = parser.parse_args()

    w, h = args.resolution.split("x")
    config = GridConfig(
        resolution=(int(w), int(h)),
        fps=args.fps,
        cell_size=args.cell_size,
        margin=args.margin,
        color_levels=args.color_levels,
        ecc_level=ECCLevel(args.ecc_level),
        redundancy=args.redundancy,
        codec=args.codec,
        crf=args.crf,
        compression=Compression(args.compression),
    )

    encode(args.input, args.output, config, quiet=args.quiet)


if __name__ == "__main__":
    main()
