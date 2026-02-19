#!/usr/bin/env python3
"""Video Steganography Decoder — reconstructs a file from an encoded video."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from core.config import GridConfig, ECCLevel, Compression, FrameType, FIDUCIAL_SIZE_CELLS, METADATA_ROWS
from core.packets import (
    decode_packet,
    decompress_data,
    StartFramePayload,
    PacketHeader,
    HEADER_SIZE,
    raw_packet_size,
    compute_chunk_size,
)
from core.datagrid import read_data_from_grid
from core.detection import process_frame, is_sync_frame
from core.report import DecodeReport


def _human_size(nbytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(nbytes) < 1024:
            return f"{nbytes:.1f} {unit}"
        nbytes /= 1024  # type: ignore[assignment]
    return f"{nbytes:.1f} TB"


def decode(
    input_path: str,
    output_path: Optional[str] = None,
    output_dir: str = "./",
    report_file: str = "decode_report.json",
    skip_frames: int = 0,
    max_frames: int = 0,
    force: bool = False,
    quiet: bool = False,
    config_hint: Optional[GridConfig] = None,
) -> DecodeReport:
    """Main decoding pipeline."""
    report = DecodeReport()

    # ── Open video ──
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {input_path}")

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not quiet:
        print("=" * 54)
        print("  Video Steganography Decoder v1.0")
        print("=" * 54)
        print(f"  Input: {input_path} ({video_width}x{video_height} "
              f"@ {video_fps:.2f}fps)")
        print(f"  Total video frames: {total_video_frames}")
        print("-" * 54)

    # ── Phase 1: Find START frame to get configuration ──
    config = None
    start_payload: Optional[StartFramePayload] = None
    packets_data: dict[int, bytes] = {}  # packet_index -> payload
    ecc_corrections = 0
    redundancy_saves = 0

    # Default config for initial detection (will be updated from START frame)
    if config_hint is not None:
        default_config = GridConfig(
            resolution=(video_width, video_height),
            fps=int(video_fps) if video_fps > 0 else 30,
            cell_size=config_hint.cell_size,
            margin=config_hint.margin,
            color_levels=config_hint.color_levels,
            ecc_level=config_hint.ecc_level,
        )
    else:
        default_config = GridConfig(
            resolution=(video_width, video_height),
            fps=int(video_fps) if video_fps > 0 else 30,
        )

    # Skip frames if requested
    if skip_frames > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)

    frames_to_process = total_video_frames - skip_frames
    if max_frames > 0:
        frames_to_process = min(frames_to_process, max_frames)

    progress = tqdm(
        total=frames_to_process,
        desc="Decoding",
        unit="frame",
        disable=quiet,
    )

    frame_idx = skip_frames
    failed_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if max_frames > 0 and (frame_idx - skip_frames) >= max_frames:
            break

        report.total_frames_processed += 1

        # Use the detected config or default
        active_config = config if config else default_config

        # Process frame
        result = process_frame(frame, active_config)

        if result is None:
            failed_frames += 1
            report.data_frames_failed += 1
            report.add_error(frame_idx, "fiducial_not_found",
                             "Could not detect fiducials")
            progress.update(1)
            frame_idx += 1
            continue

        grid, is_sync = result

        if is_sync:
            report.sync_frames_found += 1
            progress.update(1)
            frame_idx += 1
            continue

        # Read raw bytes from grid
        raw_bytes = read_data_from_grid(grid, active_config,
                                        active_config.raw_bytes_per_frame)

        # Try to decode packet
        rps = raw_packet_size(active_config)
        decoded = decode_packet(raw_bytes, active_config.ecc_level, rps)

        if decoded is None:
            # If we haven't found config yet, try different ECC levels
            if config is None:
                for try_ecc in [ECCLevel.MEDIUM, ECCLevel.LOW, ECCLevel.HIGH]:
                    tmp_cfg = GridConfig(
                        resolution=active_config.resolution,
                        fps=active_config.fps,
                        cell_size=active_config.cell_size,
                        margin=active_config.margin,
                        color_levels=active_config.color_levels,
                        ecc_level=try_ecc,
                    )
                    rps2 = raw_packet_size(tmp_cfg)
                    decoded = decode_packet(raw_bytes, try_ecc, rps2)
                    if decoded is not None:
                        break

        if decoded is None:
            failed_frames += 1
            report.data_frames_failed += 1
            report.add_error(frame_idx, "decode_failed",
                             "ECC decode failed for frame")
            progress.update(1)
            frame_idx += 1
            continue

        header, payload = decoded

        # Handle frame types
        if header.frame_type == FrameType.START_MARKER:
            report.control_frames_found += 1
            if start_payload is None:
                try:
                    start_payload = StartFramePayload.deserialize(payload)
                    # Build proper config from start frame
                    config = GridConfig(
                        resolution=(video_width, video_height),
                        fps=int(video_fps) if video_fps > 0 else 30,
                        cell_size=start_payload.cell_size,
                        margin=start_payload.margin,
                        color_levels=start_payload.color_levels,
                        ecc_level=start_payload.ecc_level,
                        compression=start_payload.compression,
                    )
                    config.validate()

                    report.original_name = start_payload.original_filename
                    report.original_size = start_payload.original_size
                    report.original_hash = f"sha256:{start_payload.sha256_hash}"
                    report.total_packets_expected = start_payload.total_packets
                    report.bytes_total = start_payload.original_size

                    if not quiet:
                        tqdm.write(
                            f"  Found START: {start_payload.original_filename} "
                            f"({_human_size(start_payload.original_size)})"
                        )
                        tqdm.write(
                            f"  Config: cell={start_payload.cell_size} "
                            f"colors={start_payload.color_levels} "
                            f"ecc={start_payload.ecc_level.value} "
                            f"packets={start_payload.total_packets}"
                        )
                except Exception as e:
                    report.add_error(frame_idx, "start_parse_failed", str(e))

        elif header.frame_type == FrameType.END_MARKER:
            report.control_frames_found += 1

        elif header.frame_type == FrameType.DATA:
            report.data_frames_found += 1
            pkt_idx = header.packet_index

            if pkt_idx in packets_data:
                # Already have this packet — redundancy copy
                redundancy_saves += 1
                report.corrected_by_redundancy += 1
            else:
                packets_data[pkt_idx] = payload
                report.data_frames_decoded += 1

        progress.update(1)
        frame_idx += 1

    progress.close()
    cap.release()

    # ── Phase 2: Reconstruct file ──
    if start_payload is None:
        if not quiet:
            print("\n  ERROR: No START frame found. Cannot reconstruct file.")
        report.status = "failed"
        report.finalize()
        report.save(report_file)
        return report

    total_packets = start_payload.total_packets

    # Check for missing packets
    missing = [i for i in range(total_packets) if i not in packets_data]
    report.missing_packets = missing

    if missing and not force:
        if not quiet:
            print(f"\n  WARNING: {len(missing)} missing packets out of {total_packets}")
            if len(missing) <= 20:
                print(f"  Missing: {missing}")

    # Reassemble compressed data
    compressed_chunks = []
    for i in range(total_packets):
        if i in packets_data:
            compressed_chunks.append(packets_data[i])
        else:
            # Missing packet — insert zeros (will likely corrupt the file)
            if packets_data:
                # Estimate chunk size from existing packets
                avg_size = sum(len(v) for v in packets_data.values()) // len(packets_data)
                compressed_chunks.append(b"\x00" * avg_size)
            else:
                compressed_chunks.append(b"")

    compressed_data = b"".join(compressed_chunks)
    report.bytes_recovered = len(compressed_data)

    # Decompress
    try:
        decompressed = decompress_data(compressed_data, start_payload.compression)
    except Exception as e:
        if not quiet:
            print(f"\n  ERROR: Decompression failed: {e}")
        if force:
            decompressed = compressed_data
        else:
            report.status = "failed"
            report.add_error(0, "decompress_failed", str(e))
            report.finalize()
            report.save(report_file)
            return report

    # Truncate to original size
    decompressed = decompressed[:start_payload.original_size]
    report.bytes_recovered = len(decompressed)

    # Verify hash
    actual_hash = hashlib.sha256(decompressed).hexdigest()
    expected_hash = start_payload.sha256_hash
    report.hash_match = (actual_hash == expected_hash)

    # Determine output path
    if output_path is None:
        output_path = os.path.join(output_dir, start_payload.original_filename)

    # Save file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(decompressed)

    report.finalize()

    if not quiet:
        print("-" * 54)
        print(f"  Status: {report.status.upper()}")
        print(f"  Packets: {report.unique_decoded}/{total_packets} "
              f"({report.unique_decoded / total_packets * 100:.1f}%)")
        print(f"  ECC corrections: {ecc_corrections}")
        print(f"  Redundancy saves: {redundancy_saves}")
        print(f"  Failed frames: {failed_frames}")
        print(f"  Hash match: {'YES' if report.hash_match else 'NO'}")
        print(f"  Output: {output_path} ({_human_size(len(decompressed))})")
        print("=" * 54)

    # Save report
    report.save(report_file)

    return report


def main():
    parser = argparse.ArgumentParser(
        description="Video Steganography Decoder — reconstruct file from encoded video"
    )
    parser.add_argument("--input", "-i", required=True, help="Input video path")
    parser.add_argument("--output", "-o", default=None,
                        help="Output file path (default: auto-detect from metadata)")
    parser.add_argument("--output-dir", default="./",
                        help="Output directory (default: ./)")
    parser.add_argument("--report-file", default="decode_report.json",
                        help="Report file path")
    parser.add_argument("--skip-frames", type=int, default=0,
                        help="Skip N frames from start")
    parser.add_argument("--max-frames", type=int, default=0,
                        help="Process at most N frames (0=all)")
    parser.add_argument("--force", action="store_true",
                        help="Try reconstruction even with errors")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode")
    parser.add_argument("--cell-size", type=int, default=None,
                        help="Cell size hint (default: auto-detect)")
    parser.add_argument("--margin", type=int, default=None,
                        help="Margin hint (default: auto-detect)")
    parser.add_argument("--color-levels", type=int, default=None, choices=[2, 4, 8],
                        help="Color levels hint (default: auto-detect)")
    parser.add_argument("--ecc-level", default=None,
                        choices=["low", "medium", "high"],
                        help="ECC level hint (default: auto-detect)")

    args = parser.parse_args()

    # Build config hint if any grid parameters provided
    config_hint = None
    if any(v is not None for v in [args.cell_size, args.margin, args.color_levels, args.ecc_level]):
        config_hint = GridConfig(
            cell_size=args.cell_size or 8,
            margin=args.margin if args.margin is not None else 40,
            color_levels=args.color_levels or 2,
            ecc_level=ECCLevel(args.ecc_level) if args.ecc_level else ECCLevel.MEDIUM,
        )

    report = decode(
        input_path=args.input,
        output_path=args.output,
        output_dir=args.output_dir,
        report_file=args.report_file,
        skip_frames=args.skip_frames,
        max_frames=args.max_frames,
        force=args.force,
        quiet=args.quiet,
        config_hint=config_hint,
    )

    sys.exit(0 if report.hash_match else 1)


if __name__ == "__main__":
    main()
