"""Multi-process parallel decoder pipeline.

Architecture:
    Process 0 (reader)  — reads frames via FFmpeg pipe → puts into work queue
    Process 1..N (workers) — fiducial detect → warp → sample → color decode → ECC
    Main thread (assembler) — collects results, reassembles file

Uses multiprocessing (not threading) because RS-decode and NumPy loops are
CPU-bound and the GIL would prevent true parallelism.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import struct
import sys
from multiprocessing import Queue, Process, Event
from typing import Optional

import cv2
import numpy as np

from .config import GridConfig, ECCLevel, FrameType
from .detection import (
    detect_fiducials,
    compute_homography,
    correct_perspective,
    sample_grid_from_frame,
    is_sync_frame,
)
from .datagrid import read_data_from_grid
from .packets import decode_packet, raw_packet_size, PacketHeader, HEADER_SIZE


# Sentinel value to signal end of queue
_SENTINEL = None


def _worker_fn(
    work_queue: Queue,
    result_queue: Queue,
    stop_event: Event,
    config_dict: dict,
) -> None:
    """Worker process: decode a single frame's data.

    Receives (frame_idx, frame_bytes, height, width) from work_queue.
    Puts (frame_idx, header_or_none, payload_or_none, is_sync, error_str) into result_queue.
    """
    # Rebuild config from dict (can't pickle GridConfig easily across processes)
    config = GridConfig(**config_dict)
    rps = raw_packet_size(config)

    while not stop_event.is_set():
        try:
            item = work_queue.get(timeout=1.0)
        except Exception:
            continue

        if item is _SENTINEL:
            break

        frame_idx, frame_flat, height, width = item
        frame = np.frombuffer(frame_flat, dtype=np.uint8).reshape(height, width, 3)

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected = detect_fiducials(gray, config)

            if detected is None:
                result_queue.put((frame_idx, None, None, False, "fiducial_not_found"))
                continue

            H = compute_homography(detected, config)
            if H is not None:
                corrected = correct_perspective(frame, H, config)
            else:
                corrected = frame

            if corrected.shape[1] != config.width or corrected.shape[0] != config.height:
                corrected = cv2.resize(corrected, (config.width, config.height))

            grid = sample_grid_from_frame(corrected, config)

            if is_sync_frame(grid, config):
                result_queue.put((frame_idx, None, None, True, None))
                continue

            raw_bytes = read_data_from_grid(grid, config, config.raw_bytes_per_frame)
            decoded = decode_packet(raw_bytes, config.ecc_level, rps)

            if decoded is None:
                result_queue.put((frame_idx, None, None, False, "decode_failed"))
                continue

            header, payload = decoded
            # Serialize header fields to pass across process boundary
            header_dict = {
                "frame_type": header.frame_type.value,
                "packet_index": header.packet_index,
                "total_packets": header.total_packets,
            }
            result_queue.put((frame_idx, header_dict, payload, False, None))

        except Exception as e:
            result_queue.put((frame_idx, None, None, False, str(e)))


class ParallelDecoder:
    """Wraps the decoder pipeline with multiprocessing workers.

    Use this for large videos where single-threaded decode is too slow.
    For small videos or sequential mode, just use the regular decode() directly.
    """

    def __init__(self, config: GridConfig, workers: int = 0, max_queue: int = 0):
        self.config = config
        self.workers = workers or max(1, os.cpu_count() - 1)
        self.max_queue = max_queue or self.workers * 3
        self._processes: list[Process] = []
        self._stop_event = mp.Event()

    def decode_frames(
        self,
        frames_iter,
        total_frames: int,
        on_result=None,
    ) -> dict[int, bytes]:
        """Decode frames in parallel.

        Args:
            frames_iter: Iterator yielding (frame_idx, np.ndarray) tuples.
            total_frames: Total number of frames expected.
            on_result: Optional callback(frame_idx, header_dict, payload, is_sync, error).

        Returns:
            Dict mapping packet_index -> payload bytes.
        """
        work_q: Queue = Queue(maxsize=self.max_queue)
        result_q: Queue = Queue(maxsize=self.max_queue * 2)

        config_dict = {
            "resolution": self.config.resolution,
            "fps": self.config.fps,
            "cell_size": self.config.cell_size,
            "margin": self.config.margin,
            "color_levels": self.config.color_levels,
            "ecc_level": self.config.ecc_level,
            "redundancy": self.config.redundancy,
            "codec": self.config.codec,
            "crf": self.config.crf,
            "compression": self.config.compression,
        }

        # Start workers
        for _ in range(self.workers):
            p = Process(
                target=_worker_fn,
                args=(work_q, result_q, self._stop_event, config_dict),
                daemon=True,
            )
            p.start()
            self._processes.append(p)

        packets: dict[int, bytes] = {}
        frames_submitted = 0
        frames_received = 0

        try:
            for frame_idx, frame in frames_iter:
                # Submit frame as raw bytes (picklable)
                h, w = frame.shape[:2]
                work_q.put((frame_idx, frame.tobytes(), h, w))
                frames_submitted += 1

                # Drain results if available
                while not result_q.empty():
                    result = result_q.get_nowait()
                    frames_received += 1
                    fidx, hdr, payload, is_sync, err = result
                    if on_result:
                        on_result(fidx, hdr, payload, is_sync, err)
                    if hdr and hdr["frame_type"] == FrameType.DATA.value:
                        pkt_idx = hdr["packet_index"]
                        if pkt_idx not in packets:
                            packets[pkt_idx] = payload

            # Send sentinels
            for _ in range(self.workers):
                work_q.put(_SENTINEL)

            # Collect remaining results
            while frames_received < frames_submitted:
                try:
                    result = result_q.get(timeout=10.0)
                    frames_received += 1
                    fidx, hdr, payload, is_sync, err = result
                    if on_result:
                        on_result(fidx, hdr, payload, is_sync, err)
                    if hdr and hdr["frame_type"] == FrameType.DATA.value:
                        pkt_idx = hdr["packet_index"]
                        if pkt_idx not in packets:
                            packets[pkt_idx] = payload
                except Exception:
                    break

        finally:
            self._stop_event.set()
            for p in self._processes:
                p.join(timeout=5)
                if p.is_alive():
                    p.terminate()
            self._processes.clear()
            self._stop_event.clear()

        return packets
