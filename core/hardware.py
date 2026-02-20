"""Runtime hardware/software capability detection."""

from __future__ import annotations

import os
import shutil
from typing import Optional


def detect_capabilities() -> dict:
    """Detect available hardware and software acceleration.

    Returns:
        Dict describing available capabilities.
    """
    caps: dict = {
        "cpu_count": os.cpu_count() or 1,
        "cuda": _check_cuda(),
        "ffmpeg": _check_ffmpeg(),
        "ffmpeg_hwaccel": _check_ffmpeg_hwaccel(),
        "creedsolo": _check_creedsolo(),
    }

    try:
        import psutil
        caps["ram_gb"] = round(psutil.virtual_memory().total / (1024 ** 3), 1)
    except ImportError:
        caps["ram_gb"] = None

    return caps


def _check_cuda() -> bool:
    try:
        import cv2
        return cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        return False


def _check_ffmpeg() -> Optional[str]:
    """Return ffmpeg path or None."""
    path = shutil.which("ffmpeg")
    if path:
        return path
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _check_ffmpeg_hwaccel() -> list[str]:
    """Return list of available FFmpeg hwaccel methods."""
    import subprocess
    ffmpeg = _check_ffmpeg()
    if not ffmpeg:
        return []
    try:
        result = subprocess.run(
            [ffmpeg, "-hwaccels"],
            capture_output=True, text=True, timeout=5,
        )
        lines = result.stdout.strip().split("\n")
        # First line is "Hardware acceleration methods:", rest are method names
        return [l.strip() for l in lines[1:] if l.strip()]
    except Exception:
        return []


def _check_creedsolo() -> bool:
    try:
        import creedsolo
        return True
    except ImportError:
        return False


def print_capabilities() -> None:
    """Print a summary of detected capabilities."""
    caps = detect_capabilities()
    print("Hardware/Software Capabilities:")
    print(f"  CPUs: {caps['cpu_count']}")
    if caps["ram_gb"] is not None:
        print(f"  RAM: {caps['ram_gb']} GB")
    print(f"  CUDA: {'Yes' if caps['cuda'] else 'No'}")
    print(f"  FFmpeg: {caps['ffmpeg'] or 'Not found'}")
    if caps["ffmpeg_hwaccel"]:
        print(f"  HW Accel: {', '.join(caps['ffmpeg_hwaccel'])}")
    print(f"  creedsolo (fast RS): {'Yes' if caps['creedsolo'] else 'No (using reedsolo)'}")
