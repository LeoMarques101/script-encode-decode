"""Decode report generation."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ErrorEntry:
    frame: int
    type: str
    detail: str
    packet: Optional[int] = None


@dataclass
class DecodeReport:
    status: str = "pending"  # "success", "partial", "failed"

    original_name: str = ""
    original_size: int = 0
    original_hash: str = ""

    total_frames_processed: int = 0
    data_frames_found: int = 0
    data_frames_decoded: int = 0
    data_frames_failed: int = 0
    control_frames_found: int = 0
    sync_frames_found: int = 0

    total_packets_expected: int = 0
    unique_decoded: int = 0
    missing_packets: list[int] = field(default_factory=list)
    corrected_by_ecc: int = 0
    corrected_by_redundancy: int = 0
    unrecoverable: int = 0

    bytes_recovered: int = 0
    bytes_total: int = 0
    recovery_percentage: float = 0.0
    hash_match: bool = False

    errors: list[ErrorEntry] = field(default_factory=list)

    def add_error(self, frame: int, err_type: str, detail: str,
                  packet: int | None = None) -> None:
        self.errors.append(ErrorEntry(frame, err_type, detail, packet))

    def finalize(self) -> None:
        """Compute derived fields."""
        if self.bytes_total > 0:
            self.recovery_percentage = (
                self.bytes_recovered / self.bytes_total * 100
            )

        all_decoded = set(range(self.total_packets_expected))
        decoded_set = set(range(self.total_packets_expected)) - set(self.missing_packets)
        self.unique_decoded = len(decoded_set)
        self.unrecoverable = len(self.missing_packets)

        if self.hash_match:
            self.status = "success"
        elif self.recovery_percentage > 0:
            self.status = "partial"
        else:
            self.status = "failed"

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "file_info": {
                "original_name": self.original_name,
                "original_size": self.original_size,
                "original_hash": self.original_hash,
            },
            "frames": {
                "total_processed": self.total_frames_processed,
                "data_frames_found": self.data_frames_found,
                "data_frames_decoded": self.data_frames_decoded,
                "data_frames_failed": self.data_frames_failed,
                "control_frames_found": self.control_frames_found,
                "sync_frames_found": self.sync_frames_found,
            },
            "packets": {
                "total_expected": self.total_packets_expected,
                "unique_decoded": self.unique_decoded,
                "missing": self.missing_packets,
                "corrected_by_ecc": self.corrected_by_ecc,
                "corrected_by_redundancy": self.corrected_by_redundancy,
                "unrecoverable": self.unrecoverable,
            },
            "reconstruction": {
                "bytes_recovered": self.bytes_recovered,
                "bytes_total": self.bytes_total,
                "recovery_percentage": round(self.recovery_percentage, 1),
                "hash_match": self.hash_match,
            },
            "errors": [
                {
                    "frame": e.frame,
                    "type": e.type,
                    **({"packet": e.packet} if e.packet is not None else {}),
                    "detail": e.detail,
                }
                for e in self.errors[:200]  # cap at 200 error entries
            ],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(self.to_json())
