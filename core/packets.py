"""Packet structure for data transmission through video frames.

Each packet contains:
  - Header (fixed size): frame_type, packet_index, total_packets, checksum, etc.
  - Payload: actual file data chunk
  - ECC is applied to (header + payload) padded to a fixed size

All packets are padded to the same raw_packet_size BEFORE ECC is applied,
so the decoder always knows the exact original length for ECC decoding.

Start/End frames carry file metadata instead of payload data.
"""

from __future__ import annotations

import hashlib
import math
import struct
import time
import zlib
from dataclasses import dataclass
from typing import List, Optional

from .config import Compression, ECCLevel, FrameType, GridConfig, PROTOCOL_VERSION
from .ecc import encode as ecc_encode, decode as ecc_decode, encoded_size


# Header format:
#   B  frame_type (1 byte)
#   I  packet_index (4 bytes)
#   I  total_packets (4 bytes)
#   I  crc32 of payload (4 bytes)
#   8s file_hash_fragment (8 bytes) - first 8 bytes of SHA-256
#   B  cell_size (1 byte)
#   B  color_levels (1 byte)
#   B  ecc_level_code (1 byte)  - 0=low, 1=medium, 2=high
#   = Total: 24 bytes
HEADER_FORMAT = "!BIII8sBBB"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)

ECC_LEVEL_CODE = {ECCLevel.LOW: 0, ECCLevel.MEDIUM: 1, ECCLevel.HIGH: 2}
CODE_ECC_LEVEL = {v: k for k, v in ECC_LEVEL_CODE.items()}


def compute_chunk_size(config: GridConfig) -> int:
    """Compute the payload chunk size for data packets.

    Finds the maximum payload size such that
    encoded_size(HEADER_SIZE + chunk, ecc_level) <= raw_bytes_per_frame.
    """
    raw_cap = config.raw_bytes_per_frame
    lo, hi = 1, raw_cap
    chunk_size = 1
    while lo <= hi:
        mid = (lo + hi) // 2
        enc = encoded_size(HEADER_SIZE + mid, config.ecc_level)
        if enc <= raw_cap:
            chunk_size = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return chunk_size


def raw_packet_size(config: GridConfig) -> int:
    """The fixed size of raw data (header+payload) before ECC, for all packets."""
    return HEADER_SIZE + compute_chunk_size(config)


@dataclass
class PacketHeader:
    frame_type: FrameType
    packet_index: int
    total_packets: int
    payload_crc32: int
    file_hash_fragment: bytes  # 8 bytes
    cell_size: int
    color_levels: int
    ecc_level: ECCLevel

    def pack(self) -> bytes:
        return struct.pack(
            HEADER_FORMAT,
            self.frame_type.value,
            self.packet_index,
            self.total_packets,
            self.payload_crc32,
            self.file_hash_fragment[:8].ljust(8, b"\x00"),
            self.cell_size,
            self.color_levels,
            ECC_LEVEL_CODE[self.ecc_level],
        )

    @classmethod
    def unpack(cls, data: bytes) -> "PacketHeader":
        (ft, idx, total, crc, hash_frag, cs, cl, ecc_code) = struct.unpack(
            HEADER_FORMAT, data[:HEADER_SIZE]
        )
        return cls(
            frame_type=FrameType(ft),
            packet_index=idx,
            total_packets=total,
            payload_crc32=crc,
            file_hash_fragment=hash_frag,
            cell_size=cs,
            color_levels=cl,
            ecc_level=CODE_ECC_LEVEL.get(ecc_code, ECCLevel.MEDIUM),
        )


@dataclass
class StartFramePayload:
    """Metadata carried in the START_MARKER frames."""
    original_filename: str
    original_size: int
    sha256_hash: str  # hex string
    total_packets: int
    cell_size: int
    color_levels: int
    ecc_level: ECCLevel
    compression: Compression
    margin: int
    timestamp: float
    protocol_version: int = PROTOCOL_VERSION

    def serialize(self) -> bytes:
        """Serialize to bytes."""
        name_bytes = self.original_filename.encode("utf-8")[:255]
        hash_bytes = bytes.fromhex(self.sha256_hash)
        buf = bytearray()
        buf += struct.pack("!B", self.protocol_version)
        buf += struct.pack("!B", len(name_bytes))
        buf += name_bytes
        buf += struct.pack("!Q", self.original_size)
        buf += hash_bytes[:32].ljust(32, b"\x00")
        buf += struct.pack("!I", self.total_packets)
        buf += struct.pack("!B", self.cell_size)
        buf += struct.pack("!B", self.color_levels)
        buf += struct.pack("!B", ECC_LEVEL_CODE[self.ecc_level])
        buf += struct.pack("!B",
                           {Compression.NONE: 0, Compression.ZLIB: 1, Compression.LZ4: 2}[self.compression])
        buf += struct.pack("!H", self.margin)
        buf += struct.pack("!d", self.timestamp)
        return bytes(buf)

    @classmethod
    def deserialize(cls, data: bytes) -> "StartFramePayload":
        offset = 0
        version = struct.unpack_from("!B", data, offset)[0]; offset += 1
        name_len = struct.unpack_from("!B", data, offset)[0]; offset += 1
        name = data[offset:offset + name_len].decode("utf-8"); offset += name_len
        size = struct.unpack_from("!Q", data, offset)[0]; offset += 8
        hash_bytes = data[offset:offset + 32]; offset += 32
        total_packets = struct.unpack_from("!I", data, offset)[0]; offset += 4
        cell_size = struct.unpack_from("!B", data, offset)[0]; offset += 1
        color_levels = struct.unpack_from("!B", data, offset)[0]; offset += 1
        ecc_code = struct.unpack_from("!B", data, offset)[0]; offset += 1
        comp_code = struct.unpack_from("!B", data, offset)[0]; offset += 1
        margin = struct.unpack_from("!H", data, offset)[0]; offset += 2
        timestamp = struct.unpack_from("!d", data, offset)[0]; offset += 8

        comp_map = {0: Compression.NONE, 1: Compression.ZLIB, 2: Compression.LZ4}
        return cls(
            original_filename=name,
            original_size=size,
            sha256_hash=hash_bytes.hex(),
            total_packets=total_packets,
            cell_size=cell_size,
            color_levels=color_levels,
            ecc_level=CODE_ECC_LEVEL.get(ecc_code, ECCLevel.MEDIUM),
            compression=comp_map.get(comp_code, Compression.NONE),
            margin=margin,
            timestamp=timestamp,
            protocol_version=version,
        )


def compress_data(data: bytes, compression: Compression) -> bytes:
    """Compress data with the selected algorithm."""
    if compression == Compression.NONE:
        return data
    elif compression == Compression.ZLIB:
        return zlib.compress(data, level=9)
    elif compression == Compression.LZ4:
        import lz4.frame
        return lz4.frame.compress(data)
    raise ValueError(f"Unknown compression: {compression}")


def decompress_data(data: bytes, compression: Compression) -> bytes:
    """Decompress data."""
    if compression == Compression.NONE:
        return data
    elif compression == Compression.ZLIB:
        return zlib.decompress(data)
    elif compression == Compression.LZ4:
        import lz4.frame
        return lz4.frame.decompress(data)
    raise ValueError(f"Unknown compression: {compression}")


def _encode_raw_to_frame(raw: bytes, config: GridConfig) -> bytes:
    """Pad raw data to fixed raw_packet_size, apply ECC, pad to raw_bytes_per_frame."""
    rps = raw_packet_size(config)
    raw_cap = config.raw_bytes_per_frame

    # Pad raw to fixed size before ECC
    if len(raw) < rps:
        raw = raw + b"\x00" * (rps - len(raw))
    raw = raw[:rps]

    encoded = ecc_encode(raw, config.ecc_level)

    # Pad encoded to frame capacity
    if len(encoded) < raw_cap:
        encoded = encoded + b"\x00" * (raw_cap - len(encoded))
    return encoded[:raw_cap]


def split_into_packets(
    data: bytes,
    config: GridConfig,
    file_hash: str,
) -> List[bytes]:
    """Split compressed data into packets, each with header + payload + ECC.

    Each packet's raw data (header+payload) is padded to raw_packet_size
    before ECC, so the decoder always knows the original length.
    """
    chunk_size = compute_chunk_size(config)
    total_packets = max(1, (len(data) + chunk_size - 1) // chunk_size)
    file_hash_frag = bytes.fromhex(file_hash)[:8] if file_hash else b"\x00" * 8

    packets = []
    for i in range(total_packets):
        start = i * chunk_size
        end = min(start + chunk_size, len(data))
        payload = data[start:end]
        # Pad payload to chunk_size so CRC matches after decode
        padded_payload = payload.ljust(chunk_size, b"\x00")

        header = PacketHeader(
            frame_type=FrameType.DATA,
            packet_index=i,
            total_packets=total_packets,
            payload_crc32=zlib.crc32(padded_payload) & 0xFFFFFFFF,
            file_hash_fragment=file_hash_frag,
            cell_size=config.cell_size,
            color_levels=config.color_levels,
            ecc_level=config.ecc_level,
        )

        raw = header.pack() + padded_payload
        packets.append(_encode_raw_to_frame(raw, config))

    return packets


def build_start_frame_data(
    config: GridConfig,
    filename: str,
    file_size: int,
    file_hash: str,
    total_packets: int,
) -> bytes:
    """Build the data for a START_MARKER frame."""
    payload = StartFramePayload(
        original_filename=filename,
        original_size=file_size,
        sha256_hash=file_hash,
        total_packets=total_packets,
        cell_size=config.cell_size,
        color_levels=config.color_levels,
        ecc_level=config.ecc_level,
        compression=config.compression,
        margin=config.margin,
        timestamp=time.time(),
    )
    payload_bytes = payload.serialize()
    file_hash_frag = bytes.fromhex(file_hash)[:8] if file_hash else b"\x00" * 8

    header = PacketHeader(
        frame_type=FrameType.START_MARKER,
        packet_index=0,
        total_packets=total_packets,
        payload_crc32=zlib.crc32(payload_bytes) & 0xFFFFFFFF,
        file_hash_fragment=file_hash_frag,
        cell_size=config.cell_size,
        color_levels=config.color_levels,
        ecc_level=config.ecc_level,
    )

    raw = header.pack() + payload_bytes
    return _encode_raw_to_frame(raw, config)


def build_end_frame_data(
    config: GridConfig,
    file_hash: str,
    total_packets: int,
) -> bytes:
    """Build the data for an END_MARKER frame."""
    payload = bytes.fromhex(file_hash)[:32].ljust(32, b"\x00")
    file_hash_frag = bytes.fromhex(file_hash)[:8] if file_hash else b"\x00" * 8

    header = PacketHeader(
        frame_type=FrameType.END_MARKER,
        packet_index=total_packets,
        total_packets=total_packets,
        payload_crc32=zlib.crc32(payload) & 0xFFFFFFFF,
        file_hash_fragment=file_hash_frag,
        cell_size=config.cell_size,
        color_levels=config.color_levels,
        ecc_level=config.ecc_level,
    )

    raw = header.pack() + payload
    return _encode_raw_to_frame(raw, config)


def decode_packet(
    raw_data: bytes,
    ecc_level: ECCLevel,
    expected_raw_size: int,
) -> Optional[tuple[PacketHeader, bytes]]:
    """Decode a packet: ECC-decode with known original size, parse header.

    Args:
        raw_data: The full frame data (raw_bytes_per_frame bytes).
        ecc_level: The ECC level used for encoding.
        expected_raw_size: The raw_packet_size (header + payload before ECC).

    Returns:
        Tuple of (header, payload) or None if decoding fails.
    """
    if not raw_data or expected_raw_size < HEADER_SIZE:
        return None

    try:
        decoded = ecc_decode(raw_data, expected_raw_size, ecc_level)
    except Exception:
        return None

    if len(decoded) < HEADER_SIZE:
        return None

    try:
        header = PacketHeader.unpack(decoded[:HEADER_SIZE])
    except Exception:
        return None

    payload = decoded[HEADER_SIZE:]

    # Validate CRC for DATA frames
    if header.frame_type == FrameType.DATA:
        actual_crc = zlib.crc32(payload) & 0xFFFFFFFF
        if actual_crc != header.payload_crc32:
            return None

    return header, payload
