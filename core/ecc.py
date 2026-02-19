"""Reed-Solomon error correction wrapper with byte interleaving."""

from __future__ import annotations

import math

from reedsolo import RSCodec, ReedSolomonError

from .config import ECCLevel

# Maximum RS block size (255 is the standard GF(2^8) limit)
RS_MAX_BLOCK = 255


def _rs_params(data_len: int, ecc_level: ECCLevel) -> tuple[int, int]:
    """Calculate RS parameters: (nsym, num_blocks) for the given data length.

    Uses interleaving: data is split into multiple RS blocks to keep each
    block within the GF(2^8) 255-byte limit and distribute burst errors.
    """
    ecc_ratio = ecc_level.ratio
    nsym_total = max(1, int(math.ceil(data_len * ecc_ratio)))
    total_len = data_len + nsym_total

    # Determine number of interleaved blocks
    num_blocks = max(1, math.ceil(total_len / RS_MAX_BLOCK))

    # nsym per block
    nsym_per_block = max(1, math.ceil(nsym_total / num_blocks))

    # Ensure each block doesn't exceed RS_MAX_BLOCK
    block_data = math.ceil(data_len / num_blocks)
    while block_data + nsym_per_block > RS_MAX_BLOCK:
        num_blocks += 1
        block_data = math.ceil(data_len / num_blocks)
        nsym_per_block = max(1, math.ceil(nsym_total / num_blocks))

    return nsym_per_block, num_blocks


def encode(data: bytes, ecc_level: ECCLevel) -> bytes:
    """Encode data with Reed-Solomon ECC using interleaving.

    Returns: data + ecc bytes (interleaved across blocks).
    """
    if not data:
        return b""

    nsym, num_blocks = _rs_params(len(data), ecc_level)
    codec = RSCodec(nsym)

    # Split data into blocks
    block_size = math.ceil(len(data) / num_blocks)
    blocks = []
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, len(data))
        blocks.append(data[start:end])

    # Encode each block
    encoded_blocks = []
    for block in blocks:
        encoded = bytes(codec.encode(block))
        encoded_blocks.append(encoded)

    # Interleave: write column by column across blocks
    max_block_len = max(len(b) for b in encoded_blocks)
    result = bytearray()
    for col in range(max_block_len):
        for block in encoded_blocks:
            if col < len(block):
                result.append(block[col])

    return bytes(result)


def decode(data: bytes, original_len: int, ecc_level: ECCLevel) -> bytes:
    """Decode Reed-Solomon encoded data with interleaving.

    Args:
        data: The interleaved encoded bytes.
        original_len: The original data length before encoding.
        ecc_level: The ECC level used during encoding.

    Returns:
        Decoded original data.

    Raises:
        ReedSolomonError: If errors exceed correction capacity.
    """
    if not data:
        return b""

    nsym, num_blocks = _rs_params(original_len, ecc_level)
    codec = RSCodec(nsym)

    # Calculate expected block sizes
    block_data_size = math.ceil(original_len / num_blocks)
    encoded_block_sizes = []
    for i in range(num_blocks):
        start = i * block_data_size
        end = min(start + block_data_size, original_len)
        d_len = end - start
        encoded_block_sizes.append(d_len + nsym)

    max_block_len = max(encoded_block_sizes) if encoded_block_sizes else 0

    # De-interleave
    blocks = [bytearray() for _ in range(num_blocks)]
    idx = 0
    for col in range(max_block_len):
        for b_idx in range(num_blocks):
            if col < encoded_block_sizes[b_idx]:
                if idx < len(data):
                    blocks[b_idx].append(data[idx])
                else:
                    blocks[b_idx].append(0)  # missing byte
                idx += 1

    # Decode each block
    result = bytearray()
    for i, block in enumerate(blocks):
        dec_result = codec.decode(bytes(block))
        # reedsolo returns (decoded_message, decoded_msgecc, errata_pos)
        decoded = bytes(dec_result[0])
        result.extend(decoded)

    return bytes(result[:original_len])


def encoded_size(data_len: int, ecc_level: ECCLevel) -> int:
    """Calculate the total encoded size for the given data length."""
    if data_len == 0:
        return 0
    nsym, num_blocks = _rs_params(data_len, ecc_level)
    block_size = math.ceil(data_len / num_blocks)
    total = 0
    for i in range(num_blocks):
        start = i * block_size
        end = min(start + block_size, data_len)
        total += (end - start) + nsym
    return total
