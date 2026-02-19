"""Tests for Reed-Solomon ECC encoding/decoding."""

import os
import sys
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.ecc import encode, decode, encoded_size
from core.config import ECCLevel


class TestECC:
    def test_roundtrip_small(self):
        data = b"Hello, World!"
        for level in [ECCLevel.LOW, ECCLevel.MEDIUM, ECCLevel.HIGH]:
            encoded = encode(data, level)
            decoded = decode(encoded, len(data), level)
            assert decoded == data, f"Roundtrip failed for {level}"

    def test_roundtrip_larger(self):
        data = os.urandom(500)
        for level in [ECCLevel.LOW, ECCLevel.MEDIUM, ECCLevel.HIGH]:
            encoded = encode(data, level)
            decoded = decode(encoded, len(data), level)
            assert decoded == data

    def test_roundtrip_exact_block_boundary(self):
        data = os.urandom(255)
        encoded = encode(data, ECCLevel.MEDIUM)
        decoded = decode(encoded, len(data), ECCLevel.MEDIUM)
        assert decoded == data

    def test_error_correction_small(self):
        data = b"Test data for ECC"
        level = ECCLevel.HIGH  # 50% redundancy — max correction
        encoded = encode(data, level)

        # Corrupt a few bytes
        corrupted = bytearray(encoded)
        for i in range(0, min(5, len(corrupted)), 3):
            corrupted[i] ^= 0xFF

        decoded = decode(bytes(corrupted), len(data), level)
        assert decoded == data

    def test_empty_data(self):
        assert encode(b"", ECCLevel.MEDIUM) == b""
        assert decode(b"", 0, ECCLevel.MEDIUM) == b""

    def test_encoded_size_grows(self):
        for level in [ECCLevel.LOW, ECCLevel.MEDIUM, ECCLevel.HIGH]:
            size = encoded_size(100, level)
            assert size > 100, f"Encoded size should be > original for {level}"

    def test_encoded_size_ordering(self):
        n = 100
        low = encoded_size(n, ECCLevel.LOW)
        med = encoded_size(n, ECCLevel.MEDIUM)
        high = encoded_size(n, ECCLevel.HIGH)
        assert low <= med <= high


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
