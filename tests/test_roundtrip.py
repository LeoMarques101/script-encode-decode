"""End-to-end roundtrip test: encode a file, decode it, compare hashes."""

import hashlib
import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import GridConfig, ECCLevel, Compression
from encoder import encode as run_encode
from decoder import decode as run_decode


def _check_ffmpeg():
    import shutil
    if shutil.which("ffmpeg"):
        return True
    try:
        import imageio_ffmpeg
        imageio_ffmpeg.get_ffmpeg_exe()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _check_ffmpeg(), reason="ffmpeg not in PATH")
class TestRoundtrip:
    def _roundtrip(self, data: bytes, filename: str = "test.bin", **config_kw):
        """Helper: write data to file, encode to video, decode back, compare."""
        defaults = dict(
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
        defaults.update(config_kw)
        config = GridConfig(**defaults)

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = os.path.join(tmpdir, filename)
            video_path = os.path.join(tmpdir, "encoded.mp4")
            output_path = os.path.join(tmpdir, "decoded_" + filename)
            report_path = os.path.join(tmpdir, "report.json")

            # Write input
            with open(input_path, "wb") as f:
                f.write(data)

            original_hash = hashlib.sha256(data).hexdigest()

            # Encode
            run_encode(input_path, video_path, config, quiet=True)
            assert os.path.isfile(video_path), "Video file not created"
            assert os.path.getsize(video_path) > 0, "Video file is empty"

            # Decode
            report = run_decode(
                input_path=video_path,
                output_path=output_path,
                report_file=report_path,
                quiet=True,
                config_hint=config,
            )

            # Verify
            assert os.path.isfile(output_path), "Output file not created"
            with open(output_path, "rb") as f:
                decoded_data = f.read()

            decoded_hash = hashlib.sha256(decoded_data).hexdigest()
            assert decoded_hash == original_hash, (
                f"Hash mismatch!\n"
                f"  Original: {original_hash}\n"
                f"  Decoded:  {decoded_hash}\n"
                f"  Original size: {len(data)}\n"
                f"  Decoded size:  {len(decoded_data)}\n"
                f"  Report status: {report.status}"
            )
            assert report.hash_match

    def test_small_text(self):
        self._roundtrip(b"Hello, Video Steganography!" * 10)

    def test_binary_data(self):
        data = os.urandom(1024)
        self._roundtrip(data)

    def test_larger_file(self):
        data = os.urandom(5000)
        self._roundtrip(data)

    def test_no_compression(self):
        data = os.urandom(500)
        self._roundtrip(data, compression=Compression.NONE)

    def test_high_ecc(self):
        data = os.urandom(500)
        self._roundtrip(data, ecc_level=ECCLevel.HIGH)

    def test_color_levels_4(self):
        data = os.urandom(2000)
        self._roundtrip(data, color_levels=4)


class TestColorRoundtrip:
    def test_color_encoding_basic(self):
        """Test that color encode/decode roundtrips correctly at the byte level."""
        from core.color import encode_byte_to_cells, decode_cells_to_bytes

        for levels in [2, 4, 8]:
            data = os.urandom(50)
            cells = encode_byte_to_cells(data, levels)
            recovered = decode_cells_to_bytes(cells, levels, len(data))
            assert recovered == data, (
                f"Color roundtrip failed for {levels} levels"
            )


class TestPacketRoundtrip:
    def test_packet_split_and_decode(self):
        """Test packet splitting and individual packet decoding."""
        from core.packets import split_into_packets, decode_packet, raw_packet_size
        from core.config import GridConfig, ECCLevel

        config = GridConfig(
            resolution=(640, 480), cell_size=8, margin=20,
            ecc_level=ECCLevel.MEDIUM, color_levels=2,
        )
        data = os.urandom(2000)
        file_hash = hashlib.sha256(data).hexdigest()

        packets = split_into_packets(data, config, file_hash)
        assert len(packets) > 0

        rps = raw_packet_size(config)
        for pkt in packets:
            result = decode_packet(pkt, config.ecc_level, rps)
            assert result is not None, "Packet decode should succeed"
            header, payload = result
            assert header.frame_type.value == 1  # DATA


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
