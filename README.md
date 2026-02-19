<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"/>
  <img src="https://img.shields.io/badge/opencv-4.8%2B-green?style=for-the-badge&logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/ffmpeg-required-red?style=for-the-badge&logo=ffmpeg&logoColor=white" alt="FFmpeg"/>
  <img src="https://img.shields.io/badge/license-MIT-yellow?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">VideoSteg</h1>

<p align="center">
  <strong>Encode any file into a video. Decode it back — bit-perfect.</strong>
</p>

<p align="center">
  A video steganography system that transforms arbitrary files into visually encoded videos<br/>
  with Reed-Solomon error correction, perspective recovery, and multi-level redundancy.<br/>
  Designed to survive real-world degradation: compression, rescaling, and screen capture.
</p>

---

## How It Works

```
                         ENCODE                                    DECODE
  ┌──────────┐    ┌──────────────┐    ┌─────────┐     ┌─────────┐    ┌──────────┐
  │          │    │  Compress     │    │  Video  │     │ Detect  │    │          │
  │  Any     │───>│  Split       │───>│  .mp4   │────>│ Correct │───>│  Original│
  │  File    │    │  ECC Encode  │    │  .mkv   │     │ Decode  │    │  File    │
  │          │    │  Frame Build │    │  .avi   │     │ Rebuild │    │          │
  └──────────┘    └──────────────┘    └─────────┘     └─────────┘    └──────────┘
```

Each frame of the output video is a **visual data grid** — a matrix of colored cells encoding binary data, surrounded by fiducial markers for alignment:

```
  ┌────────────────────────────────────────────┐
  │ margin                                     │
  │   ┌──┐                           ┌──┐     │
  │   │TL│  ▒▒▒ metadata row ▒▒▒    │TR│     │
  │   └──┘                           └──┘     │
  │        ┌──────────────────────┐            │
  │        │                      │            │
  │        │    DATA GRID         │            │
  │        │    (colored cells    │            │
  │        │     = encoded bytes) │            │
  │        │                      │            │
  │        └──────────────────────┘            │
  │   ┌──┐                           ┌──┐     │
  │   │BL│  ▒▒▒ metadata row ▒▒▒    │BR│     │
  │   └──┘                           └──┘     │
  │                                            │
  └────────────────────────────────────────────┘
```

The **4 corner fiducials** (TL, TR, BL, BR) are asymmetric patterns that allow the decoder to detect frame orientation and compute a homography for perspective correction — even from a phone camera pointed at a screen.

---

## Features

| Feature | Description |
|---------|-------------|
| **Bit-perfect recovery** | SHA-256 verified reconstruction of the original file |
| **Reed-Solomon ECC** | Corrects burst errors from video compression with configurable strength (10%/25%/50%) |
| **Byte interleaving** | Distributes RS blocks to resist localized corruption |
| **Perspective correction** | Homography-based alignment from fiducial markers |
| **Multi-level color encoding** | 2, 4, or 8 discrete levels per RGB channel (3/6/9 bits per cell) |
| **Packet redundancy** | Configurable N-copy redundancy across the video timeline |
| **Compression** | zlib or LZ4 pre-compression to reduce video length |
| **Codec flexibility** | H.264, H.265, or raw video output via FFmpeg pipe |
| **Auto-detection** | Decoder discovers all encoding parameters from the START frame |
| **Detailed reporting** | JSON decode report with frame-level error tracking |

---

## Quick Start

### Prerequisites

- **Python 3.10+**
- **FFmpeg** in PATH (or install `imageio[ffmpeg]` as fallback)

### Installation

```bash
git clone https://github.com/your-username/videosteg.git
cd videosteg
pip install -r requirements.txt
```

### Encode a file

```bash
python encoder.py -i secret_document.pdf -o encoded.mp4
```

### Decode it back

```bash
python decoder.py -i encoded.mp4 -o recovered.pdf
```

That's it. The recovered file will be byte-identical to the original.

---

## Usage

### Encoder

```
python encoder.py -i <file> -o <video> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | *required* | Input file path |
| `--output`, `-o` | `output.mp4` | Output video path |
| `--resolution` | `1920x1080` | Video resolution (`WxH`) |
| `--fps` | `30` | Frames per second |
| `--cell-size` | `8` | Cell size in pixels |
| `--margin` | `40` | Margin around the grid (px) |
| `--color-levels` | `2` | Color levels per channel: `2`, `4`, or `8` |
| `--ecc-level` | `medium` | Error correction: `low`, `medium`, `high` |
| `--redundancy` | `2` | Packet copy count |
| `--codec` | `h264` | Video codec: `h264`, `h265`, `rawvideo` |
| `--crf` | `0` | Quality factor (0 = lossless) |
| `--compression` | `zlib` | Data compression: `none`, `zlib`, `lz4` |

### Decoder

```
python decoder.py -i <video> -o <file> [options]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--input`, `-i` | *required* | Input video path |
| `--output`, `-o` | *auto-detect* | Output file path (from metadata) |
| `--output-dir` | `./` | Output directory |
| `--force` | `false` | Reconstruct even with missing packets |
| `--skip-frames` | `0` | Skip N frames from start |
| `--max-frames` | `0` | Process at most N frames (0 = all) |
| `--cell-size` | *auto* | Cell size hint for bootstrapping |
| `--margin` | *auto* | Margin hint |
| `--color-levels` | *auto* | Color levels hint |
| `--ecc-level` | *auto* | ECC level hint |

---

## Examples

### Minimal — small file, fast encode

```bash
python encoder.py -i notes.txt -o notes.mp4 \
  --resolution 640x480 --cell-size 8 --ecc-level low
```

### Maximum resilience — survives heavy compression

```bash
python encoder.py -i photo.jpg -o photo.mp4 \
  --resolution 1920x1080 --cell-size 12 --color-levels 2 \
  --ecc-level high --redundancy 4 --crf 0
```

### High density — more data per frame

```bash
python encoder.py -i data.bin -o data.mp4 \
  --cell-size 4 --color-levels 8 --ecc-level medium
```

### Decode with hints (when auto-detection struggles)

```bash
python decoder.py -i degraded_video.mp4 --force \
  --cell-size 8 --margin 40 --color-levels 2 --ecc-level medium
```

---

## Architecture

```
videosteg/
├── encoder.py           # CLI + encoding pipeline
├── decoder.py           # CLI + decoding pipeline
├── core/
│   ├── config.py        # GridConfig, enums, capacity math
│   ├── color.py         # Multi-level RGB cell encoding/decoding
│   ├── ecc.py           # Reed-Solomon with byte interleaving
│   ├── packets.py       # Packet structure, split/reassemble, start/end frames
│   ├── datagrid.py      # Cell grid <-> image conversion
│   ├── framing.py       # Frame composition with fiducials
│   ├── detection.py     # Fiducial detection, homography, perspective correction
│   └── report.py        # JSON decode report
├── tests/
│   ├── test_ecc.py      # Reed-Solomon encode/decode tests
│   ├── test_detection.py# Fiducial detection & grid sampling tests
│   └── test_roundtrip.py# Full encode -> decode integration tests
└── requirements.txt
```

### Pipeline

**Encoder:** `file` → `zlib/lz4 compress` → `split into packets` → `add 24-byte headers` → `pad to fixed size` → `Reed-Solomon ECC` → `color-encode cells` → `compose frames with fiducials` → `pipe to FFmpeg` → `video`

**Decoder:** `video` → `read frame` → `detect fiducials (template matching)` → `compute homography` → `warp perspective` → `sample cell grid (3x3 avg)` → `color-decode` → `ECC decode` → `parse header + CRC verify` → `reassemble packets` → `decompress` → `SHA-256 verify` → `file`

---

## Color Encoding

Each cell in the data grid uses discrete color levels per RGB channel to encode bits:

| Levels | Values | Bits/Channel | Bits/Cell | Robustness |
|--------|--------|-------------|-----------|------------|
| **2** | `0, 255` | 1 | 3 | Best — maximum contrast |
| **4** | `0, 85, 170, 255` | 2 | 6 | Good |
| **8** | `0, 36, 73, 109, 146, 182, 219, 255` | 3 | 9 | Fragile — needs lossless |

> **Recommendation:** Use `--color-levels 2` for anything that will go through lossy compression. Use `4` or `8` only with `--crf 0` (lossless).

---

## Error Correction

Reed-Solomon codes operate over GF(2^8) with block interleaving:

| Level | Parity Ratio | Can Correct | Best For |
|-------|-------------|-------------|----------|
| `low` | 10% | ~5% errors | Lossless / controlled environments |
| `medium` | 25% | ~12.5% errors | General use (default) |
| `high` | 50% | ~25% errors | Lossy codecs / screen capture |

Large data is split across multiple interleaved RS blocks (max 255 bytes each per GF(2^8) constraint). Byte interleaving distributes consecutive bytes across different blocks, so burst errors from localized frame corruption get spread across multiple correction domains.

---

## Capacity Reference

Approximate data capacity per frame at 1920x1080 with default settings:

| Cell Size | Color Levels | ECC | Raw Bytes/Frame | Payload Bytes/Frame |
|-----------|-------------|-----|----------------|-------------------|
| 4px | 2 | medium | ~18 KB | ~13 KB |
| 8px | 2 | medium | ~4.5 KB | ~3.3 KB |
| 8px | 4 | medium | ~9 KB | ~6.7 KB |
| 12px | 2 | medium | ~2 KB | ~1.5 KB |

A 6.3 MB JPEG encoded at 1920x1080, cell_size=8, color_levels=2, ecc=medium, redundancy=2 produces a ~1900-frame video (~63 seconds at 30fps).

---

## Running Tests

```bash
# All tests
pytest tests/ -v

# Individual test suites
pytest tests/test_ecc.py -v          # Reed-Solomon tests
pytest tests/test_detection.py -v    # Fiducial detection tests
pytest tests/test_roundtrip.py -v    # Full encode/decode integration tests
```

The roundtrip tests require FFmpeg to be available.

---

## Protocol Format

### Packet Header (24 bytes)

```
Offset  Size  Field
─────────────────────────────
0       1B    frame_type       (0=START, 1=DATA, 2=END, 3=SYNC)
1       4B    packet_index     (uint32)
5       4B    total_packets    (uint32)
9       4B    payload_crc32    (uint32)
13      8B    file_hash_frag   (first 8 bytes of SHA-256)
21      1B    cell_size        (uint8)
22      1B    color_levels     (uint8)
23      1B    ecc_level_code   (0=low, 1=medium, 2=high)
```

### Frame Sequence

```
[START x3] [DATA/SYNC...] [END x3]
     │           │              │
     │           │              └─ Carries full SHA-256 hash
     │           └─ Data packets interleaved with periodic sync frames
     └─ Carries filename, size, hash, grid config, compression mode
```

START frames are repeated 3x for reliability. The decoder uses the first successfully decoded START frame to configure itself.

---

## Resilience Design

The system is built to handle real-world degradation:

1. **Compression artifacts** → Reed-Solomon ECC corrects bit-level errors
2. **Frame drops** → Packet redundancy (N copies spread across timeline, not consecutive)
3. **Perspective distortion** → Fiducial-based homography correction
4. **Resolution changes** → Multi-scale template matching in detector
5. **Color shift** → Discrete color levels with threshold-based quantization
6. **Missing packets** → `--force` mode reconstructs partial files

---

## Limitations

- Color levels 4/8 are fragile under lossy compression — use level 2 for robustness
- Very small files (< 100 bytes) may produce disproportionately large videos due to control frame overhead
- Screen capture resilience depends on screen resolution and recording quality
- The decoder needs at least one valid START frame to reconstruct the file

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `opencv-python` | Frame I/O, template matching, homography |
| `numpy` | Array operations |
| `reedsolo` | Reed-Solomon error correction |
| `tqdm` | Progress bars |
| `lz4` | LZ4 compression (optional) |
| `imageio[ffmpeg]` | FFmpeg binary fallback |

---

## License

MIT
