# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The embedded smoke PNG must be a valid, decodable image (regression: the
previous literal had a bad IDAT CRC and every smoke check failed)."""

from __future__ import annotations

import base64
import struct
import zlib

import pytest


def _png_bytes(harness_module) -> bytes:
    return base64.b64decode(harness_module._SMOKE_PNG_B64, validate=True)


def _chunks(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    pos = 8
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos : pos + 4])
        tag = data[pos + 4 : pos + 8]
        body = data[pos + 8 : pos + 8 + length]
        (crc,) = struct.unpack(">I", data[pos + 8 + length : pos + 12 + length])
        yield tag, body, crc
        pos += 12 + length
    assert pos == len(data)


def test_smoke_png_chunk_crcs_and_idat_inflate_without_pil(harness_module):
    chunks = list(_chunks(_png_bytes(harness_module)))
    assert [t for t, _, _ in chunks] == [b"IHDR", b"IDAT", b"IEND"]
    for tag, body, crc in chunks:
        assert zlib.crc32(tag + body) & 0xFFFFFFFF == crc, tag
    width, height, depth, color = struct.unpack(">IIBB", chunks[0][1][:10])
    assert (width, height, depth, color) == (224, 224, 8, 2)
    raw = zlib.decompress(chunks[1][1])
    assert len(raw) == height * (1 + 3 * width)


def test_smoke_png_decodes_with_pil(harness_module):
    pil = pytest.importorskip("PIL.Image")
    import io

    img = pil.open(io.BytesIO(_png_bytes(harness_module))).convert("RGB")
    assert img.size == (224, 224)
    assert img.getpixel((0, 0)) == (255, 255, 255)
