# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Deterministic synthetic fixtures for DINOv2 embedding regression."""

from __future__ import annotations

import base64
from io import BytesIO
from typing import Any


def _image_module() -> Any:
    from PIL import Image

    return Image


def _image_draw_module() -> Any:
    from PIL import ImageDraw

    return ImageDraw


def _gradient() -> Any:
    image = _image_module().new("RGB", (320, 240))
    image.putdata(
        [
            (
                round(255 * x / 319),
                round(255 * y / 239),
                round(255 * (x + y) / 558),
            )
            for y in range(240)
            for x in range(320)
        ]
    )
    return image


def _checkerboard() -> Any:
    image = _image_module().new("RGB", (240, 320), (240, 180, 30))
    draw = _image_draw_module().Draw(image)
    size = 30
    for y in range(0, image.height, size):
        for x in range(0, image.width, size):
            if (x // size + y // size) % 2:
                draw.rectangle((x, y, x + size - 1, y + size - 1), fill=(20, 80, 190))
    return image


def _rings() -> Any:
    image = _image_module().new("RGB", (288, 288), (15, 20, 35))
    draw = _image_draw_module().Draw(image)
    colors = ((230, 70, 60), (245, 190, 45), (45, 170, 120), (55, 105, 210))
    for index, radius in enumerate(range(140, 15, -24)):
        color = colors[index % len(colors)]
        draw.ellipse((144 - radius, 144 - radius, 144 + radius, 144 + radius), fill=color)
    return image


def _quadrants() -> Any:
    image = _image_module().new("RGB", (360, 240))
    draw = _image_draw_module().Draw(image)
    draw.rectangle((0, 0, 179, 119), fill=(220, 45, 45))
    draw.rectangle((180, 0, 359, 119), fill=(40, 175, 75))
    draw.rectangle((0, 120, 179, 239), fill=(45, 90, 220))
    draw.rectangle((180, 120, 359, 239), fill=(235, 210, 50))
    return image


def _diagonal_stripes() -> Any:
    image = _image_module().new("RGB", (240, 360), (25, 25, 25))
    draw = _image_draw_module().Draw(image)
    colors = ((225, 80, 150), (50, 195, 210), (245, 150, 45))
    for index, offset in enumerate(range(-360, 240, 36)):
        draw.polygon(
            (
                (offset, 0),
                (offset + 18, 0),
                (offset + 378, 360),
                (offset + 360, 360),
            ),
            fill=colors[index % len(colors)],
        )
    return image


def fixture_payloads() -> dict[str, str]:
    fixtures = {
        "gradient": _gradient(),
        "checkerboard": _checkerboard(),
        "rings": _rings(),
        "quadrants": _quadrants(),
        "diagonal_stripes": _diagonal_stripes(),
    }
    payloads = {}
    for name, image in fixtures.items():
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=False, compress_level=9)
        payloads[name] = base64.b64encode(buffer.getvalue()).decode("ascii")
    return payloads
