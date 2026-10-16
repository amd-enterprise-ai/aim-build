# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

import base64
from io import BytesIO

import pytest
from dinov2_golden_fixtures import fixture_payloads

Image = pytest.importorskip("PIL.Image")


def test_golden_fixtures_are_five_distinct_valid_images():
    payloads = fixture_payloads()

    assert set(payloads) == {
        "gradient",
        "checkerboard",
        "rings",
        "quadrants",
        "diagonal_stripes",
    }
    assert len(set(payloads.values())) == 5
    for payload in payloads.values():
        with Image.open(BytesIO(base64.b64decode(payload))) as image:
            assert image.mode == "RGB"
            assert min(image.size) >= 240
