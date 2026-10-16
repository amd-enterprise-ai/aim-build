# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Capture reviewed DINOv2 golden embeddings from a running AIM."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from golden_fixtures import fixture_payloads

EMBED_DIM = 1024


def _post_predict(service_url: str, payload: dict[str, Any]) -> dict[str, Any]:
    body = json.dumps({"request": payload}).encode("utf-8")
    request = Request(
        f"{service_url.rstrip('/')}/predict",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def capture(service_url: str, output: Path, threshold: float, precision: str) -> None:
    fixtures = {}
    for name, image_base64 in fixture_payloads().items():
        response = _post_predict(service_url, {"image_base64": image_base64})
        embedding = response.get("embedding", [])
        if len(embedding) != EMBED_DIM or not all(math.isfinite(float(value)) for value in embedding):
            raise RuntimeError(f"fixture {name!r} returned an invalid embedding")
        fixtures[name] = [float(value) for value in embedding]

    manifest = {
        "model": "dinov2_vitl14",
        "precision": precision,
        "cosine_threshold": threshold,
        "fixtures": fixtures,
    }
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--service-url", default="http://localhost:8000")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.999)
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    capture(args.service_url, args.output, args.threshold, args.precision)


if __name__ == "__main__":
    main()
