#!/usr/bin/env python3

# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Echo service — a trivial BentoML service that reverses input text.

Exposes (via BentoML's class-based ``@bentoml.service`` SDK):
  GET  /healthz       -> 200 (BentoML built-in readiness probe)
  POST /predict       -> {"reversed": "<input reversed>", "length": <int>}
  POST /predict_batch -> list of {"reversed": ..., "length": ...} results

This is a play model used to demonstrate the ModelHarness pattern against a
real BentoML-hosted engine. It has no ML dependencies — just BentoML.

How AIM launches it
-------------------
The BentoML engine entry in ``engines.yaml`` runs::

    python -m bentoml serve --working-dir /workspace/model --arg port=<AIM_PORT>

``bentoml serve`` defaults its target to ``.`` and therefore loads the
``service`` field from ``bentofile.yaml`` (``service: "service:EchoService"``).
AIM forwards the serving port as a BentoML template argument (``--arg port=…``),
which we consume below via :func:`bentoml.use_arguments` so the bound port
matches the one ``aim-runtime`` expects.
"""

from __future__ import annotations

import bentoml
from pydantic import BaseModel


class BentoArgs(BaseModel):
    """Template arguments forwarded by AIM through ``bentoml serve --arg``.

    BentoML serves on port 3000 by default; AIM overrides this with the
    resolved ``AIM_PORT`` value so the harness and the service agree.
    """

    port: int = 3000


args = bentoml.use_arguments(BentoArgs)


@bentoml.service(name="echo-model", http={"port": args.port})
class EchoService:
    """Reverses input text.

    BentoML automatically exposes ``/healthz`` (and ``/readyz``/``/livez``)
    readiness probes, so the harness's health check needs no custom endpoint.
    Each ``@bentoml.api`` method is published at ``/<method-name>``.
    """

    @bentoml.api
    def predict(self, text: str = "") -> dict:
        """Reverse a single string. Request body: ``{"text": "..."}``."""
        return {"reversed": text[::-1], "length": len(text)}

    @bentoml.api
    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Reverse a batch of strings. Request body: ``{"texts": ["a", "b"]}``."""
        return [{"reversed": item[::-1], "length": len(item)} for item in texts]
