# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""BentoML engine.

A serving framework rather than an LLM engine: no ``served-model-name``
default, no AITER kernels, and forwarded (``--arg key=value``) CLI format.

The BentoML argument-validation model (:class:`BentomlEngineArgsModel`) lives
here too, next to the engine it describes; it is re-exported from
``aim_runtime.engines``.
"""

from __future__ import annotations

from typing import ClassVar, Optional

from aim_runtime.engines.base import BaseEngine
from aim_runtime.engines.engine_args_models import (
    EngineArgsFormat,
    EngineArgsModel,
)


class BentomlEngineArgsModel(EngineArgsModel):
    """Pydantic model for BentoML ``bentoml serve`` CLI arguments.

    Unlike vLLM, BentoML does **not** expose a native ``EngineArgs`` class or
    argparse-based CLI parser that can be hooked for authoritative validation.
    BentoML is a serving *framework* (wrapping inference engines) rather than
    an inference engine itself:

    * Its CLI is built with Click, not argparse, and only accepts
      framework-level flags (host, port, workers, SSL, etc.).
    * Model-specific and inference-engine configuration lives inside Python
      service code (``@bentoml.service()`` decorator) and ``bentofile.yaml``,
      not as CLI arguments.
    * BentoML's internal ``BentoMLConfiguration`` validates BentoML's own
      runtime config (tracing, logging, api_server settings) — not engine
      arguments in the AIM sense.

    BentoML *does* offer validation surfaces, but none function as a
    centralized engine-args validator:

    * ``bentoml.use_arguments()`` (v1.4.8+) lets service authors define a
      Pydantic schema for **per-service template arguments** (passed via
      ``--arg``). The schema is user-defined per service, not a built-in
      class that BentoML ships.
    * ``_bentoml_sdk.validators`` (``TensorSchema``, ``DataframeSchema``,
      ``PILImageEncoder``, etc.) validate API **request/response payloads**
      (tensors, images, dataframes), not CLI or engine configuration.

    Consequently this model uses **Pydantic-only validation** — there is no
    ``@model_validator(mode="wrap")`` delegation to a native parser. The
    fields below mirror the ``bentoml serve`` Click options as of BentoML
    v1.4.x so that AIM profiles can be validated at merge time.
    """

    port: int | None = None
    host: str | None = None
    api_workers: int | None = None
    timeout: int | None = None
    backlog: int | None = None
    reload: bool | None = None
    working_dir: str | None = None
    development: bool | None = None

    # SSL
    ssl_certfile: str | None = None
    ssl_keyfile: str | None = None
    ssl_keyfile_password: str | None = None
    ssl_version: int | None = None
    ssl_cert_reqs: int | None = None
    ssl_ca_certs: str | None = None
    ssl_ciphers: str | None = None

    # Timeouts
    timeout_keep_alive: int | None = None
    timeout_graceful_shutdown: int | None = None


class BentomlEngine(BaseEngine):
    """BentoML-backed serving engine."""

    ARGS_MODEL: ClassVar[Optional[type[EngineArgsModel]]] = BentomlEngineArgsModel
    ARGS_FORMAT: ClassVar[EngineArgsFormat] = EngineArgsFormat.FORWARDED
