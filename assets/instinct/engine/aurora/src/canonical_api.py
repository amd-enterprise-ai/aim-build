# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""The two GET endpoints served alongside the BentoML inference route.

BentoML routes ``@bentoml.api`` methods as POST, so these are served by a small
Starlette app mounted on the service:

  * ``GET /health``    — 200 once the service is up. Forwards to BentoML's own
    ``/healthz``.
  * ``GET /v1/models`` — OpenAI-shaped list whose ``data[0].id`` is the model
    the container serves.

The app is mounted at ``/`` and must not define ``/healthz`` itself, or the
forward below would have nothing to reach.

Handlers are synchronous: Starlette runs them in a threadpool, so the loopback
call in ``/health`` does not block the event loop serving a forecast.
"""

import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route

HEALTH_PATH = "/health"
MODELS_PATH = "/v1/models"
HEALTHZ_PATH = "/healthz"
FORWARD_TIMEOUT_S = 5


def resolve_model_id(default=""):
    """The id ``GET /v1/models`` reports: ``AIM_ID`` if set, else the default."""
    return os.environ.get("AIM_ID") or default


def _forward_healthz(port):
    """Ask BentoML's own ``/healthz`` over loopback and return its status code."""
    url = "http://127.0.0.1:%d%s" % (int(port), HEALTHZ_PATH)
    try:
        with urlopen(Request(url), timeout=FORWARD_TIMEOUT_S) as resp:
            return resp.status
    except HTTPError as exc:
        return exc.code
    except (URLError, OSError):
        return 503


def build_canonical_app(port, model_id):
    """Return the Starlette app to mount on the service.

    ``port`` must be the port the service binds, because ``/health`` forwards to
    ``/healthz`` on it.
    """

    def health(request):
        code = _forward_healthz(port)
        return JSONResponse(
            {"status": "ok" if code == 200 else "unavailable"},
            status_code=code,
        )

    def models(request):
        return JSONResponse({"object": "list", "data": [{"id": model_id, "object": "model"}]})

    return Starlette(
        routes=[
            Route(HEALTH_PATH, health, methods=["GET"]),
            Route(MODELS_PATH, models, methods=["GET"]),
        ]
    )
