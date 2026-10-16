# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Engine stand-in for the real-process supervisor tests.

Serves a fixed HTTP status on every path, so the caller decides whether the
supervisor's readiness probe ever succeeds: 200 makes it ready, 404 keeps it
alive but permanently unready.

Usage:
    python fake_engine.py <status> <port> [--exit-code CODE]
"""

from __future__ import annotations

import argparse
import http.server
import sys


def build_handler(status: int) -> type[http.server.BaseHTTPRequestHandler]:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802 (name mandated by BaseHTTPRequestHandler)
            self.send_response(status)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:
            pass

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("status", type=int, help="HTTP status to return on every path")
    parser.add_argument("port", type=int)
    parser.add_argument("--exit-code", type=int, default=None)
    args = parser.parse_args()

    if args.exit_code is not None:
        sys.exit(args.exit_code)

    http.server.HTTPServer(("127.0.0.1", args.port), build_handler(args.status)).serve_forever()


if __name__ == "__main__":
    main()
