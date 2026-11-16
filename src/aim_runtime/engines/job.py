# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Long-running job engine stub.

Long-running job AIMs (e.g. GROMACS) are not served via an HTTP engine —
they read input from $AIM_INPUT_PATH, run, write to $AIM_OUTPUT_PATH, and
exit. This stub satisfies the ENGINE_CLASSES registry requirement so that
Engine.JOB is a valid enum value; it is never instantiated at runtime for
a real job AIM (build_engine is not called for non-serving containers).
"""

from aim_runtime.config import AIMConfig
from aim_runtime.engine_config import EngineConfig
from aim_runtime.engines.base import BaseEngine


class JobEngine(BaseEngine):
    """Placeholder engine for long-running job containers.

    Never instantiated at runtime — long-running job AIMs do not use
    aim-runtime serve. Registered in ENGINE_CLASSES so Engine.JOB is a
    valid, mappable enum value.
    """

    def serve(self, config: AIMConfig, engine_config: EngineConfig) -> None:
        raise NotImplementedError(
            "JobEngine.serve() is never called — long-running job AIMs "
            "run via their own entrypoint, not via aim-runtime serve."
        )
