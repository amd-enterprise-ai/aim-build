# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""ENV_PREFIX for the boltz2 AIM — namespace for its BOLTZ2_* env vars.

Read statically by the Tuner (stdlib ``ast``; the module is never imported, so
the model's heavy dependencies never reach the login node).
"""

ENV_PREFIX = "BOLTZ2"
