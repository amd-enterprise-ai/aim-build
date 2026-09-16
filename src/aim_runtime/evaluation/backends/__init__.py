# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Evaluation backends: the tools that actually produce a score.

Nothing is re-exported here on purpose. Importing a backend pulls in whatever
that backend needs, and the runner imports exactly the one it was configured
with.
"""
