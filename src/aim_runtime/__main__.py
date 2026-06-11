# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""Allow ``python -m aim_runtime`` as an alternative to the ``aim-runtime`` console script."""

from entrypoint import main

if __name__ == "__main__":
    main()
