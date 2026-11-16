# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

# Makes `src.service:Boltz2Prediction` (the bentofile.yaml target) importable
# from the working dir, while the modules inside still import each other flat
# (`from runner import ...`) via the src-layout install. Mirrors the sibling
# openfold3 asset.
