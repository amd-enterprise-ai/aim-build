# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Accuracy evaluation shared by the in-image harness and the CI pipeline.

One implementation, two consumers: :meth:`aim_runtime.harness.vllm_harness.VLLMHarness.evaluate`
runs it inside the image, and ``ci/accuracy_evaluation/run_accuracy_evaluation.py``
runs it from the CI pipeline. Neither owns evaluation logic; both translate their
own inputs into :class:`aim_runtime.evaluation.settings.EvaluationSettings` and map
the results back out.

Submodules are imported explicitly rather than re-exported here, so that reading
settings never pulls in a backend.
"""
