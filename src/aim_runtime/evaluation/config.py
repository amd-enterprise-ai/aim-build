# Copyright © Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT

"""
Every tunable the evaluation path reads, and nothing behavioural.

These are the values an operator might want to change, each documented.
Behavioural modules import them rather than declaring their own: the runner, the
backends, and :mod:`aim_runtime.evaluation.settings`, which holds the two models
(``EvaluationSettings``, ``AccuracyEvalConfig``) that default to them.

The lm-eval argument defaults below are constants here but are *applied* by
``backends/lm_eval.py``, which owns the mapping from them onto lm-eval's argv.
That keeps ``EvaluationSettings`` free of one backend's spelling while still
centralizing the values.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

# --------------------------------------------------------------------------- #
# Backend selection
# --------------------------------------------------------------------------- #

#: Backend slug for lm-evaluation-harness. The results envelope reports it, so
#: a score stays attributable to the tool that produced it.
BACKEND_LM_EVAL = "lm-eval"

#: Backends that may be named in settings. One entry today; a second backend
#: replaces this check with registry dispatch.
SUPPORTED_BACKENDS = frozenset({BACKEND_LM_EVAL})

#: Command that runs the lm-eval backend. A bare name resolves on ``PATH``, which
#: is what a dev environment with the ``evaluation`` extra installed provides.
BACKEND_COMMAND = "lm_eval"

#: Environment variable that overrides :data:`BACKEND_COMMAND` at runtime, so an
#: image can relocate the backend without a code change.
BACKEND_COMMAND_ENV = "AIM_EVAL_BACKEND_COMMAND"

#: Where AIM images install the backend: its own virtualenv, so lm-eval's pinned
#: transformers/numpy never shadow the ones vLLM was built against. The base
#: Dockerfiles export this path as :data:`BACKEND_COMMAND_ENV`; the value lives
#: here so image and runtime cannot drift apart silently.
IN_IMAGE_BACKEND_COMMAND = "/workspace/tools/eval-venv/bin/lm_eval"


# --------------------------------------------------------------------------- #
# What to evaluate
# --------------------------------------------------------------------------- #

#: Tasks evaluated when nothing else is configured. gsm8k is small enough to run
#: on every CI profile and is what the pipeline records today.
DEFAULT_TASKS = ("gsm8k",)

#: In-context examples per request. Changing it changes what the score means, so
#: it travels with the results. ``None`` leaves each task's own YAML default in
#: force.
NUM_FEWSHOT = 5


# --------------------------------------------------------------------------- #
# How hard to push the endpoint
# --------------------------------------------------------------------------- #

#: Tensor-parallel width at or below which the low concurrency tier applies.
#: CPU profiles count cores in ``accelerator_count`` but are nominally TP 1, so
#: they must not be sized from that number — see :func:`concurrency_for`.
CONCURRENCY_TP_THRESHOLD = 2

#: Concurrent requests for endpoints up to :data:`CONCURRENCY_TP_THRESHOLD` wide.
CONCURRENCY_LOW_TP = 32

#: Concurrent requests for wider endpoints, which have the capacity to absorb
#: them.
CONCURRENCY_HIGH_TP = 64

#: Per-request timeout for accelerated (GPU) endpoints. Kept tight so a hung
#: request fails fast rather than blocking the whole evaluation.
REQUEST_TIMEOUT_SECONDS_ACCELERATED = 600

#: Per-request timeout for CPU endpoints, which generate far slower: a
#: full-length reasoning answer can exceed the GPU budget on every retry and
#: fail the run outright.
REQUEST_TIMEOUT_SECONDS_CPU = 1800

#: Retries the backend makes per request before giving up on it.
MAX_RETRIES = 10


# --------------------------------------------------------------------------- #
# Backend argument defaults (lm-eval vocabulary, applied by its adapter)
# --------------------------------------------------------------------------- #

#: lm-eval ``--model``: the endpoint flavour it talks to.
LM_EVAL_MODEL_TYPE = "local-completions"

#: lm-eval ``--batch_size``. "auto" lets it size batches from the endpoint.
BATCH_SIZE = "auto"

#: Generated-token ceiling per request. Reasoning models need room to finish an
#: answer; truncation scores as a wrong answer.
MAX_GEN_TOKS = 8192

#: Context-window ceiling reported to the backend, prompt plus generation.
MAX_MODEL_LENGTH = 16384

#: Endpoint path appended to the service URL for completion requests.
COMPLETIONS_PATH = "/v1/completions"

#: Directory, under the run's output directory, that the backend writes its own
#: results into. CI has always used this name, and the aggregated artifact sits
#: beside it rather than inside it.
BACKEND_OUTPUT_SUBDIR = "lm_eval_results"

#: How long the availability probe waits for the backend command to answer.
#: Importing the backend's dependencies dominates this, not the work itself.
PROBE_TIMEOUT_SECONDS = 120

#: Lines of backend output quoted in a failure message. Enough to carry the
#: traceback's last line without pasting a whole evaluation log into a check.
FAILURE_TAIL_LINES = 3


# --------------------------------------------------------------------------- #
# Backend argument names (the ``backend_args`` keys the lm-eval adapter reads)
# --------------------------------------------------------------------------- #
# Spellings are shared with ``AccuracyEvalConfig`` and with lm-eval's own
# ``--model_args`` keys, so the CI delegator can pass its wire config through
# without translation. They are constants rather than literals so a typo in a
# caller is a name error rather than a silently ignored argument.

#: lm-eval ``--model``: which endpoint flavour it talks to.
BACKEND_ARG_MODEL_TYPE = "lm_eval_model_type"

#: Whether to wrap prompts in the model's chat template.
BACKEND_ARG_APPLY_CHAT_TEMPLATE = "apply_chat_template"

#: Whether the tokenizer may execute code shipped with the model repository. Both
#: consumers derive it from the served profile's engine args, read through
#: ``aim_common.engine_args``, so the backend loads such a tokenizer the way the
#: engine did.
BACKEND_ARG_TRUST_REMOTE_CODE = "trust_remote_code"

#: Overrides :data:`BATCH_SIZE`.
BACKEND_ARG_BATCH_SIZE = "batch_size"

#: Overrides :data:`MAX_GEN_TOKS`.
BACKEND_ARG_MAX_GEN_TOKS = "max_gen_toks"

#: Overrides :data:`MAX_MODEL_LENGTH`.
BACKEND_ARG_MAX_MODEL_LENGTH = "max_length"

#: Tokenizer to score with; defaults to the served model.
BACKEND_ARG_TOKENIZER = "tokenizer"

#: Every ``backend_args`` key the lm-eval adapter understands. Anything else is
#: reported and ignored, because passing an unrecognised key through to a
#: subprocess argument list is how a typo becomes a wrong measurement.
LM_EVAL_BACKEND_ARGS = frozenset(
    {
        BACKEND_ARG_MODEL_TYPE,
        BACKEND_ARG_APPLY_CHAT_TEMPLATE,
        BACKEND_ARG_TRUST_REMOTE_CODE,
        BACKEND_ARG_BATCH_SIZE,
        BACKEND_ARG_MAX_GEN_TOKS,
        BACKEND_ARG_MAX_MODEL_LENGTH,
        BACKEND_ARG_TOKENIZER,
    }
)

#: What each overridable ``backend_args`` key falls back to. The three keys of
#: :data:`LM_EVAL_BACKEND_ARGS` missing here have no static default: the two
#: booleans are simply off when unset, and ``tokenizer`` follows the served
#: model, which is known only per run. Annotated as a ``Mapping`` so a caller
#: that tries to mutate the table fails type checking.
LM_EVAL_BACKEND_ARG_DEFAULTS: Mapping[str, str | int] = {
    BACKEND_ARG_MODEL_TYPE: LM_EVAL_MODEL_TYPE,
    BACKEND_ARG_BATCH_SIZE: BATCH_SIZE,
    BACKEND_ARG_MAX_GEN_TOKS: MAX_GEN_TOKS,
    BACKEND_ARG_MAX_MODEL_LENGTH: MAX_MODEL_LENGTH,
}


# --------------------------------------------------------------------------- #
# Which metric is the headline (lm-eval's vocabulary, its adapter's judgement)
# --------------------------------------------------------------------------- #
# Deliberately scoped to lm-eval rather than presented as the shared truth:
# "which metric is the score" is a per-backend, per-modality judgement. A
# segmentation backend's headline is a Dice coefficient, and a speech backend's
# is a word error rate where *lower* is better. Only the lm-eval adapter reads
# these; a new backend brings its own names rather than reusing them.

#: Metric names lm-eval reports that count as an accuracy score. A metric outside
#: this set is carried as detail rather than promoted to the headline number.
LM_EVAL_SCORE_METRIC_NAMES = frozenset({"acc", "acc_norm", "accuracy", "exact_match"})

#: Metric variant preferred for the headline score when a task reports several.
#: lm-eval calls these filters; gsm8k reports "strict-match" and
#: "flexible-extract" for the same generations.
LM_EVAL_PREFERRED_VARIANT = "flexible-extract"

#: Environment variable overriding :data:`LM_EVAL_PREFERRED_VARIANT`. Keeps the
#: name CI has always used, so an existing override keeps working.
LM_EVAL_PREFERRED_VARIANT_ENV = "PREFERRED_METRIC_FILTER"


# --------------------------------------------------------------------------- #
# Reading the results
# --------------------------------------------------------------------------- #

#: Version of the results envelope. Bump when a consumer would have to branch on
#: the change.
SCHEMA_VERSION = 1

#: Decimals kept on the run's elapsed time. It is a duration reported beside a
#: score, not a measurement anyone computes with.
ELAPSED_ROUND_DECIMALS = 2


# --------------------------------------------------------------------------- #
# How a harness reports the score
# --------------------------------------------------------------------------- #
# Check names are a CI-facing contract: ``ci/async/parse_validation_log.py``
# turns them into ``validation_test_results.test_name`` row values. Lowercase and
# underscored, and constants rather than literals so the harness and its tests
# cannot disagree about a spelling.

#: Check carrying the run's headline score.
PRIMARY_CHECK_NAME = "accuracy"

#: Prefix of the per-task checks, completed by the task name (``eval_gsm8k``).
TASK_CHECK_PREFIX = "eval_"


# --------------------------------------------------------------------------- #
# Where the results are written
# --------------------------------------------------------------------------- #
# Filenames CI already owns: the accuracy action declares both and uploads them
# by name (``.github/actions/accuracy-evaluation/action.yaml``), so they are a
# contract rather than a preference. Each has an environment override, following
# ``AIMBenchmark``'s ``BENCHMARK_{JSON,CSV}_FILE`` precedent — the override may
# be a bare filename or an absolute path, which is what the action sets.

#: Aggregated results, as :meth:`EvaluationResults.to_dict` publishes them.
RESULTS_JSON_FILENAME = "accuracy_evaluation_results.json"

#: Environment variable overriding :data:`RESULTS_JSON_FILENAME`.
RESULTS_JSON_FILE_ENV = "ACCURACY_JSON_FILE"

#: One row per task, as :meth:`EvaluationResults.to_csv_rows` produces them.
RESULTS_CSV_FILENAME = "accuracy_evaluation_results.csv"

#: Environment variable overriding :data:`RESULTS_CSV_FILENAME`.
RESULTS_CSV_FILE_ENV = "ACCURACY_CSV_FILE"

#: The backend's own results document, copied beside the aggregated files. Its
#: name is ours rather than CI's: nothing uploads or parses it today, and it
#: exists so a suspicious score can be traced back to what the tool actually
#: reported. Deliberately backend-neutral, since any backend's document lands here.
BACKEND_DOCUMENT_FILENAME = "evaluation_backend_results.json"


# --------------------------------------------------------------------------- #
# What the backend process inherits
# --------------------------------------------------------------------------- #

#: Variables the backend needs to reach its datasets and tokenizers, reported
#: before a run so a missing one is visible in the log rather than surfacing as a
#: download failure minutes later. The runner passes the whole environment
#: through; these are the ones worth naming. Who *sets* them for an in-image run
#: is unowned — see the plan's open items.
DATASET_ENV_VARS = ("HF_TOKEN", "HF_HOME", "HF_DATASETS_CACHE")


# --------------------------------------------------------------------------- #
# Shipped defaults
# --------------------------------------------------------------------------- #

#: Default settings file, shipped beside this module so an image can evaluate
#: with no arguments. Mirrors how ``AIMBenchmark`` finds ``benchmark-config.yaml``.
DEFAULT_CONFIG_FILENAME = "evaluation-config.yaml"


def lm_eval_preferred_variant() -> str:
    """The lm-eval metric variant to promote to the headline score.

    Read at call time rather than at import, so a test or an operator can set
    ``PREFERRED_METRIC_FILTER`` without having to control import order — which
    the CI parser's module-level global does require today.
    """
    return os.environ.get(LM_EVAL_PREFERRED_VARIANT_ENV) or LM_EVAL_PREFERRED_VARIANT
