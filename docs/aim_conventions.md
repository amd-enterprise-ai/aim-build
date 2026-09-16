<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# AIM Authoring Conventions

Conventions for authoring and reviewing an AIM under `assets/<accelerator>/<org>/<model>/`.

This document is the single source of truth for AIM conventions that require
human or LLM judgment. It is referenced by:

- the `aim-release-review` skill, which reviews an AIM before the `/release` label
- pre-commit hook error messages, which link back to the relevant section here
- [GitHub Copilot custom repository instructions](../.github/copilot-instructions.md)

**Mechanical rules are not repeated here.** Anything a pre-commit hook enforces
deterministically — file naming, schema shape, enum values, copyright headers,
YAML key ordering — lives in the hook, and the hook is authoritative. See
[Enforced automatically](#enforced-automatically) for what that covers.

The conventions below are written to hold across all accelerator families
(`instinct`, `epyc`, `radeon`, `cpu`). Where a value depends on the accelerator,
this document states how to *derive* it rather than tabulating per-SKU values.

## Profile `type` and the release lifecycle

`type` records what has actually been measured, not what is intended. Attach
the evidence for it to the PR — a developer statement, a link to a benchmark
result, or explicit performance-gate output are all acceptable, as long as the
claim is checkable rather than asserted:

- **`unoptimized`** — the default. Use it until benchmark results exist, or the
  results fall below the preview performance-gate threshold.
- **`preview`** — meets the preview performance-gate threshold; benchmarked and
  published ahead of full optimization.
- **`optimized`** — meets the optimized performance-gate threshold; benchmarked
  and tuned.
- **`general`** — engine-level default profile, not model-specific. Must live in a
  `general/` directory.

Promoting `type` in a follow-up PR once benchmarks land is expected and cheap;
a released AIM that overstates its own quality is not.

`type` also determines **auto-selectability**. Only `unoptimized` is excluded from
automatic profile selection; `optimized`, `preview`, and `general` are all
eligible and tie with each other. This is the role the removed
`manual_selection_only` flag used to play — see `AUTO_SELECTABLE_PRIORITY` in
[`src/aim_runtime/profile_selector.py`](../src/aim_runtime/profile_selector.py).
A profile that should not be selected automatically must be `type: unoptimized`;
there is no separate opt-out flag.

## `primary` is generated — do not hand-edit

Exactly one profile per `(accelerator_model, metric)` pair carries `primary: true`.
An AIM with profiles across several accelerators and both metrics therefore has
*many* primary profiles — this is correct, not a duplication error.

`primary` is set automatically by the `set-all-primary-flags` pre-commit hook.
Treat any hand-written value as a merge artifact. See
[metadata_overview.md — Primary Profiles](./metadata_overview.md#primary-profiles)
for the selection algorithm.

## Accelerator-dependent values

`accelerator_count` is interpreted by accelerator type:

- **GPU families** (`accelerator_type: gpu`) — the number of devices, i.e. the
  tensor-parallel size.
- **CPU families** (`accelerator_type: cpu`) — the number of cores.

A value of `1` is valid in both cases. The schema enforces `>= 1` only; it cannot
tell an under-provisioned CPU profile from a single-GPU one.

Engine args and environment variables are frequently accelerator-specific
(memory-space sizing, batching limits, backend selection). **Derive them from
sibling profiles in the same accelerator family and model**, never by copying
across families or models — a value tuned for one family or model is rarely
correct for another, and the schema will not catch it.

Before finalising profiles for a new model, check existing profiles for the
**same publisher and the same accelerator family**. Model families carry standard
engine args that should carry over unless there is a documented reason not to.

## Metadata that needs external verification

These fields cannot be validated from within the repository:

- **`hfToken.required`** — must match the actual Hugging Face model page. A wrong
  value fails at deployment time for every user. Check whether the model is
  gated before setting it.
- **`description.full`** — the complete upstream description, not a truncated
  copy.
- **`org.opencontainers.image.licenses`** — the upstream model's real license.

## Specialized AIMs

An AIM is *specialized* when it has an `image/Dockerfile`; this is the marker CI
uses to discover its layered image build. Model-level specialized AIMs currently
also set `specialized: true` in `config.yaml`, but that key is descriptive rather
than the build-discovery signal.

### Layout

```
assets/<accelerator>/<org>/<model>/image/
├── Dockerfile
├── requirements.txt        # or dependencies declared in pyproject.toml
├── pyproject.toml          # only with the `pip install -e .` pattern below
├── bentofile.yaml          # BentoML engine only
└── src/
    ├── harness.py          # ModelHarness subclass
    └── service.py          # BentoML engine only
```

### Making `src/` importable

Two patterns are in use and both are acceptable — match whichever the AIM already
uses rather than converting between them:

1. **`pip install -e .`** with a `pyproject.toml` declaring a src layout — used
   by the MONAI (`swinunetr`, `wholeBrainSeg_Large_UNEST_segmentation`) and
   `openfold3` images.
2. **A `.pth` file** pointing at `/workspace/model/src` — used by the `boltz2`
   and `echo-model` images:

   ```dockerfile
   RUN python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))" \
       | xargs -I{} sh -c 'echo /workspace/model/src > {}/workspace-model.pth'
   ```

Either way the image copies `src/` to `/workspace/model/src/`, so modules live at
`/workspace/model/src/<module>.py` — they are *not* flattened into
`/workspace/model/`. Comments in `bentofile.yaml` and `Dockerfile` must reflect
that.

### Dependency pinning

Every dependency in a specialized AIM's `requirements.txt` must be pinned to an
exact version (`==`) matching the validated CI build. Loose specifiers (`>=`,
`~=`, `<=`) and bare package names resolve differently on later builds, breaking
reproducibility with no code change.

The reliable source is the built image:

```bash
docker run --rm <built-image>:<tag> pip freeze
```

Record the source in a comment, e.g. `# Versions pinned to <image>:<tag>`.

Pin `torchvision` specifically, or omit it — an unpinned `torchvision` pulls in a
CPU-only torch that replaces the base image's ROCm build. Where a package must
not pull its own dependency tree, install it with `--no-deps` (see the MONAI
Dockerfiles).

External git dependencies must be pinned to a **40-character commit SHA**, not a
tag or branch. Tags are mutable; a branch is not a version at all.

Drop packages from `requirements.txt` once nothing imports them. A dependency
list that outgrows the code is the usual sign an image was copied from another
AIM without being adapted.

### Dockerfile

- **`ARG AIM_BUILD_REF` must not point at a feature branch.** The branch stops
  existing after merge and the image silently stops building.
- **Build-time verification must use `sys.exit()`, not bare `print`.** A canary
  that only prints cannot fail the build — if ROCm torch is absent, the build
  succeeds and the print shows `hip: None`. Use `sys.exit()` so a broken base
  image is caught at build time. Keep the `print` for diagnostic value:
  ```dockerfile
  RUN python3 -c "import torch, sys; print('torch:', torch.__version__, 'hip:', torch.version.hip); sys.exit(0 if torch.version.hip is not None else 'ROCm torch not installed')"
  ```
- **Use one pip invocation style throughout.** Mixing `pip3` and
  `python3 -m pip` in a single Dockerfile can resolve to different interpreters.

### Profile `env_vars` — never set secret keys

`env_vars` entries are applied as **hard overrides** (`os.environ[key] = value`)
before the service launches. Setting `HF_TOKEN: ''` in a profile silently
overwrites any pod-injected secret, causing the service to crash-loop with a
message telling the operator to set a variable they already set.

Secret keys (tokens, API keys) belong in `metadata.yaml` under `hfToken`
(or equivalent) and are injected by the operator via Kubernetes Secrets. They
must never appear in committed profile `env_vars`, even as empty strings.

### Harness — no side-effect imports

`harness.py` must be loadable in a minimal test environment (no GPU, no torch).
Any module it imports must not pull in `torch`, call `logging.basicConfig(force=True)`,
or run filesystem probes at import time.

If a constant from `config.py` is needed in the harness, extract it to a
dependency-free `constants.py` (no imports beyond stdlib) and import from there.
The harness test suite stubs heavy deps in `sys.modules` before loading the
harness; a `from config import X` that drags in `torch._dynamo` at import time
breaks the entire test suite.

When using `pip install -e .` with a `pyproject.toml`, every `.py` file in
`src/` must appear in `py-modules`. A module not listed is excluded from the
installed package, making it importable from the source tree only.

### Harness tests — import the real function

Test files must import and exercise the real function, not a local re-implementation
of it. A copy-paste twin provides zero regression protection: deleting the guard
from the real function leaves the suite green. The correct pattern stubs heavy
deps in `sys.modules` before importing the real module:

```python
sys.modules["torch"] = types.ModuleType("torch")  # stub
import pipeline  # real module

def test_raises_without_token(tmp_path, monkeypatch):
    monkeypatch.setattr(pipeline, "HF_TOKEN", None)
    with pytest.raises(RuntimeError, match="license"):
        pipeline.ensure_checkpoint(tmp_path, tmp_path / "model.pt")
```

`from config import HF_TOKEN` binds the name into the importing module's
namespace, so `monkeypatch.setattr(pipeline, "HF_TOKEN", ...)` is sufficient —
no environment manipulation needed.

### Path validation ordering

When validating caller-supplied paths for containment, the containment check
must run **before** `.exists()`. Checking existence first makes the endpoint a
filesystem existence oracle — a caller can enumerate the container filesystem by
distinguishing `FileNotFoundError` (path exists, escapes base) from `ValueError`
(path does not exist). Always validate, then stat:

## Profile coverage

A model must have a profile for every accelerator × precision × metric
combination it claims to support. Nothing validates the *absence* of a profile:
a missing combination is not a schema error, it is a deployment that silently
has no profile to select. Check the set of profiles against the claim, not just
the profiles that exist.

## Harness contract

Full guidance lives in the
[`aim-model-harness` skill](../.claude/skills/aim-model-harness/SKILL.md). The
conventions most often missed in review:

- **Resolve the service URL, never hardcode it.** Every reference must go through
  `config.resolve_service_url()`. A hardcoded URL silently ignores the
  `--service-url` flag CI passes, so CI validates the wrong endpoint — including
  when it is used only as a fallback default.
- **Fail loudly when the accelerator is missing.** Device selection must raise
  `RuntimeError` with an actionable message rather than falling back to CPU. A
  silent fallback starts the service and fails later as an OOM or extreme
  latency, with nothing pointing at the real cause.
- **Every `CheckResult` must be declared in the `CHECKS` catalog.** CI dashboards
  key off `CHECKS`; undeclared results are silently dropped. Check descriptions
  are user-facing and must describe what the code actually measures.
- **Gate checks on scope before doing the work.** `CheckScope.RUNTIME` means fast
  checks needing a live service; `CheckScope.OFFLINE` means heavier checks that
  may run for minutes. A check that requires a running service must not perform
  its network call before the scope gate — doing so blocks for the full timeout
  when the relevant scope was not requested.
- **Use `time.monotonic()` for deadlines**, not `time.time()` — wall-clock jumps
  cause spurious timeouts.
- **Use `Field(default_factory=...)` for mutable Pydantic defaults.**

## BentoML services

BentoML 1.x has two distinct health surfaces, and conflating them is a recurring
review finding:

- **`GET /healthz`** — the built-in readiness probe. Always present, requires no
  decorator. Harnesses must poll this for readiness.
- **`@bentoml.api(route="/health")`** — a *user-defined POST endpoint*. It
  duplicates the built-in probe and should not be added.

Every `@bentoml.api` route is POST-only in BentoML 1.x. Docstrings must not
describe a decorated route as a GET.
[`swinunetr/image/src/service.py`](../assets/instinct/monai/swinunetr/image/src/service.py)
documents the correct approach.

Reject conflicting inputs explicitly with HTTP 400 rather than silently ignoring
one of them, and keep docstrings in sync with the routes that actually exist.

## Public-repository hygiene

`assets/` is published. Before merging:

- **No internal ticket context in YAML.** Agent decision logs, ticket IDs, and
  controller-override narration do not belong in the public repo. A single
  neutral operational comment is fine (e.g. `# Touch to trigger CI rebuild`).
- **No template placeholders.** `__AIM_ID__`, `__MODEL_ID__`, and `[AGENT: …]`
  markers left by scaffolding cause `ProfileNotFound` at AIM startup.
- **No stray files** — `*_local_run.log`, `__pycache__`, `.DS_Store`, or
  committed binary artifacts.

## Enforced automatically

Do not spend review time on these — pre-commit rejects them:

| Convention | Hook |
|---|---|
| AMD copyright / SPDX headers | `copyright-check` |
| Profile filename matches its `metadata` fields | `check-profile-names` |
| Profile schema, enum values, unknown fields (`extra="forbid"`) | `check-profile-names` |
| `metadata.yaml` schema and unknown fields | `metadata-validation` |
| Model `canonicalName` matches `<org>/<model>` in its asset path | `check-canonical-name` |
| `config.yaml` registry allowlist and version-tag format | `validate-config` |
| `primary` flag assignment | `set-all-primary-flags` |
| YAML key ordering, parse errors, trailing whitespace, final newline | `yaml-key-order`, `check-yaml`, `trailing-whitespace`, `end-of-file-fixer` |
| Template placeholders in `assets/` | `check-placeholders` |
| Internal narration in `assets/` YAML and Python comments | `check-internal-comments`, `check-internal-comments-py` |
| Unpinned specialized-image dependencies (warning) | `check-requirements-pinned` |
| Secret key names in profile `env_vars` | `check-profile-env-vars-no-secrets` |
| ROCm torch canary missing `sys.exit()` | `check-rocm-canary` |

Run the full suite with `pre-commit run --all-files`.
