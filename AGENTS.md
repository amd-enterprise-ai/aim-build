<!--
Copyright © Advanced Micro Devices, Inc., or its affiliates.

SPDX-License-Identifier: MIT
-->

# Agent Guide

Guidance for AI coding assistants working in this repository. This file follows
the [AGENTS.md convention](https://agents.md/) and is read by Cursor, Claude
Code, OpenAI Codex, Aider, Gemini CLI, and other agents.

Human contributors should also use this as an index — for the full contributor
guide see [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Repository orientation

- `src/aim_runtime/` — the Python package that ships inside every AIM image
  (profile selection, command generation, hardware detection, harness ABC).
- `src/entrypoint.py` — Click CLI exposed as `aim-runtime`.
- `assets/<vendor>/<org>/<model>/` — per-model configuration, profiles, and
  optional `image/src/harness.py` for non-vLLM engines.
- `docs/` — long-form documentation. Agents should treat the files referenced
  in the **Task playbooks** section below as authoritative for their topic.
- `tests/` — pytest test suite mirroring the `src/` layout.

## Task playbooks

When a task matches one of these triggers, read the linked guide *before*
making changes. These are the project's source-of-truth references — prefer
them over inferring conventions from a few files.

### Custom model harness

**Read [`.claude/skills/aim-model-harness/SKILL.md`](.claude/skills/aim-model-harness/SKILL.md) when the task involves any of:**

- Implementing or modifying a `ModelHarness` for an AIM image.
- Wiring a custom AIM image so it integrates with `aim-runtime validate`,
  `aim-runtime benchmark`, `aim-runtime evaluate`, or `aim-runtime list-checks`.
- Working in a file at `assets/<vendor>/<org>/<model>/image/src/harness.py`.
- Subclassing `aim_runtime.harness.ModelHarness`, or touching
  `src/aim_runtime/harness/`.
- Designing the `CHECKS` catalog, `CheckResult` output, or `HarnessResult`
  metrics that CI consumes.
- Questions about how the harness is discovered, what each method should
  return, or how to surface useful metrics for CI.

The skill covers when to write a custom harness vs. reuse `VLLMHarness`,
discovery rules, the per-method contract, output-quality standards, and a
copy-pasteable [`template.py`](.claude/skills/aim-model-harness/template.py).
Claude Code auto-loads this skill from its `.claude/skills/` directory; other
agents read it via this AGENTS.md pointer.

## General conventions

- Python 3.12+, formatted with the project's existing style (see `.flake8`
  and CI). Match the indentation and quoting of the file you're editing.
- Tests live in `tests/<package>/test_<module>.py` and use pytest.
- Don't add comments that narrate what the code does — see `CONTRIBUTING.md`
  for the full style guide.
- Don't commit large binary artifacts or `*_local_run.log` files.

## Where to put new agent-facing skills

This repo uses the Claude Code skill convention as the canonical home for
agent-facing guidance:

- Project skills live at `.claude/skills/<skill-name>/SKILL.md` with
  YAML frontmatter (`name`, `description`) and supporting files
  (`reference.md`, `examples.md`, `template.py`, etc.) in the same directory.
- Add a pointer to each new skill in the **Task playbooks** section above so
  non-Claude agents (Cursor, Codex, Aider, Gemini CLI, …) discover it via
  this AGENTS.md.

Don't mirror skills into `.cursor/skills/` or other tool-specific folders —
the AGENTS.md pointer keeps the source of truth single.
