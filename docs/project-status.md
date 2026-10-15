# Project status and handoff

Updated 2026-10-15. Owners confirmed by Ruthvik: Track 1 Ruthvik; Track 2 Hancheng; Track 3 May.

## Current state

This Track 1 branch contains Ruthvik's task/evaluation implementation, tests and handoff documentation. Local validation rerun 2026-10-15: 25 tests pass, Ruff checks/formatting pass; the saved audit covers 64 official instances (16 core levels x seeds 42–45), all passing feasibility and gold-oracle checks. No real-model experiment has run through this implementation.

May's `may/track3-infra` branch is now available (inspected at `98b98a8`): it includes Slurm setup/serving scripts, `docs/cluster_setup.md`, a draft `docs/log_schema.md`, and reported model smoke tests in its project context. Those reports have not been independently rerun by Track 1. The branch is not yet merged into main; Hancheng's harness has not been inspected or integrated. Both tracks modify agent/dependency files: reconcile `AGENTS.md`, `pyproject.toml` and `uv.lock` when integrating, preserving the official benchmark pin and May's serving requirements. May's personal agent preferences should remain distinct from shared team defaults.

The shared repo should carry code, interface contracts, setup instructions, decisions and aggregate results. `AGENTS.md` contains portable team instructions. Each person's private notes may add context but must not be required to run the project.

## Immediate handoffs

| Owner | Next deliverable | What this unblocks |
|---|---|---|
| May | Review/integrate the published infra setup with Track 1; confirm endpoint/tunnel access and model/precision availability. | Reproducible serving and team access |
| Ruthvik | Submit this branch for review, agree the task/log interface with Hancheng and May, and help validate real episodes and failures. | Stable task and scoring inputs for Hancheng |
| Hancheng | Connect one served model to the task interface, implement the unmodified no-management loop first, and export actual per-turn snapshots/termination/usage. | First end-to-end real-model check |
| All, then Ruthvik on evaluation | Inspect the first two below-budget episodes and one pressure-level episode per task; fix protocol/formatting problems before scaling. | Evidence that model, harness, task and scorer work together |

## Hancheng: minimal implementation order

1. Read `docs/tasks-and-evaluation.md`. Reuse official `contextbench.run` if feasible; implement its three-method agent interface around May's OpenAI-compatible server. Keep gold/future operations outside model-accessible tools.
2. Complete a full no-management episode for KV R130 and Needle K4-C4, seed 42. Honor readiness/submit markers, expose the actual live context, snapshot before compaction and preserve termination/usage. Do not silently truncate context or omit failed episodes.
3. Add CLM context editing and a manager hook. Freeze the self/peer definitions and threshold before comparison: a same-size peer must be an operationally distinct session/role. Keep editing tools and the allowed information consistent between comparable conditions. Reuse the same served backend for same-size roles where possible; separate roles do not require duplicate weight loads.
4. Implement the self, same-size peer and fixed-27B peer conditions with separate actor/manager logs. Record manager triggers, context edits, tokens and termination; let May's accounting include both roles. Stop at the instructor's core grid before adding deferred baselines.

## First model smoke test

Use one of May's already-working Qwen models, exact ID and precision recorded. First run both below-budget controls (KV R130; Needle K4-C4), seed 42, no management. Then try KV R1000 and Needle K4-C24 to inspect overflow and management behavior once editing is connected. A complete but inaccurate small-model run is diagnostic, not a test failure to hide; separate model inability, protocol failure, context overflow and scorer bugs. Below-budget controls should be examined before deciding to exclude any actor size. An append-only pressure run hitting a budget wall can be expected and should be recorded honestly.

For each episode require: reproducible command/config, commit IDs, full protocol execution or explicit failure reason, snapshot-backed official score plus diagnostics, context/usage logs, and confirmation that gold was never exposed. For the managed pressure pilot also inspect that edits affect the next model input and required needles/queried values survive as intended. Expand to shared seeds and the reduced pilot grid only after these checks.

## What remains in Track 1

The initial task code and local oracle validation are done. Track 1 is not finished for the project: review with Hancheng, a real-model integration check, failure analysis and scorer fixes as needed, and final benchmark/methods/report material still remain. `docs/benchmark-literature.md` is an initial sourced brief; its supporting-paper summaries are abstract-scoped and need deeper reading before final report claims. Final figures and model runs depend on Tracks 2/3.

## Interface decisions still to freeze

- Management trigger and the operational distinction between CLM self and same-size peer.
- Tool permissions, snapshot timing and manager-visible context; common o200k vs serving-tokenizer limits.
- Model/precision availability and a feasible scheduling plan for a 27B manager under course GPU limits.
- Prompt versions, output caps, sampling settings, pilot seeds and run-log schema.

The local task config offers concrete levels/seeds; these decisions should be recorded here or linked from a versioned config before the main comparison. No cluster job, GitHub publication or teammate notification is implied by this handoff.
