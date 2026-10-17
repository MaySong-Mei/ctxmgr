# Project status and handoff

Updated 2026-10-15. Owners confirmed by Ruthvik: Track 1 Ruthvik; Track 2 Hancheng; Track 3 May.

## Current state

This Track 1 branch contains Ruthvik's task/evaluation implementation, tests and handoff documentation. Local validation rerun 2026-10-15: 25 tests pass, Ruff checks/formatting pass; the saved audit covers 64 official instances (16 core levels x seeds 42–45), all passing feasibility and gold-oracle checks. No real-model experiment has run through this implementation.

May's `may/track3-infra` branch is now available (inspected at `98b98a8`): it includes Slurm setup/serving scripts, `docs/cluster_setup.md`, a draft `docs/log_schema.md`, and reported model smoke tests in its project context. Those reports have not been independently rerun by Track 1. The branch is not yet merged into main; Hancheng's harness has not been inspected or integrated. Both tracks modify agent/dependency files: reconcile `AGENTS.md`, `pyproject.toml` and `uv.lock` when integrating, preserving the official benchmark pin and May's serving requirements. May's personal agent preferences should remain distinct from shared team defaults.

The shared repo should carry code, interface contracts, setup instructions, decisions and aggregate results. `AGENTS.md` contains portable team instructions. Each person's private notes may add context but must not be required to run the project.

### Track 2 baseline scaffold (`may/track2-baseline`, May, 2026-10-16)

Branched from this Track 1 branch to unblock the first end-to-end check; Hancheng owns Track 2 and is
free to rework or replace it. Contents: `src/ctxmgr/harness/` with a `ChatAgent` implementing the
ContextBench runner protocol on vLLM's OpenAI API and the no-management condition only (thinking off,
temperature 0, max_tokens 2048); `run_episode` writes `episode.json`, `calls.jsonl` and `score.json`
next to ContextBench's snapshots, as drafted in `docs/log_schema.md` on `may/track3-infra`;
`slurm/baseline_smoke.sbatch` serves one model (`max_len` 40960 for Qwen/o200k headroom) and runs KV R130
and Needle K4-C4 at seed 42. No-management baselines for 0.8B/2B/4B have run (results in `CLAUDE.md` on
`may/track3-infra`; issue #3).

**Draft self/peer management (2026-10-17), to freeze with the team before comparison runs**
(`src/ctxmgr/harness/management.py`, `agents.py`):
- Shared action space for every managed condition: the context is numbered blocks; the editor replies
  with one JSON object of `delete` / `replace` edits; blocks 0–1 (system prompt, task instruction) are
  protected. Same edit instructions text for all conditions.
- Edit contract (hardened after the first managed pilot, PR #4; rules also in the module docstring):
  - `{"op": "delete", "ids": [...]}` or `{"op": "delete", "from": a, "to": b}` (inclusive range, so
    long deletes stay short under the 2,048-token cap); both forms are in the edit instructions. An
    edit with both `ids` and `from`/`to` is rejected.
  - A delete is all-or-nothing: if it names a protected (0/1), missing or non-integer id, the whole
    edit is rejected and reported in `error`. Before, the bad ids were dropped and the rest applied,
    which turned 2B's "delete 0..N" into a wipe of every editable block. Range endpoints are checked
    before the range is expanded.
  - `{"op": "replace", "id": i, "text": "..."}` rewrites one block. An edit with an integer `id` and a
    string `text` but no `op` is read as a replace (4B wrote replaces this way), with the same id
    checks. Any other edit without a known `op` is rejected.
  - Edits in one reply are checked independently: a rejected edit doesn't block the others.
- Trigger: before a turn, when the live context exceeds 18,432 o200k tokens (60% of the 30,720 wall).
- `self`: the actor gets the edit request appended to its own conversation (it already holds the full
  text, so it sees block ids with short previews). `peer_same` / `peer_27b`: a separate manager session
  with its own system prompt gets the whole context rendered as numbered blocks. `peer_same` uses the
  actor's server as a separate session.
- The management exchange is never kept in the actor's context; manager calls are logged with role
  `self_manager` / `manager` in `calls.jsonl`, and edits (sizes before/after, applied ops, parse and
  rejection errors, and the manager's raw reply as `reply`) in `edits.jsonl`. Malformed edit replies
  apply nothing and are logged.
- Open: trigger value; whether self should see full text instead of previews; whether `replace` may
  rewrite user operations; manager output budget (currently the same 2,048 max_tokens; long replaces
  can still hit it); KV R1000 has no way to keep values while dropping SET-BATCH blocks.
- First managed pilot (2B/4B, 2026-10-17) ran on the pre-hardening protocol; results in `CLAUDE.md` on
  `may/track3-infra`.
Tested with fake clients only: `uv run pytest -q` 39 passed, `ruff check .` and `ruff format --check` clean (Bouchet CPU node,
2026-10-17).

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
