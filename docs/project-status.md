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
  - `{"op": "replace", "id": i, "text": "..."}` rewrites one block.
  - A missing `op` is inferred when unambiguous: `id` + `text` → replace (4B wrote replaces this way),
    `ids` or `from`/`to` without `text` → delete, with the same checks. Inferred edits are logged with
    `"inferred_op": true` so protocol slips stay countable (issue #3). Anything else is rejected.
  - Negative ids are invalid (reported as bad ids, not as protected blocks).
  - **edit-v3 (2026-10-17): a reply is all-or-nothing.** If any edit in it is invalid, none is applied
    (`error` starts with "rejected whole reply"). Was: edits checked independently, which in the
    27B pilot applied "delete 2–11" while rejecting the paired "copy the needles into block 1".
  - **`{"op": "note", "text": ...}`** writes to a notes block kept right after block 1 (created on
    first use), a legal place to consolidate what must be kept (edit-v3). **edit-v4: notes append**;
    in v3 each note rewrote the whole block, and once it held 53+ needles the rewrite hit the 2,048
    output cap and was rejected. Replacing the notes block by id prunes it (it keeps its header).
  - Conflicts in a valid reply are resolved, not rejected: a block both replaced and deleted keeps its
    replacement; a block replaced twice (or several notes) keeps the last text. Both are reported in
    `error`, and `applied` lists only what took effect.
  - **edit-v5 (2026-10-20): the notes block is kept by the protocol.** Deletes skip it (a range over
    it skips it), a replace of it is read as a note, and notes only add lines it does not hold yet.
    The context view labels it `notes, always kept`. Reason: 27B kept rebuilding the whole notes
    block in v4, which hit the 4,096 output cap at 384 needles (K8-C48).
  - **edit-v6 (2026-10-20):** same semantics as v5; the instructions now say to note the needed lines
    from every block before deleting it (v5's "never copy the existing notes again" made 27B stop
    noting after the first event).
  - **edit-v7 (2026-10-20):** v6 semantics; the context label is just `notes` and the instructions say
    the notes block holds only what the editor writes and a deleted block is gone. Under v5/v6 27B
    wrote one note and then only deleted ("always kept" read as "saved for me").
  - `EDIT_PROMPT_VERSION` (now `edit-v7`) is logged in `episode.json` for every managed episode.
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
  rewrite user operations; manager output budget (2026-10-19: edit requests get their own `manager_max_tokens`
  = 4096, actor stays 2,048; logged in `episode.json`; a server rejection of an edit request is a
  failed event, not the end of the episode); KV R1000 has no way to keep values while dropping SET-BATCH blocks.
- First managed pilot (2B/4B, 2026-10-17) ran on the pre-hardening protocol; results in `CLAUDE.md` on
  `may/track3-infra`.
Tested with fake clients only: `uv run pytest -q` 53 passed, Ruff check/format clean (local, edit-v5,
2026-10-20).

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

## Efficiency follow-up (Ruthvik, 2026-10-27)

Dedicated `ruthvik/efficiency-pilot` branch combines the current harness and published results, retaining the harness dependency lock/server helpers on merge conflicts. Added elapsed-time instrumentation without changing benchmark/edit semantics, historical per-role token/latency extraction, controlled cache-reset timing jobs and an exploratory HotpotQA-derived streaming QA pilot. Detailed design, sample selection, metrics and limitations: `docs/efficiency-pilot.md`. Submission requires passing setup/tests and a smoke job; full jobs run at most two concurrently. Historical artifacts and cached weights are read from the team's normal group-readable directories without modifying them. No completed-run claim is implied here; inspect the submission manifest and Slurm.
