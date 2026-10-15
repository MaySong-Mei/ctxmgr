# Run log schema (draft, Track 3 → for agreement with Tracks 1/2)

Status: **proposal**. Track 3 (accounting, analysis) consumes these logs; Track 2 writes them; Track 1
supplies the ground-truth fields. Nothing here is final until all three tracks sign off.

Goal: every metric in the proposal can be computed offline from logs, without rerunning jobs.
Format: one JSONL file per (actor size, condition, task, pressure level, seed) episode, one line per event.
Token counts use the model's own tokenizer (`*_tokens`) plus o200k (`*_tokens_o200k`) for cross-size comparison.

## Episode header (first line, `"event": "episode_start"`)

| Field | Type | Notes |
|---|---|---|
| `run_id` | str | Grid run identifier (git SHA + config hash) |
| `episode_id` | str | Unique per episode |
| `task` | str | `kv_store` / `needle_retention` / … (Track 1) |
| `pressure` | str | Pressure level name (Track 1) |
| `seed` | int | Instance seed |
| `condition` | str | `none` / `self` / `peer_same` / `peer_27b` (+ `scheduled_summary` / `oracle` if adopted) |
| `actor_model`, `manager_model` | str | HF ids; `manager_model` null for `none` / `self` uses actor id |
| `actor_precision`, `manager_precision` | str | `bf16` / `fp8` |
| `context_budget_o200k` | int | Fixed task-level budget, same across sizes |
| `manager_trigger_o200k` | int | Context size that triggers management |
| `required_facts` | list | Track 1 ground truth: `{fact_id, text, needed_until_step}`; `needed_until_step` lets us flag destructive edits |

## Model call (`"event": "call"`)

One line per LLM request, actor or manager.

| Field | Type | Notes |
|---|---|---|
| `step` | int | Environment step the call belongs to |
| `role` | str | `actor` / `manager` |
| `model` | str | HF id |
| `prompt_tokens`, `completion_tokens` | int | From vLLM usage |
| `cached_prompt_tokens` | int | Prefix-cache hits, if available |
| `thinking_tokens` | int | Reasoning tokens if thinking is on |
| `latency_s` | float | Wall time |

## Context edit (`"event": "edit"`)

One line per applied edit (self or peer).

| Field | Type | Notes |
|---|---|---|
| `step` | int | |
| `editor` | str | `actor` (self) / `manager` (peer) |
| `op` | str | CLM edit tool name (Track 2) |
| `context_before_o200k`, `context_after_o200k` | int | Size around the edit |
| `facts_removed` | list[str] | `fact_id`s present before and absent after (Track 1 matcher) |
| `facts_still_needed_removed` | list[str] | Subset with `needed_until_step > step` → destructive |

## Step end (`"event": "step"`)

| Field | Type | Notes |
|---|---|---|
| `step` | int | |
| `context_o200k` | int | Context size after the step |
| `required_present` | int | Required facts still needed and present in context |
| `irrelevant_o200k` | int | Tokens not attributable to required facts or instructions (Track 1 estimator) |

## Episode end (`"event": "episode_end"`)

| Field | Type | Notes |
|---|---|---|
| `answer`, `gold` | str | Final answer and ground truth |
| `correct` | bool | Exact-value match (Track 1 scorer) |
| `status` | str | `ok` / `context_overflow` / `max_steps` / `error` |

## Derived metrics (Track 3, `src/ctxmgr/analysis/`)

- Accuracy by size × condition × pressure.
- Peak context = max `context_o200k`; compression ratio = Σ`context_before` / Σ`context_after` over edits.
- Edit count and timing (steps, and context size at trigger).
- Required info kept = `required_present` / needed facts, per step and at the end.
- Destructive edits = count of non-empty `facts_still_needed_removed`.
- Compute = Σ calls by role; FLOPs via `src/ctxmgr/accounting/` (adapted from CLM `flops_metrics/`),
  always reporting actor + manager together.

## Open questions for Tracks 1/2

1. Can Track 1's generator emit `required_facts` with `needed_until_step` for both task families?
2. How does the fact matcher decide presence after a rewrite (exact string, normalized value, or LLM judge)?
3. Does the harness log the full context text per step (large, but enables re-analysis) or only sizes?
