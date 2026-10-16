# Run log schema (draft, Track 3 → for agreement with Tracks 1/2)

Status: **proposal**. Track 2 writes these logs, Track 3 (accounting, analysis) reads them, and Track 1's
contract (`docs/tasks-and-evaluation.md` on `ruthvik/track1-tasks`) is the source of truth for scoring.
Nothing here is final until all three tracks sign off.

Principle: task scores come only from ContextBench's own artifacts. Our logs add what ContextBench does not
record (per-call usage by role, edits, triggers, served-tokenizer counts), keyed so they join to its turns.

## Run directory

One directory per episode = (actor size, condition, family, level, seed), written by `contextbench.run`
(or a custom loop following the same convention), plus our files:

```
runs/<run_id>/<actor>/<condition>/<family>-<level>-s<seed>/
  context_snapshots/turn-*.json   # ContextBench: live messages + o200k tokens per turn (scored)
  result.json                     # ContextBench: incl. terminated_by
  episode.json                    # ours: header below
  calls.jsonl                     # ours: one line per LLM request
  edits.jsonl                     # ours: one line per applied context edit
  score.json                      # Track 1: score_run() output (official + diagnostics)
```

The instance directory (with `gold.json` and `manifest.json`) stays outside the run directory and outside
any model-accessible tool. Generated instances are reused unchanged across all conditions.

## `episode.json`

| Field | Notes |
|---|---|
| `run_id` | Grid run id (git SHA + config hash) |
| `family`, `level`, `seed` | ContextBench named level and seed |
| `instance_sha256` | From the instance `manifest.json`, so a run can't be scored against the wrong instance |
| `condition` | `none` / `self` / `peer_same` / `peer_27b` |
| `actor_model`, `manager_model` | HF ids incl. revision; `manager_model` = actor for `self`, null for `none` |
| `actor_precision`, `manager_precision` | `bf16` / `fp8` |
| `budget`, `reserve` | 32768 / 2048 (o200k) |
| `manager_trigger_o200k` | Context size that triggers management |
| `served_max_len` | vLLM `max_len` for each server (Qwen tokens) |
| `prompt_version`, `sampling` | Method prompt version; temperature / top_p / max_tokens / thinking on-off |
| `vllm_version`, `repo_commit`, `contextbench_commit` | Provenance |

## `calls.jsonl` (one line per LLM request)

| Field | Notes |
|---|---|
| `turn` | ContextBench turn index the call belongs to (matches `turn-*.json`) |
| `role` | `actor` / `manager` |
| `model` | HF id |
| `prompt_tokens`, `completion_tokens` | Served-model (Qwen) counts from vLLM usage |
| `prompt_tokens_o200k` | Same prompt in o200k, to measure the Qwen/o200k ratio |
| `cached_prompt_tokens` | Prefix-cache hits, if reported |
| `reasoning_tokens` | If thinking is on |
| `latency_s` | Wall time |

## `edits.jsonl` (one line per applied edit, self or peer)

| Field | Notes |
|---|---|
| `turn` | Edit applies at the start of this turn, after the previous turn was snapshotted |
| `editor` | `actor` (self) / `manager` (peer) |
| `trigger` | `threshold` / `self_initiated` |
| `op` | CLM edit tool name (Track 2) |
| `context_before_o200k`, `context_after_o200k` | Live context size around the edit |

Destroyed information is not inferred per edit here: Needle loss comes from the final snapshot
(`required_info_loss`), and a missing KV answer does not prove loss if the value was offloaded.

## Derived metrics (Track 3, `src/ctxmgr/analysis/`)

- Accuracy: `score.json` → `official.reward` by size × condition × level; strict diagnostics alongside.
- Completion: `result.json.terminated_by`; budget-wall runs are reported, never dropped or counted as done.
- Peak context: `diagnostics.observed_peak_tokens` (o200k, from snapshots).
- Edits: count, turn of first edit, compression = Σ`context_before` / Σ`context_after`.
- Compute: Σ tokens per role from `calls.jsonl`; FLOPs via `src/ctxmgr/accounting/` (adapted from CLM
  `flops_metrics/`), always actor + manager together.
- Served-length check: Qwen/o200k prompt ratio per model, to size vLLM `max_len`.

## Open questions

1. Track 2: does the harness use `contextbench.run` as-is (then calls/edits are logged from the agent
   wrapper), or a custom loop that must reproduce the snapshot convention?
2. Track 2: how is a manager call attributed to a turn when it runs between turns?
3. All: keep full prompts/completions per call (large, enables re-analysis) or only the counts above?
