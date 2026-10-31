# Pre-registration v5: identical-input replay of manager events (why the Gemma 12B helps under para3)

Status: frozen 2026-10-31 by the commit that pushes it, before any job was submitted; later changes go
to `docs/preregistration-deviations.md`. Harness: `may/manager-replay` at `b8252b3`. Jobs are round 32
(`r32-mr-*`), submitted by one script that refuses to run unless the cluster worktree is clean and
at that commit.

## Why

In round 31 (`docs/results.md` finding 10) the Gemma 4 12B manager scored +0.48 (K4-C24) and +0.38
(K8-C48) over `none` under `para3`, but −0.02 / +0.02 under v8 and −0.15 / −0.07 under `para4`. In an
episode, every manager event after the first sees a context shaped by its own earlier edits, so
episode rewards cannot tell whether a wording changes what the manager does with a given input, or
only which states it reaches.

A no-GPU check on the logged episodes (2026-10-31) looked at the first manager event. Up to that
event the 2B actor's trajectory is the same in every condition (identical context size in 31 of 32
level × seed groups), so the first events are identical inputs. Success = every needle survives and
at least 30% of the context is freed:

| manager | v8 | `para3` | `para4` | other wordings |
|---|---|---|---|---|
| Qwen3.5 27B (K4 / K8) | 22/22, 4/4 | 8/8, 8/8 | 8/8, 8/8 | 6-8 of 8 |
| Gemma 4 31B | 8/8, 8/8 | 8/8, 8/8 | 8/8, 8/8 | – |
| Gemma 4 12B | 4/8, 0/8 | 4/8, 1/8 | 0/8, 0/8 | – |
| Qwen3.5 9B | 0/8, 0/8 | 0/8, 0/8 | 0/8, 0/8 | 0-2 of 8 |

At the first event the 12B does no better under `para3` than under v8, yet its episodes differ by
+0.50. The difference should therefore be in later events. This probe sends the same later-event
inputs to each manager under each wording.

## Design

- **Inputs:** the first three manager events of each of the 16 episodes of job 31545119 (2B actor
  + Gemma 4 31B, plain edit-v8; Needle K4-C24 and K8-C48, seeds 42-49), 48 contexts in all.
  - Event 1 is the shared first-event input above.
  - Events 2 and 3 are "healthy" mid-episode states: a notes block that holds every earlier needle,
    plus the operation blocks added since.
- **Rebuild:** each context is snapshot t-1 of the event's turn. The manager request is rebuilt
  exactly as `PeerManagedAgent.request_edits` builds it. A local check matches all 48 contexts and
  all 48 v8 requests to the logged `context_before_o200k` and manager `prompt_tokens_o200k`. The job
  repeats this check and refuses to run on any mismatch.
- **Managers:** each one served alone on one H200 (util 0.90, `max_len` 81,920, 64 sequences, vLLM
  0.30, bf16): Gemma 4 12B, Gemma 4 31B, Qwen3.5 9B, Qwen3.5 27B.
- **Wordings:** v8, `para1`, `para2`, `para3`, `para4`.
- **Decoding:** greedy, thinking off, `max_tokens` 4,096, the episodes' manager settings.
- **Sampled repeats (12B only):** v8, `para3` and `para4` at Gemma's recommended setting
  (temperature 1.0, top-p 0.95, top-k 64), 4 request seeds per context. Only the manager is called,
  so the Qwen actor is not affected.
- **Calls:** 240 greedy per manager, plus 576 sampled for the 12B. 4 jobs, about 3-4 H200-hours.
- **Code:** script `scripts/manager_replay.py` and job `slurm/manager_replay.sbatch`, on harness
  branch `may/manager-replay` (based on `may/gemma4-managers` e04f356).

## Measures (per reply)

The reply is parsed with `management.parse_edits` and applied to a copy of the context with
`management.apply_edits`, the same code the episodes use.
- **kept:** the share of the context's needle ids present after the edits.
- **freed:** 1 − o200k after / o200k before.
- **success:** kept ≥ 0.99 and freed ≥ 0.30.
- **Also logged:** parse and apply errors, `_wrapped` slips, replies cut at the cap.
- **Groupings:** "first" = event 1 (16 contexts); "later" = events 2 and 3 (32 contexts).

## Predictions

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| R1 | Greedy replay reproduces the logged first events (precondition). | event-1 success matches the episode's logged first event in ≥ 90% of the (manager, wording) cases that ran in episodes | < 75%; the other tests are then reported as indicative only |
| R2 | The 12B's `para3` advantage lies in later events. | on later events, 12B greedy success under `para3` − under v8 ≥ +0.30 | ≤ +0.05 |
| R3 | Size decides robustly on identical inputs. | Gemma 31B and Qwen 27B succeed in ≥ 0.85 of all 48 contexts under each of the five wordings, and Qwen 9B in ≤ 0.15 under each | 31B or 27B < 0.70 under any wording, or 9B > 0.30 under any wording |

Results between the two thresholds are recorded as inconclusive.

## Decision rules (exploratory)

- **Is the 12B stochastic or input-determined?** Look at its sampled success per context over 4
  samples:
  - if ≥ 80% of contexts are 0/4 or 4/4 under both v8 and `para3`, input-determined;
  - if ≥ 50% are 1/4-3/4, near a threshold, so greedy episode rewards reflect chance as much as
    ability.
- **Do other wordings rescue the 12B?** `para1` and `para2` have not been run with Gemma; report their
  success on first and later events.
- **How do failures happen?** Report the share from parse errors, capped replies, deletes without
  notes, and edits that free too little.

## Not tested

- Which part of `para3` matters. Its distinctive feature is a numbered step procedure (read the task;
  judge each block; check for only copies; note, then write the reply); the probe does not ablate it.
- Contexts reached by other managers. Every input comes from the 31B's own v8 trajectory.
