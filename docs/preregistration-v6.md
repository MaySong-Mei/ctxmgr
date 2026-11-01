# Pre-registration v6: is `para3`'s numbered procedure what helps the Gemma 12B?

Status: frozen 2026-10-31 by the commit that pushes it, before any job was submitted; later changes go
to `docs/preregistration-deviations.md`. Harness: `may/manager-replay` at `da7578b`. Jobs are round 33
(`r33-abl-*`), submitted by one script that refuses to run unless the cluster worktree is clean and
at that commit.

## Why

Round 32 (`docs/results.md` finding 11) found that the Gemma 4 12B's `para3` advantage lies in later
manager events. With identical inputs, its later-event success was 0.66 under `para3` and 0.06 under
v8 (greedy), or 0.34 and 0.09 under Gemma's recommended sampling (4 samples per context).

`para3` shares almost all its content with v8:
- judge every block;
- keep only copies of needed lines;
- never delete the only copy;
- note in every reply.

What only `para3` has is a numbered five-step procedure ("Work through these steps: 1. … 5. …"). This
probe tests whether that format is what helps.

## Design

The inputs, rebuild check, scoring, server settings and code are the same as round 32
(`docs/preregistration-v5.md`):
- **Inputs:** the first three manager events of job 31545119's 16 episodes, 48 contexts.
- **Measures:** success = every needle id kept and ≥ 30% of the context freed; "later" = events 2-3
  (32 contexts).
- **Code:** `scripts/manager_replay.py` on harness branch `may/manager-replay`, with three new
  wordings built from `para3`'s own text, so no step is reworded:

| wording | what changes | tests |
|---|---|---|
| `para3` | – (control, rerun in the same job) | – |
| v8 | – (control, rerun in the same job) | – |
| `p3flat` | `para3` with the step list rewritten as one paragraph: the same sentences in the same order, without "Work through these steps:", the numbers or step 5's "Write the reply." | is the procedure format necessary? |
| `p3bullets` | `para3` with the numbers replaced by "-" bullets; every other character and line break unchanged | numbering vs any list |
| `v8steps` | v8 with `para3`'s step block (header and five numbered steps, 12 lines verbatim) inserted before the final "Reply with only one JSON object" line | is the procedure sufficient on top of v8? |

The full texts are printed by `python scripts/manager_replay.py texts`.

**Managers.** Each is served alone on one H200 (util 0.90, `max_len` 81,920, 64 sequences, vLLM
0.30, bf16):
- **Gemma 4 12B:** greedy, plus 4 sampled replies per context at Gemma's recommended setting
  (temperature 1.0, top-p 0.95, top-k 64, request seeds 0-3), for all five wordings. 1,200 calls.
- **Qwen3.5 27B:** greedy only, five wordings. 240 calls. Exploratory: under v8 it succeeded in
  0.50 of later events in round 32, and in 0.75 under `para3`.

Two jobs, under 1 H200-hour in total.

## Predictions

The primary outcome is the 12B's sampled success on later events: 32 contexts × 4 samples = 128
replies per wording. Paired comparisons are per context (the share of the 4 samples that succeed),
with a two-sided sign test over the contexts where the two wordings differ.

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| S1 | The numbered procedure is necessary: as prose, `para3` loses its advantage. | `para3` − `p3flat` ≥ +0.15, sign p < 0.05 | `para3` − `p3flat` ≤ +0.03 |
| S2 | The numbered procedure is sufficient: added to v8, it helps. | `v8steps` − v8 ≥ +0.15, sign p < 0.05 | `v8steps` − v8 ≤ +0.03 |

Results between the two thresholds are recorded as inconclusive. Precondition: the in-job greedy
controls reproduce round 32's greedy later-event success (`para3` 0.66, v8 0.06) within ±0.10. If
not, only within-job comparisons are reported, and that is noted.

## Exploratory

- **Numbering vs list:** `p3bullets` vs `para3` and `p3flat`.
- **Greedy:** the same comparisons on greedy replies (32 per wording).
- **Qwen 27B:** does `v8steps` raise its later-event success over v8, and does `p3flat` lower it
  below `para3`?
- **Failure types per wording:** deleting without notes, freeing too little, parse errors, cut replies.

## Not tested

- Effects on whole episodes (no episodes are run).
- Other model families.
- Steps 1-5 one at a time.
