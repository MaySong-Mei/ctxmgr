# Pre-registration v7: which part of `para3` makes the Gemma 12B note before deleting?

Status: frozen 2026-10-31 by the commit that pushes it, before the job was submitted; later changes go
to `docs/preregistration-deviations.md`. Harness: `may/manager-replay` at `14b1a23`. The job is round
34 (`r34-sent-g12`), submitted by a script that refuses to run unless the cluster worktree is clean
and at that commit.

## Why

Round 33 (`docs/preregistration-v6.md`, `docs/results.md` finding 11) ruled out the numbered format:
- **Bullets** score the same as numbers.
- **Prose** costs 0.06, not significant.
- **Adding `para3`'s steps to v8** does nothing for the 12B.

What tracks the 12B's later-event success is the body text. It notes before deleting in 52-59 of 128
sampled later replies under the `para3` texts, and in 3-8 under the v8 texts.

This round removes one part of `para3` at a time, and moves `para3`'s opening into v8, to find which
part carries the effect.

## Design

Same inputs, rebuild check, scoring and server settings as rounds 32-33 (`docs/preregistration-v5.md`):
the first three manager events of job 31545119's 16 episodes (48 contexts; "later" = events 2-3, 32
contexts). The manager is Gemma 4 12B alone on one H200 (util 0.90, `max_len` 81,920, 64 sequences,
vLLM 0.30, bf16).

The wordings are built from `para3` and v8's own text (`scripts/manager_replay.py`, harness branch
`may/manager-replay`). Each removes or swaps one part; the remaining lines are verbatim, and later
steps are renumbered (numbering had no effect in round 33):

| wording | change |
|---|---|
| `para3`, v8 | none (in-job controls) |
| `p3noframe` | drop the framing sentence "Your job is to edit it so that there is space again while the assistant can still carry out its task; you are judged on the second part, not on the amount of space freed." |
| `p3nostep1` | drop step 1 (read block 1; work out what must be kept) |
| `p3nostep2` | drop step 2 (judge each block: delete, or keep only needed lines by replacing, or note them and delete) |
| `p3nostep3` | drop step 3 (never delete the only copy; deleting every block is almost certainly wrong) |
| `p3nostep4` | drop step 4 (note the needed lines in this same reply, every reply; the notes block holds only what you write) |
| `p3nostep5` | drop step 5 (write the reply; blocks 0 and 1 cannot be edited; one invalid edit voids all) |
| `v8p3frame` | v8 with its three-line opening replaced by `para3`'s three-line opening (situation and framing) |

`python scripts/manager_replay.py texts <wording>` prints each text.

**Decoding.** Greedy, plus 8 sampled replies per context at Gemma's recommended setting (temperature
1.0, top-p 0.95, top-k 64, request seeds 0-7). Round 33 used 4; S1 there was inconclusive, so the
sample count is doubled. 9 wordings × 48 contexts × 9 calls = 3,888 calls, one job, about 1
H200-hour.

## Predictions

The primary outcome is the 12B's sampled success on later events: 32 contexts × 8 samples = 256
replies per wording. Each wording is compared with `para3` (or v8) by its per-context success share,
with a two-sided sign test over the contexts where the two wordings differ. The six leave-one-out
tests are Holm-adjusted together.

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| T1 | A single part of `para3` carries the effect. | for at least one leave-one-out wording X, `para3` − X ≥ +0.15 with Holm-adjusted p < 0.05 | `para3` − X ≤ +0.05 for every X |
| T2 | Step 4 (the explicit "note in this same reply" step) is necessary. | `para3` − `p3nostep4` ≥ +0.15 with Holm-adjusted p < 0.05 | ≤ +0.03 |
| T3 | `para3`'s opening transfers to v8. | `v8p3frame` − v8 ≥ +0.15 with p < 0.05 | ≤ +0.03 |

Results between the thresholds are recorded as inconclusive. Precondition: the in-job controls'
sampled later-event success is within ±0.10 of round 33 (`para3` 0.35, v8 0.09). If not, only
within-job comparisons are reported, and that is noted.

## Exploratory

- **Mechanism:** the share of replies that note, per wording.
- **Greedy:** the same comparisons on greedy replies (32 per wording).
- **Failure types:** deleting without notes, freeing too little, parse errors, cut replies, invalid
  edits. For `p3nostep5`, check whether invalid edits naming block 0 or 1 rise.
- **Distributed effect:** if no single part is necessary, compare pairs of the smallest losses.

## Not tested

- Pairs or groups of parts removed together.
- Other managers (Qwen 27B and Gemma 31B are already near ceiling, or move in other directions).
- Whole episodes.
