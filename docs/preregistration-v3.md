# Pre-registration v3: replay probe for the 27B actor's output-cap loop

Status: frozen 2026-10-31 by the commit that pushes it, before the job was submitted; later changes go
to `docs/preregistration-deviations.md`. Harness: `may/track2-baseline` at `51b6825`. Submitted from
the cluster worktree on that commit with

    sbatch --partition=gpu_h200 --qos=normal --gpus=h200:1 --mem=160G --cpus-per-task=8         --job-name=r30-replay slurm/replay_probe.sbatch

(defaults: `MODEL=Qwen/Qwen3.5-27B`, `SRC=<project>/runs/baseline-31451066/Qwen3.5-27B`,
`MAX_LEN=81920`, `GPU_UTIL=0.90`, `MAX_NUM_SEQS=64`).

## Why

Round 29 falsified P11 (`docs/preregistration-v2.md`): with the rule, the 27B actor scored 0.64 at
K4-C24 (7 of 8 seeds at 0.58) and 0.79 at K8-C48 (2 of 8 seeds near 0.15). Post hoc, from the
logs: the 27B restates every needle received so far in each reply; once the rule keeps the context
small, the reply reaches the 2,048-token output cap before
`READY_FOR_NEXT_OP`, the runner's reminder yields the same cut reply, and the episode loops to the turn
cap. The termination audit (`docs/tables/termination_needle.md`) finds the same signature in 31 of 136
managed 27B-actor episodes, 22 of them before round 29. Two adversarial reviews left these
explanations open: the cap binds, or the reply degenerates; the loop is kept alive by the actor's own previous reply (which the rule keeps), or
by its policy; restating comes from a belief that messages do not persist, or from a hedge against
deletion, or from reading the instruction; greedy decoding makes the loop absorbing, or not. This probe
separates them without running episodes.

## Design

One job, one H200 (`gpu_h200`), Qwen3.5-27B bf16, vLLM 0.30, `max_len` 81,920, `--max-num-seqs` 64:
the round-29 server settings. Script `scripts/replay_probe.py` and job `slurm/replay_probe.sbatch` on
harness branch `may/track2-baseline`. Inputs are job 31451066's `context_snapshots`: the input of actor
call t is snapshot t without its last (assistant) message. No new episodes run.

Probe points per `rule_needle` episode with a capped reply: **a** = the first capped call (K4: op 14;
K8: op 7); **b** = the next call (input ends with the capped reply and the runner's reminder; K4 loops
there, K8 escaped there in round 29); **deep** = the sixth call after **a** when all calls in between
were capped. Defaults: greedy, thinking off, `max_tokens` 2,048, as in the episodes.

| Arm | Points | Change | Calls |
|---|---|---|---|
| R1 exact | a, b, deep (K4, K8) | none; one call at a time, before every other arm | 42 |
| R2 cap 8K | a, b (K4, K8) | `max_tokens` 8,192 | 32 |
| R3a short previous reply | b (K4) | the capped reply is replaced by one that restates only the current op's needles (the 2B's format), still without the marker | 8 |
| R3b no previous reply | b (K4) | the capped reply and the reminder are removed; the input ends with the notes block and the current op | 8 |
| R4 disclosure rollouts | `none` K4-C24, turns 1-4 | 4-turn greedy rollouts from the logged ops with the instruction unchanged (D0) or with one sentence appended: D1 "messages in this conversation stay in your context until something removes them"; D2 "a context manager may remove old messages; before it does, it copies every needle line into a notes block, so no needle is lost"; D3 "do not repeat needle lines from earlier operations in your replies" | 8 seeds × 4 arms × 4 |
| R5a sampled | b (K4) | temperature 0.7, top-p 0.8, top-k 20, request seeds 0-7 | 64 |
| R5b sampled + presence penalty | b (K4) | as R5a plus presence penalty 1.5 | 64 |
| R6 cap reminder | b, deep (K4) | the runner's reminder is replaced by one that names the output cap and asks for a short reply ending with the marker | 16 |

In total 362 calls (128 of them in R4 rollouts); time limit 2 h, expected under 1 h including the
cold start of about 17 min.

Measures per reply: `finish_reason`, completion tokens, marker on a line of its own
(`contextbench.stream.is_ready_line`), needle lines, and which ops' needles it restates. "Escape" = the
marker on its own line within 2,048 tokens. "Cumulative" = restates needles of an op earlier than the
latest op it mentions. Counts below are over K4-C24 points (8 seeds) unless stated.

## Predictions

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| Q1 | Exact replay reproduces the episodes (precondition for the rest). | R1 matches the logged `finish_reason` in ≥ 90% of calls | < 75% (then every other arm is reported as indicative only) |
| Q2 | The cap binds: given room, the reply ends normally. | R2 at K4 a and b: ≥ 13/16 end with the marker after 2,049–4,096 tokens | ≥ 8/16 reach 8,192 tokens or stop without the marker (degeneration) |
| Q3 | Restating is the actor's policy, not carried by its previous reply: with no previous reply it rebuilds the full list from the notes. | R3b: ≥ 6/8 are cut at the cap and cumulative | ≥ 6/8 escape |
| Q4 | Restating is instruction-controllable. | R4 turn 4: D0 cumulative in ≥ 7/8 and D3 cumulative in ≤ 2/8 | D3 cumulative in ≥ 6/8 |
| Q5 | Greedy decoding is not what makes the loop absorbing. | R5a escape rate ≤ 0.30 | ≥ 0.60 |

Results between the two thresholds are recorded as inconclusive.

## Decision rules (exploratory arms)

- **R3a:** ≥ 6/8 escape → the long previous reply sustains the loop (fix on the rule side: do not keep
  the actor's previous reply); ≥ 6/8 cut → policy.
- **R4 D1 / D2** ("stops" = cumulative in ≤ 2/8 at turn 4, "does not stop" = ≥ 6/8): only D1 stops →
  a belief that messages do not persist; only D2 stops → a hedge against deletion (candidate A, the
  actor does not trust the manager); both stop → either piece of information suffices; neither →
  not information-driven.
- **R5b:** reported only; the penalty also acts on restating itself.
- **R6:** ≥ 13/16 escape → a runner-side fix. It changes the upstream protocol, so any episode run with
  it is a benchmark deviation and is labelled so.

## What follows

The fix with the least cost to comparability that removes the loop in the probe (in order: the D3
actor instruction, then a rule-side change, then the runner's reminder) is run as full episodes,
27B + rule at K4-C24, seeds 42-49, pre-registered separately as P11b before submission. It cannot
rescue P11, which stays falsified as registered.

## Exploratory (not confirmatory)

- K8 b under R1: does the id-range switch reproduce; R2 at K8.
- Reply length per op under D0 against the round-29 slope (~148 tokens/op at K4).
