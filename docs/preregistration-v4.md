# Pre-registration v4: second manager family (Gemma 4) on Needle

Status: frozen 2026-10-31 by the commit that pushes it, before any ladder job was submitted; later
changes go to `docs/preregistration-deviations.md`. Jobs are round 31 (`r31-g12-*`, `r31-g31-*`),
submitted by one script that refuses to run unless the cluster worktree is clean and at harness
`e04f356`. Every job calls `slurm/peer27b_pilot.sbatch` with the settings listed under Design.

## Why

The Qwen3.5 ladder (`docs/results.md` findings 1, 7, 9) says that without thinking only the largest
manager (27B) helps a small actor on Needle, and that the gain tracks how often a manager keeps needles
before deleting (protective share; Spearman 0.67 / 0.62). Both claims rest on one model family. This
ladder asks whether they hold for a second family, Gemma 4, at two sizes.

## Design

- **Actor:** Qwen3.5-2B, greedy, thinking off (as in every Qwen manager cell).
- **Managers:** `google/gemma-4-12B-it` (`peer_12b`) and `google/gemma-4-31B-it` (`peer_31b`), bf16,
  greedy, thinking off, `manager_max_tokens` 4,096.
- **Wordings:** edit-v8, `para3`, `para4` (fixed before the smoke tests; the Qwen 27B scored +0.66,
  +0.67 and +0.17 with them at K4-C24).
- **Task:** Needle K4-C24 and K8-C48, seeds 42-49 (one layout per level; seeds change words and values).
- **`none`:** 2B actor alone, rerun in the v8 jobs (16 episodes per job).
- **Settings:** those of rounds 25-29: trigger 18,432, `max_len` 81,920, `MAX_TURNS_PER_OP=2`,
  `--max-num-seqs` 64, `--enable-prompt-tokens-details`, one H200 per job. The manager and actor
  share it: 12B at 0.55 / 0.38, 31B at 0.60 / 0.33 (31B: 1.14x concurrency at 81,920 in the smoke).
- **Harness:** `may/gemma4-managers` at `e04f356` (PR #18, under review). It adds `peer_12b` and
  `peer_31b`, the `-it` size check, and reads `{"edits": {<one edit>}}` as a one-edit list, marked
  `_wrapped`. This changes no earlier Qwen event, and a Qwen 27B canary reproduced 1.00 / 1.00 on
  seeds 46-47.
- **Jobs:** 6, one per manager x wording, 16 episodes each (32 for v8 with `none`). About 10
  H200-hours in total.

**Already seen before writing this.** Smoke tests on seed 42 with edit-v8 (jobs 31540581, 31543129):
- 31B: 1.00 (K4-C24) and 0.625 (K8-C48).
- 12B: 0.21 and 0.15.

Seeds share each level's layout, so the v8 rows are largely known. The tests below take the median
over the three wordings. That median is the v8 value only if v8 falls between `para3` and `para4`,
which have not been run with Gemma.

## Measures

- **Reward:** official Needle reward per episode.
- **Paired difference:** the cell's per-seed reward minus that seed's mean greedy 2B `none` reward.
  The mean pools the rerun with all earlier greedy 2B `none` episodes on that seed, as
  `scripts/paired_table.py` does.
- **Protective share:** the share of applied manager events with a `note` or `replace`
  (`scripts/mechanism_table.py`).
- **Median over wordings:** the median of the three wordings' mean paired differences, per manager and
  level.
- **Sign test:** two-sided, over the 8 seeds, per cell.

## Predictions

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| G1 | The 31B manager helps at low pressure. | K4-C24: median ≥ +0.30, and sign p < 0.05 in ≥ 2 of 3 wordings | median ≤ +0.05 |
| G2 | The 31B manager helps at high pressure too. | K8-C48: median ≥ +0.10 | median ≤ +0.02 |
| G3 | The 12B manager does not help. | median ≤ +0.05 at both levels | median > +0.15 with sign p < 0.05 in ≥ 2 wordings, at either level |
| G4 | The mechanism transfers: gain tracks protective share across both families. | pooled Spearman(protective share, gain) over all Qwen and Gemma cells with n ≥ 4 stays ≥ 0.55 at both levels | < 0.40 at either level |
| G5 | Within Gemma, size orders both protection and reward. | the 31B is above the 12B in protective share and in reward in ≥ 5 of the 6 wording x level pairs | ≤ 3 of 6 for either measure |

Results between the two thresholds are recorded as inconclusive.

## Exploratory (not confirmatory)

- **Actor loops.** In both smoke tests, the 2B actor looped on replies cut at 2,048 tokens after
  Gemma's compaction. Reported per cell with `scripts/termination_audit.py` (`max_steps`, cap loops,
  delivered x kept), together with how much context the manager leaves after each event. Not a
  reason to exclude episodes.
- **Manager reply quality.** Parse failures, `_wrapped` slips and replies cut at the 4,096 cap, per cell.
- **Compute.** GPU-seconds only. `scripts/cost_table.py` has no FLOPs term for Gemma's sliding-window
  layers and skips these episodes.

## Not run, and why

- **12B under Gemma's recommended sampling.** The harness applies one sampling setting to both actor
  and manager, so this cell would also sample the Qwen 2B actor at temperature 1.0. Running it cleanly
  needs manager-only sampling, which is another harness change.
- **Gemma thinking, E4B and the 26B-A4B MoE.** Left out to keep the ladder to two dense sizes.
