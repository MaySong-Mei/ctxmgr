# Pre-registration: validating the context-management decision rule

Status: frozen by the git tag `prereg-v1` on the commit that contains this file (repository branch
`may/track3-infra`) together with the tag `prereg-v1-harness` on `may/track2-baseline`. No job with
seed ≥ 50 is submitted before both tags exist. This file and `scripts/prereg_analysis.py` are not
edited afterwards; any deviation goes into `docs/preregistration-deviations.md`.

Design rationale: `reports/Validation experiment design check.md`. The rule is derived from
`docs/results.md` (seeds 42–49). An adversarial review of the draft (blocking issues: test validity,
unfrozen job settings, rerun rules, R2 tested only at ceiling, analysis outputs) was addressed before
freezing.

## 1. Hypothesis: the decision rule

Derived from seeds 42–49 under greedy decoding:

- **R1 (in-context retention, Needle Retention):** a fixed, stronger external manager (27B) helps
  small actors; same-size self-management does not.
- **R2 (keyed recall, KV Store):** with archive-with-recall, a no-LLM rule manager does as well as a
  27B manager, provided the actor issues RECALL (9B does, 4B does not).
- **R3:** on Needle, the archive arms (self, same-size peer, rule) neither come close to the 27B
  manager (delete) nor improve on no management.

Scope: Qwen3.5 dense models, bf16, greedy decoding, ContextBench Needle Retention and KV Store,
32,768 o200k budget. Sampled decoding and other model families are out of scope; the 27B
role-separation effect (decoding-sensitive) is not in the confirmatory family.

## 2. Frozen setup

| Item | Value |
|---|---|
| Harness | `may/track2-baseline` tag `prereg-v1-harness` = commit `7956662` (`edit-v8`, `edit-v8-arc`, `rule-mask-v1`) |
| Job submission | `slurm/prereg_submit.sh canary` then `blocks`, run from the cluster worktree checked out detached at the tag; the script refuses any other HEAD or modified tracked files, and sets every job knob explicitly |
| ContextBench | `fb533c18c78a82d3f1b2047a9c1700735cf829ae`, reply delivery |
| Serving | vLLM 0.30.0, bf16, one H200 per job (`gpu_h200`), `--max-num-seqs 64`, `--enable-prompt-tokens-details` |
| Small-actor jobs | `peer27b_pilot.sbatch`: 27B bf16 manager at util 0.55 + actor at 0.38, max_len 81,920 for both |
| 27B-actor jobs | `baseline_smoke.sbatch`: util 0.90, max_len 81,920 |
| Management | trigger 18,432 o200k; manager max_tokens 4,096; actor max_tokens 2,048 |
| Turn cap | `max_turns` = 2 × n_ops + 10 |
| Decoding | greedy: temperature 0, top_p 1, thinking off |
| Seeds | 50–57 ("8") or 50–65 ("16") |

The analysis script enforces this in confirmatory mode: it aborts unless every episode used records
harness commit `7956662`, the ContextBench commit, bf16, max_len 81,920, the greedy sampling block,
the actor and manager models implied by its condition, the edit version of its condition, trigger and
manager max_tokens (managed conditions), and `max_turns` = 2 × n_ops + 10, and unless both episodes of
every pair ran the same instance (`instance_sha256`).

## 3. Metric, unit of analysis, test

- Score: Needle **official** reward; KV **lenient** recall (`scripts/lenient_kv.py`); KV official is
  reported but not tested.
- Unit: one generated instance (level × seed), paired across conditions. Seeds change words, values
  and queried keys, not the level's layout, so P1–P8 generalize to "this layout with new content";
  only P9–P10 (held-out levels) test a new layout.
- **Test.** Every prediction component states that the paired difference d = A − B lies beyond a
  threshold **on a majority of instances**. Exact one-sided binomial sign test: k = number of
  instances strictly on the predicted side (ties count against), p = P(Binomial(n, ½) ≥ k). The mean d
  and the exact Clopper–Pearson 95% interval for k/n are reported descriptively. (A sign-flip test on
  the mean was dropped: it assumes symmetric differences, and bounded, skewed scores inflated its
  false-positive rate in the review's simulation.)
- A prediction with several components passes only if every component passes (intersection-union;
  p = largest component p). Holm step-down across the 10 predictions, one-sided α = 0.05.
- **Outcome.** *Confirmed*: passes Holm. *Falsified*: some component has a significant opposite
  majority (binomial p < 0.05 for the count strictly on the wrong side; unadjusted across about 20
  components, which makes falsification the lenient side). Otherwise *inconclusive* (or *incomplete*
  if episodes are missing).
- Power: n = 8 gives a minimum p of 1/256 = 0.0039, so only 8/8 can pass Holm; n = 16 needs about
  14/16 for the first Holm step. The weakest prediction is P1 for 9B (6/8 beyond +0.30 on seeds
  42–49). In a dry run on seeds 42–49 (not confirmatory) P4 and P5 were 7/8, so both were raised to
  16 seeds before any seed ≥ 50 was run.

## 4. Confirmatory predictions

"d > t on a majority" is written d > t; δ = threshold, Δ = margin.

| # | Cells (actor; level; seeds) | Prediction (each on a majority of instances) |
|---|---|---|
| P1 | 2B, 4B, 9B; Needle K4-C24; 16 | `peer_27b − none` > +0.30, for each actor |
| P2 | 2B, 4B; Needle K4-C24; 8 | `self − none` < +0.05, for each actor |
| P3 | 9B; KV R1000; 8 | `mask_arc − none` > +0.70 |
| P4 | 9B; KV R1000; 16 | \|`mask_arc − peer_27b_arc`\| < 0.10 |
| P5 | 4B; KV R1000; 16 | `mask_arc − none` < +0.20 and `peer_27b_arc − none` < +0.20 |
| P6 | 27B; KV R1000; 8 | `self_arc − none_r` > +0.70 |
| P7 | 9B; Needle K4-C24; 8 | for A in {`self_arc`, `peer_same_arc`, `mask_arc`}: `peer_27b − A` > +0.20 and `A − none` < +0.10 |
| P8 | 2B and 27B; Needle K4-C24; 16 | `peer_27b` (2B actor) − `peer_same` (27B actor) > −0.10 |
| P9 | 9B; **KV R2000** (held out); 8 | `mask_arc − none` > +0.50, `peer_27b_arc − none` > +0.50, and \|`mask_arc − peer_27b_arc`\| < 0.15 |
| P10 | 2B, 9B; **Needle K2-C192** (held out); 8 | `peer_27b − none` > +0.05, for each actor |

P1–P5 and P8 are within-layout replications on new content; P6 (27B actor with archive), P7 (archive
on Needle) and P9–P10 (held-out levels) are out-of-sample conditions. Several replications are
expected to pass easily (`none` is constant at the floor or the arms are at ceiling, e.g. P2, P3, P8);
they guard against regressions, and the informative predictions are P1 (9B), P4, P6, P7, P9 and P10.

### What would falsify the rule

- R1: P1 or P10 falsified, or P2 falsified for both 2B and 4B.
- R2: P3, P4 or P9 falsified. Its actor-size boundary: P5 falsified. P6 tests whether R2 extends to
  the 27B actor; failing it restricts R2 to the 9B actor.
- R3: P7 falsified.

The script prints these rule-level verdicts as worded here.

## 5. Cost (exploratory)

Not part of the confirmatory family. Reported per role (actor, manager): (1) no-cache upper bound,
params × (prompt + completion tokens); (2) prefix-reuse FLOPs (CLM Eq. 8–9 with per-size constants
from each `config.json`) from the logged `cached_prompt_tokens`; (3) GPU-seconds per episode. Plotted
as reward vs cost for every configuration run here, with baselines labelled: 27B `none` (single
model), 27B `self` (the model manages its own context), 27B `peer_same` (two 27B sessions).

## 6. Runs

All submitted by `slurm/prereg_submit.sh`.

| Job | Actor | Level | Conditions | Seeds | Used by |
|---|---|---|---|---|---|
| C1 (canary) | 9B | Needle K4-C24, KV R1000 | `peer_27b`, `mask_arc` | 46 | canary |
| C2 (canary) | 27B | Needle K4-C24 | `none` | 46 | canary |
| J1–J3 | 2B, 4B, 9B | Needle K4-C24 | `none`, `peer_27b` | 50–65 | P1, P7, P8 |
| J4–J5 | 2B, 4B | Needle K4-C24 | `self` | 50–57 | P2 |
| J6 | 9B | Needle K4-C24 | `self_arc`, `peer_same_arc`, `mask_arc` | 50–57 | P7 |
| J7 | 9B | KV R1000 | `none`, `mask_arc`, `peer_27b_arc` | 50–65 | P3, P4 |
| J8 | 4B | KV R1000 | `none`, `mask_arc`, `peer_27b_arc` | 50–65 | P5 |
| J9 | 9B | KV R2000 | `none`, `peer_27b`, `mask_arc`, `peer_27b_arc` | 50–57 | P9 |
| J10–J11 | 2B, 9B | Needle K2-C192 | `none`, `peer_27b` | 50–57 | P10 |
| J12 | 27B | KV R1000 | `none_r`, `self_arc`, `mask_arc` | 50–57 | P6 |
| J13 | 27B | Needle K4-C24 | `none`, `self`, `peer_same` | 50–65 | P8 |

Cells run but not tested (27B `mask_arc` and `none`/`self`, 9B `peer_27b` at R2000, seeds 58–65 of
J7/J8 outside P3) feed the exploratory cost frontier only.

**Canary rule.** C1 and C2 must reproduce the stored greedy results exactly: 9B `peer_27b` Needle
K4-C24 s46 official 0.8333 (job 30127437), 9B `mask_arc` KV R1000 s46 lenient 1.0000 (job 30939997),
27B `none` Needle K4-C24 s46 official 0.2917 (job 30136264); and log non-null `cached_prompt_tokens`.
If a reward differs, the blocks are not submitted until the cause is found and recorded as a
deviation. If only the cached-token field is null, cost item (2) uses the logged-prompt approximation.

## 7. Exclusions, reruns, stopping, leakage

- Every episode that ends is scored as it ended (`stream_exhausted`, `submit`, `budget_wall`,
  `max_steps`, `server_rejected`); one that ends without a score file (e.g. rejected on turn 1)
  scores 0. Counts per ending are reported.
- An episode stopped by infrastructure (job killed, time limit, node or server failure) at any turn
  leaves no `episode.json`; it is rerun once, same seed and code, as a job for the missing cells only.
  If the rerun also fails, the prediction using it is reported *incomplete*.
- No seeds are added after any confirmatory result is seen; the seed sets above are final.
- Leakage: no episode with seed ≥ 50 has been run. During the design review, instances for seeds
  50–57 were generated locally and their structure (op counts, needle positions, queried keys) was
  inspected; no model was run on them.
- The analysis is `scripts/prereg_analysis.py` at the tagged commit, run unchanged.
