# Pre-registration: validating the context-management decision rule

Status: **DRAFT for team review. Not yet frozen.** It is frozen by the git tag `prereg-v1` on the
commit that contains it. No job with seed ≥ 50 may be submitted before that tag exists, and this file
is not edited afterwards (any deviation goes into a separate `docs/preregistration-deviations.md`).

Design rationale: `reports/Validation experiment design check.md`. Results this rule is derived from:
`docs/results.md` (seeds 42–49).

## 1. Hypothesis: the decision rule

Derived from seeds 42–49 under greedy decoding:

- **R1 (in-context retention, Needle Retention):** manager capability decides the outcome. A fixed,
  stronger external manager (27B) helps small actors; same-size management does not.
- **R2 (keyed recall, KV Store):** reversibility decides the outcome. With archive-with-recall, a
  no-LLM rule manager is as good as a 27B manager, provided the actor is large enough to issue RECALL
  (9B yes, 4B no).
- **R3:** archiving does not substitute for manager capability on Needle (no key to recall by; the
  grader reads the final context).

Scope: Qwen3.5 dense models, bf16, greedy decoding, ContextBench Needle Retention and KV Store, the
32,768 o200k budget. Claims about sampled decoding or other model families are out of scope (the 27B
role-separation effect is decoding-sensitive and is **not** part of the confirmatory family).

## 2. Frozen setup

| Item | Value |
|---|---|
| Harness | `may/track2-baseline` commit `23d9a6b` (edit protocol `edit-v8`, archive `edit-v8-arc`, rule `rule-mask-v1`) |
| ContextBench | pinned `fb533c18c78a82d3f1b2047a9c1700735cf829ae` (reply delivery) |
| Serving | vLLM 0.30.0, bf16, one H200 per job (`gpu_h200`), `--max-num-seqs 64`, `PROMPT_TOKENS_DETAILS=1` |
| Small-actor jobs | `slurm/peer27b_pilot.sbatch`: 27B manager util 0.55 + actor util 0.38, `MAX_LEN` = `ACTOR_MAX_LEN` = 81,920 |
| 27B-actor jobs | `slurm/baseline_smoke.sbatch`: util 0.90, `MAX_LEN` 81,920 |
| Management | trigger 18,432 o200k; manager max_tokens 4,096; actor max_tokens 2,048 |
| Turn cap | `max_turns` = 2 × n_ops + 10 |
| Decoding | greedy: temperature 0, top_p 1, thinking off, no `SAMPLING` |
| Seeds | 50–57; 50–65 where marked "16" |

## 3. Metrics and unit of analysis

- Needle: ContextBench **official** reward. KV: **lenient** recall (`scripts/lenient_kv.py`) as primary,
  official as secondary.
- Unit: one generated instance (level × seed), paired across conditions on the same instance.
  Population: "this level's layout × random lexical content" (seeds change words, values and queried
  keys, not the layout; see the design check). Generalization across layouts is tested only by the two
  held-out levels (P9, P10).

## 4. Confirmatory predictions

Notation: `A − B` is the mean paired difference over seeds. δ = minimum effect, Δ = margin.
A prediction with several components passes only if **every** component passes (intersection-union
test: its p-value is the largest component p-value).

| # | Cells (actor, level, seeds) | Prediction | Test |
|---|---|---|---|
| P1 | 2B, 4B, 9B; Needle K4-C24; 16 | `peer_27b − none` > +0.30 for each actor | superiority, δ = 0.30 |
| P2 | 2B, 4B; Needle K4-C24; 8 | `self − none` < +0.05 for each actor | upper bound, δ = 0.05 |
| P3 | 9B; KV R1000; 8 | `mask_arc − none` > +0.70 | superiority, δ = 0.70 |
| P4 | 9B; KV R1000; 16 | `mask_arc` ≡ `peer_27b_arc` | equivalence (TOST), Δ = 0.10 |
| P5 | 4B; KV R1000; 16 | `mask_arc − none` < +0.20 and `peer_27b_arc − none` < +0.20 | upper bound, δ = 0.20 |
| P6 | 27B; KV R1000; 8 | `self_arc − none_r` > +0.70 | superiority, δ = 0.70 |
| P7 | 9B; Needle K4-C24; 8 | for each A in {`self_arc`, `peer_same_arc`, `mask_arc`}: `peer_27b − A` > +0.20 and `A − none` < +0.10 | superiority δ = 0.20; upper bound δ = 0.10 |
| P8 | 2B vs 27B; Needle K4-C24; 16 | `peer_27b` (2B actor) − `peer_same` (27B actor) > −0.10 | non-inferiority, Δ = 0.10 |
| P9 | 9B; **KV R2000** (held-out level); 8 | `mask_arc − none` > +0.50 and `peer_27b_arc − none` > +0.50 | superiority, δ = 0.50 |
| P10 | 2B, 9B; **Needle K2-C192** (held-out level); 8 | `peer_27b − none` > +0.05 for each actor | superiority, δ = 0.05 |

P6, P7 (archive on Needle) and P9–P10 (held-out levels) are out-of-sample conditions; P1–P5 and P8 are
within-layout replications on new seeds. They are reported separately.

### Statistical procedure

- Each component: one-sided exact sign-flip permutation test on the shifted paired differences
  (`d − δ` for "> δ", `δ − d` for "< δ"; `d + Δ` for non-inferiority); all 2^n sign patterns
  (256 at n = 8, 65,536 at n = 16); zero shifted differences count as no evidence (p includes ties).
  TOST: two such tests at ±Δ, p = the larger.
- Multiplicity: Holm across the 10 predictions at family-wise α = 0.05 (one-sided).
- Intervals: report the mean difference with an exact Clopper-Pearson interval on the proportion of
  seeds favouring the prediction; no bootstrap intervals.
- **Outcome per prediction:** *confirmed* if it passes Holm; *falsified* if the mean difference is on
  the wrong side of its threshold **and** the opposite one-sided test has unadjusted p < 0.05;
  otherwise *inconclusive*.
- Secondary: count of confirmed predictions vs Binomial(10, 0.5).
- Power note: at n = 8 the smallest one-sided p is 1/256 = 0.0039, so under Holm only an 8/8 result
  passes. A dry run of `scripts/prereg_analysis.py --dry-run` on seeds 42–49 (the data the rule came
  from, not confirmatory) showed P4 and P5 at 7/8 and therefore inconclusive; both are cheap KV cells
  and were raised to 16 seeds before any seed ≥ 50 existed.

### What would falsify the rule

- R1 is falsified if P1 or P10 is falsified, or if P2 is falsified for both 2B and 4B.
- R2 is falsified if P3, P4 or P9 is falsified; its actor-size boundary is falsified if P5 is.
- R3 is falsified if P7 is falsified (some archive arm makes Needle no longer need a stronger manager).
- P6 tests whether R2 extends to the 27B actor; failing it limits R2 to the 9B actor.

## 5. Cost comparison (P8 and the secondary cost frontier)

Reported per role (actor, manager), three ways:

1. No-cache upper bound: params × (prompt + completion tokens), 2N FLOPs per token.
2. Prefix-reuse FLOPs (CLM Eq. 8–9 with per-size constants from each model's `config.json`), using
   logged `cached_prompt_tokens`; if vLLM does not report them, the approximation from logged prompts
   is used and labelled as such.
3. Measured GPU-seconds per episode (sum of call latencies on one H200).

P8's cost condition, checked descriptively: the 2B + 27B configuration uses ≤ 0.5× the prefix-reuse
FLOPs of the 27B `peer_same` configuration. Baselines are labelled correctly: 27B `none` (single
model), 27B `self` (the model manages its own context), 27B `peer_same` (two 27B sessions).
The reward-vs-cost Pareto frontier over all configurations below is exploratory.

## 6. Runs

| Block | Actor | Level | Conditions | Seeds |
|---|---|---|---|---|
| Canary (first) | 9B | Needle K4-C24 | `peer_27b` | 46 |
| A | 2B, 4B, 9B | Needle K4-C24 | `none`, `peer_27b` | 50–65 |
| A | 2B, 4B | Needle K4-C24 | `self` | 50–57 |
| B | 9B | Needle K4-C24 | `self_arc`, `peer_same_arc`, `mask_arc` | 50–57 |
| C | 9B, 4B | KV R1000 | `none`, `mask_arc`, `peer_27b_arc` | 50–65 |
| D | 27B | KV R1000 | `none_r`, `self_arc`, `mask_arc` | 50–57 |
| E | 27B | Needle K4-C24 | `none`, `self`, `peer_same` | 50–65 |
| F (held-out) | 9B | KV R2000 | `none`, `peer_27b`, `mask_arc`, `peer_27b_arc` | 50–57 |
| F (held-out) | 2B, 9B | Needle K2-C192 | `none`, `peer_27b` | 50–57 |

Canary rule: the seed-46 rerun must reproduce the stored reward (job 30127437, 9B `peer_27b`
K4-C24 s46, edit-v8: official reward 0.8333) and log non-null `cached_prompt_tokens`. If the reward differs, stop and
investigate before any block runs. If only the cached-token field is null, drop
`PROMPT_TOKENS_DETAILS` and continue (cost item 2 then uses the approximation).

Estimated cost ≈ 30–40 H200-hours (design check, section "H200-hour budget"); blocks run in parallel.

## 7. Exclusions and stopping

- Every episode is scored as it ends, including `max_steps`, `budget_wall` and `server_rejected`;
  counts are reported. A cell is rerun only for an infrastructure failure before turn 1 (job killed,
  node failure, server never healthy), with the same seed and code.
- No seeds are added after any confirmatory result is seen. Seeds 58–65 are part of the design for the
  cells marked 16, not an optional extension.
- No seed ≥ 50 has been run for any condition (checked in the local `runs/` copy and to be re-checked
  on the cluster before tagging); smoke tests and the canary use seeds 42–49.
- The analysis script (`scripts/prereg_analysis.py`) is committed before the tag and run unchanged.
