# Pre-registration v2: round 29 must-runs (27B + rule, 9B manager under sampling)

Status: written 2026-10-30 and pushed before either job was submitted; not edited afterwards
(deviations go to `docs/preregistration-deviations.md`). Both jobs come from one script,
`submit_week1.sh`, recorded in `CLAUDE.md` round 29. It uses the settings of
`slurm/prereg_submit.sh` (harness `may/track2-baseline` at submission, trigger 18,432,
max_len 81,920, `--enable-prompt-tokens-details`, `MAX_TURNS_PER_OP=2`). Needle K4-C24 and K8-C48,
seeds 42-49 (the seeds share each level's layout).

## Jobs

- **J1:** Qwen3.5-27B actor alone (`baseline_smoke.sbatch`, util 0.90), conditions `none` and
  `rule_needle`, greedy.
- **J2:** Qwen3.5-2B actor + co-located Qwen3.5-9B manager (`peer27b_pilot.sbatch`, utils
  0.55/0.38), conditions `none` and `peer_9b` (edit-v8). Decoding is the Qwen3.5 recommended
  non-thinking sampling for both models: temperature 0.7, top-p 0.8, top-k 20, presence penalty
  1.5, request seed = instance seed.

## Predictions

| # | Prediction | Confirmed if | Falsified if |
|---|---|---|---|
| P11 | The rule makes the 27B actor score like the 2B actor with the rule. | mean reward ≥ 0.95 at both levels | mean reward < 0.90 at either level |
| P12 | The 27B actor + rule costs what the 2B rule runs' calls cost with 27B constants: 8.7 PF (K4-C24), 24.1 PF (K8-C48). It is therefore ≈ 1.2× the 2B actor + 27B manager at K4-C24 (7.5 PF) and below the 2B actor + 27B manager with thinking at K8-C48 (28.2 PF). | total PF per episode (`scripts/cost_table.py`) within ±25% of the prediction at both levels | outside ±25% at either level |
| P13 | Recommended sampling does not rescue the 9B manager. | `peer_9b` − `none` (paired by seed, both sampled) ≤ +0.05 at both levels | the difference is > +0.10 with a two-sided sign test p < 0.05 at either level |
| P14 | The 9B manager's failure mechanism is unchanged under sampling: it rarely protects needles. | share of applied manager events with a `note` or `replace` (`scripts/mechanism_table.py`) < 0.10 at both levels | ≥ 0.25 at either level |

Results between the two thresholds are recorded as inconclusive.

## Exploratory (not confirmatory)

- Share of 9B manager replies cut off at the 4,096-token output cap, under sampling vs greedy.
- 27B + rule GPU-seconds against the 2B + 27B manager system.
