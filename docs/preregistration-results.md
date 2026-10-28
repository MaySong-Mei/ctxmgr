# Pre-registration results (`prereg-v1`)

Runs: jobs 30981378–30981390 (J1–J13), submitted by `slurm/prereg_submit.sh blocks` at harness tag
`prereg-v1-harness` (`33b3997`); all 13 jobs completed every episode (no infrastructure reruns).
Canaries: 30980396/7 (deviation D1 in `docs/preregistration-deviations.md`). Analysis:
`python scripts/prereg_analysis.py`, byte-identical to the `prereg-v1` tag, run once after all jobs
finished. No seed ≥ 50 result was looked at before that run.

## Confirmatory outcome

**7 of 10 predictions confirmed, 3 inconclusive, 0 falsified. Rules R1, R2 and R3 are not
falsified; the 27B extension of R2 (P6) is confirmed.**

| # | Prediction | Outcome | Key numbers |
|---|---|---|---|
| P1 | Needle K4-C24: `peer_27b − none` > +0.30 (2B, 4B, 9B) | confirmed | 16/16, 14/16, 13/16; mean +0.66 / +0.47 / +0.51 |
| P2 | `self − none` < +0.05 (2B, 4B) | confirmed | 8/8 each; mean 0.00 |
| P3 | 9B KV R1000: `mask_arc − none` > +0.70 | confirmed | 8/8; mean +1.00 |
| P4 | 9B KV R1000: \|`mask_arc − peer_27b_arc`\| < 0.10 | inconclusive | 10/16 within; mean +0.12 (rule manager higher) |
| P5 | 4B KV R1000: archive arms − `none` < +0.20 | inconclusive | `mask_arc` 16/16 (mean +0.07); `peer_27b_arc` 10/16 (mean +0.30) |
| P6 | 27B KV R1000: `self_arc − none_r` > +0.70 | confirmed | 8/8; mean +1.00 |
| P7 | 9B Needle: `peer_27b` beats each archive arm by > 0.20; no archive arm beats `none` by > 0.10 | confirmed | 8/8 on all six components; archive arms − `none` −0.13 to −0.21 |
| P8 | 2B `peer_27b` non-inferior to 27B `peer_same` (Δ 0.10) | confirmed | 15/16; mean +0.22 |
| P9 | 9B KV R2000 (held out): archive arms − `none` > +0.50; arms within 0.15 | inconclusive | 7/8 and 6/8 (means +0.88 / +0.73); 5/8 within |
| P10 | Needle K2-C192 (held out): `peer_27b − none` > +0.05 (2B, 9B) | confirmed | 8/8 each; mean +0.33 / +0.21 |

Holm thresholds, p-values, Clopper–Pearson intervals and source jobs: the frozen script's output
below.

## Exploratory (not part of the confirmatory family)

Looked at only after the confirmatory run, to understand the three inconclusive predictions.

- **P4.** 9B KV R1000 means (16 seeds): `none` 0.00, `mask_arc` 0.96 (15/16 at 1.00), `peer_27b_arc`
  0.84 (four seeds at 0.21–0.33). The equivalence fails in the direction of the rule manager being
  *better* than the 27B archive manager, which strengthens rather than contradicts R2's claim that a
  rule suffices.
- **P5.** 4B never RECALLs under `mask_arc` (0 per episode; mean 0.07). Under `peer_27b_arc` it
  RECALLs 4.3 times per episode and reaches 0.62–1.00 on 5 of 16 seeds (mean 0.30). The actor-size
  boundary holds for the rule manager but is not a hard boundary when a 27B manager runs the archive.
- **P9 (held-out R2000).** 9B means (8 seeds): `none` 0.00, `peer_27b` (delete) 0.00, `mask_arc` 0.88
  (one seed at 0.00), `peer_27b_arc` 0.73 (two seeds at 0.00). The effect is large but not unanimous
  at n = 8, so the majority test with Holm did not pass.
- D1 non-determinism applies: individual greedy episodes are near-repeats, not exact repeats.

## Frozen script output

| prediction | component | result | p | p (opposite) | outcome |
|---|---|---|---|---|---|
| P1 | 2B peer_27b-none | d > +0.30: 16/16 (95% CI 0.79-1.00), 0 opposite; mean d +0.661; jobs 30981378 | 0.0000 | 1.0000 | **confirmed** (p=0.0106, Holm 0.0125) |
|  | 4B peer_27b-none | d > +0.30: 14/16 (95% CI 0.62-0.98), 2 opposite; mean d +0.466; jobs 30981379 | 0.0021 | 0.9997 |  |
|  | 9B peer_27b-none | d > +0.30: 13/16 (95% CI 0.54-0.96), 3 opposite; mean d +0.510; jobs 30981380 | 0.0106 | 0.9979 |  |
| P2 | 2B self-none | d < +0.05: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.000; jobs 30981378, 30981381 | 0.0039 | 1.0000 | **confirmed** (p=0.0039, Holm 0.0056) |
|  | 4B self-none | d < +0.05: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.000; jobs 30981379, 30981382 | 0.0039 | 1.0000 |  |
| P3 | 9B mask_arc-none | d > +0.70: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +1.000; jobs 30981384 | 0.0039 | 1.0000 | **confirmed** (p=0.0039, Holm 0.0063) |
| P4 | 9B mask_arc vs peer_27b_arc | \|d\| < 0.10: 10/16 (95% CI 0.35-0.85), 6 opposite; mean d +0.117; jobs 30981384 | 0.2272 | 0.8949 | **inconclusive** (p=0.2272, Holm 0.0167) |
| P5 | 4B mask_arc-none | d < +0.20: 16/16 (95% CI 0.79-1.00), 0 opposite; mean d +0.065; jobs 30981385 | 0.0000 | 1.0000 | **inconclusive** (p=0.2272, Holm 0.0250) |
|  | 4B peer_27b_arc-none | d < +0.20: 10/16 (95% CI 0.35-0.85), 6 opposite; mean d +0.302; jobs 30981385 | 0.2272 | 0.8949 |  |
| P6 | 27B self_arc-none_r | d > +0.70: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +1.000; jobs 30981389 | 0.0039 | 1.0000 | **confirmed** (p=0.0039, Holm 0.0071) |
| P7 | 9B peer_27b-self_arc | d > +0.20: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.719; jobs 30981380, 30981383 | 0.0039 | 1.0000 | **confirmed** (p=0.0039, Holm 0.0083) |
|  | 9B self_arc-none | d < +0.10: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d -0.167; jobs 30981380, 30981383 | 0.0039 | 1.0000 |  |
|  | 9B peer_27b-peer_same_arc | d > +0.20: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.682; jobs 30981380, 30981383 | 0.0039 | 1.0000 |  |
|  | 9B peer_same_arc-none | d < +0.10: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d -0.130; jobs 30981380, 30981383 | 0.0039 | 1.0000 |  |
|  | 9B peer_27b-mask_arc | d > +0.20: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.760; jobs 30981380, 30981383 | 0.0039 | 1.0000 |  |
|  | 9B mask_arc-none | d < +0.10: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d -0.208; jobs 30981380, 30981383 | 0.0039 | 1.0000 |  |
| P8 | 2B peer_27b vs 27B peer_same | d > -0.10: 15/16 (95% CI 0.70-1.00), 1 opposite; mean d +0.224; jobs 30981378, 30981390 | 0.0003 | 1.0000 | **confirmed** (p=0.0003, Holm 0.0050) |
| P9 | 9B mask_arc-none | d > +0.50: 7/8 (95% CI 0.47-1.00), 1 opposite; mean d +0.875; jobs 30981386 | 0.0352 | 0.9961 | **inconclusive** (p=0.3633, Holm 0.0500) |
|  | 9B peer_27b_arc-none | d > +0.50: 6/8 (95% CI 0.35-0.97), 2 opposite; mean d +0.729; jobs 30981386 | 0.1445 | 0.9648 |  |
|  | 9B mask_arc vs peer_27b_arc | \|d\| < 0.15: 5/8 (95% CI 0.24-0.91), 3 opposite; mean d +0.146; jobs 30981386 | 0.3633 | 0.8555 |  |
| P10 | 2B peer_27b-none | d > +0.05: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.332; jobs 30981387 | 0.0039 | 1.0000 | **confirmed** (p=0.0039, Holm 0.0100) |
|  | 9B peer_27b-none | d > +0.05: 8/8 (95% CI 0.63-1.00), 0 opposite; mean d +0.210; jobs 30981388 | 0.0039 | 1.0000 |  |

Rule-level falsification (component opposite-side p < 0.05, unadjusted):
- R1: not falsified
- R2: not falsified; actor-size boundary (P5): not falsified; 27B extension (P6): confirmed
- R3: not falsified
