# Results so far

Who should manage the context? Qwen3.5 actors (0.8B–27B, bf16, vLLM 0.30 on one H200) on ContextBench
Needle Retention and KV Store under a 32,768-token o200k budget. Managers edit numbered context blocks
(delete / replace / note, edit protocol v8) once the context passes 18,432 o200k tokens. Conditions:
`none`, `self` (the actor edits its own context), `peer_same` (a separate session of the same model)
and `peer_27b` (a fixed Qwen3.5-27B manager). Managers are untrained (zero-shot). Each cell has 8 seeds
(42–49) unless noted; differences are paired on the same instance.

Sources: `docs/tables/episodes_vllm.csv` (2,360 episodes), `docs/tables/h200_bf16_pooled.md` (greedy
grid), `docs/tables/decoding_ablation.md` (greedy vs sampled), `docs/figures/`. Run log: `CLAUDE.md`.

## Findings

1. **A fixed 27B manager helps small actors on Needle.** `peer_27b` - `none` at K4-C24:

   | actor | greedy | sampled |
   |---|---|---|
   | 0.8B | +0.64 (8/0) | +0.23 (5/1, p=0.22) |
   | 2B | +0.67 (8/0) | +0.43 (8/0) |
   | 4B | +0.46 (8/0) | +0.44 (8/0) |
   | 9B | +0.48 (8/0) | +0.33 (8/0) |

   At K8-C48 (greedy) the gain is +0.11 to +0.17 (8/0 for every size). Under sampling the 0.8B actor
   degenerates (130/600 actor calls hit the 2,048-token output cap, none under greedy).

2. **Same-size management does not help small actors.** `self` and `peer_same` equal `none` for
   0.8B–4B and lower it for 9B (Needle K4-C24: `self` −0.16 greedy / −0.17 sampled, 0/8 seeds).

3. **Edit-based management hurts KV Store once it fires.** At R520, lenient recall drops from
   0.98–1.00 (`none`) to 0.03–0.28 for every manager, size and decoding (4B and 9B, 0/8 seeds each).
   At R130 / R260 (tested with 4B and 9B, `peer_27b`, 4 seeds) the trigger is never reached and
   managed runs equal `none`.

4. **27B actor: role separation depends on decoding and pressure.** `peer_same` - `self`:

   | level | greedy | sampled |
   |---|---|---|
   | Needle K4-C24 | +0.48 (8/0) | +0.06 (4/4) |
   | Needle K8-C48 | +0.20 (8/0) | +0.12 (6/1) |
   | Needle K8-C60 (n=4, greedy only) | +0.18 (4/0) | |
   | Needle K4-C96 (n=4, greedy only) | +0.16 (4/0) | |

   Under sampling `self` rises at K4-C24 (0.38 → 0.60) while `peer_same` falls (0.85 → 0.67).

5. **On KV, the harm comes from irreversible deletion, not from management.** With archive-with-recall
   (removed blocks are archived; the actor writes `RECALL <key>` to bring lines back into context),
   greedy, 8 seeds, lenient KV:

   | actor | level | none | delete arms | archive arms (self / peer_same / peer_27b / rule) |
   |---|---|---|---|---|
   | 9B | R520 | 0.98 | 0.03–0.28 | 0.91 / 0.94 / 0.91 / 0.99 |
   | 9B | R1000 | 0.00 | 0.08–0.23 | 1.00 / 1.00 / 0.97 / 0.99 |
   | 4B | R520 | 1.00 | 0.09–0.28 | 0.48 / 0.54 / 0.51 / 0.45 |
   | 4B | R1000 | 0.00 | 0.05–0.23 | 0.11 / 0.17 / 0.09 / 0.06 |

   For 9B, archiving makes management harmless at R520 and decisive at R1000 (+0.97 to +1.00 over
   `none`, 8/0), and a no-LLM rule (archive the oldest blocks) does as well as any LLM manager. 4B
   recalls on only part of the GETs at R520 and never at R1000 (it invents a hash instead), so the
   limit moves to the actor's retrieval initiative. The 2B actor's ANSWER format breaks when the
   archive paragraph is added to its instruction (all archive arms score 0), so 2B is excluded.

6. **The decision rule held up under pre-registration.** Ten predictions derived from findings 1–5
   were committed (`docs/preregistration.md`, tag `prereg-v1`) before running new seeds (50–65) and
   two held-out levels. **7 confirmed, 3 inconclusive, 0 falsified**; no rule was falsified
   (`docs/preregistration-results.md`). Confirmed: the 27B manager's Needle gain for 2B/4B/9B (also at
   the held-out K2-C192 level), small-model self-management not helping, the rule archive manager
   rescuing 9B KV R1000, the 27B actor's archive arm on KV, archive arms not replacing the 27B manager
   on Needle, and 2B + 27B manager being no worse than 27B `peer_same`. Inconclusive: the rule
   manager was *better* than the 27B archive manager rather than equivalent (P4); 4B does use RECALL
   on some instances when the 27B manager runs the archive (P5); the held-out KV R2000 gain was large
   (+0.73 to +0.88) but not unanimous at 8 seeds (P9).

7. **The manager must be large: a 9B manager hurts, only 27B helps.** Manager-size ladder on Needle
   (greedy, 8 seeds, paired with `none` run in the same job), mean reward:

   | actor | level | `none` | 4B manager | 9B manager | 27B manager |
   |---|---|---|---|---|---|
   | 0.8B | K4-C24 | 0.33 | 0.24 | 0.16 | 0.97 |
   | 2B | K4-C24 | 0.33 | 0.34 | 0.16 | 1.00 |
   | 4B | K4-C24 | 0.33 | — | 0.16 | 0.80 |
   | 0.8B / 2B / 4B | K8-C48 | 0.15–0.17 | 0.17 / 0.17 / — | 0.05–0.06 | 0.27–0.31 |

   The 9B manager lowers every actor on every seed (0/7–0/8). Its edits are well-formed and applied,
   but it deletes all blocks without first copying the needles into the notes block, while the 27B
   manager copies them before each delete. A 4B manager changes little. The threshold for zero-shot
   editing judgment lies between 9B and 27B, and 9B's harm as `self` is judgment, not the actor role.

## Caveats

- Under sampling, most managed 27B episodes at K8-C48 end at the turn cap (`max_steps`). Doubling the
  cap (job 30791880) changed means by +0.02 (`self` 0.18 → 0.20, `peer_same` 0.30 → 0.31) and
  `peer_same` - `self` from +0.12 to +0.11 (5/2), so the cap is not driving these results.
- Two synthetic tasks, one model family, untrained managers. KV uses a lenient score that ignores
  the `#hash` suffix and punctuation (`scripts/lenient_kv.py`); official KV scores are in the CSV.
- The 27B manager may help partly by absorbing protocol/format work for small actors; no
  format-assist control has been run yet.
