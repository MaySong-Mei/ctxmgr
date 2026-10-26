# Results so far

Who should manage the context? Qwen3.5 actors (0.8B–27B, bf16, vLLM 0.30 on one H200) on ContextBench
Needle Retention and KV Store under a 32,768-token o200k budget. Managers edit numbered context blocks
(delete / replace / note, edit protocol v8) once the context passes 18,432 o200k tokens. Conditions:
`none`, `self` (the actor edits its own context), `peer_same` (a separate session of the same model)
and `peer_27b` (a fixed Qwen3.5-27B manager). Managers are untrained (zero-shot). Each cell has 8 seeds
(42–49) unless noted; differences are paired on the same instance.

Sources: `docs/tables/episodes_vllm.csv` (1,375 episodes), `docs/tables/h200_bf16_pooled.md` (greedy
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

## Caveats

- Under sampling, most managed 27B episodes at K8-C48 end at the turn cap (`max_steps`). Doubling the
  cap (job 30791880) changed means by +0.02 (`self` 0.18 → 0.20, `peer_same` 0.30 → 0.31) and
  `peer_same` - `self` from +0.12 to +0.11 (5/2), so the cap is not driving these results.
- Two synthetic tasks, one model family, untrained managers. KV uses a lenient score that ignores
  the `#hash` suffix and punctuation (`scripts/lenient_kv.py`); official KV scores are in the CSV.
- The 27B manager may help partly by absorbing protocol/format work for small actors; no
  format-assist control has been run yet.
