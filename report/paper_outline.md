# Paper outline: "Who should manage the context?"

Draft 2026-10-30. Target: 4 pages plus appendix. Each section names its claim and the figure or table
that carries it; "pending" marks evidence still being run. Source of every number: `docs/results.md`
(finding numbers in brackets) and `docs/tables/`.

**One-sentence claim.** A zero-shot LLM context editor helps a small actor only to the extent that it
keeps needed content before deleting. Without reasoning, only the largest manager of each family does
so under every wording (Qwen3.5 27B, Gemma 4 31B); the Gemma 12B does under one of three wordings. And
for small actors a rule that always does so beats every LLM manager at a fraction of the compute. (Not
for the 27B actor: it restates every needle in each reply, and once management removes the context
wall its replies hit the output cap and the stream stops; `docs/tables/termination_needle.md`.)

| # | Section (pages) | Claim | Carried by | Status |
|---|---|---|---|---|
| 1 | Introduction (0.5) | Who edits a small actor's live context matters only through whether the edits preserve before deleting; capability, not role, decides it; external editing costs about T/(T−R) × the same model reading the stream. | Fig. 1: reward vs PFLOPs, Needle K4-C24 and K8-C48 (`docs/figures/needle_reward_vs_compute.png`). | exists, with the 27B + rule point; Gemma excluded (no FLOPs term for sliding-window layers; GPU-seconds in finding 10) |
| 2 | Setup: Needle as a probe with a known policy (0.6) | Needle labels what to keep and a no-LLM rule scores 1.00 at every level, so managers are scored on executing a known policy. KV tests exact recall. Conditions separate role (self vs `peer_same`) from capability (`peer_same` vs fixed larger manager). | Table 1: conditions; rule 1.00 vs `none` 1.00 → 0.04 across levels [0]. | exists |
| 3 | Who can execute it: the manager ladder (0.8) | Without thinking, only the largest manager of each family helps under every wording: Qwen 27B under seven (+0.02 to +0.67), Gemma 4 31B under three (+0.54 to +0.60 at K4-C24, +0.33 to +0.61 at K8-C48). Gemma 12B helps only under `para3`; Qwen 9B hurts; 4B and same-size do nothing. Thinking partly rescues 9B; 27B with thinking stays ahead. | Table 2: diff vs `none` with CIs (`docs/tables/paired_needle.md`), min/median/max over wordings, ± thinking [1, 7]. | exists; 9B sampled done (P13 confirmed); Gemma rows done (round 31, G1-G5 confirmed) |
| 4 | Why: preserve before delete (0.5) | Gain over `none` tracks the share of manager edits that note or replace, across two families: Spearman 0.70 / 0.64 over 133 cells (Qwen alone 0.67 / 0.62). Necessary, not sufficient: managers that note but never delete hit the wall. | Fig. 2: `docs/figures/needle_protective_vs_gain.png` (Qwen blue, Gemma magenta); event taxonomy (`docs/tables/mechanism_needle.md`) [9, 10]. | exists with Gemma; manager-side identical-input replay probe not run |
| 5 | Same-size management and decoding (0.3) | Same-size management never helps ≤ 4B and hurts 9B. The 27B role effect (`peer_same` > `self`) holds under greedy at K4-C24, where both arms receive the whole stream; at K8-C48 and under sampling it is mixed with the 27B actor's output-cap loop. | Table 3: greedy vs sampled (`docs/tables/decoding_ablation.md`), delivered × kept split (`docs/tables/termination_needle.md`) [2, 4]. | exists |
| 6 | Irreversibility (0.5) | Deletion destroys exact KV recall; archive-with-recall restores it and manager identity stops mattering; the limit becomes the actor's initiative to recall. Agrees with ARC, MemGPT, VISTA. | Table 4: delete vs archive [3, 5]. | exists |
| 7 | Cost of external management (0.4) | Without prefix-cache hits a peer manager reads about S·T/(T−R) tokens (measured/predicted median 0.94); a 27B manager that keeps little reads each streamed token about once, so a small actor + 27B manager costs about the 27B reading the stream itself. The rule on the 2B actor is cheaper than both. | Eq. 1 + Table 5 (`docs/tables/cost_law_needle.md`, `docs/tables/cost_needle.md`) [8]. | exists; 27B + rule measured (P12 falsified: the K4 number is the output-cap loop) |
| 8 | Boundary, pre-registration, limits (0.4) | When relevance needs judgment (MuSiQue), even the best cheap rule loses full evidence for most questions at K = 8, and keeping later hops requires answering earlier ones, so selection and task skill are entangled (stated as a premise with prior art: MuSiQue, IRCoT, IterCOMP). Pre-registration v1: 2 of 5 informative tests confirmed, 3 inconclusive; v2: P11 and P12 falsified (27B actor's output-cap loop), P13 confirmed, P14 inconclusive; v3 replay probe: the cap binds by ~100 tokens, restating is the 27B's policy and survives one-sentence instructions, disclosures and sampling (Q4 falsified). Limits: one layout per level, two tasks, one model family (plus Gemma), untrained managers; 27B-actor rewards are lower bounds where the output-cap loop stops the stream (report how every cell ends). | Table 6: rule full-recall by K (offline simulation); optional selector-probe result. | simulation exists (scratch); probe optional |

## Figures and tables (main text)

- Fig. 1 reward vs compute; Fig. 2 protective share vs gain.
- Tables 1-6 as above; everything else to the appendix (per-cell tables, wordings, prompts).

## Cut or appendix-only

Prompt-version history (v1-v8), KV official-metric detail, pre-registration P-by-P table, the
MuSiQue simulation details, thinking cost breakdown.
