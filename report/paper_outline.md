# Paper outline: "Who should manage the context?"

Draft 2026-10-30. Target: 4 pages plus appendix. Each section names its claim and the figure or table
that carries it; "pending" marks evidence still being run. Source of every number: `docs/results.md`
(finding numbers in brackets) and `docs/tables/`.

**One-sentence claim.** A zero-shot LLM context editor helps a small actor only to the extent that it
keeps needed content before deleting. In Qwen3.5, only the 27B does so reliably without reasoning, and
a rule that always does so beats every LLM manager at a fraction of the compute.

| # | Section (pages) | Claim | Carried by | Status |
|---|---|---|---|---|
| 1 | Introduction (0.5) | Who edits a small actor's live context matters only through whether the edits preserve before deleting; capability, not role, decides it; external editing costs about T/(T−R) × the same model reading the stream. | Fig. 1: reward vs PFLOPs, Needle K4-C24 and K8-C48 (`docs/figures/needle_reward_vs_compute.png`). | exists; adds the 27B + rule point and the Gemma rows (pending) |
| 2 | Setup: Needle as a probe with a known policy (0.6) | Needle labels what to keep and a no-LLM rule scores 1.00 at every level, so managers are scored on executing a known policy. KV tests exact recall. Conditions separate role (self vs `peer_same`) from capability (`peer_same` vs fixed larger manager). | Table 1: conditions; rule 1.00 vs `none` 1.00 → 0.04 across levels [0]. | exists |
| 3 | Who can execute it: the manager ladder (0.8) | Without thinking, only the 27B manager helps, under seven wordings (+0.02 to +0.67); 9B hurts; 4B and same-size do nothing. Thinking partly rescues 9B; 27B with thinking stays ahead. | Table 2: diff vs `none` with CIs (`docs/tables/paired_needle.md`), min/median/max over wordings, ± thinking [1, 7]. | exists; Gemma rows and 9B sampled (P13) pending |
| 4 | Why: preserve before delete (0.5) | Gain over `none` tracks the share of manager edits that note or replace: Spearman 0.67 / 0.62 over 119 cells. Necessary, not sufficient: managers that note but never delete hit the wall. | Fig. 2: `docs/figures/needle_protective_vs_gain.png`; event taxonomy (`docs/tables/mechanism_needle.md`) [9]. | exists; replay probe (identical inputs, Qwen + Gemma) pending |
| 5 | Same-size management and decoding (0.3) | Same-size management never helps ≤ 4B and hurts 9B. The 27B role effect (`peer_same` > `self`) holds under greedy but not under recommended sampling. | Table 3: greedy vs sampled (`docs/tables/decoding_ablation.md`) [2, 4]. | exists |
| 6 | Irreversibility (0.5) | Deletion destroys exact KV recall; archive-with-recall restores it and manager identity stops mattering; the limit becomes the actor's initiative to recall. Agrees with ARC, MemGPT, VISTA. | Table 4: delete vs archive [3, 5]. | exists |
| 7 | Cost of external management (0.4) | Without prefix-cache hits a peer manager reads about S·T/(T−R) tokens (measured/predicted median 0.94); a 27B manager that keeps little reads each streamed token about once, so a small actor + 27B manager costs about the 27B reading the stream itself. The rule on the 2B actor is cheaper than both. | Eq. 1 + Table 5 (`docs/tables/cost_law_needle.md`, `docs/tables/cost_needle.md`) [8]. | exists; 27B + rule point (P12) pending |
| 8 | Boundary, pre-registration, limits (0.4) | When relevance needs judgment (MuSiQue), even the best cheap rule loses full evidence for most questions at K = 8, and keeping later hops requires answering earlier ones, so selection and task skill are entangled (stated as a premise with prior art: MuSiQue, IRCoT, IterCOMP). Pre-registration v1: 2 of 5 informative tests confirmed, 3 inconclusive; v2: P11-P14. Limits: one layout per level, two tasks, one model family (plus Gemma), untrained managers. | Table 6: rule full-recall by K (offline simulation); optional selector-probe result. | simulation exists (scratch); probe optional |

## Figures and tables (main text)

- Fig. 1 reward vs compute; Fig. 2 protective share vs gain.
- Tables 1-6 as above; everything else to the appendix (per-cell tables, wordings, prompts).

## Cut or appendix-only

Prompt-version history (v1-v8), KV official-metric detail, pre-registration P-by-P table, the
MuSiQue simulation details, thinking cost breakdown.
