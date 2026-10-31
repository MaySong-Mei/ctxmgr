# Results so far

Who should manage the context? Qwen3.5 actors (0.8B–27B, bf16, vLLM 0.30 on one H200) on ContextBench
Needle Retention and KV Store under a 32,768-token o200k budget. Managers edit numbered context blocks
(delete / replace / note, edit protocol v8) once the context passes 18,432 o200k tokens. Conditions:
`none`, `self` (the actor edits its own context), `peer_same` (a separate session of the same model)
and `peer_<size>` (a fixed manager of another size; mainly `peer_27b`). Managers are untrained
(zero-shot). Each cell has 8 seeds (42–49) unless noted; differences are paired on the same instance.
Greedy decoding with thinking off unless noted.

Sources: `docs/tables/episodes_vllm.csv` (2,936 episodes), `docs/tables/h200_bf16_pooled.md` (greedy
grid), `docs/tables/decoding_ablation.md` (greedy vs sampled), `docs/tables/cost_needle.md` (compute
per episode), `docs/tables/termination_needle.md` (how episodes end; reward = delivered × kept),
`docs/figures/`. Run log: `CLAUDE.md`.
Two adversarial reviews (novelty and robustness, 2026-10-28) shaped the wording below; their main
points are in "Limits" at the end.

## What the tasks allow

0. **Needle Retention is solvable by a rule; the question is whether a zero-shot LLM manager finds and
   executes it.** Needle chunks label their own content (`NEEDLES (keep these lines verbatim...)`,
   `FILLER-BLOCK ... delete this entire block`), and the grader counts needle lines anywhere in the
   final context. A task-aware rule (`rule_needle`: copy every `[n...#hash]` line into the notes block,
   delete the older blocks; no LLM) with an actor that only advances scores **1.000 at every level**
   (K4-C4 to K2-C192, 8 seeds each; peak context ≈ 22K o200k), while `none` falls from 1.00 to 0.04
   (`scripts/simulate_rule_needle.py` on `may/track2-baseline`); run live with the 2B actor it also
   scores 1.000 on all 8 seeds at K4-C24 and K8-C48 (job 31255831). So on Needle a manager that helps is
   doing the retention itself; the actor contributes little (0.8B and 2B score alike under the 27B
   manager). The "actor that only advances" is an assumption the 27B actor breaks: with the same rule
   it scores 0.64 at K4-C24 and 0.79 at K8-C48 (job 31451066), because it restates every needle in
   each reply until the reply is cut at the output cap and the stream stops (Limits). The rule kept
   every delivered needle (kept = 1.00); 7 of 8 K4 and 2 of 8 K8 episodes ended in that loop. Results 1, 2 and 7 measure how close zero-shot managers get to this rule.

## Findings

1. **Without thinking, only the 27B manager executes the retention rule, and only at low pressure.**
   `peer_27b` − `none` at Needle K4-C24:

   | actor | greedy | sampled |
   |---|---|---|
   | 0.8B | +0.64 (8/0) | +0.23 (5/1, p=0.22) |
   | 2B | +0.67 (8/0) | +0.43 (8/0) |
   | 4B | +0.46 (8/0) | +0.44 (8/0) |
   | 9B | +0.48 (8/0) | +0.33 (8/0) |

   That is 0.80–1.00 against the rule's 1.00. At K8-C48 the gain is +0.11 to +0.17 (8/0 for every
   size), i.e. 0.27–0.31 against the rule's 1.00, so even the 27B manager falls far short of the rule
   once more needles accumulate (with manager thinking it reaches 0.63–0.82; finding 7). Under sampling the 0.8B actor degenerates (130/600 actor calls hit
   the 2,048-token output cap, none under greedy).

2. **Same-size management does not help small actors.** `self` and `peer_same` equal `none` for
   0.8B–4B and lower it for 9B (Needle K4-C24: `self` −0.16 greedy / −0.17 sampled, 0/8 seeds). The
   small models' edits often fail to apply (0.8B applies about 0 edits per episode, 2B/4B 1–2), but the
   4B self-manager does write notes in 47 of 72 edit-v8 Needle events, so the copy mechanism is within reach.

3. **On KV, forced deletion destroys exact recall (lenient metric); the official metric tells a
   different story.** At R520 the lenient recall (`scripts/lenient_kv.py`, ignores the `#hash` suffix
   and punctuation) drops from 0.98–1.00 (`none`) to 0.03–0.28 for every manager, size and decoding
   (4B and 9B, 0/8 seeds each). Under the official metric (value including its `#hash`), `none` itself
   scores low (9B 0.00, 4B 0.12) and managed arms score 0.03–0.17 (9B) and 0.06–0.16 (4B), so the
   harm exists only under the lenient metric, which we use because the hash is an arbitrary suffix the
   actors routinely drop. The harm is also forced by the trigger: R520 fits the budget, and with the
   trigger at 27K instead of 18,432 the 4B `peer_same` arm recovers to 1.00 lenient (4 seeds). The 27B
   manager and the 9B self-manager made identical edits in all 24 KV episodes checked, and the 27B
   manager wrote 1 note in 249 KV events.

4. **27B actor: separate-session management beats self-management under greedy, not robustly.**
   `peer_same` − `self`:

   | level | greedy | sampled |
   |---|---|---|
   | Needle K4-C24 | +0.48 (8/0) | +0.06 (4/4) |
   | Needle K8-C48 | +0.20 (8/0) | +0.12 (6/1) |
   | Needle K8-C60 (n=4, greedy only) | +0.18 (4/0) | |
   | Needle K4-C96 (n=4, greedy only) | +0.16 (4/0) | |

   Not a pure role effect: `self` sees its own system prompt and 80-character block previews, while
   `peer_same` sees a manager system prompt and every block in full. Under the model card's sampling
   (greedy is not a recommended setting for Qwen3.5) the gap mostly disappears.
   Splitting reward into the share of the stream delivered and the share of delivered needles kept
   (`docs/tables/termination_needle.md`): at K4-C24 greedy both arms receive almost the whole stream
   (`peer_same` 0.97, `self` 0.99 of 24 ops), so the +0.48 gap is retention (kept 0.80 vs 0.39). At
   K8-C48 `peer_same` receives only 0.45–0.65 of the stream (4 of 8 greedy episodes end at the turn
   cap) and under sampling 11 of 16 episodes in each arm do, so there the gap mixes retention with the
   output-cap loop described under Limits.

5. **With archive-with-recall, deletion stops hurting KV, and a rule manager suffices.** Removed blocks
   are archived; the actor writes `RECALL <key>` and the harness appends the archived lines with that
   key. Greedy, 8 seeds, lenient KV:

   | actor | level | none | delete arms | archive arms (self / peer_same / peer_27b / rule) |
   |---|---|---|---|---|
   | 9B | R520 | 0.98 | 0.03–0.28 | 0.91 / 0.94 / 0.91 / 0.99 |
   | 9B | R1000 | 0.00 | 0.08–0.23 | 1.00 / 1.00 / 0.97 / 0.99 |
   | 4B | R520 | 1.00 | 0.09–0.28 | 0.48 / 0.54 / 0.51 / 0.45 |
   | 4B | R1000 | 0.00 | 0.05–0.23 | 0.11 / 0.17 / 0.09 / 0.06 |

   This agrees with prior archive-and-recall systems (MemGPT; VISTA; ARC, arXiv 2607.25066, where a
   rule-based compactor with ID-addressable recall beats LLM summarization). Two qualifications: the
   harness performs the exact-key lookup, so "a rule matches 27B" largely follows from retrieval
   carrying the task; and the bottleneck becomes the actor's initiative to recall (9B recalls ~23 of 24
   GETs, 4B rarely or never). The 2B actor's ANSWER format breaks when the archive paragraph is added
   (`none_r` 0.00 vs `none` 0.46), so small actors are sensitive to instruction length; 2B is excluded.

6. **Pre-registered validation: v1 had none falsified, most informative tests inconclusive; v2 had
   two falsified by the 27B actor's output-cap loop.** Ten predictions were committed (`docs/preregistration.md`, tag `prereg-v1`) before new seeds (50–65)
   and two held-out levels: 7 confirmed, 3 inconclusive, 0 falsified
   (`docs/preregistration-results.md`). Five were structural guards that could hardly fail (P2: `self`
   equals `none`; P3/P6: `none` is 0 at KV R1000 by construction; P7: archived needles cannot count
   under final-context grading; P8: the same 27B manager on both sides). Of the five informative tests,
   2 were confirmed (P1: 27B manager gain on new seeds; P10: gain at the held-out K2-C192 level) and 3
   were inconclusive (P4, P5, P9; see the results file). New seeds reuse each level's layout, so only
   P9–P10 test a new layout.
   Pre-registration v2 (`docs/preregistration-v2.md`, round 29) has two falsified, one confirmed and
   one inconclusive:
   - **P11 falsified:** 27B actor + rule 0.64 (K4-C24) and 0.79 (K8-C48), below 0.90 at both levels.
     The cause is the output-cap loop (finding 0, Limits); this explanation is post hoc.
   - **P12 falsified:** 14.2 PF per episode at K4-C24 against 8.7 predicted (+63%); 25.5 PF at
     K8-C48 against 24.1 (+6%). At K4 the number measures the loop: 7 of 8 episodes made 58 actor
     calls for 14 of 24 ops, 45 of them cut at the cap.
   - **P13 confirmed:** recommended sampling does not rescue the 9B manager: `peer_9b` − `none`, both
     sampled, −0.19 [−0.22, −0.17] at K4-C24 (0/8 seeds) and −0.06 [−0.09, −0.03] at K8-C48 (1/7).
   - **P14 inconclusive:** protective share 0.03 at K4-C24, 0.14 at K8-C48 (between 0.10 and 0.25).
     Exploratory: 9B replies cut at the 4,096-token manager cap fall from 0.38 / 0.08 (greedy) to 0 / 0
     under sampling, and the extra applied edits are unprotected deletes.

7. **Manager-size ladder: without thinking only the 27B manager helps, under seven prompt wordings;
   with thinking both 9B and 27B help, and 27B stays ahead.** Needle, greedy, 8 seeds, `none` rerun
   in the same job:

   | actor | level | `none` | 4B manager | 9B manager | 27B manager |
   |---|---|---|---|---|---|
   | 0.8B | K4-C24 | 0.33 | 0.24 | 0.16 | 0.97 |
   | 2B | K4-C24 | 0.33 | 0.34 | 0.16 | 1.00 |
   | 4B | K4-C24 | 0.33 | — | 0.16 | 0.80 |
   | 0.8B / 2B / 4B | K8-C48 | 0.15–0.17 | 0.17 / 0.17 / — | 0.05–0.06 | 0.27–0.31 |

   The 9B manager lowers every actor on every seed (0/7–0/8): it deletes blocks without first copying
   the needles (it wrote no `note` in 347 events; 81% of its K4-C24 events applied, and 30 replies were
   runaway id lists that hit the output cap). The v8 prompt was tuned while watching the 27B manager,
   and under the early prompts (v1/v2) the 27B manager failed the same way (0.17 / 0.00 on 2B K4-C24).
   Prompt-variant check (round 27): two generic additions to v8, `example` (one worked
   note-then-delete reply) and `mustnote` (the copy rule stated as a requirement), same cells:

   | manager, prompt | 2B K4-C24 | 2B K8-C48 | 4B K4-C24 | 4B K8-C48 |
   |---|---|---|---|---|
   | 9B v8 | −0.18 (0/8) | −0.09 (0/8) | −0.17 (0/8) | −0.10 (0/8) |
   | 9B +example | +0.03 (4/3) | +0.07 (7/1) | −0.04 (2/5) | −0.02 (3/5) |
   | 9B +mustnote | −0.20 (0/8) | −0.05 (2/6) | −0.17 (0/8) | −0.10 (0/8) |
   | 27B v8 | +0.67 (8/0) | +0.17 (8/0) | +0.46 (8/0) | +0.11 (8/0) |
   | 27B +example | +0.04 (8/0) | +0.02 (6/2) | | |
   | 27B +mustnote | +0.62 (8/0) | +0.26 (8/0) | | |

   (difference vs `none` run in the same job; 27B v8 from the pooled grid.) The worked example
   removes the 9B manager's harm but does not make it help (it notes in 31 of 208 events); stating the
   rule changes nothing (4 of 231). No prompt brought 9B near 27B. The prompt matters as much for 27B:
   `example` drops it to `none` level (it notes once at the first event, then only deletes, the v6
   failure), while `mustnote` keeps K4-C24 and improves K8-C48 (0.40 vs 0.31 under v8).

   Paraphrases (round 28): four full rewrites of v8 with the same content and edit syntax and no
   example (`para1` terse bullets, `para2` prose, `para3` numbered procedure, `para4` rules plus a
   checklist), 2B actor, difference vs `none`:

   | manager, prompt | K4-C24 | K8-C48 |
   |---|---|---|
   | 9B para1 / para2 / para3 / para4 | 0.00 (0/0) / −0.10 (0/4) / −0.16 (0/6) / −0.17 (0/6) | −0.02 (1/2) / −0.10 (0/8) / −0.07 (0/8) / −0.08 (0/8) |
   | 27B para1 / para2 / para3 / para4 | +0.35 / +0.61 / +0.67 / +0.17 (8/0 each) | +0.20 / +0.39 / +0.21 / +0.05 (8/0 each) |

   Over the seven wordings (v8, `example`, `mustnote`, four paraphrases) the 9B manager never helps
   and is significantly harmful in at least one cell under five; under the paraphrases it wrote no
   note in 445 events (under `para1` all 24 K4-C24 replies were runaway delete lists cut off at the
   4,096-token cap, so nothing applied). The 27B manager beats `none` on 8/8 seeds in 13 of 14 cells,
   but by +0.02 to +0.67, and its score tracks how often it notes before deleting (`para3`: notes in
   32/32 K4-C24 events, 1.00 on every seed; `para4`: 13/32, 0.51). The direction of the 9B/27B gap
   is robust to wording; its size is not.

   Manager thinking (round 28): v8 with thinking on for the manager only (actor unchanged, manager
   output cap 16,384), means:

   | actor | level | `none` | 9B | 9B thinking | 27B | 27B thinking | 27B − 9B, both thinking |
   |---|---|---|---|---|---|---|---|
   | 2B | K4-C24 | 0.33 | 0.16 | 0.66 | 1.00 | 0.93 | +0.27 (7/1) |
   | 2B | K8-C48 | 0.15 | 0.06 | 0.34 | 0.31 | 0.63 | +0.29 (6/2) |
   | 4B | K4-C24 | 0.33 | 0.16 | 0.43 | 0.80 | 0.83 | +0.40 (8/0) |
   | 4B | K8-C48 | 0.16 | 0.06 | 0.23 | 0.27 | 0.82 | +0.59 (8/0) |

   Thinking turns the 9B manager from harmful to helpful with the 2B actor (vs 9B without thinking
   +0.51 and +0.28, 8/0 each; vs `none` +0.33 (7/0, p=0.016) and +0.20 (6/0, p=0.031)); with the 4B
   actor its gain over `none` is not significant (+0.10, +0.07). It rarely notes (3 of 234 events)
   and instead replaces chunk blocks with their needle lines or deletes selectively. The 27B manager
   with thinking is the best manager so far: at K8-C48, where every manager without thinking stayed
   near 0.3, it reaches 0.63 (2B) and 0.82 (4B; +0.54 over 27B without thinking, 8/0), noting in 42 of
   93 and 81 of 128 events. Thinking costs 5–8K output tokens per manager call; 8–42% of calls per
   cell hit the 16,384 cap before the answer (no edit), the 27B thinking jobs took 4–6 h instead of
   ~0.5 h, and 5 of 32 27B-thinking episodes ended at `max_steps` (scores are lower bounds). So
   editing judgment depends on both size and reasoning: reasoning alone rescues 9B with the 2B actor,
   and with reasoning on both, 27B stays ahead (8/0 with the 4B actor). Findings 1–2 and the ladder
   above are without thinking.

8. **A 27B compactor costs about (manager size / actor size) × the actor's compute: cheaper than a
   27B system, far above a rule.** Prefix-reuse FLOPs (CLM Eq. 8–9, constants from each
   `config.json`; the 27B gives CLM's 48.70 GFLOP/token and 3.93×10⁵ per query-key pair) with the
   prefix-cache hits vLLM measured, and GPU-seconds (summed call latency; actor and manager share one
   H200, one call at a time). Means per episode from `docs/tables/cost_needle.md`, which covers only
   runs that logged cache hits (rounds 25–29); figure `docs/figures/needle_reward_vs_compute.png`
   (`scripts/plot_cost.py`):

   | level | system | reward | PFLOPs | manager share | GPU-s |
   |---|---|---|---|---|---|
   | K4-C24 | 2B `none` | 0.33 | 0.16 | – | 3 |
   | K4-C24 | 2B + `rule_needle` | 1.00 | 0.56 | – | 10 |
   | K4-C24 | 2B + 9B manager | 0.16 | 3.7 | 89% | 80 |
   | K4-C24 | 2B + 9B manager, thinking | 0.66 | 4.2 | 92% | 190 |
   | K4-C24 | 2B + 27B manager | 0.99 | 7.5 | 94% | 90 |
   | K4-C24 | 2B + 27B manager, thinking | 0.93 | 10.0 | 95% | 356 |
   | K4-C24 | 27B `none` | 0.29 | 2.4 | – | 53 |
   | K4-C24 | 27B `self` | 0.40 | 13.0 | 10% | 576 |
   | K4-C24 | 27B `peer_same` | 0.77 | 28.5 | 48% | 1,312 |
   | K4-C24 | 27B + `rule_needle` | 0.64 | 14.2 | – | 1,461 |
   | K8-C48 | 2B `none` | 0.15 | 0.18 | – | 5 |
   | K8-C48 | 2B + `rule_needle` | 1.00 | 1.6 | – | 35 |
   | K8-C48 | 27B `none` | 0.15 | 2.8 | – | 101 |
   | K8-C48 | 27B + `rule_needle` | 0.79 | 25.5 | – | 1,152 |
   | K8-C48 | 2B + 9B manager, thinking | 0.34 | 6.7 | 93% | 423 |
   | K8-C48 | 2B + 27B manager (`para2`) | 0.53 | 18.7 | 96% | 281 |
   | K8-C48 | 2B + 27B manager, thinking | 0.63 | 28.2 | 97% | 1,503 |
   | K8-C48 | 4B + 27B manager, thinking | 0.82 | 34.8 | 91% | 2,027 |

   - The actor reads each chunk about once (about 73% of its prompt tokens are cache hits). A peer
     manager re-reads the whole current context (~30K Qwen tokens) at every event and gets almost no
     cache hits, because each event follows the previous event's edits and the request starts with
     the changing context size. So manager compute ≈ C_token(manager) / C_token(actor) × actor
     compute: under the 27B v8 manager, measured 17× for the 2B actor (ratio of constants 17.7),
     6.4× for 4B (6.8) and 3.7× for 9B (3.5). The self-manager continues the actor's own cache
     (92–96% hits), its one structural cost advantage.
   - A small actor with a 27B compactor beats the 27B model managed by a second 27B session at
     about a quarter of the FLOPs and a fifteenth of the GPU time (2B + 27B: 0.99, 7.5 PF, 90 s;
     27B `peer_same`: 0.77, 28.5 PF, 1,312 s). The task-aware rule reaches the same score at 1/13 of
     the FLOPs.
   - Thinking costs more in time than in FLOPs, because decode is memory-bound. At K4-C24 the 27B
     manager with thinking uses 1.3× the FLOPs and 4× the GPU time of v8 for no gain; at K8-C48 it
     adds +0.10 over the best wording without thinking (`para2`) for 1.5× the FLOPs and 5× the GPU
     time.
   - The 27B `self`, `peer_same` and `rule_needle` rows include actor turns spent in output-cap
     loops (Limits), so their cost is partly the loop, not management (27B + rule at K4-C24: 58 actor
     calls for 14 ops in 7 of 8 episodes).
   - `none` is cheap because it stops at the budget wall after about 7 of 24 (K4-C24) or 48 (K8-C48)
     operations. FLOPs omit norms, softmax and the DeltaNet state update; GPU-seconds are batch-1
     calls on a shared H200 and do not transfer to other deployments.
   - How much a peer manager reads follows a trigger law: it re-reads the context (about T tokens)
     at each event and each event frees about T − R, so over S streamed tokens it reads about
     S·T/(T − R). Measured against predicted, the median ratio is 0.94 (IQR 0.74–0.98, 55 cells;
     `docs/tables/cost_law_needle.md`). For the 27B manager it is 0.94–0.98, and because it leaves
     little behind (R ≪ T) it reads each streamed token only 1.04–1.20 times. So a small actor with a
     27B manager costs about as much as the 27B reading the stream itself, plus the small actor. The law
     overpredicts for managers that free almost nothing (small `peer_same`, `peer_4b`, 9B thinking),
     whose episodes end at the budget wall instead.

9. **Across every managed cell, the gain over `none` tracks how often the manager protects needles
   when it edits.** For each Needle cell (condition × manager × prompt wording × decoding, n ≥ 4),
   the share of applied manager events that `note` or `replace` (the two ways to keep needles while
   removing their block) correlates with reward − `none`: Spearman 0.67 over 65 cells at K4-C24 and
   0.62 over 54 at K8-C48 (`docs/tables/mechanism_needle.md`,
   `docs/figures/needle_protective_vs_gain.png`). The 27B cells lie close to one line, from
   `example` (0.25 protective, +0.04) to `para3` (1.00, +0.67); 9B and smaller managers sit at 0–0.04
   protective. Protecting is necessary, not sufficient: the 4B self-manager and the 0.8B `peer_same`
   add notes but never delete (0 delete events), so every episode ends at the budget wall and they
   gain nothing. Manager thinking is the main exception (27B thinking: 0.40 protective, +0.60 at
   K4-C24; it deletes in all 35 applied events). The per-event measure cannot see whether a deleted
   block's needles were noted in an earlier event, so it may understate protection there.
   Paired differences with seed-level bootstrap CIs for every cell are in
   `docs/tables/paired_needle.md`.

## Limits

- **Novelty.** Archive-and-recall beating deletion (finding 5) and "small actor + large helper at
  lower cost" are known (ARC, MemGPT, VISTA; Minions, advisor models, the Proactive Memory Agent,
  arXiv 2607.08716). Untrained managers hurting is also reported (AdaCoM's untrained 4B manager; an
  untrained Qwen3.5-27B memory agent in 2607.08716). What we have not found elsewhere: a within-family
  ladder of a larger model editing a smaller actor's live context, against a same-weights
  separate-session control, crossed with delete vs archive.
- **Prompt and decoding.** Edit protocol v1–v8 was tuned on the 27B manager with 2B/4B actors (seeds
  46–49). Most results are greedy with thinking off; the 27B role effect does not survive sampling.
  Manager thinking raises both the 9B and 27B managers a lot (finding 7), so the results without it
  understate what these models can do as managers; the 27B gain without thinking varies from +0.02
  to +0.67 across seven zero-shot wordings.
- **Sample.** Seeds change words, values and queried keys, not the layout of a level; `none` is often
  constant (0.333 on Needle K4-C24), so paired tests are close to one-sample tests. Greedy outputs vary
  with server state (deviation D1), so repeated instances are near-repeats.
- **Scope.** Two synthetic tasks, one model family, one trigger (18,432), untrained managers.
- **Turn cap and output cap (27B actor).** In every reply the 27B actor restates all needles received
  so far (smaller actors restate only the current op's, so their replies stay at ~170 tokens at K4
  and ~320 at K8). Once management keeps the context under the wall, the reply grows until it is cut
  at the 2,048-token output cap, before `READY_FOR_NEXT_OP`; the stream does not advance, the runner's
  reminder yields the same cut reply, and the episode loops until the turn cap. 47 of 136 managed
  27B-actor Needle episodes end at `max_steps`, 31 of them with their last 5 replies all cut at the
  cap (9 of the 31 are 27B + rule, round 29); for smaller actors it is 0–6 episodes per size (the 4B actor under thinking managers has 5 cap
  loops). Doubling the turn cap changed means by +0.02, which is what an absorbing loop predicts; an
  earlier version of this line read that as "the cap is not driving those results", which was wrong.
  27B-actor rewards are lower bounds wherever delivered < 1 (`docs/tables/termination_needle.md`).
  Under `none` the restating costs the 27B one op before the wall (0.29 vs 0.32–0.33 for 0.8B–9B at
  K4-C24).
