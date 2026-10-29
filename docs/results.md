# Results so far

Who should manage the context? Qwen3.5 actors (0.8B–27B, bf16, vLLM 0.30 on one H200) on ContextBench
Needle Retention and KV Store under a 32,768-token o200k budget. Managers edit numbered context blocks
(delete / replace / note, edit protocol v8) once the context passes 18,432 o200k tokens. Conditions:
`none`, `self` (the actor edits its own context), `peer_same` (a separate session of the same model)
and `peer_<size>` (a fixed manager of another size; mainly `peer_27b`). Managers are untrained
(zero-shot). Each cell has 8 seeds (42–49) unless noted; differences are paired on the same instance.
Greedy decoding with thinking off unless noted.

Sources: `docs/tables/episodes_vllm.csv` (2,872 episodes), `docs/tables/h200_bf16_pooled.md` (greedy
grid), `docs/tables/decoding_ablation.md` (greedy vs sampled), `docs/figures/`. Run log: `CLAUDE.md`.
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
   manager). Results 1, 2 and 7 measure how close zero-shot managers get to this rule.

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

6. **Pre-registered validation: no prediction falsified; most informative ones inconclusive.** Ten
   predictions were committed (`docs/preregistration.md`, tag `prereg-v1`) before new seeds (50–65)
   and two held-out levels: 7 confirmed, 3 inconclusive, 0 falsified
   (`docs/preregistration-results.md`). Five were structural guards that could hardly fail (P2: `self`
   equals `none`; P3/P6: `none` is 0 at KV R1000 by construction; P7: archived needles cannot count
   under final-context grading; P8: the same 27B manager on both sides). Of the five informative tests,
   2 were confirmed (P1: 27B manager gain on new seeds; P10: gain at the held-out K2-C192 level) and 3
   were inconclusive (P4, P5, P9; see the results file). New seeds reuse each level's layout, so only
   P9–P10 test a new layout.

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
- Under sampling, most managed 27B episodes at K8-C48 end at the turn cap; doubling it changed means by
  +0.02, so the cap is not driving those results.
