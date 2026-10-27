# Separate role from capability before deleting more context

No paper or shipped product found through late October 2026 runs a controlled self-vs-external context-manager comparison with matched tools. None uses the actor's own weights as an external manager, and none varies manager size relative to the actor. The project's same-weights `self` vs `peer_same` contrast and its same-size vs `peer_27b` contrast therefore have no direct precedent. The closest work is AdaCoM. It uses a 4B external manager that is much smaller than the agents it serves, and that manager *hurts* when untrained (average 22.01 vs ReAct 24.17); only RL training makes it win (33.60) ([AdaCoM](https://arxiv.org/html/2605.30785)). CLM makes self-management its design premise and never has one model edit another model's context ([CLM](https://arxiv.org/html/2609.37725)). On mechanisms, prior art agrees that strong systems **archive rather than delete**, that summaries lose exact values, that single-shot vs multi-step editing has never been ablated, and that trigger ablations are rare and favor moderate thresholds. The project's KV Store collapse (lenient recall from about 1.0 to 0.03–0.28 once any manager fires) is what the literature predicts for irreversible deletion when there is no recovery path. The four experiments below are ranked by novelty × feasibility × claim strength. (1) Archive-with-recall on KV Store. (2) A manager-size ladder (9B and 4B peers managing smaller actors). (3) Protocol-vs-judgment controls: a rule-based masking manager plus schema-constrained small managers. (4) As a stretch, a plan-then-edit `self` condition for the 27B actor. Together they need roughly 25–55 H200-hours and fit in one to two weeks.

## No prior work isolates the manager's role from its capability

**CLM (arXiv:2609.37725) never puts a second model in charge of a context.** A full-text pass over Sections 1–6 and Appendices D–G found swarms in which each agent owns its own context file. It also found an orchestrator that keeps scoreboards *about* subagents in its own context, and "assisted evolution" in which Claude Fable 5.1 writes offline skill documents for Qwen3.6-27B. None of these is runtime editing of another model's context. The paper says CLMs "delegate full responsibility for maintaining the context to the CLM itself" ([CLM](https://arxiv.org/html/2609.37725)). Its scale evidence is consistent with our weak 9B self results. Qwen3.5-9B makes 1.4 edits per TerminalBench task (none in half of tasks) vs 2.6 for 27B, and runs to a median peak of 30.2K of 32K vs 17.6K ([CLM](https://arxiv.org/html/2609.37725)). Two corrections to the project brief `docs/manager-literature.md` follow. The phrase "limited context-management capabilities" does not appear in Appendix F; the closest wording is that gains "grow with the model's ability to make that decision." And the 9B-vs-summary sign depends on setup. Appendix F gives CLM-9B 39.9 vs summary 37.7 on BrowseComp-Plus, but the zero-shot row of Table 2 gives **28.8 vs 34.7** ([CLM](https://arxiv.org/html/2609.37725)). One gap remains: the paper's Figure 19 (TerminalBench accuracy by size) was not read.

**AdaCoM confounds role with capability by design.** Its external manager is always Qwen3-4B-Instruct, and the frozen agents are Qwen3-Max, Kimi-K2, GLM-4.5-Air and DeepSeek-V3. Its Limitations section says larger managers were not trained for lack of compute, and the paper contains no same-size or larger-than-agent manager ([AdaCoM](https://arxiv.org/html/2605.30785)). Its untrained column is the closest analogue to our zero-shot managers. There the external manager averages 22.01, essentially tied with agent self-summarization (SumAgent, 22.06) and below ReAct (24.17). Training is what helps (AdaCoM 33.60, SumCoM 29.39). For the strongest agent, trained self-management still wins: MemAct 37.33 vs AdaCoM 36.67 on Qwen3-Max ([AdaCoM](https://arxiv.org/html/2605.30785)). Our `peer_27b` result is therefore a different claim. Zero-shot, a manager *larger* than the actor improves Needle K4-C24 by +0.46 to +0.67 under greedy decoding, while same-size managers do nothing (0.8B–4B) or harm (9B) ([results](../docs/results.md)). The unread tail of the AdaCoM appendix (about 15K characters) could hide a manager-size ablation. None appears in the main text.

**The wider literature splits deciding from writing but never ablates who does it.** Self-GC forks the agent's prefix into an untrained side-channel planner that proposes fold, mask and prune actions; the harness validates them. That is structurally close to our `peer_same`, but the paper has no self baseline and no task-success metric ([Self-GC](https://arxiv.org/html/2607.00692)). ACM lets the agent decide *when* to compress while a separate call *writes* the summary, and uses the same Qwen3.5-9B for both roles ([ACM](https://arxiv.org/html/2607.23809)). ACE's elasticizer "is set to be the same as the main LLM" ([ACE](https://arxiv.org/html/2606.31564)). Self-managed systems such as Scroll, Bud ContextCompact 2.0 and "Towards Reliable Context Compression" compare only against same-backbone baselines ([Scroll](https://arxiv.org/html/2608.21690); [Bud](https://arxiv.org/html/2610.25516)). Shipped products follow cost, not capability. Letta's summarizer defaults to a cheaper model (Haiku 4.5, gpt-5-mini, gemini-2.5-flash) ([Letta](https://docs.letta.com/guides/core-concepts/messages/compaction)). LangChain's examples pair a gpt-5.5 agent with a gpt-5.4-mini summarizer ([LangChain](https://docs.langchain.com/oss/python/langchain/middleware/built-in)). OpenAI frames compaction as a skill its model is "natively trained" for, i.e. self-compaction; that claim comes from search snippets, because the primary page returned 403 ([OpenAI](https://openai.com/index/gpt-5-1-codex-max/)). **No shipped product uses a compactor larger than the agent.** Letta's own guidance disagrees with itself on direction: its blog suggests a stronger sleep-time model, while a community guide recommends a cheap one ([Letta blog](https://www.letta.com/blog/sleep-time-compute); [Letta forum](https://forum.letta.com/t/sleeptime-agents-for-memory-consolidation-best-practices-guide/154)).

## Side models saturate near 4B, but live editors evidently need more

The only manager-size sweeps in the literature cover memory *side* roles, not editors of a live context, and they plateau early. E-mem's assistant agents rise from 28.89 F1 at 0.6B to 50.70 at 4B, then stay flat (49.80 at 8B, 50.02 at 14B) ([E-mem](https://arxiv.org/html/2601.21714)). HMARS finds a 0.6B memory sub-agent as good as an 8B one (0.838 vs 0.832) ([HMARS](https://arxiv.org/html/2606.28349)). Our grid points the other way for editing. Same-size managers from 0.8B to 9B are no-ops or harmful, and only 27B helps on Needle. Edit logs separate two failure types. The 0.8B model applies about zero edits and 2B/4B apply one or two, which is a protocol failure. The 9B model applies every edit, deletes needles, and submits early, which is a judgment failure ([results](../docs/results.md)). Our working hypothesis, which no source tests, is that deciding what a live context can safely lose is harder than retrieving or summarizing. Within-family evidence on self-management is sparse. Scroll's backbone sweep bottoms out at a 35B-total MoE model, and its authors call the failures "not protocol-level" ([Scroll](https://arxiv.org/html/2608.21690)). Our sub-10B failures are largely protocol-level, so the 0.8B–27B dense sweep covers ground the literature has not. A methodological advantage worth stating in the write-up: ContextBench's forced stream guarantees that management fires. In agentic tasks, small models often stop before it would. ACM's Qwen3-4B stops after about 2 turns (3.4% on BrowseComp-Plus), and Bud's Qwen3-8B triggered compaction once in 200 sessions ([ACM](https://arxiv.org/html/2607.23809); [Bud](https://arxiv.org/html/2610.25516)).

The format confound is real, and the literature offers a clean control for it. Small models fail format adherence independently of content quality, and hard schema decoding fixes validity but can cut accuracy. In one study, Qwen2.5-1.5B tool-call accuracy fell from 91.5% (prompt-only JSON) to 48.0% under a hard schema, and the authors recommend "reason free, constrain late." Both findings come from abstract-only reading ([Constraint Tax](https://arxiv.org/abs/2605.26128); [Chavan 2026](https://arxiv.org/abs/2609.23742)). A schema-constrained small manager therefore isolates *judgment* from *protocol* only if valid-but-harmful edits are tracked separately.

## Archiving beats deletion, and ContextBench rewards only what returns to context

**Methods that evict content to a recoverable store top the comparison tables, and summary or deletion methods sit at the bottom.** On LOCA-Bench (Gemini-3-Flash, 128K), VISTA scores 50.7% and its no-archive variant 27/75 (36%). The ordering runs from ReAct 22.7, tool-result clearing 26.7, stale-observation masking 28.0 and a SLIM summary 29.3 up to archive-based methods at 44.0–50.7 ([VISTA](https://arxiv.org/html/2606.30005v5)). The effect appears only under pressure: at 8K the methods are roughly tied (86.7 vs 84.0). Recovery itself is rare. Only 8 of 16 rescued runs that archived a block ever recalled it, so most of the gain comes from *safe* eviction ([VISTA](https://arxiv.org/html/2606.30005v5)). The irreversible methods concede the cost. MemAct states that "the system cannot recover original data once details are summarized" ([MemAct](https://arxiv.org/html/2510.12635)). Products converge on the same pattern. Manus requires compression to be restorable: keep the URL, drop the page ([Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)). Cursor writes long outputs and chat history to searchable files ([Cursor](https://cursor.com/blog/dynamic-context-discovery)). Anthropic warns the model to save to its memory tool before clearing tool results ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)), and reports +39% on an internal agentic-search eval for memory plus context editing vs +29% for editing alone ([Claude blog](https://claude.com/blog/context-management)).

Two caveats shape the KV experiment. First, **ContextBench reads the reward from context snapshots, not from disk**, so an offloaded value counts only once it is pulled back into live context ([ContextBench](https://github.com/RulinShao/ContextBench)). CLM's evolved KV skill does exactly that: it offloads SET batches to `/tmp/ctx_offload/` and greps on GET. That skill lifted KV Store from 22.3% to 83.8% on the dev split and from 38.3% to 74.2% on held-out test, the strongest evidence that offload plus keyed retrieval fixes KV ([CLM](https://arxiv.org/html/2609.37725)). Second, retrieval is not automatically safe. Semantic retrieval was the worst history baseline on AppWorld (27.4 vs 56.0 uncompressed) ([ACON](https://arxiv.org/html/2510.00615)), and A-MEM was the worst baseline in MemAct's QA table (0.399) ([MemAct](https://arxiv.org/html/2510.12635)). The design below therefore uses *exact keyed* lookup. CLM's per-family pilot numbers (summary vs offload on KV) exist only in a figure that was not read, so the report should not claim "summary < offload on KV" from CLM.

## Single-shot vs multi-step editing and triggers are almost unablated

No source compares a single-shot edit reply with an inspect-then-edit loop. The strongest systems are multi-step by construction. CLM edits its own context file with Bash over many turns (one trace shows 163 in-place edits) ([CLM](https://arxiv.org/html/2609.37725)). VISTA's agent archives and recovers repeatedly, and its dashboard ablation shows that *informed* edits matter more than edit count: without the dashboard it made 255 archives and 57 recoveries, vs 69 and 105 with it ([VISTA](https://arxiv.org/html/2606.30005v5)). Compressor-style systems (ACON, ReSum, AdaCoM) are single-shot per trigger. Our round-14 observation is that 27B as `self` emits one edit per reply while as a peer it emits a plan. That points to response mode, not separation itself, as a candidate mechanism behind the 27B role effect. The effect is fragile: +0.48 greedy, only +0.06 sampled at K4-C24 ([decoding ablation](../docs/tables/decoding_ablation.md)).

Trigger evidence is thin and points one way: compressing early or often costs accuracy. ACON finds moderate thresholds best (4,096 history / 1,024 observation tokens) and reports that lower thresholds add calls and lose accuracy ([ACON](https://arxiv.org/html/2510.00615)). MemAct's fixed every-5-turns schedule nearly matches its learned policy (0.582 vs 0.591) ([MemAct](https://arxiv.org/html/2510.12635)). Models cannot sense their own context size. Without a dashboard, median relative error in self-estimated size runs 0.43–0.84 across four frontier models, and 0.00 with one ([VISTA](https://arxiv.org/html/2606.30005v5)). CLM's Appendix G finds the same and recommends hints ([CLM](https://arxiv.org/html/2609.37725)). Products fire at fixed fractions or token counts. Gemini CLI uses a 0.5 compression threshold ([Gemini CLI](https://geminicli.com/docs/cli/settings/)), Anthropic's threshold compaction defaults to 150,000 input tokens (minimum 50,000) ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/compaction-threshold)), and its tool-result clearing defaults to 100,000 ([Claude docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)). The strongest cheap baseline is non-LLM. JetBrains' "Complexity Trap" finds that observation masking matches LLM summarization's solve rate on SWE-bench Verified at about half the cost (abstract-only reading) ([Complexity Trap](https://arxiv.org/abs/2508.21433)). A Factory vendor self-evaluation finds that all compaction methods lose artifact detail (artifact-trail scores 2.19–2.45 out of 5) ([Factory](https://factory.com/news/evaluating-compression)). Neither study measures exact-recall retention in a live context, which ContextBench does.

## Four experiments ranked for the remaining two weeks

| Rank | Experiment | Novelty vs prior art | Feasibility (1–2 wk) | Claim it strengthens | Est. H200-h |
|---|---|---|---|---|---|
| 1 | Archive-with-recall on KV Store | Medium-high: no delete-vs-archive ablation on exact-recall KV under fixed manager identities | Medium: one new actor command plus archive store | Turns the KV collapse into a mechanism claim (irreversibility) and gives KV a possible positive effect | 6–12 |
| 2 | Manager-size ladder (`peer_9b`, `peer_4b`) | High: AdaCoM never varied manager size; no live-editor sweep exists | High: config change only (if the peer manager id is a parameter, as for `peer_27b`) | Locates the capability threshold; separates 9B's edit judgment from its early submission as actor | 5–12 |
| 3 | Protocol-vs-judgment controls (masking rule, constrained JSON) | Medium: masking is a known baseline, but untested on exact-recall live context | High: a deterministic manager plus schema decoding | Rules out "27B helps just by freeing space" and "same-size fails only on format" | 5–12 |
| 4 (stretch) | 27B `self_plan` (plan-then-edit in one reply) | High: no single-shot vs plan ablation anywhere | Medium: prompt/schema change; 27B cells are slow | Tests whether the 27B role effect is response mode rather than separation | 8–20 |

All four keep the fixed setup: Qwen3.5 bf16 on vLLM 0.30, 32,768 o200k budget with a 2,048 reserve, trigger at 18,432 that re-fires on each crossing, single-shot all-or-nothing JSON edits, untrained managers, seeds 42–49, paired differences per instance, and a sign test. Run greedy first and sampled where noted. Before any new GPU run, two zero-GPU log checks would sharpen the report. The first counts how many surviving needle lines in `peer_27b` episodes sit in manager-written notes vs untouched original blocks (retention by copy vs retention by sparing). The second tabulates protocol-invalid manager replies per size, so experiment 3 has a baseline.

### 1. Archive-with-recall turns KV deletion reversible

**Question.** Is the KV R520 collapse caused by management itself, or by the irreversibility of deletion? Once deletion is reversible, does manager identity still matter?

**Conditions.** Five arms:
- `none_r`: none, run with the new instruction (a prompt-change control).
- `self_arc`, `peer_same_arc`, `peer_27b_arc`: the existing managers, with `delete` turned into `archive`.
- `mask_arc`: a rule-based oldest-first eviction to archive, the analogue of VISTA's Auto-Archive + Recover.

The existing delete arms serve as the comparison.

**Actors and levels.** 4B and 9B (both near 1.0 under `none` at R520), and 2B optionally (`none` 0.46). Levels are R520 and R1000. At R1000 the values (~35K o200k) exceed the budget, and `none` scores 0.00 because it hits the wall at turn 6, before any GET. The delete arms reach only 0.07–0.25 lenient there, from whichever last SET batch survives (round 9, 4 seeds). That makes R1000 the only level where management can show a large *positive* KV effect. Extend the R1000 `none` and delete arms to 8 seeds. 8 seeds, greedy; repeat sampled at R520 for 9B only.

**Harness change.**
- Deleted block text goes into a per-episode archive instead of being discarded, and the context keeps a one-line placeholder `[block N archived]`.
- The KV instruction allows one command, `RECALL <key>`. The harness greps the archive for lines containing that key and appends them as a new block. The recalled text enters live context, which is what ContextBench scores, and counts against the budget.
- No other tools, and the actor still answers in context.
- Log RECALL count, hit rate and recalled tokens.

**Expected outcome.** Archive arms recover most of the R520 loss for 9B, and partly for 4B, because the RECALL protocol is simple and keyed. At R1000 the archive arms beat `none`. `mask_arc` lands close to the LLM-manager archive arms. That is what VISTA's finding (gains come mostly from safe eviction, with rare recall) predicts.

**Interpretation.**

| Result | Interpretation |
|---|---|
| Archive arms ≈ `none` at R520 and > `none` at R1000 | The KV harm was irreversibility, not management. With offload, management becomes net positive on KV, and the role/capability question moves to *retrieval*. |
| Recovery for 9B but not 4B, with 4B RECALL count near 0 | The bottleneck is the actor's retrieval initiative, a capability threshold on the actor side. |
| `mask_arc` ≈ `peer_27b_arc` | Once deletion is reversible, LLM judgment adds nothing on KV. This supports rule-based eviction, as in the Complexity Trap. |
| `peer_27b_arc` > `mask_arc` and same-size arms | Manager capability matters even with offload, extending the Needle finding to KV. |
| RECALLs issued and hitting, yet scores stay low | Check value corruption from `replace` edits or placeholder parsing before interpreting. |

**Cost.** About 2–3 days of harness work. Compute is 2–3 actors × 2 levels × 2 three-condition cells × 0.25–0.7 h, plus sampled reruns: about **6–12 H200-h**.

### 2. A manager-size ladder locates where editing judgment appears

**Question.** At what manager size does zero-shot external management start helping, and is 9B's harm as `self` an editing failure or an actor-side failure? 9B as `self` deletes needles *and* submits early. As a pure manager it cannot submit, so `peer_9b` isolates its edit judgment.

**Conditions.** `peer_9b` on 0.8B, 2B and 4B actors, and `peer_4b` on 0.8B and 2B actors. Reuse the existing `none`, `self`, `peer_same` and `peer_27b` cells. Optionally add `peer_9b` on 4B at KV R520 (8 seeds).

**Levels and seeds.** Needle K4-C24 and K8-C48, 8 seeds, greedy and sampled.

**Harness change.** None beyond a manager-model setting, with the 9B or 4B manager co-located with the actor on one H200 as `peer_27b` already is.

**Expected outcome.** `peer_4b` ≈ `none`, given 4B's 1–2 applied edits as `peer_same`. `peer_9b` is the informative cell. CLM's 9B edits sparsely and runs near the wall ([CLM](https://arxiv.org/html/2609.37725)), so the prior favors ≈ `none` or slight harm.

**Interpretation.**

| Result | Interpretation |
|---|---|
| `peer_9b` helps 2B/4B substantially (e.g. ≥ +0.3 at K4-C24) | The capability threshold lies between 4B and 9B. 9B's `self` harm then comes from the actor role (early submission, editing its own work), the clearest *role* effect below 27B. |
| `peer_9b` ≈ `none` or harmful on every actor | Editor capability needs more than 9B zero-shot. The 9B `self` harm is judgment, and "capability, not role" holds across the small-actor range. |
| `peer_9b` helps only at K4-C24, not K8-C48 | Capability needed scales with pressure, matching the shrinking 27B gain at K8-C48. |
| `peer_4b` > `none` | Contradicts the protocol-failure reading of 4B. Re-check against experiment 3. |

**Cost.** Five actor–manager pairs × 2 levels × 2 decodings at about 0.1–0.3 h per condition-run: roughly **5–12 H200-h**, all parallelizable. Can start on day 1.

### 3. Protocol-vs-judgment controls bracket the 27B gain

**Question.** Does `peer_27b` help Needle by selective retention, or just by freeing space? Do same-size managers fail because they cannot emit valid edits, or because their edits are bad?

**Conditions.** Two arms per actor:
- `mask`: a deterministic manager that, at each trigger, deletes the oldest non-protected blocks until the context falls below about 50% of budget, emitting the same JSON edit list through the same code path. No LLM call; this is the JetBrains baseline.
- `self_cj` and `peer_same_cj`: the existing same-size managers with vLLM JSON-schema structured output on manager calls only. The actor stays unconstrained ("reason free, constrain late").

**Actors and levels.** `mask` on 0.8B, 2B, 4B and 9B at Needle K4-C24, K8-C48 and KV R520. Add 27B at K4-C24 with 4 seeds if a job slot is free; it is cheap because there are no manager calls. `_cj` arms on 0.8B, 2B and 4B at Needle K4-C24 and KV R520. 8 seeds, greedy. Report the schema-valid rate, applied edits and "valid but harmful" edits (needle-containing blocks deleted) separately.

**Harness change.** A rule manager class, plus a schema argument on the manager request. The schema should restrict ids to the editable blocks (an enum built per event). Otherwise 2B's "delete 0..N" replies stay schema-valid but are rejected as protected-id edits.

**Expected outcome.** `mask` ≈ `none` or below on Needle, well under `peer_27b`. It frees space, so the episode runs on past the wall, but it drops needles wholesale. The early pilot in which 27B wiped every editable block scored 0.17 vs 0.33 for `none`. On KV R520, `mask` falls into the same 0.03–0.28 band as the LLM managers. Constrained decoding raises applied edits for 0.8B–4B to near 100%, but scores stay ≈ `none` or drop.

**Interpretation.**

| Result | Interpretation |
|---|---|
| `mask` ≈ `none` ≪ `peer_27b` on Needle | The 27B gain is selective retention, not space freeing. This is the strongest single control for the headline claim. |
| `mask` ≈ LLM managers on KV | On KV, every zero-shot manager behaves like a recency rule. The KV failure is about the mechanism (experiment 1), not who manages. |
| `mask` > LLM managers on KV | Zero-shot managers delete *worse* than recency, an active judgment failure. |
| `_cj` arms ≈ `none` or worse despite valid edits | Same-size failure is judgment, not format. "Capability, not role" holds even at 0.8B–4B. |
| `_cj` arms approach `peer_27b` | Protocol was the bottleneck, so the size ladder partly measures format ability. The report must say so. |

**Cost.** `mask` runs have no manager inference (about 0.1–0.3 h per actor-level for small actors). The `_cj` arms run small managers only. Total about **5–12 H200-h**. Harness work is about 1 day.

### 4. Stretch: does planning close the 27B role gap?

**Question.** Is `peer_same` > `self` for the 27B actor caused by separation, or by the peer's planning response mode?

**Conditions.** `self_plan`: the self-management reply must contain a `plan` field (per block: keep / note / delete, with reason) before `edits`, in the same single call. Compare with the existing `self`, `peer_same` and `none`.

**Actor and levels.** 27B at Needle K4-C24 and K8-C48. 8 seeds greedy, plus sampled at K4-C24, where the gap shrank to +0.06.

**Harness change.** A prompt change and a schema field, with no extra calls.

**Expected outcome.** `self_plan` closes part of the greedy gap.

**Interpretation.** If `self_plan` ≈ `peer_same`, the "role effect" is a response-mode effect that prompting removes. The report should then reframe role separation as eliciting deliberate planning, which is also the first single-shot vs plan comparison in the literature. If `self_plan` ≈ `self`, separation itself (a fresh session, not editing one's own work) carries the effect. Mixed greedy/sampled results would confirm that the role effect is decoding-fragile, and the report should say so plainly.

**Cost.** One new condition at about one-third of a 5–17 h cell, per level and decoding: **8–20 H200-h**. Launch early in parallel, or drop it if slots are tight.

### Candidates rejected for this window

The **trigger-threshold sweep** is low marginal value. Round 12 already moved the trigger to 27,000, which took 4B `peer_same` KV from 0.27 to 1.00 while 27B stayed near 0.28. That shows the problem is *what* is deleted, not *when*, and ACON already reports the threshold direction ([ACON](https://arxiv.org/html/2510.00615)). **Manager-decided triggering and token hints** require actor-side self-triggering. CLM and VISTA show models need explicit dashboards to know their size, so this becomes a separate project. Per-block token counts could be folded into experiment 1's placeholders later. A full **multi-step agentic manager** is too large a harness change, and experiment 4 is its minimal slice. An **AdaCoM-style 4B manager on the 27B actor** costs 5–17 h per cell, and AdaCoM's untrained column (22.01 vs 24.17 ReAct) plus 4B's 1–2 applied edits predict a null. Run it only as a 4-seed K4-C24 add-on if a slot opens.

## Conclusion

The project's distinctive contribution is the decomposition itself. Prior art either assumes self-management (CLM, OpenAI, Scroll), or picks an external manager for cost and then trains it (AdaCoM, ACON, Letta), so nobody has asked whether the external manager wins because it is external or because it is stronger. Our data already favor capability over role for small actors. Experiments 2 and 3 can turn that from a two-point contrast (same-size vs 27B) into a curve with protocol controlled. Experiment 1 addresses the one result that currently undercuts any pro-management story: KV Store. If archive-with-recall reverses the collapse, the paper's message changes from "managers help Needle but wreck KV" to "managers help when their mistakes are recoverable, and only strong managers make few enough mistakes to help when they are not." That is a sharper and more transferable claim. It aligns with how products actually ship (offload plus restorable compression), and ContextBench's live-context scoring makes it testable in a way that agentic benchmarks do not.
