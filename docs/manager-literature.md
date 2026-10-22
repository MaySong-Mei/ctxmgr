# Manager / Scaling Literature Brief (Track 3)

_Status: draft by an AI research assistant (2026-10-17); key IDs (CLM arXiv:2609.37725, AdaCoM arXiv:2605.30785) re-checked against arXiv abstract pages. Entries tagged [ABS]/[WEB] need full-text reading before report claims._

Project: *Who should manage the context?* (CPSC 5880). Compiled 2026-10-17.

**Reading-depth legend.** Each entry is tagged with how closely it was read:
- **[FT]**: targeted full-text reading of the arXiv HTML. Specific sections were queried, but not every page was read line by line.
- **[ABS]**: abstract or landing page only.
- **[WEB]**: official blog or docs page.

Numbers come from the cited source. Anything I could not confirm is marked *unverified*.

Scope note: Track 1 covers ContextBench as a benchmark. This brief covers who does the managing and how that interacts with model scale.

---

## 1. The CLM paper: what it does and does not test (most important)

**Citation.** Shao, Shen, Yin, Li, Wang, Ivison, Poovendran, Lambert, Xiao, Lewis, Yih, Zettlemoyer, Koh. *Context Language Models.* arXiv:2609.37725, submitted 29 Sep 2026. https://arxiv.org/abs/2609.37725 (title and ID verified). ContextBench is released at https://github.com/RulinShao/ContextBench, and the repo README links the same paper. **[FT]**, with caveats at the end of this section.

**What it proposes.** A CLM treats its own context as a file that it can freely edit. CLMs are built zero-shot from existing models, can learn editing strategies in context or through RL weight updates, and can be steered with natural language. The paper also proposes *Suffix Cache Reuse* for serving efficiency (https://arxiv.org/abs/2609.37725).

**Context-editing "tools".** Section **"4.1 Formal Definition and Implementation with Context as a File"** describes no set of named editing primitives. The live context is mirrored to a file, the model edits that file with ordinary **Bash** commands, and the edits are synced back before the next generation. If the model does not edit, new output is appended as usual. Named tools such as `manage_context` and `query_memory` appear only in the *baseline* harnesses (Appendix D / the ContextBench repo). In the paper's words, CLMs "delegate full responsibility for maintaining the context to the CLM itself" (https://arxiv.org/html/2609.37725).

**Multi-agent experiments.**
- Section 4.1 says "an agent swarm can be implemented by initializing the workspace with multiple context files." Each file stays synchronized with its own LLM server, and subagents are started and stopped by creating and deleting context files.
- Experiments in **"5.1 Evaluating CLMs Zero-Shot on Long-Horizon Agentic Tasks"** (5.1.1 / 5.1.2):
  - *Software World*: six agents jointly optimize interdependent repositories, with GPT-5.6-Sol and a 272K budget.
  - *EdgeBench-10 with subagents*: up to five concurrent subagents, which the authors say added little on a single-repo benchmark.
- Figure 3(a) shows a CLM building "in-context scoreboards and trackers to orchestrate and monitor subagents."
- **"5.2 Learning Better Context-Management Strategies in Context or in Weights"** includes *assisted evolution*. A proposer model (Claude Fable 5.1) reads Qwen3.6-27B's traces and writes candidate *skill documents*, and the best one is selected on a dev split. This is offline skill or prompt optimization by a stronger model. It is **not** live editing of another model's context.

**What was NOT tested.** I found no experiment in which one model edits a *different* actor's context at runtime. That rules out:
- a peer manager of the same size;
- a larger fixed manager;
- any comparison that separates role separation from manager capability.

Two further gaps:
- The paper does not say whether swarm agents can read or write each other's context files, or whether all swarm agents are the same model.
- **"6 Discussion and Future Work"** discusses safety risks of editable context and distilling harnesses into CLMs. It does not propose external managers.

**Scale-related evidence in CLM.**
- **Appendix F** ("…TerminalBench 2.1, TBLite and BrowseComp-Plus with models of different sizes") compares Qwen3.5-9B and Qwen3.6-27B on TerminalBench 2.1:
  - The 9B model edits less: 1.4 edits per task, with no edits in half of tasks, versus 2.6 for 27B.
  - The 9B model runs close to the limit: median peak context 30.2K of 32K, versus 17.6K for 27B.
  - The 9B model initially trails the summary harness because of "limited context-management capabilities." On BrowseComp-Plus, CLM-9B scores 39.9% vs 37.7% for the summary harness, and CLM-27B scores 59.4%.
  - The authors say CLM gains grow with the model's ability to decide when and how to edit.
- **"3 Pilot Study with ContextBench"** and **Appendix E** run on **GPT-5.4 only** (32,768-token limit, 2,048 reserved for the response). The pilot compares Mini-SWE-Agent (no management), Codex-style summary, Context Folding, RLM, Self-Compact, ACM and CLM. Summary compaction "can lose or hallucinate information" on Needle Retention, and no method was perfect.
- **Appendix G** reports that LMs estimate their own context length poorly, and that token-count hints help.

**Caveats.**
- The Appendix D body (ContextBench task details) was not visible in my reading.
- The Software World and swarm descriptions were read only at the level summarized above.
- The negative claim ("no peer or larger manager tested") rests on the main text, Sections 4–6 and Appendices E–G. Someone should skim Appendix D and the full PDF once before we cite it as settled.

---

## 2. Memory and context systems: same model vs. separate manager

| System | Who manages memory | Separate vs. same compared? | Read |
|---|---|---|---|
| **MemGPT** (Packer et al., 2023) https://arxiv.org/abs/2310.08560 | The same LLM, through function calls: "Memory edits and retrieval are entirely self-directed" (§2.3). | No. It does report that MemGPT "has significantly degraded performance using GPT-3.5, due to its limited function calling capabilities," and that GPT-3.5 fails nested KV lookups beyond 2 levels. This is early evidence that self-management depends on capability. | [FT] |
| **Letta sleep-time agents** (Letta blog, 2025) https://www.letta.com/blog/sleep-time-compute | A **separate** sleep-time agent holds the core-memory editing tools for itself *and* the primary agent. The primary agent cannot edit core memory. | The models can differ. The blog suggests a fast primary (e.g. gpt-4o-mini) with a stronger, slower sleep-time model (e.g. gpt-4.1 / Sonnet 3.7). No controlled comparison is reported there. | [WEB] |
| **Sleep-time Compute** (Lin et al., 2025) https://arxiv.org/abs/2504.13171 | Offline precomputation over context before queries arrive. | The abstract does not say whether a different model is used. | [ABS] |
| **A-MEM** (Xu et al., 2025) https://arxiv.org/abs/2502.12110 | Zettelkasten-style notes, links and memory evolution driven by an LLM. | Tested on six foundation models. Whether the organizer is the same model as the agent is not stated in the abstract (*unverified*). | [ABS] |
| **MemoryBank** (Zhong et al., 2023) https://arxiv.org/abs/2305.10250 | Forgetting-curve-based memory update for a chatbot (SiliconFriend). | No manager comparison. | [ABS] |
| **Mem0** (Chhikara et al., 2025) https://arxiv.org/abs/2504.19413 | A fixed **GPT-4o-mini** pipeline runs extraction and update (ADD/UPDATE/DELETE/NOOP). It is effectively a dedicated manager. | "No ablation" of manager model choice was found. | [FT] |
| **Recursive summarization** (Wang et al., 2023) https://arxiv.org/abs/2308.15022 | The LLM recursively merges chunk summaries into a running memory. | Applied to open and closed LLMs. No separate-vs-same comparison. | [ABS] |
| **MemAct** (Zhang et al., 2025/26) https://arxiv.org/abs/2510.12635 | The agent itself deletes and inserts context through RL-trained actions. | It argues that external controllers "lack awareness of the agent's reasoning state," but the abstract shows no head-to-head. It reports that MemAct-RL-14B matches models 16× larger with 51% shorter context. | [ABS] |
| **MEM1** (Zhou et al., 2025) https://arxiv.org/abs/2506.15841 | RL-trained internal state consolidation (self-managed). | MEM1-7B beats Qwen2.5-14B-Instruct by 3.5× in performance with 3.7× less memory on 16-objective QA. | [ABS] |
| **Context-Folding** (Sun et al., 2025) https://arxiv.org/abs/2510.11967 | The agent branches and folds its own sub-trajectories (self-managed, RL). | No external-manager comparison is reported (*from secondary listings; abstract not directly fetched*). | [ABS*] |
| **ReSum** (Wu et al., 2025) https://arxiv.org/abs/2509.13313 | Periodic summarization by an "external tool." | Gains of 4.5% (training-free) and a further 8.2% (ReSum-GRPO) over ReAct. Whether the summarizer is a different model is not stated in the abstract. | [ABS] |
| **Self-GC** (Hao et al., 2026) https://arxiv.org/abs/2607.00692 | A **side-channel planner** proposes fold, mask and prune actions, and the harness enforces them. | Three planner backbones (unnamed in the abstract) reach 91–95% no-impact rates. There is no self-vs-external ablation in the abstract. | [ABS] |
| **VISTA** (Xu et al., 2026) https://arxiv.org/abs/2606.30005 | Self-managed, with a token-usage "proprioception" dashboard. It argues that frontier LMs are "proprioceptively blind to their own context." | Gemini-3-Flash goes from 22.7% to 50.7% on LOCA-Bench. No size study in the abstract. | [ABS] |
| **Coding-agent compaction** (Anthropic, 2025) https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents | Compaction passes the message history "to the model to summarize." It also describes note-taking and sub-agents that return condensed summaries. | Whether the compactor is the same model is not stated. | [WEB] |
| **LightMem** (Zhang et al., ACL 2026) https://arxiv.org/abs/2604.07798 | **Small LMs** run memory operations (retrieve, write, consolidate) for an agent. | The abstract claims results "across model scales." The agent model is not named in the abstract. | [ABS] |

**The closest prior art: AdaCoM.** Yi et al., *Learning Agent-Compatible Context Management for Long-Horizon Tasks*, arXiv:2605.30785, May 2026. https://arxiv.org/abs/2605.30785 **[FT]**

- **Setup.** An external **Qwen3-4B** manager, trained with SFT and then GRPO, edits the context of a *frozen* agent before each step.
  - Main agents: Qwen3-Max, Kimi-K2, GLM-4.5-Air, DeepSeek-V3.
  - Transfer agents: GPT-OSS-20B, Gemini-2.5-Flash, Seed-1.6-Flash, GPT-4o-mini.
- **Results on BrowseComp-Plus, mean@3 (Table 1, §4.2).** Averages across agents:

  | Condition | Average |
  |---|---|
  | Vanilla ReAct | 24.17 |
  | MemAct (agent self-management) | 25.72 |
  | SumAgent (agent self-management) | 22.06 |
  | **Untrained** 4B manager | 22.01 |
  | 4B summary-only manager (SumCoM) | 29.39 |
  | AdaCoM (trained 4B manager) | 33.60 |

- They report a **"Fidelity–Reliability Trade-off"**: weaker agents benefit from more aggressive compression, stronger agents from preserving more context.
- **What it leaves open for us:**
  - Its Limitations section says **larger managers were not trained**.
  - It does **not** separate "external role" from "manager capability."
  - Managers are always *smaller* than, or different from, the agent, and they are trained per agent.
  - It does not use ContextBench, and it does not test same-size or larger-than-actor managers on small actors.

Other 2026 papers that frame the design space: *Context as an Environment* (Scroll), https://arxiv.org/abs/2608.21690, and AgentEvolver's Context Manager, https://arxiv.org/abs/2511.10395. Both are **[snippet only, unverified details]**.

---

## 3. Small-vs-large delegation, prompt compression, and scaling

**A stronger model's skill distilled into a smaller manager.** ACON (Kang et al., 2025; ICML 2026), https://arxiv.org/abs/2510.00615 **[FT]**:
- It optimizes natural-language compression guidelines from failure analysis, then distills the compressor into Qwen3-14B, Qwen3-8B and Phi-4. The distilled students keep >95% of the gpt-4.1 teacher's performance (§4.3).
- With a Qwen3-14B *agent* and distilled compressors (§4.4), AppWorld rises from 25.6% to 33.9% and 8-objective QA from 0.158 to 0.23 EM. The abstract summarizes this as up to 46% gains for small agents.
- **Not tested:** a large compressor serving a small agent directly, or self-compression versus a separate compressor. All reported setups use a separate compressor.

**Small model reads, large model reasons.**
- *Minions* (Narayan et al., 2025), https://arxiv.org/abs/2502.15964 **[ABS]**:
  - The naive local–cloud chat protocol recovers only 87% of frontier performance. The authors attribute this to the small model's trouble following multi-step instructions.
  - Decomposed MinionS recovers 97.9% at 5.7× lower cost.
  - This is directly relevant to our confound: small-model *protocol following* is the bottleneck.
- *Chain of Agents* (Zhang et al., 2024), https://arxiv.org/abs/2406.02818 **[ABS]**: worker agents read the input chunks in sequence and pass information along, and a manager agent combines their output into the answer.

**Prompt and RAG compression (separate compressor, frozen reader).**
- LLMLingua reports up to 20× compression with little loss: https://arxiv.org/abs/2310.05736 **[ABS]**.
- LLMLingua-2 distills an LLM into small encoders (XLM-RoBERTa-large, mBERT). It achieves 2–5× compression and 1.6–2.9× end-to-end speedup: https://arxiv.org/abs/2403.12968 **[ABS]**.
- RECOMP trains extractive and abstractive compressors for the reader LM, reaching rates as low as 6%. Compressors transfer across LMs on language modeling: https://arxiv.org/abs/2310.04408 **[ABS]**.
- These are single-shot compressions of retrieved or prompt text, not stateful management of a stream. That makes them a weaker analogue to our setup.

**Scaling of self-management ability.**

Direct evidence:
- MemGPT: GPT-3.5 fails where GPT-4 works.
- CLM: the 9B model edits less and runs near the limit compared with 27B.
- AdaCoM: the right management policy depends on actor capability.

Counter-evidence, where RL makes small self-managers competitive:
- MemAct-14B is comparable to 16× larger models.
- MEM1-7B beats a 14B model.

I found **no systematic sweep** of self-management ability across a dense model family (e.g. 0.8B → 27B) under a fixed budget. That gap is ours to fill.

---

## 4. The confound: format and protocol compliance in small models

Our observation is that 0.8B–4B actors fail mostly on output format rather than memory. The literature consistently says this must be measured separately.

**Small models fail on format specifically.**
- *Small Models, Big Tasks* (Kavathekar et al., EASE 2025): SLMs "struggle significantly with adhering to the given output format" in function calling. https://arxiv.org/abs/2504.19277 **[ABS]**
- FoFo: open models lag closed models on format adherence, and "format-following performance is independent of their content generation quality." https://arxiv.org/abs/2402.18667 **[ABS]**
- Minions (above): small local models struggle to follow multi-step instructions.

**Format requests themselves hurt reasoning: the "format tax".**
- *Let Me Speak Freely?* (Tam et al., 2024), https://arxiv.org/abs/2408.02442 **[FT]**:
  - Format restrictions significantly reduce reasoning, and stricter formats hurt more.
  - Methodologically important for us: the authors use an **LLM answer parser** ("perfect text parser", Claude-3-Haiku, κ=0.86 vs GPT-4-turbo) to "disentangle format errors" from content errors.
  - Strict regex badly understates performance: GPT-3.5-Turbo scores 43.7 with regex vs 75.5 with the LLM parser on GSM8K (App. E).
  - They also test NL-to-Format, a two-step answer-then-reformat method.
- *The Format Tax* (Lee, D'Antoni, Berg-Kirkpatrick, 2026), https://arxiv.org/abs/2604.03616 **[ABS]**:
  - Most of the loss comes from the format *instruction*, not the decoder constraint.
  - Separating reasoning from formatting (a second reformatting pass, or extended thinking) recovers most of the loss.
  - Recent closed models show little tax.

**Constrained decoding fixes syntax, not semantics, especially for small models.**
- *The Constraint Tax* (Ray, 2026), https://arxiv.org/abs/2605.26128 **[ABS]**. Models: Qwen2.5-0.5B / 1.5B and SmolLM2-1.7B.
  - Hard schema decoding raised validity from 61.5% to 100% but cut accuracy from 19.7% to 11.0%.
  - On a tool-call task, Qwen2.5-1.5B scored 91.5% with prompt-only JSON vs 48.0% under the hard schema.
  - Recommendation: "reason free, constrain late," and report validity, accuracy and wrong-but-valid rate separately.
- *Constrained Decoding Eliminates Structural Failures in Small LLMs but Reveals a Scale-Dependent Semantic Gap* (Chavan, 2026), https://arxiv.org/abs/2609.23742 **[ABS]**. Models: 0.6B–4B.
  - Outlines and XGrammar raise schema validity from 78.6–92.9% to 100%.
  - Multi-step function-calling semantics stay broken, and the semantic gap is scale-dependent.
- JSONSchemaBench evaluates six constrained-decoding frameworks on 10K schemas: https://arxiv.org/abs/2501.10868 **[ABS]**.

**Lenient scoring precedents.**
- IFEval reports **strict** vs **loose** accuracy. Loose counts a response as compliant if any of 8 transformed versions passes (markdown stripped, first or last line dropped). The authors note this "reduces false negatives" but may add false positives. https://arxiv.org/abs/2311.07911 **[FT]**
- *Instruction-Followed Function Calling* hands function-call formatting to a smaller model, separating protocol from reasoning: https://arxiv.org/abs/2608.22472 **[ABS]**.

---

## Implications for our design

- **Report two error axes for every cell.** Track protocol or format violations (unparseable tool calls, malformed answers) separately from memory errors (well-formed but wrong answers, lost needles). Then also score each run with a lenient parser. Following IFEval and *Let Me Speak Freely?*, the effect of a manager should be judged on the lenient, memory-only metric, with the strict metric as secondary.
- **Separate the actor's format burden from the management condition.** One option is a "format-assist" control in which every actor gets constrained decoding or a reformat pass with no context management. Without it, a 27B manager might appear to "help" small actors only by absorbing protocol work. The Constraint Tax and Chavan 2026 warn that hard constraints alone can hurt small-model semantics, so prefer "reason free, constrain late."
- **Our 2×2 isolates what AdaCoM could not.** Same-size peer vs self-management isolates the **role** effect. 27B peer vs same-size peer isolates the **capability** effect. AdaCoM, ACON and Letta all bundle the two.
- **Expect an interaction with actor strength.** AdaCoM's Fidelity–Reliability trade-off and CLM's 9B-vs-27B gap suggest that management policy, and its benefit, will vary with actor size. Plot effects against actor size rather than pooling.
- **Use CLM's own diagnostics.** Log edit counts per task and peak context relative to the 32K limit, so our small-model numbers are comparable to CLM's Appendix F (9B: 1.4 edits per task, peak 30.2K).
- **Avoid summary-only managers as the main peer baseline.** CLM's pilot and AdaCoM's SumCoM both show that summarization loses verbatim needles. Give the peer manager the same edit actions the self-managing actor has.

## Open prior-art risk

- **AdaCoM (arXiv:2605.30785)** is the main risk. It already shows that a separate, smaller, *trained* manager beats agent self-management on a frozen agent. Our novelty claim must therefore be narrow:
  - untrained (zero-shot) managers;
  - **same-size vs larger** managers on a **dense size sweep of small actors**;
  - ContextBench KV Store and Needle Retention under a fixed 32K budget;
  - explicit separation of role and capability effects, with format confounds controlled.
- **CLM.** I found no peer or larger-manager experiment in CLM, but the Appendix D body and the swarm details were not fully visible. Confirm in the PDF that no "external manager" ablation exists before stating it in the report.
- **Not fully checked.** Self-GC's side-channel planner, Scroll, AgentEvolver, E-mem (https://arxiv.org/abs/2601.21714) and LightMem were checked only at abstract or snippet level. Any of them could contain a self-vs-external ablation.
