# New seeds re-test one layout per level

The proposed validation is worth running, but not as designed. Its biggest risk turned out real in a different form than the notes feared. Seeds do produce distinct instances: every one of seeds 42–49 has its own words, hashes and per-op token counts. But for Needle Retention, **every seed has the identical task structure**: the same number of ops, the same 147 lines per chunk, needles on the same lines of every chunk, and fillers in the same positions. Seeds 50–57 will repeat that structure exactly, because the pinned generator draws only words from the seed. KV Store seeds differ in one way that matters: which keys are queried. So "8 new seeds" is 8 lexical re-draws of one layout per level. That is a valid within-layout replication, but its sample size for "does the rule generalize across task structures" is the number of **levels**, not seeds. Part A's cost claim also shrinks. Recomputed from the per-call logs, 2B + 27B manager costs **0.16×** the 27B `peer_same` baseline under the current no-cache count, but about **0.32×** once prefix reuse is approximated. The 4B and 9B actors (0.81 and 0.80) do not yet match 27B `peer_same` (0.85) within any margin testable at n = 8. Part B needs typed predictions (direction plus margin), one-sided exact tests with Holm correction, equivalence margins for its "no difference" claims, 16 seeds where power demands it, and at least two held-out official levels. The revised design below costs about **34 H200-hours, or ~42 with contingency**.

## Every seed shares one Needle layout; KV seeds vary only the queried keys

The check compared the stored instance files for all eight seeds at Needle K4-C24 and K8-C48 and KV R520 and R1000 (runs `peer27b-30088182`, `peer27b-30127436`, `peer27b-30265858` and others in `runs/`). It tokenized every op with o200k and regenerated instances locally with the pinned generator. Copies of the same seed are byte-identical across runs (one chunks hash per seed everywhere). Different seeds are genuinely different texts: all eight per-op token vectors differ, with K4-C24 ops ranging 3,902–4,046 tokens and whole streams 95,082–95,503 tokens (a 0.44% spread). **Structurally there is exactly one layout per level.** Every K4-C24 seed has 24 ops of 147 lines, needles on lines 1–4 and the filler sentinels at the same offsets. K8-C48 likewise has one layout across all 8 seeds, and so does K4-C4 across its 4. The generator explains why. `needle_retention/generate.py` and `kv_store/generate.py` at the pinned commit `fb533c1` seed a `random.Random(seed)` that only draws words, values and (for KV) the query keys. The layout comes entirely from the level's knobs in `configs/paper.yaml`. Regenerating seed 42 locally reproduced the stored K4-C24 and R520 chunks bit for bit. Seeds 50–57 reproduce the same Needle layout (stream totals 95,234–95,531) with new words only.

KV differs in one consequential way. The op layout is fixed (6 SET batches then 24 GETs at R520; 10 + 24 at R1000), but `rng.sample` picks a different set of queried keys per seed. So the number of queried keys per SET batch varies. At R520, for example, seed 42 has {5, 3, 6, 3, 7, 0} across batches 0–5 and seed 46 has {3, 6, 2, 3, 9, 1}. At R1000, seeds 50–57 put 3–5 queried keys in the last two batches. This is what drives the delete arms' 0.17–0.29 scores, because those come from whichever batches survive the deletions. So KV seeds are partly structural replicates; Needle seeds are not.

The run log's line "seeds change the words, not the structure" therefore still holds for the current generator. Its corollary, "42 = 44, 43 = 45, down to token counts", does not. In job 30043145 those pairs had equal rewards and turns, but token counts differed slightly (for example, `none` K4-C24 used 165,891 vs 165,644 actor tokens) ([episodes_vllm.csv](../docs/tables/episodes_vllm.csv)). In the H200 log, the pattern holds in only **12 of 93** eight-seed cells whose rewards vary at all. What the log actually shows is outcome collapse. Of 137 eight-seed cells, **44 (32%) give the same reward on every seed** (17 of them `none` arms, which hit the budget wall at the same op every time), and the median cell has only 2 distinct values. Greedy decoding on one layout, with rewards quantized in steps of 1/24 or 1/96, produces few distinct trajectories.

| Level | Seeds checked | Distinct layouts | Per-op token vectors | What the seed changes |
|---|---|---|---|---|
| Needle K4-C24 | 42–49 (+50–57 generated) | 1 | 8 distinct (16 with new seeds) | words and hash tags only |
| Needle K8-C48 | 42–49 | 1 | 8 distinct | words and hash tags only |
| KV R520 | 42–49 | 1 op layout | 8 distinct | values plus which 24 keys are queried |
| KV R1000 | 42–49 (+50–57 generated) | 1 op layout | 8 distinct | values plus which 24 keys are queried |

## Eight instances are valid for a layout, but the rule's real sample is a handful of levels

The finding cuts both ways. **The fear that seeds are duplicates is wrong.** Instances are distinct, and greedy reruns on a fixed stack are near-deterministic ([CLAUDE.md](../CLAUDE.md), round 18). So the eight paired differences in a cell are eight genuine observations, and exact paired tests on them are valid. But the population they sample is narrow: "this level's fixed layout, with random lexical content". Miller frames inference as targeting a super-population of questions ([Miller 2024](https://arxiv.org/html/2411.00640v1)). Here the super-population that seeds sample is one structure per level. For Needle under `peer_27b`, the actor merely replies READY while the 27B manager decides what survives ([CLAUDE.md](../CLAUDE.md), round 5). So seed-to-seed variance there mostly measures how the manager reacts to different filler words. Three consequences follow for the statistics.

First, the rule's generalization has a small effective n. A claim like "on Needle, use a stronger external manager" rests on **2 layouts at 8 seeds** (K4-C24, K8-C48) plus 4 more at 4 seeds (K4-C4 to K4-C96, pressure curve). That makes 6 structural units for Needle and 2–4 for KV, and new seeds add none. In Patel et al.'s terms, new seeds are the weakest "held-out scenario" test (Criterion A). A cross-subset test (Criterion B) needs untested levels or task combinations whose results are predicted in advance ([Patel et al.](https://arxiv.org/html/2606.19704v1)).

Second, many paired tests are really one-sample tests. When `none` is constant across seeds (0.33 on K4-C24 for 2B), "8/8 positive, p = 0.004" just says the managed arm beat a fixed number on every lexical re-draw. That is still a valid claim, but it should be described that way. In ceiling cells (2B `peer_27b` = 1.00 on all 8 seeds) there is no variance to estimate at all. Any pooled analysis across levels must cluster by level, because Miller's clustered errors were up to about 3× the naive ones ([Miller 2024](https://arxiv.org/html/2411.00640v1)).

Third, power is low for anything but near-deterministic effects. With n = 8, the exact one-sided floor is 1/256 = 0.0039. That survives Holm across about 12 tests, while a two-sided 8/0 fails Bonferroni beyond k = 6 (computed in the methodology notes). Bootstrap intervals at this n under-cover, according to Bowyer et al. and Miller, so exact or paired-Bayes methods are preferable ([Bowyer et al.](https://arxiv.org/html/2503.01747); [Miller 2024](https://arxiv.org/html/2411.00640v1)). The observed spread matters for non-inferiority (NI) claims. The paired SD of 4B `peer_27b` − 27B `peer_same` on K4-C24 is 0.16, and for 9B it is 0.27. Showing non-inferiority within a 0.10 margin at 80% power therefore needs roughly **16 instances at SD 0.16 and about 45 at SD 0.27** (normal approximation, computed). Equivalence cells need 16 seeds, a wider pre-stated margin, or both.

## The "1/6 compute" claim holds only without caching and only for 2B

Part A's numbers were recomputed from `episodes_vllm.csv` and from per-call `calls.jsonl` logs (greedy, H200 bf16, trigger 18,432, 8 seeds). Cost is Σ params(B) × tokens(M) per role. The "reuse-approx" column credits an actor call with the previous actor prompt plus completion as a cached prefix when no edit came in between. It gives manager calls no reuse, since their prompts are rebuilt each time. This mirrors CLM's prefix-reuse accounting, but it is an approximation: real vLLM caching for hybrid Qwen3.5 runs in experimental, block-aligned "align" mode, and `cached_prompt_tokens` is currently logged as null ([vLLM APC docs](https://docs.vllm.ai/en/v0.30.0/features/automatic_prefix_caching/); [CLM](https://arxiv.org/html/2609.37725)).

| Needle K4-C24 (official reward) | Reward | No-cache B·Mtok | Reuse-approx B·Mtok | Ratio vs 27B `peer_same` (no-cache / reuse) |
|---|---|---|---|---|
| 2B + 27B manager (`peer_27b`) | **1.00** (8/8 at ceiling) | 4.33 | 3.73 | 0.16 / **0.32** |
| 4B + 27B manager | 0.81 | 5.25 | 4.08 | 0.20 / 0.35 |
| 9B + 27B manager | 0.80 | 7.56 | 4.93 | 0.28 / 0.42 |
| 27B `peer_same` (two 27B sessions; s42–45 v7, s46–49 v8) | 0.85 | 26.60 | ~11.7 | 1 / 1 |
| 27B `self` (v8) | 0.38 | 18.53 | 10.76 | 0.70 / 0.92 |
| 27B `none` | 0.29 | 3.67 | 1.04 | 0.14 / 0.09 |

The 2B result is strong. It is non-inferior to 27B `peer_same` at any margin (7 wins, 1 tie, 0 losses; exact one-sided p = 0.0039). It beats 27B `none` at roughly matched no-cache cost (1.00 vs 0.29 at about 4 B·Mtok). Three corrections are needed, though.

The ratio roughly doubles to about 1/3 under reuse. Append-only conditions are cached heavily (27B `none` drops to 9–28% of its naive cost), while every manager call is a fresh prompt. The 2B + 27B system is about 86% manager-dominated and gains little from caching.

**"Matches or beats" fails for 4B and 9B as tested.** Their mean differences vs 27B `peer_same` are −0.04 and −0.05, each with 3 wins, 1 tie and 4 losses. Non-inferiority at a 0.10 margin gives p = 0.18 and 0.31; even a 0.15 margin gives p = 0.05 and 0.17.

The baselines need care. 27B `peer_same` already pays for two 27B sessions and mixes edit versions. It also runs 43 turns per episode vs 24 for the small-actor systems, which inflates its cost. And because the Needle actor in `peer_27b` is a pass-through, the Needle comparison says little about capability substitution ([prior-art notes](https://arxiv.org/html/2502.15964) for the analogous Minions role reversal).

The KV half of Part A has no 27B comparator yet. The planned one is 9B `mask_arc` at R1000: 0.97 official reward at 6.72 no-cache / 0.52 reuse B·Mtok, with zero manager compute. But 27B `self_arc` has never run, and on vLLM bf16 the 27B has no R1000 data at all. The field's convention is to report cost–quality curves with matched-quality or matched-cost points, counting every role's compute rather than zeroing the "cheap" model ([AutoMix](https://arxiv.org/html/2310.12963); [FrugalGPT](https://arxiv.org/abs/2305.05176) [abstract only]; [Fireworks memo](https://fireworks.ai/research/open-source-worker-closed-source-advisor)). FLOPs, parameter counts and throughput can also disagree, so no single indicator should be reported alone ([Efficiency Misnomer](https://arxiv.org/abs/2110.12894) [abstract only]). Measured `latency_s` already exists per call, but it is confounded by co-location and concurrency. For example, 2B `peer_27b` manager time was 221 s per episode under v7 and 87 s under v8 for near-identical work.

## Part B is a sound idea that needs typed claims and a structural holdout

Pre-registering a decision rule and testing it prospectively is rare in the context-management literature. Benchmarks such as AgentMemBench state recommended defaults by task type, but derive them on the same data they report ([AgentMemBench](https://arxiv.org/html/2608.00009)). The git-commit timestamp is a defensible analogue of an OSF/AsPredicted timestamp if it lands before any seed-50+ output exists ([van Miltenburg et al.](https://ar5iv.labs.arxiv.org/html/2103.06944)). The local mirror holds no episode with seed ≥ 50. New seeds are also the right remedy for tuning edit-v5..v7 on seeds 46–49, since they are a fresh i.i.d. draw from an unchanged generator ([Dwork et al.](https://arxiv.org/abs/1506.02629) [abstract only]). Three weaknesses remain.

A bare hit rate is uncalibrated and ignores magnitude. "No difference" branches, such as "manager identity stops mattering under archive", need TOST or non-inferiority bounds, because a non-significant difference is not evidence of equivalence ([TOSTER vignette](https://cran.r-project.org/web/packages/TOSTER/vignettes/IntroductionToTOSTER.html) [summary only]).

Seeds alone cannot test the rule's structural claim, as the previous section showed. The two untested combinations are the only genuine Criterion-B tests in the proposal, and they should be labelled separately from the seed replications.

The rule must be scoped to greedy decoding. Under recommended sampling the 27B role effect at K4-C24 vanished (+0.48 → +0.06, 4/4), and the 0.8B gain lost significance ([results.md](../docs/results.md)). Batch-variant kernels can also break temperature-0 determinism across stacks ([Thinking Machines](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/)). The project itself saw this across GPU types (2B Needle K4-C4 ran 14 turns on one GPU type and 11 on another), so the GPU type and serving stack must be pinned.

Unused official levels supply the structural holdout at no generator change. KV R2000 (pressure 3.1) is tested only with 2B. Needle K2-C192 (a new needle density, pressure 23) is untested. K4-C6 and K4-C8 are sub-budget negative controls, where management is predicted not to matter. The `log_triage` family would be a true out-of-family test, but it needs a scorer check first.

## Revised design: keep the core, add levels, type every claim

**Keep:**
- The git-committed pre-registration.
- Seeds 50–57, extended to 50–65 for the non-inferiority cells, labelled "within-layout replication".
- Both untested combinations, labelled "out-of-sample conditions":
  - 27B `self_arc` on KV R1000, adding 27B `none_r` and `mask_arc` as comparators;
  - 9B archive arms on Needle K4-C24.
- The no-cache cost, relabelled as an upper bound.

**Drop:**
- The bare hit rate as the primary metric.
- Pooling seeds 42–49 into confirmatory tests (report pooled n = 16 as secondary only).
- The 27B role-separation claim from the confirmatory family, since it is decoding-sensitive (keep it exploratory).
- "Matches or beats at 1/5–1/6" for 4B and 9B, until non-inferiority is shown.
- Mixing v7 and v8 in any baseline.

**Add:** two held-out levels (KV R2000, Needle K2-C192), a single-seed canary rerun of seed 42 to confirm the stack reproduces, and cached-token logging.

| Pre-registration item (commit, push, tag before submitting jobs) | Content |
|---|---|
| Frozen code | Repo commit; edit-v8 / v8-arc / rule-mask-v1 prompts; vLLM 0.30.0; bf16; H200 only; GPU utils (manager 0.55 / actor 0.38); trigger 18,432; `max_turns` rule; greedy decoding; thinking off |
| Cells and seeds | Each prediction's actor × condition × level; seeds 50–57 (50–65 for NI cells); the levels set to 16 seeds |
| Primary metric | Needle official reward; KV lenient recall (`scripts/lenient_kv.py`) as primary and official as secondary, fixed in advance (9B `self_arc` R1000 is 1.00 lenient but 0.75 official) |
| Typed predictions | Each one is *superiority* (direction + minimum effect), *non-inferiority* (margin), or *equivalence* (TOST margin), with the pass, inconclusive and fail rule written out |
| Exclusions | `server_rejected`, `max_steps` and crashes scored as-is (no reruns), with counts reported; a rerun is allowed only for infrastructure failure before turn 1 |
| Stopping | No seeds added after unblinding; the 16-seed cells are fixed now |
| Leakage statement | Seeds ≥ 50 never run, including smoke tests |
| Aggregate and falsification | Confirmed count vs Binomial(m, 0.5); per-branch kill criteria (e.g. "small-model self-management ≥ best external manager in ≥ 2 cells falsifies 'never self-manage small'") |

**Statistical plan.**
- *Unit of analysis.* The generated instance, paired across conditions on the same instance, with the population named as "level layout × random lexical content". Rule-level generalization is assessed per held-out level, and any pooling across levels is clustered by level.
- *Superiority tests.* One-sided exact sign-flip permutation tests on paired differences (256 flips at n = 8, 65,536 at n = 16), with Holm correction across a fixed confirmatory family of about 10–12 tests.
- *Intervals.* Clopper-Pearson or Wilson intervals on win proportions, or Bowyer's paired-Bayes method, instead of bootstrap intervals.
- *Margins.* Δ = 0.10 for Needle and KV non-inferiority and equivalence. The 9B cells and 9B KV R520 archive-vs-`none` (observed −0.07) are pre-declared at Δ = 0.15, or run at 16 seeds.
- *Ceiling cells.* Reported descriptively.

The concrete confirmatory set is:

| # | Prediction |
|---|---|
| 1 | Needle K4-C24: `peer_27b` − `none` ≥ +0.30 for 2B, 4B and 9B |
| 2 | Same-size `self` − `none` ≤ +0.05 (NI-style bound) for 2B and 4B |
| 3 | 9B KV R1000: `mask_arc` − `none` ≥ +0.70 |
| 4 | 9B KV R1000: `mask_arc` ≡ `peer_27b_arc` within ±0.10 (TOST) |
| 5 | 4B KV R1000: archive − `none` ≤ +0.20 (the "actor ≥ 9B" boundary) |
| 6 | 27B `self_arc` KV R1000: ≥ 0.90 lenient, and − `none_r` ≥ +0.70 |
| 7 | 9B Needle K4-C24: `peer_27b` − best archive arm ≥ +0.20, with no archive arm exceeding `none` by more than 0.10 |
| 8 | Part A: 2B `peer_27b` non-inferior to 27B `peer_same` (Δ = 0.10) at ≤ 0.5× reuse-aware FLOPs |
| 9 | Held-out KV R2000 (9B): archive arms > `none` |
| 10 | Held-out Needle K2-C192 (2B, 9B): `peer_27b` > `none` |

**Cost-reporting plan.** Report three accountings side by side, per role (actor vs manager):

1. *No-cache upper bound:* the current B·Mtok, converted to FLOPs with 2N.
2. *Prefix-reuse FLOPs:* CLM's Eq. 8–9, with per-size C_token and C_attn derived from each model's config. This includes the full-attention quadratic term, about 8% of a 20K-token 27B turn ([CLM](https://arxiv.org/html/2609.37725)). It uses measured cached-token counts once vLLM's prompt-token details are enabled (verify the flag on 0.30), and the logged-prompt approximation otherwise.
3. *Measured GPU-seconds per episode:* run at fixed concurrency on one pinned H200, so latency is not confounded by co-location.

Plot reward against cost for every configuration (actor sizes × `none`/`self`/`peer_same`/`peer_27b`/archive arms) and draw the Pareto frontier. State two numbers: the cheapest configuration reaching ≥ 0.80 and ≥ 0.95, and the reward at matched cost. The correct baselines are 27B `none` (single model), 27B `self` (the literal "manages its own context"), and 27B `peer_same`, flagged as a two-session system. For KV, add 27B `self_arc`/`mask_arc` at R1000 and 27B `none` at R520, which is near-free under reuse (0.97 B·Mtok).

**H200-hour budget.** Estimates come from summed per-call latencies, which match past job durations (round 14: an estimated 6.4 h vs 6.5 h actual). They assume episodes run sequentially and add about 0.3–0.5 h of startup per job.

| Block | Cells | H200-h |
|---|---|---|
| 27B actor `none`/`self`/`peer_same` | K4-C24 × 16 seeds, K8-C48 × 8 | 16.5 (K8-C48 `self` cuttable: −2.6) |
| 2B/4B/9B `none` + `peer_27b` | K4-C24 × 16, K8-C48 × 8 | 3.7 |
| 9B KV arc + delete arms, 4B KV boundary | R520 + R1000 × 8 | 2.8 |
| 27B KV `none_r`/`self_arc`/`mask_arc` | R520 + R1000 × 8 | 3.2 |
| 9B Needle archive arms | K4-C24 × 8 | 1.7 |
| Held-out levels | KV R2000 (9B, 4 arms); Needle K2-C192 (2B and 9B, `none` + `peer_27b`) × 8 | 5.5 |
| Canary rerun (seed 42, one cell) | | 0.3 |
| **Total** | | **≈34 (≈42 with 25% contingency; +5 for an optional sampled-decoding check)** |

## Conclusion

Seeds were a thinner source of evidence than the plan assumed, and in a different way than the risk note feared. They are not duplicates. They are word-level perturbations of one fixed task per level. That reframes what the 8/8 sign tests established so far: robust, layout-specific effects, which is a respectable claim, but not task-family generalization. It also means the cheapest way to strengthen the rule is spending compute on new levels and the two untested combinations rather than on more seeds. Seeds only buy power where a margin is being tested.

The cost story also shifts. Small-actor + large-manager wins mainly because the large model stops re-reading its own growing transcript as an actor, not because the small actor substitutes for it. Prefix caching removes much of that saving for append-only baselines. So the defensible headline is about 1/3 the compute at matched Needle quality for 2B, with a clean KV counterpart only once 27B `self_arc` exists. It should be made with explicit per-role accounting under all three cost measures.
