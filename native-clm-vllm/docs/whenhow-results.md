# When versus how: who should manage the context?

Results of the when-versus-how study, run on the native CLM harness in this package
(`facebookresearch/context-language-models@4831982`, official `ClmStream`/`ClmAgent`,
thinking on, temperature 0.7, top_p 0.95). Qwen3.5 actors and managers in BF16 on
vLLM 0.30, one H200 per job. All numbers are aggregates; BrowseComp-Plus questions and
answers stay in owner-only cluster storage and are never written to this repository.

## Questions

1. **Scaling.** Does native CLM self-management improve from 2B to 27B?
2. **When vs how.** When a small model fails, is it at deciding *when* to edit its
   context or at *how* it edits?
3. **Separation.** Does giving the timing or the editing to a separate manager help?
4. **Role vs capability.** Is any gain from separating the roles (same-size manager) or
   from a more capable manager (fixed 27B)?

## Conditions

CLM bundles two decisions, when to edit and how. Each condition fixes who makes each one.

| condition | who decides when | who edits | implementation |
|---|---|---|---|
| `none` | — | — | team chat baseline (seeds 42–45); `none_same_harness`, the CLM loop with edits disabled (seeds 46–49) |
| `clm_self` | actor, with the task recipe (ContextBench) or budget nudges (BrowseComp-Plus) | actor | official `ClmStream` / `ClmAgent`, paper settings |
| `clm_self_noskill` / `clm_self_nonudge` | actor, from the official system prompt only | actor | `ClmStream(skill="none")`; on BrowseComp-Plus, nudges off |
| `clm_fixed` | harness: at 60% of the enforced budget, a two-turn edit window | actor | `clm_whenhow.FixedTriggerMixin` |
| `mgr_request` | actor runs `echo REQUEST_COMPACTION` | manager | `clm_whenhow.ManagedMixin` |
| `mgr_fixed` | harness, at 60% of the enforced budget | manager | `clm_whenhow.ManagedMixin` |
| `mgr_watch` | manager, after every 4,096 tokens of growth (may decline) | manager | `clm_whenhow.ManagedMixin` |

A manager is a separate session with the same bash tool on the same mirrored context
file, the official editing guidance, and the same "edit changed nothing" feedback the
actor gets. The actor's own edits are not applied in manager conditions. The manager
is a same-size session (`@<actor size>`, the actor's own server) or a fixed 27B
(`@27B`, a second server on the same GPU). Each condition's timing rule is stated in
the actor's system prompt.

**Why a no-recipe condition.** The paper's ContextBench skills prescribe both timing
("delete each chunk's filler block on the SAME turn it arrives") and the edit itself
(an exact `python3` heredoc). `clm_self` on ContextBench therefore measures recipe
execution; `clm_self_noskill` leaves both decisions to the model.

## Benchmarks

- **ContextBench** (official generator and grader, pinned): Needle Retention K4-C24 and
  K8-C48, KV Store R1000 and R2000, 32,768-token budget, 2,048 reserved. Seeds 42–49 on
  the same instances for every condition, so comparisons are paired: 16 pairs per
  comparison (2 levels × 8 seeds). Seeds share each level's layout, so they are weak
  replicates.
- **BrowseComp-Plus**: 100 questions, the first 100 query ids in SHA-256 order (chosen
  before any run, independent of performance), over the fixed 100,195-document corpus.
  Search is a loopback BM25 server (`bm25s`; the official Lucene index needs Java):
  top-10 snippets of 512 o200k tokens, documents capped at 8,192 tokens, exposed as the
  shell commands `bcp_search` and `bcp_get_document`. Agent settings from the released
  `clm/clm_harness/configs/bcp.yaml` (28,672-token budget, 2,048 reserve, 6 rollback
  retries, shrink edit gate, nudges at 25/50/75% plus a persistent 90% nudge), except
  100 task steps, the paper's cap. Judged by Qwen3.5-27B with the official
  BrowseComp-Plus grading template; an episode with no answer file is incorrect.

## Results

### BrowseComp-Plus (100 questions)

Limit hits = episodes that received the final-turn notice (ran out of context). p =
exact McNemar against `clm_self` on the same questions. PF = nominal compute,
2 × parameters × (uncached prompt + completion tokens), summed over roles.

| actor | condition | accuracy [95% CI] | limit hits | vs `clm_self` (+/−, p) | PF / episode | PF / correct |
|---|---|---|---|---|---|---|
| 9B | `clm_self` | 21% [14–30] | 78 | — | 4.5 | 21.6 |
| 9B | `clm_self_nonudge` | 10% [6–17] | 91 | +1/−12, 0.003 | 3.3 | 32.8 |
| 9B | `clm_fixed` | 24% [17–33] | 69 | +11/−8, 0.65 | 4.5 | 18.7 |
| 9B | `mgr_fixed@9B` | 46% [37–56] | 52 | +27/−2, <0.001 | 14.1 | 30.7 |
| 9B | `mgr_watch@9B` | 42% [33–52] | 64 | +24/−3, <0.001 | 13.4 | 31.9 |
| 9B | `mgr_request@9B` | 13% [8–21] | 91 | +3/−11, 0.057 | 3.4 | 25.8 |
| 9B | `mgr_fixed@27B` | 61% [51–70] | 0 | +41/−1, <0.001 | 38.2 | 62.7 |
| 9B | `mgr_watch@27B` | 62% [52–71] | 1 | +42/−1, <0.001 | 48.3 | 77.9 |
| 9B | `mgr_request@27B` | 14% [9–22] | 91 | +5/−12, 0.14 | 3.5 | 24.7 |
| 27B | `clm_self` | 50% [40–60] | 58 | — | 14.5 | 28.9 |
| 27B | `clm_self_nonudge` | 33% [25–43] | 82 | +5/−22, 0.002 | 9.6 | 29.2 |
| 27B | `clm_fixed` | 36% [27–46] | 76 | +7/−21, 0.013 | 12.2 | 34.0 |
| 27B | `mgr_fixed@27B` | 66% [56–75] | 0 | +21/−5, 0.002 | 37.1 | 56.2 |
| 27B | `mgr_watch@27B` | 65% [55–74] | 0 | +23/−8, 0.011 | 39.1 | 60.1 |
| 27B | `mgr_request@27B` | 32% [24–42] | 83 | +5/−23, 0.001 | 9.3 | 29.2 |

Further paired tests: with a 9B actor, a 27B manager beats a 9B manager (watching
+25/−5, p<0.001; fixed +23/−8, p=0.011). Harness timing beats no-nudge self for the 9B
(+16/−2, p=0.001), not for the 27B (+11/−8, n.s.).

### ContextBench (seeds 42–49)

Mean official reward, Needle (K4-C24 and K8-C48) / KV (R1000 and R2000), 16 episodes
per cell.

| actor | `none` | `clm_self` | `clm_self_noskill` | `clm_fixed` | same-size `mgr_watch` | `mgr_watch@27B` |
|---|---|---|---|---|---|---|
| 2B | 0.13 / 0 | 0.51 / 0 | 0.12 / 0 | 0.15 / 0 | 0.08 / 0 | 0.73 / 0.21 |
| 4B | 0.22 / 0.01 | 0.62 / 0.99 | 0.22 / 0 | 0.46 / 0.01 | 0.30 / 0.02 | 0.53 / 0.42 |
| 9B | 0.22 / 0 | 0.99 / 0.92 | 0.45 / 0 | 0.46 / 0.12 | 0.63 / 0.21 | 0.62 / 0.31 |
| 27B | 0.22 / 0 | 1.00 / 1.00 | 0.73 / 0 | 0.70 / 0 | 0.34 / 0.55 | — |

Paired differences against `clm_self_noskill` (mean [bootstrap 95% CI], pairs
better/worse, exact sign test):

| actor | comparison | Needle | KV |
|---|---|---|---|
| 2B | `mgr_watch@27B` | +0.61 [+0.49, +0.73], 16/0, p<0.001 | +0.21 [+0.10, +0.32], 12/0, p<0.001 |
| 2B | `mgr_watch@2B` | −0.03, 4/7, n.s. | 0 |
| 4B | `clm_fixed` | +0.24 [+0.11, +0.38], 10/1, p=0.012 | +0.01, n.s. |
| 4B | `mgr_watch@27B` | +0.31 [+0.18, +0.44], 14/2, p=0.004 | +0.42 [+0.28, +0.57], 16/0, p<0.001 |
| 9B | `mgr_watch@27B` | +0.16, 9/6, n.s. | +0.31 [+0.19, +0.44], 14/0, p<0.001 |
| 27B | `mgr_fixed@27B` | −0.39 [−0.59, −0.17], 2/11, p=0.022 | +0.59 [+0.46, +0.72], 16/0, p<0.001 |
| 27B | `mgr_request@27B` | −0.50 [−0.67, −0.32], 2/14, p=0.004 | 0 |

In request mode, actors asked for the manager 6 times in 103 episodes (seeds 42–45).

## Findings

1. **Scaling holds with the recipe, and the recipe carries KV.** With the paper's skill,
   KV recall is 0 at 2B, 0.99 at 4B and ≈1.0 at 9B and 27B. Without it, KV is 0 at every
   size, 27B included: no model offloads to files unaided. A separate 27B manager does
   invent offloading (it writes the SET batches to a file and edits the actor's context
   down).
2. **Small models fail at different halves of the job.** The 4B rarely edits on its own
   (3 edits in 16 episodes, seeds 42–45), and harness timing lifts its Needle score
   0.22 → 0.46 (p=0.012). The 2B edits often but loses what it needs; timing does not help
   it, and only a 27B manager does (0.12 → 0.73, 16/16 pairs).
3. **Agents almost never ask for help.** Request-mode management performs like no
   management on both benchmarks.
4. **A separate manager removes context exhaustion on the realistic task.** On
   BrowseComp-Plus, 58–78% of `clm_self` episodes end out of context; with a 27B
   manager, 0–1%. A 9B actor with a 27B manager (61–62%) beats the 27B managing itself
   (50%).
5. **Separation and capability both matter.** A same-size manager helps on
   BrowseComp-Plus (9B: 21% → 42–46%; 27B: 50% → 65–66%), and a 27B manager helps more.
   On ContextBench only the 27B manager helps the 2B and 4B, and a separate 27B manager
   hurts the 27B on Needle while helping it on KV.
6. **Timing cues matter.** Removing the paper's budget nudges costs 11 points at 9B and
   17 at 27B on BrowseComp-Plus.
7. **It costs compute.** Managed runs use about 2.5–3.5× the nominal compute per
   episode. The 9B with a same-size manager (46% at 14 PF) is about as efficient as the
   27B alone (50% at 14.5 PF).

## Limits

One model family (Qwen3.5). BM25 instead of the stronger embedding retriever: the 27B's
50% is near the paper's 59.4% (Qwen3.6-27B), and the 9B's 21% is below its 39.9%. 100
questions. Nominal FLOPs ignore attention and Qwen3.5's linear-attention layers.
ContextBench seeds share each level's layout. Failure policy: episodes that hit the 2 GB
container memory limit (8) or the 6-hour agent limit (1) count as failures;
infrastructure errors (container start timeout, malformed log, HTTP error, jobs
cancelled under a QOS limit or killed by host OOM) were rerun.

## Reproduction

- Native suite (`none`, `clm_self`, seeds 42–45): `scripts/clm_vllm.sh submit` with a
  config copy setting `CLM_GPU_CPUS=6` (`gpu_h200` allows at most 6 CPUs per GPU).
- When/how ContextBench grid: `scripts/whenhow.sh submit <suite>` (seeds 42–45) and
  `scripts/whenhow.sh extra <suite>` (seeds 46–49, with instance generation and matched
  controls).
- BrowseComp-Plus: `scripts/bcp.sh init <suite>`, then `pilot`, `grid` and `judge`.
  Questions 50–100 were submitted with `slurm/bcp_job.sbatch` directly (slices
  `50:75`, `75:100`).
- Analysis: `analysis/` (`extract.sh` on the login node, then `paired.py`,
  `bcp_an.py` and `bcp_cost.py`).

Run directories (course storage): native suite `runs/clm-vllm-20261112T191604Z-ADzo9i`;
when/how `runs/whenhow-20261112T194815Z` and `runs/whenhow-20261114T002536Z`;
BrowseComp-Plus `runs/bcp-20261112T193824Z` (mode 700).
