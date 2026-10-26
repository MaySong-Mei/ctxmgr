# Native CLM recovery snapshot — 2026-10-26

This is an archived intake snapshot, not a live cluster status check. The
[generated report preview](tables/native_clm_recovery_20261026.md) contains its
paired comparison, per-seed scores and termination counts.

The uploaded review archive passed SHA-256 verification. The audit checked all
101 original frozen source files, 101 recovery source files with their two
declared host-side patches, and eight frozen recovery tools against the supplied
manifests. The original suite and saved grades were not modified.

The latest saved combined report contains **132/144 validated grades**: 88
original grades and 44 recovered grades. Coverage by model is 48/48 for 0.8B,
48/48 for 2B and 36/48 for 4B. The final missing cell, 4B task seed 42, is already
recorded as job **30875473**. Its archived log contains the start of its fifth
episode and no completion summary. A current Slurm check and fresh combined
report are required before calling this experiment complete.

Completed recovery cells in the snapshot:

| Size | Task seed | Job | Selected grades |
| --- | --- | --- | --- |
| 0.8B | 42 | 30174444 | 10/10 |
| 2B | 43 | 30180281 | 10/10 |
| 2B | 44 | 30182507 | 12/12 |
| 2B | 45 | 30186786 | 12/12 |

The partial 4B report suggests gains on KV R1000/R2000 and Needle K8-C48, but
the remaining seed must be included before reporting the full grid. The small
models have mixed or negative differences; automation does not select only
favorable tasks or replace valid zero scores.

Experimental configuration: Qwen3.5 Q8_0 on llama.cpp/A40; native CLM `official`
sampling with thinking enabled, temperature 0.7, top_p 0.95 and max_tokens 8192;
task seeds 42–45; six levels; `none` versus native `clm_self`. ContextBench is
pinned at `fb533c18c78a82d3f1b2047a9c1700735cf829ae`, and native CLM at
`483198229ab7b90c0eed4651306ab3b3505aaf50`. Shared task inputs and sampling do not
make the two native protocols identical. This is not an edit-permission-only
ablation and is separate from the team's vLLM bf16/edit-v8 results.

The archive does not include all raw episode directories. This audit validates
the uploaded code and report consistency; a cluster-side report must validate
actual saved episode provenance. Task seeds are task variants, not independent
model-training or inference-seed replicates. No confidence interval or
statistical-significance claim is inferred from this snapshot.
