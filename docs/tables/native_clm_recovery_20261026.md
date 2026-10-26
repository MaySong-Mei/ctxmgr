# Archived native CLM report preview — 2026-10-26

This is the uploaded **132/144** snapshot, not a live status report. Final cell 4B
seed 42 was already submitted as job 30875473 when this archive was captured.
A new cluster report must confirm its outcome. This file demonstrates the
automatic Discussion format; it has not been posted as a Discussion comment.

<!-- ctxmgr-native-clm-report:v1:6485b8c6527ad1cd9daa6d513187b4bb30245a4aabdb9a9a37de62d5a3fc78f6 -->
<!-- ctxmgr-report-status: incomplete -->

### Native CLM recovery: 132/144 validated grades

**Coverage: incomplete.** Original grades preserved: 88; recovered grades: 44; failed: 0; unrun: 12.

This is an intermediate snapshot. Partial seed pairs are not the full-grid comparison.

Qwen3.5 0.8B/2B/4B, Q8_0; six task levels; task seeds 42–45; conditions `none` and native `clm_self`. Official sampling: temperature 0.7, top_p 0.95, max_tokens 8192, thinking enabled.

Graded means a valid saved score, not necessarily a completed task stream. `budget_wall` is early termination; missing scores remain blank rather than zero. Every mean below uses only matching graded task-seed pairs.

| Model | Task | Paired / expected | none | clm_self | clm_self − none |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B-Q8_0 | kv_store:R130 | 4/4 | 0.0625 | 0.0000 | -0.0625 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R1000 | 4/4 | 0.0000 | 0.0000 | +0.0000 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R2000 | 4/4 | 0.0000 | 0.0000 | +0.0000 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C4 | 4/4 | 1.0000 | 0.6250 | -0.3750 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C24 | 4/4 | 0.1354 | 0.1042 | -0.0312 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K8-C48 | 4/4 | 0.1198 | 0.1667 | +0.0469 |
| Qwen3.5-2B-Q8_0 | kv_store:R130 | 4/4 | 0.4375 | 0.2917 | -0.1458 |
| Qwen3.5-2B-Q8_0 | kv_store:R1000 | 4/4 | 0.0000 | 0.0208 | +0.0208 |
| Qwen3.5-2B-Q8_0 | kv_store:R2000 | 4/4 | 0.0000 | 0.0000 | +0.0000 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C4 | 4/4 | 1.0000 | 0.9375 | -0.0625 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C24 | 4/4 | 0.3333 | 0.6667 | +0.3333 |
| Qwen3.5-2B-Q8_0 | needle_retention:K8-C48 | 4/4 | 0.1667 | 0.1927 | +0.0260 |
| Qwen3.5-4B-Q8_0 | kv_store:R130 | 3/4 | 1.0000 | 0.9722 | -0.0278 |
| Qwen3.5-4B-Q8_0 | kv_store:R1000 | 3/4 | 0.0000 | 1.0000 | +1.0000 |
| Qwen3.5-4B-Q8_0 | kv_store:R2000 | 3/4 | 0.0000 | 0.9861 | +0.9861 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C4 | 3/4 | 1.0000 | 1.0000 | +0.0000 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C24 | 3/4 | 0.3194 | 0.3611 | +0.0417 |
| Qwen3.5-4B-Q8_0 | needle_retention:K8-C48 | 3/4 | 0.1458 | 0.8819 | +0.7361 |

<details>
<summary>Per-seed scores (42, 43, 44, 45)</summary>

| Model | Task | Condition | Seed 42 | Seed 43 | Seed 44 | Seed 45 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B-Q8_0 | kv_store:R130 | none | 0.0000 | 0.0417 | 0.0000 | 0.2083 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R130 | clm_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R1000 | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R1000 | clm_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R2000 | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B-Q8_0 | kv_store:R2000 | clm_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C4 | none | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C4 | clm_self | 1.0000 | 0.2500 | 1.0000 | 0.2500 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C24 | none | 0.2917 | 0.0417 | 0.0833 | 0.1250 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K4-C24 | clm_self | 0.0417 | 0.0417 | 0.2917 | 0.0417 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K8-C48 | none | 0.1458 | 0.0833 | 0.1458 | 0.1042 |
| Qwen3.5-0.8B-Q8_0 | needle_retention:K8-C48 | clm_self | 0.0625 | 0.3750 | 0.0208 | 0.2083 |
| Qwen3.5-2B-Q8_0 | kv_store:R130 | none | 0.0000 | 0.0000 | 0.8333 | 0.9167 |
| Qwen3.5-2B-Q8_0 | kv_store:R130 | clm_self | 0.9167 | 0.1667 | 0.0417 | 0.0417 |
| Qwen3.5-2B-Q8_0 | kv_store:R1000 | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B-Q8_0 | kv_store:R1000 | clm_self | 0.0833 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B-Q8_0 | kv_store:R2000 | none | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B-Q8_0 | kv_store:R2000 | clm_self | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C4 | none | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C4 | clm_self | 1.0000 | 1.0000 | 0.7500 | 1.0000 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C24 | none | 0.3333 | 0.3333 | 0.3333 | 0.3333 |
| Qwen3.5-2B-Q8_0 | needle_retention:K4-C24 | clm_self | 0.5000 | 0.7500 | 0.6667 | 0.7500 |
| Qwen3.5-2B-Q8_0 | needle_retention:K8-C48 | none | 0.1667 | 0.1667 | 0.1667 | 0.1667 |
| Qwen3.5-2B-Q8_0 | needle_retention:K8-C48 | clm_self | 0.0833 | 0.2708 | 0.3333 | 0.0833 |
| Qwen3.5-4B-Q8_0 | kv_store:R130 | none | — | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-4B-Q8_0 | kv_store:R130 | clm_self | — | 1.0000 | 0.9583 | 0.9583 |
| Qwen3.5-4B-Q8_0 | kv_store:R1000 | none | — | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-4B-Q8_0 | kv_store:R1000 | clm_self | — | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-4B-Q8_0 | kv_store:R2000 | none | — | 0.0000 | 0.0000 | 0.0000 |
| Qwen3.5-4B-Q8_0 | kv_store:R2000 | clm_self | — | 1.0000 | 1.0000 | 0.9583 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C4 | none | — | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C4 | clm_self | — | 1.0000 | 1.0000 | 1.0000 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C24 | none | — | 0.2917 | 0.3333 | 0.3333 |
| Qwen3.5-4B-Q8_0 | needle_retention:K4-C24 | clm_self | — | 1.0000 | 0.0000 | 0.0833 |
| Qwen3.5-4B-Q8_0 | needle_retention:K8-C48 | none | — | 0.1250 | 0.1458 | 0.1667 |
| Qwen3.5-4B-Q8_0 | needle_retention:K8-C48 | clm_self | — | 0.6667 | 0.9792 | 1.0000 |

</details>

Termination among graded episodes:

| Model | Condition | Graded | stream_exhausted | budget_wall | submit | max_steps |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B-Q8_0 | none | 24 | 6 | 18 | 0 | 0 |
| Qwen3.5-0.8B-Q8_0 | clm_self | 24 | 2 | 22 | 0 | 0 |
| Qwen3.5-2B-Q8_0 | none | 24 | 8 | 16 | 0 | 0 |
| Qwen3.5-2B-Q8_0 | clm_self | 24 | 5 | 19 | 0 | 0 |
| Qwen3.5-4B-Q8_0 | none | 18 | 3 | 12 | 3 | 0 |
| Qwen3.5-4B-Q8_0 | clm_self | 18 | 11 | 1 | 6 | 0 |

<details>
<summary>Model and run provenance</summary>

| Model | Model revision | Model SHA-256 |
| --- | --- | --- |
| Qwen3.5-0.8B-Q8_0 | `6ab461498e2023f6e3c1baea90a8f0fe38ab64d0` | `0ad885ffd4bb022fc4f0d33a3308fa108ef8613159d3b3a67e23abca056b7a6c` |
| Qwen3.5-2B-Q8_0 | `f6d5376be1edb4d416d56da11e5397a961aca8ae` | `1b04acba824817554f4ce23639bc8495ff70453b8fcb047900c731521021f2c1` |
| Qwen3.5-4B-Q8_0 | `e87f176479d0855a907a41277aca2f8ee7a09523` | `10cc391b403021dd11c614679d2fd92f611c3681d29e29651b717316965d61e1` |

Original jobs: 30092658, 30094049, 30111683, 30120092, 30127128, 30129246, 30130421, 30131368, 30132949, 30133702, 30138817, 30143409.
Recovery jobs represented: 30174444, 30180281, 30182507, 30186786.

</details>

Original valid grades take precedence. The frozen recovery validator checks saved metadata and task identities; this formatter checks table consistency. Neither step regrades snapshots or proves equality with historical teammate model bytes.

Task seeds vary generated tasks, not inference seeds. Shared task structure can correlate seeds, so no independent-replicate confidence interval or significance test is inferred. Native CLM prompts, Bash/file tools and budget-final-turn rules differ from the original team `none` system. Native `clm_self` is also distinct from May's JSON `self` protocol; these Q8_0 official-sampling results must not be pooled with the vLLM/bf16 runs or their greedy/recommended-sampling comparisons.

Prepared with AI-assisted reporting code. Scientific interpretation and publication remain subject to human review.
