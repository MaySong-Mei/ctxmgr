# Results tables

`episodes_vllm.csv`: one row per scored episode on the vLLM stack (A40: bf16 actors, Qwen3.5-27B-FP8
manager; H200: bf16 actors and bf16 Qwen3.5-27B manager; see `job`), built from the run directories (`episode.json`, `score.json`,
`calls.jsonl`, `edits.jsonl`) with `python scripts/episodes_table.py OUT.csv PREVIOUS.csv`. Raw runs
live in the course project directory, not in git.

- `edit_version`: edit protocol for managed conditions. `edit-v0-draft` = first draft (c717c61),
  `edit-v1-hardened` = PR #4 before instructions were versioned; later versions are logged by the
  harness (`EDIT_PROMPT_VERSION`). Empty for `none`.
- `temperature` / `top_p` / `top_k` / `presence_penalty`: decoding of actor and manager. Sampled
  (0.7 / 0.8 / 20 / 1.5, request seed = instance seed): jobs 30263988/9, 30265858/9, 30271270/1,
  30335778, 30691916, 30791880. All other jobs are greedy (temperature 0). Jobs 30260909/10 were meant as the
  ablation but ran greedy (sbatch bug) and are kept as a greedy replicate.
- `max_turns`: safety cap, 2 x n_ops + 10 where logged (empty for jobs before the cap existed);
  job 30791880 uses 4 x n_ops + 10.
- `official_reward`: ContextBench official score. `required_info_loss` (Needle) and
  `strict_exact_accuracy` (KV) are Track 1 diagnostics.
- `mgmt_events` / `events_applied` / `manager_capped`: management calls, calls with at least one
  applied edit, manager replies cut off by the output cap.
- Token columns sum prompt + completion tokens of the served model per role.
- Not comparable with llama.cpp/Q8_0 runs (different serving stack and precision).

Other tables:
- `h200_bf16_pooled.md`: greedy core grid, paired differences vs `none` (`scripts/pooled_grid.py`).
- `decoding_ablation.md`: greedy vs recommended sampling for the core grid and the 27B role-separation
  comparison (`scripts/decoding_ablation.py`).
- `cost_needle.md`: compute per Needle episode by role, prefix-reuse FLOPs with measured cache hits
  (`src/ctxmgr/accounting/flops.py`), no-cache bound and GPU-seconds (`scripts/cost_table.py`; runs
  from rounds 25-28 only, the first to log `cached_prompt_tokens`).
- Archive conditions (round 24, jobs 30937295 and 30939991/4/7): `none_r`, `self_arc`, `peer_same_arc`,
  `peer_27b_arc` (`edit_version` `edit-v8-arc`) and `mask_arc` (`rule-mask-v1`). Their per-episode
  RECALL logs are in each run's `recalls.jsonl` (not in this table).
- Pre-registered runs (round 25): canaries 30980396/7 (seed 46) and jobs 30981378–30981390
  (seeds 50–65), harness `33b3997`; analysed only by `scripts/prereg_analysis.py`
  (`docs/preregistration-results.md`).
