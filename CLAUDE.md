# CLAUDE.md

Course project (Yale, cluster course id `cpsc4880`): who should manage an LLM agent's context?
This repo is public — never commit credentials, cluster usernames, model weights or results.
Agent operating instructions and current status: @AGENTS.md

## Experiment

Core grid = actor size × 4 conditions on the official ContextBench (released; pinned at
`fb533c18c78a82d3f1b2047a9c1700735cf829ae`, Track 1 adapter and contract in `docs/tasks-and-evaluation.md`
on `ruthvik/track1-tasks`, PR #1). Families: KV Store and Needle Retention at official named levels.

- Actors: Qwen3.5 dense instruct models 0.8B, 2B, 4B, 9B, 27B (not the `-Base` variants).
- Conditions: no management · CLM self-management · same-size peer manager · fixed 27B peer manager
  (a 27B actor gets none / self / same-size peer only). Self and peer managers use the same edit tools
  and instructions; the manager triggers at a context threshold.
- Budget: 32,768 o200k tokens with 2,048 reserved, same for every actor size. Exact o200k is required;
  served-model (Qwen) token counts are logged separately.
- Metrics: official ContextBench reward graded from per-turn context snapshots (KV: ANSWER blocks across
  snapshots; Needle: lines in the final snapshot), plus Track 1's strict diagnostics, plus compute
  including manager calls (adapt CLM `flops_metrics/`). The gold oracle is an implementation check,
  never a baseline.

## Tracks / ownership

| Track | Owner | Code |
|---|---|---|
| 1 Tasks & eval | Ruthvik | `src/ctxmgr/tasks/` |
| 2 Harness & managers (vLLM agent loop, CLM edit tools, manager hook, per-step logs) | Hancheng | `src/ctxmgr/harness/` |
| 3 Infra, runs, analysis, report skeleton | May | `src/ctxmgr/{serving,accounting,analysis}/`, `slurm/`, `report/` |

Tracks 1 and 2 must agree on the interface (task format in → context/answers out); document it in `docs/`.
Branches: `<name>/<track>` (e.g. `may/track3-infra`), merged to `main` via PR.
Timeline: weeks 1–2 tasks + harness + pilot on 0.8B; weeks 3–5 core grid; weeks 6–8 failure analysis + 4-page NeurIPS write-up.

## Compute (Yale Bouchet, `education_gpu`)

- Enforced limit: **1 GPU per user, 1 job at a time, 24 h wall** (docs say 2 GPUs; QoS enforces 1).
- GPUs: A5000 24 GB (32), RTX 5000 Ada 32 GB (12), A40 48 GB (4, one node).
- Rules of Behavior: all computation through Slurm, never on login nodes; end idle interactive sessions;
  no floods of <5-min jobs; don't split work across accounts to dodge fair-use limits; course accounts are coursework-only.
- Storage: weights/results in project dir `/nfs/roberts/project/<course>/<user>` (backed up);
  scratch is purged after 30 days (caches only). Paths are set in `slurm/env.sh`.
- Connection setup: `docs/cluster_setup.md`. Login needs Duo, which only a human can approve.

## Models & precision

- Weights (bf16): 0.8B 1.7 GB, 2B 4.5 GB, 4B 9.3 GB, 9B 19.3 GB, 27B 55.6 GB. Official quants exist only for 27B
  (`Qwen3.5-27B-FP8` 30.9 GB, `-GPTQ-Int4` 30.2 GB).
- 27B fits no single GPU in bf16 → use `Qwen3.5-27B-FP8` on an A40 (Ampere: weight-only FP8, slower).
- 9B actor + 27B-FP8 manager does not fit one A40 in bf16 → candidate: 9B in FP8 for all four of its conditions.
- Keep precision identical across the four conditions of a given actor size. Plan a bf16-vs-FP8 control on one small size.
- Qwen3.5 is multimodal (`Qwen3_5ForConditionalGeneration`) with hybrid linear/full attention (1 in 4 layers full),
  so KV cache is small (~64 KB/token for 27B). Serve text-only (`--limit-mm-per-prompt` image/video = 0).
  Thinking is on by default; control it via `chat_template_kwargs.enable_thinking`.
- vLLM pinned to 0.30.0 (`uv sync --group serve`, Linux only); it pulls torch 2.13 built for CUDA 13.
  GPU nodes run driver 580 (CUDA 13 OK) but have no `nvcc`, so FlashInfer JIT kernels fail →
  `VLLM_USE_FLASHINFER_SAMPLER=0` (set in `slurm/env.sh`). Attention backend is FLASH_ATTN (no JIT).
- Pilot (2026-10-12, 0.8B on A5000): smoke test passed; 27K-token KV recall correct; ~313 tok/s decode;
  KV cache 1.36M tokens; job ~7 min, of which vLLM startup ~3 min (fixed per-job overhead).
- Pilot (2026-10-14, A40, max_len 32K, gpu_util 0.90; single-request decode is memory-bandwidth bound):
  2B: 3.6 GiB weights, KV 2.72M tok, 128 tok/s · 4B: 8.0 GiB, KV 0.88M tok, 59 tok/s ·
  9B: 16.8 GiB, KV 0.60M tok, 33 tok/s. All recalled the 27K-token KV probe. 9B on A5000 still untested.
- Pilot (2026-10-15, A40, `slurm/a40_27b.sbatch` first run, max_len 32K): every phase FAILED, all on
  configuration, none on capacity. 27B-FP8 download 5 min (setup job OK). 27B-FP8 loads: 28.0 GiB
  weights, 25 s (8 s cached); torch.compile 48 s first time (later cached). Alone at util 0.90: 8.32 GiB
  left for KV; co-location share 0.75: 3.79 GiB = 53,930 tokens, i.e. only 1.65 concurrent 32K requests
  for the manager. Both then refused to start: default `max_num_seqs` 256 exceeds the Mamba state blocks
  (173 at 0.90, 79 at 0.75); fix `--max-num-seqs 64`. 9B with online FP8 crashed in torch.compile
  (Inductor "auto_functionalized was not removed", vLLM 0.30 / torch 2.13); fix `--enforce-eager`
  (eager decode is a lower bound) or an offline FP8 checkpoint. Actors never started, so co-location
  fit is still unknown. Both fixes are in the script.
- Pilot (2026-10-16, A40 re-run with `--max-num-seqs 64`, max_len 32K): job exit 1 (3 of 5 phases fail).
  27B-FP8 alone (util 0.90) PASS: Marlin weight-only FP8 kernel, KV 9.03 GiB = 128K tokens (3.9× 32K),
  27K-token probe correct (prefill 22 s), decode 20.3 tok/s. Co-located 27B-FP8 (0.75) + 2B (0.21) PASS,
  both smoke tests concurrently: 27B KV 63K tokens (1.9× 32K), decode still 20.3 tok/s, prefill 28 s;
  2B KV 342K tokens, 58 tok/s (vs 128 alone); 43.1 of 46.1 GB used. 27B + 4B FAIL on capacity: 4B weights
  8.0 GiB leave −0.19 GiB KV in the 0.21 share. 9B online FP8 FAIL (alone and co-located): vLLM picks the
  CUTLASS w8a8 kernel, which errors on Ampere (`cutlass_scaled_mm_sm80_epilogue`); the 27B checkpoint
  works because it gets Marlin. 9B-FP8 weights are 10.4 GiB, above the 0.21 share (~9.4 GiB) anyway.
  So on one A40 only 2B (and presumably 0.8B) can sit beside the 27B manager at 32K.

## Dev

- Python 3.12, uv. `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Pilot: `sbatch slurm/setup_env.sbatch [models]` (CPU node) then `sbatch slurm/serve_smoke.sbatch [model ...]` (1 GPU; env MAX_LEN, GPU_UTIL).

## Open decisions

- Precision for 9B; how to run the 9B × fixed-27B-manager cell.
- Where the fixed 27B manager runs: co-located at util 0.75 it has ~63K KV tokens (~1.9 concurrent 32K
  requests), so manager calls would serialize, and only 2B/0.8B actors fit beside it (4B and 9B do not).
  Alternatives: 27B alone on the A40 (128K KV tokens) serving manager calls in a separate phase, or
  batching all peer_27b cells per actor.
- Instances / pressure levels / seeds per cell → total GPU-hours under the 1-GPU limit.
- Whether to ask course staff for a higher GPU limit.
- Prior-art check: does the CLM paper's multi-agent section already run an equivalent peer-manager
  comparison? Settle before the core grid; it decides the novelty claim.
- Deferred per the instructor's core grid (Track 1 AGENTS.md): scheduled summarization, next-size-up
  peers, MoE. Suggest dropping the proposal's oracle-retention baseline: ContextBench's gold oracle reads the answer
  key and is only an implementation check.
- Manager scope: the peer manager may only edit context (same edit tools as self-management), never
  answer the task or add reasoning, or task skill leaks into the peer conditions.
- Served context length: the 32K budget is in o200k tokens, but vLLM `max_len` counts Qwen tokens, which
  can be more for the same text. Measure Qwen/o200k ratio on real episodes; `max_len` = 32,768 (as in the
  pilots) may need headroom, which changes KV memory and co-location fit.
- Context-management logging from the first run (draft in `docs/log_schema.md`): per-call usage by role,
  edits, triggers, termination, on top of ContextBench snapshots.
