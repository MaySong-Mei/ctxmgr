# CLAUDE.md

Course project (Yale, cluster course id `cpsc4880`): who should manage an LLM agent's context?
This repo is public — never commit credentials, cluster usernames, model weights or results.

## Experiment

Core grid = actor size × 4 conditions, run on ContextBench-style tasks rebuilt from CLM Appendix D
(KV Store, Needle Retention; seeded, o200k token counts, pressure levels).

- Actors: Qwen3.5 dense instruct models 0.8B, 2B, 4B, 9B, 27B (not the `-Base` variants).
- Conditions: no management · CLM self-management · same-size peer manager · fixed 27B peer manager.
  Self and peer managers use the same edit tools and instructions; the manager triggers at a context threshold.
- Metrics: exact-value accuracy, loss of required info, compute including manager calls
  (adapt CLM `flops_metrics/`).

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

## Dev

- Python 3.12, uv. `uv sync`, `uv run pytest`, `uv run ruff check .`.
- Pilot: `sbatch slurm/setup_env.sbatch [models]` (CPU node) then `sbatch slurm/serve_smoke.sbatch [model ...]` (1 GPU; env MAX_LEN, GPU_UTIL).

## Open decisions

- Precision for 9B; how to run the 9B × fixed-27B-manager cell.
- Instances / pressure levels / seeds per cell → total GPU-hours under the 1-GPU limit.
- Whether to ask course staff for a higher GPU limit.
