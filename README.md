# ctxmgr

Who should manage an LLM agent's context? We compare **no management**, **CLM self-management**,
a **same-size peer manager**, and a **fixed 27B peer manager** across Qwen3.5 actor sizes
(0.8B, 2B, 4B, 9B, 27B) on ContextBench-style tasks (KV Store, Needle Retention, CLM Appendix D).

## Layout

| Path | Track | Contents |
|---|---|---|
| `src/ctxmgr/tasks/` | 1 — Tasks & eval | Seeded instance generators, o200k token counts, pressure levels, scoring |
| `src/ctxmgr/harness/` | 2 — Harness & managers | Agent loop on vLLM, CLM edit tools/prompts, manager hook, per-step logs |
| `src/ctxmgr/serving/` | 3 — Infra | vLLM launch helpers, model registry |
| `src/ctxmgr/accounting/` | 3 — Infra | Compute accounting incl. manager calls (adapted from CLM `flops_metrics/`) |
| `src/ctxmgr/analysis/` | 3 — Analysis | Grid aggregation, plots, failure taxonomy |
| `configs/` | all | Experiment grid / model configs (YAML) |
| `slurm/` | 3 | Batch scripts for the Bouchet `education_gpu` partition |
| `scripts/` | all | Entry-point scripts |
| `report/` | all | 4-page NeurIPS write-up, reproducibility checklist |
| `docs/` | all | Interface specs (task format in → context/answers out), notes |

## Setup

```bash
uv sync                 # local dev (no GPU)
uv run pytest
```

On the cluster GPU nodes, vLLM is installed separately (Linux + CUDA only); see `slurm/`.

## Compute notes (Yale Bouchet, `education_gpu`)

- Limit: **1 GPU per user, 1 job at a time, 24 h wall time**. GPUs: A5000 (24 GB), RTX 5000 Ada (32 GB), A40 (48 GB, only 4).
- Run all computation through Slurm, never on login nodes.
- Store model weights and results in the course **project** directory (backed up). Scratch is purged after 30 days.
- Qwen3.5-27B (bf16, ~56 GB) fits on no single GPU → use the official `Qwen3.5-27B-FP8` (~31 GB) on an A40.
