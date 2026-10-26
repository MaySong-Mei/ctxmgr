# ctxmgr

Who should manage an LLM agent's context? We compare **no management**, **self-management**, a
**same-size peer manager** and a **fixed 27B peer manager** for Qwen3.5 actors (0.8B–27B) on the
official [ContextBench](https://github.com/RulinShao/ContextBench) KV Store and Needle Retention
tasks (32,768-token o200k budget).

## Current implementation

| Path | Contents |
|---|---|
| `slurm/` | vLLM 0.30 environment setup and serving jobs for Yale Bouchet (A40 / H200 GPUs) |
| `scripts/smoke_test.py` | Server check: short chat, long-context recall, throughput |
| `scripts/*.py` (other) | Analysis: lenient KV score, paired comparisons vs no management, figures |
| `docs/results.md`, `docs/tables/`, `docs/figures/` | Results so far: summary, 1,375 episodes (vLLM, bf16), pooled and decoding-ablation tables, figures |
| `docs/` | Cluster setup, log schema, manager literature notes |

The task adapter (Track 1) and the agent harness with the four management conditions (Track 2)
are on the `ruthvik/track1-tasks` and `may/track2-baseline` branches and will be merged after review.

## Usage

```bash
uv sync --locked
uv run python scripts/pooled_grid.py   # needs run directories under runs/
```

On the cluster: `sbatch slurm/setup_env.sbatch`, then `sbatch slurm/serve_smoke.sbatch`.
