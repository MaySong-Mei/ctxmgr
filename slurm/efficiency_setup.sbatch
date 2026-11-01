#!/bin/bash
#SBATCH --job-name=ctx-eff-setup
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
hostname
uv sync --locked --group serve
uv run pytest -q
uv run ruff check src/ctxmgr/accounting/efficiency.py scripts/efficiency_grid.py scripts/multidoc_pilot.py tests/test_efficiency.py src/ctxmgr/harness/episode.py
uv run python scripts/multidoc_pilot.py --data "${PROJECT_DIR}/datasets/hotpot_dev_distractor_v1.json" --prepare-only
# Record verified, existing weights without downloading or changing a teammate's cache.
for size in 2B 4B 9B 27B; do
    path=$(bash slurm/efficiency_models.sh "$size")
    test -r "$path/config.json"
    test -r "$path/tokenizer.json"
done
# Verify flags against installed vLLM, not assumptions about another version. `vllm serve --help`
# cannot run here: it builds the engine config and fails to infer a device on a CPU node
# (job 31564027). Read the installed server CLI definition instead; the GPU smoke exercises it.
site=$(uv run python -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
cli_args="$site/vllm/entrypoints/launchers/cli_args.py"
grep -Eq '^\s+enable_prompt_tokens_details: bool' "$cli_args"
uv run python -c "import importlib.metadata as m; print('vllm', m.version('vllm'))"
