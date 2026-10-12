#!/bin/bash
# One-time (or after dependency changes) environment setup on a CPU node:
# installs uv, syncs the venv with vLLM, and downloads model weights.
# Usage (from repo root): sbatch slurm/setup_env.sbatch [model ...]
#SBATCH --job-name=ctxmgr-setup
#SBATCH --partition=education
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version

uv sync --group serve
uv run python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda)"

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("Qwen/Qwen3.5-0.8B")
for m in "${MODELS[@]}"; do
    uv run hf download "$m"
done
du -sh "${HF_HOME}"/hub/models--* 2>/dev/null || true
