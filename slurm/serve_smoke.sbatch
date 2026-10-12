#!/bin/bash
# Pilot: start a vLLM OpenAI-compatible server on one GPU, run a smoke test against it,
# then shut down so the GPU is not held idle.
# Usage (from repo root): sbatch slurm/serve_smoke.sbatch [model] [max_model_len]
#SBATCH --job-name=ctxmgr-smoke
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh

MODEL="${1:-Qwen/Qwen3.5-0.8B}"
MAX_LEN="${2:-32768}"
PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "host=$(hostname) job=${SLURM_JOB_ID} model=${MODEL} max_len=${MAX_LEN} port=${PORT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# Text-only use: disable image/video inputs so the vision encoder does not reserve memory.
uv run vllm serve "${MODEL}" \
    --host 127.0.0.1 --port "${PORT}" \
    --max-model-len "${MAX_LEN}" \
    --gpu-memory-utilization 0.85 \
    --limit-mm-per-prompt '{"image": 0, "video": 0}' \
    > "logs/vllm-${SLURM_JOB_ID}.log" 2>&1 &
SERVER_PID=$!
trap 'kill ${SERVER_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 120); do
    if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then break; fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
        echo "vLLM server exited during startup"; tail -50 "logs/vllm-${SLURM_JOB_ID}.log"; exit 1
    fi
    sleep 5
done
curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null || { echo "server not healthy after 10 min"; exit 1; }

uv run python scripts/smoke_test.py --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" \
    --out "logs/smoke-${SLURM_JOB_ID}.json"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
