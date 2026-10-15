#!/bin/bash
# Pilot: for each model, start a vLLM OpenAI-compatible server on one GPU, run the smoke test
# against it, then shut it down. Several models share one job to pay the queue wait once.
# Usage (from repo root):
#   sbatch slurm/serve_smoke.sbatch [model ...]          # default: Qwen/Qwen3.5-0.8B
#   MAX_LEN=32768 GPU_UTIL=0.90 sbatch slurm/serve_smoke.sbatch ...
#SBATCH --job-name=ctxmgr-smoke
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("Qwen/Qwen3.5-0.8B")
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "host=$(hostname) job=${SLURM_JOB_ID} max_len=${MAX_LEN} gpu_util=${GPU_UTIL} port=${PORT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

failed=()
for m in "${MODELS[@]}"; do
    tag="${m##*/}"
    log="logs/vllm-${SLURM_JOB_ID}-${tag}.log"
    echo "===== ${m} ====="
    start_server "${m}" "${PORT}" "${GPU_UTIL}" "${MAX_LEN}" "${log}"
    if wait_healthy "${PORT}" "${SERVER_PID}" "${log}" \
        && smoke "${PORT}" "${m}" "logs/smoke-${SLURM_JOB_ID}-${tag}.json"; then
        gpu_mem
    else
        failed+=("${m}")
    fi
    stop_server "${SERVER_PID}"
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL PASSED"
