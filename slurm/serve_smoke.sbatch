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

MODELS=("$@")
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("Qwen/Qwen3.5-0.8B")
MAX_LEN="${MAX_LEN:-32768}"
GPU_UTIL="${GPU_UTIL:-0.90}"
PORT=$((20000 + SLURM_JOB_ID % 10000))

echo "host=$(hostname) job=${SLURM_JOB_ID} max_len=${MAX_LEN} gpu_util=${GPU_UTIL} port=${PORT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

run_one() {
    local model="$1" tag="${1##*/}" server_pid
    local log="logs/vllm-${SLURM_JOB_ID}-${tag}.log"
    echo "===== ${model} ====="

    # Text-only use: disable image/video inputs so the vision encoder does not reserve memory.
    uv run vllm serve "${model}" \
        --host 127.0.0.1 --port "${PORT}" \
        --max-model-len "${MAX_LEN}" \
        --gpu-memory-utilization "${GPU_UTIL}" \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' \
        > "${log}" 2>&1 &
    server_pid=$!

    local healthy=0
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null; then healthy=1; break; fi
        kill -0 "${server_pid}" 2>/dev/null || break
        sleep 5
    done

    local rc=1
    if [[ ${healthy} -eq 1 ]]; then
        grep -E "Model loading took|GPU KV cache size" "${log}"
        uv run python scripts/smoke_test.py --base-url "http://127.0.0.1:${PORT}/v1" --model "${model}" \
            --out "logs/smoke-${SLURM_JOB_ID}-${tag}.json"
        rc=$?
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    else
        echo "vLLM server failed to start for ${model}"; grep -E "Error|error" "${log}" | tail -5
    fi

    kill "${server_pid}" 2>/dev/null
    wait "${server_pid}" 2>/dev/null
    return ${rc}
}

failed=()
for m in "${MODELS[@]}"; do
    run_one "${m}" || failed+=("${m}")
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL PASSED"
