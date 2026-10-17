#!/bin/bash
# First end-to-end check: served Qwen model -> official ContextBench task -> Track 1 scorer,
# no-management condition only (project-status.md "First model smoke test").
# Generates the instances inside the job (Python never runs on login nodes), serves one model,
# runs each (family, level) at seed 42, and prints the scores.
# Usage (from repo root):
#   sbatch slurm/baseline_smoke.sbatch                      # default: Qwen/Qwen3.5-2B
#   MODEL=Qwen/Qwen3.5-0.8B LEVELS="kv_store:R130 needle_retention:K4-C4" sbatch slurm/baseline_smoke.sbatch
#SBATCH --job-name=ctxmgr-baseline
#SBATCH --partition=education_gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

MODEL="${MODEL:-Qwen/Qwen3.5-2B}"
LEVELS="${LEVELS:-kv_store:R130 needle_retention:K4-C4}"
SEED="${SEED:-42}"
# The benchmark budget is 32,768 o200k tokens; Qwen's tokenizer can need more for the same text, so
# serve with headroom and let ContextBench's o200k wall decide when an episode ends.
MAX_LEN="${MAX_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
J="${SLURM_JOB_ID}"
TAG="${MODEL##*/}"
RUNS="${PROJECT_DIR}/runs/baseline-${J}"

echo "host=$(hostname) job=${J} model=${MODEL} max_len=${MAX_LEN} levels=${LEVELS} seed=${SEED}"
echo "commit=$(git rev-parse --short HEAD) runs=${RUNS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --group serve --quiet || exit 1

log="logs/vllm-${J}-${TAG}.log"
start_server "${MODEL}" "${PORT}" "${GPU_UTIL}" "${MAX_LEN}" "${log}" --max-num-seqs "${MAX_NUM_SEQS}"
if ! wait_healthy "${PORT}" "${SERVER_PID}" "${log}"; then
    stop_server "${SERVER_PID}"; exit 1
fi

failed=()
for item in ${LEVELS}; do
    family="${item%%:*}" level="${item##*:}"
    name="${family}-${level}-s${SEED}"
    echo "===== ${name} ====="
    instance="${RUNS}/instances/${name}"
    if ! uv run python -m ctxmgr.tasks generate "${family}" --level "${level}" --seed "${SEED}" \
        --out "${instance}" > /dev/null; then
        failed+=("${name}:generate"); continue
    fi
    if ! uv run python -m ctxmgr.harness run "${instance}" \
        --out "${RUNS}/${TAG}/none/${name}" --condition none \
        --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" \
        --served-max-len "${MAX_LEN}" --precision "${PRECISION:-bf16}"; then
        failed+=("${name}:run")
    fi
done
gpu_mem
stop_server "${SERVER_PID}"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL EPISODES COMPLETED (scores above; low reward is a result, not a job failure)"
