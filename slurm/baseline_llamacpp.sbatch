#!/bin/bash
# No-management baseline on the llama.cpp stack (GGUF Q8_0 actors; see issue #2 for the switch from
# vLLM/bf16). Same harness and scorer as slurm/baseline_smoke.sbatch; only the server differs.
# Generates the instances inside the job (Python never runs on login nodes), then for each model:
# serve it, run every (family, level, seed), stop the server. Pinned to the A40 because the managed
# and peer_27b cells need it, and temperature 0 is not reproducible across GPU types.
# Run slurm/llamacpp_setup.sbatch (on ruthvik/llamacpp-a40) once first for the llama.cpp build.
# Usage (from repo root):
#   SIZES=2B LEVELS="kv_store:R130" SEEDS=42 sbatch --time=01:00:00 slurm/baseline_llamacpp.sbatch
#   sbatch slurm/baseline_llamacpp.sbatch          # core grid: 4 sizes x 6 levels x 4 seeds
#SBATCH --job-name=ctxmgr-baseline-gguf
#SBATCH --partition=education_gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh
source slurm/llamacpp.sh

SIZES="${SIZES:-0.8B 2B 4B 9B}"
QUANT="${QUANT:-Q8_0}"
LEVELS="${LEVELS:-kv_store:R130 kv_store:R1000 kv_store:R2000 needle_retention:K4-C4 needle_retention:K4-C24 needle_retention:K8-C48}"
SEEDS="${SEEDS:-42 43 44 45}"
# The benchmark budget is 32,768 o200k tokens, and ContextBench's o200k wall should end episodes, not
# the server. Qwen needs 1.24x as many tokens on KV but up to 1.78x on Needle when a small model
# echoes digit-heavy text (2B, job 29904662: rejected at 47.6K Qwen = 27.4K o200k with ctx 49,152).
# 1.78 x 32,768 + 2,048 reply = 60.4K, so serve 80K for margin.
CTX="${CTX:-81920}"
PORT=$((20000 + SLURM_JOB_ID % 10000))
J="${SLURM_JOB_ID}"
RUNS="${PROJECT_DIR}/runs/baseline-gguf-${J}"

echo "host=$(hostname) job=${J} sizes=${SIZES} quant=${QUANT} ctx=${CTX} levels=${LEVELS} seeds=${SEEDS}"
echo "commit=$(git rev-parse --short HEAD) llama.cpp=${LLAMA_BUILD} runs=${RUNS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --quiet || exit 1

failed=()
# Generate each instance once; every model sees identical streams.
for item in ${LEVELS}; do
    family="${item%%:*}" level="${item##*:}"
    for seed in ${SEEDS}; do
        name="${family}-${level}-s${seed}"
        if ! uv run python -m ctxmgr.tasks generate "${family}" --level "${level}" --seed "${seed}" \
            --out "${RUNS}/instances/${name}" > /dev/null; then
            failed+=("${name}:generate")
        fi
    done
done

export HF_HUB_DISABLE_XET=1
for size in ${SIZES}; do
    TAG="Qwen3.5-${size}-${QUANT}"
    gguf="${GGUF_DIR}/Qwen3.5-${size}/${TAG}.gguf"
    [[ -f "${gguf}" ]] || uvx --from huggingface_hub hf download "unsloth/Qwen3.5-${size}-GGUF" \
        "${TAG}.gguf" --local-dir "${GGUF_DIR}/Qwen3.5-${size}" > /dev/null
    echo "######## ${TAG} ########"
    log="logs/llama-${J}-${TAG}.log"
    start_llama "${gguf}" "${PORT}" "${CTX}" "${log}"
    if ! wait_llama "${PORT}" "${SERVER_PID}" "${log}"; then
        # A model that cannot start is a result; keep going with the others.
        stop_server "${SERVER_PID}"; failed+=("${TAG}:server"); continue
    fi
    for item in ${LEVELS}; do
        family="${item%%:*}" level="${item##*:}"
        for seed in ${SEEDS}; do
            name="${family}-${level}-s${seed}"
            instance="${RUNS}/instances/${name}"
            [[ -d "${instance}" ]] || continue
            echo "===== ${TAG} none ${name} ====="
            if ! uv run python -m ctxmgr.harness run "${instance}" \
                --out "${RUNS}/${TAG}/none/${name}" --condition none \
                --base-url "http://127.0.0.1:${PORT}/v1" --model "${TAG}" \
                --served-max-len "${CTX}" --precision "gguf-${QUANT}/llama.cpp-${LLAMA_BUILD}"; then
                failed+=("${TAG}/none/${name}:run")
            fi
        done
    done
    gpu_mem
    stop_server "${SERVER_PID}"
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL EPISODES COMPLETED (scores above; low reward is a result, not a job failure)"
