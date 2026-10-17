#!/bin/bash
# A40 pilot for a GGUF 27B manager under llama.cpp (replaces 27B-FP8 + vLLM if it passes).
# Phases on one 48 GB A40:
#   1. 27B manager alone: loads, buffer sizes, recall probe, decode speed.
#   2. Co-location: 27B manager + each Q8_0 actor as two servers on the same GPU, smoke-tested
#      concurrently. If the 9B pair fails with Q6_K, it is retried with the Q5_K_M manager.
# Run slurm/llamacpp_setup.sbatch first. Usage (from repo root): sbatch slurm/a40_gguf.sbatch
#SBATCH --job-name=ctxmgr-a40-gguf
#SBATCH --partition=education_gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh
source slurm/llamacpp.sh

BIG_QUANT="${BIG_QUANT:-Q6_K}"
FALLBACK_QUANT="${FALLBACK_QUANT:-Q5_K_M}"
ACTORS=(9B 4B)
# Qwen tokens run 1.22-1.40x o200k on our tasks, so a 32,768 o200k budget needs ~46K served tokens.
CTX="${CTX:-49152}"
P1=$((20000 + SLURM_JOB_ID % 10000))
P2=$((P1 + 1))
J="${SLURM_JOB_ID}"

big() { echo "${GGUF_DIR}/Qwen3.5-27B/Qwen3.5-27B-$1.gguf"; }
actor() { echo "${GGUF_DIR}/Qwen3.5-$1/Qwen3.5-$1-Q8_0.gguf"; }

echo "host=$(hostname) job=${J} llama.cpp=${LLAMA_BUILD} ctx=${CTX} big=${BIG_QUANT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"${LLAMA_BIN}/llama-server" --version 2>&1 | tail -2
declare -A status

# coloc <big_quant> <actor_size>: 0 if both servers start and both smoke tests pass.
coloc() {
    local q="$1" a="$2" tag="27b-$1+$2" ok=1
    echo "===== co-location: 27B ${q} + ${a} Q8_0 ====="
    start_llama "$(big "${q}")" "${P1}" "${CTX}" "logs/llama-${J}-${tag}-big.log"
    local big_pid=${SERVER_PID}
    if wait_llama "${P1}" "${big_pid}" "logs/llama-${J}-${tag}-big.log"; then
        start_llama "$(actor "${a}")" "${P2}" "${CTX}" "logs/llama-${J}-${tag}-actor.log"
        local actor_pid=${SERVER_PID}
        if wait_llama "${P2}" "${actor_pid}" "logs/llama-${J}-${tag}-actor.log"; then
            gpu_mem
            # Both servers answer at the same time, as manager and actor would in the grid.
            smoke "${P1}" "27b-${q}" "logs/smoke-${J}-${tag}-big.json" & local s1=$!
            smoke "${P2}" "${a}-Q8_0" "logs/smoke-${J}-${tag}-actor.json" & local s2=$!
            wait ${s1} && wait ${s2} && ok=0
            gpu_mem
        fi
        stop_server "${actor_pid}"
    fi
    stop_server "${big_pid}"
    return ${ok}
}

echo "===== phase 1: 27B ${BIG_QUANT} alone ====="
start_llama "$(big "${BIG_QUANT}")" "${P1}" "${CTX}" "logs/llama-${J}-27b-alone.log"
if wait_llama "${P1}" "${SERVER_PID}" "logs/llama-${J}-27b-alone.log" \
    && smoke "${P1}" "27b-${BIG_QUANT}" "logs/smoke-${J}-27b-alone.json"; then
    status[27b_${BIG_QUANT}_alone]=PASS; gpu_mem
else
    status[27b_${BIG_QUANT}_alone]=FAIL
fi
stop_server "${SERVER_PID}"

for a in "${ACTORS[@]}"; do
    if coloc "${BIG_QUANT}" "${a}"; then
        status[coloc_${BIG_QUANT}_${a}]=PASS
    else
        status[coloc_${BIG_QUANT}_${a}]=FAIL
        if [[ "${a}" == "9B" ]]; then
            coloc "${FALLBACK_QUANT}" "${a}" && status[coloc_${FALLBACK_QUANT}_${a}]=PASS \
                || status[coloc_${FALLBACK_QUANT}_${a}]=FAIL
        fi
    fi
done

echo "===== summary ====="
for k in "${!status[@]}"; do echo "${k}: ${status[$k]}"; done | sort
[[ " ${status[*]} " == *" FAIL "* ]] && exit 1
exit 0
