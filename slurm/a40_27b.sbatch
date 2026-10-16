#!/bin/bash
# A40 pilot for the 27B cells. Three phases on one 48 GB A40:
#   1. 27B-FP8 alone: does it load, how much KV cache is left, decode speed.
#   2. 9B with online FP8 alone: memory and speed (candidate precision for the 9B row).
#   3. Co-location: 27B-FP8 manager + a small actor as two servers on the same GPU,
#      smoke-tested concurrently. Tried for each actor in ACTORS.
# Usage (from repo root): sbatch slurm/a40_27b.sbatch
#SBATCH --job-name=ctxmgr-a40
#SBATCH --partition=education_gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

BIG="Qwen/Qwen3.5-27B-FP8"
ACTORS=("Qwen/Qwen3.5-2B" "Qwen/Qwen3.5-4B" "Qwen/Qwen3.5-9B")
MAX_LEN="${MAX_LEN:-32768}"
# Memory split for co-location; the manager starts first and takes its share.
BIG_UTIL="${BIG_UTIL:-0.75}"
ACTOR_UTIL="${ACTOR_UTIL:-0.21}"
# Hybrid attention: each running sequence needs one Mamba state block, and vLLM refuses to start if
# max_num_seqs (default 256) exceeds the blocks that fit (27B-FP8: 173 at util 0.90, 79 at 0.75).
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
# Online FP8 crashes torch.compile in vLLM 0.30 / torch 2.13 ("auto_functionalized was not removed"),
# so 9B-FP8 runs eager (no CUDA graphs): its decode speed is a lower bound.
FP8=(--quantization fp8 --enforce-eager)
P1=$((20000 + SLURM_JOB_ID % 10000))
P2=$((P1 + 1))
J="${SLURM_JOB_ID}"

echo "host=$(hostname) job=${J} max_len=${MAX_LEN} big_util=${BIG_UTIL} actor_util=${ACTOR_UTIL}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
declare -A status

echo "===== phase 1: ${BIG} alone ====="
start_server "${BIG}" "${P1}" 0.90 "${MAX_LEN}" "logs/vllm-${J}-27b-alone.log" \
    --max-num-seqs "${MAX_NUM_SEQS}"
if wait_healthy "${P1}" "${SERVER_PID}" "logs/vllm-${J}-27b-alone.log" \
    && smoke "${P1}" "${BIG}" "logs/smoke-${J}-27b-alone.json"; then
    status[27b_alone]=PASS; gpu_mem
else
    status[27b_alone]=FAIL
fi
stop_server "${SERVER_PID}"

echo "===== phase 2: Qwen3.5-9B online FP8 alone ====="
start_server "Qwen/Qwen3.5-9B" "${P1}" 0.90 "${MAX_LEN}" "logs/vllm-${J}-9b-fp8.log" \
    --max-num-seqs "${MAX_NUM_SEQS}" "${FP8[@]}"
if wait_healthy "${P1}" "${SERVER_PID}" "logs/vllm-${J}-9b-fp8.log" \
    && smoke "${P1}" "Qwen/Qwen3.5-9B" "logs/smoke-${J}-9b-fp8.json"; then
    status[9b_fp8]=PASS; gpu_mem
else
    status[9b_fp8]=FAIL
fi
stop_server "${SERVER_PID}"

for actor in "${ACTORS[@]}"; do
    tag="${actor##*/}"
    extra=(--max-num-seqs "${MAX_NUM_SEQS}")
    # 9B in bf16 cannot fit beside 27B-FP8; try it as online FP8 (the candidate 9B precision).
    [[ "${tag}" == "Qwen3.5-9B" ]] && extra+=("${FP8[@]}")
    echo "===== phase 3: ${BIG} (${BIG_UTIL}) + ${actor} (${ACTOR_UTIL}) ${extra[*]} ====="

    start_server "${BIG}" "${P1}" "${BIG_UTIL}" "${MAX_LEN}" "logs/vllm-${J}-coloc-${tag}-big.log" \
        --max-num-seqs "${MAX_NUM_SEQS}"
    big_pid=${SERVER_PID}
    ok=0
    if wait_healthy "${P1}" "${big_pid}" "logs/vllm-${J}-coloc-${tag}-big.log"; then
        start_server "${actor}" "${P2}" "${ACTOR_UTIL}" "${MAX_LEN}" \
            "logs/vllm-${J}-coloc-${tag}-actor.log" "${extra[@]}"
        actor_pid=${SERVER_PID}
        if wait_healthy "${P2}" "${actor_pid}" "logs/vllm-${J}-coloc-${tag}-actor.log"; then
            gpu_mem
            # Both servers answer at the same time, as manager and actor would in the grid.
            smoke "${P1}" "${BIG}" "logs/smoke-${J}-coloc-${tag}-big.json" & s1=$!
            smoke "${P2}" "${actor}" "logs/smoke-${J}-coloc-${tag}-actor.json" & s2=$!
            wait ${s1} && wait ${s2} && ok=1
        fi
        stop_server "${actor_pid}"
    fi
    stop_server "${big_pid}"
    status[coloc_${tag}]=$([[ ${ok} -eq 1 ]] && echo PASS || echo FAIL)
done

echo "===== summary ====="
for k in "${!status[@]}"; do echo "${k}: ${status[$k]}"; done | sort
[[ " ${status[*]} " == *" FAIL "* ]] && exit 1
exit 0
