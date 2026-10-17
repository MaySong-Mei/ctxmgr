#!/bin/bash
# peer_27b pilot: a fixed 27B-FP8 manager and a small actor co-located on one A40, plus the same
# actor's none / self / peer_same conditions on the same instances for comparison.
# Only 2B (and presumably 0.8B) fits beside the 27B manager (a40_27b.sbatch re-run, CLAUDE.md).
# Usage (from repo root):
#   sbatch slurm/peer27b_pilot.sbatch
#   ACTOR=Qwen/Qwen3.5-0.8B LEVELS="needle_retention:K4-C24" CONDITIONS="none peer_27b" sbatch ...
#SBATCH --job-name=ctxmgr-peer27b
#SBATCH --partition=education_gpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

ACTOR="${ACTOR:-Qwen/Qwen3.5-2B}"
MANAGER="${MANAGER:-Qwen/Qwen3.5-27B-FP8}"
LEVELS="${LEVELS:-needle_retention:K4-C24}"
CONDITIONS="${CONDITIONS:-none self peer_same peer_27b}"
SEED="${SEED:-42}"
TRIGGER="${TRIGGER:-18432}"
MAX_LEN="${MAX_LEN:-49152}"
BIG_UTIL="${BIG_UTIL:-0.75}"
ACTOR_UTIL="${ACTOR_UTIL:-0.21}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
P1=$((20000 + SLURM_JOB_ID % 10000))
P2=$((P1 + 1))
J="${SLURM_JOB_ID}"
TAG="${ACTOR##*/}"
RUNS="${PROJECT_DIR}/runs/peer27b-${J}"

echo "host=$(hostname) job=${J} actor=${ACTOR} manager=${MANAGER} conditions=${CONDITIONS}"
echo "levels=${LEVELS} seed=${SEED} trigger=${TRIGGER} max_len=${MAX_LEN} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --group serve --quiet || exit 1

failed=()
for item in ${LEVELS}; do
    family="${item%%:*}" level="${item##*:}"
    name="${family}-${level}-s${SEED}"
    uv run python -m ctxmgr.tasks generate "${family}" --level "${level}" --seed "${SEED}" \
        --out "${RUNS}/instances/${name}" > /dev/null || failed+=("${name}:generate")
done

# The manager starts first and takes its share; the actor fits in what is left.
start_server "${MANAGER}" "${P1}" "${BIG_UTIL}" "${MAX_LEN}" "logs/vllm-${J}-manager.log" \
    --max-num-seqs "${MAX_NUM_SEQS}"
big_pid=${SERVER_PID}
wait_healthy "${P1}" "${big_pid}" "logs/vllm-${J}-manager.log" || { stop_server "${big_pid}"; exit 1; }
start_server "${ACTOR}" "${P2}" "${ACTOR_UTIL}" "${MAX_LEN}" "logs/vllm-${J}-actor.log" \
    --max-num-seqs "${MAX_NUM_SEQS}"
actor_pid=${SERVER_PID}
if ! wait_healthy "${P2}" "${actor_pid}" "logs/vllm-${J}-actor.log"; then
    stop_server "${actor_pid}"; stop_server "${big_pid}"; exit 1
fi
gpu_mem

for item in ${LEVELS}; do
    family="${item%%:*}" level="${item##*:}"
    name="${family}-${level}-s${SEED}"
    instance="${RUNS}/instances/${name}"
    [[ -d "${instance}" ]] || continue
    for cond in ${CONDITIONS}; do
        echo "===== ${TAG} ${cond} ${name} ====="
        extra=()
        [[ "${cond}" == "peer_27b" ]] && extra=(--manager-base-url "http://127.0.0.1:${P1}/v1" \
            --manager-model "${MANAGER}")
        if ! uv run python -m ctxmgr.harness run "${instance}" \
            --out "${RUNS}/${TAG}/${cond}/${name}" --condition "${cond}" \
            --base-url "http://127.0.0.1:${P2}/v1" --model "${ACTOR}" \
            --served-max-len "${MAX_LEN}" --precision bf16 --trigger "${TRIGGER}" "${extra[@]}"; then
            failed+=("${cond}/${name}:run")
        fi
    done
done
gpu_mem
stop_server "${actor_pid}"
stop_server "${big_pid}"

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL EPISODES COMPLETED (low reward is a result, not a job failure)"
