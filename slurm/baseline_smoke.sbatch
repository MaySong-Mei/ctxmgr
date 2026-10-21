#!/bin/bash
# First end-to-end check: served Qwen model -> official ContextBench task -> Track 1 scorer,
# conditions none / self / peer_same (project-status.md "First model smoke test" and first managed pilot).
# Generates the instances inside the job (Python never runs on login nodes), then for each model:
# serve it, run each (family, level) at seed 42, print the scores, stop the server.
# Several models share one job because education_gpu allows one submitted job per user.
# Usage (from repo root):
#   sbatch slurm/baseline_smoke.sbatch                      # default: Qwen/Qwen3.5-2B
#   MODEL=Qwen/Qwen3.5-0.8B LEVELS="kv_store:R130 needle_retention:K4-C4" sbatch slurm/baseline_smoke.sbatch
#   MODELS="Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B" sbatch slurm/baseline_smoke.sbatch
#   CONDITIONS="none self peer_same" sbatch slurm/baseline_smoke.sbatch
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

MODELS="${MODELS:-${MODEL:-Qwen/Qwen3.5-2B}}"
LEVELS="${LEVELS:-kv_store:R130 needle_retention:K4-C4}"
SEEDS="${SEEDS:-${SEED:-42}}"
# Conditions served by one model: none, self, peer_same (peer_27b needs a second server).
CONDITIONS="${CONDITIONS:-none}"
TRIGGER="${TRIGGER:-18432}"
# The benchmark budget is 32,768 o200k tokens; Qwen's tokenizer can need more for the same text, so
# serve with headroom and let ContextBench's o200k wall decide when an episode ends.
MAX_LEN="${MAX_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT=$(free_ports 1)
J="${SLURM_JOB_ID}"
RUNS="${PROJECT_DIR}/runs/baseline-${J}"

echo "host=$(hostname) job=${J} models=${MODELS} conditions=${CONDITIONS} trigger=${TRIGGER} max_len=${MAX_LEN} levels=${LEVELS} seeds=${SEEDS}"
echo "commit=$(git rev-parse --short HEAD) runs=${RUNS}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --group serve --quiet || exit 1

failed=()
# Generate each instance once; every model sees identical streams.
for SEED in ${SEEDS}; do
    for item in ${LEVELS}; do
        family="${item%%:*}" level="${item##*:}"
        name="${family}-${level}-s${SEED}"
        if ! uv run python -m ctxmgr.tasks generate "${family}" --level "${level}" --seed "${SEED}" \
            --out "${RUNS}/instances/${name}" > /dev/null; then
            failed+=("${name}:generate")
        fi
    done
done

for MODEL in ${MODELS}; do
    TAG="${MODEL##*/}"
    echo "######## ${MODEL} ########"
    log="logs/vllm-${J}-${TAG}.log"
    start_server "${MODEL}" "${PORT}" "${GPU_UTIL}" "${MAX_LEN}" "${log}" --max-num-seqs "${MAX_NUM_SEQS}"
    if ! { wait_healthy "${PORT}" "${SERVER_PID}" "${log}" && check_model "${PORT}" "${MODEL}"; }; then
        # A model that cannot start is a result; keep going with the others.
        stop_server "${SERVER_PID}"; failed+=("${TAG}:server"); continue
    fi
    for SEED in ${SEEDS}; do
        for item in ${LEVELS}; do
            family="${item%%:*}" level="${item##*:}"
            name="${family}-${level}-s${SEED}"
            instance="${RUNS}/instances/${name}"
            [[ -d "${instance}" ]] || continue
            # peer_same reuses this server as a separate manager session (no second weight copy).
            for cond in ${CONDITIONS}; do
                echo "===== ${TAG} ${cond} ${name} ====="
                if ! uv run python -m ctxmgr.harness run "${instance}" \
                    --out "${RUNS}/${TAG}/${cond}/${name}" --condition "${cond}" \
                    --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" \
                    --served-max-len "${MAX_LEN}" --precision "${PRECISION:-bf16}" \
                    --trigger "${TRIGGER}"; then
                    failed+=("${TAG}/${cond}/${name}:run")
                fi
            done
        done
    done
    gpu_mem
    stop_server "${SERVER_PID}"
done

if [[ ${#failed[@]} -gt 0 ]]; then
    echo "FAILED: ${failed[*]}"; exit 1
fi
echo "ALL EPISODES COMPLETED (scores above; low reward is a result, not a job failure)"
