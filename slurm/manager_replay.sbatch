#!/bin/bash
# Identical-input replay of Needle manager events (scripts/manager_replay.py; pre-registered in
# docs/preregistration-v5.md on may/week1). Serves one manager model alone and re-sends the rebuilt
# requests of SRC's first three events per episode under every wording. No episode is run.
# Usage (from repo root):
#   MODEL=google/gemma-4-12B-it SAMPLED=1 sbatch --partition=gpu_h200 --qos=normal --gpus=h200:1 \
#       --mem=160G --cpus-per-task=8 slurm/manager_replay.sbatch
#SBATCH --job-name=manager-replay
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

MODEL="${MODEL:?set MODEL}"
SRC="${SRC:-${PROJECT_DIR}/runs/peer27b-31545119/Qwen3.5-2B/peer_31b}"
SAMPLED="${SAMPLED:-}"
# Comma lists (scripts/manager_replay.py --wordings / --sampled-wordings); empty = the script default.
WORDINGS="${WORDINGS:-}"
SAMPLED_WORDINGS="${SAMPLED_WORDINGS:-}"
MAX_LEN="${MAX_LEN:-81920}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT=$(free_ports 1)
J="${SLURM_JOB_ID}"
OUT="${PROJECT_DIR}/runs/mreplay-${J}"

echo "host=$(hostname) job=${J} model=${MODEL} src=${SRC} sampled=${SAMPLED:-0} max_len=${MAX_LEN}"
echo "wordings=${WORDINGS:-default} sampled_wordings=${SAMPLED_WORDINGS:-default}"
echo "commit=$(git rev-parse --short HEAD) out=${OUT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --group serve --quiet || exit 1
mkdir -p "${OUT}"
uv run python scripts/manager_replay.py plan "${SRC}" | tail -1 || exit 1

log="logs/vllm-${J}-${MODEL##*/}.log"
start_server "${MODEL}" "${PORT}" "${GPU_UTIL}" "${MAX_LEN}" "${log}" --max-num-seqs "${MAX_NUM_SEQS}"
if ! { wait_healthy "${PORT}" "${SERVER_PID}" "${log}" && check_model "${PORT}" "${MODEL}"; }; then
    stop_server "${SERVER_PID}"; exit 1
fi
uv run python scripts/manager_replay.py run "${SRC}" "${OUT}/replay.jsonl" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}" ${SAMPLED:+--sampled} \
    ${WORDINGS:+--wordings "${WORDINGS}"} ${SAMPLED_WORDINGS:+--sampled-wordings "${SAMPLED_WORDINGS}"}
status=$?
gpu_mem
stop_server "${SERVER_PID}"
[[ ${status} -eq 0 ]] || { echo "MANAGER REPLAY FAILED"; exit 1; }
uv run python scripts/manager_replay.py summarize "${OUT}/replay.jsonl"
echo "MANAGER REPLAY COMPLETED"
