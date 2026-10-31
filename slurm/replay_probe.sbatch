#!/bin/bash
# Replay probe for the 27B actor's output-cap loop (scripts/replay_probe.py; pre-registered in
# docs/preregistration-v3.md on may/week1). Serves the actor model with the round-29 settings and
# re-sends logged actor inputs from SRC, one change per arm. No episode is run.
# Usage (from repo root):
#   sbatch --partition=gpu_h200 --qos=normal --gpus=h200:1 --mem=160G --cpus-per-task=8 \
#       slurm/replay_probe.sbatch
#   SRC=/path/to/runs/baseline-<job>/Qwen3.5-27B sbatch ... slurm/replay_probe.sbatch
#SBATCH --job-name=replay-probe
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x-%j.out

set -uo pipefail
source slurm/env.sh
source slurm/lib.sh

MODEL="${MODEL:-Qwen/Qwen3.5-27B}"
SRC="${SRC:-${PROJECT_DIR}/runs/baseline-31451066/Qwen3.5-27B}"
# Same server settings as the episodes being replayed (round 29: max_len 81,920, 64 sequences).
MAX_LEN="${MAX_LEN:-81920}"
GPU_UTIL="${GPU_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
PORT=$(free_ports 1)
J="${SLURM_JOB_ID}"
OUT="${PROJECT_DIR}/runs/replay-${J}"

echo "host=$(hostname) job=${J} model=${MODEL} src=${SRC} max_len=${MAX_LEN}"
echo "commit=$(git rev-parse --short HEAD) out=${OUT}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

uv sync --locked --group serve --quiet || exit 1
mkdir -p "${OUT}"
uv run python scripts/replay_probe.py plan "${SRC}" || exit 1

log="logs/vllm-${J}-${MODEL##*/}.log"
start_server "${MODEL}" "${PORT}" "${GPU_UTIL}" "${MAX_LEN}" "${log}" --max-num-seqs "${MAX_NUM_SEQS}"
if ! { wait_healthy "${PORT}" "${SERVER_PID}" "${log}" && check_model "${PORT}" "${MODEL}"; }; then
    stop_server "${SERVER_PID}"; exit 1
fi
uv run python scripts/replay_probe.py run "${SRC}" "${OUT}/replay.jsonl" \
    --base-url "http://127.0.0.1:${PORT}/v1" --model "${MODEL}"
status=$?
gpu_mem
stop_server "${SERVER_PID}"
[[ ${status} -eq 0 ]] || { echo "REPLAY FAILED"; exit 1; }
uv run python scripts/replay_probe.py summarize "${OUT}/replay.jsonl"
echo "REPLAY COMPLETED"
