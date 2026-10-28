#!/bin/bash
#SBATCH --job-name=ctx-eff
#SBATCH --partition=gpu_h200
#SBATCH --qos=normal
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
source slurm/lib.sh
: "${ACTOR_SIZE:?Set ACTOR_SIZE}"
MODE=${MODE:-full}
RUNS="${PROJECT_DIR}/runs/efficiency-${SLURM_JOB_ID}"
mkdir -p "$RUNS"
pids=()
cleanup() {
    for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
hostname
git rev-parse HEAD > "$RUNS/source_commit.txt"
nvidia-smi --query-gpu=uuid,name,memory.total,driver_version --format=csv > "$RUNS/hardware.csv"
nvidia-smi --query-gpu=timestamp,uuid,memory.used,power.draw,utilization.gpu --format=csv -l 2 > "$RUNS/gpu-telemetry.csv" &
pids+=("$!")
uv sync --locked --group serve --quiet
read -r actor_port manager_port <<< "$(free_ports 2)"
startup=$(date +%s)
export PROMPT_TOKENS_DETAILS=1
if [[ "$ACTOR_SIZE" != 27B ]]; then
    manager_path=$(bash slurm/efficiency_models.sh 27B)
    start_server "$manager_path" "$manager_port" 0.55 81920 "$RUNS/manager.log" \
        --served-model-name Qwen/Qwen3.5-27B --max-num-seqs 64 --enable-prefix-caching
    pids+=("$SERVER_PID")
    wait_healthy "$manager_port" "$SERVER_PID" "$RUNS/manager.log"
    check_model "$manager_port" Qwen/Qwen3.5-27B
    actor_util=0.38
else
    actor_util=0.90
fi
actor_path=$(bash slurm/efficiency_models.sh "$ACTOR_SIZE")
start_server "$actor_path" "$actor_port" "$actor_util" 81920 "$RUNS/actor.log" \
    --served-model-name "Qwen/Qwen3.5-${ACTOR_SIZE}" --max-num-seqs 64 --enable-prefix-caching
pids+=("$SERVER_PID")
wait_healthy "$actor_port" "$SERVER_PID" "$RUNS/actor.log"
check_model "$actor_port" "Qwen/Qwen3.5-${ACTOR_SIZE}"
printf '{"startup_s":%s,"gpus":1,"actor_size":"%s","mode":"%s"}\n' \
    "$(( $(date +%s) - startup ))" "$ACTOR_SIZE" "$MODE" > "$RUNS/job-metadata.json"
extra=()
qaextra=()
if [[ "$ACTOR_SIZE" != 27B ]]; then
    extra+=(--manager-url "http://127.0.0.1:${manager_port}/v1")
    qaextra+=(--manager-url "http://127.0.0.1:${manager_port}/v1" --manager-model Qwen/Qwen3.5-27B)
    conditions=(none self peer_27b)
else
    conditions=(none self peer_same)
fi
[[ "$MODE" == smoke ]] && extra+=(--smoke)
uv run python scripts/efficiency_grid.py --actor "$ACTOR_SIZE" \
    --base-url "http://127.0.0.1:${actor_port}/v1" --out "$RUNS/contextbench" "${extra[@]}"
cases=8
[[ "$MODE" == smoke ]] && cases=1
uv run python scripts/multidoc_pilot.py \
    --data "${PROJECT_DIR}/datasets/hotpot_dev_distractor_v1.json" \
    --out "$RUNS/multidoc" --model "Qwen/Qwen3.5-${ACTOR_SIZE}" \
    --base-url "http://127.0.0.1:${actor_port}/v1" --conditions "${conditions[@]}" \
    --cases "$cases" "${qaextra[@]}"
uv run python -m ctxmgr.accounting.efficiency "$RUNS/contextbench" --out "$RUNS/analysis" --lenient
echo "ALL EFFICIENCY AND MULTIDOC EPISODES COMPLETED: $RUNS"
