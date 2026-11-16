#!/usr/bin/env bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
set -euo pipefail
ROOT=${1:?root} BASE=${2:?base} ACTOR=${3:?actor} MANAGER=${4:?manager} CONDITIONS=${5:?conditions}
QUESTIONS=${6:-0:25} PARALLEL=${7:-6}
source "$ROOT/source/slurm/bcp_common.sh"
exec "$client_python" -m ctxmgr.bcp.suite job --base-suite "$BASE" --root "$ROOT" --data "$DATA" \
    --actor "$ACTOR" --manager "$MANAGER" --conditions "$CONDITIONS" \
    --questions "$QUESTIONS" --parallel "$PARALLEL"
