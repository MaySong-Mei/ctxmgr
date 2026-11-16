#!/usr/bin/env bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=128G
#SBATCH --time=08:00:00
set -euo pipefail
ROOT=${1:?root} BASE=${2:?base}
source "$ROOT/source/slurm/bcp_common.sh"
exec "$client_python" -m ctxmgr.bcp.suite judge --base-suite "$BASE" --root "$ROOT" --data "$DATA"
