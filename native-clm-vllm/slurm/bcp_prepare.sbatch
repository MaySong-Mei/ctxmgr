#!/usr/bin/env bash
#SBATCH --partition=day
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
# Locks envs/bcp, decrypts the question subset (owner-only file) and builds the BM25 index.
set -euo pipefail
ROOT=${1:?root} BASE=${2:?base suite} N=${3:-100}
source "$ROOT/source/slurm/bcp_common.sh"
uv sync --locked --python 3.12 --project "$ROOT/source/envs/bcp"
umask 077
"$BCP_PYTHON" -m ctxmgr.bcp.data prepare --out "$DATA" --questions "$N"
