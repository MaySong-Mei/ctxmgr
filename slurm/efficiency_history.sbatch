#!/bin/bash
#SBATCH --job-name=ctx-eff-history
#SBATCH --partition=education
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source slurm/env.sh
: "${TEAM_RUNS:?Set TEAM_RUNS to the existing readable team run directory}"
uv run python -m ctxmgr.accounting.efficiency "$TEAM_RUNS" \
    --out "${PROJECT_DIR}/runs/efficiency-history-${SLURM_JOB_ID}"
