#!/usr/bin/env bash
#SBATCH --partition=day
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# Generates and validates task instances for the added seeds (46-49).
set -euo pipefail
root=${1:?root} base=${2:?base suite}
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: run inside a Slurm allocation.' >&2; exit 2; }
case "$(hostname -s)" in login*) echo 'ERROR: refusing Python on a login node.' >&2; exit 2 ;; esac
cd "$root/source" && sha256sum --check --quiet "$root/source.sha256"
set -a; source "$base/config.env"; set +a
export PYTHONPATH="$root/source/src" LITELLM_LOCAL_MODEL_COST_MAP=True
export TIKTOKEN_CACHE_DIR="${PROJECT_DIR}/tiktoken_cache"
exec "$base/source/envs/clm/.venv/bin/python" -m ctxmgr.harness.whenhow_suite instances \
    --out "$root/instances"
