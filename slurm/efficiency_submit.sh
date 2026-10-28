#!/bin/bash
set -euo pipefail
: "${TEAM_MODEL_CACHE:?Set TEAM_MODEL_CACHE}"
: "${TEAM_RUNS:?Set TEAM_RUNS}"
manifest=${1:?Usage: efficiency_submit.sh NEW_MANIFEST_PATH}
(set -o noclobber; : > "$manifest")
mkdir -p logs
submit() {
    local label=$1
    shift
    local job
    job=$(sbatch --parsable "$@")
    [[ "$job" =~ ^[0-9]+$ ]] || { echo "Unexpected sbatch output: $job" >&2; exit 1; }
    printf '%s\t%s\n' "$label" "$job" >> "$manifest"
    printf '%s\n' "$job"
}
setup=$(submit setup slurm/efficiency_setup.sbatch)
history=$(submit history --dependency="afterok:$setup" slurm/efficiency_history.sbatch)
smoke=$(ACTOR_SIZE=2B MODE=smoke submit smoke --dependency="afterok:$setup" --time=06:00:00 --job-name=ctx-eff-smoke slurm/efficiency_gpu.sbatch)
a=$(ACTOR_SIZE=2B MODE=full submit actor2b --dependency="afterok:$smoke" --job-name=ctx-eff-2B slurm/efficiency_gpu.sbatch)
b=$(ACTOR_SIZE=4B MODE=full submit actor4b --dependency="afterok:$smoke" --job-name=ctx-eff-4B slurm/efficiency_gpu.sbatch)
c=$(ACTOR_SIZE=9B MODE=full submit actor9b --dependency="afterok:$smoke,afterany:$a" --job-name=ctx-eff-9B slurm/efficiency_gpu.sbatch)
d=$(ACTOR_SIZE=27B MODE=full submit actor27b --dependency="afterok:$smoke,afterany:$b" --job-name=ctx-eff-27B slurm/efficiency_gpu.sbatch)
cat "$manifest"
