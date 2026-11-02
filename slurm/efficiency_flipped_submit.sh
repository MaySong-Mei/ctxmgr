#!/bin/bash
# Flipped roles: 27B actor with 2B/4B/9B managers (peer_small, peer_small_arc, QA long peer_small).
# Setup re-runs tests/lint/dataset checks first; at most two GPU jobs run concurrently.
set -euo pipefail
: "${TEAM_MODEL_CACHE:?Set TEAM_MODEL_CACHE}"
manifest=${1:?Usage: efficiency_flipped_submit.sh NEW_MANIFEST_PATH}
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
a=$(ACTOR_SIZE=27B MANAGER_SIZE=2B MODE=flipped submit flip2b --dependency="afterok:$setup" --job-name=ctx-flip-2B slurm/efficiency_gpu.sbatch)
b=$(ACTOR_SIZE=27B MANAGER_SIZE=4B MODE=flipped submit flip4b --dependency="afterok:$setup" --job-name=ctx-flip-4B slurm/efficiency_gpu.sbatch)
c=$(ACTOR_SIZE=27B MANAGER_SIZE=9B MODE=flipped submit flip9b --dependency="afterok:$setup,afterany:$a" --job-name=ctx-flip-9B slurm/efficiency_gpu.sbatch)
cat "$manifest"
