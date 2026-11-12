#!/usr/bin/env bash
# Login-node launcher for the when-versus-how grid. Runs no Python.
#
#   bash scripts/whenhow.sh submit BASE_SUITE [ROOT]
#
# BASE_SUITE is a finished `clm_vllm.sh submit` suite (models, environments, instances).
# Submits 28 single-GPU jobs: actors 2B/4B/9B/27B with their own server (seeds 42-45),
# plus actors 2B/4B/9B with a 27B manager (seeds 42-45). The source is frozen into ROOT.
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:?usage: whenhow.sh submit BASE_SUITE [ROOT]}
base=${2:?BASE_SUITE}
[[ "$mode" == submit ]] || { echo "Unknown mode: $mode" >&2; exit 2; }
[[ -f "$base/ready.json" ]] || { echo "Base suite is not ready: $base" >&2; exit 1; }
root=${3:-$(dirname "$base")/whenhow-$(date -u +%Y%m%dT%H%M%SZ)}
[[ ! -e "$root" ]] || { echo "Refusing to reuse existing root: $root" >&2; exit 1; }
mkdir -p "$root/source" "$root/logs"
for item in src slurm scripts SHA256SUMS; do cp -a "$here/$item" "$root/source/"; done
git -C "$here" rev-parse HEAD > "$root/source/COMMIT"
(cd "$root/source" && find src slurm scripts -type f -print0 | sort -z | xargs -0 sha256sum) \
    > "$root/source.sha256"
printf 'actor\tmanager\tseed\tjob_id\n' > "$root/jobs.tsv"
submit() {
    local actor=$1 manager=$2 seed=$3 id
    id=$(sbatch --parsable --job-name="whenhow-${actor}-${manager}-s${seed}" \
        --output="$root/logs/%x-%j.out" "$root/source/slurm/whenhow.sbatch" \
        "$root" "$base" "$actor" "$manager" "$seed")
    printf '%s\t%s\t%s\t%s\n' "$actor" "$manager" "$seed" "$id" >> "$root/jobs.tsv"
}
for seed in 42 43 44 45; do
    for actor in 2B 4B 9B 27B; do submit "$actor" same "$seed"; done
    for actor in 2B 4B 9B; do submit "$actor" 27B "$seed"; done
done
echo "Root: $root"
column -t "$root/jobs.tsv"
