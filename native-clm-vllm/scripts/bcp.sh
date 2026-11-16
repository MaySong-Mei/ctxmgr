#!/usr/bin/env bash
# Login-node launcher for BrowseComp-Plus. Runs no Python.
#
#   bash scripts/bcp.sh init BASE_SUITE [ROOT] [N_QUESTIONS]   freeze source, prepare data
#   bash scripts/bcp.sh pilot ROOT                             5 questions, clm_self, 9B and 27B
#   bash scripts/bcp.sh grid ROOT [N_SHARDS]                   all conditions, 25 questions per shard
#   bash scripts/bcp.sh judge ROOT                             judge every finished episode
set -euo pipefail
here=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
mode=${1:?usage: see header}
submit() { sbatch --parsable --output="$root/logs/%x-%j.out" "$@"; }
case "$mode" in
init)
    base=${2:?BASE_SUITE}
    [[ -f "$base/ready.json" ]] || { echo "Base suite is not ready: $base" >&2; exit 1; }
    root=${3:-$(dirname "$base")/bcp-$(date -u +%Y%m%dT%H%M%SZ)}
    n=${4:-100}
    [[ ! -e "$root" ]] || { echo "Refusing to reuse existing root: $root" >&2; exit 1; }
    mkdir -p "$root/source" "$root/logs"
    chmod 700 "$root"
    for item in src slurm scripts envs SHA256SUMS; do cp -a "$here/$item" "$root/source/"; done
    rm -rf "$root/source/envs/"*/.venv
    git -C "$here" rev-parse HEAD > "$root/source/COMMIT"
    (cd "$root/source" && find src slurm scripts envs -type f -print0 | sort -z | xargs -0 sha256sum) \
        > "$root/source.sha256"
    echo "$base" > "$root/BASE"
    id=$(submit --job-name=bcp-prepare "$root/source/slurm/bcp_prepare.sbatch" "$root" "$base" "$n")
    printf 'prepare\t%s\n' "$id" > "$root/jobs.tsv"
    echo "Root: $root (prepare job $id)"
    ;;
pilot)
    root=${2:?ROOT}; base=$(cat "$root/BASE")
    for actor in 9B 27B; do
        id=$(submit --job-name="bcp-pilot-$actor" "$root/source/slurm/bcp_job.sbatch" \
            "$root" "$base" "$actor" same clm_self 0:5 5)
        printf 'pilot-%s\t%s\n' "$actor" "$id" >> "$root/jobs.tsv"
    done
    tail -2 "$root/jobs.tsv"
    ;;
grid)
    root=${2:?ROOT}; base=$(cat "$root/BASE"); shards=${3:-2}
    self=clm_self,clm_self_nonudge,clm_fixed
    mgr=mgr_fixed,mgr_request,mgr_watch
    for ((s = 0; s < shards; s++)); do
        q="$((s * 25)):$((s * 25 + 25))"
        for spec in "9B same $self" "9B same $mgr" "9B 27B $mgr" "27B same $self" "27B same $mgr"; do
            set -- $spec
            id=$(submit --job-name="bcp-$1-$2-${3//,/+}-$q" "$root/source/slurm/bcp_job.sbatch" \
                "$root" "$base" "$1" "$2" "$3" "$q" 6)
            printf 'grid\t%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$q" "$id" >> "$root/jobs.tsv"
        done
    done
    grep '^grid' "$root/jobs.tsv" | column -t
    ;;
judge)
    root=${2:?ROOT}; base=$(cat "$root/BASE")
    id=$(submit --job-name=bcp-judge "$root/source/slurm/bcp_judge.sbatch" "$root" "$base")
    printf 'judge\t%s\n' "$id" >> "$root/jobs.tsv"; echo "judge job $id"
    ;;
*) echo "Unknown mode: $mode" >&2; exit 2 ;;
esac
