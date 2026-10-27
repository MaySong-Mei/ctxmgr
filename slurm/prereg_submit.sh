#!/bin/bash
# Submit the pre-registered runs (docs/preregistration.md on may/track3-infra). Every setting is fixed
# here; nothing is taken from the caller's environment.
#
#   git fetch --tags && git checkout --detach prereg-v1-harness   # in the cluster worktree
#   bash slurm/prereg_submit.sh canary                  # C1, C2 (seed 46), first
#   bash slurm/prereg_submit.sh blocks                  # J1-J13 (seeds 50-65), after the canary passes
#   bash slurm/prereg_submit.sh blocks J7 J8            # only the named jobs (e.g. after a failed submit)
#   bash slurm/prereg_submit.sh rerun 9B kv_store:R1000 mask_arc 53   # one cell, same settings
#
# Refuses to submit unless HEAD is the tagged commit and tracked files are unmodified. Stops at the
# first failed sbatch.
set -euo pipefail

TAG=prereg-v1-harness
want=$(git rev-list -n 1 "${TAG}")
if [[ "$(git rev-parse HEAD)" != "${want}" ]]; then
    echo "HEAD is not ${TAG} (${want}); check it out detached first"; exit 1
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "tracked files are modified; refusing"; exit 1
fi

S8="50 51 52 53 54 55 56 57"
S16="50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65"
H200=(--partition=gpu_h200 --qos=normal --gpus=h200:1 --mem=160G --cpus-per-task=8)
# Drop every knob the job scripts read that is not set explicitly below (sbatch exports the env).
CLEAR=(-u ACTOR_GPU -u MANAGER_GPU -u CUDA_VISIBLE_DEVICES -u MODEL -u SEED -u SAMPLING
       -u SPLIT_GPUS -u WAIT_HEALTHY_S MAX_NUM_SEQS=64 MAX_TURNS_PER_OP=2)

# small actor + co-located 27B bf16 manager (peer27b_pilot.sbatch)
small() {  # small <job name> <actor> <seeds> <levels> <conditions> <time>
    env "${CLEAR[@]}" \
        ACTOR="Qwen/Qwen3.5-$2" MANAGER="Qwen/Qwen3.5-27B" SEEDS="$3" LEVELS="$4" CONDITIONS="$5" \
        TRIGGER=18432 BIG_UTIL=0.55 ACTOR_UTIL=0.38 MAX_LEN=81920 ACTOR_MAX_LEN=81920 \
        PROMPT_TOKENS_DETAILS=1 \
        sbatch --parsable "${H200[@]}" --time="$6" --job-name="$1" slurm/peer27b_pilot.sbatch
}

# 27B actor alone (baseline_smoke.sbatch); peer_same reuses the same server
big() {  # big <job name> <seeds> <levels> <conditions> <time>
    env "${CLEAR[@]}" \
        MODELS="Qwen/Qwen3.5-27B" SEEDS="$2" LEVELS="$3" CONDITIONS="$4" \
        TRIGGER=18432 MAX_LEN=81920 GPU_UTIL=0.90 PRECISION=bf16 PROMPT_TOKENS_DETAILS=1 \
        sbatch --parsable "${H200[@]}" --time="$5" --job-name="$1" slurm/baseline_smoke.sbatch
}

NK4="needle_retention:K4-C24"
job() {  # job <name>: submit one pre-registered job; print "<name> <slurm id>" or stop
    local id
    case "$1" in
        C1) id=$(small prereg-C1 9B 46 "${NK4} kv_store:R1000" "peer_27b mask_arc" 06:00:00) ;;
        C2) id=$(big prereg-C2 46 "${NK4}" "none" 06:00:00) ;;
        J1) id=$(small prereg-J1 2B "${S16}" "${NK4}" "none peer_27b" 24:00:00) ;;
        J2) id=$(small prereg-J2 4B "${S16}" "${NK4}" "none peer_27b" 24:00:00) ;;
        J3) id=$(small prereg-J3 9B "${S16}" "${NK4}" "none peer_27b" 24:00:00) ;;
        J4) id=$(small prereg-J4 2B "${S8}" "${NK4}" "self" 12:00:00) ;;
        J5) id=$(small prereg-J5 4B "${S8}" "${NK4}" "self" 12:00:00) ;;
        J6) id=$(small prereg-J6 9B "${S8}" "${NK4}" "self_arc peer_same_arc mask_arc" 24:00:00) ;;
        J7) id=$(small prereg-J7 9B "${S16}" kv_store:R1000 "none mask_arc peer_27b_arc" 24:00:00) ;;
        J8) id=$(small prereg-J8 4B "${S16}" kv_store:R1000 "none mask_arc peer_27b_arc" 24:00:00) ;;
        J9) id=$(small prereg-J9 9B "${S8}" kv_store:R2000 "none peer_27b mask_arc peer_27b_arc" 24:00:00) ;;
        J10) id=$(small prereg-J10 2B "${S8}" needle_retention:K2-C192 "none peer_27b" 36:00:00) ;;
        J11) id=$(small prereg-J11 9B "${S8}" needle_retention:K2-C192 "none peer_27b" 36:00:00) ;;
        J12) id=$(big prereg-J12 "${S8}" kv_store:R1000 "none_r self_arc mask_arc" 24:00:00) ;;
        J13) id=$(big prereg-J13 "${S16}" "${NK4}" "none self peer_same" 36:00:00) ;;
        *) echo "unknown job $1"; exit 2 ;;
    esac
    [[ -n "${id}" ]] || { echo "$1: sbatch returned no job id; stopping"; exit 1; }
    echo "$1 ${id}"
}

case "${1:-}" in
    canary)
        job C1; job C2
        ;;
    blocks)
        shift
        names=("$@")
        [[ ${#names[@]} -gt 0 ]] || names=(J1 J2 J3 J4 J5 J6 J7 J8 J9 J10 J11 J12 J13)
        for name in "${names[@]}"; do job "${name}"; done
        ;;
    rerun)  # rerun <actor> <family:level> <condition> <seed>: one cell stopped by infrastructure
        [[ $# -eq 5 ]] || { echo "usage: $0 rerun <actor> <family:level> <condition> <seed>"; exit 2; }
        if [[ "$2" == 27B ]]; then
            id=$(big "prereg-R-$4-$5" "$5" "$3" "$4" 36:00:00)
        else
            id=$(small "prereg-R-$2-$4-$5" "$2" "$5" "$3" "$4" 36:00:00)
        fi
        [[ -n "${id}" ]] || { echo "sbatch returned no job id"; exit 1; }
        echo "rerun $2 $3 $4 $5 ${id}"
        ;;
    *)
        echo "usage: $0 canary | blocks [J1 ... J13] | rerun <actor> <family:level> <condition> <seed>"
        exit 2
        ;;
esac
