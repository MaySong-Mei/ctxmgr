#!/usr/bin/env bash
# Continue only a prepared suite whose four 27B cells never created results.
# All Python and result validation run in a CPU allocation, not on login nodes.
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
[[ $# -ge 1 && $# -le 2 ]] || fail 'Usage: bash scripts/clm_resume.sh SUITE [--check]'
[[ $# -eq 1 || "$2" == --check ]] || fail 'Only --check is accepted after SUITE.'
[[ "$1" = /* && -d "$1/source" ]] || fail 'Expected an absolute existing suite path.'
suite=$(cd -- "$1" && pwd -P)
for name in sbatch squeue scontrol sha256sum flock awk; do
    command -v "$name" >/dev/null || fail "Required command not found: $name"
done
for item in config.env config.sha256 source.sha256 jobs.tsv ready.json; do
    [[ -f "$suite/$item" ]] || fail "Missing $suite/$item; this is not a prepared suite."
done
(cd "$suite" && sha256sum --check --quiet config.sha256)
# This is the same user-owned, frozen resource config used by the original run.
set -a
source "$suite/config.env"
set +a
for key in CLM_ACCOUNT CLM_QOS CLM_CPU_PARTITION CLM_GPU_27B_PARTITION CLM_GPU_27B_TYPE; do
    [[ "${!key:-}" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail "Invalid or missing resource setting: $key"
done
for key in CLM_GPU_CPUS CLM_REPORT_CPUS; do
    [[ "${!key:-}" =~ ^[1-9][0-9]*$ ]] || fail "Invalid or missing CPU setting: $key"
done
for key in CLM_GPU_27B_MEM CLM_REPORT_MEM; do
    [[ "${!key:-}" =~ ^[1-9][0-9]*[KMGT]?$ ]] || fail "Invalid or missing memory setting: $key"
done
for key in CLM_GPU_TIME CLM_REPORT_TIME; do
    [[ "${!key:-}" =~ ^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || fail "Invalid or missing time setting: $key"
done
base=(--account="$CLM_ACCOUNT" --qos="$CLM_QOS" --nodes=1 --ntasks=1)
cpu=("${base[@]}" --partition="$CLM_CPU_PARTITION" --cpus-per-task="$CLM_REPORT_CPUS"
     --mem="$CLM_REPORT_MEM" --time="$CLM_REPORT_TIME")
gpu=("${base[@]}" --partition="$CLM_GPU_27B_PARTITION" --gpus-per-node="${CLM_GPU_27B_TYPE}:1"
     --cpus-per-task="$CLM_GPU_CPUS" --mem="$CLM_GPU_27B_MEM" --time="$CLM_GPU_TIME")
if [[ ${2:-} == --check ]]; then
    sbatch --test-only "${cpu[@]}" --wrap=true
    sbatch --test-only "${gpu[@]}" --wrap=true
    printf 'Resource requests accepted. No jobs were submitted; the full preflight runs on a CPU worker.\n'
    exit 0
fi
exec 9> "$suite/submit.lock"
flock -w 10 9 || fail 'Another submitter holds this suite lock.'
resume_dir="$suite/continuation-27b-v1"
if [[ -e "$resume_dir" ]]; then
    printf 'A continuation is already recorded: %s\n' "$resume_dir"
    [[ ! -f "$resume_dir/state" ]] || cat "$resume_dir/state"
    [[ ! -f "$resume_dir/jobs.tsv" ]] || cat "$resume_dir/jobs.tsv"
    printf 'No duplicate jobs were submitted. Inspect its logs before any retry.\n'
    exit 0
fi
# Query the user queue once: completed IDs need not be accepted by squeue -j.
queue=$(squeue -h -u "$USER" -o '%F') || fail 'Could not inspect the current queue; nothing submitted.'
while IFS=$'\t' read -r role job; do
    [[ "$job" =~ ^[0-9]+$ ]] || continue
    if printf '%s\n' "$queue" | awk -v id="$job" '$1==id {found=1} END {exit !found}'; then
        fail "Suite job $job ($role) is still queued or running; nothing submitted."
    fi
done < "$suite/jobs.tsv"
for item in 16-27B-s42 17-27B-s43 18-27B-s44 19-27B-s45; do
    [[ ! -e "$suite/cells/$item" ]] || fail "Cell $item already exists; partial cells require a separate recovery, not an overwrite."
done
mkdir "$resume_dir"
mkdir "$resume_dir/logs"
cp -- "$script_dir/clm_resume.sh" "$resume_dir/launcher.sh"
cp -- "$script_dir/clm_resume.py" "$resume_dir/runner.py"
cp -- "$script_dir/clm_resume_worker.sh" "$resume_dir/worker.sh"
(cd "$resume_dir" && sha256sum launcher.sh runner.py worker.sh > helper.sha256)
printf 'role\tjob_id\n' > "$resume_dir/jobs.tsv"
printf 'created\n' > "$resume_dir/state"
record_submit() {
    local role=$1 response
    shift
    printf '%q ' sbatch "$@" > "$resume_dir/${role}.command"
    printf '\n' >> "$resume_dir/${role}.command"
    printf 'submitting:%s\n' "$role" > "$resume_dir/state"
    if ! sbatch "$@" > "$resume_dir/${role}.stdout" 2> "$resume_dir/${role}.stderr"; then
        printf 'uncertain:%s\n' "$role" > "$resume_dir/state"
        cat "$resume_dir/${role}.stderr" >&2
        fail "Submission was not confirmed for $role. Do not repeat: inspect $resume_dir; preflight remains held."
    fi
    response=$(cat "$resume_dir/${role}.stdout")
    [[ "$response" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]] || {
        printf 'uncertain:%s\n' "$role" > "$resume_dir/state"
        fail "Unexpected sbatch response for $role; inspect the recorded command/stdout and queue."
    }
    submitted_id=${response%%;*}
    printf '%s\t%s\n' "$role" "$submitted_id" >> "$resume_dir/jobs.tsv"
    printf '%s\t%s\n' "$role" "$submitted_id" >> "$suite/jobs.tsv"
    printf 'recorded:%s:%s\n' "$role" "$submitted_id" > "$resume_dir/state"
    printf '%s job: %s\n' "$role" "$submitted_id"
}
common=(--parsable --export=ALL --no-requeue --chdir="$suite/source")
record_submit resume_preflight "${common[@]}" "${cpu[@]}" --hold \
    --job-name=clm-resume-check --output="$resume_dir/logs/preflight-%j.out" \
    "$resume_dir/worker.sh" preflight "$suite"
preflight_id=$submitted_id
record_submit resume_27B "${common[@]}" "${gpu[@]}" --array=16-19%1 \
    --job-name=clm-resume-27B --dependency="afterok:$preflight_id" --kill-on-invalid-dep=yes \
    --output="$resume_dir/logs/cell-%A_%a.out" "$resume_dir/worker.sh" cell "$suite"
array_id=$submitted_id
record_submit resume_report "${common[@]}" "${cpu[@]}" --job-name=clm-resume-report \
    --dependency="afterok:$preflight_id,afterany:$array_id" --kill-on-invalid-dep=yes \
    --output="$resume_dir/logs/report-%j.out" "$resume_dir/worker.sh" report "$suite"
printf 'release-requested:%s\n' "$preflight_id" > "$resume_dir/state"
scontrol release "$preflight_id" || fail "Release was not confirmed; inspect job $preflight_id and $resume_dir/state."
printf 'submitted\n' > "$resume_dir/state"
printf '\nQueued a CPU preflight, four 27B cells (48 episodes, one GPU at a time), and a final report.\n'
printf 'GPU cells start only after the original frozen suite passes preflight.\n'
printf 'Logs: %s/logs\nStatus: bash %q status %q\n' "$resume_dir" "$suite/source/scripts/clm_vllm.sh" "$suite"
