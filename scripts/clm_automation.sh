#!/usr/bin/env bash
# Login-node launcher. Python executes only in the allocated controller job.
set -euo pipefail

die() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
usage() {
    cat <<'HELP'
Usage:
  bash scripts/clm_automation.sh check RECOVERY_DIR [options]
  bash scripts/clm_automation.sh start RECOVERY_DIR [options]
  bash scripts/clm_automation.sh status RECOVERY_DIR
  bash scripts/clm_automation.sh stop RECOVERY_DIR

Options for check/start:
  --partition NAME       Controller CPU partition (default: day)
  --account NAME         Slurm account (default: cpsc4880)
  --qos NAME             Slurm QOS (default: normal)
  --time LIMIT           Controller allocation limit (default: 1-00:00:00)
  --publish              Publish the validated complete report to GitHub
  --repository OWNER/REPO  Target repository (default: MaySong-Mei/ctxmgr)
  --discussion NUMBER    Target Discussion (default: 6)

check uses sbatch --test-only and does not create a job.
start waits for recorded jobs, fills unsubmitted cells, and validates the report.
start resumes a stopped controller; it never retries a recorded failed GPU cell.
Without --publish the controller writes a local GitHub report preview only.
stop requests a graceful controller stop; submitted experiment jobs keep running.
HELP
}

action=${1:-help}
case "$action" in help|-h|--help) usage; exit 0;; check|start|status|stop) ;; *) die 'Unknown action.';; esac
[[ $# -ge 2 ]] || die 'Missing recovery directory.'
target=$(realpath -- "$2")
shift 2
[[ -f "$target/runtime.env" && -f "$target/plan-ready" && -f "$target/tools/recover.sh" ]] || die 'Expected an existing, prepared recovery directory.'
source_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../tools/clm_automation" && pwd)
auto_dir="$target/automation"
mkdir -p -- "$auto_dir/logs"

if [[ "$action" == stop ]]; then
    [[ $# -eq 0 ]] || die 'stop does not take options.'
    touch "$auto_dir/STOP"
    echo 'Stop requested. The controller will stop at its next checkpoint; submitted jobs keep running.'
    exit 0
fi
if [[ "$action" == status ]]; then
    [[ $# -eq 0 ]] || die 'status does not take options.'
    if [[ -s "$auto_dir/controller-jobs.tsv" ]]; then
        cat "$auto_dir/controller-jobs.tsv"
        ids=$(awk -F '\t' 'NR>1 {print $1}' "$auto_dir/controller-jobs.tsv" | paste -sd, -)
        [[ -z "$ids" ]] || sacct -X -j "$ids" --format=JobID,State,ExitCode,Elapsed
    else
        echo 'No automation controller has been recorded.'
    fi
    for item in controller-state.json submission-intent.json launch-intent.txt; do
        if [[ -f "$auto_dir/$item" ]]; then printf '\n%s\n' "$item"; cat "$auto_dir/$item"; fi
    done
    bash "$target/tools/recover.sh" status "$target"
    exit 0
fi

partition=day
account=cpsc4880
qos=normal
time_limit=1-00:00:00
repository=MaySong-Mei/ctxmgr
discussion=6
publish=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --publish) publish=true; shift;;
        --partition|--account|--qos|--time|--repository|--discussion)
            [[ $# -ge 2 && -n "$2" ]] || die "Missing value for $1."
            case "$1" in
                --partition) partition=$2;; --account) account=$2;; --qos) qos=$2;;
                --time) time_limit=$2;; --repository) repository=$2;; --discussion) discussion=$2;;
            esac
            shift 2;;
        *) die "Unknown option: $1";;
    esac
done
[[ "$discussion" =~ ^[1-9][0-9]*$ ]] || die 'Discussion number must be positive.'
[[ "$repository" =~ ^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$ ]] || die 'Expected repository OWNER/REPO.'
# A separate CPU partition avoids occupying the single education GPU job slot.
[[ "$partition" != education && "$partition" != education_gpu ]] || die 'Use a separate CPU partition for the controller, such as day.'
if "$publish"; then command -v gh >/dev/null || die 'GitHub CLI gh is required for --publish.'; fi
for name in ${!SBATCH_@}; do unset "$name"; done
worker_args=("$target" --repository "$repository" --discussion "$discussion")
if "$publish"; then worker_args+=(--publish); fi
scheduler_args=(--account="$account" --qos="$qos" --partition="$partition" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=2G --time="$time_limit" --export=ALL --job-name=ctxmgr-clm-auto)
if [[ "$action" == check ]]; then
    sbatch --test-only "${scheduler_args[@]}" "$source_dir/worker.sbatch" "${worker_args[@]}"
    exit 0
fi

exec 8>"$auto_dir/launch.lock"
flock -w 10 8 || die 'Another controller launch is active.'
if [[ -f "$auto_dir/controller-jobs.tsv" ]]; then
    while IFS=$'\t' read -r id snapshot; do
        [[ "$id" == job_id ]] && continue
        [[ "$id" =~ ^[0-9]+$ ]] || die 'Invalid controller ledger.'
        raw=$(timeout 30 sacct -X -n -j "$id" --format=JobIDRaw,State%40 -P) || die 'Cannot verify an earlier controller.'
        state=$(awk -F '|' -v wanted="$id" '$1==wanted {gsub(/^ +| +$/, "", $2); print $2; exit}' <<< "$raw")
        case "${state%% *}" in
            COMPLETED|FAILED|CANCELLED*|TIMEOUT|OUT_OF_MEMORY|NODE_FAIL|BOOT_FAIL|DEADLINE|PREEMPTED|REVOKED) ;;
            PENDING|RUNNING|CONFIGURING|COMPLETING|SUSPENDED|RESIZING|REQUEUED|REQUEUE_FED|REQUEUE_HOLD)
                printf 'Controller %s is %s. No duplicate submitted.\n' "$id" "$state"
                exit 0;;
            *) die "Controller $id has unknown state ($state). Nothing submitted.";;
        esac
    done < "$auto_dir/controller-jobs.tsv"
fi
[[ ! -e "$auto_dir/launch-intent.txt" ]] || die 'An earlier controller submission is unresolved. Inspect launch-intent.txt and squeue before proceeding.'
# The allocated controller reconciles experiment/report intent against frozen ledgers.

code_dir=$(mktemp -d "$auto_dir/code-XXXXXXXX")
for item in controller.py github_report.py worker.sbatch; do cp -- "$source_dir/$item" "$code_dir/$item"; done
(cd "$code_dir" && sha256sum controller.py github_report.py worker.sbatch > SHA256SUMS)
[[ -f "$auto_dir/controller-jobs.tsv" ]] || printf 'job_id\tsource_snapshot\n' > "$auto_dir/controller-jobs.tsv"
printf 'started_utc=%s\nsource_snapshot=%s\n' "$(date -u +%FT%TZ)" "$code_dir" > "$auto_dir/launch-intent.txt"
rm -f -- "$auto_dir/STOP"
if ! response=$(timeout 60 sbatch --parsable --hold "${scheduler_args[@]}" --chdir="$code_dir" --output="$auto_dir/logs/%x-%j.out" "$code_dir/worker.sbatch" "${worker_args[@]}" 2>"$auto_dir/launch-error.txt"); then
    cat "$auto_dir/launch-error.txt" >&2
    die 'Submission was not confirmed. Inspect the queue and launch-intent.txt; do not submit again blindly.'
fi
id=${response%%;*}
[[ "$id" =~ ^[0-9]+$ ]] || die 'Ambiguous scheduler response. Inspect the queue and launch-intent.txt.'
printf '%s\t%s\n' "$id" "$code_dir" >> "$auto_dir/controller-jobs.tsv"
rm -- "$auto_dir/launch-intent.txt"
if ! scontrol release "$id"; then die "Controller $id was recorded but release failed. Inspect and release that job; do not resubmit."; fi
printf 'Controller job: %s\nLog: %s/logs/ctxmgr-clm-auto-%s.out\n' "$id" "$auto_dir" "$id"
if "$publish"; then
    printf 'Complete reports will be published to https://github.com/%s/discussions/%s\n' "$repository" "$discussion"
else
    echo 'GitHub publication is disabled. A local report preview will be generated.'
fi
