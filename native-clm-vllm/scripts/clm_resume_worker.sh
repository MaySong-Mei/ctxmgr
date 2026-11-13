#!/usr/bin/env bash
set -euo pipefail
mode=${1:?Expected preflight, cell, or report}
suite=${2:?Expected existing suite}
case "$mode" in preflight|cell|report) ;; *) exit 2 ;; esac
[[ "$suite" = /* && -d "$suite/source" ]] || exit 2
resume_dir="$suite/continuation-27b-v1"
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
    echo 'ERROR: Run this worker only through Slurm.' >&2; exit 2;
}
worker_host=$(hostname -s)
case "$worker_host" in login*) echo 'ERROR: Refusing Python on a login node.' >&2; exit 2 ;; esac
allocated_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
printf '%s\n' "$allocated_hosts" | sed 's/\..*$//' | grep -Fxq "$worker_host" || {
    echo 'ERROR: Host is outside this allocation.' >&2; exit 2;
}
(cd "$resume_dir" && sha256sum --check --quiet helper.sha256)
(cd "$suite" && sha256sum --check --quiet config.sha256)
(cd "$suite/source" && sha256sum --check --quiet "$suite/source.sha256")
set -a
source "$suite/config.env"
set +a
export CLM_RESUME_DIR="$resume_dir" CLM_RESUME_RUNNER="$resume_dir/runner.py"
export PYTHONPATH="$suite/source/src${PYTHONPATH:+:$PYTHONPATH}"
client_python="$suite/source/envs/clm/.venv/bin/python"
[[ -x "$client_python" && -x "$suite/source/.venv/bin/python" ]] || {
    echo 'ERROR: Original prepared Python environments are missing; no experiment was started.' >&2; exit 2;
}
if [[ "$mode" == preflight ]]; then
    # Preflight regenerates/validates the report under the original report lock.
    exec 8> "$suite/report.lock"
    flock -w 600 8
    exec "$client_python" "$CLM_RESUME_RUNNER" preflight --suite "$suite"
fi
[[ -s "$resume_dir/preflight.json" ]] || { echo 'ERROR: Missing successful preflight.' >&2; exit 2; }
(cd "$resume_dir" && sha256sum --check --quiet generated.sha256)
if [[ "$mode" == cell ]]; then
    [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^1[6-9]$ ]] || { echo 'ERROR: Only indices 16–19 are allowed.' >&2; exit 2; }
    exec 7> "$resume_dir/cell-${SLURM_ARRAY_TASK_ID}.lock"
    flock -n 7 || { echo 'ERROR: This continuation cell already has a worker.' >&2; exit 2; }
fi
exec bash "$resume_dir/runtime.sh" "$mode" "$suite"
