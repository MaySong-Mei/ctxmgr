#!/usr/bin/env bash
# Resource requests and log paths are supplied by scripts/clm_vllm.sh.
set -euo pipefail

mode=${1:?Expected setup, cell, or report}
suite=${2:?Expected an absolute suite path}
case "$mode" in setup|cell|report) ;; *) echo 'ERROR: Unknown worker mode.' >&2; exit 2 ;; esac
[[ "$suite" = /* && -d "$suite/source" ]] || {
    echo 'ERROR: Invalid suite directory.' >&2
    exit 2
}
[[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_JOB_NODELIST:-}" ]] || {
    echo 'ERROR: This worker must run in a Slurm batch allocation.' >&2
    exit 2
}
worker_host=$(hostname -s)
case "$worker_host" in login*) echo 'ERROR: Refusing Python on a login node.' >&2; exit 2 ;; esac
allocated_hosts=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
if ! printf '%s\n' "$allocated_hosts" | sed 's/\..*$//' | grep -Fxq "$worker_host"; then
    echo 'ERROR: The current host is outside the allocated node list.' >&2
    exit 2
fi

cd "$suite"
sha256sum --check --quiet config.sha256
cd "$suite/source"
sha256sum --check --quiet "$suite/source.sha256"
set -a
# This is the resource configuration frozen by the submitter, verified above.
source "$suite/config.env"
set +a
printf 'CLM vLLM %s job %s on %s\n' "$mode" "$SLURM_JOB_ID" "$worker_host"
exec bash "$suite/source/scripts/clm_vllm_runtime.sh" "$mode" "$suite"
