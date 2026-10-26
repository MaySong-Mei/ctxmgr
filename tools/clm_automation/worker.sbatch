#!/bin/bash
# Submit this controller on a CPU partition separate from recovery GPU jobs.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=1-00:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Run this script through sbatch}"
case "$(hostname -s)" in
    login*) echo 'ERROR: The controller must run on an allocated compute node.' >&2; exit 2;;
esac
recovery_dir=$(realpath "${1:?Missing recovery directory}")
shift
worker_dir="$PWD"
[[ -f "$worker_dir/controller.py" && -f "$worker_dir/github_report.py" ]] || {
    echo 'ERROR: Submit with --chdir pointing to the frozen automation code directory.' >&2
    exit 2
}
sha256sum -c "$worker_dir/SHA256SUMS"
source "$recovery_dir/runtime.env"
: "${CLM_PYTHON:?Missing existing recovery Python environment}"
export PYTHONDONTWRITEBYTECODE=1
printf 'CLM automation controller %s on %s\n' "$SLURM_JOB_ID" "$(hostname)"
exec "$CLM_PYTHON" "$worker_dir/controller.py" "$recovery_dir" "$@"
