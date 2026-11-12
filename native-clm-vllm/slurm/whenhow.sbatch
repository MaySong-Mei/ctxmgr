#!/usr/bin/env bash
#SBATCH --partition=gpu_h200
#SBATCH --gpus=h200:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=160G
#SBATCH --time=1-00:00:00
# One when-versus-how job: one actor size, one manager kind, one task seed.
# Submitted by scripts/whenhow.sh with: ROOT BASE_SUITE ACTOR MANAGER SEED.
set -euo pipefail
root=${1:?root} base=${2:?base suite} actor=${3:?actor} manager=${4:?manager} seed=${5:?seed}

[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: run inside a Slurm allocation.' >&2; exit 2; }
host=$(hostname -s)
case "$host" in login*) echo 'ERROR: refusing Python on a login node.' >&2; exit 2 ;; esac

cd "$root/source" && sha256sum --check --quiet "$root/source.sha256"
set -a
source "$base/config.env"
set +a
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$root/source/src"
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_ENGINE_READY_TIMEOUT_S="$CLM_SERVER_READY_TIMEOUT"
export LITELLM_LOCAL_MODEL_COST_MAP=True
unset OPENAI_BASE_URL OPENAI_API_BASE
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}" no_proxy="127.0.0.1,localhost"
export CLM_SERVE_PYTHON="$base/source/.venv/bin/python"
client_python="$base/source/envs/clm/.venv/bin/python"
source "$root/source/slurm/clm_env.sh"
printf 'whenhow job %s on %s: actor=%s manager=%s seed=%s\n' "$SLURM_JOB_ID" "$host" "$actor" "$manager" "$seed"
exec "$client_python" -m ctxmgr.harness.whenhow_suite job \
    --base-suite "$base" --root "$root" --actor "$actor" --manager "$manager" --seed "$seed"
