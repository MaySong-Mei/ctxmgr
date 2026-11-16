# Sourced by the BrowseComp-Plus sbatch scripts: ROOT and BASE must be set.
[[ -n "${SLURM_JOB_ID:-}" ]] || { echo 'ERROR: run inside a Slurm allocation.' >&2; exit 2; }
case "$(hostname -s)" in login*) echo 'ERROR: refusing Python on a login node.' >&2; exit 2 ;; esac
cd "$ROOT/source" && sha256sum --check --quiet "$ROOT/source.sha256"
set -a
source "$BASE/config.env"
set +a
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$ROOT/source/src"
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_ENGINE_READY_TIMEOUT_S="$CLM_SERVER_READY_TIMEOUT"
export LITELLM_LOCAL_MODEL_COST_MAP=True
unset OPENAI_BASE_URL OPENAI_API_BASE
export NO_PROXY="127.0.0.1,localhost" no_proxy="127.0.0.1,localhost"
export CLM_SERVE_PYTHON="$BASE/source/.venv/bin/python"
export BCP_PYTHON="$ROOT/source/envs/bcp/.venv/bin/python"
client_python="$BASE/source/envs/clm/.venv/bin/python"
DATA="$ROOT/data"
source "$ROOT/source/slurm/clm_env.sh"
