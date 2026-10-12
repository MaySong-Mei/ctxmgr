# Shared environment for Bouchet jobs. Source from the repo root: `source slurm/env.sh`.
# Course accounts are named <course>_<netid>; storage lives under the course group.

COURSE="${USER%%_*}"
export PROJECT_DIR="/nfs/roberts/project/${COURSE}/${USER}"
export SCRATCH_DIR="/nfs/roberts/scratch/${COURSE}/${USER}"

# Model weights are large and reused across runs -> backed-up project dir, not home or scratch.
export HF_HOME="${PROJECT_DIR}/hf_cache"
# Package/build caches are disposable -> scratch (purged after 30 days).
export UV_CACHE_DIR="${SCRATCH_DIR}/uv_cache"
export VLLM_CACHE_ROOT="${SCRATCH_DIR}/vllm_cache"
export TRITON_CACHE_DIR="${SCRATCH_DIR}/triton_cache"

export PATH="${HOME}/.local/bin:${PATH}"
export HF_HUB_DISABLE_TELEMETRY=1
export VLLM_NO_USAGE_STATS=1

mkdir -p "${HF_HOME}" "${UV_CACHE_DIR}" "${VLLM_CACHE_ROOT}" "${TRITON_CACHE_DIR}" logs
