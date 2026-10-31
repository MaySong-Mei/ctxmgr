# Source after May's env.sh, from the repository root on an allocated node.
export LITELLM_LOCAL_MODEL_COST_MAP=True
export TIKTOKEN_CACHE_DIR="${PROJECT_DIR}/tiktoken_cache"
export APPTAINER_CACHEDIR="${SCRATCH_DIR}/apptainer_cache"
export SINGULARITY_CACHEDIR="${APPTAINER_CACHEDIR}"
export TMPDIR="${SLURM_TMPDIR:-${SCRATCH_DIR}/clm-tmp-${SLURM_JOB_ID:-local}}"
export APPTAINER_TMPDIR="${TMPDIR}"
export SINGULARITY_TMPDIR="${TMPDIR}"
export CLM_IMAGE="${CLM_IMAGE:-${PROJECT_DIR}/containers/contextbench.sif}"
mkdir -p "${TMPDIR}" "${TIKTOKEN_CACHE_DIR}" "${APPTAINER_CACHEDIR}"

# Harbor 0.16.1 invokes the executable name `singularity`, including on Apptainer sites.
# Supply CONTAINER_MODULE only when the cluster requires a module (use its exact name).
if [[ -n "${CONTAINER_MODULE:-}" ]]; then
    module load "${CONTAINER_MODULE}"
fi
if ! command -v singularity >/dev/null && command -v apptainer >/dev/null; then
    mkdir -p "${TMPDIR}/bin"
    cat > "${TMPDIR}/bin/singularity" <<'SH'
#!/bin/bash
exec apptainer "$@"
SH
    chmod +x "${TMPDIR}/bin/singularity"
    export PATH="${TMPDIR}/bin:${PATH}"
fi
if ! command -v singularity >/dev/null; then
    echo 'Container runtime missing. Check `module spider Apptainer` / `module spider Singularity`.' >&2
    echo 'Resubmit with --export=ALL,CONTAINER_MODULE=<the available module name>.' >&2
    return 1
fi
singularity --version
