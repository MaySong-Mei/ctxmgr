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

# No nvcc on GPU nodes (cluster CUDA modules top out at 12.9; our torch is cu13), so FlashInfer's
# JIT-compiled sampler fails. Use vLLM's PyTorch sampler; attention uses FLASH_ATTN (no JIT).
export VLLM_USE_FLASHINFER_SAMPLER=0
# Same cause on Hopper (H200): vLLM's FP8 path uses DeepGEMM, which JIT-compiles with the CUDA
# toolkit and kills the engine at startup (job 30085999). Use the precompiled CUTLASS FP8 kernels.
export VLLM_USE_DEEP_GEMM=0
# vLLM's own engine startup limit (default 600 s) is too short for a cold 27B bf16 start on H200
# (~17 min of compile + CUDA graph capture; job 30087088). Compile caches make later starts faster.
export VLLM_ENGINE_READY_TIMEOUT_S=3600
