#!/bin/bash
# One-time setup for the llama.cpp GGUF pilot on a CPU node: builds llama.cpp from source with the
# cluster CUDA module, syncs the base venv for scripts/smoke_test.py, and downloads GGUFs.
# Usage (from repo root): sbatch slurm/llamacpp_setup.sbatch
#SBATCH --job-name=ctxmgr-llamacpp-setup
#SBATCH --partition=education
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
source slurm/env.sh
source slurm/llamacpp.sh

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
uv --version
uv sync

# Pinned source tag b11306 (2026-09-30). The prebuilt Ubuntu binaries need glibc 2.38 and Bouchet
# (RHEL 9.6) has 2.34, so build here; nvcc comes from the CUDA module (llamacpp.sh loads it).
if [[ ! -x "${LLAMA_BIN}/llama-server" ]]; then
    src="${SCRATCH_DIR}/llama.cpp-${LLAMA_BUILD}"
    rm -rf "${src}"
    git clone -q --depth 1 --branch "${LLAMA_BUILD}" https://github.com/ggml-org/llama.cpp "${src}"
    # A40 only (sm_86); no -march=native because the build node's CPU may differ from the GPU node's.
    cmake -S "${src}" -B "${src}/build" -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES=86 -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
    cmake --build "${src}/build" --target llama-server -j "${SLURM_CPUS_PER_TASK:-8}"
    mkdir -p "${LLAMA_BIN}"
    cp -a "${src}"/build/bin/* "${LLAMA_BIN}/"
fi
"${LLAMA_BIN}/llama-server" --version
ls "${LLAMA_BIN}" | head -40

# Manager candidates (Q6_K first, Q5_K_M fallback) and Q8_0 actors. Text only: no mmproj.
# One file per call over plain HTTP: hf-xet fetching two 20 GB files at once peaked above 16 GB RSS
# and was OOM-killed (job 29901625).
export HF_HUB_DISABLE_XET=1
fetch() { uvx --from huggingface_hub hf download "$1" "$2" --local-dir "${GGUF_DIR}/$3"; }
fetch unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q6_K.gguf Qwen3.5-27B
fetch unsloth/Qwen3.5-27B-GGUF Qwen3.5-27B-Q5_K_M.gguf Qwen3.5-27B
for size in 9B 4B; do
    fetch "unsloth/Qwen3.5-${size}-GGUF" "Qwen3.5-${size}-Q8_0.gguf" "Qwen3.5-${size}"
done
du -sh "${GGUF_DIR}"/* || true
df -h "${PROJECT_DIR}" || true
