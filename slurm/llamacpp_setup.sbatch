#!/bin/bash
# One-time setup for the llama.cpp GGUF pilot on a CPU node: prebuilt llama.cpp CUDA binaries
# (no nvcc on GPU nodes, so no local build), the base venv for scripts/smoke_test.py, and GGUFs.
# Usage (from repo root): sbatch slurm/llamacpp_setup.sbatch
#SBATCH --job-name=ctxmgr-llamacpp-setup
#SBATCH --partition=education
#SBATCH --cpus-per-task=4
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

# Pinned build b11306 (2026-09-30); CUDA 12.8 runtime is bundled, the 580 driver supports it.
if [[ ! -x "${LLAMA_BIN}/llama-server" ]]; then
    tmp="$(mktemp -d)"
    base="https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}"
    curl -fsSL -o "${tmp}/bin.tar.gz" "${base}/llama-${LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    curl -fsSL -o "${tmp}/cudart.tar.gz" "${base}/cudart-llama-${LLAMA_BUILD}-bin-ubuntu-cuda-12.8-x64.tar.gz"
    mkdir -p "${tmp}/x"
    tar -xzf "${tmp}/bin.tar.gz" -C "${tmp}/x"
    tar -xzf "${tmp}/cudart.tar.gz" -C "${tmp}/x"
    # Flatten: llama-server and every shared library it needs in one directory.
    mkdir -p "${LLAMA_BIN}"
    find "${tmp}/x" -type f \( -name 'llama-*' -o -name '*.so*' \) -exec cp -a {} "${LLAMA_BIN}/" \;
    find "${tmp}/x" -type l -name '*.so*' -exec cp -a {} "${LLAMA_BIN}/" \;
    rm -rf "${tmp}"
fi
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
