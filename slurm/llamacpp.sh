# Helpers for running llama.cpp servers inside a Slurm job. Source after slurm/env.sh.
# Same interface as slurm/lib.sh (start/wait/stop), so jobs can swap serving stacks.

export LLAMA_BUILD="b11306"
export LLAMA_BIN="${PROJECT_DIR}/llama.cpp/${LLAMA_BUILD}"
export GGUF_DIR="${PROJECT_DIR}/gguf"
# Built against these modules (slurm/llamacpp_setup.sbatch): nvcc at build time, and libcudart,
# libcublas and GCC 13's libstdc++ at run time. Lmod reads unset variables, so relax -u around it.
set +u
type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module load GCC/13.3.0 CUDA/12.8.0 CMake/3.31.8-GCCcore-13.3.0
set -u
export LD_LIBRARY_PATH="${LLAMA_BIN}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

# start_llama <gguf> <port> <ctx> <log> [extra llama-server args...]
# Sets SERVER_PID. One slot (-np 1): an episode sends one request at a time. All layers on GPU.
start_llama() {
    local gguf="$1" port="$2" ctx="$3" log="$4"
    shift 4
    "${LLAMA_BIN}/llama-server" -m "${gguf}" \
        --host 127.0.0.1 --port "${port}" \
        --ctx-size "${ctx}" --parallel 1 --n-gpu-layers 999 \
        --jinja --alias "$(basename "${gguf}" .gguf)" \
        "$@" > "${log}" 2>&1 &
    SERVER_PID=$!
}

# wait_llama <port> <pid> <log>: up to 10 min; prints the CUDA buffer sizes llama.cpp reports.
wait_llama() {
    local port="$1" pid="$2" log="$3"
    for _ in $(seq 1 120); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
            grep -E "CUDA0 (model|KV|RS|compute) buffer size|ggml_cuda_init|compute capability" "${log}"
            return 0
        fi
        kill -0 "${pid}" 2>/dev/null || break
        sleep 5
    done
    echo "llama-server failed to start (see ${log})"
    grep -iE "error|failed|out of memory" "${log}" | tail -5
    return 1
}
