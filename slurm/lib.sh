# Helpers for running vLLM servers inside a Slurm job. Source after slurm/env.sh.

# start_server <model> <port> <gpu_util> <max_len> <log> [extra vllm args...]
# Sets SERVER_PID. Text-only: image/video inputs are disabled so the vision encoder reserves nothing.
start_server() {
    local model="$1" port="$2" util="$3" max_len="$4" log="$5"
    shift 5
    uv run vllm serve "${model}" \
        --host 127.0.0.1 --port "${port}" \
        --max-model-len "${max_len}" \
        --gpu-memory-utilization "${util}" \
        --limit-mm-per-prompt '{"image": 0, "video": 0}' \
        "$@" > "${log}" 2>&1 &
    SERVER_PID=$!
}

# wait_healthy <port> <pid> <log>: up to WAIT_HEALTHY_S (default 1 h; a cold 27B bf16 compile + CUDA
# graph capture on H200 took ~17 min); returns 1 if the server dies or never comes up.
wait_healthy() {
    local port="$1" pid="$2" log="$3"
    for _ in $(seq 1 $(( ${WAIT_HEALTHY_S:-3600} / 5 ))); do
        if curl -sf "http://127.0.0.1:${port}/health" >/dev/null; then
            grep -E "Model loading took|GPU KV cache size" "${log}"
            return 0
        fi
        kill -0 "${pid}" 2>/dev/null || break
        sleep 5
    done
    echo "vLLM server failed to start (see ${log})"
    grep -E "Error|error" "${log}" | tail -5
    return 1
}

stop_server() {
    kill "$1" 2>/dev/null
    wait "$1" 2>/dev/null
}

# smoke <port> <model> <json_out>
smoke() {
    uv run python scripts/smoke_test.py --base-url "http://127.0.0.1:$1/v1" --model "$2" --out "$3"
}

gpu_mem() {
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
}

# free_ports <n>: print n distinct free TCP ports on this node. Ports derived from the job id
# collided when two of our jobs shared a node (30088183 and 30088184 both used 28184).
free_ports() {
    python3 - "$1" <<'PY'
import socket, sys
socks = [socket.socket() for _ in range(int(sys.argv[1]))]
for s in socks:
    s.bind(("127.0.0.1", 0))
print(" ".join(str(s.getsockname()[1]) for s in socks))
PY
}

# check_model <port> <model>: fail unless the server on <port> serves <model> (guards against
# talking to another job's server).
check_model() {
    if curl -sf "http://127.0.0.1:$1/v1/models" | grep -q "\"$2\""; then return 0; fi
    echo "port $1 does not serve $2; aborting"; return 1
}
