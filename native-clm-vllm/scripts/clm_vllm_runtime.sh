#!/usr/bin/env bash
# Compute-node implementation; use clm_vllm.sh from the login node.
set -euo pipefail
mode=${1:?Expected setup, cell, or report}
suite=${2:?Expected suite directory}
source "$suite/config.env"
source_dir="$suite/source"
cd "$source_dir"
export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="$source_dir/src${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_DISABLE_TELEMETRY=1 VLLM_NO_USAGE_STATS=1
export VLLM_USE_FLASHINFER_SAMPLER=0 VLLM_USE_DEEP_GEMM=0
export VLLM_ENGINE_READY_TIMEOUT_S="$CLM_SERVER_READY_TIMEOUT"
export LITELLM_LOCAL_MODEL_COST_MAP=True
export OPENAI_API_KEY=EMPTY
unset OPENAI_BASE_URL OPENAI_API_BASE
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="$NO_PROXY"
mkdir -p "$HF_HOME" "$UV_CACHE_DIR" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR"
client_python="$source_dir/envs/clm/.venv/bin/python"
export CLM_SERVE_PYTHON="$source_dir/.venv/bin/python"

case "$mode" in
    setup)
        command -v uv >/dev/null || {
            echo 'ERROR: uv is required. Install uv in ~/.local/bin before submission.' >&2
            exit 1
        }
        source slurm/clm_env.sh
        [[ -f "$CLM_IMAGE" ]] || {
            echo "ERROR: Existing ContextBench container not found: $CLM_IMAGE" >&2
            echo 'Set CLM_IMAGE in configs/clm-vllm.env to the container from your previous CLM run.' >&2
            exit 1
        }
        # Separate environments prevent serving and native-client dependency conflicts.
        uv sync --locked --python 3.12 --project "$source_dir" --group serve --no-dev
        uv sync --locked --python 3.12 --project "$source_dir/envs/clm"
        "$client_python" -m ctxmgr.harness.clm_upstream check
        "$client_python" -m unittest discover -s tests -p 'test_vllm_*.py'
        singularity exec --containall --no-home "$CLM_IMAGE" /bin/bash -lc \
            'test -x /usr/bin/python3 && command -v tmux && /opt/harbor-server/bin/python3 -c "import fastapi, uvicorn"'
        for size in 0.8B 2B 4B 9B 27B; do
            "$CLM_SERVE_PYTHON" -m ctxmgr.harness.clm_vllm_models prepare \
                --size "$size" --out "$suite/models/Qwen3.5-${size}-BF16"
        done
        "$client_python" -m ctxmgr.harness.clm_vllm_suite setup --suite "$suite"
        ;;
    cell)
        source slurm/clm_env.sh
        "$client_python" -m ctxmgr.harness.clm_vllm_suite cell \
            --suite "$suite" --index "${SLURM_ARRAY_TASK_ID:?Missing array index}" </dev/null
        ;;
    report)
        # Serialize manual snapshots and the final dependent report.
        exec 8>"$suite/report.lock"
        flock -w 600 8
        if [[ ! -x "$client_python" || ! -f "$suite/ready.json" ]]; then
            mkdir -p "$suite/report"
            cat > "$suite/report/report-status.json" <<'JSON'
{"schema_version":1,"status":"preparation_not_ready","planned":240,"graded":0,"unrun":240,"error":"Preparation did not finish; inspect the setup job and its log."}
JSON
            echo 'Preparation is not ready. No GPU grades are expected; inspect the setup log.' >&2
            exit 1
        fi
        "$client_python" -m ctxmgr.harness.clm_vllm_suite report --suite "$suite"
        ;;
    *) echo "ERROR: Unknown runtime mode: $mode" >&2; exit 2 ;;
esac
