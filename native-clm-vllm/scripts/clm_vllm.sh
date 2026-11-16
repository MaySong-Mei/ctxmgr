#!/usr/bin/env bash
# Login-node entry point. All Python execution is delegated to Slurm workers.
set -euo pipefail

package_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
model_sizes=(0.8B 2B 4B 9B 27B)
model_keys=(08B 2B 4B 9B 27B)
gpu_keys=()
for key in "${model_keys[@]}"; do
    gpu_keys+=("CLM_GPU_${key}_PARTITION" "CLM_GPU_${key}_TYPE" "CLM_GPU_${key}_MEM")
done
config_keys=(CLM_ACCOUNT CLM_QOS CLM_CPU_PARTITION "${gpu_keys[@]}"
    CLM_PARALLEL PROJECT_DIR SCRATCH_DIR CLM_RUNS_DIR HF_HOME UV_CACHE_DIR
    VLLM_CACHE_ROOT TRITON_CACHE_DIR CLM_IMAGE CONTAINER_MODULE
    CLM_SETUP_CPUS CLM_SETUP_MEM CLM_SETUP_TIME CLM_GPU_CPUS
    CLM_GPU_TIME CLM_REPORT_CPUS CLM_REPORT_MEM CLM_REPORT_TIME
    CLM_MAX_MODEL_LEN CLM_GPU_MEMORY_UTILIZATION CLM_SERVER_READY_TIMEOUT)

usage() {
    cat <<'USAGE'
Usage:
  bash scripts/clm_vllm.sh check [--config FILE]
  bash scripts/clm_vllm.sh submit [--config FILE]
  bash scripts/clm_vllm.sh status SUITE
  bash scripts/clm_vllm.sh report SUITE

check   Tests resource requests with sbatch --test-only; submits no jobs.
submit  Creates a NEW suite: 5 models x 4 seeds x 6 levels x 2 conditions.
        Submits setup, five four-task model GPU arrays, and a dependent report.
status  Displays recorded jobs, scheduler state, and the latest report status.
report  Submits a CPU-only report snapshot; never reruns model episodes.

Each submit invocation requests a separate full 240-episode experiment.
Keep the printed suite path. Do not repeat submit to check an existing run.
USAGE
}

fail() { printf 'ERROR: %s\n' "$*" >&2; exit 2; }
require_command() { command -v "$1" >/dev/null 2>&1 || fail "Required command not found: $1"; }

validate_config() {
    local key value
    for key in "${config_keys[@]}"; do
        [[ -v "$key" ]] || fail "Missing configuration key: $key"
        value=${!key}
        [[ "$value" != *$'\n'* && "$value" != *$'\r'* ]] || fail "Newlines are not allowed in $key"
        if [[ "$key" != CONTAINER_MODULE ]]; then
            [[ -n "$value" ]] || fail "Empty configuration key: $key"
        fi
    done
    for key in CLM_ACCOUNT CLM_QOS CLM_CPU_PARTITION; do
        [[ "${!key}" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail "Invalid resource name: $key"
    done
    for key in "${gpu_keys[@]}"; do
        [[ "$key" != *_MEM ]] || continue
        [[ "${!key}" =~ ^[a-zA-Z0-9_.-]+$ ]] || fail "Invalid resource name: $key"
    done
    for key in PROJECT_DIR SCRATCH_DIR CLM_RUNS_DIR HF_HOME UV_CACHE_DIR VLLM_CACHE_ROOT TRITON_CACHE_DIR CLM_IMAGE; do
        [[ "${!key}" = /* ]] || fail "An absolute path is required: $key"
    done
    for key in CLM_SETUP_CPUS CLM_GPU_CPUS CLM_REPORT_CPUS CLM_SERVER_READY_TIMEOUT; do
        [[ "${!key}" =~ ^[1-9][0-9]*$ ]] || fail "A positive integer is required: $key"
    done
    [[ "$CLM_PARALLEL" =~ ^[1-4]$ ]] || fail 'CLM_PARALLEL must be between 1 and 4 per model.'
    [[ "$CLM_MAX_MODEL_LEN" = 81920 ]] || fail 'This release requires CLM_MAX_MODEL_LEN=81920.'
    [[ "$CLM_GPU_MEMORY_UTILIZATION" =~ ^0\.[0-9]+$ ]] || fail 'GPU memory utilization must be a decimal between 0 and 1.'
    for key in CLM_SETUP_MEM CLM_REPORT_MEM "${gpu_keys[@]}"; do
        [[ "$key" = *_MEM ]] || continue
        [[ "${!key}" =~ ^[1-9][0-9]*[KMGT]?$ ]] || fail "Invalid memory value: $key"
    done
    for key in CLM_SETUP_TIME CLM_GPU_TIME CLM_REPORT_TIME; do
        [[ "${!key}" =~ ^([0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || fail "Invalid time limit: $key"
    done
}

load_config() {
    local override=''
    if (( $# )); then
        [[ $# -eq 2 && "$1" = --config ]] || fail 'Expected --config FILE.'
        override=$2
        [[ -f "$override" ]] || fail "Configuration file does not exist: $override"
    fi
    # These files are user-controlled shell resource settings, not downloaded data.
    source "$package_root/configs/clm-vllm.env"
    [[ -z "$override" ]] || source "$override"
    validate_config
}

resource_args() {
    local mode=$1 size=${2:-} key partition type memory
    args=(--account="$CLM_ACCOUNT" --qos="$CLM_QOS" --nodes=1 --ntasks=1)
    case "$mode" in
        setup)
            args+=(--partition="$CLM_CPU_PARTITION" --cpus-per-task="$CLM_SETUP_CPUS"
                --mem="$CLM_SETUP_MEM" --time="$CLM_SETUP_TIME") ;;
        cell)
            case "$size" in 0.8B) key=08B ;; 2B|4B|9B|27B) key=$size ;; *) fail "Invalid cell size: $size" ;; esac
            partition="CLM_GPU_${key}_PARTITION"
            type="CLM_GPU_${key}_TYPE"
            memory="CLM_GPU_${key}_MEM"
            args+=(--partition="${!partition}" --gpus-per-node="${!type}:1"
                --cpus-per-task="$CLM_GPU_CPUS" --mem="${!memory}" --time="$CLM_GPU_TIME") ;;
        report)
            args+=(--partition="$CLM_CPU_PARTITION" --cpus-per-task="$CLM_REPORT_CPUS"
                --mem="$CLM_REPORT_MEM" --time="$CLM_REPORT_TIME") ;;
    esac
}

freeze_source() {
    local item key
    local items=()
    for item in src envs configs scripts slurm docs tests pyproject.toml uv.lock README.md AGENTS.md VALIDATION.md RELEASE.json .gitignore LICENSE NOTICE; do
        [[ ! -e "$package_root/$item" ]] || items+=("$item")
    done
    mkdir -p "$suite/source" "$suite/logs" "$suite/submission"
    tar -C "$package_root" --exclude='.venv' --exclude='__pycache__' --exclude='*.pyc' \
        --exclude='.pytest_cache' --exclude='.ruff_cache' -cf - "${items[@]}" |
        tar -C "$suite/source" -xf -
    if [[ -n "$(find "$suite/source" -type l -print -quit)" ]]; then
        fail 'The source package contains symbolic links; no jobs were submitted.'
    fi
    {
        printf '# Frozen resource configuration. Do not edit after submission.\n'
        for key in "${config_keys[@]}"; do
            printf 'export %s=%q\n' "$key" "${!key}"
        done
    } > "$suite/config.env"
    (cd "$suite/source" && find . -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum) > "$suite/source.sha256"
    (cd "$suite" && sha256sum config.env) > "$suite/config.sha256"
    printf 'role\tjob_id\n' > "$suite/jobs.tsv"
    printf 'index\tsize\tseed\n' > "$suite/cells.tsv"
    local index=0 size seed
    for size in "${model_sizes[@]}"; do
        for seed in 42 43 44 45; do
            printf '%s\t%s\t%s\n' "$index" "$size" "$seed" >> "$suite/cells.tsv"
            index=$((index + 1))
        done
    done
    printf 'created\n' > "$suite/submission/state"
}

record_submit() {
    local role=$1
    shift
    local stem="$suite/submission/${role}-$(date -u +%Y%m%dT%H%M%SZ)-$$"
    local response
    printf '%q ' sbatch "$@" > "$stem.command"
    printf '\n' >> "$stem.command"
    printf 'submitting:%s\n' "$role" > "$suite/submission/state"
    if ! sbatch "$@" > "$stem.stdout" 2> "$stem.stderr"; then
        cat "$stem.stderr" >&2
        printf 'uncertain:%s\n' "$role" > "$suite/submission/state"
        fail "Submission was not confirmed for $role. Inspect squeue and $suite/submission before retrying; any setup hold is retained."
    fi
    response=$(cat "$stem.stdout")
    submitted_id=${response%%;*}
    [[ "$response" =~ ^[0-9]+(\;[a-zA-Z0-9_.-]+)?$ ]] || {
        printf 'uncertain:%s\n' "$role" > "$suite/submission/state"
        fail "Unexpected scheduler response for $role. Inspect squeue and $stem.stdout before retrying; any setup hold is retained."
    }
    printf '%s\t%s\n' "$role" "$submitted_id" >> "$suite/jobs.tsv"
    printf 'recorded:%s:%s\n' "$role" "$submitted_id" > "$suite/submission/state"
    printf '%s job: %s\n' "$role" "$submitted_id"
}

load_suite() {
    [[ $# -eq 1 ]] || fail 'Expected exactly one suite directory.'
    [[ -d "$1/source" && -f "$1/jobs.tsv" && -f "$1/config.env" ]] || fail 'Invalid suite directory.'
    suite=$(cd -- "$1" && pwd -P)
    (cd "$suite" && sha256sum --check --quiet config.sha256)
    source "$suite/config.env"
    validate_config
}

command_name=${1:-help}
(( $# == 0 )) || shift
case "$command_name" in
    help|-h|--help) usage ;;
    check)
        load_config "$@"
        require_command sbatch
        printf 'Resource test only; no jobs will be submitted.\n'
        for mode in setup report; do
            resource_args "$mode"
            printf 'Checking %s allocation...\n' "$mode"
            sbatch --test-only "${args[@]}" --wrap='true'
        done
        for size in "${model_sizes[@]}"; do
            resource_args cell "$size"
            printf 'Checking %s GPU allocation...\n' "$size"
            sbatch --test-only "${args[@]}" --wrap='true'
        done
        ;;
    submit)
        load_config "$@"
        for command in sbatch scontrol sha256sum tar flock; do require_command "$command"; done
        mkdir -p "$CLM_RUNS_DIR"
        suite=$(mktemp -d "$CLM_RUNS_DIR/clm-vllm-$(date -u +%Y%m%dT%H%M%SZ)-XXXXXX")
        printf 'Suite: %s\n' "$suite"
        exec 9> "$suite/submit.lock"
        flock -w 10 9
        freeze_source
        worker="$suite/source/slurm/clm_vllm.sbatch"
        common=(--parsable --export=ALL --no-requeue --chdir="$suite/source")

        resource_args setup
        record_submit setup "${common[@]}" "${args[@]}" --hold --job-name=ctxmgr-clm-setup \
            --output="$suite/logs/clm-setup-%j.out" "$worker" setup "$suite"
        setup_id=$submitted_id
        array_ids=()
        for i in "${!model_sizes[@]}"; do
            size=${model_sizes[$i]}
            key=${model_keys[$i]}
            start=$((i * 4))
            end=$((start + 3))
            resource_args cell "$size"
            record_submit "array_${key}" "${common[@]}" "${args[@]}" \
                --job-name="ctxmgr-clm-${key}" --array="${start}-${end}%${CLM_PARALLEL}" \
                --dependency="afterok:$setup_id" --kill-on-invalid-dep=yes \
                --output="$suite/logs/clm-cell-%A_%a.out" "$worker" cell "$suite"
            array_ids+=("$submitted_id")
        done
        printf -v array_dependency '%s:' "${array_ids[@]}"
        array_dependency=${array_dependency%:}
        resource_args report
        record_submit report "${common[@]}" "${args[@]}" --job-name=ctxmgr-clm-report \
            --dependency="afterany:$array_dependency" --kill-on-invalid-dep=yes \
            --output="$suite/logs/clm-report-%j.out" "$worker" report "$suite"
        printf 'release-requested:%s\n' "$setup_id" > "$suite/submission/state"
        if ! scontrol release "$setup_id"; then
            fail "Could not confirm release of setup job $setup_id. All job IDs are recorded; inspect status before releasing or resubmitting."
        fi
        printf 'submitted\n' > "$suite/submission/state"
        printf '\nSubmitted 240 episodes in 20 GPU tasks across five model arrays (per-model parallel limit: %s).\n' "$CLM_PARALLEL"
        printf 'Suite: %s\nStatus: bash %q status %q\n' "$suite" "$package_root/scripts/clm_vllm.sh" "$suite"
        printf 'Original experiment directories are unchanged.\n'
        ;;
    status)
        load_suite "$@"
        printf 'Suite: %s\n' "$suite"
        cat "$suite/jobs.tsv"
        job_ids=$(awk -F '\t' 'NR>1 && $2 ~ /^[0-9]+$/ { if (ids != "") ids=ids ","; ids=ids $2 } END { print ids }' "$suite/jobs.tsv")
        if [[ -n "$job_ids" ]]; then
            sacct -X -j "$job_ids" --format=JobID%22,State%18,ExitCode,Elapsed
            squeue -j "$job_ids" -o '%.22i %.18T %.12M %.30R' || true
        fi
        if [[ -f "$suite/submission/state" ]]; then
            printf 'Submission state: '
            cat "$suite/submission/state"
        fi
        if [[ -f "$suite/report/report-status.json" ]]; then cat "$suite/report/report-status.json"; fi
        if [[ -f "$suite/report/summary.txt" ]]; then cat "$suite/report/summary.txt"; fi
        ;;
    report)
        load_suite "$@"
        require_command sbatch
        require_command flock
        exec 9> "$suite/submit.lock"
        flock -w 10 9
        resource_args report
        record_submit snapshot --parsable --export=ALL --no-requeue --chdir="$suite/source" "${args[@]}" \
            --job-name=ctxmgr-clm-report --output="$suite/logs/clm-report-%j.out" \
            "$suite/source/slurm/clm_vllm.sbatch" report "$suite"
        printf 'Report log: %s/logs/clm-report-%s.out\n' "$suite" "$submitted_id"
        ;;
    *) usage >&2; exit 2 ;;
esac
