"""Fresh 240-episode native CLM evaluation on vLLM/BF16; compute nodes only."""

import argparse
import csv
import hashlib
import io
import json
import os
import secrets
import shutil
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .clm_vllm_models import MODEL_SIZES, model_identity, verify_model

SEEDS = (42, 43, 44, 45)
LEVELS = (
    "kv_store:R130",
    "kv_store:R1000",
    "kv_store:R2000",
    "needle_retention:K4-C4",
    "needle_retention:K4-C24",
    "needle_retention:K8-C48",
)
CONDITIONS = ("none", "clm_self")
TRANSPORT = "reasoning-alias-v1"
SAMPLING = {"temperature": 0.7, "top_p": 0.95, "max_tokens": 8192, "enable_thinking": True}
GPU_NAME_MARKERS = {
    "a5000": "RTX A5000",
    "rtx_5000_ada": "RTX 5000 Ada",
    "a40": "A40",
    "l40s": "L40S",
    "h100": "H100",
    "h200": "H200",
    "rtx_pro_6000_blackwell": "RTX PRO 6000",
}


def gpu_allocation(size):
    suffix = size.replace(".", "")
    return {
        "partition": os.environ[f"CLM_GPU_{suffix}_PARTITION"],
        "type": os.environ[f"CLM_GPU_{suffix}_TYPE"],
        "host_memory": os.environ[f"CLM_GPU_{suffix}_MEM"],
    }


def gpu_name_matches(gpu_csv, expected_type):
    """Check the actual GPU name, not a driver version or the CSV heading."""
    marker = GPU_NAME_MARKERS.get(expected_type)
    if marker is None:
        raise ValueError(f"Unknown GPU type in frozen allocation: {expected_type}")
    return any(marker.lower() in line.split(",", 1)[0].lower() for line in gpu_csv.splitlines()[1:])


def now():
    return datetime.now(UTC).isoformat()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_json(path, value):
    """Atomic publication prevents readers observing truncated JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    stage = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    stage.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    stage.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def cells(suite):
    suite = Path(suite).resolve()
    return [
        {
            "index": i,
            "size": size,
            "seed": seed,
            "model": model_identity(size)[0],
            "root": str(suite / "cells" / f"{i:02d}-{size}-s{seed}"),
        }
        for i, (size, seed) in enumerate((s, n) for s in MODEL_SIZES for n in SEEDS)
    ]


def expected_rows(suite):
    return [
        {
            "cell": cell["index"],
            "model": cell["model"],
            "task": task,
            "seed": cell["seed"],
            "condition": condition,
        }
        for cell in cells(suite)
        for task in LEVELS
        for condition in CONDITIONS
    ]


def row_key(row):
    return row["model"], row["task"], row["seed"], row["condition"]


def setup(suite):
    from . import clm_official, clm_upstream
    from .clm_settings import resolve_settings

    suite = Path(suite).resolve()
    if (suite / "ready.json").exists() or (suite / "shared").exists():
        raise FileExistsError("Preparation is not repeatable in place; submit a new suite")
    verified = clm_upstream.verify_upstream()
    settings = resolve_settings("official")
    if settings["sampling"] != SAMPLING or settings["profile_overrides"]:
        raise ValueError("Pinned official CLM sampling differs from this experiment")
    if os.environ.get("CLM_MAX_MODEL_LEN", "81920") != "81920":
        raise ValueError("This fixed experiment requires served context 81920")
    server_python = os.environ["CLM_SERVE_PYTHON"]
    installed_version = subprocess.check_output(
        [server_python, "-c", "from importlib.metadata import version; print(version('vllm'))"],
        text=True,
    ).strip()
    if installed_version != "0.30.0":
        raise ValueError(f"Expected vLLM 0.30.0, observed {installed_version}")
    manifests = {}
    for size in MODEL_SIZES:
        alias = model_identity(size)[0]
        manifest = suite / "models" / alias / "model.json"
        verify_model(manifest)
        manifests[alias] = sha256(manifest)
    image = Path(os.environ["CLM_IMAGE"]).resolve(strict=True)
    plan = clm_official.prepare(
        suite / "shared",
        sizes=MODEL_SIZES,
        levels=LEVELS,
        seeds=SEEDS,
        conditions=CONDITIONS,
        served_max_len=81920,
        settings=settings,
    )
    # prepare() is allowed to return None; the canonical artifact is on disk.
    plan = clm_official.load_plan(suite / "shared")
    if len(plan["episodes"]) != 240:
        raise ValueError("Full grid must contain exactly 240 episodes")
    receipt = {
        "schema_version": 1,
        "created_at": now(),
        "planned": 240,
        "cells": cells(suite),
        "sampling": SAMPLING,
        "client_transport": TRANSPORT,
        "backend": "vllm",
        "server_version": installed_version,
        "precision": "bfloat16",
        "gpu_allocations": {size: gpu_allocation(size) for size in MODEL_SIZES},
        "served_max_len": 81920,
        "upstream": verified,
        "source_manifest_sha256": sha256(suite / "source.sha256"),
        "config_sha256": sha256(suite / "config.env"),
        "plan_sha256": sha256(suite / "shared" / "official-run.json"),
        "model_manifests": manifests,
        "image": str(image),
        "image_sha256": sha256(image),
        "fresh_run": True,
        "reuses_previous_grades": False,
    }
    write_json(suite / "ready.json", receipt)
    print("Preparation complete: 5 models x 6 levels x 4 seeds x 2 conditions = 240 episodes.")


def validate_ready(suite, *, verify_image=False):
    suite = Path(suite).resolve()
    ready = read_json(suite / "ready.json")
    expected = {
        "schema_version": 1,
        "planned": 240,
        "cells": cells(suite),
        "sampling": SAMPLING,
        "backend": "vllm",
        "precision": "bfloat16",
        "gpu_allocations": {size: gpu_allocation(size) for size in MODEL_SIZES},
        "server_version": "0.30.0",
        "client_transport": TRANSPORT,
        "served_max_len": 81920,
        "fresh_run": True,
        "reuses_previous_grades": False,
    }
    for key, value in expected.items():
        if ready.get(key) != value:
            raise ValueError(f"Prepared suite mismatch: {key}")
    for key, path in (
        ("source_manifest_sha256", suite / "source.sha256"),
        ("config_sha256", suite / "config.env"),
        ("plan_sha256", suite / "shared" / "official-run.json"),
    ):
        if sha256(path) != ready[key]:
            raise ValueError(f"Prepared suite artifact changed: {path.name}")
    for alias in (model_identity(size)[0] for size in MODEL_SIZES):
        if sha256(suite / "models" / alias / "model.json") != ready["model_manifests"].get(alias):
            raise ValueError(f"Prepared model manifest changed: {alias}")
    if verify_image and sha256(ready["image"]) != ready["image_sha256"]:
        raise ValueError("Container bytes differ from preparation")
    return ready


def server_argv(model, python, port, memory_utilization="0.90"):
    return [
        python,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        model["model_path"],
        "--served-model-name",
        model["model_alias"],
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "81920",
        "--max-num-seqs",
        "1",
        "--gpu-memory-utilization",
        str(memory_utilization),
        "--language-model-only",
        "--generation-config",
        "vllm",
        "--reasoning-parser",
        "qwen3",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "qwen3_coder",
        "--default-chat-template-kwargs",
        '{"enable_thinking":true}',
        "--seed",
        "0",
    ]


def unused_port():
    # Authenticated model validation detects an endpoint borrowed by a bind race.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def wait_healthy(process, url, timeout):
    from .clm_vllm_process import listener_owner

    endpoint = urlsplit(url)
    if endpoint.scheme != "http" or endpoint.hostname != "127.0.0.1" or not endpoint.port:
        raise ValueError("Readiness requires an explicit loopback HTTP port")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with {process.returncode}; inspect server logs")
        owner = listener_owner(endpoint.port, process.pid)
        if owner == "foreign":
            raise RuntimeError("Readiness port belongs to a different process group")
        if owner == "free":
            time.sleep(2)
            continue
        try:
            with urlopen(url, timeout=3) as response:
                if response.status == 200:
                    # A process that failed its bind cannot be accepted as ready.
                    time.sleep(0.5)
                    if process.poll() is not None:
                        raise RuntimeError("Server exited during readiness")
                    if listener_owner(endpoint.port, process.pid) != "owned":
                        raise RuntimeError("Listener ownership changed during readiness")
                    return
        except (URLError, TimeoutError):
            pass
        time.sleep(2)
    raise TimeoutError(f"Server did not become healthy within {timeout} seconds")


def stop_group(process):
    if process is None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=10)


def cell(suite, index):
    from . import clm_official
    from .clm_settings import resolve_settings

    suite = Path(suite).resolve()
    if not 0 <= index < 16:
        raise ValueError("Cell index must be in 0..15")
    ready = validate_ready(suite, verify_image=True)
    selection = cells(suite)[index]
    root = Path(selection["root"])
    root.mkdir(parents=True, exist_ok=False)
    status = {
        **selection,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": now(),
        "status": "initializing",
        "graded": 0,
        "planned": 12,
        "error": None,
    }
    write_json(root / "cell-status.json", status)
    server = proxy = episode_process = None
    handles = []
    old_handlers = {}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        old_handlers[signum] = signal.signal(signum, interrupted)
    try:
        original_manifest = suite / "models" / selection["model"] / "model.json"
        model = verify_model(original_manifest)
        clm_official.prepare(
            root,
            sizes=[selection["size"]],
            seeds=[selection["seed"]],
            levels=LEVELS,
            conditions=CONDITIONS,
            settings=resolve_settings("official"),
            served_max_len=81920,
            reference_instances=suite / "shared" / "instances",
        )
        manifest_dir = root / "models" / selection["model"]
        manifest_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original_manifest, manifest_dir / "model.json")
        port, proxy_port = unused_port(), unused_port()
        while proxy_port == port:
            proxy_port = unused_port()
        argv = server_argv(
            model,
            os.environ["CLM_SERVE_PYTHON"],
            port,
            os.environ.get("CLM_GPU_MEMORY_UTILIZATION", "0.90"),
        )
        # The token is local to this job and is never written into provenance.
        # A competing process on a raced port cannot impersonate this endpoint.
        endpoint_key = secrets.token_urlsafe(32)
        child_environment = dict(os.environ, VLLM_API_KEY=endpoint_key, OPENAI_API_KEY=endpoint_key)
        version = subprocess.check_output(
            [
                os.environ["CLM_SERVE_PYTHON"],
                "-c",
                "from importlib.metadata import version; print(version('vllm'))",
            ],
            text=True,
        ).strip()
        if version != ready["server_version"]:
            raise ValueError("Serving version changed after preparation")
        gpu = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv"],
            text=True,
        )
        (root / "gpu.txt").write_text(gpu)
        allocation = gpu_allocation(selection["size"])
        if not gpu_name_matches(gpu, allocation["type"]):
            raise ValueError(f"Expected {allocation['type']} for {selection['size']}; observed {gpu!r}")
        write_json(
            root / "server-config.json",
            {
                "argv": argv,
                "backend": "vllm",
                "version": version,
                "precision": "bfloat16",
                "gpu_allocation": allocation,
                "client_transport": TRANSPORT,
                "server_seed": 0,
                "request_seed": None,
                "endpoint_auth": "per-cell-ephemeral",
                "model_manifest_sha256": sha256(original_manifest),
                "image_sha256": ready["image_sha256"],
                "started_at": now(),
                "environment": {
                    key: os.environ.get(key)
                    for key in (
                        "VLLM_USE_FLASHINFER_SAMPLER",
                        "VLLM_USE_DEEP_GEMM",
                        "VLLM_ENGINE_READY_TIMEOUT_S",
                    )
                },
            },
        )
        server_log = (root / "server.log").open("w")
        handles.append(server_log)
        server = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=child_environment,
        )
        status["status"] = "server_starting"
        write_json(root / "cell-status.json", status)
        wait_healthy(
            server,
            f"http://127.0.0.1:{port}/health",
            int(os.environ.get("CLM_SERVER_READY_TIMEOUT", "3600")),
        )
        proxy_log = (root / "transport.log").open("w")
        handles.append(proxy_log)
        proxy = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "ctxmgr.harness.clm_vllm_proxy",
                "--port",
                str(proxy_port),
                "--upstream",
                f"http://127.0.0.1:{port}",
            ],
            stdin=subprocess.DEVNULL,
            stdout=proxy_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        base_url = f"http://127.0.0.1:{proxy_port}/v1"
        wait_healthy(proxy, f"http://127.0.0.1:{proxy_port}/health", 60)
        request = Request(f"{base_url}/models", headers={"Authorization": f"Bearer {endpoint_key}"})
        with urlopen(request, timeout=20) as response:
            model_list = json.load(response)
        if selection["model"] not in [item["id"] for item in model_list["data"]]:
            raise ValueError("Endpoint serves a different model alias")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ctxmgr.harness.clm_probe",
                "--base-url",
                base_url,
                "--model",
                selection["model"],
                "--out",
                str(root / "transport-probe.json"),
                "--profile",
                "official",
                "--served-max-len",
                "81920",
            ],
            stdin=subprocess.DEVNULL,
            check=True,
            env=child_environment,
        )
        for row in clm_official.load_plan(root)["episodes"]:
            if server.poll() is not None or proxy.poll() is not None:
                raise RuntimeError("Serving process exited; remaining episodes are unrun")
            print(f"===== {row['model']} {row['condition']} {row['name']} =====", flush=True)
            status.update(status="running", active_episode=f"{row['condition']}/{row['name']}")
            write_json(root / "cell-status.json", status)
            log_path = root / f"episode-{row['condition']}-{row['name']}.log"
            with log_path.open("w") as log:
                episode_process = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "ctxmgr.harness.clm_official",
                        "run",
                        "--root",
                        str(root),
                        "--model",
                        row["model"],
                        "--name",
                        row["name"],
                        "--condition",
                        row["condition"],
                        "--base-url",
                        base_url,
                        "--image",
                        ready["image"],
                    ],
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=child_environment,
                )
                returncode = episode_process.wait()
            # Stop any lingering children even when the episode leader exited normally.
            stop_group(episode_process)
            episode_process = None
            if returncode:
                print(f"Episode failed with exit {returncode}: {log_path}", flush=True)
            rows = clm_official.summary(root)
            status["graded"] = sum(row["status"] == "completed" for row in rows)
            write_json(root / "cell-status.json", status)
        rows = clm_official.summary(root)
        status.update(
            graded=sum(row["status"] == "completed" for row in rows),
            status="complete"
            if all(row["status"] == "completed" for row in rows)
            else "incomplete",
        )
        print((root / "results.txt").read_text(), flush=True)
        print(f"Validated grades for this cell: {status['graded']}/12", flush=True)
        return 0 if status["status"] == "complete" else 1
    except BaseException as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        # Do not permit a second scheduler signal to interrupt child cleanup.
        for signum in old_handlers:
            signal.signal(signum, signal.SIG_IGN)
        for process in (episode_process, proxy, server):
            stop_group(process)
        for handle in handles:
            handle.close()
        status["finished_at"] = now()
        write_json(root / "cell-status.json", status)
        for signum, handler in old_handlers.items():
            signal.signal(signum, handler)


def paired_comparison(rows):
    by_key = {row_key(row): row for row in rows}
    if len(by_key) != len(rows):
        raise ValueError("Duplicate episode identities in report")
    comparison = []
    for size in MODEL_SIZES:
        model = model_identity(size)[0]
        for task in LEVELS:
            pairs = []
            for seed in SEEDS:
                left = by_key[model, task, seed, "none"]
                right = by_key[model, task, seed, "clm_self"]
                if left["status"] == right["status"] == "completed":
                    pairs.append((left["reward"], right["reward"]))
            comparison.append(
                {
                    "model": model,
                    "task": task,
                    "paired": len(pairs),
                    "expected": len(SEEDS),
                    "none": sum(x for x, _ in pairs) / len(pairs) if pairs else None,
                    "clm_self": sum(y for _, y in pairs) / len(pairs) if pairs else None,
                    "delta": sum(y - x for x, y in pairs) / len(pairs) if pairs else None,
                }
            )
    return comparison


def table_tsv(rows):
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def format_score(value, signed=False):
    return "—" if value is None else format(value, "+.4f" if signed else ".4f")


TERMINAL_SLURM_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
    "REVOKED",
}


def scheduler_states(job_ids):
    """Read terminal state without inferring failure from an empty live queue."""
    identifiers = sorted({str(value) for value in job_ids if value is not None})
    if not identifiers or any(not value.isdecimal() for value in identifiers):
        return {}
    try:
        output = subprocess.check_output(
            ["sacct", "-X", "-n", "-P", "-j", ",".join(identifiers), "--format=JobIDRaw,State"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return {}
    result = {}
    for line in output.splitlines():
        fields = line.split("|")
        if len(fields) >= 2 and fields[0].strip() in identifiers and fields[1].strip():
            result[fields[0].strip()] = fields[1].strip().split()[0].rstrip("+")
    return result


def render_report(rows, comparison, status):
    lines = [
        f"# Native CLM on vLLM/BF16: {status['graded']}/240 validated grades",
        "",
        (
            f"Coverage: **{status['status']}**. Failed/invalid: {status['failed']}; "
            f"unrun: {status['unrun']}; in progress: {status['running']}."
        ),
        "",
        (
            "Fresh evaluation: Qwen3.5 0.8B/2B/4B/9B/27B; 6 task levels; task seeds 42–45; "
            "conditions `none` and native `clm_self`. No previous Q8_0 results are reused."
        ),
        "",
        (
            "vLLM 0.30.0; BF16; one size-matched GPU per cell (see provenance); "
            "served context 81920; one concurrent sequence. "
            "Official CLM sampling: thinking enabled, temperature 0.7, top_p 0.95, max_tokens 8192. "
            "Model generation defaults are disabled with `--generation-config vllm`. "
            "The server seed is 0; task seeds are not sent as inference seeds."
        ),
        "",
        (
            "Budget: 32768 raw o200k tokens with 2048 reserved (`CLM_TOK_RATIO=0`). "
            "Graded means a validated saved score, not necessarily completion of the task stream. "
            "`budget_wall` and `max_steps` are early termination. Missing or invalid scores are blank, never zero."
        ),
        "",
        "| Model | Task | Paired / expected | none | clm_self | clm_self − none |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in comparison:
        lines.append(
            f"| {row['model']} | {row['task']} | {row['paired']}/{row['expected']} | "
            f"{format_score(row['none'])} | {format_score(row['clm_self'])} | "
            f"{format_score(row['delta'], True)} |"
        )
    lines += [
        "",
        "<details>",
        "<summary>Per-seed scores (42, 43, 44, 45)</summary>",
        "",
        "| Model | Task | Condition | 42 | 43 | 44 | 45 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    lookup = {row_key(row): row for row in rows}
    for size in MODEL_SIZES:
        alias = model_identity(size)[0]
        for task in LEVELS:
            for condition in CONDITIONS:
                scores = [lookup[alias, task, seed, condition]["reward"] for seed in SEEDS]
                lines.append(
                    f"| {alias} | {task} | {condition} | "
                    + " | ".join(format_score(value) for value in scores)
                    + " |"
                )
    lines += [
        "",
        "</details>",
        "",
        "Termination among validated grades:",
        "",
        "| Model | Condition | Graded | stream_exhausted | budget_wall | submit | max_steps |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for size in MODEL_SIZES:
        alias = model_identity(size)[0]
        for condition in CONDITIONS:
            selected = [
                r
                for r in rows
                if r["model"] == alias
                and r["condition"] == condition
                and r["status"] == "completed"
            ]
            counts = Counter(r["termination"] for r in selected)
            lines.append(
                f"| {alias} | {condition} | {len(selected)} | "
                + " | ".join(
                    str(counts[k])
                    for k in ("stream_exhausted", "budget_wall", "submit", "max_steps")
                )
                + " |"
            )
    lines += [
        "",
        "## Interpretation and provenance",
        "",
        (
            "Every comparison uses matching graded task-seed pairs. Seeds share task structure; "
            "no independent-replicate confidence interval or significance test is inferred. "
            "Inspect per-seed scores and termination counts before making claims."
        ),
        (
            "GPU hardware differs by model size; cross-size wall time, throughput and GPU-cost "
            "comparisons must account for the recorded allocation. BF16 precision and generation "
            "settings remain fixed across sizes."
        ),
        "",
        (
            "Native `clm_self` retains the pinned CLM prompts, Bash/file tools, reasoning history "
            "and budget final turn. The original chat `none` baseline retains its own prompts, "
            "content-only history and no-extra-call budget wall. This is a comparison of systems, "
            "not a controlled ablation of editing permissions alone. `reasoning-alias-v1` translates "
            "the vLLM response field `reasoning` to `reasoning_content` without changing content, "
            "tool calls, sampling, or either agent implementation."
        ),
        "",
        (
            "This is the requested two-condition, five-model grid, not the instructor's full "
            "peer-manager grid. Native `clm_self` is distinct from May's JSON `self` protocol. "
            "Do not pool this BF16 series with prior GGUF/Q8_0 scores or teammate experiments "
            "using different sampling, protocols, or budgets."
        ),
        "",
        (
            "Immutable HF revisions and model-file SHA-256 inventories are recorded in "
            "`provenance.json`. The aggregate model SHA-256 is a canonical snapshot-file-tree "
            "digest, not the hash of one weight file. The reporter validates saved metadata and "
            "task identities; it does not independently regrade snapshots."
        ),
        "",
        (
            "Prepared with AI-assisted orchestration and reporting code. Scientific interpretation "
            "and publication require human review."
        ),
        "",
    ]
    return "\n".join(lines)


def report(suite):
    from . import clm_official

    suite = Path(suite).resolve()
    ready = validate_ready(suite)
    shared = clm_official.load_plan(suite / "shared")
    shared_hashes = {row["name"]: row["instance_sha256"] for row in shared["episodes"]}
    collected = {}
    errors = {}
    jobs = {}
    finished_cells = set()
    for selection in cells(suite):
        root = Path(selection["root"])
        if (root / "cell-status.json").exists():
            try:
                info = read_json(root / "cell-status.json")
                jobs[selection["index"]] = info.get("job_id")
                if info.get("finished_at"):
                    finished_cells.add(selection["index"])
                if info.get("error"):
                    errors[selection["index"]] = info["error"]
            except (ValueError, OSError) as exc:
                errors[selection["index"]] = str(exc)
        if not (root / "official-run.json").exists():
            continue
        try:
            plan = clm_official.load_plan(root)
            config = plan["config"]
            if (
                config["sizes"] != [selection["size"]]
                or config["seeds"] != [selection["seed"]]
                or config["levels"] != list(LEVELS)
                or config["conditions"] != list(CONDITIONS)
                or config["settings"] != shared["config"]["settings"]
                or config["served_max_len"] != 81920
            ):
                raise ValueError("Cell grid/settings differ from the prepared suite")
            for item in plan["episodes"]:
                if item["instance_sha256"] != shared_hashes[item["name"]]:
                    raise ValueError("Cell task hashes differ from the shared suite")
            if (
                sha256(root / "models" / selection["model"] / "model.json")
                != ready["model_manifests"][selection["model"]]
            ):
                raise ValueError("Cell model manifest differs from the prepared suite")
            result_rows = clm_official.summary(root, write_outputs=False)
            expected_keys = {
                row_key(row) for row in expected_rows(suite) if row["cell"] == selection["index"]
            }
            observed_keys = [row_key(row) for row in result_rows]
            if len(observed_keys) != len(expected_keys) or set(observed_keys) != expected_keys:
                raise ValueError("Cell summary has missing, duplicate, or unexpected identities")
            pending = {}
            for row in result_rows:
                key = row_key(row)
                if key in collected:
                    raise ValueError("Duplicate result identity")
                pending[key] = row
            collected.update(pending)
        except (ValueError, KeyError, OSError, TypeError) as exc:
            errors[selection["index"]] = f"Invalid cell: {exc}"
    states = scheduler_states(jobs.values())
    rows = []
    for expected in expected_rows(suite):
        observed = collected.get(row_key(expected), {})
        valid = observed.get("status") == "completed"
        cell_error = errors.get(expected["cell"])
        row_status = observed.get("status", "error" if cell_error else "unrun")
        slurm_state = states.get(str(jobs.get(expected["cell"])))
        if row_status == "running" and (
            expected["cell"] in finished_cells or slurm_state in TERMINAL_SLURM_STATES
        ):
            row_status = "error"
            cell_error = (
                cell_error
                or f"Cell job ended without a valid final score ({slurm_state or 'finished'})"
            )
        rows.append(
            {
                **expected,
                "status": row_status,
                "reward": observed.get("reward") if valid else None,
                "termination": observed.get("termination"),
                "ops": observed.get("ops"),
                "edits": observed.get("edits"),
                "error": observed.get("error") or cell_error,
                "source_job_id": jobs.get(expected["cell"]),
                "slurm_state": slurm_state,
            }
        )
    counts = Counter(row["status"] for row in rows)
    status = {
        "schema_version": 1,
        "updated_at": now(),
        "planned": 240,
        "graded": counts["completed"],
        "failed": counts["error"],
        "unrun": counts["unrun"],
        "running": counts["running"],
        "status": "complete" if counts["completed"] == 240 else "partial",
    }
    comparisons = paired_comparison(rows)
    summaries = []
    for size in MODEL_SIZES:
        alias = model_identity(size)[0]
        for task in LEVELS:
            for condition in CONDITIONS:
                selected = [
                    r
                    for r in rows
                    if r["model"] == alias
                    and r["task"] == task
                    and r["condition"] == condition
                    and r["status"] == "completed"
                ]
                summaries.append(
                    {
                        "model": alias,
                        "task": task,
                        "condition": condition,
                        "graded": len(selected),
                        "expected": 4,
                        "mean": sum(r["reward"] for r in selected) / len(selected)
                        if selected
                        else None,
                    }
                )
    out = suite / "report"
    out.mkdir(exist_ok=True)
    contents = {
        "results.tsv": table_tsv(rows),
        "comparison.tsv": table_tsv(comparisons),
        "summary.tsv": table_tsv(summaries),
        "report.md": render_report(rows, comparisons, status),
    }
    for filename, content in contents.items():
        stage = out / f".{filename}.{os.getpid()}.tmp"
        stage.write_text(content, encoding="utf-8")
        stage.replace(out / filename)
    write_json(
        out / "provenance.json",
        {
            "backend": ready["backend"],
            "server_version": ready["server_version"],
            "precision": ready["precision"],
            "gpu_allocations": ready["gpu_allocations"],
            "client_transport": TRANSPORT,
            "upstream": ready["upstream"],
            "team_source_commit": shared["team_source_commit"],
            "team_source_sha256": shared["team_source_sha256"],
            "image_sha256": ready["image_sha256"],
            "source_manifest_sha256": ready["source_manifest_sha256"],
            "models": {
                model_identity(size)[0]: read_json(
                    suite / "models" / model_identity(size)[0] / "model.json"
                )
                for size in MODEL_SIZES
            },
            "jobs": jobs,
        },
    )
    status["report_sha256"] = {name: sha256(out / name) for name in (*contents, "provenance.json")}
    write_json(out / "report-status.json", status)
    print(f"Validated grades: {status['graded']}/240. Coverage: {status['status']}.")
    print(f"Report: {out / 'report.md'}")
    for item in comparisons:
        print(
            f"{item['model']}\t{item['task']}\t{item['paired']}/4\t"
            f"{format_score(item['none'])}\t{format_score(item['clm_self'])}\t{format_score(item['delta'], True)}"
        )
    return 0 if status["status"] == "complete" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("setup", "cell", "report"))
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args(argv)
    if args.command == "setup":
        setup(args.suite)
        return 0
    if args.command == "cell":
        if args.index is None:
            parser.error("cell requires --index")
        return cell(args.suite, args.index)
    return report(args.suite)


if __name__ == "__main__":
    raise SystemExit(main())
