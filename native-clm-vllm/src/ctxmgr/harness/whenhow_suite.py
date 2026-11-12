"""Run the when-versus-how grid on a prepared native-CLM vLLM suite; compute nodes only.

The base suite (``clm_vllm.sh submit``) supplies the verified BF16 model snapshots, the
container, the locked environments and the shared ContextBench task instances, so these
episodes see the same tasks as that suite's ``none`` and ``clm_self`` episodes.

One job = one actor size, one manager kind and one task seed:

    python -m ctxmgr.harness.whenhow_suite job --base-suite S --root R \
        --actor 4B --manager 27B --seed 42

``--manager same`` runs the self-timed, harness-timed and same-size manager conditions on
the actor's own server. ``--manager 27B`` adds a second server on the same GPU and runs
the three manager conditions with the 27B manager. Episodes whose output directory
already exists are skipped, so a job can be resubmitted after an interruption.
"""

from __future__ import annotations

import argparse
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from .clm_vllm_suite import (
    read_json,
    server_argv,
    stop_group,
    unused_port,
    validate_ready,
    verify_model,
    wait_healthy,
    write_json,
)

SIZES = ("2B", "4B", "9B", "27B")
SEEDS = (42, 43, 44, 45)
LEVELS = (
    ("needle_retention", "K4-C24"),
    ("needle_retention", "K8-C48"),
    ("kv_store", "R1000"),
    ("kv_store", "R2000"),
)
SAME_CONDITIONS = ("clm_self_noskill", "clm_fixed", "mgr_fixed", "mgr_request", "mgr_watch")
MANAGER_CONDITIONS = ("mgr_fixed", "mgr_request", "mgr_watch")
LARGE_MANAGER = "27B"
# Two servers share one H200: the 27B manager first, then the actor.
MEMORY = {"single": "0.90", "manager": "0.55", "actor": "0.30"}


def alias(size: str) -> str:
    return f"Qwen3.5-{size}-BF16"


def plan(actor: str, manager: str, seed: int) -> list[dict]:
    if actor not in SIZES:
        raise ValueError(f"actor must be one of {SIZES}")
    if manager == "same":
        conditions = SAME_CONDITIONS
    elif manager == LARGE_MANAGER and actor != LARGE_MANAGER:
        conditions = MANAGER_CONDITIONS
    else:
        raise ValueError("manager must be 'same', or '27B' for a smaller actor")
    if seed not in SEEDS:
        raise ValueError(f"seed must be one of {SEEDS}")
    rows = []
    for condition in conditions:
        managed = condition in MANAGER_CONDITIONS
        manager_size = (actor if manager == "same" else manager) if managed else None
        tag = condition if not managed else f"{condition}@{manager_size}"
        for family, level in LEVELS:
            name = f"{family}-{level}-s{seed}"
            rows.append(
                {
                    "actor": actor,
                    "manager": manager_size,
                    "condition": condition,
                    "tag": tag,
                    "family": family,
                    "level": level,
                    "seed": seed,
                    "name": name,
                }
            )
    return rows


def episode_out(root: Path, row: dict) -> Path:
    return root / alias(row["actor"]) / row["tag"] / row["name"]


class Server:
    """One vLLM server plus the reasoning-field gateway, on loopback only."""

    def __init__(self, model: dict, memory: str, log_dir: Path, environment: dict, label: str):
        self.model, self.memory, self.log_dir, self.env, self.label = (
            model,
            memory,
            log_dir,
            environment,
            label,
        )
        self.server = self.proxy = None
        self.handles = []
        self.base_url = None

    def start(self, timeout: int) -> str:
        port, proxy_port = unused_port(), unused_port()
        while proxy_port == port:
            proxy_port = unused_port()
        argv = server_argv(self.model, os.environ["CLM_SERVE_PYTHON"], port, self.memory)
        write_json(
            self.log_dir / f"server-{self.label}.json", {"argv": argv, "memory": self.memory}
        )
        log = (self.log_dir / f"server-{self.label}.log").open("w")
        self.handles.append(log)
        self.server = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=self.env,
        )
        wait_healthy(self.server, f"http://127.0.0.1:{port}/health", timeout)
        plog = (self.log_dir / f"transport-{self.label}.log").open("w")
        self.handles.append(plog)
        self.proxy = subprocess.Popen(
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
            stdout=plog,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        wait_healthy(self.proxy, f"http://127.0.0.1:{proxy_port}/health", 60)
        self.base_url = f"http://127.0.0.1:{proxy_port}/v1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "ctxmgr.harness.clm_probe",
                "--base-url",
                self.base_url,
                "--model",
                self.model["model_alias"],
                "--out",
                str(self.log_dir / f"transport-probe-{self.label}.json"),
                "--profile",
                "official",
                "--served-max-len",
                "81920",
            ],
            stdin=subprocess.DEVNULL,
            check=True,
            env=self.env,
        )
        return self.base_url

    def alive(self) -> bool:
        return all(p is not None and p.poll() is None for p in (self.server, self.proxy))

    def stop(self) -> None:
        for process in (self.proxy, self.server):
            stop_group(process)
        for handle in self.handles:
            handle.close()


def episode_argv(row: dict, *, instance: Path, out: Path, ready: dict, models: dict, urls: dict):
    actor = models[row["actor"]]
    argv = [
        sys.executable,
        "-m",
        "ctxmgr.harness.clm_upstream",
        "run",
        str(instance),
        "--out",
        str(out),
        "--model",
        actor["model_alias"],
        "--condition",
        row["condition"],
        "--base-url",
        urls[row["actor"]],
        "--image",
        ready["image"],
        "--environment",
        "singularity",
        "--profile",
        "official",
        "--backend",
        "vllm",
        "--server-version",
        ready["server_version"],
        "--precision",
        "bfloat16",
        "--model-revision",
        actor["revision"],
        "--model-sha256",
        actor["sha256"],
        "--served-max-len",
        "81920",
        "--budget-ruler",
        "o200k",
    ]
    if row["manager"] is not None:
        argv += [
            "--manager-model",
            models[row["manager"]]["model_alias"],
            "--manager-base-url",
            urls[row["manager"]],
        ]
    return argv


def job(base_suite: Path, root: Path, actor: str, manager: str, seed: int) -> int:
    base_suite, root = base_suite.resolve(), root.resolve()
    ready = validate_ready(base_suite, verify_image=True)
    rows = plan(actor, manager, seed)
    instances = base_suite / "shared" / "instances"
    for row in rows:
        if not (instances / row["name"]).is_dir():
            raise FileNotFoundError(f"Missing shared instance {row['name']} in {instances}")
    sizes = [actor] if manager == "same" else [manager, actor]
    models = {
        size: verify_model(base_suite / "models" / alias(size) / "model.json") for size in sizes
    }
    log_dir = (
        root
        / "jobs"
        / f"{alias(actor)}-{manager}-s{seed}-{os.environ.get('SLURM_JOB_ID', 'local')}"
    )
    log_dir.mkdir(parents=True, exist_ok=False)
    status = {
        "actor": actor,
        "manager": manager,
        "seed": seed,
        "base_suite": str(base_suite),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "planned": len(rows),
        "done": 0,
        "skipped_existing": 0,
        "failed": [],
    }
    write_json(log_dir / "status.json", status)
    key = secrets.token_urlsafe(32)
    environment = dict(os.environ, VLLM_API_KEY=key, OPENAI_API_KEY=key)
    timeout = int(os.environ.get("CLM_SERVER_READY_TIMEOUT", "3600"))
    servers, urls = [], {}
    old = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    for signum in old:
        signal.signal(signum, interrupted)
    try:
        if manager == "same":
            servers.append(Server(models[actor], MEMORY["single"], log_dir, environment, "actor"))
            urls[actor] = servers[0].start(timeout)
        else:
            servers.append(
                Server(models[manager], MEMORY["manager"], log_dir, environment, "manager")
            )
            urls[manager] = servers[0].start(timeout)
            servers.append(Server(models[actor], MEMORY["actor"], log_dir, environment, "actor"))
            urls[actor] = servers[1].start(timeout)
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv"],
            check=False,
            capture_output=True,
            text=True,
        ).stdout
        (log_dir / "gpu.txt").write_text(gpu)
        for row in rows:
            if not all(server.alive() for server in servers):
                raise RuntimeError("A serving process exited; remaining episodes are unrun")
            out = episode_out(root, row)
            if out.exists():
                status["skipped_existing"] += 1
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            print(f"===== {row['tag']} {row['actor']} {row['name']} =====", flush=True)
            argv = episode_argv(
                row,
                instance=instances / row["name"],
                out=out,
                ready=ready,
                models=models,
                urls=urls,
            )
            log_path = log_dir / f"episode-{row['tag'].replace('@', '-')}-{row['name']}.log"
            started = time.monotonic()
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    env=environment,
                )
                code = process.wait()
            stop_group(process)
            if code:
                status["failed"].append({"episode": f"{row['tag']}/{row['name']}", "exit": code})
            status["done"] += 1
            status.setdefault("episode_seconds", {})[f"{row['tag']}/{row['name']}"] = round(
                time.monotonic() - started, 1
            )
            write_json(log_dir / "status.json", status)
        status["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return 0 if not status["failed"] else 1
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for signum in old:
            signal.signal(signum, signal.SIG_IGN)
        for server in reversed(servers):
            server.stop()
        write_json(log_dir / "status.json", status)
        for signum, handler in old.items():
            signal.signal(signum, handler)


def collect(root: Path) -> list[dict]:
    """One row per finished episode: reward, termination and management summary."""
    rows = []
    for score_path in sorted(root.glob("Qwen3.5-*-BF16/*/*/score.json")):
        out = score_path.parent
        episode = read_json(out / "episode.json") if (out / "episode.json").exists() else {}
        score = read_json(score_path)
        whenhow = {}
        for candidate in out.rglob("whenhow.json"):
            whenhow = read_json(candidate).get("summary", {})
            break
        rows.append(
            {
                "actor": out.parent.parent.name,
                "tag": out.parent.name,
                "name": out.name,
                "family": episode.get("family"),
                "level": episode.get("level"),
                "seed": episode.get("seed"),
                "reward": (score.get("official") or {}).get("reward"),
                "terminated_by": episode.get("terminated_by"),
                "error": bool(episode.get("error")),
                **{f"wh_{k}": v for k, v in whenhow.items()},
            }
        )
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("job")
    run.add_argument("--base-suite", type=Path, required=True)
    run.add_argument("--root", type=Path, required=True)
    run.add_argument("--actor", choices=SIZES, required=True)
    run.add_argument("--manager", choices=("same", LARGE_MANAGER), required=True)
    run.add_argument("--seed", type=int, choices=SEEDS, required=True)
    show = commands.add_parser("collect")
    show.add_argument("--root", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "job":
        return job(args.base_suite, args.root, args.actor, args.manager, args.seed)
    rows = collect(args.root)
    write_json(args.root / "results.json", rows)
    print(json.dumps({"episodes": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
