"""BrowseComp-Plus jobs on a prepared native-CLM suite; compute nodes only (envs/clm).

    python -m ctxmgr.bcp.suite job --base-suite S --root R --data D \
        --actor 9B --manager same --conditions clm_self,clm_fixed --questions 0:25
    python -m ctxmgr.bcp.suite judge --base-suite S --root R --data D

A job serves the actor (and a 27B manager on the same GPU when ``--manager 27B``) with
``--max-num-seqs`` equal to the episode concurrency, starts the corpus search server, and
runs its episodes ``--parallel`` at a time. Concurrency changes wall-clock time, not the
episodes; token counts are logged per call. Existing episode directories are skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import secrets
import signal
import subprocess
import sys
import time
from pathlib import Path

from ctxmgr.harness import whenhow_suite as base
from ctxmgr.harness.clm_vllm_suite import (
    server_argv,
    stop_group,
    unused_port,
    validate_ready,
    verify_model,
    wait_healthy,
    write_json,
)

from .agent import CONDITIONS

MANAGED = {"mgr_fixed", "mgr_request", "mgr_watch"}


def question_ids(data: Path) -> list[str]:
    rows = (data / "questions.jsonl").read_text(encoding="utf-8").splitlines()
    return [str(json.loads(line)["query_id"]) for line in rows if line.strip()]


def tag(condition: str, actor: str, manager: str) -> str:
    if condition not in MANAGED:
        return condition
    return f"{condition}@{actor if manager == 'same' else manager}"


class BatchServer(base.Server):
    """``whenhow_suite.Server`` with a configurable number of concurrent sequences."""

    def __init__(self, *args, max_seqs: int = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_seqs = max_seqs

    def start(self, timeout: int) -> str:
        original = base.server_argv

        def argv(*a, **kw):
            out = server_argv(*a, **kw)
            out[out.index("--max-num-seqs") + 1] = str(self.max_seqs)
            return out

        base.server_argv = argv
        try:
            return super().start(timeout)
        finally:
            base.server_argv = original


def start_search(data: Path, log_dir: Path, timeout: int = 1800):
    port = unused_port()
    log = (log_dir / "search.log").open("w")
    process = subprocess.Popen(
        [
            os.environ["BCP_PYTHON"],
            "-m",
            "ctxmgr.bcp.server",
            "--data",
            str(data),
            "--port",
            str(port),
        ],
        stdin=subprocess.DEVNULL,
        stdout=log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    wait_healthy(process, f"http://127.0.0.1:{port}/health", timeout)
    return process, log, f"http://127.0.0.1:{port}"


def job(args: argparse.Namespace) -> int:
    base_suite, root, data = args.base_suite.resolve(), args.root.resolve(), args.data.resolve()
    ready = validate_ready(base_suite, verify_image=True)
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for condition in conditions:
        if condition not in CONDITIONS:
            raise ValueError(f"unknown condition {condition}")
        if args.manager == "27B" and condition not in MANAGED:
            raise ValueError("a 27B-manager job runs only manager conditions")
    start, stop = (int(x) for x in args.questions.split(":"))
    ids = question_ids(data)[start:stop]
    sizes = [args.actor] if args.manager == "same" else [args.manager, args.actor]
    models = {s: verify_model(base_suite / "models" / base.alias(s) / "model.json") for s in sizes}
    stamp = os.environ.get("SLURM_JOB_ID", "local")
    log_dir = root / "jobs" / f"{base.alias(args.actor)}-{args.manager}-{start}-{stop}-{stamp}"
    log_dir.mkdir(parents=True, exist_ok=False)
    key = secrets.token_urlsafe(32)
    environment = dict(os.environ, VLLM_API_KEY=key, OPENAI_API_KEY=key)
    timeout = int(os.environ.get("CLM_SERVER_READY_TIMEOUT", "3600"))
    servers, urls, search = [], {}, None
    status = {"planned": len(ids) * len(conditions), "done": 0, "failed": [], "seconds": {}}

    def interrupted(signum, _frame):
        raise InterruptedError(f"Received signal {signum}")

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, interrupted)
    try:
        search, search_log, search_url = start_search(data, log_dir)
        if args.manager == "same":
            servers.append(
                BatchServer(
                    models[args.actor],
                    "0.90",
                    log_dir,
                    environment,
                    "actor",
                    max_seqs=args.parallel,
                )
            )
            urls[args.actor] = servers[0].start(timeout)
        else:
            servers.append(
                BatchServer(
                    models[args.manager],
                    "0.55",
                    log_dir,
                    environment,
                    "manager",
                    max_seqs=args.parallel,
                )
            )
            urls[args.manager] = servers[0].start(timeout)
            servers.append(
                BatchServer(
                    models[args.actor],
                    "0.30",
                    log_dir,
                    environment,
                    "actor",
                    max_seqs=args.parallel,
                )
            )
            urls[args.actor] = servers[1].start(timeout)
        work = []
        for condition in conditions:
            for qid in ids:
                out = (
                    root
                    / base.alias(args.actor)
                    / tag(condition, args.actor, args.manager)
                    / f"q{qid}"
                )
                if not out.exists():
                    work.append((condition, qid, out))

        def run_one(item):
            condition, qid, out = item
            out.parent.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable, "-m", "ctxmgr.bcp.agent", "run",
                "--question-id", qid, "--data", str(data), "--out", str(out),
                "--condition", condition, "--model", models[args.actor]["model_alias"],
                "--base-url", urls[args.actor], "--search-url", search_url, "--image", ready["image"],
            ]  # fmt: skip
            if condition in MANAGED:
                manager = args.actor if args.manager == "same" else args.manager
                argv += [
                    "--manager-model", models[manager]["model_alias"],
                    "--manager-base-url", urls[manager],
                ]  # fmt: skip
            began = time.monotonic()
            log_path = (
                log_dir / f"{tag(condition, args.actor, args.manager).replace('@', '-')}-q{qid}.log"
            )
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    argv, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                    start_new_session=True, env=environment,
                )  # fmt: skip
                code = process.wait()
            stop_group(process)
            return condition, qid, code, round(time.monotonic() - began, 1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            for condition, qid, code, seconds in pool.map(run_one, work):
                status["done"] += 1
                status["seconds"][f"{condition}/q{qid}"] = seconds
                if code:
                    status["failed"].append(f"{condition}/q{qid}")
                write_json(log_dir / "status.json", status)
                if not all(s.alive() for s in servers) or search.poll() is not None:
                    raise RuntimeError("A serving process exited; remaining episodes are unrun")
        return 0 if not status["failed"] else 1
    except BaseException as exc:
        status["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        for server in reversed(servers):
            server.stop()
        if search is not None:
            stop_group(search)
            search_log.close()
        write_json(log_dir / "status.json", status)


def judge(args: argparse.Namespace) -> int:
    from .judge import judge_root

    base_suite, root, data = args.base_suite.resolve(), args.root.resolve(), args.data.resolve()
    validate_ready(base_suite)
    model = verify_model(base_suite / "models" / base.alias("27B") / "model.json")
    key = secrets.token_urlsafe(32)
    environment = dict(os.environ, VLLM_API_KEY=key, OPENAI_API_KEY=key)
    log_dir = root / "jobs" / f"judge-{os.environ.get('SLURM_JOB_ID', 'local')}"
    log_dir.mkdir(parents=True, exist_ok=False)
    server = base.Server(model, "0.90", log_dir, environment, "judge")
    try:
        url = server.start(int(os.environ.get("CLM_SERVER_READY_TIMEOUT", "3600")))
        rows = judge_root(root, data, url, model["model_alias"], key)
    finally:
        server.stop()
    write_json(root / "judged.json", rows)
    print(json.dumps({"judged": len(rows), "correct": sum(r["correct"] for r in rows)}))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("job", "judge"):
        sub = commands.add_parser(name)
        sub.add_argument("--base-suite", type=Path, required=True)
        sub.add_argument("--root", type=Path, required=True)
        sub.add_argument("--data", type=Path, required=True)
        if name == "job":
            sub.add_argument("--actor", choices=("4B", "9B", "27B"), required=True)
            sub.add_argument("--manager", choices=("same", "27B"), required=True)
            sub.add_argument("--conditions", required=True)
            sub.add_argument("--questions", default="0:25", help="slice START:STOP of the subset")
            sub.add_argument("--parallel", type=int, default=6)
    args = parser.parse_args(argv)
    return job(args) if args.command == "job" else judge(args)


if __name__ == "__main__":
    raise SystemExit(main())
